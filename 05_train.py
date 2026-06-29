"""
Step 5 — Train a diffusion policy on the augmented data.

Faithful re-implementation of the original Tool-as-Interface training approach
(Sec. 3 Problem Statement: O^r = single-view RGB image + proprioception):

  Observation:
    n_obs_steps (default 2) consecutive frames. Each frame is represented by
    a SINGLE image (N_VIEWS=1). For each obs frame, the candidate image is
    drawn at random from a pool of {real masked cam0, real masked cam1,
    rendered masked novel views novel00..05} — every candidate shares the
    same task-space action label (viewpoint-invariant), so this acts as the
    original paper's novel-view data augmentation, effectively giving each
    frame several "alternate episode" recordings from different viewpoints.
    At inference, the deployed real camera (cam0) is used. Each frame is
    also paired with the tracked tool pose (9D), used as the proprioceptive
    "robot end-effector pose" signal (the original paper retargets the
    tracked tool pose to a virtual end-effector frame for human demos).

  Action:
    Sequence of the NEXT action_horizon (default 8) tool poses in the
    robot-base frame (tool_poses_base.npz, falling back to task/cam0 frame if
    unavailable) — i.e. the policy sees the current state and predicts where
    the tool should be over the next horizon steps. Each pose is 9D:
    [tx, ty, tz, r6d(6)]. This matches the original paper's use of robot EEF
    poses shifted by +1.

  Normalisation:
    Max-abs scaling → [-1, 1] on each pose dimension independently (shared
    between action targets and proprioceptive observations, since both use
    the same 9D representation). Computed once from all training data, same
    as the original paper.

  Episodes without tool_poses_cam{track_cam}.npz (fallback mode) are skipped — temporal
  consistency requires the real-depth cam0 tracking from step 4.

Usage:
    python 05_train.py --data_dir data/episodes/hammer \\
        --output_dir data/checkpoints/hammer
"""

import os, glob, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

from policy_common import (N_VIEWS, PROPRIO_DIM, pose_matrix_to_9d,
                            MaxAbsNormalizer, DiffusionPolicyNet,
                            gather_obs_pool, select_obs_views, sample_obs_view)


# ── Dataset ───────────────────────────────────────────────────────────────────

class EpisodeWindowDataset(Dataset):
    """
    Temporal window sampler over episodes.

    For each episode:
      - frame_ids: sorted list of sampled frame indices (e.g. 0, 15, 30, ...)
      - cam0_poses: frame_id → (4,4) tool pose in cam0 frame

    Each sample:
      obs_pools  — list[n_obs_steps] of image-path pools (real cam0, real
                    cam1, rendered novel views) for each obs frame. A single
                    image is drawn from each pool per __getitem__ call
                    (random during training, cam0 during eval/inference).
      proprio    — (n_obs_steps, 9) normalised tool pose per obs step
      action_seq — (action_horizon, 9) normalised task/cam0 poses
    """

    def __init__(self, data_dir: str, n_obs_steps: int = 2,
                 action_horizon: int = 8, image_size: int = 128,
                 crop_size: int = 115, training: bool = True,
                 cam0_only: bool = False, track_cam: int = 0,
                 n_views: int = 1, subsample: int = 1,
                 action_frame: str = 'task'):
        self.n_obs_steps    = n_obs_steps
        self.action_horizon = action_horizon
        self.training       = training
        self.cam0_only      = cam0_only
        self.track_cam      = track_cam
        self.n_views        = n_views
        self.subsample      = subsample
        self.action_frame   = action_frame

        if training:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomCrop(crop_size),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

        # ── collect raw samples and all poses for normaliser ──────────────────
        raw_samples = []   # (obs_pools, obs_proprio, act_seq)
        all_poses   = []   # flat list of 9D poses for normaliser fitting

        for ep_dir in sorted(glob.glob(os.path.join(data_dir, '*'))):
            base_path = os.path.join(ep_dir, 'augmented', 'tool_poses_base.npz')
            task_path = os.path.join(ep_dir, 'augmented', 'tool_poses_task.npz')
            cam_path  = os.path.join(ep_dir, 'augmented', f'tool_poses_cam{track_cam}.npz')
            if action_frame == 'base':
                pose_path = base_path
            elif action_frame == 'task':
                pose_path = task_path
            else:
                pose_path = cam_path
            aug_dir   = os.path.join(ep_dir, 'augmented')
            novel_dir = os.path.join(aug_dir, 'masked_novel')
            if not os.path.exists(pose_path):
                continue
            if not cam0_only and n_views == 1 and not os.path.isdir(novel_dir):
                continue

            poses_raw  = dict(np.load(pose_path))            # str(fid) → (4,4)
            frame_ids  = sorted(poses_raw.keys(), key=int)
            if subsample > 1:
                frame_ids = frame_ids[::subsample]          # e.g. every 3rd → 10Hz from 30fps
            n_frames   = len(frame_ids)

            if n_frames < n_obs_steps + action_horizon:
                continue

            # Convert all poses to 9D for normaliser stats
            for fid in frame_ids:
                all_poses.append(pose_matrix_to_9d(poses_raw[fid]))

            # Sliding window: obs = frames[i-n_obs+1 … i], action = frames[i+1 … i+H]
            for i in range(n_obs_steps - 1, n_frames - action_horizon):
                obs_fids = frame_ids[i - n_obs_steps + 1 : i + 1]
                act_fids = frame_ids[i + 1 : i + 1 + action_horizon]

                # Gather image pools (real cam0/cam1 + rendered novel views) per obs frame
                obs_pools   = []
                obs_proprio = []
                valid = True
                for fid in obs_fids:
                    pool = gather_obs_pool(aug_dir, fid, cam0_only=cam0_only,
                                           track_cam=track_cam)
                    if not pool:
                        valid = False
                        break
                    obs_pools.append(pool)
                    obs_proprio.append(pose_matrix_to_9d(poses_raw[fid]))

                if not valid:
                    continue

                obs_proprio = np.stack(obs_proprio)   # (n_obs_steps, 9)

                # Build action sequence — absolute poses in task/cam0 frame
                act_seq = np.stack([pose_matrix_to_9d(poses_raw[fid])
                                    for fid in act_fids])   # (H, 9)
                raw_samples.append((obs_pools, obs_proprio, act_seq))

        if not raw_samples:
            raise RuntimeError(
                f"No temporal samples found under {data_dir}.\n"
                "  Ensure step 4 ran in TEMPORAL mode (produces tool_poses_cam0.npz).\n"
                "  Episodes processed in FALLBACK mode are skipped.")

        # ── fit normaliser from all pose data ─────────────────────────────────
        all_poses = np.stack(all_poses)   # (N_total, 9)
        self.normalizer = MaxAbsNormalizer(all_poses)

        # ── store normalised samples ──────────────────────────────────────────
        self.samples = []
        for obs_pools, obs_proprio, act_seq in raw_samples:
            self.samples.append((obs_pools,
                                  self.normalizer.normalize(obs_proprio),
                                  self.normalizer.normalize(act_seq)))

        if cam0_only:
            mode = "cam0 only (no novel views)"
        elif n_views > 1:
            mode = f"dual-cam (cam{track_cam}+other, no novel views)"
        else:
            mode = "cam0+cam1+novel views"
        sub_str = f"  subsample={subsample}" if subsample > 1 else ""
        print(f"Dataset: {len(self.samples)} windows  |  "
              f"obs_steps={n_obs_steps}  n_views={n_views}  "
              f"action_horizon={action_horizon}  action_dim=9  proprio_dim={PROPRIO_DIM}  "
              f"image_pool={mode}{sub_str}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs_pools, proprio, act_seq = self.samples[idx]   # already normalised

        # Select n_views images per obs step → (n_obs_steps, n_views, 3, H, W)
        step_tensors = []
        for pool in obs_pools:
            paths = select_obs_views(pool, self.n_views, self.training)
            view_tensors = [self.transform(Image.open(p).convert('RGB')) for p in paths]
            step_tensors.append(torch.stack(view_tensors))   # (n_views, 3, H, W)
        obs_imgs = torch.stack(step_tensors)

        return (obs_imgs,                          # (n_obs_steps, n_views, 3, H, W)
                torch.from_numpy(proprio),          # (n_obs_steps, 9)
                torch.from_numpy(act_seq))          # (action_horizon, 9)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = EpisodeWindowDataset(
        data_dir=args.data_dir,
        n_obs_steps=args.n_obs_steps,
        action_horizon=args.action_horizon,
        image_size=args.image_size,
        crop_size=args.crop_size,
        training=True,
        cam0_only=args.cam0_only,
        track_cam=args.track_cam,
        n_views=args.n_views,
        subsample=args.subsample,
        action_frame=args.action_frame,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=4, pin_memory=True)

    model = DiffusionPolicyNet(
        action_dim=9,
        action_horizon=args.action_horizon,
        n_obs_steps=args.n_obs_steps,
        n_views=args.n_views,
        unet_dims=tuple(args.unet_dims),
        unet_kernel=args.unet_kernel,
    ).to(device)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon',
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6)
    lr_sched  = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,
        num_training_steps=args.num_epochs * len(dataloader),
    )

    print(f"Image: {args.image_size}×{args.image_size} → crop {args.crop_size}×{args.crop_size}")
    print(f"Obs steps: {args.n_obs_steps}  |  Action horizon: {args.action_horizon}")
    print(f"Training: {args.num_epochs} epochs  |  batch {args.batch_size}  "
          f"|  {len(dataloader)} steps/epoch")

    best_loss        = float('inf')
    epochs_no_improve = 0
    ema_loss         = None
    ema_alpha        = 0.05   # smooth over ~20 epochs

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0

        for obs_imgs, proprio, actions in dataloader:
            obs_imgs = obs_imgs.to(device)    # (B, n_obs_steps, N_VIEWS, 3, H, W)
            proprio  = proprio.to(device)     # (B, n_obs_steps, 9)        — normalised
            actions  = actions.to(device)     # (B, action_horizon, 9)     — normalised

            actions_flat = actions.reshape(len(obs_imgs), -1)   # (B, horizon*9)

            noise     = torch.randn_like(actions_flat)
            timesteps = torch.randint(0, 100, (len(obs_imgs),), device=device).long()
            noisy     = noise_scheduler.add_noise(actions_flat, noise, timesteps)

            obs_emb    = model.encode_obs(obs_imgs, proprio)
            pred_noise = model(noisy, timesteps, obs_emb)

            loss = F.mse_loss(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_sched.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(dataloader)
        ema_loss = avg if ema_loss is None else ema_alpha * avg + (1 - ema_alpha) * ema_loss

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1:4d}/{args.num_epochs}]  loss={avg:.4f}  ema={ema_loss:.4f}")

        if (epoch + 1) % args.checkpoint_every == 0:
            _save(args, epoch + 1, model, optimizer, noise_scheduler, dataset)

        # Early stopping — track EMA loss improvement
        if args.patience > 0:
            if ema_loss < best_loss - args.min_delta:
                best_loss         = ema_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1} "
                      f"(no improvement for {args.patience} epochs, "
                      f"best ema_loss={best_loss:.4f})")
                _save(args, epoch + 1, model, None, noise_scheduler, dataset,
                      name='policy_final.pt')
                return

    _save(args, args.num_epochs, model, None, noise_scheduler, dataset,
          name='policy_final.pt')
    print(f"\nTraining complete → {args.output_dir}/policy_final.pt")


def _save(args, epoch, model, optimizer, noise_scheduler, dataset, name=None):
    name = name or f'policy_epoch{epoch:04d}.pt'
    ckpt = {
        'epoch':            epoch,
        'model':            model.state_dict(),
        'noise_scheduler':  noise_scheduler,
        'action_horizon':   args.action_horizon,
        'n_obs_steps':      args.n_obs_steps,
        'n_views':          args.n_views,
        'proprio_dim':      PROPRIO_DIM,
        'image_size':       args.image_size,
        'crop_size':        args.crop_size,
        'normalizer':       dataset.normalizer.state_dict(),
        'unet_dims':        list(args.unet_dims),
        'unet_kernel':      args.unet_kernel,
        'action_frame':     args.action_frame,
    }
    if optimizer is not None:
        ckpt['optimizer'] = optimizer.state_dict()
    path = os.path.join(args.output_dir, name)
    torch.save(ckpt, path)
    print(f"  Saved checkpoint: {path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',         required=True,
                   help='Task directory containing episode subdirectories')
    p.add_argument('--output_dir',       default='data/checkpoints')
    p.add_argument('--num_epochs',       type=int,   default=3050,
                   help='Matches original paper')
    p.add_argument('--batch_size',       type=int,   default=32,
                   help='Matches original paper')
    p.add_argument('--lr',               type=float, default=1e-4)
    p.add_argument('--image_size',       type=int,   default=128,
                   help='Resize images to this size before cropping (original: 128)')
    p.add_argument('--crop_size',        type=int,   default=115,
                   help='Random crop size during training (original: 115)')
    p.add_argument('--n_obs_steps',      type=int,   default=2,
                   help='Number of consecutive frames as observation')
    p.add_argument('--action_horizon',   type=int,   default=16,
                   help='Number of future poses to predict (pred horizon)')
    p.add_argument('--n_views',          type=int,   default=1,
                   help='Number of camera views per obs step. 1=single-cam (random pool '
                        'augmentation), 2=dual-cam (always use track_cam + one other, no random).')
    p.add_argument('--subsample',        type=int,   default=1,
                   help='Use every Nth frame. Set to 3 when recording at 30fps and deploying '
                        'at 10Hz. Subsample=1 means use all frames.')
    p.add_argument('--unet_dims',        type=int,   nargs='+', default=[128, 256, 512],
                   help='Channel sizes for each UNet-1D encoder stage (default: 256 512 1024). '
                        'Number of values = number of encoder stages. '
                        'action_horizon must be divisible by 2^(n_stages-1).')
    p.add_argument('--unet_kernel',      type=int,   default=5,
                   help='Conv1d kernel size in UNet-1D residual blocks (default: 5).')
    p.add_argument('--checkpoint_every', type=int,   default=100)
    p.add_argument('--patience',         type=int,   default=500,
                   help='Early stopping: stop if EMA loss does not improve for this many epochs. '
                        'Set to 0 to disable.')
    p.add_argument('--min_delta',        type=float, default=1e-4,
                   help='Minimum EMA loss improvement to count as progress.')
    p.add_argument('--device',           default='cuda')
    p.add_argument('--action_frame',      default='task',
                   choices=['task', 'base', 'cam'],
                   help='Coordinate frame for action targets. '
                        '"task" = ChArUco board origin (default). '
                        '"base" = robot base frame (requires robot_extrinsics). '
                        '"cam"  = tracking camera frame.')
    p.add_argument('--cam0_only',        action='store_true',
                   help='Use only the track_cam real images — no other cams, no novel views. '
                        'Eliminates viewpoint mismatch between training and deployment.')
    p.add_argument('--track_cam',        type=int, default=0,
                   help='Camera index used at deployment (default: 0). '
                        'Ensures pool[0] = track_cam image so inference always uses the '
                        'correct camera. Match this to --track_cam in 04_track.py and 07_deploy.py.')
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
