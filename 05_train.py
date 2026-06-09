"""
Step 5 — Train a diffusion policy on the augmented data.

Faithful re-implementation of the original Tool-as-Interface training approach:

  Observation:
    n_obs_steps (default 2) consecutive frames, each represented by one
    masked novel view image randomly sampled from available views per frame.
    Images are stacked along the channel dimension.

  Action:
    Sequence of the NEXT action_horizon (default 8) tool poses in the cam0
    frame — i.e. the policy sees the current state and predicts where the tool
    should be over the next horizon steps. Each pose is 9D: [tx, ty, tz, r6d(6)].
    This matches the original paper's use of robot EEF poses shifted by +1.

  Normalisation:
    Max-abs scaling → [-1, 1] on each action dimension independently.
    Computed once from all training data, same as the original paper.

  Episodes without tool_poses_cam0.npz (fallback mode) are skipped — temporal
  consistency requires the real-depth cam0 tracking from step 4.

Usage:
    python 05_train.py --data_dir data/episodes/hammer \\
        --output_dir data/checkpoints/hammer
"""

import os, glob, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup


# ── Rotation / pose utilities ─────────────────────────────────────────────────

def matrix_to_6d(R: np.ndarray) -> np.ndarray:
    """(3,3) rotation matrix → 6D (first two columns, row-major)."""
    return R[:, :2].T.reshape(6)


def pose_matrix_to_9d(T: np.ndarray) -> np.ndarray:
    """(4,4) pose → 9D [tx, ty, tz, rot6d] — translation first, matches original."""
    return np.concatenate([T[:3, 3], matrix_to_6d(T[:3, :3])]).astype(np.float32)


def rot6d_to_matrix(r6d: np.ndarray) -> np.ndarray:
    """6D → (3,3) via Gram-Schmidt. Inverse of matrix_to_6d."""
    a1, a2 = r6d[:3], r6d[3:]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def action_9d_to_pose(a: np.ndarray) -> np.ndarray:
    """9D action [tx, ty, tz, rot6d] → (4,4) pose."""
    T = np.eye(4, dtype=np.float64)
    T[:3, 3]  = a[:3]
    T[:3, :3] = rot6d_to_matrix(a[3:])
    return T


# ── Normaliser ────────────────────────────────────────────────────────────────

class MaxAbsNormalizer:
    """
    Scales each action dimension independently so that the max absolute value
    maps to 1. Matches the original paper's normalizer_from_stat() logic.
    """
    def __init__(self, actions: np.ndarray):
        # actions: (N, action_dim)
        max_abs = np.maximum(actions.max(axis=0), np.abs(actions.min(axis=0)))
        max_abs = np.where(max_abs < 1e-8, 1.0, max_abs)
        self.scale  = (1.0 / max_abs).astype(np.float32)   # (action_dim,)
        self.offset = np.zeros_like(self.scale)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return x * self.scale + self.offset

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.offset) / self.scale

    def state_dict(self):
        return {'scale': self.scale, 'offset': self.offset}

    @classmethod
    def from_state_dict(cls, d):
        obj = cls.__new__(cls)
        obj.scale  = d['scale']
        obj.offset = d['offset']
        return obj


# ── Dataset ───────────────────────────────────────────────────────────────────

class EpisodeWindowDataset(Dataset):
    """
    Temporal window sampler over episodes.

    For each episode:
      - frame_ids: sorted list of sampled frame indices (e.g. 0, 15, 30, ...)
      - cam0_poses: frame_id → (4,4) tool pose in cam0 frame

    Each sample:
      obs_paths  — list[n_obs_steps] of lists[n_novel_views] of image paths
      action_seq — (action_horizon, 9) normalised cam0 poses
    """

    def __init__(self, data_dir: str, n_obs_steps: int = 2,
                 action_horizon: int = 8, image_size: int = 128,
                 crop_size: int = 115, training: bool = True):
        self.n_obs_steps    = n_obs_steps
        self.action_horizon = action_horizon
        self.training       = training

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

        # ── collect raw samples and all actions for normaliser ────────────────
        raw_samples  = []   # (episode_dir, obs_frame_ids, act_frame_ids)
        all_actions  = []   # flat list of 9D actions for normaliser fitting

        for ep_dir in sorted(glob.glob(os.path.join(data_dir, '*'))):
            # Prefer task-frame poses (calibrated via 00_calibrate.py) over cam0 poses
            task_path = os.path.join(ep_dir, 'augmented', 'tool_poses_task.npz')
            cam0_path = os.path.join(ep_dir, 'augmented', 'tool_poses_cam0.npz')
            pose_path = task_path if os.path.exists(task_path) else cam0_path
            novel_dir = os.path.join(ep_dir, 'augmented', 'masked_novel')
            if not os.path.exists(pose_path) or not os.path.isdir(novel_dir):
                continue

            poses_raw = dict(np.load(pose_path))            # str(fid) → (4,4)
            frame_ids = sorted(poses_raw.keys(), key=int)
            n_frames  = len(frame_ids)

            if n_frames < n_obs_steps + action_horizon:
                continue

            # Convert all poses to 9D for normaliser stats
            for fid in frame_ids:
                all_actions.append(pose_matrix_to_9d(poses_raw[fid]))

            # Sliding window: obs = frames[i-n_obs+1 … i], action = frames[i+1 … i+H]
            for i in range(n_obs_steps - 1, n_frames - action_horizon):
                obs_fids = frame_ids[i - n_obs_steps + 1 : i + 1]
                act_fids = frame_ids[i + 1 : i + 1 + action_horizon]

                # Verify all novel view images exist for obs frames
                obs_paths = []
                valid = True
                for fid in obs_fids:
                    paths = sorted(glob.glob(
                        os.path.join(novel_dir, f'{int(fid):06d}_novel*.jpg')))
                    if not paths:
                        valid = False
                        break
                    obs_paths.append(paths)

                if not valid:
                    continue

                # Build action sequence — absolute poses in task/cam0 frame
                act_seq = np.stack([pose_matrix_to_9d(poses_raw[fid])
                                    for fid in act_fids])   # (H, 9)
                raw_samples.append((obs_paths, act_seq))

        if not raw_samples:
            raise RuntimeError(
                f"No temporal samples found under {data_dir}.\n"
                "  Ensure step 4 ran in TEMPORAL mode (produces tool_poses_cam0.npz).\n"
                "  Episodes processed in FALLBACK mode are skipped.")

        # ── fit normaliser from all action data ───────────────────────────────
        all_actions = np.stack(all_actions)   # (N_total, 9)
        self.normalizer = MaxAbsNormalizer(all_actions)

        # ── store normalised samples ──────────────────────────────────────────
        self.samples = []
        for obs_paths, act_seq in raw_samples:
            self.samples.append((obs_paths, self.normalizer.normalize(act_seq)))

        print(f"Dataset: {len(self.samples)} windows  |  "
              f"obs_steps={n_obs_steps}  action_horizon={action_horizon}  "
              f"action_dim=9")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs_paths, act_seq = self.samples[idx]   # act_seq already normalised

        imgs = []
        for frame_paths in obs_paths:
            # Training: random novel view. Eval: first view.
            path = (frame_paths[np.random.randint(len(frame_paths))]
                    if self.training else frame_paths[0])
            imgs.append(self.transform(Image.open(path).convert('RGB')))

        # Stack along channel dim: (n_obs_steps*3, H, W)
        obs_img = torch.cat(imgs, dim=0)
        return obs_img, torch.from_numpy(act_seq)   # (C, H, W), (H, 9)


# ── Model ─────────────────────────────────────────────────────────────────────

class DiffusionPolicyNet(nn.Module):
    """
    DDPM-based policy.

    Observation encoder:
      - Each of n_obs_steps frames encoded independently by ResNet-18
      - Temporal embeddings averaged → obs_dim (512)

    Noise predictor:
      - Conditioned on obs embedding via FiLM (scale + shift per layer)
      - Predicts noise over the flattened action sequence (action_horizon × 9)
    """

    def __init__(self, action_dim: int = 9, obs_dim: int = 512,
                 action_horizon: int = 8, n_obs_steps: int = 2):
        super().__init__()
        import torchvision.models as tvm

        self.n_obs_steps    = n_obs_steps
        self.action_horizon = action_horizon
        self.action_dim     = action_dim
        flat_action         = action_dim * action_horizon

        # Shared ResNet-18 backbone for all obs frames
        backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])  # → (B, 512, 1, 1)

        # Project concatenated obs embeddings
        self.obs_proj = nn.Sequential(
            nn.Linear(obs_dim * n_obs_steps, obs_dim),
            nn.SiLU(),
        )

        # Timestep embedding (sinusoidal-inspired, simple learned)
        self.time_emb = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )

        # Noise predictor with FiLM conditioning on obs
        hidden = 512
        self.net = nn.ModuleList([
            nn.Linear(flat_action + 64, hidden),
            nn.Linear(hidden, hidden),
            nn.Linear(hidden, hidden),
            nn.Linear(hidden, flat_action),
        ])
        # FiLM scale + shift from obs per hidden layer (first 3 layers)
        self.film = nn.ModuleList([
            nn.Linear(obs_dim, hidden * 2) for _ in range(3)
        ])
        self.act = nn.SiLU()

    def encode_obs(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs: (B, n_obs_steps*3, H, W) — obs frames stacked along channels
        Returns: (B, obs_dim)
        """
        B = imgs.shape[0]
        # Split back into individual frames and encode each
        frame_imgs = imgs.chunk(self.n_obs_steps, dim=1)   # n_obs_steps × (B, 3, H, W)
        embs = [self.encoder(f).squeeze(-1).squeeze(-1) for f in frame_imgs]
        return self.obs_proj(torch.cat(embs, dim=-1))      # (B, obs_dim)

    def forward(self, noisy_actions: torch.Tensor, timesteps: torch.Tensor,
                obs_emb: torch.Tensor) -> torch.Tensor:
        """
        noisy_actions: (B, action_horizon * action_dim)
        timesteps:     (B,)
        obs_emb:       (B, obs_dim)
        Returns predicted noise: (B, action_horizon * action_dim)
        """
        t_emb = self.time_emb(timesteps.float().unsqueeze(-1) / 100.0)  # (B, 64)
        x = torch.cat([noisy_actions, t_emb], dim=-1)

        for i, layer in enumerate(self.net[:-1]):
            x = layer(x)
            # FiLM: scale + shift from obs embedding
            film_out = self.film[i](obs_emb)
            scale, shift = film_out.chunk(2, dim=-1)
            x = x * (1 + scale) + shift
            x = self.act(x)

        return self.net[-1](x)


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
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=4, pin_memory=True)

    model = DiffusionPolicyNet(
        action_dim=9,
        action_horizon=args.action_horizon,
        n_obs_steps=args.n_obs_steps,
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

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0

        for obs_imgs, actions in dataloader:
            obs_imgs = obs_imgs.to(device)    # (B, n_obs*3, H, W)
            actions  = actions.to(device)     # (B, action_horizon, 9)  — already normalised

            actions_flat = actions.reshape(len(obs_imgs), -1)   # (B, horizon*9)

            noise     = torch.randn_like(actions_flat)
            timesteps = torch.randint(0, 100, (len(obs_imgs),), device=device).long()
            noisy     = noise_scheduler.add_noise(actions_flat, noise, timesteps)

            obs_emb    = model.encode_obs(obs_imgs)
            pred_noise = model(noisy, timesteps, obs_emb)

            loss = F.mse_loss(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_sched.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1:4d}/{args.num_epochs}]  loss={avg:.4f}")

        if (epoch + 1) % args.checkpoint_every == 0:
            _save(args, epoch + 1, model, optimizer, noise_scheduler, dataset)

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
        'image_size':       args.image_size,
        'crop_size':        args.crop_size,
        'normalizer':       dataset.normalizer.state_dict(),
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
                   help='Number of consecutive frames as observation (original: 2)')
    p.add_argument('--action_horizon',   type=int,   default=8,
                   help='Number of future poses to predict (original: 8)')
    p.add_argument('--checkpoint_every', type=int,   default=100)
    p.add_argument('--device',           default='cuda')
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
