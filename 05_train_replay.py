#!/usr/bin/env python3
"""
Step 5 variant — train the diffusion policy on REPLAY images instead of the
original human-demo images. Does NOT modify 05_train.py; reuses its training
loop, model, optimizer, and checkpoint format completely unchanged (loaded as
a module, with only its Dataset class substituted for the one below).

WHY: 05_train.py's default dataset sources observation images from
<episode_dir>/augmented/masked_real/ + masked_novel/ -- the ORIGINAL human
demo, with the human hand masked out (since a bare hand is never present at
deployment). Once an episode has been REPLAYED on the robot
(replay_episode.py --execute --capture_camera) and had novel views generated
from those replay frames (02_augment_noposplat.py --episode_dir
<episode_dir>/replay), you can instead train on those UNMASKED replay images.
Rationale: replay images already show the ROBOT'S OWN gripper/arm performing
the task -- the same thing deployment's live camera will see -- so there's no
human hand to mask out, and training on them can shrink the visual gap
between training and deployment.

ACTION/PROPRIO LABELS ARE UNCHANGED: they still come from the original demo's
tracked trajectory (<episode_dir>/augmented/tool_poses_*.npz), since that's
the only place pose labels exist. This is intentional, not a shortcut --
replay is the robot re-executing that exact tracked trajectory, so the pose
label at a given frame_id is still what the tool was doing; only the image
(what the scene visually looks like) changes to show the robot instead of a
human hand.

IMPORTANT -- --subsample must match whatever --subsample replay_episode.py
used (default 3): replay images only exist at the ORIGINAL frame_ids that
became waypoints (e.g. 0, 3, 6, ...), not a dense 0..449 sequence. If
--subsample here doesn't line up with that spacing, most requested frames
won't exist in replay/augmented/real/ and will be silently dropped as
incomplete windows -- not fatal, but wasteful. Matching --subsample to the
replay's own subsample (3 by default) makes every requested frame align
exactly with an existing replay image.

Corresponding deployment flag: run 07_deploy.py with --no_arm_mask, since a
policy trained on unmasked (arm-visible) images should also be DEPLOYED on
unmasked live images -- masking live but training unmasked (or vice versa)
is a train/deploy mismatch.

Usage:
    python 05_train_replay.py --data_dir data/episodes/PastaTransfer_force \\
        --output_dir data/checkpoints/PastaTransfer_force_replay \\
        --subsample 3 --action_frame task

VALIDATION SPLIT: pass --val_episodes (one or more episode IDs, e.g. "020") to
hold those episodes out of training entirely and track validation loss
alongside training loss every epoch -- this is what actually shows overfitting
(training loss alone keeps dropping even as a model overfits; validation loss
plateauing/rising while training loss keeps falling is the overfitting
signal). The validation set reuses the TRAINING set's normalizer (fit once,
from training data only -- standard practice, and required for a meaningful
loss comparison) rather than fitting its own.

    python 05_train_replay.py --data_dir data/episodes/PastaTransfer_force \\
        --output_dir data/checkpoints/PastaTransfer_force_replay \\
        --subsample 3 --action_frame task --val_episodes 020
"""
import os, sys, glob, argparse, importlib.util
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)

from policy_common import (pose_matrix_to_9d, MaxAbsNormalizer, PROPRIO_DIM,
                            select_obs_views, DiffusionPolicyNet)


def _load_train05_module():
    """'05_train' starts with a digit, not a valid module name for `import` --
    load it by file path instead. Zero modification to that file."""
    spec = importlib.util.spec_from_file_location(
        '_train05', os.path.join(PIPELINE_DIR, '05_train.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gather_replay_obs_pool(ep_dir: str, fid, cam0_only: bool = False,
                          track_cam: int = 0) -> list:
    """Like policy_common.gather_obs_pool, but sources UNMASKED images from
    the REPLAY capture's augmentation output (<ep_dir>/replay/augmented/
    {real,novel}/) instead of the demo's masked_real/masked_novel. No mask
    fallback -- if the replay image for this exact frame_id wasn't produced
    (e.g. --sample_every didn't align with the replay's --subsample), this
    returns [] and the caller (window sampler) drops that frame, same as any
    other missing-frame case."""
    real_dir  = os.path.join(ep_dir, 'replay', 'augmented', 'real')
    novel_dir = os.path.join(ep_dir, 'replay', 'augmented', 'novel')

    tc_path = os.path.join(real_dir, f'{int(fid):06d}_cam{track_cam}.jpg')
    if not os.path.exists(tc_path):
        return []
    if cam0_only:
        return [tc_path]

    all_real   = sorted(glob.glob(os.path.join(real_dir, f'{int(fid):06d}_cam*.jpg')))
    other_real = [p for p in all_real if p != tc_path]
    novel_paths = sorted(glob.glob(os.path.join(novel_dir, f'{int(fid):06d}_novel*.jpg')))
    return [tc_path] + other_real + novel_paths


class ReplayImageWindowDataset(Dataset):
    """
    Same windowing/normalisation contract as 05_train.EpisodeWindowDataset
    (drop-in replacement -- same constructor signature, same __getitem__
    return shape), but observation images come from gather_replay_obs_pool
    instead of policy_common.gather_obs_pool. See module docstring for why
    action/proprio labels are still sourced from the original demo tracking.
    """

    def __init__(self, data_dir: str, n_obs_steps: int = 2,
                 action_horizon: int = 8, image_size: int = 128,
                 crop_size: int = 115, training: bool = True,
                 cam0_only: bool = False, track_cam: int = 0,
                 n_views: int = 1, subsample: int = 1,
                 action_frame: str = 'task',
                 exclude_episodes: list = None, include_episodes: list = None,
                 external_normalizer: MaxAbsNormalizer = None):
        """
        exclude_episodes / include_episodes: episode directory basenames
        (e.g. "020") to filter data_dir's episodes by -- used to build
        disjoint train/validation splits from the same data_dir. At most one
        of these should be set.

        external_normalizer: if given, use this instead of fitting a fresh
        one from this dataset's own poses. Required for a validation split
        (must share the training set's normalizer for a meaningful loss
        comparison) -- fitting an independent one from just 1-2 held-out
        episodes would put val losses on a different scale than train losses.
        """
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

        raw_samples = []
        all_poses   = []

        ep_dirs = sorted(glob.glob(os.path.join(data_dir, '*')))
        if exclude_episodes:
            ep_dirs = [d for d in ep_dirs if os.path.basename(d) not in exclude_episodes]
        if include_episodes:
            ep_dirs = [d for d in ep_dirs if os.path.basename(d) in include_episodes]

        for ep_dir in ep_dirs:
            base_path = os.path.join(ep_dir, 'augmented', 'tool_poses_base.npz')
            task_path = os.path.join(ep_dir, 'augmented', 'tool_poses_task.npz')
            cam_path  = os.path.join(ep_dir, 'augmented', f'tool_poses_cam{track_cam}.npz')
            if action_frame == 'base':
                pose_path = base_path
            elif action_frame == 'task':
                pose_path = task_path
            else:
                pose_path = cam_path

            real_dir  = os.path.join(ep_dir, 'replay', 'augmented', 'real')
            novel_dir = os.path.join(ep_dir, 'replay', 'augmented', 'novel')
            if not os.path.exists(pose_path):
                continue
            if not os.path.isdir(real_dir):
                continue
            if not cam0_only and n_views == 1 and not os.path.isdir(novel_dir):
                continue

            poses_raw = dict(np.load(pose_path))
            frame_ids = sorted(poses_raw.keys(), key=int)
            if subsample > 1:
                frame_ids = frame_ids[::subsample]
            n_frames = len(frame_ids)

            if n_frames < n_obs_steps + action_horizon:
                continue

            for fid in frame_ids:
                all_poses.append(pose_matrix_to_9d(poses_raw[fid]))

            for i in range(n_obs_steps - 1, n_frames - action_horizon):
                obs_fids = frame_ids[i - n_obs_steps + 1: i + 1]
                act_fids = frame_ids[i + 1: i + 1 + action_horizon]

                obs_pools   = []
                obs_proprio = []
                valid = True
                for fid in obs_fids:
                    pool = gather_replay_obs_pool(ep_dir, fid, cam0_only=cam0_only,
                                                  track_cam=track_cam)
                    if not pool:
                        valid = False
                        break
                    obs_pools.append(pool)
                    obs_proprio.append(pose_matrix_to_9d(poses_raw[fid]))

                if not valid:
                    continue

                obs_proprio = np.stack(obs_proprio)
                act_seq = np.stack([pose_matrix_to_9d(poses_raw[fid]) for fid in act_fids])
                raw_samples.append((obs_pools, obs_proprio, act_seq))

        if not raw_samples:
            raise RuntimeError(
                f"No temporal samples found under {data_dir} sourcing REPLAY images "
                f"(exclude={exclude_episodes} include={include_episodes}).\n"
                "  Ensure each episode has been replayed (replay_episode.py --execute "
                "--capture_camera) and augmented (02_augment_noposplat.py --episode_dir "
                "<episode_dir>/replay).\n"
                "  Also check --subsample here matches the --subsample used at replay time "
                "(default 3) -- a mismatch means requested frame_ids won't exist in "
                "replay/augmented/real/.")

        all_poses = np.stack(all_poses)
        self.normalizer = external_normalizer if external_normalizer is not None \
            else MaxAbsNormalizer(all_poses)

        self.samples = []
        for obs_pools, obs_proprio, act_seq in raw_samples:
            self.samples.append((obs_pools,
                                  self.normalizer.normalize(obs_proprio),
                                  self.normalizer.normalize(act_seq)))

        if cam0_only:
            mode = "replay cam0 only (no novel views)"
        elif n_views > 1:
            mode = f"replay dual-cam (cam{track_cam}+other, no novel views)"
        else:
            mode = "replay cam0+cam1+novel views (unmasked)"
        sub_str = f"  subsample={subsample}" if subsample > 1 else ""
        split_str = " [VALIDATION]" if external_normalizer is not None else ""
        print(f"Dataset{split_str} (REPLAY IMAGES): {len(self.samples)} windows from "
              f"{len(ep_dirs)} episode(s)  |  "
              f"obs_steps={n_obs_steps}  n_views={n_views}  "
              f"action_horizon={action_horizon}  action_dim=9  proprio_dim={PROPRIO_DIM}  "
              f"image_pool={mode}{sub_str}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs_pools, proprio, act_seq = self.samples[idx]

        step_tensors = []
        for pool in obs_pools:
            paths = select_obs_views(pool, self.n_views, self.training)
            view_tensors = [self.transform(Image.open(p).convert('RGB')) for p in paths]
            step_tensors.append(torch.stack(view_tensors))
        obs_imgs = torch.stack(step_tensors)

        return (obs_imgs,
                torch.from_numpy(proprio),
                torch.from_numpy(act_seq))


def _run_epoch(model, dataloader, noise_scheduler, device, optimizer=None, lr_sched=None):
    """One pass over dataloader. Trains (with optimizer step) if optimizer is
    given, otherwise runs in eval/no_grad mode. Returns average MSE loss."""
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss = 0.0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for obs_imgs, proprio, actions in dataloader:
            obs_imgs = obs_imgs.to(device)
            proprio  = proprio.to(device)
            actions  = actions.to(device)
            actions_flat = actions.reshape(len(obs_imgs), -1)

            noise     = torch.randn_like(actions_flat)
            timesteps = torch.randint(0, 100, (len(obs_imgs),), device=device).long()
            noisy     = noise_scheduler.add_noise(actions_flat, noise, timesteps)

            obs_emb    = model.encode_obs(obs_imgs, proprio)
            pred_noise = model(noisy, timesteps, obs_emb)
            loss = F.mse_loss(pred_noise, noise)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_sched.step()

            total_loss += loss.item()
    return total_loss / len(dataloader)


def train_with_validation(args, train05, val_episodes, include_episodes=None):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    train_ds = ReplayImageWindowDataset(
        data_dir=args.data_dir, n_obs_steps=args.n_obs_steps,
        action_horizon=args.action_horizon, image_size=args.image_size,
        crop_size=args.crop_size, training=True, cam0_only=args.cam0_only,
        track_cam=args.track_cam, n_views=args.n_views, subsample=args.subsample,
        action_frame=args.action_frame, exclude_episodes=val_episodes,
        include_episodes=include_episodes,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

    val_loader = None
    if val_episodes:
        val_ds = ReplayImageWindowDataset(
            data_dir=args.data_dir, n_obs_steps=args.n_obs_steps,
            action_horizon=args.action_horizon, image_size=args.image_size,
            crop_size=args.crop_size, training=False, cam0_only=args.cam0_only,
            track_cam=args.track_cam, n_views=args.n_views, subsample=args.subsample,
            action_frame=args.action_frame, include_episodes=val_episodes,
            external_normalizer=train_ds.normalizer,   # share train's normaliser -- see class docstring
        )
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=2, pin_memory=True)

    model = DiffusionPolicyNet(
        action_dim=9, action_horizon=args.action_horizon, n_obs_steps=args.n_obs_steps,
        n_views=args.n_views, unet_dims=tuple(args.unet_dims), unet_kernel=args.unet_kernel,
    ).to(device)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100, beta_start=0.0001, beta_end=0.02,
        beta_schedule='squaredcos_cap_v2', clip_sample=True, prediction_type='epsilon',
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6)
    lr_sched = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=500, num_training_steps=args.num_epochs * len(train_loader))

    print(f"Image: {args.image_size}x{args.image_size} -> crop {args.crop_size}x{args.crop_size}")
    print(f"Obs steps: {args.n_obs_steps}  |  Action horizon: {args.action_horizon}")
    print(f"Training: {args.num_epochs} epochs  |  batch {args.batch_size}  "
          f"|  {len(train_loader)} steps/epoch"
          + (f"  |  validation on episode(s) {val_episodes} ({len(val_loader.dataset)} windows)"
             if val_loader else "  |  NO validation split (pass --val_episodes to add one)"))

    best_criterion   = float('inf')
    epochs_no_improve = 0
    train_ema = val_ema = None
    ema_alpha = 0.05

    for epoch in range(args.num_epochs):
        train_avg = _run_epoch(model, train_loader, noise_scheduler, device,
                               optimizer=optimizer, lr_sched=lr_sched)
        train_ema = train_avg if train_ema is None else ema_alpha * train_avg + (1 - ema_alpha) * train_ema

        val_avg = None
        if val_loader is not None:
            val_avg = _run_epoch(model, val_loader, noise_scheduler, device)
            val_ema = val_avg if val_ema is None else ema_alpha * val_avg + (1 - ema_alpha) * val_ema

        if (epoch + 1) % 10 == 0:
            msg = f"Epoch [{epoch+1:4d}/{args.num_epochs}]  train={train_avg:.4f} (ema={train_ema:.4f})"
            if val_avg is not None:
                gap = val_ema - train_ema
                msg += f"  val={val_avg:.4f} (ema={val_ema:.4f})  gap={gap:+.4f}"
            print(msg)

        if (epoch + 1) % args.checkpoint_every == 0:
            train05._save(args, epoch + 1, model, optimizer, noise_scheduler, train_ds)

        # Early stopping: on validation EMA if we have one (the correct overfitting
        # signal), otherwise falls back to training EMA (matches 05_train.py's
        # original behaviour when no --val_episodes is given).
        criterion = val_ema if val_ema is not None else train_ema
        if args.patience > 0:
            if criterion < best_criterion - args.min_delta:
                best_criterion    = criterion
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= args.patience:
                which = "val" if val_ema is not None else "train"
                print(f"\nEarly stopping at epoch {epoch+1} "
                      f"(no {which}-loss improvement for {args.patience} epochs, "
                      f"best {which}_ema={best_criterion:.4f})")
                train05._save(args, epoch + 1, model, None, noise_scheduler, train_ds,
                             name='policy_final.pt')
                return

    train05._save(args, args.num_epochs, model, None, noise_scheduler, train_ds,
                  name='policy_final.pt')
    print(f"\nTraining complete -> {args.output_dir}/policy_final.pt")


def main():
    # Pull --val_episodes out of argv before handing the rest to 05_train.py's
    # own parse_args() -- that parser doesn't know this flag, and it isn't
    # being modified to add it.
    val_parser = argparse.ArgumentParser(add_help=False)
    val_parser.add_argument('--val_episodes', nargs='+', default=None,
                            help='Episode IDs to hold out for validation (e.g. 020). '
                                 'Excluded from training entirely; validation loss is '
                                 'tracked every epoch alongside training loss.')
    val_parser.add_argument('--include_episodes', nargs='+', default=None,
                            help='Restrict training (and, if given, validation) to only '
                                 'these episode IDs (e.g. 021 022 ... 037). Applied on top '
                                 'of --val_episodes exclusion -- lets you train on a subset '
                                 'of data_dir without a separate symlinked directory.')
    val_args, remaining = val_parser.parse_known_args(sys.argv[1:])

    train05 = _load_train05_module()
    sys.argv = [sys.argv[0]] + remaining
    args = train05.parse_args()

    train_with_validation(args, train05, val_args.val_episodes, val_args.include_episodes)


if __name__ == '__main__':
    main()
