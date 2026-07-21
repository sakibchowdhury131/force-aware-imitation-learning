#!/usr/bin/env python3
"""
Step 5 variant — like 05_train_replay.py (trains on REPLAY images), but the
proprioception/action vectors are extended from 9D pose to 12D [pose9,
force3]: past force is fed into the observation alongside proprio, and
future force is predicted alongside future pose over the same action
horizon. Does NOT modify 05_train.py or 05_train_replay.py -- reuses
05_train.py's parse_args()/model config, with its own dataset, training
loop, and checkpoint save (see "Why not just call train05._save()" below).

WHY REPLAY IMAGES, NOT THE ORIGINAL DEMO IMAGES: force was only ever
measured during REPLAY (replay_episode.py --execute), logged to
<episode_dir>/replay/replay_full_forces.npz (dense, ~47Hz) and
<episode_dir>/replay/torque_log.npz (frame_id -> t, exactly the subsampled
frame grid). The original human-demo images have no corresponding force
measurement at all -- there's no clock to align them on. Replay images are
frame_id-aligned with the SAME tool_poses_base.npz labels 05_train.py uses
(replay re-executes that exact tracked trajectory), so pairing "frame_id's
image and pose" with "frame_id's force, read off the replay force log" is
well-defined. This is why this script is a variant of 05_train_replay.py,
not of 05_train.py directly.

FORCE PREPROCESSING (per episode):
  1. Load the DENSE replay force log (~47Hz) -- NOT the already-sparse
     per-frame log (replay_forces_per_frame.npz has only ~150 samples over
     ~60s, i.e. ~2.5Hz average rate, right at the Nyquist edge for a 2Hz
     filter and too coarse to filter meaningfully).
  2. Tare: subtract the mean force over the first --tare_window_s seconds
     (pose-dependent gravity-residual bias varies episode-to-episode by
     several Newtons even at rest -- confirmed empirically across episodes
     020/021/025/030/035/037 -- so an untared force is partly just
     recalibration noise, not contact signal).
  3. Low-pass filter at --force_cutoff_hz (default 2.0, matching what we
     settled on for the visualization in plot_episode_forces.py) via
     scipy.signal.filtfilt (offline/non-causal -- fine here, this is a
     training-label pass, not a real-time loop).
  4. For each frame_id used by the dataset window sampler, look up that
     frame's timestamp (torque_log.npz's frame_id->t, already nearest-
     matched to the dense log with sub-20ms error) and take the nearest
     tared+filtered dense sample. This is the frame's 3D force label, used
     both as a past-observation feature and a future-prediction target.

LOSS WEIGHTING: force is a noisier, more contact-geometry-sensitive signal
than pose across demonstrations (two demos of "the same" phase can have
different contact force even with similar poses), so --force_loss_weight
(default 0.2) down-weights the 3 force dimensions relative to the 9 pose
dimensions in the per-dimension-weighted noise-prediction MSE, keeping pose
prediction the dominant training signal.

CHECKPOINT FORMAT: this checkpoint stores proprio_dim=12/action_dim=12 (vs.
05_train.py's implicit 9/9) plus predicts_force=True/force_dim=3. Not using
train05._save() here because it hardcodes 'proprio_dim': PROPRIO_DIM (9)
regardless of what was actually trained -- would silently mismatch. NOTE for
the deploy-side follow-up (separate task, not done in this script):
test_policy.py's load_model() currently hardcodes action_dim=9 when
reconstructing DiffusionPolicyNet from any checkpoint -- loading THIS
checkpoint there will need that changed to
ckpt.get('action_dim', 9) before it will load correctly.

CONDITIONING-ONLY VARIANT (--no_predict_force): force still goes into the
observation (proprio stays 12D), but the action target is pure 9D pose --
the model is never asked to predict future force, only better positions.
Isolates "does force-conditioning improve pose prediction" from "can this
model also predict force well" (the combined variant's force-prediction
accuracy was found to generalize poorly to a held-out high-drift episode --
see POLICY_EXPERIMENTS.md -- this variant tests the more basic claim without
that confound). Deployment-side: 07_deploy_force.py reads the checkpoint's
own force_in_proprio/predicts_force fields independently, so this variant
deploys through the ordinary position controller with no admittance
correction available (there's no predicted force to use as a reference).

Usage:
    python 05_train_replay_force.py --data_dir data/episodes/PastaTransfer_force \\
        --output_dir data/checkpoints/PastaTransfer_force_replay_force \\
        --subsample 3 --action_frame task --val_episodes 020

    python 05_train_replay_force.py --data_dir data/episodes/PastaTransfer_force \\
        --output_dir data/checkpoints/PastaTransfer_force_cond_only \\
        --subsample 3 --action_frame task --val_episodes 020 --no_predict_force
"""
import os, sys, glob, argparse, importlib.util
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from scipy.signal import butter, filtfilt
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)

from policy_common import pose_matrix_to_9d, MaxAbsNormalizer, select_obs_views, DiffusionPolicyNet

POSE_DIM  = 9
FORCE_DIM = 3
COMBINED_DIM = POSE_DIM + FORCE_DIM   # 12


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
    """Same as 05_train_replay.py's helper of the same name -- unmasked replay
    images, no mask fallback."""
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


def load_frame_forces(ep_dir: str, cutoff_hz: float = 2.0,
                      tare_window_s: float = 1.0, order: int = 2):
    """Returns dict[int frame_id] -> (3,) tared + low-pass-filtered external
    force (N) for this episode, or None if the replay force logs don't exist
    (episode was never replayed / never had --capture_camera force logging).
    See module docstring for why filtering happens on the dense log and is
    then matched to each frame, rather than filtering the sparse per-frame
    log directly."""
    dense_path = os.path.join(ep_dir, 'replay', 'replay_full_forces.npz')
    frame_path = os.path.join(ep_dir, 'replay', 'torque_log.npz')
    if not os.path.exists(dense_path) or not os.path.exists(frame_path):
        return None

    dd = np.load(dense_path)
    t_dense = dd['t']
    F_dense = dd['external_force_xyz'].astype(np.float64)

    tare = F_dense[(t_dense - t_dense[0]) < tare_window_s].mean(axis=0)
    F_tared = F_dense - tare

    if cutoff_hz > 0:
        fs = 1.0 / np.mean(np.diff(t_dense))
        b, a = butter(order, cutoff_hz, btype='low', fs=fs)
        F_filt = np.stack([filtfilt(b, a, F_tared[:, i]) for i in range(3)], axis=1)
    else:
        F_filt = F_tared

    fd = np.load(frame_path)
    frame_ids = fd['frame_id']
    frame_t   = fd['t']

    out = {}
    for fid, t in zip(frame_ids, frame_t):
        idx = int(np.clip(np.searchsorted(t_dense, t), 0, len(t_dense) - 1))
        if idx > 0 and abs(t_dense[idx - 1] - t) < abs(t_dense[idx] - t):
            idx -= 1
        out[int(fid)] = F_filt[idx].astype(np.float32)
    return out


class ReplayImageForceWindowDataset(Dataset):
    """
    Like 05_train_replay.ReplayImageWindowDataset, but obs_proprio and
    act_seq are 12D [pose9, force3] instead of 9D pose -- past force is
    concatenated onto proprio (observation), future force is concatenated
    onto each predicted pose (action target).
    """

    def __init__(self, data_dir: str, n_obs_steps: int = 2,
                 action_horizon: int = 8, image_size: int = 128,
                 crop_size: int = 115, training: bool = True,
                 cam0_only: bool = False, track_cam: int = 0,
                 n_views: int = 1, subsample: int = 1,
                 action_frame: str = 'task',
                 exclude_episodes: list = None, include_episodes: list = None,
                 external_normalizer: MaxAbsNormalizer = None,
                 force_cutoff_hz: float = 2.0, tare_window_s: float = 1.0,
                 predict_force: bool = True):
        """
        predict_force: if False, force is still concatenated onto the
        OBSERVATION (proprio stays 12D [pose9,force3] -- force as a
        conditioning signal only), but the ACTION target is pure 9D pose --
        the model is never asked to predict future force, only better
        positions. Isolates "does force-conditioning improve pose
        prediction" from "can this model also predict force," which the
        combined variant conflates.
        """
        self.n_obs_steps    = n_obs_steps
        self.action_horizon = action_horizon
        self.training       = training
        self.cam0_only      = cam0_only
        self.track_cam      = track_cam
        self.n_views        = n_views
        self.subsample      = subsample
        self.action_frame   = action_frame
        self.predict_force  = predict_force

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
        all_vecs    = []   # flat list of 12D [pose9,force3] vectors for normaliser stats

        ep_dirs = sorted(glob.glob(os.path.join(data_dir, '*')))
        if exclude_episodes:
            ep_dirs = [d for d in ep_dirs if os.path.basename(d) not in exclude_episodes]
        if include_episodes:
            ep_dirs = [d for d in ep_dirs if os.path.basename(d) in include_episodes]

        n_skipped_no_force = 0
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

            force_dict = load_frame_forces(ep_dir, cutoff_hz=force_cutoff_hz,
                                           tare_window_s=tare_window_s)
            if force_dict is None:
                n_skipped_no_force += 1
                continue

            poses_raw = dict(np.load(pose_path))
            frame_ids = sorted(poses_raw.keys(), key=int)
            if subsample > 1:
                frame_ids = frame_ids[::subsample]
            n_frames = len(frame_ids)

            if n_frames < n_obs_steps + action_horizon:
                continue

            for fid in frame_ids:
                if int(fid) not in force_dict:
                    continue
                all_vecs.append(np.concatenate([pose_matrix_to_9d(poses_raw[fid]),
                                                force_dict[int(fid)]]))

            for i in range(n_obs_steps - 1, n_frames - action_horizon):
                obs_fids = frame_ids[i - n_obs_steps + 1: i + 1]
                act_fids = frame_ids[i + 1: i + 1 + action_horizon]

                obs_pools   = []
                obs_proprio = []
                valid = True
                for fid in obs_fids:
                    if int(fid) not in force_dict:
                        valid = False
                        break
                    pool = gather_replay_obs_pool(ep_dir, fid, cam0_only=cam0_only,
                                                  track_cam=track_cam)
                    if not pool:
                        valid = False
                        break
                    obs_pools.append(pool)
                    obs_proprio.append(np.concatenate([pose_matrix_to_9d(poses_raw[fid]),
                                                       force_dict[int(fid)]]))
                if not valid:
                    continue
                # Force is only required at act_fids when it's part of the
                # action target -- the conditioning-only variant (predict_force
                # =False) never reads force at future frames, so those windows
                # shouldn't be dropped just because a future force sample is
                # missing (more usable windows than the combined variant).
                if predict_force and any(int(fid) not in force_dict for fid in act_fids):
                    continue

                obs_proprio = np.stack(obs_proprio)   # (n_obs_steps, 12)
                if predict_force:
                    act_seq = np.stack([np.concatenate([pose_matrix_to_9d(poses_raw[fid]),
                                                        force_dict[int(fid)]])
                                        for fid in act_fids])   # (H, 12)
                else:
                    act_seq = np.stack([pose_matrix_to_9d(poses_raw[fid])
                                        for fid in act_fids])   # (H, 9)
                raw_samples.append((obs_pools, obs_proprio, act_seq))

        if not raw_samples:
            raise RuntimeError(
                f"No temporal samples found under {data_dir} sourcing REPLAY images + force "
                f"(exclude={exclude_episodes} include={include_episodes}, "
                f"{n_skipped_no_force} episode(s) skipped for missing replay force logs).\n"
                "  Ensure each episode has been replayed with force capture "
                "(replay/replay_full_forces.npz + replay/torque_log.npz present) and "
                "augmented (02_augment_noposplat.py --episode_dir <episode_dir>/replay).")

        all_vecs = np.stack(all_vecs)   # (N_total, 12)
        self.normalizer = external_normalizer if external_normalizer is not None \
            else MaxAbsNormalizer(all_vecs)

        self.samples = []
        for obs_pools, obs_proprio, act_seq in raw_samples:
            # act_seq is 9D in the conditioning-only variant -- normalize()
            # slices the (12D-fit) normalizer's scale/offset to match
            # automatically (policy_common.MaxAbsNormalizer._matched), so
            # pose scaling is identical between both variants (fair
            # comparison) without fitting a separate 9D normalizer.
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
        skip_str = f"  ({n_skipped_no_force} episode(s) skipped, no replay force log)" \
            if n_skipped_no_force else ""
        action_dim = COMBINED_DIM if predict_force else POSE_DIM
        action_desc = f"pose{POSE_DIM}+force{FORCE_DIM}" if predict_force else f"pose{POSE_DIM} only"
        print(f"Dataset{split_str} (REPLAY IMAGES + FORCE{'' if predict_force else ' COND-ONLY'}): "
              f"{len(self.samples)} windows from "
              f"{len(ep_dirs) - n_skipped_no_force} episode(s){skip_str}  |  "
              f"obs_steps={n_obs_steps}  n_views={n_views}  "
              f"action_horizon={action_horizon}  action_dim={action_dim} ({action_desc})  "
              f"proprio_dim={COMBINED_DIM}  image_pool={mode}{sub_str}")

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


def _run_epoch(model, dataloader, noise_scheduler, device, weight_vec,
              optimizer=None, lr_sched=None):
    """Like 05_train_replay.py's _run_epoch, but the noise-prediction MSE is
    weighted per-dimension by weight_vec (1.0 for the 9 pose dims,
    args.force_loss_weight for the 3 force dims, repeated per horizon step)
    so a noisier force target doesn't dominate the shared visual encoder."""
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
            loss = (((pred_noise - noise) ** 2) * weight_vec).mean()

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_sched.step()

            total_loss += loss.item()
    return total_loss / len(dataloader)


def _save_force(args, epoch, model, optimizer, noise_scheduler, dataset, name=None):
    """Local checkpoint save -- NOT train05._save(), which hardcodes
    'proprio_dim': PROPRIO_DIM (9). Proprio is always 12D here (force is
    always a conditioning input); action_dim/predicts_force depend on
    --no_predict_force. See module docstring for the test_policy.py::
    load_model follow-up needed before this can be deployed. force_in_proprio
    is stored explicitly (rather than left for 07_deploy_force.py to infer
    from proprio_dim>9) so the deploy script can unambiguously decide whether
    to build a 12D proprio observation regardless of predicts_force."""
    name = name or f'policy_epoch{epoch:04d}.pt'
    action_dim = COMBINED_DIM if args.predict_force else POSE_DIM
    ckpt = {
        'epoch':             epoch,
        'model':             model.state_dict(),
        'noise_scheduler':   noise_scheduler,
        'action_horizon':    args.action_horizon,
        'n_obs_steps':       args.n_obs_steps,
        'n_views':           args.n_views,
        'proprio_dim':       COMBINED_DIM,
        'action_dim':        action_dim,
        'predicts_force':    args.predict_force,
        'force_in_proprio':  True,
        'force_dim':         FORCE_DIM,
        'force_cutoff_hz':   args.force_cutoff_hz,
        'force_loss_weight': args.force_loss_weight if args.predict_force else None,
        'tare_window_s':     args.tare_window_s,
        'image_size':        args.image_size,
        'crop_size':         args.crop_size,
        'normalizer':        dataset.normalizer.state_dict(),
        'unet_dims':         list(args.unet_dims),
        'unet_kernel':       args.unet_kernel,
        'action_frame':      args.action_frame,
    }
    if optimizer is not None:
        ckpt['optimizer'] = optimizer.state_dict()
    path = os.path.join(args.output_dir, name)
    torch.save(ckpt, path)
    print(f"  Saved checkpoint: {path}")


def train_with_validation(args, val_episodes, include_episodes=None):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    train_ds = ReplayImageForceWindowDataset(
        data_dir=args.data_dir, n_obs_steps=args.n_obs_steps,
        action_horizon=args.action_horizon, image_size=args.image_size,
        crop_size=args.crop_size, training=True, cam0_only=args.cam0_only,
        track_cam=args.track_cam, n_views=args.n_views, subsample=args.subsample,
        action_frame=args.action_frame, exclude_episodes=val_episodes,
        include_episodes=include_episodes,
        force_cutoff_hz=args.force_cutoff_hz, tare_window_s=args.tare_window_s,
        predict_force=args.predict_force,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

    val_loader = None
    if val_episodes:
        val_ds = ReplayImageForceWindowDataset(
            data_dir=args.data_dir, n_obs_steps=args.n_obs_steps,
            action_horizon=args.action_horizon, image_size=args.image_size,
            crop_size=args.crop_size, training=False, cam0_only=args.cam0_only,
            track_cam=args.track_cam, n_views=args.n_views, subsample=args.subsample,
            action_frame=args.action_frame, include_episodes=val_episodes,
            external_normalizer=train_ds.normalizer,
            force_cutoff_hz=args.force_cutoff_hz, tare_window_s=args.tare_window_s,
            predict_force=args.predict_force,
        )
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=2, pin_memory=True)

    action_dim = COMBINED_DIM if args.predict_force else POSE_DIM
    model = DiffusionPolicyNet(
        action_dim=action_dim, action_horizon=args.action_horizon,
        n_obs_steps=args.n_obs_steps, n_views=args.n_views,
        proprio_dim=COMBINED_DIM,
        unet_dims=tuple(args.unet_dims), unet_kernel=args.unet_kernel,
    ).to(device)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100, beta_start=0.0001, beta_end=0.02,
        beta_schedule='squaredcos_cap_v2', clip_sample=True, prediction_type='epsilon',
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6)
    lr_sched = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=500, num_training_steps=args.num_epochs * len(train_loader))

    # Per-dimension loss weight: 1.0 for the 9 pose dims, force_loss_weight for
    # the 3 force dims (only present when predicting force), repeated once per
    # horizon step (matches the (horizon, dim) -> flat reshape order used for
    # actions_flat/pred_noise/noise). Conditioning-only variant: plain
    # all-ones weight over the 9 pose dims -- equivalent to unweighted MSE,
    # matching 05_train_replay.py's original loss exactly.
    if args.predict_force:
        weight_pattern = torch.tensor([1.0] * POSE_DIM + [args.force_loss_weight] * FORCE_DIM)
    else:
        weight_pattern = torch.tensor([1.0] * POSE_DIM)
    weight_vec = weight_pattern.repeat(args.action_horizon).to(device)

    print(f"Image: {args.image_size}x{args.image_size} -> crop {args.crop_size}x{args.crop_size}")
    print(f"Obs steps: {args.n_obs_steps}  |  Action horizon: {args.action_horizon}")
    if args.predict_force:
        print(f"Force: cutoff={args.force_cutoff_hz}Hz  tare_window={args.tare_window_s}s  "
              f"loss_weight={args.force_loss_weight} (pose dims weighted 1.0)  "
              f"[predicting force -- combined variant]")
    else:
        print(f"Force: cutoff={args.force_cutoff_hz}Hz  tare_window={args.tare_window_s}s  "
              f"[CONDITIONING-ONLY variant -- force is an input, action target is pose-only]")
    print(f"Training: {args.num_epochs} epochs  |  batch {args.batch_size}  "
          f"|  {len(train_loader)} steps/epoch"
          + (f"  |  validation on episode(s) {val_episodes} ({len(val_loader.dataset)} windows)"
             if val_loader else "  |  NO validation split (pass --val_episodes to add one)"))

    best_criterion   = float('inf')
    epochs_no_improve = 0
    train_ema = val_ema = None
    ema_alpha = 0.05

    for epoch in range(args.num_epochs):
        train_avg = _run_epoch(model, train_loader, noise_scheduler, device, weight_vec,
                               optimizer=optimizer, lr_sched=lr_sched)
        train_ema = train_avg if train_ema is None else ema_alpha * train_avg + (1 - ema_alpha) * train_ema

        val_avg = None
        if val_loader is not None:
            val_avg = _run_epoch(model, val_loader, noise_scheduler, device, weight_vec)
            val_ema = val_avg if val_ema is None else ema_alpha * val_avg + (1 - ema_alpha) * val_ema

        if (epoch + 1) % 10 == 0:
            msg = f"Epoch [{epoch+1:4d}/{args.num_epochs}]  train={train_avg:.4f} (ema={train_ema:.4f})"
            if val_avg is not None:
                gap = val_ema - train_ema
                msg += f"  val={val_avg:.4f} (ema={val_ema:.4f})  gap={gap:+.4f}"
            print(msg)

        if (epoch + 1) % args.checkpoint_every == 0:
            _save_force(args, epoch + 1, model, optimizer, noise_scheduler, train_ds)

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
                _save_force(args, epoch + 1, model, None, noise_scheduler, train_ds,
                           name='policy_final.pt')
                return

    _save_force(args, args.num_epochs, model, None, noise_scheduler, train_ds,
               name='policy_final.pt')
    print(f"\nTraining complete -> {args.output_dir}/policy_final.pt")


def main():
    # Pull force-specific + validation flags out of argv before handing the
    # rest to 05_train.py's own parse_args() -- that parser doesn't know
    # these flags, and it isn't being modified to add them.
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument('--val_episodes', nargs='+', default=None,
                              help='Episode IDs to hold out for validation (e.g. 020).')
    extra_parser.add_argument('--include_episodes', nargs='+', default=None,
                              help='Restrict training (and validation) to only these episode IDs.')
    extra_parser.add_argument('--force_cutoff_hz', type=float, default=2.0,
                              help='Low-pass cutoff (Hz) applied to the dense replay force log '
                                   'before matching to each training frame. 0 disables filtering.')
    extra_parser.add_argument('--tare_window_s', type=float, default=1.0,
                              help='Per-episode tare window (s) at the start of the dense force '
                                   'log, subtracted before filtering.')
    extra_parser.add_argument('--force_loss_weight', type=float, default=0.2,
                              help='Weight on the 3 force dims in the noise-prediction loss, '
                                   'relative to 1.0 on the 9 pose dims -- actions stay the '
                                   'dominant training signal, force is auxiliary. Ignored if '
                                   '--no_predict_force is set.')
    extra_parser.add_argument('--no_predict_force', action='store_true',
                              help='Force is still fed into the OBSERVATION (proprio stays 12D), '
                                   'but the action target is pure 9D pose -- the model is never '
                                   'asked to predict future force. Isolates whether force-'
                                   'conditioning improves pose prediction on its own, separate '
                                   'from whether the model can also predict force well.')
    extra_args, remaining = extra_parser.parse_known_args(sys.argv[1:])

    sys.argv = [sys.argv[0]] + remaining
    train05 = _load_train05_module()
    args = train05.parse_args()
    args.force_cutoff_hz   = extra_args.force_cutoff_hz
    args.tare_window_s     = extra_args.tare_window_s
    args.force_loss_weight = extra_args.force_loss_weight
    args.predict_force     = not extra_args.no_predict_force

    train_with_validation(args, extra_args.val_episodes, extra_args.include_episodes)


if __name__ == '__main__':
    main()
