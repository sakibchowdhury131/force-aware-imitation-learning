#!/usr/bin/env python3
"""
Offline diagnostic (no robot, no live camera): feed a REPLAY-trained
checkpoint the exact training-distribution images/proprio it saw at train
time, and compare its predicted action-horizon poses to the ground-truth
demo-tracked labels for the same window.

WHY: isolates "did the model learn the task at all" from "is there a live
deployment/perception bug". A model that converged to train_ema ~0.02-0.04
should reproduce training-episode trajectories closely. If it doesn't even
here, the checkpoint itself is the problem (data quantity / action_horizon /
architecture). If it DOES reproduce training data well but failed live on
the robot, the bug is in live perception/calibration/frame-conversion
instead -- collecting more data or changing action_horizon would not fix
that.

Usage:
    python eval_replay_offline.py \
        --checkpoint data/checkpoints/PastaTransfer_force_replay/policy_final.pt \
        --data_dir data/episodes/PastaTransfer_force \
        --episodes 001 020
"""
import os, sys, argparse, importlib.util
import numpy as np
import torch

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)

from policy_common import MaxAbsNormalizer, DiffusionPolicyNet, action_9d_to_pose


def _load_module(fname, alias):
    spec = importlib.util.spec_from_file_location(alias, os.path.join(PIPELINE_DIR, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pose_errors(T_pred, T_gt):
    from scipy.spatial.transform import Rotation
    trans_err = float(np.linalg.norm(T_pred[:3, 3] - T_gt[:3, 3]) * 100.0)  # cm
    R_rel = Rotation.from_matrix(T_pred[:3, :3].T @ T_gt[:3, :3])
    rot_err = float(np.degrees(R_rel.magnitude()))
    return trans_err, rot_err


@torch.no_grad()
def predict_window(model, noise_scheduler, obs_imgs, proprio, device):
    obs_imgs = obs_imgs.unsqueeze(0).to(device)   # (1, T, V, 3, H, W)
    proprio  = proprio.unsqueeze(0).to(device)    # (1, T, 9)
    obs_emb  = model.encode_obs(obs_imgs, proprio)
    action_dim = model.action_horizon * 9
    noisy = torch.randn(1, action_dim, device=device)
    noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps)
    for t in noise_scheduler.timesteps:
        pred = model(noisy, t.unsqueeze(0).to(device), obs_emb)
        noisy = noise_scheduler.step(pred, t, noisy).prev_sample
    return noisy[0].cpu().numpy().reshape(model.action_horizon, 9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data_dir', default='data/episodes/PastaTransfer_force')
    p.add_argument('--episodes', nargs='+', required=True,
                   help='Episode IDs to evaluate, e.g. 001 020')
    p.add_argument('--track_cam', type=int, default=1)
    p.add_argument('--n_windows', type=int, default=8,
                   help='Number of windows to sample (evenly spaced across the episodes)')
    p.add_argument('--full_sweep', action='store_true',
                   help='Evaluate every window in order and print predicted vs ground-truth '
                        'displacement magnitude per window, to see where in the trajectory '
                        'the model stalls (predicts near-zero motion instead of progressing).')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    device = torch.device(args.device)
    train05_replay = _load_module('05_train_replay.py', '_train05_replay')

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = DiffusionPolicyNet(
        action_dim=9, action_horizon=ckpt['action_horizon'], n_obs_steps=ckpt['n_obs_steps'],
        n_views=ckpt['n_views'], proprio_dim=ckpt['proprio_dim'],
        unet_dims=tuple(ckpt['unet_dims']), unet_kernel=ckpt['unet_kernel'],
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    normalizer = MaxAbsNormalizer.from_state_dict(ckpt['normalizer'])
    noise_scheduler = ckpt['noise_scheduler']

    print(f"Checkpoint: {args.checkpoint}")
    print(f"  n_views={ckpt['n_views']} n_obs_steps={ckpt['n_obs_steps']} "
          f"action_horizon={ckpt['action_horizon']} action_frame={ckpt.get('action_frame')}")

    ds = train05_replay.ReplayImageWindowDataset(
        data_dir=args.data_dir, n_obs_steps=ckpt['n_obs_steps'],
        action_horizon=ckpt['action_horizon'], image_size=ckpt['image_size'],
        crop_size=ckpt['crop_size'], training=False, track_cam=args.track_cam,
        n_views=ckpt['n_views'], subsample=3, action_frame=ckpt.get('action_frame', 'task'),
        include_episodes=args.episodes, external_normalizer=normalizer,
    )

    n = len(ds)
    if args.full_sweep:
        idxs = np.arange(n)
    else:
        idxs = np.linspace(0, n - 1, min(args.n_windows, n)).astype(int)

    all_trans, all_rot = [], []
    for idx in idxs:
        obs_imgs, proprio, act_seq = ds[idx]
        pred_norm = predict_window(model, noise_scheduler, obs_imgs, proprio, device)
        pred = normalizer.denormalize(pred_norm)
        gt   = normalizer.denormalize(act_seq.numpy())

        step_trans, step_rot = [], []
        for k in range(pred.shape[0]):
            T_pred = action_9d_to_pose(pred[k])
            T_gt   = action_9d_to_pose(gt[k])
            te, re = pose_errors(T_pred, T_gt)
            step_trans.append(te); step_rot.append(re)
        all_trans.append(step_trans); all_rot.append(step_rot)
        mid = len(step_trans) // 2

        if args.full_sweep:
            # displacement magnitude (first->last horizon step) -- collapses
            # toward 0 if the model predicts "stay put" instead of progressing
            gt_disp   = float(np.linalg.norm(gt[-1, :3] - gt[0, :3]) * 100.0)
            pred_disp = float(np.linalg.norm(pred[-1, :3] - pred[0, :3]) * 100.0)
            cur_x     = float(normalizer.denormalize(proprio.numpy())[-1, 0] * 100.0)
            print(f"  window {idx:4d}  x={cur_x:6.2f}cm  gt_disp={gt_disp:5.2f}cm  "
                  f"pred_disp={pred_disp:5.2f}cm  trans_err(last)={step_trans[-1]:.2f}cm")
        else:
            print(f"  window {idx:4d}: trans_err cm  step1={step_trans[0]:.2f}  "
                  f"mid={step_trans[mid]:.2f}  last={step_trans[-1]:.2f}   "
                  f"rot_err deg  step1={step_rot[0]:.2f}  last={step_rot[-1]:.2f}")

    all_trans = np.array(all_trans)   # (n_windows, horizon)
    all_rot   = np.array(all_rot)
    mid = all_trans.shape[1] // 2
    print(f"\nMEAN over {len(idxs)} windows (episodes={args.episodes}):")
    print(f"  translation error (cm):  step1={all_trans[:,0].mean():.2f}  "
          f"mid={all_trans[:,mid].mean():.2f}  last={all_trans[:,-1].mean():.2f}   "
          f"overall={all_trans.mean():.2f}")
    print(f"  rotation error (deg):    step1={all_rot[:,0].mean():.2f}  "
          f"last={all_rot[:,-1].mean():.2f}   overall={all_rot.mean():.2f}")


if __name__ == '__main__':
    main()
