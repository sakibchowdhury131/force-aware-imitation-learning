#!/usr/bin/env python3
"""
Offline evaluation of a force-conditioned checkpoint's (05_train_replay_force.py)
predicted force against the actual recorded force, on a held-out episode --
no robot needed. Reuses 05_train_replay_force.py's dataset class and
test_policy.py's real DDPM inference path (same predict_action_sequence used
live), just against already-recorded data instead of a live camera/robot.

Compares two things per horizon step:
  - model:    the checkpoint's predicted force at that step
  - baseline: "force stays whatever it was in the last observed step" (naive
              zero-change predictor) -- context for whether the model is
              adding real predictive signal, not just tracking a slowly
              varying baseline.

Usage:
    python eval_force_prediction.py \\
        --checkpoint data/checkpoints/PastaTransfer_force_replay_force/policy_final.pt \\
        --episode 037
"""
import os, sys, argparse, importlib.util
import numpy as np
import torch

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)

from test_policy import load_model, predict_action_sequence


def _load_train_force_module():
    spec = importlib.util.spec_from_file_location(
        '_train_force', os.path.join(PIPELINE_DIR, '05_train_replay_force.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data_dir', default='data/episodes/PastaTransfer_force')
    p.add_argument('--episode', default='037', help='Held-out episode ID to evaluate on.')
    p.add_argument('--track_cam', type=int, default=1)
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    train_force = _load_train_force_module()

    print(f"Loading checkpoint: {args.checkpoint}")
    model, normalizer, ckpt = load_model(args.checkpoint, device)
    if not ckpt.get('predicts_force', False):
        raise SystemExit("This checkpoint has predicts_force=False -- nothing to evaluate.")

    noise_scheduler = ckpt['noise_scheduler']
    noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps)

    n_obs_steps    = ckpt.get('n_obs_steps', 2)
    n_views        = ckpt.get('n_views', 1)
    action_horizon = ckpt['action_horizon']
    action_frame   = ckpt.get('action_frame', 'task')
    force_cutoff_hz = ckpt.get('force_cutoff_hz', 2.0)
    tare_window_s   = ckpt.get('tare_window_s', 1.0)

    print(f"Building validation set: episode {args.episode}  "
          f"(force_cutoff_hz={force_cutoff_hz}, tare_window_s={tare_window_s})")
    val_ds = train_force.ReplayImageForceWindowDataset(
        data_dir=args.data_dir, n_obs_steps=n_obs_steps, action_horizon=action_horizon,
        image_size=ckpt.get('image_size', 128), crop_size=ckpt.get('crop_size', 115),
        training=False, cam0_only=False, track_cam=args.track_cam, n_views=n_views,
        subsample=3, action_frame=action_frame,
        include_episodes=[args.episode], external_normalizer=normalizer,
        force_cutoff_hz=force_cutoff_hz, tare_window_s=tare_window_s,
    )
    n = len(val_ds)
    print(f"{n} windows to evaluate\n")

    # Per-window: (action_horizon, 3) errors, accumulated across all windows.
    model_err = np.zeros((n, action_horizon, 3))
    base_err  = np.zeros((n, action_horizon, 3))
    gt_force_all = np.zeros((n, action_horizon, 3))

    model.eval()
    for i in range(n):
        obs_imgs, proprio_norm, act_seq_norm = val_ds[i]
        view_tensors = [obs_imgs[t] for t in range(n_obs_steps)]
        proprio_list = [proprio_norm[t].numpy() for t in range(n_obs_steps)]

        actions_raw, _ = predict_action_sequence(
            model, normalizer, noise_scheduler, view_tensors, proprio_list, device)
        force_pred = actions_raw[:, 9:12]

        act_seq_raw = normalizer.denormalize(act_seq_norm.numpy())
        force_gt = act_seq_raw[:, 9:12]

        proprio_raw = normalizer.denormalize(proprio_norm.numpy())
        force_last_obs = proprio_raw[-1, 9:12]   # last observed force -- the naive baseline

        model_err[i] = force_pred - force_gt
        base_err[i]  = force_last_obs[None, :] - force_gt
        gt_force_all[i] = force_gt

        if (i + 1) % 20 == 0 or i == n - 1:
            print(f"  evaluated {i+1}/{n} windows")

    print("\n" + "=" * 70)
    print(f"Force prediction accuracy on episode {args.episode} ({n} windows, "
          f"{action_horizon}-step horizon)")
    print("=" * 70)

    gt_std = gt_force_all.reshape(-1, 3).std(axis=0)
    print(f"\nGround-truth force std per axis (context for MAE below): "
          f"Fx={gt_std[0]:.2f}  Fy={gt_std[1]:.2f}  Fz={gt_std[2]:.2f} N")

    model_mae_overall = np.abs(model_err).mean()
    base_mae_overall  = np.abs(base_err).mean()
    model_rmse_overall = np.sqrt((model_err ** 2).mean())
    base_rmse_overall  = np.sqrt((base_err ** 2).mean())
    print(f"\nOverall (all axes, all horizon steps):")
    print(f"  model:    MAE={model_mae_overall:.3f}N   RMSE={model_rmse_overall:.3f}N")
    print(f"  baseline: MAE={base_mae_overall:.3f}N   RMSE={base_rmse_overall:.3f}N")
    improvement = (base_mae_overall - model_mae_overall) / base_mae_overall * 100
    print(f"  model {'beats' if improvement > 0 else 'is worse than'} the naive baseline "
          f"by {improvement:+.1f}% (MAE)")

    print(f"\nPer-axis MAE (N):")
    for k, label in enumerate(['Fx', 'Fy', 'Fz']):
        m = np.abs(model_err[:, :, k]).mean()
        b = np.abs(base_err[:, :, k]).mean()
        print(f"  {label}: model={m:.3f}  baseline={b:.3f}")

    print(f"\nMAE by horizon step (model vs baseline, all axes averaged):")
    for h in [0, 3, 7, 11, 15] if action_horizon > 15 else range(action_horizon):
        m = np.abs(model_err[:, h, :]).mean()
        b = np.abs(base_err[:, h, :]).mean()
        print(f"  step {h+1:2d}: model={m:.3f}N  baseline={b:.3f}N")

    out_path = '/tmp/eval_force_prediction_results.npz'
    np.savez(out_path, model_err=model_err, base_err=base_err, gt_force=gt_force_all)
    print(f"\nRaw per-window errors saved -> {out_path}")


if __name__ == '__main__':
    main()
