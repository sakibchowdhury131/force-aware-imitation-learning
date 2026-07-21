#!/usr/bin/env python3
"""
Full external-force analysis of a replay_episode.py dense log, using the
FINAL recommended calibration pipeline from FORCE_SENSING_FINDINGS.md:
  1. firmware gravity matrix   (already baked into gravity_free_torque)
  2. software gravity regressor (contact_detector.gravity_regressor @ phi_task_only)
  3. gravity-residual NN        (fit_gravity_residual_nn -- static-data-only correction)
  4. mass/Coriolis REGRESSOR    (full_dynamics_regressor @ dynamics_residual_pi)

Three tiers reported: firmware-only, + gravity (regressor+NN), final estimate
(+ mass/Coriolis regressor). Same three tiers diag_replay_forces.py plots
live -- this is the saved-log/post-hoc equivalent.

Saves the actual computed force data (not just a plot/printout) to
<log_dir>/replay_full_forces.npz: per-timestep t, external_force_xyz (N, the
pure external force at the EEF, base frame -- this is the number that
matters), external_moment_xyz (N*m), plus all three tiers' ||F|| norms and
q_deg/qdot_deg for reference. Pass --output_npz "" to skip saving.

Also tags the force at each WAYPOINT time instant (not just the dense
samples), if <log_dir>/torque_log.npz is present (replay_episode.py always
saves this alongside torque_log_dense.npz) -- these are the exact time
instants --capture_camera saves images at (data/.../replay/cam{N}/{frame_id}.jpg),
so this is how you map images <-> forces. For each waypoint's (frame_id, t),
the nearest dense sample's already-computed force is looked up (dense data
has continuous qdot for the dynamics correction; the sparse per-waypoint log
does not, so this resamples the dense result rather than recomputing cruder
sparse-only estimates). Saved to <log_dir>/replay_forces_per_frame.npz:
frame_id, t, external_force_xyz, external_moment_xyz, Fn_final, and
match_dt (how far the matched dense sample's timestamp was from the
waypoint's -- should be small, ~1/dense_hz; large values mean sparse
sampling or a gap in the dense log).

Usage:
    python analyze_replay_full.py --episode_dir data/episodes/pastaTransfer4/007
    python analyze_replay_full.py --log_path data/episodes/.../replay_smooth/torque_log_dense.npz
"""
import os, sys, argparse
import numpy as np
import torch

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)
from contact_detector import torque_to_wrench, gravity_regressor, full_dynamics_regressor, VelocityDifferentiator
from fit_gravity_residual_nn import GravityResidualNet, predict as predict_gravity_residual


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--episode_dir', default=None, help='Reads <episode_dir>/replay/torque_log_dense.npz')
    p.add_argument('--log_path', default=None, help='Explicit path to torque_log_dense.npz (overrides --episode_dir)')
    p.add_argument('--phi', default='data/gravity_phi_task_only.npy')
    p.add_argument('--gravity_nn', default='data/gravity_residual_nn.pt')
    p.add_argument('--dynamics_pi', default='data/dynamics_residual_pi.npy')
    p.add_argument('--damping', type=float, default=0.05)
    p.add_argument('--qddot_smoothing', type=float, default=0.3)
    p.add_argument('--plot_path', default=None)
    p.add_argument('--output_npz', default=None,
                   help='Where to save the computed force time-series (default: '
                        '<log_dir>/replay_full_forces.npz). Pass "" to skip saving.')
    p.add_argument('--torque_log_path', default=None,
                   help='Explicit path to the sparse per-waypoint torque_log.npz, for frame '
                        'tagging. Default: torque_log.npz next to the dense log.')
    p.add_argument('--output_per_frame_npz', default=None,
                   help='Where to save per-waypoint (frame_id-tagged) forces (default: '
                        '<log_dir>/replay_forces_per_frame.npz). Pass "" to skip.')
    return p.parse_args()


def main():
    args = parse_args()
    if args.log_path:
        log_path = args.log_path
    elif args.episode_dir:
        log_path = os.path.join(args.episode_dir, 'replay', 'torque_log_dense.npz')
    else:
        raise SystemExit('Pass --episode_dir or --log_path')
    d = np.load(log_path)
    t, q_deg, qdot_deg, gf = d['t'], d['q_deg'], d['qdot_deg'], d['gravity_free_torque']
    n = len(t)
    print(f'Loaded {log_path}: {n} dense samples over {t[-1]-t[0]:.1f}s')

    phi = np.load(os.path.join(PIPELINE_DIR, args.phi))
    ckpt = torch.load(os.path.join(PIPELINE_DIR, args.gravity_nn), weights_only=False)
    g_nn = GravityResidualNet(); g_nn.load_state_dict(ckpt['state_dict']); g_nn.eval()
    g_x_mean, g_x_std = ckpt['x_mean'], ckpt['x_std']
    pi = np.load(os.path.join(PIPELINE_DIR, args.dynamics_pi))

    qdiff = VelocityDifferentiator(smoothing=args.qddot_smoothing)
    Fn_fw, Fn_grav, Fn_final = np.zeros(n), np.zeros(n), np.zeros(n)
    # Full 6D wrench [Fx,Fy,Fz,Mx,My,Mz] at the EEF, base frame -- the actual
    # pure external force/moment, not just its norm. This is the real "saved
    # force data" output; Fn_* above are just a convenience scalar for plotting.
    wrench_final = np.zeros((n, 6))

    for i in range(n):
        qddot_i = qdiff.update(qdot_deg[i], t[i])
        F_fw = torque_to_wrench(q_deg[i], gf[i], damping=args.damping)

        g_res_lin = gravity_regressor(q_deg[i]) @ phi
        g_res_nn = predict_gravity_residual(g_nn, g_x_mean, g_x_std, q_deg[i])
        tau_after_gravity = gf[i] - g_res_lin - g_res_nn
        F_grav = torque_to_wrench(q_deg[i], tau_after_gravity, damping=args.damping)

        dyn_pred = full_dynamics_regressor(q_deg[i], qdot_deg[i], qddot_i) @ pi
        tau_final = tau_after_gravity - dyn_pred
        F_final = torque_to_wrench(q_deg[i], tau_final, damping=args.damping)

        Fn_fw[i]    = np.linalg.norm(F_fw[:3])
        Fn_grav[i]  = np.linalg.norm(F_grav[:3])
        Fn_final[i] = np.linalg.norm(F_final[:3])
        wrench_final[i] = F_final

    print(f'\n{"":24} {"mean":>8} {"std":>8} {"min":>8} {"max":>8}')
    for label, arr in [('firmware-only', Fn_fw), ('+ gravity (regressor+NN)', Fn_grav),
                       ('final estimate (+dynamics)', Fn_final)]:
        print(f'{label:24} {arr.mean():>8.3f} {arr.std():>8.3f} {arr.min():>8.3f} {arr.max():>8.3f}')

    print(f'\nReduction, firmware -> +gravity: {(Fn_fw.mean()-Fn_grav.mean())/Fn_fw.mean()*100:.1f}%')
    print(f'Reduction, +gravity -> final:    {(Fn_grav.mean()-Fn_final.mean())/Fn_grav.mean()*100:.1f}%')
    print(f'Reduction, firmware -> final (total): {(Fn_fw.mean()-Fn_final.mean())/Fn_fw.mean()*100:.1f}%')

    peak_i = int(np.argmax(Fn_final))
    print(f'\nPeak final-estimate ||F|| = {Fn_final[peak_i]:.2f} N at t={t[peak_i]:.2f}s '
         f'(check this against the mean/std above -- a real push looks like a smooth '
         f'rise-peak-fall over ~0.3-0.5s, not an isolated single-sample spike)')

    if args.output_npz != '':
        npz_path = args.output_npz or os.path.join(os.path.dirname(log_path), 'replay_full_forces.npz')
        np.savez(npz_path,
                 t=t,
                 external_force_xyz=wrench_final[:, :3],   # the pure external force, base frame (N)
                 external_moment_xyz=wrench_final[:, 3:],  # accompanying moment, base frame (N*m)
                 Fn_firmware_only=Fn_fw,
                 Fn_plus_gravity=Fn_grav,
                 Fn_final=Fn_final,
                 q_deg=q_deg, qdot_deg=qdot_deg)
        print(f'Saved calibrated force data -> {npz_path}')

    # ── Tag the force at each WAYPOINT instant (image/force alignment) ──────
    if args.output_per_frame_npz != '':
        sparse_path = args.torque_log_path or os.path.join(os.path.dirname(log_path), 'torque_log.npz')
        if not os.path.exists(sparse_path):
            print(f'\n(per-frame tagging skipped: {sparse_path} not found)')
        else:
            sd = np.load(sparse_path)
            wp_frame_id, wp_t = sd['frame_id'], sd['t']
            n_wp = len(wp_t)

            # Nearest dense sample per waypoint (dense t is monotonically increasing).
            idx = np.searchsorted(t, wp_t)
            idx = np.clip(idx, 0, n - 1)
            left = np.clip(idx - 1, 0, n - 1)
            use_left = np.abs(t[left] - wp_t) < np.abs(t[idx] - wp_t)
            nearest = np.where(use_left, left, idx)
            match_dt = np.abs(t[nearest] - wp_t)

            per_frame_path = (args.output_per_frame_npz
                             or os.path.join(os.path.dirname(log_path), 'replay_forces_per_frame.npz'))
            np.savez(per_frame_path,
                     frame_id=wp_frame_id,
                     t=wp_t,
                     external_force_xyz=wrench_final[nearest, :3],
                     external_moment_xyz=wrench_final[nearest, 3:],
                     Fn_final=Fn_final[nearest],
                     match_dt=match_dt)
            print(f'\nSaved per-frame (waypoint) tagged forces -> {per_frame_path}  '
                 f'({n_wp} waypoints, mean|match_dt|={match_dt.mean()*1000:.1f}ms, '
                 f'max={match_dt.max()*1000:.1f}ms)')
            print('  frame_id here matches --capture_camera\'s saved image filenames '
                 '({frame_id:06d}.jpg) -- use it to pair images with forces.')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t, Fn_fw, color='#f58231', linewidth=1, label='firmware-only', alpha=0.7)
        ax.plot(t, Fn_grav, color='#3cb44b', linewidth=1.2, label='+ gravity (regressor+NN)')
        ax.plot(t, Fn_final, color='#4363d8', linewidth=1.6, label='final estimate (+dynamics)')
        ax.set_xlabel('time (s)'); ax.set_ylabel('||F|| (N)')
        ax.set_title('External force estimate — three correction tiers')
        ax.legend()
        fig.tight_layout()
        plot_path = args.plot_path or os.path.join(os.path.dirname(log_path), 'replay_full_forces.png')
        fig.savefig(plot_path, dpi=110)
        print(f'\nPlot -> {plot_path}')
    except Exception as e:
        print(f'\n(plot skipped: {e!r})')


if __name__ == '__main__':
    main()
