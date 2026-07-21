#!/usr/bin/env python3
"""
Post-process a replay_episode.py torque_log.npz into EEF wrenches (raw and
gravity-free), via contact_detector.torque_to_wrench. replay_episode.py logs
joint torque only — this is the analysis step that turns it into forces.

Usage:
    python analyze_replay_forces.py --episode_dir data/episodes/newExperiment/001
    python analyze_replay_forces.py --log_path data/episodes/newExperiment/001/replay/torque_log.npz
"""
import os, sys, argparse
import numpy as np

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)
from contact_detector import torque_to_wrench


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--episode_dir', default=None,
                   help='Reads <episode_dir>/replay/torque_log.npz')
    p.add_argument('--log_path', default=None,
                   help='Explicit path to torque_log.npz (overrides --episode_dir)')
    p.add_argument('--damping', type=float, default=0.05,
                   help='Tikhonov damping for the wrench pinv (default 0.05)')
    p.add_argument('--plot_path', default=None,
                   help='Where to save the ||F|| vs time plot (default: alongside the log)')
    return p.parse_args()


def main():
    args = parse_args()
    if args.log_path:
        log_path = args.log_path
    elif args.episode_dir:
        log_path = os.path.join(args.episode_dir, 'replay', 'torque_log.npz')
    else:
        raise SystemExit('Pass --episode_dir or --log_path')

    d = np.load(log_path)
    frame_id = d['frame_id']
    t        = d['t']
    q_deg    = d['q_deg']
    raw      = d['raw_torque']
    gf       = d['gravity_free_torque']
    n = len(frame_id)
    print(f'Loaded {log_path}: {n} steps')

    Fn_raw = np.zeros(n)
    Fn_gf  = np.zeros(n)
    conds  = np.zeros(n)
    F_gf_full = np.zeros((n, 6))
    for i in range(n):
        F_raw = torque_to_wrench(q_deg[i], raw[i], damping=args.damping)
        F_gf  = torque_to_wrench(q_deg[i], gf[i],  damping=args.damping)
        Fn_raw[i] = np.linalg.norm(F_raw[:3])
        Fn_gf[i]  = np.linalg.norm(F_gf[:3])
        F_gf_full[i] = F_gf
        conds[i] = np.linalg.cond(__import__('contact_detector').compute_jacobian(q_deg[i]))

    print(f'\n{"step":>5} {"frame":>6} {"t(s)":>7} {"||F|| raw":>10} {"||F|| gf":>9} {"cond(J)":>8}')
    for i in range(0, n, max(1, n // 25)):   # thin the printout to ~25 rows
        print(f'{i:>5} {frame_id[i]:>6} {t[i]:>7.2f} {Fn_raw[i]:>10.3f} {Fn_gf[i]:>9.3f} {conds[i]:>8.1f}')

    print('\n' + '=' * 60)
    print('SUMMARY (gravity-free ||F||, N)')
    print('=' * 60)
    print(f'  mean = {Fn_gf.mean():.3f}   std = {Fn_gf.std():.3f}')
    print(f'  min  = {Fn_gf.min():.3f}   max = {Fn_gf.max():.3f}')
    print(f'  95th percentile = {np.percentile(Fn_gf, 95):.3f}')
    worst_i = int(np.argmax(Fn_gf))
    print(f'  worst step: {worst_i} (frame {frame_id[worst_i]}, t={t[worst_i]:.2f}s), '
          f'cond(J)={conds[worst_i]:.1f}')
    print(f'\nSUMMARY (raw ||F||, N) -- for reference: mean = {Fn_raw.mean():.3f}, '
          f'max = {Fn_raw.max():.3f}')

    high_cond = conds > 1e4
    if np.any(high_cond):
        print(f'\nWARNING: {high_cond.sum()} steps had cond(J) > 1e4 — wrench there is unreliable.')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(t, Fn_raw, color='#f58231', label='raw')
        axes[0].plot(t, Fn_gf, color='#42d4f4', label='gravity-free')
        axes[0].set_ylabel('||F|| (N)'); axes[0].legend(); axes[0].set_title('EEF force magnitude vs time')
        for j, lbl in enumerate(['Fx', 'Fy', 'Fz']):
            axes[1].plot(t, F_gf_full[:, j], label=lbl)
        axes[1].set_ylabel('Force (N)'); axes[1].set_xlabel('time (s)')
        axes[1].set_title('Gravity-free EEF force components'); axes[1].legend()
        fig.tight_layout()
        plot_path = args.plot_path or os.path.join(os.path.dirname(log_path), 'replay_forces.png')
        fig.savefig(plot_path, dpi=110)
        print(f'\nPlot -> {plot_path}')
    except Exception as e:
        print(f'\n(plot skipped: {e!r})')


if __name__ == '__main__':
    main()
