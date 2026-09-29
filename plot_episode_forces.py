#!/usr/bin/env python3
"""
Plot Fx, Fy, Fz from an episode's replay_full_forces.npz as three separate
(stacked) subplots. Pass --include_moments to also plot Mx, My, Mz (the
external_moment_xyz field analyze_replay_full.py already saves alongside
external_force_xyz, but which nothing in this project's training/deploy
pipeline actually uses -- see FORCE_AWARE_POLICY_REPORT.txt section 4.2/4.4).

Usage:
    python plot_episode_forces.py --episode_dir data/episodes/PastaTransfer_force_ep20-37/021
    python plot_episode_forces.py --log_path <path>/replay_full_forces.npz --save_path out.png
    python plot_episode_forces.py --episode_dir ... --include_moments   # full 6D wrench
"""
import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--episode_dir', default='data/episodes/PastaTransfer_force_ep20-37/021',
                   help='Reads <episode_dir>/replay/replay_full_forces.npz')
    p.add_argument('--log_path', default=None,
                   help='Explicit path to replay_full_forces.npz (overrides --episode_dir)')
    p.add_argument('--cutoff_hz', type=float, default=5.0,
                   help='Low-pass cutoff (Hz) to smooth out high-frequency oscillations. '
                        '0 disables filtering.')
    p.add_argument('--filter_order', type=int, default=2, help='Butterworth filter order.')
    p.add_argument('--hide_raw', action='store_true',
                   help='Only plot the filtered trace (default overlays raw, faded, underneath it).')
    p.add_argument('--downsample_hz', type=float, default=None,
                   help='Also overlay the filtered signal decimated to this rate (e.g. 2.0) -- '
                        'nearest-sample match into the filtered trace, same approach '
                        '05_train_replay_force.py uses to build per-frame training labels. '
                        'Shows whether coarse sampling loses visible detail beyond what the '
                        'low-pass filter itself already removes.')
    p.add_argument('--include_moments', action='store_true',
                   help='Also plot Mx, My, Mz (external_moment_xyz) below the 3 force rows -- '
                        'the full 6D wrench instead of just the 3D force. Same filter/downsample '
                        'treatment as the force rows.')
    p.add_argument('--save_path', default=None,
                   help='Save the figure here (e.g. forces.png) instead of / in addition to showing it')
    p.add_argument('--no_show', action='store_true', help='Skip plt.show() (useful with --save_path)')
    return p.parse_args()


def _filter_and_downsample(t, X, cutoff_hz, filter_order, downsample_hz):
    """X: (N, 3). Returns (X_filt or None, t_ds or None, X_ds or None)."""
    X_filt = None
    if cutoff_hz > 0:
        fs = 1.0 / np.mean(np.diff(t))   # samples aren't perfectly uniform; mean rate is close enough for filtfilt
        sos_b, sos_a = butter(filter_order, cutoff_hz, btype='low', fs=fs)
        X_filt = np.stack([filtfilt(sos_b, sos_a, X[:, i]) for i in range(3)], axis=1)

    t_ds = X_ds = None
    if downsample_hz is not None:
        if X_filt is None:
            raise SystemExit('--downsample_hz requires filtering to be enabled (--cutoff_hz > 0).')
        t_ds = np.arange(t[0], t[-1], 1.0 / downsample_hz)
        idx = np.searchsorted(t, t_ds)
        idx = np.clip(idx, 0, len(t) - 1)
        closer = (idx > 0) & (np.abs(t[np.clip(idx - 1, 0, None)] - t_ds) < np.abs(t[idx] - t_ds))
        idx = np.where(closer, idx - 1, idx)
        X_ds = X_filt[idx]
    return X_filt, t_ds, X_ds


def _plot_rows(axes, t, X, X_filt, t_ds, X_ds, labels, unit, colors, args):
    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        if X_filt is not None:
            if not args.hide_raw:
                ax.plot(t, X[:, i], color=color, linewidth=0.8, alpha=0.3, label='raw (47Hz)')
            ax.plot(t, X_filt[:, i], color=color, linewidth=1.6,
                    label=f'filtered <{args.cutoff_hz:.0f}Hz (47Hz)')
            if X_ds is not None:
                ax.plot(t_ds, X_ds[:, i], color='black', linewidth=1.0, linestyle='--',
                        marker='o', markersize=4, alpha=0.8,
                        label=f'downsampled ({args.downsample_hz:.0f}Hz)')
            ax.legend(loc='upper right', fontsize=8, framealpha=0.7)
        else:
            ax.plot(t, X[:, i], color=color, linewidth=1.2)
        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        ax.set_ylabel(f'{label} ({unit})')
        ax.grid(True, alpha=0.3)


def main():
    args = parse_args()
    log_path = args.log_path or os.path.join(args.episode_dir, 'replay', 'replay_full_forces.npz')
    d = np.load(log_path)
    t = d['t']
    F = d['external_force_xyz']
    print(f"Loaded {log_path}: {len(t)} samples over {t[-1]-t[0]:.1f}s")

    F_filt, t_ds, F_ds = _filter_and_downsample(t, F, args.cutoff_hz, args.filter_order, args.downsample_hz)
    if args.cutoff_hz > 0:
        fs = 1.0 / np.mean(np.diff(t))
        print(f"Low-pass filtered at {args.cutoff_hz:.1f}Hz (fs~={fs:.1f}Hz, order={args.filter_order})")
    if F_ds is not None:
        print(f"Downsampled (nearest-match into the filtered trace) to {args.downsample_hz:.1f}Hz "
              f"-> {len(t_ds)} samples")

    M = M_filt = M_ds = None
    if args.include_moments:
        if 'external_moment_xyz' not in d.files:
            raise SystemExit(f"--include_moments requested but 'external_moment_xyz' not found in {log_path} "
                              f"(re-run analyze_replay_full.py on this episode to regenerate it).")
        M = d['external_moment_xyz']
        M_filt, _, M_ds = _filter_and_downsample(t, M, args.cutoff_hz, args.filter_order, args.downsample_hz)

    n_rows = 6 if args.include_moments else 3
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 16 if args.include_moments else 8), sharex=True)
    force_colors = ['#2a78d6', '#1baf7a', '#eda100']
    moment_colors = ['#a23b8f', '#d6544e', '#4e9e9e']

    _plot_rows(axes[:3], t, F, F_filt, t_ds, F_ds, ['Fx', 'Fy', 'Fz'], 'N', force_colors, args)
    if args.include_moments:
        _plot_rows(axes[3:], t, M, M_filt, t_ds, M_ds, ['Mx', 'My', 'Mz'], 'N·m', moment_colors, args)

    axes[-1].set_xlabel('time (s)')
    title = 'Contact wrench (force + moment)' if args.include_moments else 'Contact force'
    fig.suptitle(f'{title} — {os.path.basename(os.path.dirname(os.path.dirname(log_path)))}')
    fig.tight_layout()

    if args.save_path:
        fig.savefig(args.save_path, dpi=150)
        print(f"Saved -> {args.save_path}")
    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
