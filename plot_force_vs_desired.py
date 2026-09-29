#!/usr/bin/env python3
"""
Plot F_live (actual, tared+filtered live force) vs f_desired (the pose+force
policy's predicted future force -- the admittance reference) side by side,
per axis, from a 07_deploy_force_mdk.py (or 07_deploy_force.py) force_log
.npz. Requires a log that has 'f_desired' saved -- only force_log_mdk_*.npz
files produced after the f_desired-logging addition have this field; older
force_log_*.npz files from 07_deploy_force.py do not.

Usage:
    python plot_force_vs_desired.py --log_path deploymentRuns/.../force_log_mdk_20260824_132833.npz
"""
import os, argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--log_path', required=True, help='force_log_mdk_*.npz with an f_desired field')
    p.add_argument('--save_path', default=None,
                   help='Save the figure here (e.g. force_vs_desired.png) instead of / in addition to showing it')
    p.add_argument('--no_show', action='store_true', help='Skip plt.show() (useful with --save_path)')
    return p.parse_args()


def main():
    args = parse_args()
    d = np.load(args.log_path)
    if 'f_desired' not in d.files:
        raise SystemExit(f"'f_desired' not found in {args.log_path} -- this log predates the "
                         f"f_desired-logging addition, or was produced by a script that never "
                         f"had it (e.g. 07_deploy_force.py). Re-run with the current "
                         f"07_deploy_force_mdk.py to get a log with this field.")

    t = d['t']
    F_live = d['F_live']
    f_desired = d['f_desired']
    valid = ~np.isnan(f_desired[:, 0])   # NaN on "waiting for inference"/paused ticks
    print(f"Loaded {args.log_path}: {len(t)} samples, {valid.sum()} with a valid f_desired")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ['Fx', 'Fy', 'Fz']
    colors = ['#2a78d6', '#1baf7a', '#eda100']
    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.plot(t, F_live[:, i], color=color, linewidth=1.4, label='F_live (actual)')
        ax.plot(t[valid], f_desired[valid, i], color='black', linewidth=1.6, linestyle='--',
                marker='o', markersize=3, alpha=0.85, label='f_desired (predicted)')
        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        ax.set_ylabel(f'{label} (N)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.7)
    axes[-1].set_xlabel('time (s)')
    fig.suptitle(f'Actual vs. predicted (desired) force — {os.path.basename(args.log_path)}')
    fig.tight_layout()

    if args.save_path:
        fig.savefig(args.save_path, dpi=150)
        print(f"Saved -> {args.save_path}")
    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
