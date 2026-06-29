"""
Live plot of actual tool trajectory vs policy predicted targets from deploy log.

Converts everything to task frame / tool position so they are directly comparable:
  - Blue  solid line : actual tool xyz in task frame (FP tracking)
  - Orange dashed line: policy predicted tool xyz in task frame
                        (predicted EEF base frame → tool base frame → task frame)

Usage:
    python plot_deploy.py
    python plot_deploy.py --log /tmp/deploy.log
"""

import os, re, argparse
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_calibration():
    """Load T_base_task and T_tool_eef to convert predicted EEF (base) → tool (task)."""
    T_base_task = np.load(os.path.join(PIPELINE_DIR, 'data', 'robot_extrinsics.npy'))  # (4,4)
    T_task_base = np.linalg.inv(T_base_task)

    T_tool_eef  = np.load(os.path.join(PIPELINE_DIR, 'data', 'T_tool_eef.npy'))  # (4,4)
    T_eef_tool  = np.linalg.inv(T_tool_eef)
    return T_task_base, T_eef_tool


def eef_base_to_tool_task(xyz_cm, T_task_base, T_eef_tool):
    """Convert predicted EEF position (base frame, cm) → tool position (task frame, cm)."""
    xyz_m = np.array([*xyz_cm]) / 100.0
    T_base_eef       = np.eye(4); T_base_eef[:3, 3] = xyz_m
    T_task_tool_pred = T_task_base @ T_base_eef @ T_eef_tool
    return T_task_tool_pred[:3, 3] * 100.0   # back to cm


def parse_log(path, T_task_base, T_eef_tool):
    actual = {}   # step -> (x,y,z) tool in task frame, cm
    pred   = {}   # step -> (x,y,z) tool in task frame, cm (converted from predicted EEF)

    pat_actual = re.compile(
        r'step\s+(\d+):\s+tool xyz\(task\)\s*=\s*\(\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\)')
    pat_pred   = re.compile(
        r'step\s+(\d+)\s+k=0:.*?->\s*clamped\s*\(\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\)')

    try:
        with open(path) as f:
            for line in f:
                m = pat_actual.search(line)
                if m:
                    s = int(m.group(1))
                    actual[s] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))

                m = pat_pred.search(line)
                if m:
                    s = int(m.group(1))
                    xyz_eef_base = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
                    if T_task_base is not None:
                        pred[s] = tuple(eef_base_to_tool_task(xyz_eef_base, T_task_base, T_eef_tool))
                    else:
                        pred[s] = xyz_eef_base
    except FileNotFoundError:
        pass

    return actual, pred


def arrays(d):
    if not d:
        return np.array([]), np.array([]), np.array([]), np.array([])
    steps = np.array(sorted(d.keys()))
    xyz   = np.array([d[s] for s in steps])
    return steps, xyz[:, 0], xyz[:, 1], xyz[:, 2]


def setup_ax(ax, ylabel):
    ax.set_facecolor('#2a2a2a')
    ax.tick_params(colors='white', labelsize=8)
    ax.set_ylabel(ylabel, fontsize=8, color='white')
    ax.set_xlabel('step', fontsize=8, color='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#555')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--log',      default='/tmp/deploy.log')
    p.add_argument('--interval', type=float, default=0.5)
    args = p.parse_args()

    os.environ.setdefault('DISPLAY', os.environ.get('DISPLAY', ':1'))

    try:
        T_task_base, T_eef_tool = load_calibration()
        print("Calibration loaded — predictions converted to task-frame tool positions.")
    except Exception as e:
        print(f"WARNING: Could not load calibration ({e}). Plotting raw values (not comparable).")
        T_task_base = T_eef_tool = None

    fig = plt.figure(figsize=(14, 8), facecolor='#1e1e1e')
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)
    ax3 = fig.add_subplot(gs[:, 0], projection='3d')
    axx = fig.add_subplot(gs[0, 1])
    axy = fig.add_subplot(gs[1, 1])
    axz = fig.add_subplot(gs[2, 1])

    plt.ion()
    plt.show()

    print(f"Reading {args.log} — Ctrl+C to stop.")

    while True:
        if T_task_base is not None:
            actual, pred = parse_log(args.log, T_task_base, T_eef_tool)
        else:
            actual, pred = parse_log(args.log, None, None)

        sa, xa, ya, za = arrays(actual)
        sp, xp, yp, zp = arrays(pred)

        for ax in [ax3, axx, axy, axz]:
            ax.cla()

        for ax, ya_v, yp_v, lbl in [(axx, xa, xp, 'X (cm)'),
                                     (axy, ya, yp, 'Y (cm)'),
                                     (axz, za, zp, 'Z (cm)')]:
            setup_ax(ax, lbl)
            if len(sa): ax.plot(sa, ya_v, color='#4fc3f7', lw=1.5, label='actual tool (task)')
            if len(sp): ax.plot(sp, yp_v, color='#ffb74d', lw=1.5, ls='--', label='predicted tool (task)')
            ax.legend(fontsize=7, loc='upper left',
                      facecolor='#333', edgecolor='#555', labelcolor='white')

        ax3.set_facecolor('#2a2a2a')
        ax3.tick_params(colors='white', labelsize=7)
        ax3.set_xlabel('X (cm)', fontsize=7, color='white')
        ax3.set_ylabel('Y (cm)', fontsize=7, color='white')
        ax3.set_zlabel('Z (cm)', fontsize=7, color='white')
        ax3.set_title('Both in task frame — tool positions', color='white', fontsize=8)
        if len(sa):
            ax3.plot(xa, ya, za, color='#4fc3f7', lw=1.5, label='actual')
            ax3.scatter(xa[-1:], ya[-1:], za[-1:], color='#4fc3f7', s=40)
        if len(sp):
            ax3.plot(xp, yp, zp, color='#ffb74d', lw=1.5, ls='--', label='predicted')
            ax3.scatter(xp[-1:], yp[-1:], zp[-1:], color='#ffb74d', s=40)
        if len(sa) or len(sp):
            ax3.legend(fontsize=8, facecolor='#333', edgecolor='#555', labelcolor='white')

        gap = ''
        if len(sa) and len(sp):
            common = sorted(set(actual) & set(pred))
            if common:
                last = common[-1]
                d = np.linalg.norm(np.array(actual[last]) - np.array(pred[last]))
                gap = f'  |  last-step gap = {d:.1f} cm'

        fig.suptitle(
            f'Actual (blue) vs Predicted tool pos — task frame  —  {len(actual)} steps{gap}',
            color='white', fontsize=11)

        fig.canvas.draw_idle()
        plt.pause(args.interval)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nDone.")
