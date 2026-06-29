"""
Live 3D plot of robot EEF pose and predicted action horizon during deployment.

Reads /tmp/deploy_state.npz written by 07_deploy.py each control step.

Shows:
  - Blue sphere  : current EEF position (base frame)
  - Blue arrows  : current EEF X/Y/Z axes (orientation)
  - Orange dots  : predicted future EEF positions (step+1 = brightest, step+8 = fading)
  - Grey trail   : history of actual EEF positions

Run alongside 07_deploy.py:
    python plot_3d_deploy.py
"""

import os, time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

STATE_FILE   = '/tmp/deploy_state.npz'
TRAIL_LEN    = 80     # how many past EEF positions to show
AXIS_LEN     = 0.02   # metres — length of orientation arrows
REFRESH_S    = 0.1    # seconds between plot updates

# EEF axis colours: X=red, Y=green, Z=blue
AXIS_COLORS  = ['#e74c3c', '#2ecc71', '#3498db']


def draw_axes_3d(ax, T, length=AXIS_LEN):
    """Draw XYZ arrows at the pose given by 4x4 matrix T (base frame, metres)."""
    o = T[:3, 3]
    for i, col in enumerate(AXIS_COLORS):
        d = T[:3, i] * length
        ax.quiver(o[0], o[1], o[2], d[0], d[1], d[2],
                  color=col, linewidth=2, arrow_length_ratio=0.3)


def setup_ax(ax):
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='white', labelsize=7)
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    ax.set_xlabel('X (m)', fontsize=8)
    ax.set_ylabel('Y (m)', fontsize=8)
    ax.set_zlabel('Z (m)', fontsize=8)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#333')


def main():
    fig = plt.figure(figsize=(9, 8), facecolor='#0d0d1a')
    ax  = fig.add_subplot(111, projection='3d')
    fig.suptitle('EEF pose (blue) + predicted horizon (orange) — base frame',
                 color='white', fontsize=11)
    plt.ion()
    plt.show()

    trail = []   # list of (3,) arrays — actual EEF history

    print(f"Watching {STATE_FILE} — Ctrl+C to stop.")

    last_step = -1
    while True:
        try:
            if not os.path.exists(STATE_FILE):
                plt.pause(REFRESH_S)
                continue

            data = np.load(STATE_FILE)
            step = int(data['step'][0])
            if step == last_step:
                plt.pause(REFRESH_S)
                continue
            last_step = step

            T_eef      = data['T_base_eef']          # (4,4)
            poses_pred = data['poses_eef_pred']       # (N,4,4)

            eef_pos = T_eef[:3, 3]
            trail.append(eef_pos.copy())
            if len(trail) > TRAIL_LEN:
                trail.pop(0)

            ax.cla()
            setup_ax(ax)

            # --- trail ---
            if len(trail) > 1:
                tr = np.array(trail)
                ax.plot(tr[:, 0], tr[:, 1], tr[:, 2],
                        color='#5dade2', lw=1, alpha=0.5, label='actual trail')

            # --- current EEF ---
            ax.scatter(*eef_pos, s=120, color='#5dade2',
                       edgecolors='white', linewidths=1.5,
                       depthshade=False, label='current EEF', zorder=5)
            draw_axes_3d(ax, T_eef)

            # --- predicted horizon ---
            H = len(poses_pred)
            for k in range(H - 1, -1, -1):
                alpha  = 0.3 + 0.7 * (H - k) / H
                size   = 20 + 80 * (H - k) / H
                color  = (1.0, 0.4 * alpha, 0.0)   # orange fading to dark
                pos    = poses_pred[k][:3, 3]
                ax.scatter(*pos, s=size, color=[color],
                           edgecolors='white', linewidths=0.5,
                           depthshade=False, alpha=float(alpha))

            # draw full axes for step+1 (k=0, the nearest prediction)
            if H > 0:
                draw_axes_3d(ax, poses_pred[0], length=AXIS_LEN)
                ax.scatter(*poses_pred[0][:3, 3], s=180, color='#f39c12',
                           edgecolors='white', linewidths=1.5,
                           depthshade=False, zorder=5, label='predicted step+1')

            # --- axis limits: centre on current EEF with ±0.25 m window ---
            cx, cy, cz = eef_pos
            r = 0.25
            ax.set_xlim(cx - r, cx + r)
            ax.set_ylim(cy - r, cy + r)
            ax.set_zlim(max(0.0, cz - r), cz + r)

            ax.legend(fontsize=8, loc='upper left',
                      facecolor='#222', edgecolor='#555', labelcolor='white')
            ax.set_title(f'step {step}', color='#aaa', fontsize=9, pad=2)

            fig.canvas.draw_idle()
            plt.pause(REFRESH_S)

        except KeyboardInterrupt:
            print("\nDone.")
            break
        except Exception as e:
            print(f"[plotter] {e}")
            plt.pause(REFRESH_S)


if __name__ == '__main__':
    main()
