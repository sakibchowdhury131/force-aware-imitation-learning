#!/usr/bin/env python3
"""
Live plot of joint torques while replay_episode.py is running.

Unlike diag_joint_torques.py (which opens its own Kinova USB connection),
this script does NOT talk to the arm at all — only one process can hold the
USB connection at a time, and replay_episode.py already has it. Instead this
reads /tmp/replay_state.npz, written every executed step by replay_episode.py
(same pattern as 07_deploy.py + deploy_viz.py).

Usage — run in a SEPARATE terminal, alongside replay_episode.py --execute:
    python diag_replay_torques.py
    python diag_replay_torques.py --gravity_free
    python diag_replay_torques.py --window 15

Press Q or close the window to quit (does not stop the replay).
"""
import os, time, argparse, collections
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

STATE_FILE = '/tmp/replay_state.npz'

JOINT_COLORS = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4']
JOINT_LABELS = [f'J{i}' for i in range(1, 7)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--window', type=float, default=8.0,
                   help='Seconds of history to display (default 8)')
    p.add_argument('--gravity_free', action='store_true',
                   help='Show only gravity-compensated channel (cleaner for external loads)')
    p.add_argument('--poll_hz', type=float, default=30.0,
                   help='How often to check the state file for a new step (default 30 Hz)')
    return p.parse_args()


def main():
    args = parse_args()
    poll_dt = 1.0 / args.poll_hz

    print(f"Waiting for {STATE_FILE} (start replay_episode.py --execute in another terminal)...")
    while not os.path.exists(STATE_FILE):
        time.sleep(0.2)
    print("Found it — plotting.\n")

    maxlen  = 2000  # generous — actual window is time-based via t_buf
    buf_raw = collections.deque(maxlen=maxlen)
    buf_gf  = collections.deque(maxlen=maxlen)
    t_buf   = collections.deque(maxlen=maxlen)
    step_buf = collections.deque(maxlen=maxlen)

    last_step = [None]
    t0 = [None]

    n_rows = 1 if args.gravity_free else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 4 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]
    fig.suptitle('Replay — Joint Torques (N·m)', fontsize=13, fontweight='bold')

    titles = (['Gravity-free torque (external load)'] if args.gravity_free
              else ['Raw torque (GetAngularForce)',
                    'Gravity-free torque (GetAngularForceGravityFree)'])

    lines = []
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=10)
        ax.set_ylabel('Torque (N·m)')
        ax.set_xlabel('Replay time (s)')
        ax.axhline(0, color='white', linewidth=0.5, alpha=0.4)
        ax.set_facecolor('#111111')
        fig.patch.set_facecolor('#1a1a1a')
        ax.tick_params(colors='#cccccc')
        ax.yaxis.label.set_color('#cccccc')
        ax.xaxis.label.set_color('#cccccc')
        ax.title.set_color('#eeeeee')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')
        row_lines = []
        for col, lbl in zip(JOINT_COLORS, JOINT_LABELS):
            ln, = ax.plot([], [], color=col, linewidth=1.4, label=lbl)
            row_lines.append(ln)
        ax.legend(loc='upper left', fontsize=8, framealpha=0.4,
                  labelcolor='#eeeeee', facecolor='#222222')
        lines.append(row_lines)

    readouts = []
    for ax, row_lines in zip(axes, lines):
        row_ro = []
        for j, (col, ln) in enumerate(zip(JOINT_COLORS, row_lines)):
            txt = ax.text(1.001, 0.85 - j * 0.14, '', transform=ax.transAxes,
                          color=col, fontsize=7.5, va='center', ha='left',
                          fontfamily='monospace')
            row_ro.append(txt)
        readouts.append(row_ro)

    frame_txt = fig.text(0.5, 0.965, '', ha='center', fontsize=9, color='#00c8ff')

    def update(_frame):
        if not os.path.exists(STATE_FILE):
            return []
        try:
            state = np.load(STATE_FILE)
            step  = int(state['step'][0])
            if step == last_step[0]:
                return []
            last_step[0] = step
            t_now  = float(state['t'][0])
            raw    = state['raw_torque']
            gf     = state['gravity_free_torque']
            fid    = int(state['frame_id'][0])
        except Exception:
            return []  # file mid-write — skip this tick

        if t0[0] is None:
            t0[0] = t_now
        buf_raw.append(raw)
        buf_gf.append(gf)
        t_buf.append(t_now - t0[0])
        step_buf.append(step)

        if len(t_buf) < 2:
            return []

        t_arr   = np.array(t_buf)
        raw_arr = np.array(buf_raw)
        gf_arr  = np.array(buf_gf)
        bufs = [gf_arr] if args.gravity_free else [raw_arr, gf_arr]

        frame_txt.set_text(f'step {step}  frame {fid}  t={t_arr[-1]:.1f}s')

        updated = []
        for ax, row_lines, row_bufs, row_ro in zip(axes, lines, bufs, readouts):
            for j, (ln, txt) in enumerate(zip(row_lines, row_ro)):
                ln.set_data(t_arr, row_bufs[:, j])
                txt.set_text(f'J{j+1}: {row_bufs[-1, j]:+7.2f} N·m')
                updated.append(ln)
                updated.append(txt)
            ax.relim()
            ax.autoscale_view(scalex=True, scaley=True)
            x_max = t_arr[-1]
            x_min = max(0.0, x_max - args.window)
            ax.set_xlim(x_min, x_max + 0.1)

        return updated

    ani = animation.FuncAnimation(fig, update, interval=max(20, int(poll_dt * 1000)),
                                  blit=False, cache_frame_data=False)

    def on_key(event):
        if event.key in ('q', 'Q'):
            plt.close('all')
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.tight_layout(rect=[0, 0, 0.97, 0.94])
    plt.show()


if __name__ == '__main__':
    main()
