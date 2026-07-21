#!/usr/bin/env python3
"""
Live plot of END-EFFECTOR FORCE while replay_episode.py is running, using the
FINAL recommended calibration pipeline from FORCE_SENSING_FINDINGS.md:
  1. firmware gravity matrix   (baked into gravity_free_torque via apply_saved_gravity_params)
  2. software gravity regressor (contact_detector.gravity_regressor @ phi_task_only)
  3. gravity-residual NN        (fit_gravity_residual_nn -- static-data-only correction)
  4. mass/Coriolis REGRESSOR    (full_dynamics_regressor @ dynamics_residual_pi -- NOT the
                                 NN here: the regressor and NN tied on held-out accuracy,
                                 and the regressor is the one to trust, see FORCE_SENSING_FINDINGS.md §4)

Like diag_replay_torques.py, this does NOT talk to the arm at all -- only one
process can hold the USB connection, and replay_episode.py already has it.
Instead this reads /tmp/replay_state.npz (q_deg, qdot_deg, gravity_free_torque),
written every executed step by replay_episode.py.

PERFORMANCE NOTE: full_dynamics_regressor costs ~25ms/call (72 unit-perturbation
RNEA evaluations) -- right at the edge of a 30Hz redraw interval. Earlier version
called this directly inside the matplotlib animation callback, so on any tick
slower than ~33ms the Tk event loop fell behind and never caught up (observed:
severe, growing display lag over a ~100-step replay). Fixed by running the
state-file-watching + heavy computation in a background thread; the animation
callback only ever reads already-computed values from shared buffers and
redraws -- rendering can never be blocked by the model, mirroring
diag_joint_torques.py's poll/redraw split (there the poll itself is cheap
enough to not need a separate thread; here it isn't, so the split is explicit).

Usage — run in a SEPARATE terminal, alongside replay_episode.py --execute:
    python diag_replay_forces.py
    python diag_replay_forces.py --window 15

Press Q or close the window to quit (does not stop the replay).
"""
import os, sys, time, threading, argparse, collections
import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)
from contact_detector import torque_to_wrench, gravity_regressor, full_dynamics_regressor, VelocityDifferentiator
from fit_gravity_residual_nn import GravityResidualNet, predict as predict_gravity_residual

STATE_FILE = '/tmp/replay_state.npz'

FIRMWARE_COLOR = '#f58231'
GRAV_COLOR     = '#3cb44b'
FINAL_COLOR    = '#4363d8'
AXIS_COLORS    = ['#e6194b', '#3cb44b', '#4363d8']
AXIS_LABELS    = ['Fx', 'Fy', 'Fz']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--phi', default='data/gravity_phi_task_only.npy')
    p.add_argument('--gravity_nn', default='data/gravity_residual_nn.pt')
    p.add_argument('--dynamics_pi', default='data/dynamics_residual_pi.npy')
    p.add_argument('--qddot_smoothing', type=float, default=0.3,
                   help='EMA smoothing for the live finite-differenced qddot (default 0.3)')
    p.add_argument('--window', type=float, default=8.0,
                   help='Seconds of history to display (default 8)')
    p.add_argument('--damping', type=float, default=0.05,
                   help='Tikhonov damping for the wrench pinv (default 0.05)')
    p.add_argument('--redraw_hz', type=float, default=15.0,
                   help='GUI redraw rate -- decoupled from computation, always cheap (default 15)')
    p.add_argument('--watch_hz', type=float, default=20.0,
                   help='Background-thread state-file poll rate (default 20)')
    return p.parse_args()


def main():
    args = parse_args()

    phi = np.load(os.path.join(PIPELINE_DIR, args.phi))
    print(f'Loaded gravity regressor phi from {args.phi}')

    ckpt = torch.load(os.path.join(PIPELINE_DIR, args.gravity_nn), weights_only=False)
    g_nn = GravityResidualNet(); g_nn.load_state_dict(ckpt['state_dict']); g_nn.eval()
    g_x_mean, g_x_std = ckpt['x_mean'], ckpt['x_std']
    print(f'Loaded gravity-residual NN from {args.gravity_nn}')

    pi = np.load(os.path.join(PIPELINE_DIR, args.dynamics_pi))
    print(f'Loaded mass/Coriolis regressor pi from {args.dynamics_pi}')

    print(f"Waiting for {STATE_FILE} (start replay_episode.py --execute in another terminal)...")
    while not os.path.exists(STATE_FILE):
        time.sleep(0.2)
    print("Found it — plotting.\n")

    maxlen = 2000
    buf_fw, buf_grav, buf_final = (collections.deque(maxlen=maxlen) for _ in range(3))
    # Single shared buffer (one deque, one atomic append per new step) --
    # previously used 5 separate deques appended one-at-a-time from the
    # worker thread, which let the GUI thread catch them mid-update with
    # mismatched lengths (a real race: "shape mismatch (11,) vs (12,)"),
    # crashing the Tkinter callback and silently freezing the plot at
    # whatever frame last rendered successfully. A single deque.append() is
    # atomic under the GIL, so there's no length-mismatch window anymore.
    records = collections.deque(maxlen=maxlen)
    frame_info = ['']

    stop_flag = threading.Event()

    def worker():
        """Background thread: watches the state file, does ALL the heavy
        computation (gravity regressor + NN + 72-param dynamics regressor,
        ~25ms/step), and appends to the shared deques. Never touches
        matplotlib -- the GUI thread only ever reads these buffers."""
        qdiff = VelocityDifferentiator(smoothing=args.qddot_smoothing)
        last_step = None
        t0 = None
        watch_dt = 1.0 / args.watch_hz
        while not stop_flag.is_set():
            time.sleep(watch_dt)
            if not os.path.exists(STATE_FILE):
                continue
            try:
                state = np.load(STATE_FILE)
                step = int(state['step'][0])
                if step == last_step:
                    continue
                last_step = step
                t_now = float(state['t'][0])
                q_deg = state['q_deg']
                qdot_deg = state['qdot_deg'] if 'qdot_deg' in state.files else np.zeros(6)
                gf = state['gravity_free_torque']
                fid = int(state['frame_id'][0])
            except Exception:
                continue  # file mid-write — skip this tick

            F_fw = torque_to_wrench(q_deg, gf, damping=args.damping)

            g_res_lin = gravity_regressor(q_deg) @ phi
            g_res_nn = predict_gravity_residual(g_nn, g_x_mean, g_x_std, q_deg)
            tau_after_gravity = gf - g_res_lin - g_res_nn
            F_grav = torque_to_wrench(q_deg, tau_after_gravity, damping=args.damping)

            qddot_deg = qdiff.update(qdot_deg, t_now)
            dyn_pred = full_dynamics_regressor(q_deg, qdot_deg, qddot_deg) @ pi
            tau_final = tau_after_gravity - dyn_pred
            F_final = torque_to_wrench(q_deg, tau_final, damping=args.damping)

            if t0 is None:
                t0 = t_now
            records.append((
                t_now - t0,
                float(np.linalg.norm(F_fw[:3])),
                float(np.linalg.norm(F_grav[:3])),
                float(np.linalg.norm(F_final[:3])),
                F_final[:3].copy(),
            ))
            frame_info[0] = f'step {step}  frame {fid}  t={t_now - t0:.1f}s'

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    fig, (ax_mag, ax_axes) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('Replay — End-Effector Force (full pipeline: firmware + gravity regressor '
                '+ gravity-residual NN + mass/Coriolis regressor)', fontsize=12, fontweight='bold')

    for ax in (ax_mag, ax_axes):
        ax.set_facecolor('#111111')
        ax.tick_params(colors='#cccccc')
        ax.yaxis.label.set_color('#cccccc')
        ax.xaxis.label.set_color('#cccccc')
        ax.title.set_color('#eeeeee')
        ax.axhline(0, color='white', linewidth=0.5, alpha=0.4)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')
    fig.patch.set_facecolor('#1a1a1a')

    ax_mag.set_title('||F|| (linear part) — try pushing on the arm and watch "final estimate"', fontsize=10)
    ax_mag.set_ylabel('Force (N)')
    ln_fw,    = ax_mag.plot([], [], color=FIRMWARE_COLOR, linewidth=1.1, label='firmware-only')
    ln_grav,  = ax_mag.plot([], [], color=GRAV_COLOR, linewidth=1.3, label='+ gravity (regressor + NN)')
    ln_final, = ax_mag.plot([], [], color=FINAL_COLOR, linewidth=2.0, label='final estimate (+ mass/Coriolis regressor)')
    ro_fw    = ax_mag.text(1.001, 0.85, '', transform=ax_mag.transAxes, color=FIRMWARE_COLOR,
                           fontsize=9, va='center', ha='left', fontfamily='monospace')
    ro_grav  = ax_mag.text(1.001, 0.70, '', transform=ax_mag.transAxes, color=GRAV_COLOR,
                           fontsize=9, va='center', ha='left', fontfamily='monospace')
    ro_final = ax_mag.text(1.001, 0.55, '', transform=ax_mag.transAxes, color=FINAL_COLOR,
                           fontsize=9, va='center', ha='left', fontfamily='monospace')
    ax_mag.legend(loc='upper left', fontsize=9, framealpha=0.4,
                  labelcolor='#eeeeee', facecolor='#222222')

    ax_axes.set_title('Final-estimate wrench components (base frame)', fontsize=10)
    ax_axes.set_ylabel('Force (N)')
    ax_axes.set_xlabel('Replay time (s)')
    axis_lines, axis_ro = [], []
    for j, (col, lbl) in enumerate(zip(AXIS_COLORS, AXIS_LABELS)):
        ln, = ax_axes.plot([], [], color=col, linewidth=1.4, label=lbl)
        axis_lines.append(ln)
        txt = ax_axes.text(1.001, 0.85 - j * 0.14, '', transform=ax_axes.transAxes,
                           color=col, fontsize=9, va='center', ha='left', fontfamily='monospace')
        axis_ro.append(txt)
    ax_axes.legend(loc='upper left', fontsize=9, framealpha=0.4,
                  labelcolor='#eeeeee', facecolor='#222222')

    frame_txt = fig.text(0.5, 0.965, '', ha='center', fontsize=9, color='#00c8ff')
    last_drawn_len = [0]

    def update(_frame):
        snapshot = list(records)   # one atomic-ish copy of the deque; every
                                   # tuple in it was appended as a whole, so
                                   # no per-field length mismatch is possible
        n = len(snapshot)
        if n < 2 or n == last_drawn_len[0]:
            return []
        last_drawn_len[0] = n

        t_arr      = np.array([r[0] for r in snapshot])
        fw_arr     = np.array([r[1] for r in snapshot])
        grav_arr   = np.array([r[2] for r in snapshot])
        final_arr  = np.array([r[3] for r in snapshot])
        axes_arr   = np.array([r[4] for r in snapshot])

        frame_txt.set_text(frame_info[0])

        ln_fw.set_data(t_arr, fw_arr)
        ln_grav.set_data(t_arr, grav_arr)
        ln_final.set_data(t_arr, final_arr)
        ro_fw.set_text(f'firmware-only: {fw_arr[-1]:+6.2f} N')
        ro_grav.set_text(f'+gravity:     {grav_arr[-1]:+6.2f} N')
        ro_final.set_text(f'final:        {final_arr[-1]:+6.2f} N')
        ax_mag.relim(); ax_mag.autoscale_view(scalex=True, scaley=True)

        for j, ln in enumerate(axis_lines):
            ln.set_data(t_arr, axes_arr[:, j])
            axis_ro[j].set_text(f'{AXIS_LABELS[j]}: {axes_arr[-1, j]:+6.2f} N')
        ax_axes.relim(); ax_axes.autoscale_view(scalex=True, scaley=True)

        x_max = t_arr[-1]
        x_min = max(0.0, x_max - args.window)
        for ax in (ax_mag, ax_axes):
            ax.set_xlim(x_min, x_max + 0.1)

        return [ln_fw, ln_grav, ln_final, ro_fw, ro_grav, ro_final, frame_txt] + axis_lines + axis_ro

    redraw_interval_ms = max(20, int(1000.0 / args.redraw_hz))
    ani = animation.FuncAnimation(fig, update, interval=redraw_interval_ms,
                                  blit=False, cache_frame_data=False)

    def on_key(event):
        if event.key in ('q', 'Q'):
            plt.close('all')
    fig.canvas.mpl_connect('key_press_event', on_key)

    try:
        plt.tight_layout(rect=[0, 0, 0.88, 0.94])
        plt.show()
    finally:
        stop_flag.set()
        thread.join(timeout=1.0)


if __name__ == '__main__':
    main()
