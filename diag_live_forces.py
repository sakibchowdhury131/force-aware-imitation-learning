#!/usr/bin/env python3
"""
Live plot of END-EFFECTOR FORCE while you hand-guide the arm with the
JOYSTICK -- no replay_episode.py needed, since joystick motion doesn't go
through this API (this script just READS live position/torque, same as
test_residual_repeatability.py's interactive mode; it never sends a motion
command). Connects directly to the robot itself (unlike diag_replay_forces.py,
which reads a state file written by a separate replay process).

Uses the FINAL recommended calibration pipeline from FORCE_SENSING_FINDINGS.md:
  1. firmware gravity matrix   (apply_saved_gravity_params, applied at startup)
  2. software gravity regressor (contact_detector.gravity_regressor @ phi_task_only)
  3. gravity-residual NN        (fit_gravity_residual_nn -- static-data-only correction)
  4. mass/Coriolis REGRESSOR    (full_dynamics_regressor @ dynamics_residual_pi)

Same threaded architecture as diag_replay_forces.py, adapted after two bugs
found live-testing that script: (1) heavy per-step computation (~25ms,
mostly the 72-parameter dynamics regressor) must run in a background thread,
never inside the matplotlib animation callback, or the GUI falls further and
further behind; (2) that thread must append ONE atomic record per step to a
SINGLE shared deque, not several deques one-at-a-time, or the GUI can catch
them mid-update with mismatched lengths and crash/freeze.

Usage:
    python diag_live_forces.py
    python diag_live_forces.py --hz 30      # polling rate (default 25)

Press Q or close the window to quit.
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
from calibrate_firmware_gravity import load_api, connect, get_q, get_qdot, get_tau_gf, apply_saved_gravity_params

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
    p.add_argument('--hz', type=float, default=25.0, help='Hardware poll rate (default 25)')
    p.add_argument('--qddot_smoothing', type=float, default=0.3)
    p.add_argument('--window', type=float, default=10.0, help='Seconds of history to display')
    p.add_argument('--damping', type=float, default=0.05)
    p.add_argument('--redraw_hz', type=float, default=15.0)
    return p.parse_args()


def main():
    args = parse_args()

    phi = np.load(os.path.join(PIPELINE_DIR, args.phi))
    ckpt = torch.load(os.path.join(PIPELINE_DIR, args.gravity_nn), weights_only=False)
    g_nn = GravityResidualNet(); g_nn.load_state_dict(ckpt['state_dict']); g_nn.eval()
    g_x_mean, g_x_std = ckpt['x_mean'], ckpt['x_std']
    pi = np.load(os.path.join(PIPELINE_DIR, args.dynamics_pi))
    print('Loaded gravity regressor, gravity-residual NN, and mass/Coriolis regressor.')

    api = load_api()
    connect(api, control=False)   # READ-ONLY: never sends a motion command
    grav_ok = apply_saved_gravity_params(api)
    if not grav_ok:
        print('ERROR: firmware gravity params failed to reapply. The Kinova does NOT retain '
              'the calibrated gravity matrix across a power cycle -- continuing would silently '
              'corrupt every force estimate in this session. Aborting rather than showing bad '
              'data. See the message above for why apply_saved_gravity_params failed.')
        api.CloseAPI()
        sys.exit(1)
    print('Firmware gravity params reapplied: OK')
    print('Move the arm with the joystick now. Press Q or close the window to quit.\n')

    maxlen = 4000
    records = collections.deque(maxlen=maxlen)   # single shared buffer, see module docstring
    stop_flag = threading.Event()

    def worker():
        qdiff = VelocityDifferentiator(smoothing=args.qddot_smoothing)
        dt = 1.0 / args.hz
        t0 = time.time()
        while not stop_flag.is_set():
            loop_start = time.time()
            q_deg = get_q(api)
            qdot_deg = get_qdot(api)
            gf = get_tau_gf(api)
            t_now = time.time() - t0

            F_fw = torque_to_wrench(q_deg, gf, damping=args.damping)

            g_res_lin = gravity_regressor(q_deg) @ phi
            g_res_nn = predict_gravity_residual(g_nn, g_x_mean, g_x_std, q_deg)
            tau_after_gravity = gf - g_res_lin - g_res_nn
            F_grav = torque_to_wrench(q_deg, tau_after_gravity, damping=args.damping)

            qddot_deg = qdiff.update(qdot_deg, t_now)
            dyn_pred = full_dynamics_regressor(q_deg, qdot_deg, qddot_deg) @ pi
            tau_final = tau_after_gravity - dyn_pred
            F_final = torque_to_wrench(q_deg, tau_final, damping=args.damping)

            records.append((
                t_now,
                float(np.linalg.norm(F_fw[:3])),
                float(np.linalg.norm(F_grav[:3])),
                float(np.linalg.norm(F_final[:3])),
                F_final[:3].copy(),
            ))

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, dt - elapsed))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    fig, (ax_mag, ax_axes) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('Live joystick demo — End-Effector Force (firmware + gravity regressor '
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

    ax_mag.set_title('||F|| (linear part) — push on the arm and watch "final estimate"', fontsize=10)
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
    ax_axes.set_xlabel('Time (s)')
    axis_lines, axis_ro = [], []
    for j, (col, lbl) in enumerate(zip(AXIS_COLORS, AXIS_LABELS)):
        ln, = ax_axes.plot([], [], color=col, linewidth=1.4, label=lbl)
        axis_lines.append(ln)
        txt = ax_axes.text(1.001, 0.85 - j * 0.14, '', transform=ax_axes.transAxes,
                           color=col, fontsize=9, va='center', ha='left', fontfamily='monospace')
        axis_ro.append(txt)
    ax_axes.legend(loc='upper left', fontsize=9, framealpha=0.4,
                  labelcolor='#eeeeee', facecolor='#222222')

    last_drawn_len = [0]

    def update(_frame):
        snapshot = list(records)
        n = len(snapshot)
        if n < 2 or n == last_drawn_len[0]:
            return []
        last_drawn_len[0] = n

        t_arr     = np.array([r[0] for r in snapshot])
        fw_arr    = np.array([r[1] for r in snapshot])
        grav_arr  = np.array([r[2] for r in snapshot])
        final_arr = np.array([r[3] for r in snapshot])
        axes_arr  = np.array([r[4] for r in snapshot])

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

        return [ln_fw, ln_grav, ln_final, ro_fw, ro_grav, ro_final] + axis_lines + axis_ro

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
        api.CloseAPI()
        print('API closed.')


if __name__ == '__main__':
    main()
