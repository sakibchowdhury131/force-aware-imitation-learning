#!/usr/bin/env python3
"""
Quick sanity check: read torques at the current pose, command a small
(default 10 deg) commanded move on a few joints, read torques there, then
return to the starting pose and read again. Reports raw and gravity-free
joint torques plus the Jacobian-transpose EEF wrench (raw and gravity-free)
at all three stops, and flags anything that looks wrong (NaN/inf, huge
jumps, high cond(J), poor return-to-start repeatability, arrival errors).

Read-only except for the commanded motion itself — no firmware writes.
Reuses load_api/connect/get_q/get_qdot/clamp_pose/move_and_settle from
calibrate_firmware_gravity.py and torque_to_wrench/compute_jacobian from
contact_detector.py, same pattern as the rest of this pipeline.

Usage:
    python quick_pose_check.py                # dry run: just prints the plan
    python quick_pose_check.py --execute       # actually moves the arm
    python quick_pose_check.py --execute --delta 15   # bigger nudge
"""
import os, sys, ctypes, time, argparse
import numpy as np

_LIB_DIR = os.path.join(os.path.expanduser('~/working_dir/kinovaDrivers'), 'sdk', 'lib')
if _LIB_DIR not in os.environ.get('LD_LIBRARY_PATH', '').split(':'):
    os.environ['LD_LIBRARY_PATH'] = _LIB_DIR + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    os.execv(sys.executable, [sys.executable] + sys.argv)

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)
from contact_detector import compute_jacobian, torque_to_wrench
from calibrate_firmware_gravity import (
    load_api as _base_load_api, connect, get_q, get_qdot, clamp_pose,
    move_and_settle, apply_saved_gravity_params, JOINT_LIMITS_DEG, SAFE_SPEED_DPS,
)

NO_ERROR_KINOVA = 1


def load_api():
    """Extends calibrate_firmware_gravity's load_api with GetAngularForce
    (raw torque) — that script only reads gravity-free, this one needs both."""
    api = _base_load_api()
    api.GetAngularForce.restype = ctypes.c_int
    return api


def get_tau_raw(api):
    from calibrate_firmware_gravity import AngularPosition
    s = AngularPosition()
    api.GetAngularForce(ctypes.byref(s))
    a = s.Actuators
    return np.array([a.Actuator1, a.Actuator2, a.Actuator3,
                     a.Actuator4, a.Actuator5, a.Actuator6], np.float64)


def get_tau_gf(api):
    from calibrate_firmware_gravity import AngularPosition
    s = AngularPosition()
    api.GetAngularForceGravityFree(ctypes.byref(s))
    a = s.Actuators
    return np.array([a.Actuator1, a.Actuator2, a.Actuator3,
                     a.Actuator4, a.Actuator5, a.Actuator6], np.float64)


def capture(api, n_samples=100, hz=100.0):
    """Returns (mean_raw, mean_gf, std_raw, std_gf) over n_samples."""
    raws, gfs = [], []
    dt = 1.0 / hz
    for _ in range(n_samples):
        raws.append(get_tau_raw(api))
        gfs.append(get_tau_gf(api))
        time.sleep(dt)
    raws = np.stack(raws); gfs = np.stack(gfs)
    return raws.mean(0), gfs.mean(0), raws.std(0), gfs.std(0)


def report_stop(label, q, tau_raw, tau_gf, damping=0.0):
    J = compute_jacobian(q)
    cond = np.linalg.cond(J)
    F_raw = torque_to_wrench(q, tau_raw, damping=damping)
    F_gf  = torque_to_wrench(q, tau_gf,  damping=damping)
    print(f'\n--- {label} ---')
    print(f'  q (deg): {np.round(q, 1)}')
    print(f'  cond(J) = {cond:.2e}' + ('  <-- near-singular, wrench unreliable' if cond > 1e4 else ''))
    print(f'  {"joint":>6} {"raw (N*m)":>12} {"grav-free (N*m)":>17}')
    for i in range(6):
        print(f'  J{i+1:<5} {tau_raw[i]:>12.3f} {tau_gf[i]:>17.3f}')
    print(f'  ||F|| raw  (linear part) = {np.linalg.norm(F_raw[:3]):.3f} N   '
          f'wrench = {np.round(F_raw, 2)}')
    print(f'  ||F|| grav-free          = {np.linalg.norm(F_gf[:3]):.3f} N   '
          f'wrench = {np.round(F_gf, 2)}')
    return dict(q=q, tau_raw=tau_raw, tau_gf=tau_gf, F_raw=F_raw, F_gf=F_gf, cond=cond)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--execute', action='store_true',
                   help='Actually send motion. Without this flag, only prints the plan.')
    p.add_argument('--delta', type=float, default=10.0,
                   help='Degrees to nudge each of J2,J3,J5 (default 10)')
    p.add_argument('--speed', type=float, default=SAFE_SPEED_DPS,
                   help=f'Commanded speed, deg/s (default {SAFE_SPEED_DPS})')
    p.add_argument('--damping', type=float, default=0.05,
                   help='Tikhonov damping for the wrench pinv (default 0.05, guards against '
                        'singularity blow-up during the small move)')
    return p.parse_args()


def main():
    args = parse_args()
    api = load_api()
    try:
        model, serial = connect(api, control=True)
        grav_ok = apply_saved_gravity_params(api)
        print(f'Firmware gravity params (data/gravity_params.npy) reapplied: '
              f'{"OK" if grav_ok else "not applied — see message above"}')

        q0 = get_q(api)
        print(f'Current pose (deg): {np.round(q0, 1)}')

        # Small nudge on J2, J3, J5 only (the joints with real hard limits —
        # J1/J4/J6 are continuous so a delta there is directionally meaningless
        # for "a little bit around the current pose"). Alternate sign so the
        # move is a genuine perturbation, not a drift in one direction.
        q1 = q0.copy()
        q1[1] += args.delta
        q1[2] -= args.delta
        q1[4] += args.delta
        q1 = clamp_pose(q1)

        print(f'Planned nudge target (deg): {np.round(q1, 1)}  '
              f'(delta = {np.round(q1 - q0, 1)})')

        if not args.execute:
            print('\nDry run only — pass --execute to actually move the arm.')
            return

        results = []

        print('\nCapturing baseline at current pose (stationary, ~1s)...')
        tau_raw0, tau_gf0, std_raw0, std_gf0 = capture(api)
        results.append(('start', report_stop('START (original pose)', q0, tau_raw0, tau_gf0, args.damping)))

        print(f'\nMoving to nudged pose (speed={args.speed} deg/s)...')
        err, settled = move_and_settle(api, q1, speed_dps=args.speed)
        print(f'  arrived, max error {err:.2f} deg, settled={settled}')
        q1_actual = get_q(api)
        tau_raw1, tau_gf1, std_raw1, std_gf1 = capture(api)
        results.append(('nudged', report_stop('NUDGED pose', q1_actual, tau_raw1, tau_gf1, args.damping)))

        print(f'\nReturning to start pose (speed={args.speed} deg/s)...')
        err2, settled2 = move_and_settle(api, q0, speed_dps=args.speed)
        print(f'  arrived, max error {err2:.2f} deg, settled={settled2}')
        q2_actual = get_q(api)
        tau_raw2, tau_gf2, std_raw2, std_gf2 = capture(api)
        results.append(('return', report_stop('RETURN (back to start pose)', q2_actual, tau_raw2, tau_gf2, args.damping)))

        # ── verdict ──────────────────────────────────────────────────────────
        print('\n' + '=' * 70)
        print('VERDICT')
        print('=' * 70)
        problems = []

        all_tau = np.concatenate([tau_raw0, tau_gf0, tau_raw1, tau_gf1, tau_raw2, tau_gf2])
        if not np.all(np.isfinite(all_tau)):
            problems.append('NaN/inf in torque readings — sensor read failure.')

        if np.max(np.abs(tau_raw0)) > 40 or np.max(np.abs(tau_raw2)) > 40:
            problems.append(f'Very large raw torque at rest (>40 N*m on some joint) — '
                            f'check for a stuck/faulted actuator.')

        if not settled or not settled2:
            problems.append('Velocity never settled below threshold after a move — '
                            'possible stiction, fault, or brake issue.')

        if err > 2.0 or err2 > 2.0:
            problems.append(f'Large arrival error (start->nudge {err:.2f} deg, '
                            f'nudge->return {err2:.2f} deg) — position control may not be tracking well.')

        # Return-to-start repeatability: gravity-free torque at start vs return
        # should match closely (same pose, same load) if sensors/dynamics are healthy.
        gf_diff = np.abs(tau_gf2 - tau_gf0)
        if np.max(gf_diff) > 1.5:
            problems.append(f'Gravity-free torque at RETURN differs from START by up to '
                            f'{np.max(gf_diff):.2f} N*m at the same pose — possible drift, '
                            f'thermal effect, or hysteresis (see project_force_sensing memory).')

        for label, r in results:
            if r['cond'] > 1e4:
                problems.append(f'{label}: cond(J)={r["cond"]:.2e} — near a kinematic '
                                f'singularity, wrench numbers there are not trustworthy.')

        if problems:
            print('Issues found:')
            for p_ in problems:
                print(f'  - {p_}')
        else:
            print('No issues detected: torques finite and within normal range, both moves '
                  'converged and settled cleanly, gravity-free torque repeatable at start vs '
                  'return, no singular poses. Looks fine.')

    finally:
        api.CloseAPI()
        print('\nAPI closed.')


if __name__ == '__main__':
    main()
