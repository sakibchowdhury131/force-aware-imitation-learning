#!/usr/bin/env python3
"""
Verify the currently-saved firmware gravity params (data/gravity_params.npy,
fitted by calibrate_firmware_gravity.py) actually generalize, by visiting a
handful of poses that were NOT part of the 8-pose calibration set and
reporting ||F|| (raw and gravity-free) at each.

Re-applies apply_saved_gravity_params() at the start (harmless whether or not
firmware state persisted since the calibration run) so this is a fair test of
"do the saved params work", independent of whether this process is a fresh
API session.

Usage:
    python verify_gravity_params.py                 # dry run, prints planned poses
    python verify_gravity_params.py --execute        # actually moves the arm
    python verify_gravity_params.py --execute --n_poses 8 --seed 4242
"""
import os, sys, ctypes, time, argparse
import numpy as np

_LIB_DIR = os.path.join(os.path.expanduser('~/working_dir/kinovaDrivers'), 'sdk', 'lib')
if _LIB_DIR not in os.environ.get('LD_LIBRARY_PATH', '').split(':'):
    os.environ['LD_LIBRARY_PATH'] = _LIB_DIR + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    os.execv(sys.executable, [sys.executable] + sys.argv)

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)

from calibrate_firmware_gravity import (
    load_api as _base_load_api, connect, get_q, get_qdot, clamp_pose,
    move_and_settle, apply_saved_gravity_params, JOINT_LIMITS_DEG, SAFE_SPEED_DPS,
    AngularPosition, TEST_POSES_DEG,
)
from calibrate_gravity_residual import ANCHOR_POSES_DEG, generate_poses
from contact_detector import compute_jacobian, torque_to_wrench


def load_api():
    api = _base_load_api()
    api.GetAngularForce.restype = ctypes.c_int
    return api


def get_tau_raw(api):
    s = AngularPosition()
    api.GetAngularForce(ctypes.byref(s))
    a = s.Actuators
    return np.array([a.Actuator1, a.Actuator2, a.Actuator3,
                     a.Actuator4, a.Actuator5, a.Actuator6], np.float64)


def get_tau_gf(api):
    s = AngularPosition()
    api.GetAngularForceGravityFree(ctypes.byref(s))
    a = s.Actuators
    return np.array([a.Actuator1, a.Actuator2, a.Actuator3,
                     a.Actuator4, a.Actuator5, a.Actuator6], np.float64)


def capture(api, n_samples=200, hz=100.0):
    raws, gfs = [], []
    dt = 1.0 / hz
    for _ in range(n_samples):
        raws.append(get_tau_raw(api))
        gfs.append(get_tau_gf(api))
        time.sleep(dt)
    return np.stack(raws).mean(0), np.stack(gfs).mean(0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--execute', action='store_true',
                   help='Actually move the arm. Without this flag, only prints the planned poses.')
    p.add_argument('--n_poses', type=int, default=6,
                   help='Number of held-out verification poses (default 6)')
    p.add_argument('--seed', type=int, default=4242,
                   help='RNG seed for generate_poses — distinct from any seed used '
                        'while collecting the software-regressor training data, so these '
                        'poses were never used to fit anything (default 4242)')
    p.add_argument('--speed', type=float, default=SAFE_SPEED_DPS)
    p.add_argument('--skip_reapply', action='store_true',
                   help='Do NOT call apply_saved_gravity_params() first — use this to test '
                        'whether firmware gravity params survived a power cycle on their own. '
                        'Without this flag the test is meaningless for that question, since '
                        'reapplying would mask whatever the firmware actually retained.')
    return p.parse_args()


def main():
    args = parse_args()

    poses = generate_poses(args.n_poses, seed=args.seed, max_cond=50.0, min_spread_deg=10.0)
    print(f'Generated {len(poses)} held-out verification poses (seed={args.seed}, '
          f'not part of TEST_POSES_DEG\'s calibration set):')
    for i, q in enumerate(poses):
        print(f'  [{i}] {np.round(q, 1)}')

    if not args.execute:
        print('\nDry run only — pass --execute to actually move the arm.')
        return

    api = load_api()
    try:
        model, serial = connect(api, control=True)

        if args.skip_reapply:
            print('\n--skip_reapply set — NOT calling apply_saved_gravity_params(). '
                  'Whatever ||F|| we see now reflects whatever the firmware actually retained.')
        else:
            print('\nRe-applying saved firmware gravity params (data/gravity_params.npy) — '
                  'harmless if they already persisted...')
            ok = apply_saved_gravity_params(api)
            print(f'  apply_saved_gravity_params -> {"OK" if ok else "FAILED (see printed codes above)"}')

        records = []
        for i, raw_pose in enumerate(poses):
            pose = clamp_pose(raw_pose)
            print(f'\n[pose {i}] target: {np.round(pose, 1)}')
            err, settled = move_and_settle(api, pose, speed_dps=args.speed)
            if not settled:
                print('  WARNING: velocity never settled — sampling anyway.')
            tau_raw, tau_gf = capture(api)
            q_actual = get_q(api)
            cond = np.linalg.cond(compute_jacobian(q_actual))
            F_raw = torque_to_wrench(q_actual, tau_raw, damping=0.05)
            F_gf  = torque_to_wrench(q_actual, tau_gf,  damping=0.05)
            Fn_raw = float(np.linalg.norm(F_raw[:3]))
            Fn_gf  = float(np.linalg.norm(F_gf[:3]))
            print(f'  arrived (max err {err:.2f} deg, settled={settled})  cond(J)={cond:.1f}')
            print(f'  ||F|| raw = {Fn_raw:.3f} N   ||F|| gravity-free = {Fn_gf:.3f} N')
            records.append(dict(pose_id=i, q=q_actual, Fn_raw=Fn_raw, Fn_gf=Fn_gf, cond=cond))

        print('\n' + '=' * 78)
        print('HELD-OUT VERIFICATION REPORT (poses never used to fit gravity_params.npy)')
        print('=' * 78)
        print(f'{"pose":>4} {"joint angles (deg)":>42} {"||F|| raw":>10} {"||F|| gf":>10} {"cond(J)":>9}')
        for r in records:
            print(f'{r["pose_id"]:>4} {np.array2string(r["q"], precision=1):>42} '
                  f'{r["Fn_raw"]:>10.3f} {r["Fn_gf"]:>10.3f} {r["cond"]:>9.1f}')
        mean_raw = float(np.mean([r['Fn_raw'] for r in records]))
        mean_gf  = float(np.mean([r['Fn_gf']  for r in records]))
        print(f'\nMean ||F|| raw (with firmware gravity model): {mean_raw:.3f} N')
        print(f'Mean ||F|| gravity-free                     : {mean_gf:.3f} N')
        print(f'(For reference, calibration-set result was 21.408N -> 1.688N gravity-free.)')
        if mean_gf < 3.0:
            print('\nVERDICT: generalizes well to new poses — comparable to the calibration-set result.')
        elif mean_gf < 8.0:
            print('\nVERDICT: some residual on new poses, worse than the calibration set but still a '
                  'large improvement over the ~21N pre-calibration baseline — expected some falloff '
                  'away from the fitted poses.')
        else:
            print('\nVERDICT: residual on new poses is close to the pre-calibration baseline — the fit '
                  'may be overfit to the 8 calibration poses specifically.')

    finally:
        api.CloseAPI()
        print('\nAPI closed.')


if __name__ == '__main__':
    main()
