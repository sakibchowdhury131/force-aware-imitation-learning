#!/usr/bin/env python3
"""
Upload the calibrated firmware gravity matrix (data/gravity_params.npy) to
the robot. This is a session-only firmware setting — confirmed on hardware
to reset after a power cycle (residual jumps back from ~1.7-1.9N to ~27N
until reapplied) — so run this once at the start of any session that needs
accurate gravity-free torque readings.

Read/write of firmware state only — never sends a motion command, safe to
run any time regardless of arm pose.

(Note: this already happens automatically at startup in diag_contact.py,
calibrate_contact_baseline.py, diag_joint_torques.py, replay_episode.py,
07_deploy.py, teach_by_hand.py, quick_pose_check.py, and
test_residual_repeatability.py. Use this script when you just want to
apply it standalone — e.g. before hand-driving the arm with the joystick,
or before a script that doesn't do this itself.)

Usage:
    python upload_gravity_params.py
    python upload_gravity_params.py --params_path data/gravity_params.npy
"""
import os, sys, argparse

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)
from calibrate_firmware_gravity import load_api, connect, apply_saved_gravity_params


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--params_path', default=os.path.join(PIPELINE_DIR, 'data', 'gravity_params.npy'),
                   help='Fitted gravity params from calibrate_firmware_gravity.py '
                        '(default: data/gravity_params.npy)')
    return p.parse_args()


def main():
    args = parse_args()
    api = load_api()
    try:
        connect(api, control=True)
        ok = apply_saved_gravity_params(api, path=args.params_path)
        if ok:
            print('\nSUCCESS — calibrated gravity params are now active on the robot.')
        else:
            print('\nFAILED — see message above (missing file or SDK error). '
                  'Gravity-free readings will use the uncalibrated firmware default.')
    finally:
        api.CloseAPI()
        print('API closed.')


if __name__ == '__main__':
    main()
