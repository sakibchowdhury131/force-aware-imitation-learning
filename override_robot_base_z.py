#!/usr/bin/env python3
"""
Overrides T_base_task's Z translation with a directly-measured (ruler/caliper)
board-to-base height, keeping the touch-calibrated XY translation and yaw
rotation as-is.

WHY: the touch-based Z estimate in 06_calibrate_robot.py (mean EEF_z during
the 4 touches) has now been observed wrong by ~4.4-4.5cm on two independent
calibration sessions (see project memory / FORCE_SENSING findings: a
2026-06-21/22 session found the same ~4.5cm gap). This looks like a
systematic bias -- most likely GetCartesianPosition()'s reference point
sitting a few cm behind the physical Kinova fingertips, not at them -- rather
than random touch imprecision. XY/yaw were never flagged as suspect, so this
script only overrides Z.

Does NOT touch 06_calibrate_robot.py. Reads any T_base_task .npy and writes
a separate, clearly-named corrected file.

Usage:
    python override_robot_base_z.py \\
        --input data/robot_extrinsics_stick_corrected.npy \\
        --board_below_base_cm 5.3
    # -> data/robot_extrinsics_stick_corrected_zmeasured.npy
"""
import argparse
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', default='data/robot_extrinsics.npy')
    p.add_argument('--output', default=None, help='Default: <input>_zmeasured.npy')
    p.add_argument('--board_below_base_cm', type=float, required=True,
                   help='Directly measured vertical distance: how far BELOW the robot base '
                        'plane the ChArUco board sits (cm). Positive = board below base.')
    return p.parse_args()


def main():
    args = parse_args()
    T = np.load(args.input).astype(np.float64)
    z_new = -args.board_below_base_cm / 100.0

    print(f"Loaded {args.input}")
    print(f"  Z translation (touch-calibrated): {T[2, 3]*100:+.2f} cm")
    print(f"  Z translation (measured override): {z_new*100:+.2f} cm "
         f"(board {args.board_below_base_cm:.1f} cm below robot base)")

    T_new = T.copy()
    T_new[2, 3] = z_new

    out_path = args.output or args.input.replace('.npy', '_zmeasured.npy')
    np.save(out_path, T_new)
    print(f"\nXY translation and yaw kept from the touch-based fit (unaffected by this override):")
    print(T_new.round(4))
    print(f"\nSaved -> {out_path}")
    print(f"Point downstream scripts at this file via --robot_extrinsics {out_path}, "
         f"or rename/copy it over robot_extrinsics.npy once you're satisfied.")


if __name__ == '__main__':
    main()
