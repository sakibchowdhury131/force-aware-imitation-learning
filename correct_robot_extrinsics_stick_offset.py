#!/usr/bin/env python3
"""
Corrects robot_extrinsics.npy (T_base_task, from 06_calibrate_robot.py) for a
known rigid stick/probe offset used during the 4-point touch calibration.

WHY: 06_calibrate_robot.py's touch procedure records GetCartesianPosition()
(the Kinova flange/tool reference point, which sits close to the natural
fingertip point) at each touch -- it has no notion of a probe/stick extending
beyond that. If you touched the board with a stick attached to the fingertips
instead of the bare fingertips, every recorded point is offset from the true
touch point by the stick's length, along the direction the tool was pointing.

ASSUMPTION (must hold for this correction to be valid): the wrist orientation
was kept IDENTICAL for all 4 touches (approaching the board from directly
above, stick pointing straight down, only translating in XY between points --
the natural way to jog a Cartesian-mode arm without touching orientation).
Under that assumption the stick's offset is a single constant vector, and
since 06_calibrate_robot.py's Kabsch fit mean-centers before solving rotation,
a constant offset has ZERO effect on the fitted yaw -- it only shows up as a
Z shift in the translation: the recorded flange height is `stick_length`
ABOVE the true touched surface (the stick reaches down to the board FROM the
flange), so:

    true_z_offset = recorded_z_offset - stick_length

If the wrist orientation was NOT kept constant across the 4 touches, this
correction is NOT valid (the offset direction changes per point, contaminating
XY/yaw too, not just Z) -- redo the touches with fixed orientation instead.

Does NOT modify 06_calibrate_robot.py or overwrite its output -- reads the
given input and writes a separate, clearly-named corrected file.

Usage:
    python correct_robot_extrinsics_stick_offset.py \\
        --input data/robot_extrinsics.npy --stick_length_cm 18.0
    # -> writes data/robot_extrinsics_stick_corrected.npy
"""
import argparse
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', default='data/robot_extrinsics.npy')
    p.add_argument('--output', default=None,
                   help='Default: <input>_stick_corrected.npy')
    p.add_argument('--stick_length_cm', type=float, required=True,
                   help='Stick length from the Kinova fingertips to its tip (cm)')
    return p.parse_args()


def main():
    args = parse_args()
    T = np.load(args.input).astype(np.float64)
    L = args.stick_length_cm / 100.0

    print(f"Loaded {args.input}")
    print(f"T_base_task (before correction):\n{T.round(4)}")
    print(f"  Z translation (before): {T[2, 3]*100:.2f} cm")

    T_corrected = T.copy()
    T_corrected[2, 3] -= L

    print(f"\nSubtracting stick length {args.stick_length_cm:.1f} cm "
         f"(assumes constant, straight-down wrist orientation across all 4 touches)")
    print(f"  Z translation (after):  {T_corrected[2, 3]*100:.2f} cm")
    print(f"T_base_task (after correction):\n{T_corrected.round(4)}")

    out_path = args.output or args.input.replace('.npy', '_stick_corrected.npy')
    np.save(out_path, T_corrected)
    print(f"\nSaved corrected T_base_task -> {out_path}")
    print(f"XY translation and yaw are UNCHANGED by this correction -- it is only "
         f"valid if the wrist orientation was fixed/vertical for all 4 touches. "
         f"Point downstream scripts at this file via --robot_extrinsics "
         f"{out_path}, or rename/copy it over robot_extrinsics.npy once you're "
         f"satisfied.")


if __name__ == '__main__':
    main()
