#!/usr/bin/env python3
"""
Standalone sanity-check viewer: overlays a robot base frame origin + XYZ axes
onto a live camera feed, given any T_base_task (4x4) .npy file. Lets you
visually confirm a candidate T_base_task (e.g. the stick-offset-corrected one
from correct_robot_extrinsics_stick_offset.py) against where the robot base
actually sits in the image -- without re-running the whole touch-point
calibration in 06_calibrate_robot.py (whose built-in post-calibration check
only ever shows the T_base_task it just solved, not an arbitrary file).

Chain (same as 06_calibrate_robot.py's _show_robot_base_in_camera):
    robot base origin (0,0,0) in base frame
      -> task frame:   inv(T_base_task)
      -> camera frame: tf_world2cam @ point_task
      -> pixels:       K @ point_cam / z

Does NOT touch 06_calibrate_robot.py.

Usage:
    python visualize_robot_base_in_camera.py \\
        --T_base_task data/robot_extrinsics_stick_corrected.npy --track_cam 1

Press Q to close.
"""
import os, argparse
import numpy as np
import cv2

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--T_base_task', default='data/robot_extrinsics.npy',
                   help='Any T_base_task .npy to visualize (e.g. the stick-corrected file)')
    p.add_argument('--task_frame', default=None,
                   help='tf_world2cam .npy. Default: data/cam_extrinsics.npy for cam0, '
                        'data/cam{N}_extrinsics.npy for camN.')
    p.add_argument('--track_cam', type=int, default=1, help='Camera index (default: 1)')
    p.add_argument('--axis_len', type=float, default=0.10, help='Axis arrow length (m)')
    return p.parse_args()


def main():
    args = parse_args()
    if args.task_frame is None:
        args.task_frame = ('data/cam_extrinsics.npy' if args.track_cam == 0
                           else f'data/cam{args.track_cam}_extrinsics.npy')

    T_base_task_path = os.path.join(PIPELINE_DIR, args.T_base_task)
    task_frame_path  = os.path.join(PIPELINE_DIR, args.task_frame)

    T_base_task = np.load(T_base_task_path).astype(np.float64)
    tf_world2cam = np.load(task_frame_path).astype(np.float64)
    T_task_base = np.linalg.inv(T_base_task)

    print(f"T_base_task <- {T_base_task_path}")
    print(T_base_task.round(4))
    print(f"tf_world2cam <- {task_frame_path}")

    axis_len = args.axis_len
    pts_base = np.array([
        [0, 0, 0],
        [axis_len, 0, 0],
        [0, axis_len, 0],
        [0, 0, axis_len],
    ], dtype=np.float64)

    pts_task = (T_task_base[:3, :3] @ pts_base.T).T + T_task_base[:3, 3]
    pts_cam  = (tf_world2cam[:3, :3] @ pts_task.T).T + tf_world2cam[:3, 3]

    import pyrealsense2 as rs
    ctx = rs.context()
    devices = list(ctx.devices)
    if not devices or args.track_cam >= len(devices):
        raise RuntimeError(f"--track_cam {args.track_cam} requested but only "
                           f"{len(devices)} camera(s) connected.")
    serial = devices[args.track_cam].get_info(rs.camera_info.serial_number)
    print(f"Using camera {args.track_cam} (serial {serial})")

    pipe = rs.pipeline()
    cfg  = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float64)

    def proj(p_cam):
        if p_cam[2] <= 0:
            return None
        u = int(K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2])
        v = int(K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2])
        return (u, v)

    COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]   # BGR: X=red, Y=green, Z=blue
    LABELS = ['X', 'Y', 'Z']

    cv2.namedWindow("Robot base in camera — press Q to close", cv2.WINDOW_NORMAL)
    print("\nShowing robot base frame in camera. Press Q to close.")

    for _ in range(60):
        pipe.wait_for_frames()

    try:
        while True:
            fs  = pipe.wait_for_frames(timeout_ms=3000)
            img = np.asanyarray(fs.get_color_frame().get_data())
            vis = img.copy()

            origin_px = proj(pts_cam[0])
            if origin_px:
                for i, (col, lbl) in enumerate(zip(COLORS, LABELS)):
                    tip_px = proj(pts_cam[i + 1])
                    if tip_px:
                        cv2.arrowedLine(vis, origin_px, tip_px, col, 3,
                                        tipLength=0.2, line_type=cv2.LINE_AA)
                        cv2.putText(vis, lbl, tip_px, cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, col, 2, cv2.LINE_AA)
                cv2.circle(vis, origin_px, 7, (255, 255, 255), -1)
                cv2.putText(vis, "ROBOT BASE", (origin_px[0] + 10, origin_px[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(vis, "Robot base origin behind/outside camera",
                            (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

            cv2.putText(vis, f"{os.path.basename(args.T_base_task)}  |  Q = close",
                        (16, vis.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow("Robot base in camera — press Q to close", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
