"""
Quick visualization — project the robot base frame (origin + XYZ axes) onto
a live camera frame, using the calibrations from 00_calibrate.py and
06_calibrate_robot.py.

Usage:
  python show_robot_frame.py                  # cam0 (default)
  python show_robot_frame.py --track_cam 1    # cam1
"""

import os
import argparse
import cv2
import numpy as np
import pyrealsense2 as rs


def _project(pts_3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    h = (K @ pts_3d.T).T
    return (h[:, :2] / h[:, 2:3]).astype(int)


def draw_frame_axes(vis, T_cam_frame, K, scale=0.08, label='ROBOT BASE'):
    """T_cam_frame: transforms the target frame's points -> camera frame."""
    pts_local = np.array([[0, 0, 0], [scale, 0, 0], [0, scale, 0], [0, 0, scale]], dtype=np.float64)
    pts_cam = (T_cam_frame[:3, :3] @ pts_local.T).T + T_cam_frame[:3, 3]
    if np.any(pts_cam[:, 2] <= 0):
        print("WARNING: frame origin/axes behind camera — cannot draw")
        return vis

    pts_2d = _project(pts_cam, K)
    origin = tuple(pts_2d[0])
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]   # BGR: X=red Y=green Z=blue
    labels = ['X', 'Y', 'Z']
    for k in range(3):
        tip = tuple(pts_2d[k + 1])
        cv2.arrowedLine(vis, origin, tip, colors[k], 3, tipLength=0.2)
        cv2.putText(vis, labels[k], tip, cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, colors[k], 2, cv2.LINE_AA)

    cv2.circle(vis, origin, 6, (0, 255, 255), -1)
    cv2.putText(vis, label, (origin[0] + 8, origin[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return vis


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--track_cam', type=int, default=0,
                   help='RealSense device index (default: 0)')
    p.add_argument('--robot_extrinsics', default='data/robot_extrinsics.npy')
    args = p.parse_args()

    cam_extr_path = ('data/cam_extrinsics.npy' if args.track_cam == 0
                     else f'data/cam{args.track_cam}_extrinsics.npy')

    T_base_task  = np.load(args.robot_extrinsics)
    tf_world2cam = np.load(cam_extr_path)
    tf_cam2world = np.linalg.inv(tf_world2cam)

    # T_cam_base: transforms robot-base-frame points -> camera frame
    T_base_cam = T_base_task @ tf_cam2world
    T_cam_base = np.linalg.inv(T_base_cam)

    pos = T_cam_base[:3, 3]
    print(f"Robot base origin in cam{args.track_cam} frame (cm): "
          f"x={pos[0]*100:+.1f}  y={pos[1]*100:+.1f}  z={pos[2]*100:+.1f}")

    ctx = rs.context()
    devices = list(ctx.devices)
    if not devices or args.track_cam >= len(devices):
        print(f"ERROR: camera {args.track_cam} not found ({len(devices)} camera(s) connected)")
        return
    serial = devices[args.track_cam].get_info(rs.camera_info.serial_number)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float64)

    try:
        for _ in range(30):  # warmup
            pipe.wait_for_frames()
        fs = pipe.wait_for_frames()
        img = np.asanyarray(fs.get_color_frame().get_data())
    finally:
        pipe.stop()

    vis = draw_frame_axes(img, T_cam_base, K, label=f'ROBOT BASE (cam{args.track_cam})')
    suffix = '' if args.track_cam == 0 else f'_cam{args.track_cam}'
    out_path = f'data/robot_extrinsics_vis{suffix}.jpg'
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, vis)
    print(f"Saved -> {out_path}")


if __name__ == '__main__':
    main()
