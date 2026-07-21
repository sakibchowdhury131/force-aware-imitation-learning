#!/usr/bin/env python3
"""
Checks whether a camera has physically moved since it was last calibrated
with 00_calibrate.py, WITHOUT overwriting the saved extrinsics.

Re-detects the ChArUco board live (same detection + averaging logic as
00_calibrate.py's auto mode — reused via import, that file is untouched),
computes a fresh tf_world2cam, and compares it against the saved
data/cam{N}_extrinsics.npy: how far the camera's position/orientation in the
task/world frame appears to have drifted.

CAVEAT: this measures camera pose RELATIVE TO THE BOARD'S CURRENT PLACEMENT.
If the board itself moved (not just the camera), this reports the combined
drift, not camera movement in isolation — only trust this check if you're
confident the board hasn't been nudged since the original calibration.

Usage:
    python check_camera_extrinsics_drift.py                    # checks cam0 and cam1
    python check_camera_extrinsics_drift.py --track_cam 1       # just cam1
    python check_camera_extrinsics_drift.py --track_cam 0 1 --n_stable 15

Place the ChArUco board flat in the task workspace before running (same as
00_calibrate.py) — it must be visible to whichever camera(s) you're checking.
"""
import os, sys, argparse
import cv2
import numpy as np

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)

# Reused, not modified — see module docstring. '00_calibrate' starts with a
# digit, not a valid module name for `import`, so load it by file path.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    '_calibrate00', os.path.join(PIPELINE_DIR, '00_calibrate.py'))
_calibrate00 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_calibrate00)

detect_and_estimate = _calibrate00.detect_and_estimate
draw_world_axes = _calibrate00.draw_world_axes
_average_transforms = _calibrate00._average_transforms
_overlay = _calibrate00._overlay


def pose_drift(tf_old, tf_new):
    """(trans_cm, rot_deg) of camera position/orientation in world frame."""
    cam2world_old = np.linalg.inv(tf_old)
    cam2world_new = np.linalg.inv(tf_new)
    trans_cm = float(np.linalg.norm(cam2world_new[:3, 3] - cam2world_old[:3, 3]) * 100)
    R_rel = cam2world_old[:3, :3].T @ cam2world_new[:3, :3]
    from scipy.spatial.transform import Rotation
    rot_deg = float(np.degrees(Rotation.from_matrix(R_rel).magnitude()))
    return trans_cm, rot_deg


def draw_axes_dashed_style(vis, tf_world2cam, K, scale=0.08, label='SAVED', color=(255, 255, 255)):
    """Thin/hollow-tip axes for the OLD (saved) calibration, distinguishable
    from detect_and_estimate's own (solid) live-detection overlay."""
    pts_world = np.array([[0, 0, 0], [scale, 0, 0], [0, scale, 0], [0, 0, scale]], dtype=np.float64)
    pts_cam = (tf_world2cam[:3, :3] @ pts_world.T).T + tf_world2cam[:3, 3]
    if np.any(pts_cam[:, 2] <= 0):
        return vis
    h = (K @ pts_cam.T).T
    pts_2d = (h[:, :2] / h[:, 2:3]).astype(int)
    origin = tuple(pts_2d[0])
    for k in range(3):
        tip = tuple(pts_2d[k + 1])
        cv2.line(vis, origin, tip, color, 1, cv2.LINE_AA)
        cv2.circle(vis, tip, 4, color, 1, cv2.LINE_AA)
    cv2.circle(vis, origin, 5, color, 1, cv2.LINE_AA)
    cv2.putText(vis, label, (origin[0] + 8, origin[1] + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis


def check_one_camera(track_cam, extrinsics_path, n_stable, warmup_frames, no_display):
    import pyrealsense2 as rs

    extrinsics_path = os.path.join(PIPELINE_DIR, extrinsics_path)
    if not os.path.exists(extrinsics_path):
        print(f"cam{track_cam}: SKIPPED — {extrinsics_path} not found (never calibrated)")
        return None

    tf_old = np.load(extrinsics_path).astype(np.float64)

    ctx = rs.context()
    devices = list(ctx.devices)
    if track_cam >= len(devices):
        print(f"cam{track_cam}: SKIPPED — camera index not found "
             f"({len(devices)} camera(s) connected)")
        return None
    serial = devices[track_cam].get_info(rs.camera_info.serial_number)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)
    profile = pipe.start(cfg)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float64)
    D = np.array(intr.coeffs, dtype=np.float64)

    print(f"\ncam{track_cam} (serial {serial}): warming up ({warmup_frames} frames)...")
    for _ in range(warmup_frames):
        pipe.wait_for_frames()

    print(f"cam{track_cam}: collecting {n_stable} stable board detections...")
    show = not no_display
    win_name = f"Camera drift check — cam{track_cam}"
    if show:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    collected = []
    last_vis = None
    try:
        while len(collected) < n_stable:
            fs = pipe.wait_for_frames(timeout_ms=3000)
            bgr = cv2.cvtColor(np.asanyarray(fs.get_color_frame().get_data()), cv2.COLOR_RGB2BGR)
            tf_now, n_corners, vis = detect_and_estimate(bgr, K, D)

            if tf_now is not None:
                collected.append(tf_now)
                last_vis = vis.copy()
                print(f"\r  [{len(collected)}/{n_stable}] good detection ({n_corners} corners)"
                     f"{' ' * 20}", end='', flush=True)

            if show:
                vis_disp = vis.copy()
                draw_axes_dashed_style(vis_disp, tf_old, K, label='SAVED (old)', color=(180, 180, 180))
                cv2.putText(vis_disp, f"[{len(collected)}/{n_stable}]  gray=saved calibration, "
                                      f"colored=live detection  |  Q=abort",
                            (16, vis_disp.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(win_name, vis_disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print(f"\ncam{track_cam}: aborted.")
                    return None
    finally:
        pipe.stop()
        if show:
            cv2.destroyAllWindows()
    print()

    tf_now_avg = _average_transforms(collected)
    trans_cm, rot_deg = pose_drift(tf_old, tf_now_avg)

    print(f"cam{track_cam} drift vs. {extrinsics_path}:")
    print(f"  translation: {trans_cm:.2f} cm")
    print(f"  rotation:    {rot_deg:.2f} deg")

    # Save a comparison snapshot: saved (gray, thin) axes + fresh (colored, thick) axes
    if last_vis is not None:
        vis_out = last_vis.copy()
        draw_axes_dashed_style(vis_out, tf_old, K, label='SAVED (old)', color=(180, 180, 180))
        draw_world_axes(vis_out, tf_now_avg, K)
        cv2.putText(vis_out, f"drift: trans={trans_cm:.2f}cm  rot={rot_deg:.2f}deg",
                    (16, vis_out.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2, cv2.LINE_AA)
        out_path = os.path.join(PIPELINE_DIR, f'data/cam{track_cam}_drift_check.jpg')
        cv2.imwrite(out_path, vis_out)
        print(f"  Comparison snapshot -> {out_path}")

    return dict(track_cam=track_cam, trans_cm=trans_cm, rot_deg=rot_deg,
               extrinsics_path=extrinsics_path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--track_cam', type=int, nargs='+', default=[0, 1],
                   help='Camera index/indices to check (default: 0 1)')
    p.add_argument('--n_stable', type=int, default=10)
    p.add_argument('--warmup_frames', type=int, default=30)
    p.add_argument('--no_display', action='store_true')
    p.add_argument('--trans_warn_cm', type=float, default=1.0,
                   help='Flag as likely-moved beyond this translation drift (cm)')
    p.add_argument('--rot_warn_deg', type=float, default=1.5,
                   help='Flag as likely-moved beyond this rotation drift (deg)')
    return p.parse_args()


def main():
    args = parse_args()
    results = []
    for cam in args.track_cam:
        path = 'data/cam_extrinsics.npy' if cam == 0 else f'data/cam{cam}_extrinsics.npy'
        r = check_one_camera(cam, path, args.n_stable, args.warmup_frames, args.no_display)
        if r is not None:
            results.append(r)

    print(f"\n{'='*60}\nSummary\n{'='*60}")
    for r in results:
        moved = r['trans_cm'] > args.trans_warn_cm or r['rot_deg'] > args.rot_warn_deg
        verdict = "LIKELY MOVED — consider re-running 00_calibrate.py" if moved else "OK, within noise"
        print(f"cam{r['track_cam']}: trans={r['trans_cm']:.2f}cm  rot={r['rot_deg']:.2f}deg  "
             f"-> {verdict}")
    if not results:
        print("No cameras checked.")


if __name__ == '__main__':
    main()
