"""
Step 0 (optional) — Estimate camera extrinsics from a ChArUco board.

DEFAULT (auto) mode:
  Connect a RealSense camera, place the ChArUco board flat in the task workspace,
  run the script. It warms up the camera, detects the board, averages N stable
  frames, saves the result, and exits — no keypresses needed.

  python 00_calibrate.py                          # cam0 → data/cam_extrinsics.npy
  python 00_calibrate.py --track_cam 1            # cam1 → data/cam1_extrinsics.npy

IMAGE mode (offline, board not physically present):
  python 00_calibrate.py --image /path/to/frame.jpg \\
      --meta data/episodes/spoonFood/001/meta.json \\
      --track_cam 1

Board spec — matches charuco_board_400x300mm.png:
  10×7 squares, 40 mm square, 30 mm marker, DICT_4X4_250

What gets saved:
  tf_world2cam (4×4 float64) — transforms task/world-frame points to camera frame.
  Invert to go camera → task:  tf_cam2world = inv(tf_world2cam)

Used by 04_track.py:
  python 04_track.py ... --task_frame data/cam_extrinsics.npy
  → produces tool_poses_task.npz in stable world/task frame
"""

import os, sys, argparse, json, time
import cv2
import numpy as np

# ── Board definition ──────────────────────────────────────────────────────────
_DICT  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
_BOARD = cv2.aruco.CharucoBoard((10, 7), squareLength=0.04,
                                 markerLength=0.03, dictionary=_DICT)

# Rotation aligning OpenCV's board frame with the desired world/task frame:
#   Board (OpenCV): X right, Y down-along-board, Z toward-camera
#   World/task:     X right, Y forward-on-table, Z up
_TF_WORLD2BOARD = np.eye(4, dtype=np.float64)
_TF_WORLD2BOARD[:3, :3] = np.array([[1, 0, 0],
                                     [0, -1, 0],
                                     [0,  0, -1]], dtype=np.float64)


# ── Core detection ────────────────────────────────────────────────────────────

def detect_and_estimate(frame_bgr: np.ndarray, K: np.ndarray, D: np.ndarray):
    """
    Detect ChArUco board and estimate pose.
    Returns (tf_world2cam, n_corners, vis_bgr).
    tf_world2cam is None if detection failed.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, _DICT)
    vis = frame_bgr.copy()

    if ids is None or len(ids) == 0:
        _overlay(vis, "Searching for board...", (0, 100, 255))
        return None, 0, vis

    cv2.aruco.drawDetectedMarkers(vis, corners, ids)

    _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, _BOARD, cameraMatrix=K, distCoeffs=D)

    n = 0 if charuco_corners is None else len(charuco_corners)
    if n < 6:
        _overlay(vis, f"Board partially visible ({n} corners, need ≥6)", (0, 165, 255))
        return None, n, vis

    cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)

    ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners, charuco_ids, _BOARD, K, D, None, None)

    if not ok:
        _overlay(vis, "Pose estimation failed — check lighting", (0, 0, 255))
        return None, n, vis

    R = cv2.Rodrigues(rvec)[0]
    t = tvec.ravel()
    tf_board2cam = np.eye(4, dtype=np.float64)
    tf_board2cam[:3, :3] = R
    tf_board2cam[:3, 3]  = t
    tf_world2cam = tf_board2cam @ _TF_WORLD2BOARD

    draw_world_axes(vis, tf_world2cam, K)

    dist = np.linalg.norm(t)
    _overlay(vis, f"Detected  {n} corners  dist={dist*100:.0f}cm", (0, 220, 0), y=40)
    return tf_world2cam, n, vis


def _project(pts_3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    """pts_3d: (N, 3) in camera frame -> (N, 2) pixel coords (no distortion)."""
    h = (K @ pts_3d.T).T
    return (h[:, :2] / h[:, 2:3]).astype(int)


def draw_world_axes(vis: np.ndarray, tf_world2cam: np.ndarray, K: np.ndarray,
                     scale: float = 0.08) -> np.ndarray:
    """Draw the calibrated WORLD/TASK frame axes (X=right, Y=forward, Z=up)."""
    pts_world = np.array([[0, 0, 0], [scale, 0, 0], [0, scale, 0], [0, 0, scale]],
                          dtype=np.float64)
    pts_cam = (tf_world2cam[:3, :3] @ pts_world.T).T + tf_world2cam[:3, 3]
    if np.any(pts_cam[:, 2] <= 0):
        return vis  # behind camera, skip

    pts_2d = _project(pts_cam, K)
    origin = tuple(pts_2d[0])
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]   # BGR: X=red Y=green Z=blue
    labels = ['X (right)', 'Y (fwd)', 'Z (up)']
    for k in range(3):
        tip = tuple(pts_2d[k + 1])
        cv2.arrowedLine(vis, origin, tip, colors[k], 3, tipLength=0.2)
        cv2.putText(vis, labels[k], tip, cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, colors[k], 2, cv2.LINE_AA)

    cv2.circle(vis, origin, 5, (255, 255, 255), -1)
    cv2.putText(vis, 'WORLD ORIGIN', (origin[0] + 8, origin[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def _overlay(img, text, color, y=40):
    cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color,   2, cv2.LINE_AA)


# ── Pose averaging ────────────────────────────────────────────────────────────

def _average_transforms(tfs):
    """Average a list of (4,4) SE3 matrices: mean translation + Gram-Schmidt mean rotation."""
    t_mean = np.mean([tf[:3, 3] for tf in tfs], axis=0)

    # Average rotations via SVD on the sum of rotation matrices
    R_sum = sum(tf[:3, :3] for tf in tfs)
    U, _, Vt = np.linalg.svd(R_sum)
    R_mean = U @ Vt
    if np.linalg.det(R_mean) < 0:
        U[:, -1] *= -1
        R_mean = U @ Vt

    tf_avg = np.eye(4, dtype=np.float64)
    tf_avg[:3, :3] = R_mean
    tf_avg[:3, 3]  = t_mean
    return tf_avg


# ── Auto mode (default) ───────────────────────────────────────────────────────

def run_auto(args):
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("ERROR: pyrealsense2 not found.")
        print("Install it or use --image mode for offline calibration.")
        sys.exit(1)

    ctx = rs.context()
    devices = list(ctx.devices)
    if not devices:
        print("ERROR: No RealSense cameras detected. Check USB connection.")
        sys.exit(1)

    serials = [d.get_info(rs.camera_info.serial_number) for d in devices]
    print(f"Found {len(serials)} RealSense camera(s): {serials}")
    print(f"Using camera index {args.camera} (serial {serials[args.camera]})")

    sn   = serials[args.camera]
    pipe = rs.pipeline()
    cfg  = rs.config()
    cfg.enable_device(sn)
    cfg.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)
    profile = pipe.start(cfg)

    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0, 0, 1]], dtype=np.float64)
    D = np.array(intr.coeffs, dtype=np.float64)
    print(f"Camera intrinsics: fx={intr.fx:.1f}  fy={intr.fy:.1f}  "
          f"cx={intr.ppx:.1f}  cy={intr.ppy:.1f}")

    # ── Warmup ────────────────────────────────────────────────────────────────
    warmup = args.warmup_frames
    print(f"Warming up ({warmup} frames)...", end='', flush=True)
    for i in range(warmup):
        pipe.wait_for_frames()
        if (i + 1) % 10 == 0:
            print('.', end='', flush=True)
    print(" done.")

    # ── Collect stable detections ─────────────────────────────────────────────
    n_needed   = args.n_stable
    collected  = []
    last_vis   = None   # last annotated frame — saved as _vis.jpg on success
    print(f"\nPlace the ChArUco board flat in the task workspace.")
    print(f"Collecting {n_needed} stable detections...\n")

    show = not args.no_display
    if show:
        cv2.namedWindow("00_calibrate — place board flat on table", cv2.WINDOW_NORMAL)

    try:
        while len(collected) < n_needed:
            fs  = pipe.wait_for_frames(timeout_ms=3000)
            bgr = cv2.cvtColor(
                np.asanyarray(fs.get_color_frame().get_data()), cv2.COLOR_RGB2BGR)

            tf, n_corners, vis = detect_and_estimate(bgr, K, D)

            progress = f"[{len(collected)}/{n_needed}]"
            if tf is not None:
                collected.append(tf)
                last_vis = vis.copy()
                msg = f"{progress} Good detection ({n_corners} corners)"
                print(f"\r{msg:<60}", end='', flush=True)
                _overlay(vis, msg, (0, 255, 0), y=80)
            else:
                print(f"\r{progress} Waiting...{' '*40}", end='', flush=True)

            if show:
                cv2.imshow("00_calibrate — place board flat on table", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nAborted.")
                    pipe.stop()
                    cv2.destroyAllWindows()
                    sys.exit(0)

    finally:
        pipe.stop()
        if show:
            cv2.destroyAllWindows()

    print()  # newline after progress line

    # ── Average and save ──────────────────────────────────────────────────────
    tf_avg = _average_transforms(collected)
    _save_result(tf_avg, K, args.output, last_vis)


# ── Image mode ────────────────────────────────────────────────────────────────

def run_image(args):
    if args.meta:
        with open(args.meta) as f:
            meta = json.load(f)
        K = np.array(meta['intrinsics'][args.camera]['K'], dtype=np.float64)
        D = np.zeros(5, dtype=np.float64)
        print(f"Intrinsics from {args.meta}  cam{args.camera}")
    else:
        K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float64)
        D = np.zeros(5, dtype=np.float64)
        print("WARNING: no --meta given, using estimated intrinsics (less accurate)")

    bgr = cv2.imread(args.image)
    if bgr is None:
        print(f"ERROR: Cannot read image: {args.image}")
        sys.exit(1)
    print(f"Processing {args.image}  ({bgr.shape[1]}×{bgr.shape[0]})")

    tf, n_corners, vis = detect_and_estimate(bgr, K, D)

    if tf is None:
        vis_path = os.path.splitext(args.output)[0] + '_vis.jpg'
        cv2.imwrite(vis_path, vis)
        print(f"Visualisation → {vis_path}")
        print("\nERROR: Board not detected.")
        print("  • Ensure the entire board is visible and well-lit")
        print("  • Board should be lying flat on the table")
        sys.exit(1)

    _save_result(tf, K, args.output, vis)


# ── Save + report ─────────────────────────────────────────────────────────────

def _save_result(tf_world2cam: np.ndarray, K: np.ndarray, output_path: str,
                 vis_frame=None):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.save(output_path, tf_world2cam)

    # Save annotated snapshot so you can verify axis directions
    vis_path = os.path.splitext(output_path)[0] + '_vis.jpg'
    if vis_frame is not None:
        cv2.imwrite(vis_path, vis_frame)
        print(f"\nVisualisation → {vis_path}")
        print("  Check: red=X(right)  green=Y(forward)  blue=Z(up)")

    tf_cam2world = np.linalg.inv(tf_world2cam)
    cam_pos = tf_cam2world[:3, 3]

    print(f"\ntf_world2cam:\n{tf_world2cam.round(4)}")
    print(f"\nCamera position in task frame (should match your physical setup):")
    print(f"  x = {cam_pos[0]*100:+.1f} cm   (left/right of board origin)")
    print(f"  y = {cam_pos[1]*100:+.1f} cm   (forward/back from board origin)")
    print(f"  z = {cam_pos[2]*100:+.1f} cm   (height above table)  ← should be ~40–80 cm")
    print(f"\nSaved → {output_path}")
    print(f"\nNext step:")
    print(f"  python 04_track.py --task_dir data/episodes/<task> "
          f"--tool_prompt <prompt> --mesh <mesh.obj> --task_frame {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Estimate camera extrinsics from a ChArUco board in the task workspace.\n'
                    'Default: auto mode — connect camera, detect board, save, exit.')
    p.add_argument('--image',  default=None,
                   help='Offline mode: path to a saved frame with the board visible')
    p.add_argument('--meta',   default=None,
                   help='Path to episode meta.json (for intrinsics in --image mode)')
    p.add_argument('--output', default=None,
                   help='Output .npy path. Defaults to data/cam_extrinsics.npy for cam0, '
                        'data/cam1_extrinsics.npy for cam1, etc.')
    p.add_argument('--track_cam', type=int, default=0,
                   help='RealSense device index to calibrate (default: 0)')
    p.add_argument('--camera', type=int, default=None,
                   help='Alias for --track_cam (deprecated)')
    p.add_argument('--warmup_frames', type=int, default=30,
                   help='Frames to discard for camera warmup (default: 30 ≈ 1 s)')
    p.add_argument('--n_stable', type=int, default=10,
                   help='Number of good detections to average (default: 10)')
    p.add_argument('--no_display', action='store_true',
                   help='Suppress the preview window (useful on headless machines)')
    args = p.parse_args()
    if args.camera is not None:
        args.track_cam = args.camera
    if args.output is None:
        args.output = ('data/cam_extrinsics.npy' if args.track_cam == 0
                       else f'data/cam{args.track_cam}_extrinsics.npy')
    # used internally by run_auto / run_image
    args.camera = args.track_cam
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.image:
        run_image(args)
    else:
        run_auto(args)
