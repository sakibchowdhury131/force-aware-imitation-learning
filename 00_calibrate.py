"""
Step 0 (optional) — Estimate camera extrinsics from a ChArUco board.

Place the printed ChArUco board flat in the task workspace (on the table,
where the actual task happens). Point cam0 at it and either:
  (a) pass a saved image from that session, or
  (b) run live with a connected RealSense camera.

Board spec — matches charuco_board_400x300mm.png:
  10×7 squares, 40 mm square, 30 mm marker, DICT_4X4_250

What gets saved:
  tf_world2cam  (4×4 float64) — transforms a point in task/world frame to camera frame.
  To go the other direction (camera → task):  tf_cam2world = inv(tf_world2cam)

This is then used by 04_track.py --task_frame to produce tool_poses_task.npz,
whose poses are in the stable world/task frame instead of the camera frame.

Usage — from a saved image (recommended):
  # First capture one frame with the board visible in the task workspace:
  #   just copy any frame from cam0/ that shows the board, e.g.:
  #   cp data/episodes/spoonFood/001/cam0/000000.jpg /tmp/board_frame.jpg
  python 00_calibrate.py \\
      --image /tmp/board_frame.jpg \\
      --meta  data/episodes/spoonFood/001/meta.json \\
      --output data/cam_extrinsics.npy

Usage — live RealSense:
  python 00_calibrate.py --live --output data/cam_extrinsics.npy

Then re-run tracking with the task frame:
  python 04_track.py --task_dir data/episodes/spoonFood \\
      --tool_prompt "spoon" --mesh spoon.obj \\
      --task_frame data/cam_extrinsics.npy
"""

import os, sys, argparse, json
import cv2
import numpy as np

# Board parameters — must match build_charuko_board.py / charuco_board_400x300mm.png
_DICT  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
_BOARD = cv2.aruco.CharucoBoard((10, 7), squareLength=0.04,
                                 markerLength=0.03, dictionary=_DICT)

# Rotation that aligns OpenCV's board frame with our desired world/task frame:
#   Board (OpenCV): X right, Y down-along-board, Z toward-camera
#   World/task:     X right, Y forward-on-table, Z up
# Flipping Y and Z achieves this (same convention as the original calibrate_cameras.py).
_TF_WORLD2BOARD = np.eye(4, dtype=np.float64)
_TF_WORLD2BOARD[:3, :3] = np.array([[1, 0, 0],
                                     [0, -1, 0],
                                     [0,  0, -1]], dtype=np.float64)


def detect_and_estimate(frame_bgr: np.ndarray, K: np.ndarray, D: np.ndarray):
    """
    Detect ChArUco board in frame_bgr and estimate tf_world2cam.

    Returns (tf_world2cam, vis_bgr) where tf_world2cam is None on failure.
    tf_world2cam: (4,4) — transforms task/world-frame points to camera frame.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, _DICT)
    vis = frame_bgr.copy()

    if ids is None or len(ids) == 0:
        cv2.putText(vis, "No ArUco markers detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return None, vis

    cv2.aruco.drawDetectedMarkers(vis, corners, ids)
    cv2.putText(vis, f"{len(ids)} markers detected", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, _BOARD, cameraMatrix=K, distCoeffs=D)

    if charuco_corners is None or len(charuco_corners) < 6:
        cv2.putText(vis, f"Only {0 if charuco_corners is None else len(charuco_corners)} "
                        f"ChArUco corners — need ≥6", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        return None, vis

    cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)

    ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners, charuco_ids, _BOARD, K, D, None, None)

    if not ok:
        cv2.putText(vis, "Pose estimation failed", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return None, vis

    cv2.drawFrameAxes(vis, K, D, rvec, tvec, 0.05)

    R = cv2.Rodrigues(rvec)[0]
    t = tvec.ravel()
    tf_board2cam = np.eye(4, dtype=np.float64)
    tf_board2cam[:3, :3] = R
    tf_board2cam[:3, 3]  = t

    # T^cam_board  @  T^board_world  =  T^cam_world
    tf_world2cam = tf_board2cam @ _TF_WORLD2BOARD

    dist = np.linalg.norm(t)
    cv2.putText(vis, f"Board: {dist*100:.1f} cm away  |  {len(charuco_corners)} corners",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(vis, "Board detected — press S to save", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return tf_world2cam, vis


def _print_result(tf_world2cam: np.ndarray, output_path: str):
    print(f"\ntf_world2cam (W2C):\n{tf_world2cam.round(4)}")
    cam_in_task = np.linalg.inv(tf_world2cam)[:3, 3]
    print(f"Camera origin in task frame: "
          f"x={cam_in_task[0]*100:.1f}  y={cam_in_task[1]*100:.1f}  "
          f"z={cam_in_task[2]*100:.1f} cm  ← should equal camera height above table")
    print(f"\nSaved → {output_path}")
    print(f"\nNext: python 04_track.py ... --task_frame {output_path}")


def run_image(args):
    # Load intrinsics
    if args.meta:
        with open(args.meta) as f:
            meta = json.load(f)
        cam_idx = args.camera
        K = np.array(meta['intrinsics'][cam_idx]['K'], dtype=np.float64)
        D = np.zeros(5, dtype=np.float64)   # RealSense SDK already un-distorts JPEG output
        print(f"Loaded intrinsics from {args.meta}  (cam{cam_idx})")
    else:
        K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float64)
        D = np.zeros(5, dtype=np.float64)
        print("WARNING: no --meta given, using estimated intrinsics (may be inaccurate)")

    frame_bgr = cv2.imread(args.image)
    if frame_bgr is None:
        raise FileNotFoundError(f"Cannot read: {args.image}")
    print(f"Processing {args.image}  ({frame_bgr.shape[1]}×{frame_bgr.shape[0]})")

    tf, vis = detect_and_estimate(frame_bgr, K, D)

    vis_path = os.path.splitext(args.output)[0] + '_vis.jpg'
    cv2.imwrite(vis_path, vis)
    print(f"Visualisation → {vis_path}")

    if tf is None:
        print("\nERROR: Board not detected in this image.")
        print("Tips:")
        print("  • Make sure the entire board is visible and well-lit")
        print("  • The board should be lying flat on the table")
        print("  • Try a different frame where the board is more visible")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.save(args.output, tf)
    _print_result(tf, args.output)


def run_live(args):
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("pyrealsense2 not found. Use --image mode instead.")
        sys.exit(1)

    ctx = rs.context()
    serials = [d.get_info(rs.camera_info.serial_number) for d in ctx.devices]
    if not serials:
        raise RuntimeError("No RealSense cameras found")
    print(f"Found {len(serials)} camera(s): {serials}  — using index {args.camera}")

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

    for _ in range(20):
        pipe.wait_for_frames()

    print("\nLive feed started.")
    print("Place the ChArUco board flat in the task workspace, fully in view.")
    print("  S = save calibration    Q = quit\n")

    last_good = None
    while True:
        fs  = pipe.wait_for_frames(timeout_ms=2000)
        bgr = cv2.cvtColor(
            np.asanyarray(fs.get_color_frame().get_data()), cv2.COLOR_RGB2BGR)

        tf, vis = detect_and_estimate(bgr, K, D)
        if tf is not None:
            last_good = tf

        cv2.imshow("00_calibrate — hold board flat on table  |  S=save  Q=quit", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Quit without saving.")
            break
        if key == ord('s'):
            if last_good is None:
                print("No good detection yet — keep the board visible.")
            else:
                os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
                np.save(args.output, last_good)
                _print_result(last_good, args.output)
                break

    pipe.stop()
    cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(
        description='Estimate cam0 extrinsics from a ChArUco board in the task workspace.')
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--image', help='Path to a frame (JPG/PNG) with the board visible')
    mode.add_argument('--live',  action='store_true', help='Live RealSense feed')
    p.add_argument('--meta',   default=None,
                   help='Path to episode meta.json (used to read camera intrinsics K)')
    p.add_argument('--camera', type=int, default=0,
                   help='Camera index in meta.json / RealSense device list (default: 0)')
    p.add_argument('--output', default='data/cam_extrinsics.npy',
                   help='Output .npy path for tf_world2cam (default: data/cam_extrinsics.npy)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.image:
        run_image(args)
    else:
        run_live(args)
