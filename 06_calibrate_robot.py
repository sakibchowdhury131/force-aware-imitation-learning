"""
Step 6 (optional) — Calibrate the Kinova Jaco2 base frame relative to the
task frame established by 00_calibrate.py (the ChArUco board).

ASSUMPTION: both the robot base Z-axis and the task-frame Z-axis are
vertical (gravity-aligned) — true for a table-mounted arm with the ChArUco
board lying flat on the table. Under this assumption T_base_task reduces to
a 2D rigid transform (yaw rotation about Z + XY translation) plus a single
Z offset (table height relative to the robot base origin).

VISUAL GUIDANCE (default, requires RealSense camera connected):
  Live cam0 feed shows all 4 target points reprojected using the calibrated
  tf_world2cam — the current target is drawn as a large yellow crosshair,
  the others as dim circles. Jog the arm until the tip is on the crosshair,
  then press SPACE in the image window to record that point (Q to abort).

PROCEDURE (text-only fallback, --no_visual or no camera found):
  1. Open data/cam_extrinsics_vis.jpg — it shows the WORLD ORIGIN marker and
     the calibrated task-frame axes (red=X right, green=Y forward, blue=Z up)
     drawn at the origin corner of the ChArUco board.
  2. Attach a pointed tip to the gripper (or use a fingertip as reference).
  3. Using the Kinova joystick / GUI, jog the arm so the tip touches each of
     the 4 marked points below (in order). Press Enter at each to record the
     robot's current Cartesian position (base frame).
  4. The script solves T_base_task and saves it.

NOTE: the green (+Y / "forward") arrow in _vis.jpg points AWAY from the
board (off its edge) — the board itself extends in the -Y direction from
the origin corner. The 4 touch points below are the board's interior-corner-
grid extremes (task frame, z=0):
  P0 = origin corner               (  0.00,   0.00, 0) cm  <- WORLD ORIGIN marker
  P1 = +36cm along red (X) axis    ( 36.00,   0.00, 0) cm
  P2 = -24cm along the board's      (  0.00, -24.00, 0) cm  <- opposite the green arrow
       short edge (perpendicular to red, on the board surface)
  P3 = diagonal corner             ( 36.00, -24.00, 0) cm

Usage:
  python 06_calibrate_robot.py --output data/robot_extrinsics.npy
  python 06_calibrate_robot.py --no_visual               # text-only, no camera
"""

import os, sys, ctypes, argparse
import numpy as np
import cv2

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_DIR = os.path.join(os.path.expanduser('~/working_dir/kinovaDrivers'), 'sdk', 'lib')

if _LIB_DIR not in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
    os.environ["LD_LIBRARY_PATH"] = _LIB_DIR + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.execv(sys.executable, [sys.executable] + sys.argv)

NO_ERROR_KINOVA   = 1
SERIAL_LENGTH     = 20
MAX_KINOVA_DEVICE = 20

LIB_PATH      = os.path.join(_LIB_DIR, "USBCommandLayerUbuntu.so")
COMM_LIB_PATH = os.path.join(_LIB_DIR, "USBCommLayerUbuntu.so")

# Same world/board alignment as 00_calibrate.py
_TF_WORLD2BOARD = np.eye(4, dtype=np.float64)
_TF_WORLD2BOARD[:3, :3] = np.array([[1, 0, 0],
                                     [0, -1, 0],
                                     [0,  0, -1]], dtype=np.float64)

# Touch points in BOARD frame (z=0); converted to task/world frame below via
# _TF_WORLD2BOARD. P0 = origin corner = WORLD ORIGIN marker in _vis.jpg.
# P1 = +36cm along red (X) axis. P2/P3 = along the board's other edge, which
# is the OPPOSITE direction from the green (+Y) arrow drawn in _vis.jpg.
BOARD_POINTS = {
    'P0 origin corner (WORLD ORIGIN marker in _vis.jpg)':         (0.00, 0.00, 0.0),
    'P1 corner +36cm along red (X) axis':                         (0.36, 0.00, 0.0),
    'P2 corner -24cm along board short edge (opp. green arrow)':  (0.00, 0.24, 0.0),
    'P3 diagonal far corner':                                     (0.36, 0.24, 0.0),
}


# ---------- Joint-angle FK (same chain as deploy_viz.py / 07_deploy.py) ----------

_PI = np.pi
_FK_JOINT_PARAMS = [
    ([0,       0,       0.15675], [0,      _PI,   0    ]),
    ([0,       0.0016, -0.11875], [-_PI/2, 0,     _PI  ]),
    ([0,      -0.410,  0       ], [0,      _PI,   0    ]),
    ([0,       0.2073, -0.0114 ], [-_PI/2, 0,     _PI  ]),
    ([0,       0,      -0.10375], [ _PI/2, 0,     _PI  ]),
    ([0,       0.10375, 0      ], [-_PI/2, 0,     _PI  ]),
]
_FK_EEF_PARAMS = ([0, 0, -0.1600], [_PI, 0, _PI/2])


def _fk_T(xyz, rpy):
    from scipy.spatial.transform import Rotation
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    T[:3,  3] = xyz
    return T


def joint_angles_to_eef_xyz(q_deg: np.ndarray) -> np.ndarray:
    """Return EEF xyz (metres) from joint angles (degrees) using the DH chain."""
    from scipy.spatial.transform import Rotation
    q = np.deg2rad(q_deg)
    T = np.eye(4, dtype=np.float64)
    for (xyz, rpy), qi in zip(_FK_JOINT_PARAMS, q):
        Tj = np.eye(4, dtype=np.float64)
        Tj[:3, :3] = Rotation.from_euler('z', float(qi)).as_matrix()
        T = T @ _fk_T(xyz, rpy) @ Tj
    T = T @ _fk_T(*_FK_EEF_PARAMS)
    return T[:3, 3]


# ---------- ctypes structs ----------

class KinovaDevice(ctypes.Structure):
    _fields_ = [
        ("SerialNumber",   ctypes.c_char * SERIAL_LENGTH),
        ("Model",          ctypes.c_char * SERIAL_LENGTH),
        ("VersionMajor",   ctypes.c_int),
        ("VersionMinor",   ctypes.c_int),
        ("VersionRelease", ctypes.c_int),
        ("DeviceType",     ctypes.c_int),
        ("DeviceID",       ctypes.c_int),
    ]

class CartesianInfo(ctypes.Structure):
    _fields_ = [("X", ctypes.c_float), ("Y", ctypes.c_float), ("Z", ctypes.c_float),
                ("ThetaX", ctypes.c_float), ("ThetaY", ctypes.c_float), ("ThetaZ", ctypes.c_float)]

class FingersPosition(ctypes.Structure):
    _fields_ = [("Finger1", ctypes.c_float),
                ("Finger2", ctypes.c_float),
                ("Finger3", ctypes.c_float)]

class CartesianPosition(ctypes.Structure):
    _fields_ = [("Coordinates", CartesianInfo),
                ("Fingers",     FingersPosition)]

class AngularInfo(ctypes.Structure):
    _fields_ = [(f"Actuator{i}", ctypes.c_float) for i in range(1, 8)]

class AngularPosition(ctypes.Structure):
    _fields_ = [("Actuators", AngularInfo), ("Fingers", FingersPosition)]


def load_api():
    ctypes.CDLL(COMM_LIB_PATH, mode=ctypes.RTLD_GLOBAL)
    api = ctypes.CDLL(LIB_PATH)
    for fn, restype in [
        ("InitAPI", ctypes.c_int),
        ("CloseAPI", ctypes.c_int),
        ("RefresDevicesList", ctypes.c_int),
        ("GetDevices", ctypes.c_int),
        ("SetActiveDevice", ctypes.c_int),
        ("StartControlAPI", ctypes.c_int),
        ("StopControlAPI", ctypes.c_int),
        ("GetCartesianPosition", ctypes.c_int),
        ("GetAngularPosition",   ctypes.c_int),
    ]:
        getattr(api, fn).restype = restype
    return api


def ok(result):
    return result == NO_ERROR_KINOVA


def connect():
    print("Loading Kinova USB API...")
    api = load_api()
    r = api.InitAPI()
    if not ok(r):
        print(f"ERROR: InitAPI() = {r}")
        if r == 2002:
            print("  Run:  sudo ./install_udev.sh   (in ~/working_dir/kinovaDrivers)")
        sys.exit(1)

    api.RefresDevicesList()
    devices = (KinovaDevice * MAX_KINOVA_DEVICE)()
    err = ctypes.c_int(NO_ERROR_KINOVA)
    n = api.GetDevices(devices, ctypes.byref(err))
    if n == 0:
        print("ERROR: No arm found.")
        api.CloseAPI()
        sys.exit(1)
    api.SetActiveDevice(devices[0])

    # Cold-boot kick-start (from kinova_comm.cpp)
    api.StartControlAPI()
    api.StopControlAPI()
    api.StartControlAPI()
    print(f"Connected to {devices[0].Model.decode()} (serial {devices[0].SerialNumber.decode()})")
    return api


def get_cartesian_position(api) -> np.ndarray:
    pos = CartesianPosition()
    r = api.GetCartesianPosition(ctypes.byref(pos))
    if not ok(r):
        print(f"  WARNING: GetCartesianPosition() = {r}")
    c = pos.Coordinates
    return np.array([c.X, c.Y, c.Z], dtype=np.float64)


def get_eef_xyz(api, use_joint_fk: bool) -> np.ndarray:
    """Return EEF xyz (metres).  use_joint_fk=True uses the DH chain
    (same as deploy_viz gold mesh); False uses GetCartesianPosition."""
    if not use_joint_fk:
        return get_cartesian_position(api)
    pos = AngularPosition()
    api.GetAngularPosition(ctypes.byref(pos))
    a = pos.Actuators
    q_deg = np.array([a.Actuator1, a.Actuator2, a.Actuator3,
                      a.Actuator4, a.Actuator5, a.Actuator6], dtype=np.float64)
    return joint_angles_to_eef_xyz(q_deg)


# ---------- 2D rigid-transform fit (Kabsch) ----------

def kabsch_2d(P: np.ndarray, Q: np.ndarray):
    """Find R(2x2) (proper rotation), t(2,) such that R @ P[i] + t ~= Q[i]."""
    P_mean, Q_mean = P.mean(0), Q.mean(0)
    Pc, Qc = P - P_mean, Q - Q_mean
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, d])
    R = Vt.T @ D @ U.T
    t = Q_mean - R @ P_mean
    return R, t


# ---------- Visual guidance (RealSense + reprojection) ----------

def _project(pts_3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    h = (K @ pts_3d.T).T
    return (h[:, :2] / h[:, 2:3]).astype(int)


def draw_targets(img_bgr, pts_2d, current_idx, names):
    vis = img_bgr.copy()
    for i, (px, py) in enumerate(pts_2d):
        if i == current_idx:
            color = (0, 255, 255)  # yellow — current target
            cv2.drawMarker(vis, (px, py), color, cv2.MARKER_CROSS, 30, 3)
            cv2.circle(vis, (px, py), 14, color, 2)
            cv2.putText(vis, f'TOUCH {names[i]}', (px + 18, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        else:
            color = (120, 120, 120)
            cv2.circle(vis, (px, py), 6, color, -1)
            cv2.putText(vis, names[i], (px + 10, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(vis, "SPACE = record point   |   Q = abort",
                (16, vis.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def collect_points_visual(api, task_pts_world, track_cam=0, task_frame=None,
                          use_joint_fk=False):
    """RealSense live view with reprojected targets. Returns base_pts or None on abort.

    Always opens the camera feed. Target crosshairs are projected onto the image
    if task_frame (cam_extrinsics.npy) exists. If it doesn't exist yet, the feed
    still shows so the user can see the workspace — they just press SPACE without
    a visual guide.
    """
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("pyrealsense2 not found — falling back to text-only mode.")
        return None

    ctx = rs.context()
    devices = list(ctx.devices)
    if not devices:
        print("No RealSense camera detected — falling back to text-only mode.")
        return None
    if track_cam >= len(devices):
        print(f"--track_cam {track_cam} requested but only {len(devices)} camera(s) connected "
              f"— falling back to text-only mode.")
        return None

    serial = devices[track_cam].get_info(rs.camera_info.serial_number)
    print(f"Visual guidance using camera {track_cam} (serial {serial})")

    # Load extrinsics for target projection — optional; still show feed without them
    have_extrinsics = task_frame and os.path.exists(task_frame)
    if have_extrinsics:
        tf_world2cam = np.load(task_frame).astype(np.float64)
        print(f"  Loaded extrinsics: {task_frame}")
    else:
        print(f"  WARNING: {task_frame} not found — showing camera feed without target overlay.")
        print(f"  Run: python 00_calibrate.py --track_cam {track_cam}  to generate it first.")
        tf_world2cam = None

    pipe = rs.pipeline()
    cfg  = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float64)

    if have_extrinsics:
        pts_cam = (tf_world2cam[:3, :3] @ task_pts_world.T).T + tf_world2cam[:3, 3]
        pts_2d  = _project(pts_cam, K)
    else:
        pts_2d = None
    names = [f'P{i}' for i in range(len(task_pts_world))]

    cv2.namedWindow("06_calibrate_robot — touch each target", cv2.WINDOW_NORMAL)

    base_pts = []
    try:
        idx = 0
        while idx < len(task_pts_world):
            fs  = pipe.wait_for_frames(timeout_ms=3000)
            img = np.asanyarray(fs.get_color_frame().get_data())
            if pts_2d is not None:
                vis = draw_targets(img, pts_2d, idx, names)
            else:
                vis = img.copy()
                cv2.putText(vis, f"Touch {names[idx]} then press SPACE",
                            (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(vis, "SPACE = record point   |   Q = abort",
                            (16, vis.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2)
            cv2.imshow("06_calibrate_robot — touch each target", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                base_xyz = get_eef_xyz(api, use_joint_fk)
                fk_label = "joint-FK" if use_joint_fk else "GCP"
                print(f"  P{idx} [{fk_label}]: recorded base-frame xyz = "
                      f"({base_xyz[0]*100:.1f}, {base_xyz[1]*100:.1f}, {base_xyz[2]*100:.1f}) cm")
                base_pts.append(base_xyz)
                idx += 1
            elif key == ord('q'):
                print("Aborted.")
                return None
    finally:
        pipe.stop()
        cv2.destroyAllWindows()

    return np.array(base_pts)


def _show_robot_base_in_camera(T_base_task, task_frame, track_cam):
    """
    Post-calibration sanity check: open the camera feed and overlay the robot
    base frame origin + XYZ axes projected into the image.

    Chain: robot base origin (0,0,0) in base frame
        → task frame:  inv(T_base_task)
        → camera frame: tf_world2cam @ point_task
        → pixels: K @ point_cam / z
    Press Q to close.
    """
    try:
        import pyrealsense2 as rs
    except ImportError:
        return

    tf_world2cam = np.load(task_frame).astype(np.float64)
    T_task_base  = np.linalg.inv(T_base_task)

    # Robot base frame origin and axis tips (in base frame, metres)
    axis_len = 0.10  # 10 cm arrows
    pts_base = np.array([
        [0, 0, 0],           # origin
        [axis_len, 0, 0],    # +X
        [0, axis_len, 0],    # +Y
        [0, 0, axis_len],    # +Z
    ], dtype=np.float64)

    # Transform to task frame then camera frame
    pts_task = (T_task_base[:3, :3] @ pts_base.T).T + T_task_base[:3, 3]
    pts_cam  = (tf_world2cam[:3, :3] @ pts_task.T).T + tf_world2cam[:3, 3]

    ctx = rs.context()
    devices = list(ctx.devices)
    if not devices or track_cam >= len(devices):
        return
    serial = devices[track_cam].get_info(rs.camera_info.serial_number)

    pipe = rs.pipeline()
    cfg  = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float64)

    # Project all points to pixels
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

    for _ in range(90):   # warmup
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

            cv2.putText(vis, "Q = close", (16, vis.shape[0] - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow("Robot base in camera — press Q to close", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='data/robot_extrinsics.npy',
                    help='Output .npy path for T_base_task (4x4)')
    p.add_argument('--task_frame', default=None,
                    help='Path to tf_world2cam from 00_calibrate.py. Defaults to '
                         'data/cam_extrinsics.npy for cam0, data/cam1_extrinsics.npy for cam1.')
    p.add_argument('--track_cam', type=int, default=0,
                    help='Camera index to use for the live preview (default: 0). '
                         'Match this to --track_cam used in 00_calibrate.py.')
    p.add_argument('--no_visual', action='store_true',
                    help='Skip the live camera overlay and use the text-only Enter-key procedure')
    p.add_argument('--use_joint_fk', action='store_true',
                    help='Record EEF positions using joint-angle FK (same DH chain as deploy_viz '
                         'and 07_deploy.py) instead of GetCartesianPosition.  Use this when the '
                         'gold spoon mesh in deploy_viz matches the physical arm but the cyan '
                         'circle (from GetCartesianPosition) does not.  Save to a separate file '
                         '(e.g. data/robot_extrinsics_jfk.npy) and pass it to 07_deploy.py via '
                         '--robot_extrinsics_proprio.')
    args = p.parse_args()

    if args.task_frame is None:
        args.task_frame = ('data/cam_extrinsics.npy' if args.track_cam == 0
                           else f'data/cam{args.track_cam}_extrinsics.npy')

    api = connect()

    task_pts = np.array([_TF_WORLD2BOARD[:3, :3] @ np.array([bx, by, bz])
                          for (bx, by, bz) in BOARD_POINTS.values()])
    names = list(BOARD_POINTS.keys())

    fk_mode = "joint-angle FK" if args.use_joint_fk else "GetCartesianPosition"
    base_pts = None
    if not args.no_visual:
        print("\n" + "=" * 70)
        print(f"Robot base <-> task frame calibration (visual guidance, {fk_mode})")
        print("A live camera window will show each target point — jog the arm")
        print("until the tip is on the yellow crosshair, then press SPACE.")
        print("=" * 70 + "\n")
        base_pts = collect_points_visual(api, task_pts,
                                         track_cam=args.track_cam,
                                         task_frame=args.task_frame,
                                         use_joint_fk=args.use_joint_fk)

    if base_pts is None:
        print("\n" + "=" * 70)
        print(f"Robot base <-> task frame calibration (text-only, {fk_mode})")
        print("Open data/cam_extrinsics_vis.jpg to see the task-frame axes drawn")
        print("on the ChArUco board (red=X, green=Y, origin = axes origin).")
        print("Attach a pointed tip to the gripper. For each point below, jog")
        print("the arm until the tip touches that point, then press Enter.")
        print("=" * 70 + "\n")

        base_pts = []
        for name, p_world in zip(names, task_pts):
            input(f"Touch {name}  ->  task frame xyz = "
                  f"({p_world[0]*100:.1f}, {p_world[1]*100:.1f}, {p_world[2]*100:.1f}) cm. "
                  f"Press Enter when in position...")
            base_xyz = get_eef_xyz(api, args.use_joint_fk)
            print(f"  [{fk_mode}] Recorded base-frame xyz = "
                  f"({base_xyz[0]*100:.1f}, {base_xyz[1]*100:.1f}, {base_xyz[2]*100:.1f}) cm\n")
            base_pts.append(base_xyz)
        base_pts = np.array(base_pts)

    api.CloseAPI()

    # 2D fit (XY) under the gravity-aligned assumption; Z is a constant offset
    R2, t2 = kabsch_2d(task_pts[:, :2], base_pts[:, :2])
    z_offset = float(np.mean(base_pts[:, 2] - task_pts[:, 2]))

    T_base_task = np.eye(4, dtype=np.float64)
    T_base_task[:2, :2] = R2
    T_base_task[:3, 3]  = [t2[0], t2[1], z_offset]

    # Residuals
    pred_base_xy = (R2 @ task_pts[:, :2].T).T + t2
    residuals = np.linalg.norm(pred_base_xy - base_pts[:, :2], axis=1) * 1000  # mm
    print(f"XY fit residuals (mm): {residuals.round(1)}  (mean {residuals.mean():.1f})")

    yaw_deg = np.degrees(np.arctan2(R2[1, 0], R2[0, 0]))
    print(f"\nT_base_task:\n{T_base_task.round(4)}")
    print(f"  yaw (task→base, about Z) = {yaw_deg:.1f} deg")
    print(f"  translation = ({T_base_task[0,3]*100:.1f}, "
          f"{T_base_task[1,3]*100:.1f}, {T_base_task[2,3]*100:.1f}) cm")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.save(args.output, T_base_task)
    print(f"\nSaved → {args.output}")

    print(f"\nDeployment formula:")
    print(f"  T_base_eef = T_base_task @ T_task_tool @ T_tool_eef")
    print(f"  T_base_task = np.load('{args.output}')")

    # Live post-calibration check: project robot base frame onto the camera feed.
    # The robot base origin + XYZ axes are transformed to camera frame and drawn.
    # Press Q to close.
    if not args.no_visual and os.path.exists(args.task_frame):
        _show_robot_base_in_camera(T_base_task, args.task_frame,
                                   args.track_cam)


if __name__ == '__main__':
    main()
