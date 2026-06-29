"""
Diagnostic: compare FoundationPose tool tracking vs. Kinova FK.

For each frame:
  - FoundationPose tracks the spoon/paddle in the camera frame, converts to
    robot-base frame via the camera calibration chain:
        T_base_tool_fp = T_base_task @ inv(tf_world2cam) @ T_cam_tool

  - Kinova FK gives the EEF pose; combined with the cached T_tool_eef offset:
        T_base_tool_fk = T_base_eef @ inv(T_tool_eef)

The difference between the two reveals whether T_tool_eef (or the camera
calibration) is off, and by how much.

Usage:
    python check_tool_eef_error.py --mesh spoon.obj --tool_prompt "spoon"
    python check_tool_eef_error.py --mesh table_tennis_paddle.obj \\
                                   --tool_prompt "table tennis paddle"

Press Q to quit.  Results are also saved to --output_csv.
"""

import os, sys, time, ctypes, argparse, csv
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import multiprocessing as mp

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_DIR = os.path.join(os.path.expanduser('~/working_dir/kinovaDrivers'), 'sdk', 'lib')

if _LIB_DIR not in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
    os.environ["LD_LIBRARY_PATH"] = _LIB_DIR + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.execv(sys.executable, [sys.executable] + sys.argv)

THIRD_PARTY   = os.path.join(PIPELINE_DIR, '..', 'Tool_as_Interface', 'third_party')
FP_DIR        = os.path.join(THIRD_PARTY, 'FoundationPose')
GDINO_WEIGHTS = os.path.join(PIPELINE_DIR, 'checkpoints', 'groundingdino_swint_ogc.pth')
SAM_WEIGHTS   = os.path.join(PIPELINE_DIR, 'checkpoints', 'sam_vit_h_4b8939.pth')
sys.path.insert(0, FP_DIR)
sys.path.insert(0, THIRD_PARTY)


# ── Kinova SDK (copy from 07_deploy.py) ──────────────────────────────────────

NO_ERROR_KINOVA    = 1
SERIAL_LENGTH      = 20
MAX_KINOVA_DEVICE  = 20
CARTESIAN_POSITION = 1
HAND_NOMOVEMENT    = 0

LIB_PATH      = os.path.join(_LIB_DIR, "USBCommandLayerUbuntu.so")
COMM_LIB_PATH = os.path.join(_LIB_DIR, "USBCommLayerUbuntu.so")

class KinovaDevice(ctypes.Structure):
    _fields_ = [("SerialNumber", ctypes.c_char * SERIAL_LENGTH),
                ("Model",        ctypes.c_char * SERIAL_LENGTH),
                ("VersionMajor", ctypes.c_int), ("VersionMinor",   ctypes.c_int),
                ("VersionRelease", ctypes.c_int), ("DeviceType",   ctypes.c_int),
                ("DeviceID",     ctypes.c_int)]

class CartesianInfo(ctypes.Structure):
    _fields_ = [("X", ctypes.c_float), ("Y", ctypes.c_float), ("Z", ctypes.c_float),
                ("ThetaX", ctypes.c_float), ("ThetaY", ctypes.c_float), ("ThetaZ", ctypes.c_float)]

class FingersPosition(ctypes.Structure):
    _fields_ = [("Finger1", ctypes.c_float), ("Finger2", ctypes.c_float), ("Finger3", ctypes.c_float)]

class CartesianPosition(ctypes.Structure):
    _fields_ = [("Coordinates", CartesianInfo), ("Fingers", FingersPosition)]


def load_api():
    ctypes.CDLL(COMM_LIB_PATH, mode=ctypes.RTLD_GLOBAL)
    api = ctypes.CDLL(LIB_PATH)
    for fn, rt in [("InitAPI", ctypes.c_int), ("CloseAPI", ctypes.c_int),
                   ("RefresDevicesList", ctypes.c_int), ("GetDevices", ctypes.c_int),
                   ("SetActiveDevice", ctypes.c_int), ("StartControlAPI", ctypes.c_int),
                   ("StopControlAPI", ctypes.c_int), ("GetCartesianPosition", ctypes.c_int),
                   ("SetCartesianControl", ctypes.c_int)]:
        getattr(api, fn).restype = rt
    return api


def connect_robot():
    api = load_api()
    if api.InitAPI() != NO_ERROR_KINOVA:
        raise RuntimeError("Kinova InitAPI failed")
    api.RefresDevicesList()
    devices = (KinovaDevice * MAX_KINOVA_DEVICE)()
    err = ctypes.c_int(NO_ERROR_KINOVA)
    n = api.GetDevices(devices, ctypes.byref(err))
    if n == 0:
        raise RuntimeError("No Kinova arm found")
    api.SetActiveDevice(devices[0])
    api.StartControlAPI()
    api.SetCartesianControl()
    print(f"Connected: {devices[0].Model.decode()} (serial {devices[0].SerialNumber.decode()})")
    return api


_euler_conv = 'xyz'   # overridden from args in main()

def get_eef_pose(api, verbose=False) -> np.ndarray:
    """Returns (4,4) T_base_eef."""
    pos = CartesianPosition()
    api.GetCartesianPosition(ctypes.byref(pos))
    c = pos.Coordinates
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler(_euler_conv, [c.ThetaX, c.ThetaY, c.ThetaZ]).as_matrix()
    T[:3,  3] = [c.X, c.Y, c.Z]
    if verbose:
        rv = Rotation.from_matrix(T[:3, :3]).as_rotvec()
        ang = np.degrees(np.linalg.norm(rv))
        print(f"  [FK raw] Theta=({c.ThetaX:.3f},{c.ThetaY:.3f},{c.ThetaZ:.3f}) rad "
              f"| conv={_euler_conv} | rotvec_angle={ang:.1f}°")
    return T


# ── Camera ───────────────────────────────────────────────────────────────────

def start_realsense(camera_index=0):
    import pyrealsense2 as rs
    ctx     = rs.context()
    devices = ctx.query_devices()
    serial  = devices[camera_index].get_info(rs.camera_info.serial_number)
    pipe    = rs.pipeline()
    cfg     = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8,  30)
    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16,   30)
    profile = pipe.start(cfg)
    intr    = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float64)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    print(f"Camera {camera_index}: serial {serial} — warming up (60 frames)...")
    for _ in range(60):
        pipe.wait_for_frames()
    return pipe, align, K, depth_scale


def capture(pipe, align, depth_scale):
    import pyrealsense2 as rs
    frames    = align.process(pipe.wait_for_frames(timeout_ms=3000))
    rgb       = np.asanyarray(frames.get_color_frame().get_data())
    depth_raw = np.asanyarray(frames.get_depth_frame().get_data())
    return rgb, depth_raw.astype(np.float32) * depth_scale


# ── GroundedSAM segmentation ─────────────────────────────────────────────────

def load_gdino_sam(device):
    import groundingdino
    gdino_config = os.path.join(os.path.dirname(groundingdino.__file__),
                                'config', 'GroundingDINO_SwinT_OGC.py')
    from groundingdino.util.inference import load_model as load_gdino
    from segment_anything import sam_model_registry, SamPredictor
    print("Loading GroundingDINO...")
    gdino = load_gdino(gdino_config, GDINO_WEIGHTS).to(device).eval()
    print("Loading SAM...")
    sam = sam_model_registry['vit_h'](checkpoint=SAM_WEIGHTS).to(device)
    return gdino, SamPredictor(sam)


def segment_tool(gdino, sam_pred, img_rgb, prompt, box_thr, text_thr, device):
    import torch
    import torchvision.transforms as T
    from groundingdino.util.inference import predict
    from PIL import Image
    H, W = img_rgb.shape[:2]
    transform = T.Compose([T.Resize(800), T.ToTensor(),
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img_t = transform(Image.fromarray(img_rgb)).to(device)
    with torch.no_grad():
        boxes, logits, _ = predict(gdino, img_t, prompt, box_thr, text_thr, device=device)
    if boxes is None or len(boxes) == 0:
        return np.zeros((H, W), dtype=np.uint8)
    boxes_px = boxes.clone()
    boxes_px[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * W
    boxes_px[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * H
    boxes_px[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * W
    boxes_px[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * H
    sam_pred.set_image(img_rgb)
    mask = np.zeros((H, W), dtype=bool)
    for box in boxes_px:
        m, _, _ = sam_pred.predict(box=box.cpu().numpy(), multimask_output=False)
        mask |= m[0].astype(bool)
    return mask.astype(np.uint8)


# ── Visualization helpers ─────────────────────────────────────────────────────

def _project_pt(pt3d, K):
    """Project a single 3D point (x,y,z) to (u,v) pixel using K."""
    u = int(K[0, 0] * pt3d[0] / pt3d[2] + K[0, 2])
    v = int(K[1, 1] * pt3d[1] / pt3d[2] + K[1, 2])
    return (u, v)


def draw_bbox3d(img, T_cam_obj, mesh_bounds, K, color=(0, 255, 255), thickness=1):
    """
    Draw the 3D bounding box of the mesh projected into the image.
    mesh_bounds: (2,3) array from mesh.bounds — [[xmin,ymin,zmin],[xmax,ymax,zmax]]
    """
    lo, hi = mesh_bounds
    # 8 corners of the axis-aligned bounding box in object frame
    corners = np.array([[x, y, z]
                        for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1])
                        for z in (lo[2], hi[2])], dtype=np.float64)
    # Transform to camera frame
    R, t = T_cam_obj[:3, :3], T_cam_obj[:3, 3]
    corners_cam = (R @ corners.T).T + t

    # Only draw if all corners are in front of camera
    if np.any(corners_cam[:, 2] <= 0.01):
        return img

    px = np.array([_project_pt(c, K) for c in corners_cam])

    # 12 edges of the box: connect corners that differ in exactly one bit
    for i in range(8):
        for j in range(i + 1, 8):
            if bin(i ^ j).count('1') == 1:   # adjacent corners
                cv2.line(img, tuple(px[i]), tuple(px[j]), color, thickness, cv2.LINE_AA)
    return img


def draw_axes(img, T_cam, K, length=0.07, thickness=2,
              x_color=(0,0,255), y_color=(0,255,0), z_color=(255,0,0),
              label=None, label_color=(255,255,255), filled_tips=True):
    """
    Draw XYZ axes of a 4×4 camera-frame pose onto img (BGR).
    filled_tips=True  → solid circles at tips  (FP style)
    filled_tips=False → hollow circles at tips (FK style)
    """
    origin = T_cam[:3, 3]
    if origin[2] <= 0.01:
        return img

    o_px = _project_pt(origin, K)
    tip_r = max(3, thickness)

    for axis_idx, color in enumerate([x_color, y_color, z_color]):
        tip = origin + T_cam[:3, axis_idx] * length
        if tip[2] <= 0.01:
            continue
        t_px = _project_pt(tip, K)
        cv2.line(img, o_px, t_px, color, thickness, cv2.LINE_AA)
        fill = -1 if filled_tips else 1
        cv2.circle(img, t_px, tip_r, color, fill, cv2.LINE_AA)

    # Origin marker: filled for FP, hollow for FK
    origin_fill = -1 if filled_tips else 1
    cv2.circle(img, o_px, 5, (255, 255, 255), origin_fill, cv2.LINE_AA)

    if label:
        cv2.putText(img, label, (o_px[0] + 7, o_px[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(img, label, (o_px[0] + 7, o_px[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)
    return img


# ── Pose error utilities ──────────────────────────────────────────────────────

def pose_error(T_a: np.ndarray, T_b: np.ndarray):
    """
    Returns (trans_err_m, rot_err_deg) between two 4x4 poses.
    trans_err = |t_a - t_b|
    rot_err   = angle of R_a^T @ R_b
    """
    trans_err = np.linalg.norm(T_a[:3, 3] - T_b[:3, 3])
    R_rel     = T_a[:3, :3].T @ T_b[:3, :3]
    rot_err   = np.degrees(Rotation.from_matrix(R_rel).magnitude())
    return trans_err, rot_err


def _plot3d_worker(queue: mp.Queue, display: str = ':1'):
    """
    Runs in a separate process — owns the interactive matplotlib window.
    Receives (R_fp, R_fk, t_err_cm, r_err_deg) tuples from the queue and
    redraws the 3D axes plot.  The window is fully mouse-interactable.
    """
    import os as _os
    _os.environ.setdefault('DISPLAY', display)
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    L      = 0.08
    COLORS = ['#ff4444', '#44ff44', '#4488ff']
    LABELS = ['X', 'Y', 'Z']

    plt.ion()
    fig = plt.figure('3D Pose Comparison — drag to rotate', figsize=(6, 6))
    ax  = fig.add_subplot(111, projection='3d')
    plt.show(block=False)

    T_fp = T_fk = np.eye(4)
    t_err_cm = r_err_deg = 0.0

    while True:
        # Drain the queue, keep only the latest update
        latest = None
        try:
            while True:
                latest = queue.get_nowait()
        except Exception:
            pass

        if latest is not None:
            T_fp, T_fk, t_err_cm, r_err_deg = latest

        ax.cla()
        ax.set_facecolor('#1a1a1a')
        fig.patch.set_facecolor('#1a1a1a')

        fp_o = T_fp[:3, 3]
        fk_o = T_fk[:3, 3]

        for col_idx, (color, lbl) in enumerate(zip(COLORS, LABELS)):
            fp_tip = fp_o + T_fp[:3, col_idx] * L
            ax.plot([fp_o[0], fp_tip[0]], [fp_o[1], fp_tip[1]], [fp_o[2], fp_tip[2]],
                    color=color, linewidth=3, linestyle='-')
            ax.text(*fp_tip, f'FP-{lbl}', color=color, fontsize=8)

            fk_tip = fk_o + T_fk[:3, col_idx] * L
            ax.plot([fk_o[0], fk_tip[0]], [fk_o[1], fk_tip[1]], [fk_o[2], fk_tip[2]],
                    color=color, linewidth=1.5, linestyle='--')
            ax.text(*fk_tip, f'FK-{lbl}', color=color, fontsize=8)

        # Draw a line connecting the two origins to show translation error
        ax.plot([fp_o[0], fk_o[0]], [fp_o[1], fk_o[1]], [fp_o[2], fk_o[2]],
                color='white', linewidth=1, linestyle=':', alpha=0.6)
        ax.scatter(*fp_o, color='white', s=60, zorder=5)
        ax.scatter(*fk_o, color='gray', s=60, zorder=5, marker='^')

        all_pts = np.stack([fp_o, fk_o])
        mid = all_pts.mean(axis=0)
        span = max(np.linalg.norm(fp_o - fk_o) / 2 + L * 1.5, L * 2)
        ax.set_xlim(mid[0]-span, mid[0]+span)
        ax.set_ylim(mid[1]-span, mid[1]+span)
        ax.set_zlim(mid[2]-span, mid[2]+span)
        ax.set_xlabel('X', color='#ff4444'); ax.set_ylabel('Y', color='#44ff44')
        ax.set_zlabel('Z', color='#4488ff')
        ax.tick_params(colors='#888888', labelsize=6)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False; pane.set_edgecolor('#333333')
        ax.grid(True, color='#333333', linewidth=0.5)
        ax.set_title(f'FP (solid)  vs  FK (dashed)\n'
                     f'trans {t_err_cm:.1f} cm   rot {r_err_deg:.1f}°',
                     color='white', fontsize=10)

        fig.canvas.draw_idle()
        plt.pause(0.05)

        if not plt.get_fignums():   # window was closed
            break


def draw_text(img, lines, start_y=25, color=(0, 255, 0)):
    for i, line in enumerate(lines):
        cv2.putText(img, line, (10, start_y + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(img, line, (10, start_y + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mesh',         required=True, help='Tool mesh (.obj)')
    p.add_argument('--tool_prompt',  required=True, help='GroundedSAM prompt for the tool')
    p.add_argument('--task_frame',   default='data/cam_extrinsics.npy')
    p.add_argument('--robot_extrinsics', default='data/robot_extrinsics.npy')
    p.add_argument('--tool_eef_cache',   default='data/T_tool_eef.npy')
    p.add_argument('--camera',       type=int,   default=0)
    p.add_argument('--box_threshold',type=float, default=0.3)
    p.add_argument('--text_threshold',type=float,default=0.25)
    p.add_argument('--est_refine_iter',  type=int, default=5)
    p.add_argument('--track_refine_iter',type=int, default=2)
    p.add_argument('--output_csv',   default='/tmp/tool_eef_error.csv')
    p.add_argument('--device',       default='cuda')
    p.add_argument('--euler_conv',   default='XYZ',
                   help='Euler convention for Kinova angles: XYZ (default, intrinsic), xyz, ZYX, zyx')
    p.add_argument('--diag',         action='store_true',
                   help='Print raw ThetaX/Y/Z and rotvec every frame')
    return p.parse_args()


def main():
    global _euler_conv
    args   = parse_args()
    device = args.device
    _euler_conv = args.euler_conv
    print(f"Euler convention: {_euler_conv}")

    # ── Load calibration ────────────────────────────────────────────────────
    tf_world2cam = np.load(args.task_frame).astype(np.float64)
    T_base_task  = np.load(args.robot_extrinsics).astype(np.float64)
    T_tool_eef   = np.load(args.tool_eef_cache).astype(np.float64)
    tf_cam2world = np.linalg.inv(tf_world2cam)
    print(f"Loaded calibration files.")
    print(f"  T_tool_eef:\n{T_tool_eef.round(4)}")

    # ── cv2 window FIRST (before any CUDA model loads) ───────────────────────
    cv2.namedWindow("Tool EEF Error", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tool EEF Error", 848, 480)

    # ── Load GroundedSAM (before FoundationPose/nvdiffrast) ──────────────────
    gdino, sam_pred = load_gdino_sam(device)

    # ── Load FoundationPose ──────────────────────────────────────────────────
    import trimesh
    loaded = trimesh.load(args.mesh)
    mesh = (trimesh.util.concatenate(list(loaded.geometry.values()))
            if isinstance(loaded, trimesh.Scene) else loaded)
    if mesh.bounding_box.extents.max() > 0.5:
        mesh.apply_scale(0.01)
        print("Mesh rescaled x0.01 (cm → m)")
    # OBB centering — same as live_foundationpose.py
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    print("Loading FoundationPose...")
    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
    import nvdiffrast.torch as dr
    from Utils import draw_xyz_axis, draw_posed_3d_box
    scorer  = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx   = dr.RasterizeCudaContext()
    os.makedirs('/tmp/fp_eef_debug', exist_ok=True)
    est = FoundationPose(
        model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
        scorer=scorer, refiner=refiner, glctx=glctx,
        debug_dir='/tmp/fp_eef_debug', debug=0,
    )
    print("FoundationPose ready.")

    # ── Start camera + robot ─────────────────────────────────────────────────
    pipe, align, K, depth_scale = start_realsense(args.camera)
    api = connect_robot()

    # ── Initial registration ─────────────────────────────────────────────────
    def register_tool(rgb, depth):
        print(f"\nDetecting tool ('{args.tool_prompt}')...")
        mask = segment_tool(gdino, sam_pred, rgb, args.tool_prompt,
                            args.box_threshold, args.text_threshold, device)
        if mask.sum() < 100:
            print("  Tool not detected — reposition and press SPACE.")
            return None
        pose = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask,
                            iteration=args.est_refine_iter)
        print("  Registered.")
        return pose

    print(f"\nRegistering tool — press SPACE to re-register, Q to quit.\n")
    rgb, depth = capture(pipe, align, depth_scale)
    pose_cam = register_tool(rgb, depth)
    initialized = pose_cam is not None

    # ── Launch interactive 3D plot in a subprocess ───────────────────────────
    display = os.environ.get('DISPLAY', ':1')
    plot_queue = mp.Queue(maxsize=1)
    plot_proc  = mp.Process(target=_plot3d_worker, args=(plot_queue, display), daemon=True)
    plot_proc.start()

    # ── CSV log ──────────────────────────────────────────────────────────────
    csv_file = open(args.output_csv, 'w', newline='')
    writer   = csv.writer(csv_file)
    writer.writerow([
        'step',
        'fp_x', 'fp_y', 'fp_z',
        'fk_x', 'fk_y', 'fk_z',
        'trans_err_m', 'trans_err_cm', 'rot_err_deg',
    ])

    step = 0
    trans_errors, rot_errors = [], []
    calibration_flash = 0   # frames to show "Calibrated!" banner

    try:
        while True:
            rgb, depth = capture(pipe, align, depth_scale)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord(' '):
                pose_cam = register_tool(rgb, depth)
                initialized = pose_cam is not None
                step = 0
                trans_errors.clear(); rot_errors.clear()
                continue
            if key == ord('c') and initialized:
                # Capture T_tool_eef from live FP tracking + robot FK
                _T_base_eef_cal = get_eef_pose(api)
                _T_cam_tool_cal = pose_cam.astype(np.float64)
                _T_base_tool_cal = T_base_task @ tf_cam2world @ _T_cam_tool_cal
                T_tool_eef = np.linalg.inv(_T_base_tool_cal) @ _T_base_eef_cal
                np.save(args.tool_eef_cache, T_tool_eef)
                print(f"\n[C] Calibrated T_tool_eef saved → {args.tool_eef_cache}")
                print(f"    T_tool_eef:\n{T_tool_eef.round(4)}\n")
                step = 0
                trans_errors.clear(); rot_errors.clear()
                calibration_flash = 60   # show banner for ~60 frames
                continue

            if not initialized:
                vis_bgr = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
                cv2.putText(vis_bgr, "Registration failed — press SPACE to retry",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
                cv2.imshow("Tool EEF Error", vis_bgr)
                continue

            # ── FoundationPose tracking ──────────────────────────────────────
            pose_cam = est.track_one(rgb=rgb, depth=depth, K=K,
                                      iteration=args.track_refine_iter)
            T_cam_tool  = pose_cam.astype(np.float64)
            T_base_tool_fp = T_base_task @ tf_cam2world @ T_cam_tool

            # ── Robot FK ────────────────────────────────────────────────────
            T_base_eef     = get_eef_pose(api, verbose=args.diag)
            T_base_tool_fk = T_base_eef @ np.linalg.inv(T_tool_eef)

            # ── Error ────────────────────────────────────────────────────────
            t_err, r_err = pose_error(T_base_tool_fp, T_base_tool_fk)
            trans_errors.append(t_err)
            rot_errors.append(r_err)

            fp_xyz = T_base_tool_fp[:3, 3] * 100   # cm
            fk_xyz = T_base_tool_fk[:3, 3] * 100

            writer.writerow([step,
                             *fp_xyz.round(3), *fk_xyz.round(3),
                             round(t_err, 5), round(t_err * 100, 3), round(r_err, 3)])
            csv_file.flush()

            eef_xyz = T_base_eef[:3, 3] * 100
            print(f"step {step:4d} | "
                  f"EEF ({eef_xyz[0]:+6.1f},{eef_xyz[1]:+6.1f},{eef_xyz[2]:+6.1f}) cm | "
                  f"FP  ({fp_xyz[0]:+6.1f},{fp_xyz[1]:+6.1f},{fp_xyz[2]:+6.1f}) cm | "
                  f"FK  ({fk_xyz[0]:+6.1f},{fk_xyz[1]:+6.1f},{fk_xyz[2]:+6.1f}) cm | "
                  f"err  trans={t_err*100:.2f} cm  rot={r_err:.1f}°")

            # ── Visualization ────────────────────────────────────────────────
            # Shared axis colours: X=red, Y=green, Z=blue for BOTH poses.
            # FP  — thick lines (4px) + filled tip circles  → solid appearance
            # FK  — thin  lines (2px) + hollow tip circles  → dashed appearance
            AX = [(0,0,255), (0,255,0), (255,0,0)]   # BGR: X, Y, Z

            center_pose = T_cam_tool @ np.linalg.inv(to_origin)
            vis_bgr = draw_posed_3d_box(K, img=cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR),
                                        ob_in_cam=center_pose, bbox=bbox)
            draw_axes(vis_bgr, center_pose, K, length=0.08, thickness=4,
                      x_color=AX[0], y_color=AX[1], z_color=AX[2],
                      label="FP", label_color=(255,255,255))

            T_cam_tool_fk = tf_world2cam @ np.linalg.inv(T_base_task) @ T_base_tool_fk
            draw_axes(vis_bgr, T_cam_tool_fk, K, length=0.08, thickness=2,
                      x_color=AX[0], y_color=AX[1], z_color=AX[2],
                      label="FK", label_color=(200,200,200), filled_tips=False)

            # Legend — same colour, solid=FP / thin=FK
            lx = vis_bgr.shape[1] - 190
            for i, (name, lbl_col, thick) in enumerate([("FP (thick)", (255,255,255), 3),
                                                         ("FK (thin)",  (180,180,180), 1)]):
                base_y = 20 + i * 60
                cv2.putText(vis_bgr, name, (lx, base_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, lbl_col, 1)
                for j, (ax_lbl, col) in enumerate([("X", AX[0]), ("Y", AX[1]), ("Z", AX[2])]):
                    y = base_y + 16 + j * 14
                    cv2.line(vis_bgr, (lx, y), (lx + 22, y), col, thick)
                    cv2.putText(vis_bgr, ax_lbl, (lx + 26, y + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

            mean_t = np.mean(trans_errors[-50:]) * 100
            mean_r = np.mean(rot_errors[-50:])
            delta_xyz = fp_xyz - fk_xyz
            err_color = (0, 200, 0) if t_err * 100 < 2.0 else (0, 100, 255)
            lines = [
                f"FP  ({fp_xyz[0]:+.1f}, {fp_xyz[1]:+.1f}, {fp_xyz[2]:+.1f}) cm",
                f"FK  ({fk_xyz[0]:+.1f}, {fk_xyz[1]:+.1f}, {fk_xyz[2]:+.1f}) cm",
                f"dXYZ({delta_xyz[0]:+.1f}, {delta_xyz[1]:+.1f}, {delta_xyz[2]:+.1f}) cm",
                f"err  {t_err*100:.2f} cm  {r_err:.1f} deg",
                f"avg  {mean_t:.2f} cm  {mean_r:.1f} deg  (last 50)",
                f"SPACE=re-register  C=calibrate T_tool_eef  Q=quit",
            ]
            draw_text(vis_bgr, lines, color=err_color)

            if calibration_flash > 0:
                cv2.putText(vis_bgr, "T_tool_eef CALIBRATED!",
                            (vis_bgr.shape[1]//2 - 180, vis_bgr.shape[0]//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
                cv2.putText(vis_bgr, "T_tool_eef CALIBRATED!",
                            (vis_bgr.shape[1]//2 - 180, vis_bgr.shape[0]//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 128), 2)
                calibration_flash -= 1

            try:
                plot_queue.put_nowait((
                    T_base_tool_fp.copy(),
                    T_base_tool_fk.copy(),
                    t_err * 100, r_err,
                ))
            except Exception:
                pass
            cv2.imshow("Tool EEF Error", vis_bgr)
            step += 1

    finally:
        api.CloseAPI()
        pipe.stop()
        cv2.destroyAllWindows()
        csv_file.close()
        plot_proc.terminate()
        plot_proc.join(timeout=2)

        if trans_errors:
            print(f"\n── Summary ({len(trans_errors)} frames) ──")
            print(f"  Translation error:  mean={np.mean(trans_errors)*100:.2f} cm  "
                  f"std={np.std(trans_errors)*100:.2f} cm  "
                  f"max={np.max(trans_errors)*100:.2f} cm")
            print(f"  Rotation error:     mean={np.mean(rot_errors):.1f}°  "
                  f"std={np.std(rot_errors):.1f}°  "
                  f"max={np.max(rot_errors):.1f}°")
            print(f"  Results saved → {args.output_csv}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
