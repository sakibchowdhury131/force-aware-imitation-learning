#!/usr/bin/env python3
"""
Calibration debug viewer — robot arm + calibration frames + live spoon.

Everything shown in robot BASE frame coordinates in one Open3D window:
  [1] Robot arm mesh  — live FK from joint angles (grey links)
  [2] Robot base frame — large RGB axes at (0,0,0)
  [3] Task/world frame — where the ChArUco board sits in robot space
  [4] Spoon from FP    — live FoundationPose tracking

If calibration is correct:
  - task frame appears at the physical ChArUco board location
  - spoon follows the physical spoon in robot-base space

Controls: SPACE = re-register spoon | Q = quit

Usage:
    python calib_viz_3d.py --mesh spoon.obj --tool_prompt "spoon"
    python calib_viz_3d.py --mesh spoon.obj --tool_prompt "spoon" --camera 1
"""

import os, sys, argparse, time, copy
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
MESH_DIR     = os.path.join(
    os.path.expanduser('~/working_dir/kinovaDrivers'),
    'kinova-ros', 'kinova_description', 'meshes',
)
_LIB_DIR    = os.path.join(os.path.expanduser('~/working_dir/kinovaDrivers'), 'sdk', 'lib')
THIRD_PARTY = os.path.join(PIPELINE_DIR, '..', 'Tool_as_Interface', 'third_party')
FP_DIR      = os.path.join(THIRD_PARTY, 'FoundationPose')
GDINO_WEIGHTS = os.path.join(PIPELINE_DIR, 'checkpoints', 'groundingdino_swint_ogc.pth')
SAM_WEIGHTS   = os.path.join(PIPELINE_DIR, 'checkpoints', 'sam_vit_h_4b8939.pth')

if _LIB_DIR not in os.environ.get('LD_LIBRARY_PATH', '').split(':'):
    os.environ['LD_LIBRARY_PATH'] = _LIB_DIR + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    os.execv(sys.executable, [sys.executable] + sys.argv)

LIB_PATH      = os.path.join(_LIB_DIR, 'USBCommandLayerUbuntu.so')
COMM_LIB_PATH = os.path.join(_LIB_DIR, 'USBCommLayerUbuntu.so')
sys.path.insert(0, FP_DIR)
sys.path.insert(0, THIRD_PARTY)


# ── Kinova SDK ─────────────────────────────────────────────────────────────────
import ctypes

class KinovaDevice(ctypes.Structure):
    _fields_ = [('SerialNumber', ctypes.c_char * 20), ('Model', ctypes.c_char * 20),
                ('VersionMajor', ctypes.c_int), ('VersionMinor', ctypes.c_int),
                ('VersionRelease', ctypes.c_int), ('DeviceType', ctypes.c_int),
                ('DeviceID', ctypes.c_int)]

class AngularInfo(ctypes.Structure):
    _fields_ = [(f'Actuator{i}', ctypes.c_float) for i in range(1, 8)]

class FingersPosition(ctypes.Structure):
    _fields_ = [('Finger1', ctypes.c_float), ('Finger2', ctypes.c_float),
                ('Finger3', ctypes.c_float)]

class AngularPosition(ctypes.Structure):
    _fields_ = [('Actuators', AngularInfo), ('Fingers', FingersPosition)]


def connect_robot():
    ctypes.CDLL(COMM_LIB_PATH, mode=ctypes.RTLD_GLOBAL)
    api = ctypes.CDLL(LIB_PATH)
    for fn in ['InitAPI', 'RefresDevicesList', 'GetDevices',
               'SetActiveDevice', 'GetAngularPosition', 'CloseAPI']:
        getattr(api, fn).restype = ctypes.c_int
    api.InitAPI()
    api.RefresDevicesList()
    devices = (KinovaDevice * 20)()
    err = ctypes.c_int(1)
    n = api.GetDevices(devices, ctypes.byref(err))
    if n == 0:
        raise RuntimeError('No Kinova device found')
    api.SetActiveDevice(devices[0])
    print(f'Connected: {devices[0].Model.decode()} ({devices[0].SerialNumber.decode()})')
    return api


def get_joint_angles_deg(api):
    pos = AngularPosition()
    api.GetAngularPosition(ctypes.byref(pos))
    a = pos.Actuators
    return np.array([a.Actuator1, a.Actuator2, a.Actuator3,
                     a.Actuator4, a.Actuator5, a.Actuator6], dtype=np.float64)


# ── FK chain (from kinova-ros j2s6s300.xacro) ─────────────────────────────────
_PI = np.pi
_JOINT_PARAMS = [
    ([0,       0,       0.15675], [0,      _PI,   0    ]),
    ([0,       0.0016, -0.11875], [-_PI/2, 0,     _PI  ]),
    ([0,      -0.410,  0       ], [0,      _PI,   0    ]),
    ([0,       0.2073, -0.0114 ], [-_PI/2, 0,     _PI  ]),
    ([0,       0,      -0.10375], [ _PI/2, 0,     _PI  ]),
    ([0,       0.10375, 0      ], [-_PI/2, 0,     _PI  ]),
]
_EEF_PARAMS = ([0, 0, -0.1600], [_PI, 0, _PI/2])

_LINK_MESHES = ['base', 'shoulder', 'arm', 'forearm',
                'wrist_spherical_1', 'wrist_spherical_2', 'hand_3finger']
_LINK_COLORS = [
    [0.35, 0.35, 0.40],
    [0.45, 0.48, 0.52],
    [0.50, 0.53, 0.57],
    [0.55, 0.58, 0.62],
    [0.60, 0.63, 0.67],
    [0.65, 0.68, 0.72],
    [0.70, 0.73, 0.77],
]


def _make_T(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    T[:3,  3] = xyz
    return T


def compute_fk(q_deg):
    q = np.deg2rad(q_deg)
    T = np.eye(4)
    frames = [T.copy()]
    for (xyz, rpy), qi in zip(_JOINT_PARAMS, q):
        Tj = np.eye(4)
        Tj[:3, :3] = Rotation.from_euler('z', qi).as_matrix()
        T = T @ _make_T(xyz, rpy) @ Tj
        frames.append(T.copy())
    frames.append(T @ _make_T(*_EEF_PARAMS))
    return frames   # 8 entries: base + 6 links + EEF


def load_robot_meshes():
    import open3d as o3d
    meshes = []
    for name in _LINK_MESHES:
        path = os.path.join(MESH_DIR, f'{name}.STL')
        if not os.path.exists(path):
            meshes.append(None)
            continue
        m = o3d.io.read_triangle_mesh(path)
        m.compute_vertex_normals()
        verts = np.asarray(m.vertices)
        if np.ptp(verts, axis=0).max() > 10:
            m.scale(0.001, center=np.zeros(3))
        meshes.append(m)
    return meshes


def build_robot_geoms(q_deg, base_meshes):
    frames = compute_fk(q_deg)
    geoms  = []
    for link_idx, (tmpl, color) in enumerate(zip(base_meshes, _LINK_COLORS)):
        if tmpl is None:
            continue
        m = copy.deepcopy(tmpl)
        m.transform(frames[link_idx])
        m.paint_uniform_color(color)
        geoms.append(m)
    return geoms


# ── Camera-image FP overlay ───────────────────────────────────────────────────

def draw_fp_axes(img_bgr, T_cam_tool, K, length=0.08):
    """Draw X/Y/Z axes of T_cam_tool on the camera image."""
    o = T_cam_tool[:3, 3]
    if o[2] <= 0.01:
        return img_bgr

    def proj(pt3):
        u = int(K[0,0] * pt3[0] / pt3[2] + K[0,2])
        v = int(K[1,1] * pt3[1] / pt3[2] + K[1,2])
        return (u, v)

    o_px = proj(o)
    colors_bgr = [(0,0,255), (0,255,0), (255,0,0)]   # X=red, Y=green, Z=blue
    for i, col in enumerate(colors_bgr):
        tip = o + T_cam_tool[:3, i] * length
        if tip[2] > 0.01:
            cv2.line(img_bgr, o_px, proj(tip), col, 2, cv2.LINE_AA)
            cv2.circle(img_bgr, proj(tip), 4, col, -1, cv2.LINE_AA)
    cv2.circle(img_bgr, o_px, 5, (255, 255, 255), -1, cv2.LINE_AA)
    return img_bgr


# ── Open3D helpers ─────────────────────────────────────────────────────────────

def make_frame(T, size=0.08):
    import open3d as o3d
    o   = T[:3, 3]
    pts = [o, o + T[:3,0]*size, o + T[:3,1]*size, o + T[:3,2]*size]
    ls  = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector([[0,1],[0,2],[0,3]]),
    )
    ls.colors = o3d.utility.Vector3dVector([[1,0,0],[0,1,0],[0,0,1]])
    return ls


def make_sphere(center, r=0.015, color=(1,1,0)):
    import open3d as o3d
    s = o3d.geometry.TriangleMesh.create_sphere(radius=r)
    s.translate(np.array(center, dtype=float))
    s.paint_uniform_color(list(color))
    s.compute_vertex_normals()
    return s


def make_grid(cx, cy, cz, size=0.5, step=0.05):
    import open3d as o3d
    pts, lines = [], []
    n = int(size / step)
    for i in range(-n, n+1):
        x = cx + i*step
        j = len(pts)
        pts += [[x, cy - n*step, cz], [x, cy + n*step, cz]]
        lines.append([j, j+1])
    for i in range(-n, n+1):
        y = cy + i*step
        j = len(pts)
        pts += [[cx - n*step, y, cz], [cx + n*step, y, cz]]
        lines.append([j, j+1])
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.paint_uniform_color([0.25, 0.25, 0.25])
    return ls


# ── Camera ─────────────────────────────────────────────────────────────────────

def start_realsense(camera_index):
    import pyrealsense2 as rs
    ctx  = rs.context()
    devs = ctx.query_devices()
    sn   = devs[camera_index].get_info(rs.camera_info.serial_number)
    pipe = rs.pipeline()
    cfg  = rs.config()
    cfg.enable_device(sn)
    cfg.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)
    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16,  30)
    profile     = pipe.start(cfg)
    intr        = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K           = np.array([[intr.fx,0,intr.ppx],[0,intr.fy,intr.ppy],[0,0,1]], np.float64)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align       = rs.align(rs.stream.color)
    print(f'Camera {camera_index} warming up...')
    for _ in range(60):
        pipe.wait_for_frames()
    return pipe, align, K, depth_scale


def capture(pipe, align, depth_scale):
    import pyrealsense2 as rs
    frames = align.process(pipe.wait_for_frames(timeout_ms=3000))
    rgb    = np.asanyarray(frames.get_color_frame().get_data())
    depth  = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32) * depth_scale
    return rgb, depth


# ── GroundedSAM + FP ───────────────────────────────────────────────────────────

def load_gdino_sam(device):
    import groundingdino
    gdino_cfg = os.path.join(os.path.dirname(groundingdino.__file__),
                             'config', 'GroundingDINO_SwinT_OGC.py')
    from groundingdino.util.inference import load_model as load_gdino
    from segment_anything import sam_model_registry, SamPredictor
    print('Loading GroundingDINO...')
    gdino = load_gdino(gdino_cfg, GDINO_WEIGHTS).to(device).eval()
    print('Loading SAM...')
    sam   = sam_model_registry['vit_h'](checkpoint=SAM_WEIGHTS).to(device)
    return gdino, SamPredictor(sam)


def segment_tool(gdino, sam_pred, img_rgb, prompt, box_thr, text_thr, device):
    import torch
    import torchvision.transforms as T
    from groundingdino.util.inference import predict
    from PIL import Image
    H, W = img_rgb.shape[:2]
    transform = T.Compose([T.Resize(800), T.ToTensor(),
                            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
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


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mesh',           required=True)
    p.add_argument('--tool_prompt',    required=True)
    p.add_argument('--camera',         type=int,   default=0)
    p.add_argument('--box_threshold',  type=float, default=0.3)
    p.add_argument('--text_threshold', type=float, default=0.25)
    p.add_argument('--est_refine_iter',    type=int, default=5)
    p.add_argument('--track_refine_iter',  type=int, default=2)
    p.add_argument('--device',         default='cuda')
    p.add_argument('--cam_extrinsics',    default='data/cam_extrinsics.npy')
    p.add_argument('--robot_extrinsics',  default='data/robot_extrinsics.npy')
    p.add_argument('--T_eef_spoon',       default='data/T_eef_spoon.npy',
                   help='path to load/save the EEF→spoon calibration')
    return p.parse_args()


def main():
    import open3d as o3d
    args   = parse_args()
    device = args.device

    # ── calibration ──────────────────────────────────────────────────────────
    tf_world2cam = np.load(os.path.join(PIPELINE_DIR, args.cam_extrinsics)).astype(np.float64)
    T_base_task  = np.load(os.path.join(PIPELINE_DIR, args.robot_extrinsics)).astype(np.float64)
    tf_cam2world = np.linalg.inv(tf_world2cam)

    # T_base_task maps world→base, so the board origin in base is just T_base_task[:3,3]
    tp = T_base_task[:3, 3]
    print('Calibration loaded.')
    print(f'  Task origin in base: ({tp[0]:.3f}, {tp[1]:.3f}, {tp[2]:.3f}) m')

    # ── EEF→spoon calibration (optional, can be built with C key) ─────────────
    T_eef_spoon = None
    calib_samples = []
    eef_spoon_path = os.path.join(PIPELINE_DIR, args.T_eef_spoon)
    if os.path.exists(eef_spoon_path):
        T_eef_spoon = np.load(eef_spoon_path).astype(np.float64)
        print(f'  Loaded T_eef_spoon from {eef_spoon_path}')
        t = T_eef_spoon[:3, 3]
        print(f'    translation: ({t[0]*100:.1f}, {t[1]*100:.1f}, {t[2]*100:.1f}) cm')

    # ── connect robot ─────────────────────────────────────────────────────────
    kinova_api = connect_robot()
    q_deg = get_joint_angles_deg(kinova_api)
    print(f'  Joint angles (deg): {np.round(q_deg, 1)}')

    # ── Open3D window FIRST — before any CUDA ────────────────────────────────
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Calibration Debug (Base Frame)', width=1280, height=800)
    ro = vis.get_render_option()
    ro.background_color = np.array([0.10, 0.10, 0.10])
    ro.light_on = True
    ro.mesh_show_back_face = True

    # ── load robot meshes ─────────────────────────────────────────────────────
    print('Loading robot meshes...')
    base_meshes = load_robot_meshes()

    # ── static geometry ───────────────────────────────────────────────────────
    # Robot base frame — large, prominent
    base_frame_geom = make_frame(np.eye(4), size=0.15)

    # Task / ChArUco board frame — T_base_task maps world→base, so use it directly
    task_frame_geom  = make_frame(T_base_task, size=0.12)
    task_origin_geom = make_sphere(tp, r=0.018, color=(1.0, 0.55, 0.0))

    # Reference grid at task-frame table level
    grid_geom = make_grid(tp[0], tp[1], tp[2], size=0.5, step=0.05)

    static_geoms = [base_frame_geom, task_frame_geom, task_origin_geom, grid_geom]
    for g in static_geoms:
        vis.add_geometry(g)

    # Robot arm links (dynamic)
    robot_geoms = build_robot_geoms(q_deg, base_meshes)
    for g in robot_geoms:
        vis.add_geometry(g)

    # FP-tracked spoon placeholder (dynamic, cyan)
    spoon_geoms = [make_frame(np.eye(4), size=0.08),
                   make_sphere([0, 0, 0], r=0.018, color=(0.2, 0.9, 1.0))]
    for g in spoon_geoms:
        vis.add_geometry(g)

    # FK-predicted spoon placeholder (dynamic, green) — shown when T_eef_spoon is known
    pred_spoon_geoms = []

    # ── SAM first, then FP ───────────────────────────────────────────────────
    gdino, sam_pred = load_gdino_sam(device)

    import trimesh
    loaded = trimesh.load(args.mesh)
    mesh = (trimesh.util.concatenate(list(loaded.geometry.values()))
            if isinstance(loaded, trimesh.Scene) else loaded)
    if mesh.bounding_box.extents.max() > 0.5:
        mesh.apply_scale(0.01)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    inv_to_origin = np.linalg.inv(to_origin.astype(np.float64))

    # Build Open3D spoon mesh in OBB frame (same units/transform as FP uses)
    import open3d as o3d
    spoon_o3d_base = o3d.geometry.TriangleMesh()
    spoon_o3d_base.vertices  = o3d.utility.Vector3dVector(np.asarray(mesh.vertices,  np.float64))
    spoon_o3d_base.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces,     np.int32))
    spoon_o3d_base.compute_vertex_normals()
    spoon_o3d_base.transform(to_origin.astype(np.float64))   # now in OBB frame
    spoon_o3d_base.paint_uniform_color([0.85, 0.65, 0.25])   # gold

    print('Loading FoundationPose...')
    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
    import nvdiffrast.torch as dr
    scorer  = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx   = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
        scorer=scorer, refiner=refiner, glctx=glctx,
        debug_dir='/tmp/calib_debug', debug=0,
    )

    # ── camera ───────────────────────────────────────────────────────────────
    pipe, align, K, depth_scale = start_realsense(args.camera)
    cv2.namedWindow('Camera (FP tracking)', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera (FP tracking)', 848, 480)

    # ── camera position in 3D (static — extrinsics don't change) ─────────────
    T_cam_in_base = T_base_task @ tf_cam2world
    cam_pos = T_cam_in_base[:3, 3]
    cam_frame_geom  = make_frame(T_cam_in_base, size=0.08)
    cam_sphere_geom = make_sphere(cam_pos, r=0.022, color=(1.0, 0.85, 0.0))  # yellow
    for g in [cam_frame_geom, cam_sphere_geom]:
        vis.add_geometry(g)
    print(f'  Camera in base: ({cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f}) m')

    # ── EEF frame placeholder (dynamic, updated with arm) ─────────────────────
    fk_frames_init = compute_fk(q_deg)
    T_eef_init = fk_frames_init[-1]
    eef_geoms = [make_frame(T_eef_init, size=0.07),
                 make_sphere(T_eef_init[:3, 3], r=0.014, color=(1.0, 0.15, 0.8))]  # magenta
    for g in eef_geoms:
        vis.add_geometry(g)

    # ── registration helper ───────────────────────────────────────────────────
    def register():
        rgb, depth = capture(pipe, align, depth_scale)
        print(f"\nDetecting '{args.tool_prompt}'...")
        mask = segment_tool(gdino, sam_pred, rgb, args.tool_prompt,
                            args.box_threshold, args.text_threshold, device)
        if mask.sum() < 100:
            print('  Not detected — reposition and press SPACE.')
            return None
        pose = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask,
                            iteration=args.est_refine_iter)
        print('  Registered.')
        return pose

    pose_cam = register()
    initialized = pose_cam is not None

    print('\nLegend (all in ROBOT BASE frame):')
    print('  Large RGB axes at origin    — robot BASE frame')
    print('  RGB axes + orange sphere    — TASK frame (ChArUco board)')
    print('  Grey meshes                 — robot arm (live FK)')
    print('  RGB axes + magenta sphere   — EEF from FK')
    print('  RGB axes + yellow sphere    — camera (from extrinsics)')
    print('  Cyan axes + sphere          — SPOON from FoundationPose (FP)')
    print('  Green axes + sphere         — SPOON predicted from FK (once calibrated)')
    print('\nSPACE = re-register FP | C = capture calib sample | S = save T_eef_spoon | Q = quit')
    if T_eef_spoon is not None:
        print('  [T_eef_spoon already loaded — green predicted spoon active]')
    else:
        print('  [Move arm to 5+ poses, press C at each, then S to save]')

    last_q = q_deg.copy()
    T_base_tool_last = None  # most recent FP spoon pose in base frame

    def _refresh_pred_spoon(T_eef_cur):
        """Rebuild FK-predicted spoon geoms (mesh + axes) from current EEF pose."""
        nonlocal T_eef_spoon
        if T_eef_spoon is None:
            return
        T_pred = T_eef_cur @ T_eef_spoon   # OBB frame → base frame
        # Spoon mesh: spoon_o3d_base is already in OBB frame, just apply T_pred
        m = copy.deepcopy(spoon_o3d_base)
        m.transform(T_pred)
        new_g = [m,
                 make_frame(T_pred, size=0.06),
                 make_sphere(T_pred[:3, 3], r=0.010, color=(0.1, 0.9, 0.2))]
        for g in pred_spoon_geoms:
            vis.remove_geometry(g, reset_bounding_box=False)
        pred_spoon_geoms.clear()
        pred_spoon_geoms.extend(new_g)
        for g in new_g:
            vis.add_geometry(g, reset_bounding_box=False)

    T_eef_cur = fk_frames_init[-1]  # keep current EEF pose in scope

    try:
        while True:
            if not vis.poll_events():
                break
            vis.update_renderer()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            if key == ord(' '):
                pose_cam = register()
                initialized = pose_cam is not None
                continue

            # ── C: capture calibration sample ────────────────────────────────
            if key == ord('c'):
                if not initialized or T_base_tool_last is None:
                    print('\n  [C] Need active FP tracking — press SPACE first.')
                else:
                    sample = np.linalg.inv(T_eef_cur) @ T_base_tool_last
                    calib_samples.append(sample)
                    t = sample[:3, 3]
                    print(f'\n  [C] Sample {len(calib_samples):2d} captured — '
                          f'T_eef_spoon t=({t[0]*100:.1f},{t[1]*100:.1f},{t[2]*100:.1f}) cm')

            # ── S: save averaged T_eef_spoon ──────────────────────────────────
            if key == ord('s'):
                if len(calib_samples) < 2:
                    print(f'\n  [S] Need ≥2 samples (have {len(calib_samples)}) — move arm and press C more.')
                else:
                    translations = np.array([T[:3, 3] for T in calib_samples])
                    rotations    = Rotation.from_matrix([T[:3, :3] for T in calib_samples])
                    T_eef_spoon  = np.eye(4)
                    T_eef_spoon[:3, :3] = rotations.mean().as_matrix()
                    T_eef_spoon[:3,  3] = translations.mean(axis=0)
                    np.save(eef_spoon_path, T_eef_spoon)
                    t = T_eef_spoon[:3, 3]
                    print(f'\n  [S] Saved T_eef_spoon ({len(calib_samples)} samples) → {eef_spoon_path}')
                    print(f'      translation: ({t[0]*100:.1f},{t[1]*100:.1f},{t[2]*100:.1f}) cm')
                    _refresh_pred_spoon(T_eef_cur)

            # ── update robot arm + EEF ────────────────────────────────────────
            now_q = get_joint_angles_deg(kinova_api)
            arm_moved = np.max(np.abs(now_q - last_q)) > 0.5
            if arm_moved:
                last_q = now_q.copy()

                new_robot = build_robot_geoms(now_q, base_meshes)
                for g in robot_geoms:
                    vis.remove_geometry(g, reset_bounding_box=False)
                robot_geoms.clear()
                robot_geoms.extend(new_robot)
                for g in new_robot:
                    vis.add_geometry(g, reset_bounding_box=False)

                fk_frames  = compute_fk(now_q)
                T_eef_cur  = fk_frames[-1]
                new_eef = [make_frame(T_eef_cur, size=0.07),
                           make_sphere(T_eef_cur[:3, 3], r=0.014, color=(1.0, 0.15, 0.8))]
                for g in eef_geoms:
                    vis.remove_geometry(g, reset_bounding_box=False)
                eef_geoms.clear()
                eef_geoms.extend(new_eef)
                for g in new_eef:
                    vis.add_geometry(g, reset_bounding_box=False)

                _refresh_pred_spoon(T_eef_cur)

            # ── capture camera frame ──────────────────────────────────────────
            rgb, depth = capture(pipe, align, depth_scale)
            vis_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # ── update spoon (FP) + draw on camera image ──────────────────────
            if initialized:
                pose_cam = est.track_one(rgb=rgb, depth=depth, K=K,
                                         iteration=args.track_refine_iter)
                T_cam_tool   = pose_cam.astype(np.float64)
                T_cam_center = T_cam_tool @ inv_to_origin
                T_base_tool  = T_base_task @ tf_cam2world @ T_cam_center
                T_base_tool_last = T_base_tool

                draw_fp_axes(vis_bgr, T_cam_center, K, length=0.08)

                n_s = len(calib_samples)
                label = f'FP tracking  |  {n_s} calib samples (C=add, S=save)'
                cv2.putText(vis_bgr, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3)
                cv2.putText(vis_bgr, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,230,100), 1)

                sp   = T_base_tool[:3, 3]
                eef_p = T_eef_cur[:3, 3]
                err = np.linalg.norm(sp - eef_p) * 100
                print(f'\rFP spoon ({sp[0]:+.3f},{sp[1]:+.3f},{sp[2]:+.3f})  '
                      f'EEF ({eef_p[0]:+.3f},{eef_p[1]:+.3f},{eef_p[2]:+.3f})  '
                      f'|err|={err:.1f}cm  samples={n_s}   ',
                      end='', flush=True)

                new_spoon = [make_frame(T_base_tool, size=0.08),
                             make_sphere(sp, r=0.018, color=(0.2, 0.9, 1.0))]
                for g in spoon_geoms:
                    vis.remove_geometry(g, reset_bounding_box=False)
                spoon_geoms.clear()
                spoon_geoms.extend(new_spoon)
                for g in new_spoon:
                    vis.add_geometry(g, reset_bounding_box=False)
            else:
                cv2.putText(vis_bgr, 'Press SPACE to register spoon', (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
                cv2.putText(vis_bgr, 'Press SPACE to register spoon', (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 1)

            cv2.imshow('Camera (FP tracking)', vis_bgr)

    finally:
        kinova_api.CloseAPI()
        pipe.stop()
        vis.destroy_window()
        cv2.destroyAllWindows()
        print()


if __name__ == '__main__':
    main()
