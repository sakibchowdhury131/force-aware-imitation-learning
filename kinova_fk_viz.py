#!/usr/bin/env python3
"""
Kinova j2s6s300 — Interactive FK Visualizer

Reads live joint angles from the robot, applies forward kinematics from the
official kinova-ros URDF, and renders each link mesh in its correct 3D pose
using Open3D.

Also shows:
  - Camera frame and world origin (from cam_extrinsics.npy)
  - Robot base frame (from robot_extrinsics.npy)
  - GetCartesianPosition EEF vs FK-computed EEF (for validation)

Press Q or Escape to quit.  Press R to refresh joint angles.

Usage:
    python kinova_fk_viz.py
    python kinova_fk_viz.py --no_robot          # offline, all joints = 0
    python kinova_fk_viz.py --joints 0,270,90,0,0,0   # offline, specific angles
"""

import os, sys, ctypes, argparse, time
import numpy as np
from scipy.spatial.transform import Rotation

# ── paths ─────────────────────────────────────────────────────────────────────
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
MESH_DIR     = os.path.join(
    os.path.expanduser('~/working_dir/kinovaDrivers'),
    'kinova-ros', 'kinova_description', 'meshes',
)
_LIB_DIR = os.path.join(os.path.expanduser('~/working_dir/kinovaDrivers'), 'sdk', 'lib')

if not os.environ.get('_FK_VIZ_RELAUNCHED') and \
        _LIB_DIR not in os.environ.get('LD_LIBRARY_PATH', '').split(':'):
    os.environ['LD_LIBRARY_PATH'] = _LIB_DIR + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['_FK_VIZ_RELAUNCHED'] = '1'
    os.execv(sys.executable, [sys.executable] + sys.argv)


# ── Kinova SDK structs ─────────────────────────────────────────────────────────
LIB_PATH      = os.path.join(_LIB_DIR, 'USBCommandLayerUbuntu.so')
COMM_LIB_PATH = os.path.join(_LIB_DIR, 'USBCommLayerUbuntu.so')

class KinovaDevice(ctypes.Structure):
    _fields_ = [('SerialNumber', ctypes.c_char * 20), ('Model', ctypes.c_char * 20),
                ('VersionMajor', ctypes.c_int), ('VersionMinor', ctypes.c_int),
                ('VersionRelease', ctypes.c_int), ('DeviceType', ctypes.c_int),
                ('DeviceID', ctypes.c_int)]

class AngularInfo(ctypes.Structure):
    _fields_ = [(f'Actuator{i}', ctypes.c_float) for i in range(1, 8)]

class FingersPosition(ctypes.Structure):
    _fields_ = [('Finger1', ctypes.c_float), ('Finger2', ctypes.c_float), ('Finger3', ctypes.c_float)]

class AngularPosition(ctypes.Structure):
    _fields_ = [('Actuators', AngularInfo), ('Fingers', FingersPosition)]

class CartesianInfo(ctypes.Structure):
    _fields_ = [('X', ctypes.c_float), ('Y', ctypes.c_float), ('Z', ctypes.c_float),
                ('ThetaX', ctypes.c_float), ('ThetaY', ctypes.c_float), ('ThetaZ', ctypes.c_float)]

class CartesianPosition(ctypes.Structure):
    _fields_ = [('Coordinates', CartesianInfo), ('Fingers', FingersPosition)]


def connect_robot():
    ctypes.CDLL(COMM_LIB_PATH, mode=ctypes.RTLD_GLOBAL)
    api = ctypes.CDLL(LIB_PATH)
    for fn in ['InitAPI', 'RefresDevicesList', 'GetDevices', 'SetActiveDevice',
               'GetAngularPosition', 'GetCartesianPosition', 'CloseAPI']:
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
    """Returns 6 joint angles in degrees."""
    pos = AngularPosition()
    api.GetAngularPosition(ctypes.byref(pos))
    a = pos.Actuators
    return np.array([a.Actuator1, a.Actuator2, a.Actuator3,
                     a.Actuator4, a.Actuator5, a.Actuator6], dtype=np.float64)


def get_cartesian_eef(api):
    """Returns (4,4) T_base_eef from GetCartesianPosition (xyz + extrinsic Euler)."""
    pos = CartesianPosition()
    api.GetCartesianPosition(ctypes.byref(pos))
    c = pos.Coordinates
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('XYZ', [c.ThetaX, c.ThetaY, c.ThetaZ]).as_matrix()
    T[:3,  3] = [c.X, c.Y, c.Z]
    return T, (c.X, c.Y, c.Z, c.ThetaX, c.ThetaY, c.ThetaZ)


# ── FK from URDF j2s6s300.xacro ────────────────────────────────────────────────
# Each entry is (xyz, rpy) of the joint origin relative to the parent link.
# All rotation angles in radians.  Source: kinova-ros j2s6s300.xacro
_PI = np.pi
_JOINT_PARAMS = [
    ([0,       0,       0.15675 ], [0,       _PI,    0     ]),   # joint_1
    ([0,       0.0016, -0.11875 ], [-_PI/2,  0,      _PI   ]),   # joint_2
    ([0,      -0.410,  0        ], [0,       _PI,    0     ]),   # joint_3
    ([0,       0.2073, -0.0114  ], [-_PI/2,  0,      _PI   ]),   # joint_4
    ([0,       0,      -0.10375 ], [ _PI/2,  0,      _PI   ]),   # joint_5
    ([0,       0.10375, 0       ], [-_PI/2,  0,      _PI   ]),   # joint_6
]
_EEF_PARAMS = ([0, 0, -0.1600], [_PI, 0, _PI/2])


def _make_T(xyz, rpy):
    """Build 4x4 homogeneous transform from URDF origin (xyz + rpy)."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    T[:3,  3] = xyz
    return T


def compute_fk(joint_angles_deg):
    """
    Forward kinematics for j2s6s300 using URDF joint chain.

    Parameters
    ----------
    joint_angles_deg : array-like, length 6 (degrees, as returned by GetAngularPosition)

    Returns
    -------
    frames : list of 8 (4,4) arrays
        T_base_link0, T_base_link1, ..., T_base_link6, T_base_eef
        (link0 = base link, frames[i] is the pose of link i in the robot base frame)
    """
    q = np.deg2rad(joint_angles_deg)
    frames = [np.eye(4)]   # frame[0] = base link (identity)

    T = np.eye(4)
    for (xyz, rpy), qi in zip(_JOINT_PARAMS, q):
        T_origin = _make_T(xyz, rpy)
        T_joint  = np.eye(4)
        T_joint[:3, :3] = Rotation.from_euler('z', qi).as_matrix()
        T = T @ T_origin @ T_joint
        frames.append(T.copy())

    # Fixed EEF frame (no joint angle)
    T = T @ _make_T(*_EEF_PARAMS)
    frames.append(T.copy())
    return frames   # 8 elements: base + 6 links + EEF


# ── Mesh loading ───────────────────────────────────────────────────────────────
# Mesh name for each link (same order as frames[1..6]).
# Source: link_N_mesh properties in j2s6s300.xacro
_LINK_MESHES = [
    'base',             # link_base  (frames[0])
    'shoulder',         # link_1     (frames[1])
    'arm',              # link_2     (frames[2])
    'forearm',          # link_3     (frames[3])
    'wrist_spherical_1',# link_4     (frames[4])
    'wrist_spherical_2',# link_5     (frames[5])
    'hand_3finger',     # link_6     (frames[6])
]


def load_meshes():
    """Load all link STL files.  Returns list of (o3d.TriangleMesh, link_idx)."""
    import open3d as o3d
    meshes = []
    for idx, name in enumerate(_LINK_MESHES):
        path = os.path.join(MESH_DIR, f'{name}.STL')
        if not os.path.exists(path):
            print(f'  [warn] mesh not found: {path}')
            meshes.append(None)
            continue
        m = o3d.io.read_triangle_mesh(path)
        m.compute_vertex_normals()

        # Detect mm vs m: kinova meshes span ~10-500mm range
        verts = np.asarray(m.vertices)
        span = np.ptp(verts, axis=0).max()
        if span > 10:   # clearly in mm
            m.scale(0.001, center=np.zeros(3))
            print(f'  {name}.STL  span={span:.0f}mm → scaled to metres')
        else:
            print(f'  {name}.STL  span={span*1000:.0f}mm  (already in m)')

        meshes.append(m)
    return meshes


def _make_frame_geometry(T, size=0.05):
    """Create Open3D LineSet for a coordinate frame at pose T."""
    import open3d as o3d
    o = T[:3, 3]
    pts  = [o, o + T[:3,0]*size, o + T[:3,1]*size, o + T[:3,2]*size]
    lines = [[0,1],[0,2],[0,3]]
    colors = [[1,0,0],[0,1,0],[0,0,1]]   # X=red, Y=green, Z=blue
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def _make_sphere(center, radius=0.015, color=(1,1,0)):
    import open3d as o3d
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    s.translate(center)
    s.paint_uniform_color(color)
    s.compute_vertex_normals()
    return s


def build_link_colors():
    """Distinct colours for each link mesh."""
    return [
        [0.40, 0.40, 0.45],   # base   — dark grey
        [0.50, 0.55, 0.60],   # link_1 — grey-blue
        [0.55, 0.60, 0.65],   # link_2
        [0.60, 0.65, 0.70],   # link_3
        [0.65, 0.70, 0.75],   # link_4
        [0.70, 0.75, 0.80],   # link_5
        [0.75, 0.80, 0.85],   # link_6 (gripper)
    ]


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--no_robot', action='store_true',
                   help='Offline mode — no Kinova connection')
    p.add_argument('--joints', default=None,
                   help='Comma-separated joint angles (degrees) for offline mode')
    p.add_argument('--cam_extrinsics',   default='data/cam_extrinsics.npy')
    p.add_argument('--robot_extrinsics', default='data/robot_extrinsics.npy')
    p.add_argument('--loop_hz', type=float, default=2.0,
                   help='Refresh rate for live joint angle polling (Hz)')
    return p.parse_args()


def main():
    import open3d as o3d
    args = parse_args()

    # ── Load calibration frames (optional) ──────────────────────────────────
    tf_world2cam = T_base_task = None
    cam_path  = os.path.join(PIPELINE_DIR, args.cam_extrinsics)
    rob_path  = os.path.join(PIPELINE_DIR, args.robot_extrinsics)
    if os.path.exists(cam_path):
        tf_world2cam = np.load(cam_path).astype(np.float64)
        print(f'Loaded cam_extrinsics: {cam_path}')
    if os.path.exists(rob_path):
        T_base_task  = np.load(rob_path).astype(np.float64)
        print(f'Loaded robot_extrinsics: {rob_path}')

    # ── Connect to robot ────────────────────────────────────────────────────
    api = None
    if not args.no_robot:
        try:
            api = connect_robot()
        except Exception as e:
            print(f'[warn] Robot connection failed: {e}')
            print('[warn] Running in offline mode.')

    # ── Initial joint angles ─────────────────────────────────────────────────
    if args.joints:
        q_deg = np.array([float(x) for x in args.joints.split(',')])
    elif api is not None:
        q_deg = get_joint_angles_deg(api)
    else:
        q_deg = np.zeros(6)

    print(f'\nJoint angles (deg): {q_deg}')

    # ── Load meshes ──────────────────────────────────────────────────────────
    print('\nLoading link meshes...')
    base_meshes = load_meshes()
    link_colors = build_link_colors()

    # ── Build initial FK + scene ─────────────────────────────────────────────
    def build_scene(q_deg_vals):
        import copy
        frames = compute_fk(q_deg_vals)   # 8 frames: base + 6 links + EEF
        geometries = []

        # Robot links
        for link_idx, (mesh_template, color) in enumerate(zip(base_meshes, link_colors)):
            if mesh_template is None:
                continue
            T_link = frames[link_idx]
            m = copy.deepcopy(mesh_template)
            m.transform(T_link)
            m.paint_uniform_color(color)
            geometries.append(m)

        # Joint spheres + frame axes
        joint_sphere_colors = [
            [1.0, 0.3, 0.3],   # J1 red
            [1.0, 0.6, 0.2],   # J2 orange
            [0.9, 0.9, 0.1],   # J3 yellow
            [0.2, 0.9, 0.2],   # J4 green
            [0.2, 0.7, 1.0],   # J5 cyan
            [0.7, 0.3, 1.0],   # J6 purple
        ]
        for i in range(1, 7):   # frames[1..6] are joint frames
            T_j = frames[i]
            geometries.append(_make_sphere(T_j[:3, 3], radius=0.012,
                                           color=joint_sphere_colors[i-1]))
            geometries.append(_make_frame_geometry(T_j, size=0.04))

        # EEF frame (frame[7]) — larger axes
        T_eef_fk = frames[7]
        geometries.append(_make_frame_geometry(T_eef_fk, size=0.08))
        geometries.append(_make_sphere(T_eef_fk[:3, 3], radius=0.018, color=[1,1,1]))

        return frames, geometries

    # ── GetCartesianPosition EEF for comparison ──────────────────────────────
    T_cart_eef = None
    if api is not None:
        T_cart_eef, raw = get_cartesian_eef(api)
        print(f'\nGetCartesianPosition:')
        print(f'  XYZ  = ({raw[0]:.4f}, {raw[1]:.4f}, {raw[2]:.4f}) m')
        print(f'  Euler = ({raw[3]:.4f}, {raw[4]:.4f}, {raw[5]:.4f}) rad (xyz extrinsic)')

    frames, scene_geoms = build_scene(q_deg)

    # Print FK vs CartesianPosition comparison
    T_eef_fk = frames[7]
    print(f'\nFK EEF position:   ({T_eef_fk[0,3]:.4f}, {T_eef_fk[1,3]:.4f}, {T_eef_fk[2,3]:.4f}) m')
    if T_cart_eef is not None:
        dt = np.linalg.norm(T_eef_fk[:3,3] - T_cart_eef[:3,3])
        dR = np.degrees(Rotation.from_matrix(T_eef_fk[:3,:3].T @ T_cart_eef[:3,:3]).magnitude())
        print(f'Cartesian EEF:     ({T_cart_eef[0,3]:.4f}, {T_cart_eef[1,3]:.4f}, {T_cart_eef[2,3]:.4f}) m')
        print(f'FK vs Cart error:  trans={dt*100:.1f} cm   rot={dR:.1f} deg')

    # Also show where the FK EEF and CartesianPosition project into the camera
    if tf_world2cam is not None and T_base_task is not None:
        K = np.array([[606,0,424],[0,606,240],[0,0,1]], dtype=np.float64)   # approx
        cam_path2 = os.path.join(PIPELINE_DIR, 'data/cam_extrinsics.npy')
        if os.path.exists(cam_path2):
            # use actual intrinsics if available via realsense (skip here for offline)
            pass

        def project(T_base_pt):
            T_cam = tf_world2cam @ np.linalg.inv(T_base_task) @ T_base_pt
            p = T_cam[:3, 3]
            if p[2] > 0:
                u = int(K[0,0]*p[0]/p[2] + K[0,2])
                v = int(K[1,1]*p[1]/p[2] + K[1,2])
                return u, v, p[2]
            return None

        res = project(T_eef_fk)
        if res:
            print(f'\nFK EEF projected  → pixel ({res[0]}, {res[1]})  depth={res[2]:.3f}m  '
                  f'(image center=424,240 for 848×480)')
        if T_cart_eef is not None:
            res2 = project(T_cart_eef)
            if res2:
                print(f'Cart EEF projected → pixel ({res2[0]}, {res2[1]})  depth={res2[2]:.3f}m')

    # ── Calibration frame overlays ──────────────────────────────────────────
    extra_geoms = []
    if tf_world2cam is not None:
        T_cam_in_world = np.linalg.inv(tf_world2cam)
        # Camera frame in robot base frame (for 3D overlay)
        if T_base_task is not None:
            T_cam_in_base = np.linalg.inv(T_base_task) @ T_cam_in_world
            cam_frame = _make_frame_geometry(T_cam_in_base, size=0.10)
            cam_label = _make_sphere(T_cam_in_base[:3,3], radius=0.012, color=[0,0.8,1])
            extra_geoms += [cam_frame, cam_label]

    if T_base_task is not None:
        # World origin in base frame
        world_in_base = np.linalg.inv(T_base_task)
        world_frame = _make_frame_geometry(world_in_base, size=0.08)
        world_origin = _make_sphere(world_in_base[:3,3], radius=0.012, color=[1,0.5,0])
        extra_geoms += [world_frame, world_origin]

    # CartesianPosition EEF sphere (magenta) for visual comparison
    if T_cart_eef is not None:
        extra_geoms.append(_make_sphere(T_cart_eef[:3,3], radius=0.018, color=[1,0,1]))
        extra_geoms.append(_make_frame_geometry(T_cart_eef, size=0.06))

    # ── Open3D viewer ───────────────────────────────────────────────────────
    all_geoms = scene_geoms + extra_geoms

    print('\nLegend:')
    print('  Coloured meshes  — robot links (from joint angles FK)')
    print('  Coloured spheres — joint origins (J1=red→J6=purple)')
    print('  White sphere+axes — FK EEF (from joint angle chain)')
    print('  Magenta sphere+axes — GetCartesianPosition EEF')
    print('  Orange sphere+axes — world/task frame origin')
    print('  Cyan sphere+axes  — camera frame')
    print('\nClose window or press Q to quit.')

    # Static render (one-shot)
    if args.no_robot or api is None:
        o3d.visualization.draw_geometries(
            all_geoms,
            window_name=f'Kinova j2s6s300 FK  —  joints: {np.round(q_deg, 1)}',
            width=1280, height=800,
            point_show_normal=False,
        )
        return

    # Live refresh loop — rebuild full scene each update
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Kinova j2s6s300 FK (live)', width=1280, height=800)
    ro = vis.get_render_option()
    ro.light_on = True
    ro.mesh_show_back_face = True

    current_geoms = list(all_geoms)
    for g in current_geoms:
        vis.add_geometry(g)

    dt_s = 1.0 / args.loop_hz
    last_q = q_deg.copy()

    try:
        while True:
            if not vis.poll_events():
                break
            vis.update_renderer()

            now_q = get_joint_angles_deg(api)
            if np.max(np.abs(now_q - last_q)) > 0.5:
                last_q = now_q.copy()
                print(f'\rJoints: {np.round(now_q,1)}', end='', flush=True)

                new_frames, new_robot_geoms = build_scene(now_q)

                T_cart, _ = get_cartesian_eef(api)
                dt = np.linalg.norm(new_frames[7][:3,3] - T_cart[:3,3])
                dR = np.degrees(Rotation.from_matrix(
                    new_frames[7][:3,:3].T @ T_cart[:3,:3]).magnitude())
                print(f'  FK vs Cart: {dt*100:.1f} cm  {dR:.1f}°', end='', flush=True)

                # Replace all geometries
                vis.clear_geometries()
                new_all = new_robot_geoms + extra_geoms
                for g in new_all:
                    vis.add_geometry(g, reset_bounding_box=False)
                current_geoms = new_all

            time.sleep(dt_s)

    finally:
        if api:
            api.CloseAPI()
        vis.destroy_window()
        print()


if __name__ == '__main__':
    main()
