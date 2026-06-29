#!/usr/bin/env python3
"""
Live 3D visualisation companion for 07_deploy.py.

Run in a separate terminal alongside 07_deploy.py:
    python deploy_viz.py --mesh spoon.obj --T_eef_spoon data/T_eef_spoon.npy

Reads /tmp/deploy_state.npz (written every step by 07_deploy.py) and shows:
  - Robot arm mesh        (live FK from joint angles)
  - Spoon mesh            (gold, FK + T_eef_spoon)
  - Current tool pose     (cyan axes)
  - Predicted actions     (sphere waypoints, green→red gradient)
  - Task / ChArUco frame  (orange)
  - Camera frame          (yellow)
  - Robot base frame      (large RGB axes)

Q to quit.
"""

import os, sys, copy, time, argparse
import numpy as np
from scipy.spatial.transform import Rotation

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
MESH_DIR = os.path.join(
    os.path.expanduser('~/working_dir/kinovaDrivers'),
    'kinova-ros', 'kinova_description', 'meshes',
)

STATE_FILE = '/tmp/deploy_state.npz'


# ── FK chain ──────────────────────────────────────────────────────────────────
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
    [0.30, 0.30, 0.35], [0.38, 0.40, 0.45], [0.42, 0.44, 0.50],
    [0.46, 0.48, 0.54], [0.50, 0.52, 0.58], [0.54, 0.56, 0.62],
    [0.58, 0.60, 0.66],
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
    return frames   # base + 6 links + EEF


# ── Open3D helpers ─────────────────────────────────────────────────────────────

def make_frame(T, size=0.08):
    import open3d as o3d
    o = T[:3, 3]
    pts = [o, o + T[:3, 0]*size, o + T[:3, 1]*size, o + T[:3, 2]*size]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector([[0,1],[0,2],[0,3]]),
    )
    ls.colors = o3d.utility.Vector3dVector([[1,0,0],[0,1,0],[0,0,1]])
    return ls


def make_sphere(center, r=0.015, color=(1,1,0)):
    import open3d as o3d
    s = o3d.geometry.TriangleMesh.create_sphere(radius=r)
    s.translate(np.asarray(center, dtype=float))
    s.paint_uniform_color(list(color))
    s.compute_vertex_normals()
    return s


def make_line(p0, p1, color=(0.5, 0.5, 0.5)):
    import open3d as o3d
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([p0, p1]),
        lines=o3d.utility.Vector2iVector([[0, 1]]),
    )
    ls.colors = o3d.utility.Vector3dVector([list(color)])
    return ls


def make_grid(cx, cy, cz, size=0.5, step=0.05):
    import open3d as o3d
    pts, lines = [], []
    n = int(size / step)
    for i in range(-n, n+1):
        x = cx + i*step
        j = len(pts)
        pts += [[x, cy-n*step, cz], [x, cy+n*step, cz]]
        lines.append([j, j+1])
    for i in range(-n, n+1):
        y = cy + i*step
        j = len(pts)
        pts += [[cx-n*step, y, cz], [cx+n*step, y, cz]]
        lines.append([j, j+1])
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.paint_uniform_color([0.20, 0.20, 0.20])
    return ls


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
        if np.ptp(np.asarray(m.vertices), axis=0).max() > 10:
            m.scale(0.001, center=np.zeros(3))
        meshes.append(m)
    return meshes


def build_robot_geoms(q_deg, base_meshes):
    frames = compute_fk(q_deg)
    geoms = []
    for i, (tmpl, col) in enumerate(zip(base_meshes, _LINK_COLORS)):
        if tmpl is None:
            continue
        m = copy.deepcopy(tmpl)
        m.transform(frames[i])
        m.paint_uniform_color(col)
        geoms.append(m)
    return geoms, frames[-1]   # (link geoms, T_eef)


def build_action_waypoints(poses_tool_pred, inv_to_origin):
    """
    Render predicted tool poses as a gradient sphere chain + connecting lines.
    poses_tool_pred: (N, 4, 4) base-frame tool poses (T_base_task @ T_pred).
    Returns list of o3d geometries.
    """
    if poses_tool_pred is None or len(poses_tool_pred) == 0:
        return []

    N = len(poses_tool_pred)
    geoms = []
    positions = []

    for i, T in enumerate(poses_tool_pred):
        # T_base_task @ T_pred has to_origin baked in; apply inv_to_origin to get
        # the mesh-origin position — same convention as the gold spoon mesh display.
        T_obb = T @ inv_to_origin
        pos = T_obb[:3, 3]
        positions.append(pos)

        frac = i / max(N - 1, 1)           # 0 = nearest (green), 1 = farthest (red)
        color = (frac, 1.0 - frac, 0.05)

        r = 0.012 if i < N - 1 else 0.016  # last waypoint slightly bigger
        geoms.append(make_sphere(pos, r=r, color=color))

    # Connecting line strip
    for i in range(len(positions) - 1):
        frac = i / max(N - 1, 1)
        col = (frac, 1.0 - frac, 0.05)
        geoms.append(make_line(positions[i], positions[i+1], color=col))

    return geoms


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mesh',              required=True)
    p.add_argument('--T_eef_spoon',       default='data/T_eef_spoon.npy')
    p.add_argument('--cam_extrinsics',    default='data/cam_extrinsics.npy')
    p.add_argument('--robot_extrinsics',  default='data/robot_extrinsics.npy')
    p.add_argument('--poll_hz',           type=float, default=10.0,
                   help='How often to poll the state file (Hz)')
    return p.parse_args()


def main():
    import open3d as o3d
    args = parse_args()

    # ── calibration ────────────────────────────────────────────────────────────
    tf_world2cam = np.load(os.path.join(PIPELINE_DIR, args.cam_extrinsics)).astype(np.float64)
    T_base_task  = np.load(os.path.join(PIPELINE_DIR, args.robot_extrinsics)).astype(np.float64)
    tf_cam2world = np.linalg.inv(tf_world2cam)

    T_eef_spoon  = np.load(os.path.join(PIPELINE_DIR, args.T_eef_spoon)).astype(np.float64)

    # ── mesh / to_origin ───────────────────────────────────────────────────────
    import trimesh
    loaded = trimesh.load(args.mesh)
    mesh = (trimesh.util.concatenate(list(loaded.geometry.values()))
            if isinstance(loaded, trimesh.Scene) else loaded)
    if mesh.bounding_box.extents.max() > 0.5:
        mesh.apply_scale(0.01)
    to_origin, _ = trimesh.bounds.oriented_bounds(mesh)
    inv_to_origin = np.linalg.inv(to_origin.astype(np.float64))

    # Spoon mesh in OBB frame (same as calib_viz_3d.py)
    spoon_o3d_base = o3d.geometry.TriangleMesh()
    spoon_o3d_base.vertices  = o3d.utility.Vector3dVector(np.asarray(mesh.vertices,  np.float64))
    spoon_o3d_base.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces,     np.int32))
    spoon_o3d_base.compute_vertex_normals()
    spoon_o3d_base.transform(to_origin.astype(np.float64))
    spoon_o3d_base.paint_uniform_color([0.85, 0.65, 0.25])

    # ── Open3D window ──────────────────────────────────────────────────────────
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Deploy 3D View', width=1280, height=800)
    ro = vis.get_render_option()
    ro.background_color = np.array([0.08, 0.08, 0.08])
    ro.light_on = True
    ro.mesh_show_back_face = True

    # ── static geometry ────────────────────────────────────────────────────────
    tp = T_base_task[:3, 3]

    base_frame_g  = make_frame(np.eye(4), size=0.15)
    task_frame_g  = make_frame(T_base_task, size=0.12)
    task_sphere_g = make_sphere(tp, r=0.018, color=(1.0, 0.55, 0.0))
    grid_g        = make_grid(tp[0], tp[1], tp[2], size=0.6, step=0.05)

    T_cam_in_base = T_base_task @ tf_cam2world
    cam_frame_g   = make_frame(T_cam_in_base, size=0.08)
    cam_sphere_g  = make_sphere(T_cam_in_base[:3, 3], r=0.022, color=(1.0, 0.85, 0.0))

    static_geoms = [base_frame_g, task_frame_g, task_sphere_g,
                    grid_g, cam_frame_g, cam_sphere_g]
    for g in static_geoms:
        vis.add_geometry(g)

    # ── robot meshes ───────────────────────────────────────────────────────────
    print('Loading robot meshes...')
    base_meshes = load_robot_meshes()

    # Dynamic geometry buckets (replaced every poll)
    robot_geoms  = []
    spoon_geoms  = []
    action_geoms = []
    tool_geoms   = []

    last_q    = None
    last_step = -1
    poll_dt   = 1.0 / args.poll_hz

    print('\nLegend (robot BASE frame):')
    print('  Large RGB axes        — robot base frame')
    print('  Orange sphere/axes    — task frame (ChArUco board)')
    print('  Yellow sphere/axes    — camera')
    print('  Grey meshes           — robot arm (FK from joint angles)')
    print('  Gold mesh             — spoon (FK + T_eef_spoon)')
    print('  Cyan axes             — current tool pose from policy proprio')
    print('  Green→red spheres     — predicted action waypoints (1 per step)')
    print('\nWaiting for /tmp/deploy_state.npz ... (start 07_deploy.py)')
    print('Q to quit.')

    def _replace(bucket, new_geoms):
        for g in bucket:
            vis.remove_geometry(g, reset_bounding_box=False)
        bucket.clear()
        bucket.extend(new_geoms)
        for g in new_geoms:
            vis.add_geometry(g, reset_bounding_box=False)

    t_last_poll = 0.0

    while True:
        if not vis.poll_events():
            break
        vis.update_renderer()

        now = time.time()
        if now - t_last_poll < poll_dt:
            continue
        t_last_poll = now

        # ── read state file ────────────────────────────────────────────────────
        if not os.path.exists(STATE_FILE):
            continue
        try:
            state = np.load(STATE_FILE, allow_pickle=False)
            step = int(state['step'][0])
            if step == last_step:
                continue
            last_step = step
            q_deg          = state['q_deg']
            T_base_tool    = state['T_base_tool']
            poses_tool_pred_raw = state.get('poses_tool_pred', np.zeros((0, 4, 4)))
        except Exception:
            continue  # file mid-write or truncated
        # State file stores task-frame predictions; convert to base for 3D display
        if len(poses_tool_pred_raw) > 0:
            poses_tool_pred = np.stack([T_base_task @ p for p in poses_tool_pred_raw])
        else:
            poses_tool_pred = poses_tool_pred_raw

        # ── robot arm + EEF ───────────────────────────────────────────────────
        if last_q is None or np.max(np.abs(q_deg - last_q)) > 0.3:
            last_q = q_deg.copy()
            new_robot, T_eef = build_robot_geoms(q_deg, base_meshes)
            _replace(robot_geoms, new_robot)

            # Spoon mesh at T_base_eef @ T_eef_spoon
            T_spoon_display = T_eef @ T_eef_spoon
            m = copy.deepcopy(spoon_o3d_base)
            m.transform(T_spoon_display)
            _replace(spoon_geoms, [m])

        # ── current tool pose (proprio) ────────────────────────────────────────
        # T_base_tool[:3,3] is the OBB center in base frame — exactly what the
        # policy receives as proprio.  (The old code applied @ inv_to_origin which
        # showed the raw-mesh origin, a different point on the spoon.)
        _replace(tool_geoms, [make_frame(T_base_tool, size=0.07),
                               make_sphere(T_base_tool[:3, 3], r=0.012,
                                           color=(0.1, 0.9, 1.0))])

        # ── predicted action waypoints ─────────────────────────────────────────
        _replace(action_geoms,
                 build_action_waypoints(poses_tool_pred, inv_to_origin))

        mode = 'FK' if len(poses_tool_pred) > 0 else 'waiting'
        print(f'\r  step={step:5d}  q=[{",".join(f"{v:.1f}" for v in q_deg)}]  '
              f'tool=({T_base_tool[0,3]*100:.1f},{T_base_tool[1,3]*100:.1f},{T_base_tool[2,3]*100:.1f})cm  '
              f'actions={len(poses_tool_pred)}  [{mode}]   ',
              end='', flush=True)

    vis.destroy_window()
    print()


if __name__ == '__main__':
    main()
