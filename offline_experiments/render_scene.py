#!/usr/bin/env python3
"""
offline_experiments/render_scene.py

Offline 3D scene renderer — no robot, no cameras connected.

Static mode (default):
    Shows the calibration geometry in world frame using Open3D with real robot
    link meshes (same STLs as kinova_fk_viz.py).

Episode animation mode (--episode):
    Loads a recorded episode's tool_poses_task.npz, solves IK for each frame
    to infer robot joint angles, then plays back the episode as an animated
    3D scene.  Two sets of spoon poses are shown simultaneously:
      • Inferred robot arm + EEF (gold mesh) — what the robot WOULD look like
      • Actual recorded positions (cyan spheres) — FoundationPose truth

Usage (run from the pipeline/ directory):
    # Static scene at any joint angles
    python offline_experiments/render_scene.py
    python offline_experiments/render_scene.py --joints "270,180,90,0,0,0"

    # Animate episode 001
    python offline_experiments/render_scene.py \\
        --episode data/episodes/pastaTransfer4/001

    # Animate with trajectory overlay from all episodes
    python offline_experiments/render_scene.py \\
        --episode data/episodes/pastaTransfer4/001 \\
        --task_dir data/episodes/pastaTransfer4

Controls (animation mode):
    SPACE  — pause / resume
    Q      — quit
"""

import os, sys, argparse, glob, copy, time
import numpy as np
from scipy.spatial.transform import Rotation

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
MESH_DIR     = os.path.join(
    os.path.expanduser('~/working_dir/kinovaDrivers'),
    'kinova-ros', 'kinova_description', 'meshes',
)

# ══════════════════════════════════════════════════════════════════════════════
# FK  (mirrors kinova_fk_viz.py / calib_viz_3d.py exactly)
# ══════════════════════════════════════════════════════════════════════════════

_PI = np.pi
_JOINT_PARAMS = [
    ([0,        0,        0.15675], [0,       _PI,   0    ]),
    ([0,        0.0016,  -0.11875], [-_PI/2,  0,     _PI  ]),
    ([0,       -0.410,   0       ], [0,       _PI,   0    ]),
    ([0,        0.2073,  -0.0114 ], [-_PI/2,  0,     _PI  ]),
    ([0,        0,       -0.10375], [ _PI/2,  0,     _PI  ]),
    ([0,        0.10375,  0      ], [-_PI/2,  0,     _PI  ]),
]
_EEF_PARAMS = ([0, 0, -0.1600], [_PI, 0, _PI/2])

_LINK_MESHES = [
    'base', 'shoulder', 'arm', 'forearm',
    'wrist_spherical_1', 'wrist_spherical_2', 'hand_3finger',
]
_LINK_COLORS = [
    [0.35, 0.35, 0.38], [0.45, 0.50, 0.55], [0.50, 0.55, 0.60],
    [0.55, 0.60, 0.65], [0.60, 0.65, 0.70], [0.65, 0.70, 0.75],
    [0.70, 0.75, 0.80],
]
_JOINT_COLORS = [
    [1.0, 0.3, 0.3], [1.0, 0.6, 0.2], [0.9, 0.9, 0.1],
    [0.2, 0.9, 0.2], [0.2, 0.7, 1.0], [0.7, 0.3, 1.0],
]


def _make_T(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    T[:3,  3] = xyz
    return T


def compute_fk(joint_angles_deg):
    """Returns 8 poses (4×4) in robot-BASE frame: base + link1..6 + EEF."""
    q = np.deg2rad(joint_angles_deg)
    T = np.eye(4)
    frames = [T.copy()]
    for (xyz, rpy), qi in zip(_JOINT_PARAMS, q):
        Tr = np.eye(4)
        Tr[:3, :3] = Rotation.from_euler('z', qi).as_matrix()
        T = T @ _make_T(xyz, rpy) @ Tr
        frames.append(T.copy())
    frames.append(T @ _make_T(*_EEF_PARAMS))
    return frames  # 8 entries: base + 6 links + EEF


# ══════════════════════════════════════════════════════════════════════════════
# Numerical IK
# ══════════════════════════════════════════════════════════════════════════════

def _ik_objective(q_deg, T_target):
    T = compute_fk(q_deg)[7]
    dp  = T[:3, 3] - T_target[:3, 3]
    rot = Rotation.from_matrix(T[:3, :3].T @ T_target[:3, :3]).magnitude()
    return np.dot(dp, dp) * 1e4 + rot ** 2


def solve_ik(T_target_base, q_init_deg):
    """Solve IK numerically; returns (q_deg, pos_err_m)."""
    from scipy.optimize import minimize
    r = minimize(_ik_objective, q_init_deg.copy(), args=(T_target_base,),
                 method='L-BFGS-B',
                 options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-9})
    pos_err = np.linalg.norm(compute_fk(r.x)[7][:3, 3] - T_target_base[:3, 3])
    return r.x, pos_err


def precompute_ik(T_eef_targets, q_start):
    """
    Solve IK for every frame, warm-starting from the previous frame's solution.
    Returns (joint_angles_array, pos_errors_array).
    """
    N = len(T_eef_targets)
    solutions = np.zeros((N, 6))
    errors    = np.zeros(N)
    q = q_start.copy()
    print(f'  Pre-computing IK for {N} frames ...')
    for i, T_tgt in enumerate(T_eef_targets):
        q, err = solve_ik(T_tgt, q)
        solutions[i] = q
        errors[i]    = err
        if (i + 1) % 20 == 0 or i == N - 1:
            print(f'\r    {i+1}/{N}  pos_err={err*100:.2f} cm       ', end='', flush=True)
    print()
    bad = (errors > 0.01).sum()
    if bad:
        print(f'  [warn] {bad} frames with IK error > 1 cm')
    else:
        print(f'  IK complete — max pos error: {errors.max()*100:.2f} cm')
    return solutions, errors


# ══════════════════════════════════════════════════════════════════════════════
# Open3D geometry helpers
# ══════════════════════════════════════════════════════════════════════════════

def _frame_geom(T, size=0.05):
    import open3d as o3d
    o   = T[:3, 3]
    pts = [o, o + T[:3,0]*size, o + T[:3,1]*size, o + T[:3,2]*size]
    ls  = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector([[0,1],[0,2],[0,3]]),
    )
    ls.colors = o3d.utility.Vector3dVector([[1,0,0],[0,1,0],[0,0,1]])
    return ls


def _sphere(center, radius=0.015, color=(1,1,0)):
    import open3d as o3d
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    s.translate(np.array(center, dtype=np.float64))
    s.paint_uniform_color(list(color))
    s.compute_vertex_normals()
    return s


def _camera_frustum(tf_world2cam, img_w=848, img_h=480,
                    fx=606, fy=606, depth=0.20, color=(0, 0.8, 1)):
    import open3d as o3d
    T_c2w  = np.linalg.inv(tf_world2cam)
    origin = T_c2w[:3, 3]
    cx, cy = img_w / 2.0, img_h / 2.0
    tips   = []
    for u, v in [(0,0),(img_w,0),(img_w,img_h),(0,img_h)]:
        r = np.array([(u-cx)/fx, (v-cy)/fy, 1.0])
        r /= np.linalg.norm(r)
        tips.append(T_c2w[:3,:3] @ (r * depth) + origin)
    pts   = [origin] + tips
    lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
    ls    = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls


def _charuco_board(w=0.40, h=0.28, n_sq_x=10, n_sq_y=7):
    import open3d as o3d
    pts, lines, colors = [], [], []
    def add_line(a, b, c):
        ia = len(pts); pts.append(a)
        ib = len(pts); pts.append(b)
        lines.append([ia, ib]); colors.append(c)
    gc = [0.55, 0.38, 0.10]
    for i in range(n_sq_x + 1):
        add_line([i*w/n_sq_x, 0, 0], [i*w/n_sq_x, -h, 0], gc)
    for j in range(n_sq_y + 1):
        add_line([0, -j*h/n_sq_y, 0], [w, -j*h/n_sq_y, 0], gc)
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.colors = o3d.utility.Vector3dVector(colors)
    corners = np.array([[0,0,-0.001],[w,0,-0.001],[w,-h,-0.001],[0,-h,-0.001]])
    board   = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(corners),
        triangles=o3d.utility.Vector3iVector([[0,1,2],[0,2,3]]),
    )
    board.paint_uniform_color([0.76, 0.69, 0.50])
    board.compute_vertex_normals()
    return [ls, board]


def _table_plane():
    import open3d as o3d
    corners = np.array([[-0.65,-0.90,-0.004],[0.70,-0.90,-0.004],
                         [0.70, 0.30,-0.004],[-0.65, 0.30,-0.004]])
    m = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(corners),
        triangles=o3d.utility.Vector3iVector([[0,1,2],[0,2,3]]),
    )
    m.paint_uniform_color([0.22, 0.25, 0.28])
    m.compute_vertex_normals()
    return m


def _load_spoon_mesh_base(obj_path):
    """
    Load spoon.obj, scale to metres, pre-apply to_origin.
    The resulting Open3D mesh is in OBB-centered frame.
    To place it in world: mesh.transform(T_task_base @ T_base_eef @ T_eef_spoon)
    (i.e. the raw-mesh-wrt-world transform, which equals T_obb_wrt_world when
    combined with the already-applied to_origin).
    Returns (o3d_mesh_template, to_origin).
    """
    import open3d as o3d, trimesh
    loaded = trimesh.load(obj_path, force='mesh', process=False)
    mesh   = (trimesh.util.concatenate(list(loaded.geometry.values()))
              if isinstance(loaded, trimesh.Scene) else loaded)
    if mesh.bounding_box.extents.max() > 0.5:
        mesh.apply_scale(0.01)
    to_origin, _ = trimesh.bounds.oriented_bounds(mesh)

    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(np.array(mesh.vertices, np.float64)),
        triangles=o3d.utility.Vector3iVector(np.array(mesh.faces,   np.int64)),
    )
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.transform(to_origin.astype(np.float64))   # now in OBB frame
    o3d_mesh.paint_uniform_color([0.83, 0.63, 0.13])   # gold
    return o3d_mesh, to_origin


def _load_link_meshes():
    import open3d as o3d
    meshes = []
    for name in _LINK_MESHES:
        path = os.path.join(MESH_DIR, f'{name}.STL')
        if not os.path.exists(path):
            print(f'  [warn] not found: {path}')
            meshes.append(None)
            continue
        m = o3d.io.read_triangle_mesh(path)
        m.compute_vertex_normals()
        if np.ptp(np.asarray(m.vertices), axis=0).max() > 10:
            m.scale(0.001, center=np.zeros(3))
        meshes.append(m)
        print(f'  {name}.STL  OK')
    return meshes


def _trajectory_pc(T_world_poses, color):
    """Point cloud from a list of (4×4) world-frame poses."""
    import open3d as o3d
    pts = np.array([T[:3, 3] for T in T_world_poses])
    pc  = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc.colors = o3d.utility.Vector3dVector(np.tile(color, (len(pts), 1)))
    return pc


def _build_robot_geoms(q_deg, base_meshes, T_task_base):
    """Build robot arm + joint sphere geometries in WORLD frame."""
    frames_base  = compute_fk(q_deg)
    frames_world = [T_task_base @ f for f in frames_base]
    geoms = []
    for link_idx, (tmpl, col) in enumerate(zip(base_meshes, _LINK_COLORS)):
        if tmpl is None:
            continue
        m = copy.deepcopy(tmpl)
        m.transform(T_task_base @ frames_base[link_idx])
        m.paint_uniform_color(col)
        geoms.append(m)
    for i in range(1, 7):
        T_j = frames_world[i]
        geoms.append(_sphere(T_j[:3, 3], radius=0.012, color=_JOINT_COLORS[i-1]))
        geoms.append(_frame_geom(T_j, size=0.035))
    T_eef = frames_world[7]
    geoms.append(_frame_geom(T_eef, size=0.08))
    geoms.append(_sphere(T_eef[:3, 3], radius=0.018, color=(1, 1, 1)))
    return geoms, frames_world


# ══════════════════════════════════════════════════════════════════════════════
# Episode loading + conversion
# ══════════════════════════════════════════════════════════════════════════════

def load_episode(episode_dir, T_base_task, T_eef_spoon, spoon_mesh_path):
    """
    Load tool_poses_task.npz for an episode.

    Returns:
        fids             — sorted list of frame id strings
        T_world_tools    — list of (4×4) OBB spoon poses in WORLD frame
        T_base_eef_list  — list of (4×4) EEF poses in BASE frame (for IK)
                           Empty list if T_eef_spoon is None.
    """
    import trimesh

    traj_path = os.path.join(episode_dir, 'augmented', 'tool_poses_task.npz')
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f'No tool_poses_task.npz found in {episode_dir}/augmented/')

    data  = dict(np.load(traj_path))
    fids  = sorted(data.keys(), key=int)
    T_task_tools = [data[k].astype(np.float64) for k in fids]
    T_task_base  = np.linalg.inv(T_base_task)

    # Convert task-frame OBB poses to world frame for display
    T_world_tools = [T_task_base @ T for T in T_task_tools]

    # Compute EEF targets for IK
    T_base_eef_list = []
    if T_eef_spoon is not None:
        loaded = trimesh.load(spoon_mesh_path, force='mesh', process=False)
        mesh   = (trimesh.util.concatenate(list(loaded.geometry.values()))
                  if isinstance(loaded, trimesh.Scene) else loaded)
        if mesh.bounding_box.extents.max() > 0.5:
            mesh.apply_scale(0.01)
        to_origin, _ = trimesh.bounds.oriented_bounds(mesh)
        T_tool_eef   = np.linalg.inv(to_origin) @ np.linalg.inv(T_eef_spoon)
        for T_task_tool in T_task_tools:
            T_base_tool = T_base_task @ T_task_tool
            T_base_eef_list.append(T_base_tool @ T_tool_eef)

    return fids, T_world_tools, T_base_eef_list


# ══════════════════════════════════════════════════════════════════════════════
# Static render
# ══════════════════════════════════════════════════════════════════════════════

def static_render(args, tf_world2cam, tf_world2cam1, T_base_task, T_eef_spoon,
                  base_meshes, ep_data):
    import open3d as o3d
    q_deg = np.array([float(x) for x in args.joints.split(',')])
    T_task_base  = np.linalg.inv(T_base_task) if T_base_task is not None else np.eye(4)

    frames_base  = compute_fk(q_deg)
    frames_world = [T_task_base @ f for f in frames_base]
    T_world_eef  = frames_world[7]

    T_world_spoon_raw = None
    if T_eef_spoon is not None and T_base_task is not None:
        T_world_spoon_raw = T_task_base @ frames_base[7] @ T_eef_spoon

    geoms = []
    geoms.append(_table_plane())
    geoms += _charuco_board()
    geoms.append(_frame_geom(np.eye(4), size=0.09))
    geoms.append(_sphere([0,0,0], radius=0.012, color=(1.0, 0.5, 0.0)))

    if tf_world2cam is not None:
        T_c0 = np.linalg.inv(tf_world2cam)
        geoms += [_camera_frustum(tf_world2cam, depth=0.18, color=(0, 0.8, 1)),
                  _frame_geom(T_c0, size=0.07),
                  _sphere(T_c0[:3,3], radius=0.018, color=(0, 0.8, 1))]
    if tf_world2cam1 is not None:
        T_c1 = np.linalg.inv(tf_world2cam1)
        geoms += [_camera_frustum(tf_world2cam1, depth=0.18, color=(0.2, 1.0, 0.2)),
                  _frame_geom(T_c1, size=0.07),
                  _sphere(T_c1[:3,3], radius=0.018, color=(0.2, 1.0, 0.2))]
    if T_base_task is not None:
        geoms += [_frame_geom(T_task_base, size=0.08),
                  _sphere(T_task_base[:3,3], radius=0.022, color=(1.0, 0.50, 0.10))]

    for link_idx, (tmpl, col) in enumerate(zip(base_meshes, _LINK_COLORS)):
        if tmpl is None: continue
        m = copy.deepcopy(tmpl)
        m.transform(T_task_base @ frames_base[link_idx])
        m.paint_uniform_color(col)
        geoms.append(m)
    for i in range(1, 7):
        T_j = frames_world[i]
        geoms += [_sphere(T_j[:3,3], radius=0.012, color=_JOINT_COLORS[i-1]),
                  _frame_geom(T_j, size=0.035)]
    geoms += [_frame_geom(T_world_eef, size=0.08),
              _sphere(T_world_eef[:3,3], radius=0.018, color=(1,1,1))]

    if T_world_spoon_raw is not None:
        spoon_path = os.path.join(PIPELINE_DIR, args.spoon_mesh)
        spoon_tmpl, _ = _load_spoon_mesh_base(spoon_path)
        sp = copy.deepcopy(spoon_tmpl)
        sp.transform(T_world_spoon_raw)
        geoms += [sp, _frame_geom(T_world_spoon_raw, size=0.05)]

    for color, T_world_list in ep_data:
        geoms.append(_trajectory_pc(T_world_list, color))

    joints_str = ','.join([f'{v:.0f}' for v in q_deg])
    o3d.visualization.draw_geometries(
        geoms,
        window_name=f'Tool-as-Interface scene  |  joints=[{joints_str}]',
        width=1280, height=800,
        mesh_show_back_face=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Animated episode playback
# ══════════════════════════════════════════════════════════════════════════════

def animate_episode(args, tf_world2cam, tf_world2cam1, T_base_task, T_eef_spoon,
                    base_meshes, ep_data,
                    fids, T_world_tools, T_base_eef_list, ik_solutions):
    import open3d as o3d
    T_task_base = np.linalg.inv(T_base_task)

    # ── Spoon mesh template (OBB-pre-transformed, in OBB frame) ─────────────
    spoon_path = os.path.join(PIPELINE_DIR, args.spoon_mesh)
    spoon_tmpl = None
    if T_eef_spoon is not None:
        spoon_tmpl, _ = _load_spoon_mesh_base(spoon_path)

    def _spoon_at_frame(fi):
        """Gold spoon mesh placed at IK-derived EEF position."""
        if spoon_tmpl is None:
            return []
        q = ik_solutions[fi]
        T_ws = T_task_base @ compute_fk(q)[7] @ T_eef_spoon
        m = copy.deepcopy(spoon_tmpl)
        m.transform(T_ws)
        return [m]

    def _dynamic_at_frame(fi):
        """Build ALL dynamic geometry for frame fi (robot + spoons + trail)."""
        q    = ik_solutions[fi]
        geoms, _ = _build_robot_geoms(q, base_meshes, T_task_base)

        # Actual recorded spoon position — cyan sphere + axes
        T_act = T_world_tools[fi]
        geoms.append(_sphere(T_act[:3, 3], radius=0.018, color=(0.1, 0.9, 1.0)))
        geoms.append(_frame_geom(T_act, size=0.06))

        # IK-inferred spoon — gold mesh
        geoms += _spoon_at_frame(fi)

        # Played-so-far trail — magenta
        trail_pts = np.array([T_world_tools[k][:3, 3] for k in range(fi + 1)])
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(trail_pts)
        pc.colors = o3d.utility.Vector3dVector(
            np.tile([1.0, 0.2, 0.8], (len(trail_pts), 1)))
        geoms.append(pc)

        return geoms

    # ── Visualizer ──────────────────────────────────────────────────────────
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f'Episode playback — {os.path.basename(args.episode.rstrip("/"))}',
        width=1280, height=800)
    ro = vis.get_render_option()
    ro.background_color = np.array([0.08, 0.08, 0.12])
    ro.light_on = True
    ro.mesh_show_back_face = True
    ro.point_size = 4.0

    # ── Static geometry (added once, never touched again) ────────────────────
    static_geoms = []
    static_geoms.append(_table_plane())
    static_geoms += _charuco_board()
    static_geoms.append(_frame_geom(np.eye(4), size=0.09))
    static_geoms.append(_sphere([0, 0, 0], radius=0.012, color=(1.0, 0.5, 0.0)))

    if tf_world2cam is not None:
        T_c0 = np.linalg.inv(tf_world2cam)
        static_geoms += [_camera_frustum(tf_world2cam, depth=0.18, color=(0, 0.8, 1)),
                         _frame_geom(T_c0, size=0.07),
                         _sphere(T_c0[:3, 3], radius=0.018, color=(0, 0.8, 1))]
    if tf_world2cam1 is not None:
        T_c1 = np.linalg.inv(tf_world2cam1)
        static_geoms += [_camera_frustum(tf_world2cam1, depth=0.18, color=(0.2, 1.0, 0.2)),
                         _frame_geom(T_c1, size=0.07),
                         _sphere(T_c1[:3, 3], radius=0.018, color=(0.2, 1.0, 0.2))]

    static_geoms += [_frame_geom(T_task_base, size=0.08),
                     _sphere(T_task_base[:3, 3], radius=0.022, color=(1.0, 0.50, 0.10))]

    # Full episode trajectory — dim grey backdrop
    static_geoms.append(_trajectory_pc(T_world_tools, [0.50, 0.50, 0.50]))

    for color, T_world_list in ep_data:
        static_geoms.append(_trajectory_pc(T_world_list, [c * 0.4 for c in color]))

    for g in static_geoms:
        vis.add_geometry(g)

    # ── First frame dynamic geometry ─────────────────────────────────────────
    N         = len(fids)
    dyn_geoms = _dynamic_at_frame(0)
    for g in dyn_geoms:
        vis.add_geometry(g)

    # ── Animation callback (called by vis.run() each tick) ───────────────────
    state = {'frame': 0, 'last_t': time.time()}
    dt    = 1.0 / args.play_fps

    def _anim_cb(vis_ref):
        now = time.time()
        if now - state['last_t'] < dt:
            return False   # too soon — skip, keep running

        state['last_t'] = now
        fi = state['frame']

        for g in dyn_geoms:
            vis_ref.remove_geometry(g, reset_bounding_box=False)
        dyn_geoms.clear()
        new_geoms = _dynamic_at_frame(fi)
        dyn_geoms.extend(new_geoms)
        for g in new_geoms:
            vis_ref.add_geometry(g, reset_bounding_box=False)

        T_act = T_world_tools[fi]
        q     = ik_solutions[fi]
        sys.stdout.write(
            f'\r  frame {fi+1:3d}/{N}  '
            f'spoon=({T_act[0,3]*100:+.1f},{T_act[1,3]*100:+.1f},{T_act[2,3]*100:+.1f}) cm  '
            f'q=[{",".join(f"{x:.0f}" for x in q)}]   ')
        sys.stdout.flush()

        state['frame'] = (fi + 1) % N
        return False   # False = keep running

    vis.register_animation_callback(_anim_cb)

    print(f'\nPlaying {N} frames at {args.play_fps:.0f} fps  (close window to stop)')
    print('  Grey dots     = full episode trajectory (actual FP positions)')
    print('  Cyan sphere   = current actual spoon position')
    print('  Gold mesh     = current IK-inferred spoon (attached to arm)')
    print('  Magenta trail = frames played so far\n')

    vis.run()
    vis.destroy_window()
    print()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--joints', default='0,270,90,0,0,0',
                   help='6 joint angles (degrees) for static render')
    p.add_argument('--episode', default=None,
                   help='Episode directory to animate '
                        '(e.g. data/episodes/pastaTransfer4/001). '
                        'Enables animation mode: IK is solved per frame.')
    p.add_argument('--ik_start', default='80,220,60,70,-120,85',
                   help='Initial joint angles (degrees) for IK warm-start '
                        '(should be near the actual working configuration)')
    p.add_argument('--play_fps', type=float, default=15.0,
                   help='Playback framerate (fps)')
    p.add_argument('--cam_extrinsics',   default='data/cam_extrinsics.npy')
    p.add_argument('--cam1_extrinsics',  default='data/cam1_extrinsics.npy')
    p.add_argument('--robot_extrinsics', default='data/robot_extrinsics.npy')
    p.add_argument('--T_eef_spoon',      default='data/T_eef_spoon.npy')
    p.add_argument('--spoon_mesh',       default='spoon.obj')
    p.add_argument('--task_dir',         default=None,
                   help='Optional background trajectory overlay '
                        '(e.g. data/episodes/pastaTransfer4)')
    p.add_argument('--max_episodes', type=int, default=50)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import open3d as o3d
    args = parse_args()

    def ppath(rel):
        return os.path.join(PIPELINE_DIR, rel)

    # ── Calibration ───────────────────────────────────────────────────────────
    def load_npy(rel, name):
        path = ppath(rel)
        if os.path.exists(path):
            print(f'  [OK] {name}')
            return np.load(path).astype(np.float64)
        print(f'  [--] {name} not found')
        return None

    print('\n── Calibration ─────────────────────────────────────────────────────')
    tf_world2cam  = load_npy(args.cam_extrinsics,   'cam0 extrinsics')
    tf_world2cam1 = load_npy(args.cam1_extrinsics,  'cam1 extrinsics')
    T_base_task   = load_npy(args.robot_extrinsics,  'robot extrinsics (T_base_task)')
    T_eef_spoon   = load_npy(args.T_eef_spoon,       'T_eef_spoon')

    if T_base_task is None:
        sys.exit('ERROR: robot_extrinsics.npy is required.')

    # ── Robot meshes ──────────────────────────────────────────────────────────
    print('\n── Loading robot STL meshes ────────────────────────────────────────')
    base_meshes = _load_link_meshes()

    # ── Background episode trajectories ──────────────────────────────────────
    ep_data = []   # list of (color_rgb, list[T_world])
    if args.task_dir:
        task_dir = (args.task_dir if os.path.isabs(args.task_dir)
                    else ppath(args.task_dir))
        ep_dirs  = sorted(
            d for d in glob.glob(os.path.join(task_dir, '*'))
            if os.path.isdir(d) and
               os.path.exists(os.path.join(d, 'meta.json'))
        )[:args.max_episodes]
        import matplotlib.cm as cm_mod
        cmap = cm_mod.plasma
        T_task_base = np.linalg.inv(T_base_task)
        print(f'\n── Background trajectories ({len(ep_dirs)} episodes) ─────────────────')
        for i, ep_dir in enumerate(ep_dirs):
            tpath = os.path.join(ep_dir, 'augmented', 'tool_poses_task.npz')
            if not os.path.exists(tpath):
                continue
            data  = dict(np.load(tpath))
            fids  = sorted(data.keys(), key=int)
            rgba  = cmap(i / max(1, len(ep_dirs) - 1))
            T_world_list = [T_task_base @ data[k].astype(np.float64) for k in fids]
            ep_data.append((list(rgba[:3]), T_world_list))

    # ── Static mode ───────────────────────────────────────────────────────────
    if args.episode is None:
        print('\n── Static render ───────────────────────────────────────────────────')
        print('  (use --episode <dir> to animate an episode)')
        static_render(args, tf_world2cam, tf_world2cam1, T_base_task, T_eef_spoon,
                      base_meshes, ep_data)
        return

    # ── Animation mode ────────────────────────────────────────────────────────
    episode_dir = (args.episode if os.path.isabs(args.episode)
                   else ppath(args.episode))
    print(f'\n── Loading episode: {episode_dir} ─────────────────────────────────')
    spoon_path = ppath(args.spoon_mesh)
    fids, T_world_tools, T_base_eef_list = load_episode(
        episode_dir, T_base_task, T_eef_spoon, spoon_path)
    print(f'  {len(fids)} frames loaded')

    if not T_base_eef_list:
        print('[warn] T_eef_spoon not available — robot arm will not animate.')
        ik_solutions = np.tile(
            [float(x) for x in args.ik_start.split(',')], (len(fids), 1))
    else:
        print('\n── Solving IK ──────────────────────────────────────────────────────')
        q_start = np.array([float(x) for x in args.ik_start.split(',')])
        ik_solutions, _ = precompute_ik(T_base_eef_list, q_start)

    print('\n── Starting animation ──────────────────────────────────────────────')
    animate_episode(args, tf_world2cam, tf_world2cam1, T_base_task, T_eef_spoon,
                    base_meshes, ep_data,
                    fids, T_world_tools, T_base_eef_list, ik_solutions)


if __name__ == '__main__':
    main()
