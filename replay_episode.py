#!/usr/bin/env python3
"""
Replay a tracked episode's tool trajectory on the real Kinova Jaco2, logging
joint torque (and optionally recaptured RGB) at every waypoint.

This is the "replay-to-sense" step discussed for adding tactile feedback to
the pipeline: human demonstrations carry no force signal (bare hand, no
sensor), so torque can only be harvested by having the ROBOT re-execute the
tracked trajectory and read its own joint-torque sensors while doing so.

Prerequisites for --episode_dir:
  Must already have gone through 04_track.py (+ 04c_to_base_frame.py), i.e.
  contain augmented/tool_poses_base.npz (preferred) or tool_poses_task.npz.

IMPORTANT — tool-specific calibration:
  T_eef_spoon.npy (--T_eef_spoon) and the mesh (--mesh) encode the rigid
  offset from the gripper to the SPECIFIC tool used in that episode. If you
  record a new task with a different tool (e.g. a rigid stylus for a poke
  task instead of the spoon), you must recalibrate T_eef_spoon for that tool
  via calib_viz_3d.py and pass the new mesh/offset here — reusing the spoon's
  calibration with a different tool will silently send wrong EEF targets.

SAFETY:
  Dry run (default) connects to the robot and prints every target EEF pose
  without moving it, so you can sanity-check the trajectory (smooth, small
  steps, sane range) before adding --execute.
  Replay defaults to a SLOW execution (--speed_scale 0.2, 1 cm safety clamps)
  so joint torque reflects contact force rather than motion inertia.

Usage:
  # Dry run — verify the trajectory and targets look sane
  python replay_episode.py --episode_dir data/episodes/pokeTask/001

  # Execute slowly on the real arm, logging torque only
  python replay_episode.py --episode_dir data/episodes/pokeTask/001 --execute

  # Also recapture RGB from cam0 at every waypoint (for image+torque pairing)
  python replay_episode.py --episode_dir data/episodes/pokeTask/001 \\
      --execute --capture_camera
"""
import os, sys, time, ctypes, argparse, json
import numpy as np

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_DIR = os.path.join(os.path.expanduser('~/working_dir/kinovaDrivers'), 'sdk', 'lib')
if _LIB_DIR not in os.environ.get('LD_LIBRARY_PATH', '').split(':'):
    os.environ['LD_LIBRARY_PATH'] = _LIB_DIR + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    os.execv(sys.executable, [sys.executable] + sys.argv)
sys.path.insert(0, PIPELINE_DIR)
from calibrate_firmware_gravity import apply_saved_gravity_params

LIB_PATH      = os.path.join(_LIB_DIR, 'USBCommandLayerUbuntu.so')
COMM_LIB_PATH = os.path.join(_LIB_DIR, 'USBCommLayerUbuntu.so')
NO_ERROR_KINOVA    = 1
SERIAL_LENGTH      = 20
MAX_KINOVA_DEVICE  = 20
CARTESIAN_POSITION = 1
HAND_NOMOVEMENT    = 0


# ════════════════════════════════════════════════════════════════════════════
# Kinova Jaco2 USB SDK bindings — mirrors 07_deploy.py (control) +
# diag_joint_torques.py (torque reads), kept self-contained like every other
# script in this pipeline.
# ════════════════════════════════════════════════════════════════════════════

class KinovaDevice(ctypes.Structure):
    _fields_ = [("SerialNumber",   ctypes.c_char * SERIAL_LENGTH),
                ("Model",          ctypes.c_char * SERIAL_LENGTH),
                ("VersionMajor",   ctypes.c_int),
                ("VersionMinor",   ctypes.c_int),
                ("VersionRelease", ctypes.c_int),
                ("DeviceType",     ctypes.c_int),
                ("DeviceID",       ctypes.c_int)]

# NOTE: the real SDK struct has 7 actuator slots (up to 7-DOF arms) regardless
# of this being a 6-DOF Jaco2 — matches 07_deploy.py. diag_joint_torques.py
# declares only 6 and is missing the 7th field.
class AngularInfo(ctypes.Structure):
    _fields_ = [(f"Actuator{i}", ctypes.c_float) for i in range(1, 8)]

class CartesianInfo(ctypes.Structure):
    _fields_ = [("X", ctypes.c_float), ("Y", ctypes.c_float), ("Z", ctypes.c_float),
                ("ThetaX", ctypes.c_float), ("ThetaY", ctypes.c_float), ("ThetaZ", ctypes.c_float)]

class FingersPosition(ctypes.Structure):
    _fields_ = [("Finger1", ctypes.c_float), ("Finger2", ctypes.c_float), ("Finger3", ctypes.c_float)]

class AngularPosition(ctypes.Structure):
    _fields_ = [("Actuators", AngularInfo), ("Fingers", FingersPosition)]

class CartesianPosition(ctypes.Structure):
    _fields_ = [("Coordinates", CartesianInfo), ("Fingers", FingersPosition)]

class Limitation(ctypes.Structure):
    _fields_ = [("speedParameter1", ctypes.c_float), ("speedParameter2", ctypes.c_float),
                ("speedParameter3", ctypes.c_float), ("forceParameter1", ctypes.c_float),
                ("forceParameter2", ctypes.c_float), ("forceParameter3", ctypes.c_float),
                ("accelerationParameter1", ctypes.c_float), ("accelerationParameter2", ctypes.c_float),
                ("accelerationParameter3", ctypes.c_float)]

class UserPosition(ctypes.Structure):
    _fields_ = [("Type", ctypes.c_int), ("Delay", ctypes.c_float),
                ("CartesianPosition", CartesianInfo), ("Actuators", AngularInfo),
                ("HandMode", ctypes.c_int), ("Fingers", FingersPosition)]

class TrajectoryPoint(ctypes.Structure):
    _fields_ = [("Position", UserPosition), ("LimitationsActive", ctypes.c_int),
                ("SynchroType", ctypes.c_int), ("Limitations", Limitation)]


def load_api():
    ctypes.CDLL(COMM_LIB_PATH, mode=ctypes.RTLD_GLOBAL)
    api = ctypes.CDLL(LIB_PATH)
    for fn, restype in [
        ("InitAPI", ctypes.c_int), ("CloseAPI", ctypes.c_int),
        ("RefresDevicesList", ctypes.c_int), ("GetDevices", ctypes.c_int),
        ("SetActiveDevice", ctypes.c_int),
        ("StartControlAPI", ctypes.c_int), ("StopControlAPI", ctypes.c_int),
        ("GetCartesianPosition", ctypes.c_int), ("GetAngularPosition", ctypes.c_int),
        ("GetAngularVelocity", ctypes.c_int),
        ("SetCartesianControl", ctypes.c_int), ("SendBasicTrajectory", ctypes.c_int),
        ("EraseAllTrajectories", ctypes.c_int),
        ("GetAngularForce", ctypes.c_int), ("GetAngularForceGravityFree", ctypes.c_int),
    ]:
        getattr(api, fn).restype = restype
    api.SendBasicTrajectory.argtypes = [TrajectoryPoint]
    return api


def ok(result):
    return result == NO_ERROR_KINOVA


def connect():
    print("Loading Kinova USB API...")
    api = load_api()
    r = api.InitAPI()
    if not ok(r):
        print(f"ERROR: InitAPI() = {r}")
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
    api.StartControlAPI()
    api.StopControlAPI()
    api.StartControlAPI()
    api.SetCartesianControl()
    print(f"Connected to {devices[0].Model.decode()} (serial {devices[0].SerialNumber.decode()})")
    grav_ok = apply_saved_gravity_params(api)
    if not grav_ok:
        print('ERROR: firmware gravity params (data/gravity_params.npy) failed to reapply. '
              'The Kinova does NOT retain the calibrated gravity matrix across a power cycle '
              '(confirmed 2026-07-02) -- continuing would silently corrupt gravity_free_torque '
              'and every downstream force estimate for this whole session. Aborting rather than '
              'recording bad data. See the message above for why apply_saved_gravity_params failed.')
        api.CloseAPI()
        sys.exit(1)
    print('Firmware gravity params (data/gravity_params.npy) reapplied: OK')
    return api


def get_joint_angles_deg(api) -> np.ndarray:
    pos = AngularPosition()
    api.GetAngularPosition(ctypes.byref(pos))
    a = pos.Actuators
    return np.array([a.Actuator1, a.Actuator2, a.Actuator3,
                     a.Actuator4, a.Actuator5, a.Actuator6], dtype=np.float64)


def get_joint_velocity_deg(api) -> np.ndarray:
    """GetAngularVelocity reuses the AngularPosition struct layout to report
    joint angular velocity (deg/s) instead of position — same convention as
    diag_contact.py."""
    pos = AngularPosition()
    api.GetAngularVelocity(ctypes.byref(pos))
    a = pos.Actuators
    return np.array([a.Actuator1, a.Actuator2, a.Actuator3,
                     a.Actuator4, a.Actuator5, a.Actuator6], dtype=np.float64)


def get_cartesian_pose(api) -> np.ndarray:
    """Returns [X, Y, Z, ThetaX, ThetaY, ThetaZ] (metres, radians)."""
    pos = CartesianPosition()
    r = api.GetCartesianPosition(ctypes.byref(pos))
    if not ok(r):
        print(f"  WARNING: GetCartesianPosition() = {r}")
    c = pos.Coordinates
    return np.array([c.X, c.Y, c.Z, c.ThetaX, c.ThetaY, c.ThetaZ], dtype=np.float64)


def get_torques(api):
    """Returns (raw6, gravity_free6) joint torques in N*m."""
    r = AngularPosition()
    g = AngularPosition()
    api.GetAngularForce(ctypes.byref(r))
    api.GetAngularForceGravityFree(ctypes.byref(g))
    raw = np.array([getattr(r.Actuators, f'Actuator{i}') for i in range(1, 7)], dtype=np.float64)
    gf  = np.array([getattr(g.Actuators, f'Actuator{i}') for i in range(1, 7)], dtype=np.float64)
    return raw, gf


def send_cartesian_pose(api, xyz_theta: np.ndarray, trans_speed: float = 0.0, rot_speed: float = 0.0):
    tp = TrajectoryPoint()
    ctypes.memset(ctypes.byref(tp), 0, ctypes.sizeof(tp))
    tp.Position.Type = CARTESIAN_POSITION
    tp.Position.CartesianPosition.X = float(xyz_theta[0])
    tp.Position.CartesianPosition.Y = float(xyz_theta[1])
    tp.Position.CartesianPosition.Z = float(xyz_theta[2])
    tp.Position.CartesianPosition.ThetaX = float(xyz_theta[3])
    tp.Position.CartesianPosition.ThetaY = float(xyz_theta[4])
    tp.Position.CartesianPosition.ThetaZ = float(xyz_theta[5])
    tp.Position.HandMode = HAND_NOMOVEMENT
    if trans_speed > 0 or rot_speed > 0:
        tp.LimitationsActive = 1
        tp.Limitations.speedParameter1 = float(trans_speed)  # m/s translation
        tp.Limitations.speedParameter2 = float(rot_speed)    # rad/s rotation
    api.SendBasicTrajectory(tp)


# Kinova convention: orientation is Euler-XYZ, Rot = Rx(ThetaX) @ Ry(ThetaY) @ Rz(ThetaZ)
# (scipy intrinsic 'XYZ') — matches 07_deploy.py exactly.

def kinova_pose_to_matrix(xyz_theta: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_euler('XYZ', xyz_theta[3:]).as_matrix()
    T[:3, 3]  = xyz_theta[:3]
    return T


def matrix_to_kinova_pose(T: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    theta = Rotation.from_matrix(T[:3, :3]).as_euler('XYZ')
    return np.concatenate([T[:3, 3], theta])


def clamp_pose_step(T_current: np.ndarray, T_target: np.ndarray,
                     max_trans: float, max_rot_rad: float) -> np.ndarray:
    """Limit the per-command translation/rotation delta from T_current to T_target."""
    from scipy.spatial.transform import Rotation, Slerp

    t_cur, t_tgt = T_current[:3, 3], T_target[:3, 3]
    delta = t_tgt - t_cur
    dist  = np.linalg.norm(delta)
    if dist > max_trans:
        delta = delta * (max_trans / dist)
    t_out = t_cur + delta

    r_cur = Rotation.from_matrix(T_current[:3, :3])
    r_tgt = Rotation.from_matrix(T_target[:3, :3])
    rel   = r_cur.inv() * r_tgt
    angle = rel.magnitude()
    frac  = 1.0 if angle <= max_rot_rad else max_rot_rad / angle
    slerp = Slerp([0, 1], Rotation.concatenate([r_cur, r_tgt]))
    r_out = slerp([frac])[0]

    T_out = np.eye(4)
    T_out[:3, :3] = r_out.as_matrix()
    T_out[:3, 3]  = t_out
    return T_out


# ════════════════════════════════════════════════════════════════════════════
# Episode loading
# ════════════════════════════════════════════════════════════════════════════

def load_tool_trajectory(episode_dir, robot_extrinsics_path):
    """
    Returns a sorted list of (frame_id:int, T_base_tool:4x4) for the episode.
    Prefers augmented/tool_poses_base.npz (already robot-base frame); falls
    back to tool_poses_task.npz converted via T_base_task, matching the same
    preference order documented in README.md.
    """
    aug_dir   = os.path.join(episode_dir, 'augmented')
    base_path = os.path.join(aug_dir, 'tool_poses_base.npz')
    task_path = os.path.join(aug_dir, 'tool_poses_task.npz')

    if os.path.exists(base_path):
        d = np.load(base_path)
        frame_ids = sorted(int(k) for k in d.files)
        print(f"  Using {base_path} ({len(frame_ids)} frames, robot-base frame)")
        return [(fid, d[str(fid)].astype(np.float64)) for fid in frame_ids]

    if not os.path.exists(task_path):
        raise FileNotFoundError(
            f"Neither tool_poses_base.npz nor tool_poses_task.npz found in {aug_dir}. "
            f"Run 04_track.py (+ 04c_to_base_frame.py) on this episode first.")
    d = np.load(task_path)
    T_base_task = np.load(robot_extrinsics_path).astype(np.float64)
    frame_ids = sorted(int(k) for k in d.files)
    print(f"  Using {task_path} + {robot_extrinsics_path} "
          f"({len(frame_ids)} frames, task frame -> base)")
    return [(fid, T_base_task @ d[str(fid)].astype(np.float64)) for fid in frame_ids]


def load_T_tool_eef(mesh_path, T_eef_spoon_path):
    """Same derivation as 07_deploy.py: T_tool_eef = inv_to_origin @ inv(T_eef_spoon)."""
    import trimesh
    loaded = trimesh.load(mesh_path)
    mesh = (trimesh.util.concatenate(list(loaded.geometry.values()))
            if isinstance(loaded, trimesh.Scene) else loaded)
    if mesh.bounding_box.extents.max() > 0.5:
        print("  Mesh extents suggest cm units — rescaling x0.01 to metres")
        mesh.apply_scale(0.01)
    to_origin, _ = trimesh.bounds.oriented_bounds(mesh)
    inv_to_origin = np.linalg.inv(to_origin)
    T_eef_spoon = np.load(T_eef_spoon_path).astype(np.float64)
    return inv_to_origin @ np.linalg.inv(T_eef_spoon)


def capture_rgb(pipe, align):
    frames = align.process(pipe.wait_for_frames(timeout_ms=3000))
    return np.asanyarray(frames.get_color_frame().get_data())


def start_all_realsense():
    """Opens every connected camera (matches 01_record.py's MultiCamera behaviour --
    the original demo saves all connected cameras, so replay's --capture_camera
    should too, not just a single one). Returns [(cam_index, pipe, align), ...]."""
    import pyrealsense2 as rs
    ctx = rs.context()
    devices = list(ctx.devices)
    if not devices:
        raise RuntimeError("--capture_camera requested but no RealSense cameras connected.")
    cams = []
    for i, dev in enumerate(devices):
        serial = dev.get_info(rs.camera_info.serial_number)
        pipe = rs.pipeline()
        cfg  = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)
        pipe.start(cfg)
        align = rs.align(rs.stream.color)
        cams.append((i, pipe, align))
    print(f"Capturing from {len(cams)} camera(s): warming up (30 frames each)...")
    for _, pipe, _ in cams:
        for _ in range(30):
            pipe.wait_for_frames()
    return cams


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--episode_dir', required=True,
                   help='e.g. data/episodes/pokeTask/001')
    p.add_argument('--mesh', default=os.path.join(PIPELINE_DIR, 'spoon.obj'),
                   help='Tool mesh (.obj) used to derive T_tool_eef — must match the tool '
                        'actually used in this episode, not necessarily the spoon.')
    p.add_argument('--T_eef_spoon', default='data/T_eef_spoon.npy',
                   help='EEF->tool rigid offset from calib_viz_3d.py, for the SAME tool as --mesh.')
    p.add_argument('--robot_extrinsics', default='data/robot_extrinsics.npy',
                   help='T_base_task — only used as a fallback when tool_poses_base.npz is absent.')
    p.add_argument('--subsample', type=int, default=3,
                   help='Replay every Nth tracked frame (matches --subsample in 05_train.py).')
    p.add_argument('--start_frame', type=int, default=0)
    p.add_argument('--end_frame', type=int, default=-1, help='-1 = last tracked frame')
    p.add_argument('--x_offset_cm', type=float, default=0.0,
                   help='Constant correction (cm) added to every tracked tool pose\'s '
                        'X (base frame) before converting to EEF commands. Use this to probe '
                        'a suspected systematic robot-base X calibration error.')
    p.add_argument('--y_offset_cm', type=float, default=0.0,
                   help='Constant correction (cm) added to every tracked tool pose\'s '
                        'Y (base frame) before converting to EEF commands. Use this to probe '
                        'a suspected systematic robot-base Y calibration error.')
    p.add_argument('--z_offset_cm', type=float, default=0.0,
                   help='Constant height correction (cm) added to every tracked tool pose\'s '
                        'Z (base frame) before converting to EEF commands. Use this to probe '
                        'a suspected systematic height error in the training-data actions — '
                        'e.g. --z_offset_cm 5.0 raises every commanded height by 5cm. '
                        'Positive = up.')
    p.add_argument('--speed_scale', type=float, default=0.2,
                   help='Replay speed relative to the recorded demo (0.2 = 5x slower). '
                        'Keep this low so torque reflects contact force, not joint '
                        'acceleration from fast motion.')
    p.add_argument('--max_trans_step', type=float, default=0.01,
                   help='Max per-command translation step (m) — safety clamp.')
    p.add_argument('--max_rot_step_deg', type=float, default=8.0,
                   help='Max per-command rotation step (deg) — safety clamp.')
    p.add_argument('--arm_trans_speed', type=float, default=0.03,
                   help='Cartesian translation speed limit sent to the Kinova (m/s). Keep low.')
    p.add_argument('--wait_convergence', action='store_true', default=True,
                   help='Block each step until the arm reaches the target (default: on).')
    p.add_argument('--no_wait_convergence', dest='wait_convergence', action='store_false')
    p.add_argument('--convergence_threshold', type=float, default=0.005,
                   help='EEF position threshold (m) for convergence. Default 5mm.')
    p.add_argument('--convergence_timeout', type=float, default=2.0,
                   help='Max seconds to wait per step for convergence.')
    p.add_argument('--capture_camera', action='store_true',
                   help='Recapture RGB from EVERY connected camera at every settled waypoint '
                        '(matches 01_record.py, which also saves all cameras). Needed because '
                        'the original human-demo frames are NOT valid observations of the '
                        'robot executing this replay.')
    p.add_argument('--execute', action='store_true',
                   help='Actually send commands to the robot (default: dry run / print only).')
    p.add_argument('--output_dir', default=None,
                   help='Defaults to <episode_dir>/replay/')
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = args.output_dir or os.path.join(args.episode_dir, 'replay')
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading tracked trajectory from {args.episode_dir} ...")
    traj = load_tool_trajectory(args.episode_dir, args.robot_extrinsics)
    end = args.end_frame if args.end_frame >= 0 else traj[-1][0]
    traj = [(fid, T) for fid, T in traj
            if args.start_frame <= fid <= end and (fid - args.start_frame) % args.subsample == 0]
    if not traj:
        raise RuntimeError("No waypoints left after --start_frame/--end_frame/--subsample filtering.")
    print(f"  {len(traj)} waypoints after subsample={args.subsample}, "
          f"frame range=[{traj[0][0]}, {traj[-1][0]}]")

    offset_cm = np.array([args.x_offset_cm, args.y_offset_cm, args.z_offset_cm])
    if np.any(offset_cm != 0.0):
        traj = [(fid, T.copy()) for fid, T in traj]
        for _, T in traj:
            T[:3, 3] += offset_cm / 100.0
        print(f"  Applied offset (x,y,z)=({args.x_offset_cm:+.2f},{args.y_offset_cm:+.2f},"
              f"{args.z_offset_cm:+.2f}) cm to every tool pose (base frame)")

    print(f"Deriving T_tool_eef from {args.mesh} + {args.T_eef_spoon} ...")
    T_tool_eef = load_T_tool_eef(args.mesh, args.T_eef_spoon)

    meta_path  = os.path.join(args.episode_dir, 'meta.json')
    record_fps = 30.0
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            record_fps = float(json.load(f).get('fps', 30.0))
    dt_nominal = args.subsample / record_fps
    dt_replay  = dt_nominal / max(args.speed_scale, 1e-3)
    print(f"  record fps={record_fps}  nominal step dt={dt_nominal*1000:.0f}ms  "
          f"-> replay step dt={dt_replay*1000:.0f}ms (speed_scale={args.speed_scale})")

    # Sanity-check summary before touching the robot
    xyz = np.stack([T[:3, 3] for _, T in traj])
    print(f"  tool xyz range (m): x[{xyz[:,0].min():.3f},{xyz[:,0].max():.3f}] "
          f"y[{xyz[:,1].min():.3f},{xyz[:,1].max():.3f}] z[{xyz[:,2].min():.3f},{xyz[:,2].max():.3f}]")
    path_len = np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()
    print(f"  total tool path length: {path_len*100:.1f} cm over {len(traj)} waypoints")

    capture_cams = []   # [(cam_index, pipe, align, rgb_dir), ...]
    if args.capture_camera:
        for cam_idx, pipe, align in start_all_realsense():
            rgb_dir = os.path.join(out_dir, f'cam{cam_idx}')
            os.makedirs(rgb_dir, exist_ok=True)
            capture_cams.append((cam_idx, pipe, align, rgb_dir))

    api = connect()
    if not args.execute:
        print("\n*** DRY RUN — no commands will be sent to the robot. Use --execute to enable. ***\n")

    max_rot_step = np.radians(args.max_rot_step_deg)
    T_base_eef_cur = kinova_pose_to_matrix(get_cartesian_pose(api))

    # ── Pre-position: go directly to frame[0]'s target once, unclamped ──────
    # The arm is almost never already sitting at the episode's first waypoint
    # (it's wherever it was last left). The per-step clamp below is meant to
    # keep the QUASI-STATIC REPLAY smooth/contact-safe — it isn't meant to
    # throttle the initial "walk over to the start of the trajectory" move,
    # which is safe to do in one shot since --arm_trans_speed already caps
    # how fast the arm physically moves.
    T_base_eef_start = traj[0][1] @ T_tool_eef
    start_dist = np.linalg.norm(T_base_eef_start[:3, 3] - T_base_eef_cur[:3, 3])
    print(f"\nPre-positioning to frame {traj[0][0]} "
          f"({start_dist*100:.1f} cm from current pose)...")
    t0 = time.time()   # dense-log clock starts here so pre-position samples land at t>=0
    dense = {'t': [], 'q_deg': [], 'qdot_deg': [], 'raw_torque': [], 'gravity_free_torque': []}

    dense_step = [0]

    def sample_dense():
        raw_t, gf_t = get_torques(api)
        q_deg_now = get_joint_angles_deg(api)
        qdot_deg_now = get_joint_velocity_deg(api)
        t_now = time.time() - t0
        dense['t'].append(t_now)
        dense['q_deg'].append(q_deg_now)
        dense['qdot_deg'].append(qdot_deg_now)
        dense['raw_torque'].append(raw_t)
        dense['gravity_free_torque'].append(gf_t)

        # Live state file, updated at the DENSE rate (not just once per
        # waypoint) so a live viewer (diag_replay_forces.py) gets frequent
        # updates regardless of --no_wait_convergence -- with only one
        # write per waypoint, a live plot only refreshed ~1-4 times/sec,
        # too sparse to see a quick manual push land in real time.
        dense_step[0] += 1
        np.savez('/tmp/replay_state_tmp.npz',
                 step=np.array([dense_step[0]]), frame_id=np.array([-1]),
                 t=np.array([t_now]), q_deg=q_deg_now, qdot_deg=qdot_deg_now,
                 raw_torque=raw_t, gravity_free_torque=gf_t)
        os.replace('/tmp/replay_state_tmp.npz', '/tmp/replay_state.npz')

    if args.execute and start_dist > args.max_trans_step:
        send_cartesian_pose(api, matrix_to_kinova_pose(T_base_eef_start),
                            trans_speed=args.arm_trans_speed)
        arm_speed = args.arm_trans_speed if args.arm_trans_speed > 0 else 0.03
        preposition_timeout = max(args.convergence_timeout, start_dist / arm_speed + 1.0)
        t_wait, converged = time.time(), False
        while time.time() - t_wait < preposition_timeout:
            actual = get_cartesian_pose(api)[:3]
            sample_dense()
            if np.linalg.norm(actual - T_base_eef_start[:3, 3]) < args.convergence_threshold:
                converged = True
                break
            time.sleep(0.02)
        print(f"  Pre-position [{'OK' if converged else 'TIMEOUT'}] "
              f"in {time.time()-t_wait:.1f}s (budget {preposition_timeout:.1f}s)")
        T_base_eef_cur = kinova_pose_to_matrix(get_cartesian_pose(api))
    elif not args.execute:
        print("  (dry run — skipping actual move)")
    else:
        print("  Already within one clamp step — no separate move needed.")

    log = {'frame_id': [], 't': [], 'q_deg': [], 'raw_torque': [], 'gravity_free_torque': [],
           'T_base_eef_cmd': [], 'T_base_eef_actual': []}
    # NOTE: dense/sample_dense/t0 (the continuously-sampled log used to recover
    # qdot/qddot DURING motion, unlike `log` which only records one settled
    # sample per waypoint) were already set up above, before pre-positioning,
    # so pre-position motion is captured too.

    try:
        for i, (fid, T_base_tool) in enumerate(traj):
            T_base_eef_target  = T_base_tool @ T_tool_eef
            T_base_eef_clamped = clamp_pose_step(T_base_eef_cur, T_base_eef_target,
                                                 args.max_trans_step, max_rot_step)
            T_base_eef_cur = T_base_eef_clamped
            t_tgt, t_clm = T_base_eef_target[:3, 3], T_base_eef_clamped[:3, 3]
            print(f"[{i+1}/{len(traj)}] frame {fid}: "
                  f"target=({t_tgt[0]*100:.1f},{t_tgt[1]*100:.1f},{t_tgt[2]*100:.1f})cm  "
                  f"clamped=({t_clm[0]*100:.1f},{t_clm[1]*100:.1f},{t_clm[2]*100:.1f})cm", end='')

            if not args.execute:
                print()
                time.sleep(0.01)
                continue

            send_cartesian_pose(api, matrix_to_kinova_pose(T_base_eef_clamped),
                                trans_speed=args.arm_trans_speed)

            if args.wait_convergence:
                t_wait, converged = time.time(), False
                while time.time() - t_wait < args.convergence_timeout:
                    actual = get_cartesian_pose(api)[:3]
                    sample_dense()
                    if np.linalg.norm(actual - t_clm) < args.convergence_threshold:
                        converged = True
                        break
                    time.sleep(0.01)
                print(f"  [{'OK' if converged else 'TIMEOUT'}]")
            else:
                print()
                t_sleep_start = time.time()
                while time.time() - t_sleep_start < dt_replay:
                    sample_dense()
                    time.sleep(0.01)

            raw_t, gf_t = get_torques(api)
            q_deg_now = get_joint_angles_deg(api)
            qdot_deg_now = get_joint_velocity_deg(api)
            log['frame_id'].append(fid)
            log['t'].append(time.time() - t0)
            log['q_deg'].append(q_deg_now)
            log['raw_torque'].append(raw_t)
            log['gravity_free_torque'].append(gf_t)
            log['T_base_eef_cmd'].append(T_base_eef_clamped)
            log['T_base_eef_actual'].append(kinova_pose_to_matrix(get_cartesian_pose(api)))

            # Live state file for a separate viewer process (e.g. diag_replay_torques.py,
            # diag_replay_forces.py) — mirrors 07_deploy.py's /tmp/deploy_state.npz pattern.
            # Only one process can hold the Kinova USB connection, so live plotting has to
            # read this file instead of polling the arm itself. qdot_deg included (not just
            # q_deg) so a live viewer can finite-difference qddot for dynamics-residual models.
            dense_step[0] += 1
            np.savez('/tmp/replay_state_tmp.npz',
                     step=np.array([dense_step[0]]), frame_id=np.array([fid]),
                     t=np.array([log['t'][-1]]), q_deg=q_deg_now, qdot_deg=qdot_deg_now,
                     raw_torque=raw_t, gravity_free_torque=gf_t)
            os.replace('/tmp/replay_state_tmp.npz', '/tmp/replay_state.npz')

            if capture_cams:
                import cv2
                for cam_idx, pipe, align, rgb_dir in capture_cams:
                    rgb = capture_rgb(pipe, align)
                    cv2.imwrite(os.path.join(rgb_dir, f'{fid:06d}.jpg'),
                                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    except KeyboardInterrupt:
        print("\nInterrupted — stopping.")
    finally:
        if args.execute:
            api.EraseAllTrajectories()
        api.CloseAPI()
        for _, pipe, _, _ in capture_cams:
            pipe.stop()

    if args.execute and log['frame_id']:
        out_path = os.path.join(out_dir, 'torque_log.npz')
        np.savez(out_path,
                 frame_id=np.array(log['frame_id']),
                 t=np.array(log['t']),
                 q_deg=np.stack(log['q_deg']),
                 raw_torque=np.stack(log['raw_torque']),
                 gravity_free_torque=np.stack(log['gravity_free_torque']),
                 T_base_eef_cmd=np.stack(log['T_base_eef_cmd']),
                 T_base_eef_actual=np.stack(log['T_base_eef_actual']))
        print(f"\nSaved torque log -> {out_path}  ({len(log['frame_id'])} steps)")
        if dense['t']:
            dense_path = os.path.join(out_dir, 'torque_log_dense.npz')
            np.savez(dense_path,
                     t=np.array(dense['t']),
                     q_deg=np.stack(dense['q_deg']),
                     qdot_deg=np.stack(dense['qdot_deg']),
                     raw_torque=np.stack(dense['raw_torque']),
                     gravity_free_torque=np.stack(dense['gravity_free_torque']))
            print(f"Saved dense torque/velocity log -> {dense_path}  ({len(dense['t'])} samples)")
        for cam_idx, _, _, rgb_dir in capture_cams:
            print(f"Saved recaptured RGB (cam{cam_idx}) -> {rgb_dir}")

    print("Done.")


if __name__ == '__main__':
    main()
