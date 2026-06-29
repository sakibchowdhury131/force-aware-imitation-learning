"""
Step 7 — Deploy the trained diffusion policy on the real Kinova Jaco2 6DOF
spherical-wrist arm (fingers rigidly holding the tool, e.g. the spoon).

Pipeline (mirrors the original Tool-as-Interface real-robot deployment,
adapted to the Kinova Jaco2 USB SDK used by 00_calibrate.py / 06_calibrate_robot.py):

  1. Robot forward kinematics gives T_base_eef every step (no per-step
     FoundationPose tracking — avoids tracking drift). Combined with the
     one-time rigid grasp offset T_tool_eef, this gives T_base_tool, the
     proprioceptive signal x^r — directly in the robot-base frame, the same
     frame the policy was trained on (tool_poses_base.npz).
  2. GroundedSAM masks the robot arm/gripper out of the cam0 image (blacked
     out), mirroring how the training data had the human demonstrator's
     hand/arm masked out — keeps the observation distribution consistent.
  3. The diffusion policy (policy_final.pt) takes the last n_obs_steps
     (image, proprio) pairs and predicts action_horizon future tool poses,
     directly in the robot-base frame.
  4. Tool->gripper transform T_tool_eef is calibrated ONCE at startup (rigid
     grasp assumption), via FoundationPose registration + the
     camera->task->base chain (cam_extrinsics.npy, T_base_task):
        T_base_tool_0 = T_base_task @ inv(tf_world2cam) @ T_cam_tool_0
        T_tool_eef    = inv(T_base_tool_0) @ T_base_eef_0
     Then for every predicted tool pose (already in base frame):
        T_base_eef = T_base_tool_pred @ T_tool_eef
     This is the ONLY place cam_extrinsics.npy / T_base_task are used —
     camera recalibration after this one-time step no longer affects action
     correctness.
  5. Receding horizon: execute the first --exec_steps predicted poses (with
     per-step translation/rotation clamping for safety), then re-observe and
     replan.

SAFETY:
  --dry_run (default) only prints/logs target poses and saves a debug video —
  no robot motion. Once the printed targets look sane (smooth, small steps,
  near the current tool position), re-run with --execute.
  Press Q in the preview window to stop — EraseAllTrajectories() is called
  and the arm holds its last commanded position.

Usage:
  # Dry run (no robot motion) — verify perception + predicted targets
  python 07_deploy.py --checkpoint data/checkpoints/pastaTransfer/policy_final.pt \\
      --mesh spoon.obj --tool_prompt "spoon"

  # Execute on the real arm
  python 07_deploy.py --checkpoint data/checkpoints/pastaTransfer/policy_final.pt \\
      --mesh spoon.obj --tool_prompt "spoon" --execute
"""

import os, sys, time, ctypes, argparse, threading, collections
import numpy as np
import cv2

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

from policy_common import pose_matrix_to_9d
from test_policy import (load_model, make_transform, predict_action_sequence,
                         _project, draw_axes_with_horizon, draw_axes_simple)
_ROBOT_UNET_H, _ROBOT_UNET_W = 288, 512
_ROBOT_UNET_MEAN = [0.485, 0.456, 0.406]
_ROBOT_UNET_STD  = [0.229, 0.224, 0.225]


def load_unet(checkpoint_path, device):
    import torch
    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name='resnet34', encoder_weights=None,
                     in_channels=3, classes=1).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_key = 'model_state' if 'model_state' in ckpt else 'model'
    model.load_state_dict(ckpt[state_key])
    model.eval()
    return model


def unet_mask(model, img_rgb, threshold, device):
    """Returns bool mask (H, W) at original resolution."""
    import torch
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF
    from PIL import Image
    t = TF.to_tensor(Image.fromarray(img_rgb).resize(
            (_ROBOT_UNET_W, _ROBOT_UNET_H), Image.BILINEAR))
    t = TF.normalize(t, _ROBOT_UNET_MEAN, _ROBOT_UNET_STD).unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logit = model(t)
    prob = logit.sigmoid().squeeze()
    prob_full = F.interpolate(
        prob.unsqueeze(0).unsqueeze(0).float(),
        size=img_rgb.shape[:2], mode='bilinear', align_corners=False
    ).squeeze().cpu().numpy()
    return prob_full > threshold


# ════════════════════════════════════════════════════════════════════════════
# Joint-angle forward kinematics (same DH chain as deploy_viz.py)
# Used for proprio — more accurate than GetCartesianPosition for spoon height.
# ════════════════════════════════════════════════════════════════════════════

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


def _make_fk_T(xyz, rpy):
    from scipy.spatial.transform import Rotation
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    T[:3,  3] = xyz
    return T


def joint_angles_to_eef(q_deg: np.ndarray) -> np.ndarray:
    """Return 4×4 EEF pose in robot base frame from joint angles (degrees).
    Matches the FK chain in deploy_viz.py — aligns with the gold spoon mesh."""
    from scipy.spatial.transform import Rotation
    q = np.deg2rad(q_deg)
    T = np.eye(4, dtype=np.float64)
    for (xyz, rpy), qi in zip(_FK_JOINT_PARAMS, q):
        Tj = np.eye(4, dtype=np.float64)
        Tj[:3, :3] = Rotation.from_euler('z', float(qi)).as_matrix()
        T = T @ _make_fk_T(xyz, rpy) @ Tj
    return T @ _make_fk_T(*_FK_EEF_PARAMS)


# ════════════════════════════════════════════════════════════════════════════
# Kinova Jaco2 USB SDK bindings (CARTESIAN_POSITION trajectory control)
# ════════════════════════════════════════════════════════════════════════════

NO_ERROR_KINOVA   = 1
SERIAL_LENGTH     = 20
MAX_KINOVA_DEVICE = 20
CARTESIAN_POSITION = 1
HAND_NOMOVEMENT    = 0

LIB_PATH      = os.path.join(_LIB_DIR, "USBCommandLayerUbuntu.so")
COMM_LIB_PATH = os.path.join(_LIB_DIR, "USBCommLayerUbuntu.so")


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
        ("InitAPI", ctypes.c_int),
        ("CloseAPI", ctypes.c_int),
        ("RefresDevicesList", ctypes.c_int),
        ("GetDevices", ctypes.c_int),
        ("SetActiveDevice", ctypes.c_int),
        ("StartControlAPI", ctypes.c_int),
        ("StopControlAPI", ctypes.c_int),
        ("GetCartesianPosition", ctypes.c_int),
        ("GetAngularPosition",   ctypes.c_int),
        ("SetCartesianControl", ctypes.c_int),
        ("SendBasicTrajectory", ctypes.c_int),
        ("EraseAllTrajectories", ctypes.c_int),
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
        if r == 1015:
            print("  ERROR_NO_DEVICE_FOUND (1015): robot not detected on USB.")
            print("  Check: arm powered on? USB cable seated? Try: lsusb | grep -i kinova")
        elif r == 2002:
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

    api.StartControlAPI()
    api.StopControlAPI()
    api.StartControlAPI()
    api.SetCartesianControl()
    print(f"Connected to {devices[0].Model.decode()} (serial {devices[0].SerialNumber.decode()})")
    return api


def get_joint_angles_deg(api) -> np.ndarray:
    pos = AngularPosition()
    api.GetAngularPosition(ctypes.byref(pos))
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


def send_cartesian_pose(api, xyz_theta: np.ndarray,
                         trans_speed: float = 0.0,
                         rot_speed: float = 0.0):
    """Send a CARTESIAN_POSITION trajectory point (fingers untouched).

    trans_speed: max translation speed in m/s (0 = use robot default, ~slow)
    rot_speed:   max rotation speed in rad/s  (0 = use robot default)
    """
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


# ── Pose <-> Kinova Cartesian conversion ──────────────────────────────────────
# Kinova convention: orientation is Euler-XYZ with Rot = Rx(ThetaX) @ Ry(ThetaY) @ Rz(ThetaZ),
# which is scipy's intrinsic 'XYZ' Euler convention.

def kinova_pose_to_matrix(xyz_theta: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    T = np.eye(4, dtype=np.float64)
    # Kinova SDK uses intrinsic XYZ (body-fixed roll-pitch-yaw).
    # Must match matrix_to_kinova_pose which uses as_euler('XYZ').
    T[:3, :3] = Rotation.from_euler('XYZ', xyz_theta[3:]).as_matrix()
    T[:3, 3]  = xyz_theta[:3]
    return T


def matrix_to_kinova_pose(T: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    theta = Rotation.from_matrix(T[:3, :3]).as_euler('XYZ')
    return np.concatenate([T[:3, 3], theta])


# ════════════════════════════════════════════════════════════════════════════
# Perception: RealSense, GroundedSAM segmentation, FoundationPose tracking
# ════════════════════════════════════════════════════════════════════════════

def start_realsense(camera_index=0):
    import pyrealsense2 as rs
    ctx     = rs.context()
    devices = ctx.query_devices()
    n       = len(devices)
    if n == 0:
        raise RuntimeError("No RealSense devices found.")
    if camera_index >= n:
        raise RuntimeError(f"--camera {camera_index} requested but only {n} device(s) connected.")
    serial = devices[camera_index].get_info(rs.camera_info.serial_number)
    print(f"Using camera {camera_index}: serial {serial}  ({n} device(s) connected)")

    pipe = rs.pipeline()
    cfg  = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)
    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    profile = pipe.start(cfg)

    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0, 0, 1]], dtype=np.float64)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale  = depth_sensor.get_depth_scale()

    sensor = profile.get_device().first_color_sensor()
    sensor.set_option(rs.option.enable_auto_exposure, 1)
    sensor.set_option(rs.option.enable_auto_white_balance, 1)

    align = rs.align(rs.stream.color)

    print("Warming up camera (90 frames)...")
    for _ in range(90):
        pipe.wait_for_frames()

    return pipe, align, K, depth_scale


def capture(pipe, align, depth_scale):
    import pyrealsense2 as rs
    frames    = align.process(pipe.wait_for_frames(timeout_ms=3000))
    rgb       = np.asanyarray(frames.get_color_frame().get_data())
    depth_raw = np.asanyarray(frames.get_depth_frame().get_data())
    depth     = depth_raw.astype(np.float32) * depth_scale
    return rgb, depth


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


def segment(gdino, sam_pred, img_rgb, prompt, box_thr, text_thr, device, y_max_frac=None):
    import torchvision.transforms as T
    import torch
    from groundingdino.util.inference import predict
    from PIL import Image
    H, W = img_rgb.shape[:2]
    transform = T.Compose([
        T.Resize(800), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
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

    if y_max_frac is not None:
        y_center = (boxes_px[:, 1] + boxes_px[:, 3]) / 2
        keep = y_center < (y_max_frac * H)
        boxes_px = boxes_px[keep]

    if len(boxes_px) == 0:
        return np.zeros((H, W), dtype=np.uint8)

    sam_pred.set_image(img_rgb)
    mask_all = np.zeros((H, W), dtype=bool)
    for box in boxes_px:
        m, _, _ = sam_pred.predict(box=box.cpu().numpy(), multimask_output=False)
        mask_all |= m[0].astype(bool)
    return mask_all.astype(np.uint8)


def apply_mask(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = img_rgb.copy()
    out[mask.astype(bool)] = 0
    return out


# ════════════════════════════════════════════════════════════════════════════
# Safety: per-step pose clamping
# ════════════════════════════════════════════════════════════════════════════

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
# Main
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, help='Path to policy_final.pt')
    p.add_argument('--mesh', default=os.path.join(PIPELINE_DIR, 'spoon.obj'),
                   help='Tool mesh for FoundationPose (.obj)')
    p.add_argument('--tool_prompt', default='spoon',
                   help='GroundedSAM prompt to segment the tool on the first frame')
    p.add_argument('--unet_checkpoint', required=True,
                   help='UNet checkpoint for masking the robot arm out of the observation image')
    p.add_argument('--unet_threshold', type=float, default=0.5,
                   help='Sigmoid threshold for UNet arm mask')
    p.add_argument('--task_frame', default=None,
                   help='tf_world2cam from 00_calibrate.py. Defaults to '
                        'data/cam_extrinsics.npy for cam0, data/cam1_extrinsics.npy for cam1.')
    p.add_argument('--robot_extrinsics', default='data/robot_extrinsics.npy',
                   help='T_base_task from 06_calibrate_robot.py — used for converting '
                        'policy predictions (task frame) to robot commands (base frame).')
    p.add_argument('--robot_extrinsics_proprio', default=None,
                   help='Optional separate T_base_task for proprioception, calibrated with '
                        'joint-angle FK (06_calibrate_robot.py --use_joint_fk). If omitted, '
                        'falls back to --robot_extrinsics. Use this when GetCartesianPosition '
                        'and the joint-angle FK disagree on EEF height.')
    p.add_argument('--track_cam', type=int, default=0,
                   help='Camera index to use for FP tracking and policy observation (default: 0). '
                        'Must match --track_cam used in 00_calibrate.py, 04_track.py, 05_train.py.')
    p.add_argument('--camera', type=int, default=None,
                   help='Alias for --track_cam (deprecated)')
    p.add_argument('--box_threshold',  type=float, default=0.3)
    p.add_argument('--text_threshold', type=float, default=0.25)
    p.add_argument('--est_refine_iter',   type=int, default=5)
    p.add_argument('--track_refine_iter', type=int, default=2)
    p.add_argument('--tool_eef_cache', default='data/T_tool_eef.npy',
                   help='Cache for the rigid spoon->gripper offset T_tool_eef. Once calibrated, '
                        'this fixed mechanical offset is reused on every run regardless of '
                        'camera position, so camera recalibration no longer affects action '
                        'correctness. Delete the file or pass --recalibrate_tool_eef to redo it.')
    p.add_argument('--recalibrate_tool_eef', action='store_true',
                   help='Recompute T_tool_eef via FoundationPose registration even if a cached '
                        'value exists, and overwrite the cache.')
    p.add_argument('--T_eef_spoon', default='data/T_eef_spoon.npy',
                   help='EEF→spoon calibration from calib_viz_3d.py. When present, skips '
                        'per-step FoundationPose tracking entirely — FK + this offset drives '
                        'proprio and spoon viz. Delete to fall back to FP tracking.')
    p.add_argument('--frequency', type=float, default=10.0,
                   help='Control loop rate (Hz). Must match training: record_fps / subsample. '
                        'E.g. 30fps recording with --subsample 3 → 10 Hz; --subsample 2 → 15 Hz.')
    p.add_argument('--exec_steps', type=int, default=8,
                   help='Actions to execute per inference cycle (receding horizon). '
                        'Inference fires when 2 actions remain, giving 2/frequency seconds of runway.')
    p.add_argument('--max_steps', type=int, default=0,
                   help='Stop after this many control iterations (0 = run until Q)')
    p.add_argument('--max_trans_step', type=float, default=0.02,
                   help='Max per-command translation step (metres) — safety clamp')
    p.add_argument('--max_rot_step_deg', type=float, default=10.0,
                   help='Max per-command rotation step (degrees) — safety clamp')
    p.add_argument('--arm_trans_speed', type=float, default=0.0,
                   help='Cartesian translation speed limit sent to Kinova (m/s). '
                        '0 = robot default (slow). Try 0.15–0.25 for faster execution.')
    p.add_argument('--step_mode', action='store_true',
                   help='Step-by-step execution: send one action, wait for convergence, '
                        'then block until SPACE is pressed before sending the next. '
                        'Useful for manual inspection. Combine with --wait_convergence.')
    p.add_argument('--execute', action='store_true',
                   help='Actually send commands to the robot (default: dry run / print only)')
    p.add_argument('--wait_convergence', action='store_true',
                   help='After each command, poll GetCartesianPosition until the EEF reaches the '
                        'target (within --convergence_threshold) before sending the next command. '
                        'Replaces the fixed-frequency sleep. Requires --execute.')
    p.add_argument('--convergence_threshold', type=float, default=0.010,
                   help='EEF position threshold (metres) for --wait_convergence. Default: 1 cm.')
    p.add_argument('--convergence_timeout', type=float, default=0.5,
                   help='Max seconds to wait per step in --wait_convergence mode. Default: 0.5 s.')
    p.add_argument('--output_dir', default='/tmp/policy_deploy')
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def main():
    args = parse_args()
    import torch

    if args.camera is not None:
        args.track_cam = args.camera
    if args.task_frame is None:
        args.task_frame = ('data/cam_extrinsics.npy' if args.track_cam == 0
                           else f'data/cam{args.track_cam}_extrinsics.npy')

    os.makedirs(args.output_dir, exist_ok=True)

    # Create cv2 windows FIRST — OpenGL context must exist before CUDA/nvdiffrast.
    cv2.namedWindow("Deploy",      cv2.WINDOW_NORMAL)
    cv2.namedWindow("Policy View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Deploy",      848, 480)
    cv2.resizeWindow("Policy View", 848, 480)

    # ── Load calibration ────────────────────────────────────────────────────
    tf_world2cam = np.load(args.task_frame).astype(np.float64)
    T_base_task  = np.load(args.robot_extrinsics).astype(np.float64)
    tf_cam2world = np.linalg.inv(tf_world2cam)
    print(f"Loaded tf_world2cam from {args.task_frame}")
    print(f"Loaded T_base_task (commands) from {args.robot_extrinsics}")

    # Separate T_base_task for proprio — calibrated with joint-angle FK so that
    # the computed spoon pose matches the gold-mesh position in deploy_viz.
    _prop_path = args.robot_extrinsics_proprio or args.robot_extrinsics
    T_base_task_prop = np.load(_prop_path).astype(np.float64)
    _using_separate_prop = (args.robot_extrinsics_proprio is not None
                            and args.robot_extrinsics_proprio != args.robot_extrinsics)
    print(f"Loaded T_base_task (proprio)  from {_prop_path}"
          + ("  [joint-FK calibration]" if _using_separate_prop else "  [same as commands]"))

    # ── Load policy ──────────────────────────────────────────────────────────
    device = torch.device(args.device)
    print(f"Loading policy checkpoint: {args.checkpoint}")
    model, normalizer, ckpt = load_model(args.checkpoint, device)
    image_size   = ckpt.get('image_size', 128)
    crop_size    = ckpt.get('crop_size', 115)
    n_obs_steps  = ckpt.get('n_obs_steps', 2)
    n_views      = ckpt.get('n_views', 1)
    action_frame = ckpt.get('action_frame', 'task')
    noise_scheduler = ckpt['noise_scheduler']
    noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps)
    transform = make_transform(image_size, crop_size)
    print(f"Model: obs_steps={n_obs_steps}  n_views={n_views}  action_horizon={model.action_horizon}  "
          f"image={image_size}x{image_size} -> crop {crop_size}x{crop_size}  action_frame={action_frame}")

    # ── Load UNet arm segmentor ──────────────────────────────────────────────
    print(f"Loading UNet arm segmentor: {args.unet_checkpoint}")
    unet = load_unet(args.unet_checkpoint, device)
    print(f"  threshold={args.unet_threshold}")

    # ── Load mesh for to_origin (always needed for proprio + action decoding) ───
    import trimesh
    print(f"Loading mesh: {args.mesh}")
    loaded = trimesh.load(args.mesh)
    mesh = (trimesh.util.concatenate(list(loaded.geometry.values()))
            if isinstance(loaded, trimesh.Scene) else loaded)
    if mesh.bounding_box.extents.max() > 0.5:
        print(f"  Mesh extents suggest cm units — rescaling x0.01 to metres")
        mesh.apply_scale(0.01)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    inv_to_origin = np.linalg.inv(to_origin)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    # ── T_eef_spoon: FK-based proprio (no per-step FP needed) ────────────────
    eef_spoon_path = os.path.join(PIPELINE_DIR, args.T_eef_spoon)
    T_eef_spoon = None
    if os.path.exists(eef_spoon_path):
        T_eef_spoon = np.load(eef_spoon_path).astype(np.float64)
        # T_base_tool = T_base_eef @ T_eef_spoon @ to_origin  (matches training convention)
        # T_tool_eef  = inv_to_origin @ inv(T_eef_spoon)      (for action → EEF)
        T_tool_eef_from_calib = inv_to_origin @ np.linalg.inv(T_eef_spoon)
        t = T_eef_spoon[:3, 3]
        print(f"Loaded T_eef_spoon from {eef_spoon_path} "
              f"(t={t[0]*100:.1f},{t[1]*100:.1f},{t[2]*100:.1f} cm)")
        print("  → FK-based proprio mode: FoundationPose NOT used per-step.")

    # ── Load GDino + SAM + FP only when T_eef_spoon absent or T_tool_eef missing ─
    # Only load FP when we can't derive T_tool_eef from T_eef_spoon
    need_fp = T_eef_spoon is None
    need_registration = (not os.path.exists(args.tool_eef_cache)
                         or args.recalibrate_tool_eef)

    est = gdino = sam_pred = None
    if need_fp:
        print("Loading GroundedSAM + FoundationPose (T_eef_spoon not found or T_tool_eef cache missing)...")
        gdino, sam_pred = load_gdino_sam(args.device)
        from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
        import nvdiffrast.torch as dr
        scorer  = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx   = dr.RasterizeCudaContext()
        os.makedirs('/tmp/fp_deploy_debug', exist_ok=True)
        est = FoundationPose(
            model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
            scorer=scorer, refiner=refiner, glctx=glctx,
            debug_dir='/tmp/fp_deploy_debug', debug=0,
        )
        print("FoundationPose ready.")
    try:
        from Utils import draw_posed_3d_box
    except ImportError:
        def draw_posed_3d_box(K, img, **kw): return img

    # ── Start camera(s) ──────────────────────────────────────────────────────
    pipe, align, K, depth_scale = start_realsense(args.track_cam)
    pipe_other = align_other = K_other = depth_scale_other = None
    if n_views > 1:
        other_cam = 0 if args.track_cam != 0 else 1
        print(f"Dual-cam mode: also starting cam{other_cam} ...")
        pipe_other, align_other, K_other, depth_scale_other = start_realsense(other_cam)
        print(f"  cam{other_cam} ready.")

    # ── Connect robot ────────────────────────────────────────────────────────
    api = connect()

    if not args.execute:
        print("\n*** DRY RUN — no commands will be sent to the robot. Use --execute to enable. ***\n")

    try:
        # ── T_tool_eef: for converting predicted tool poses → EEF commands ──────
        if T_eef_spoon is not None:
            # Derived analytically from calib_viz_3d.py calibration — no FP needed
            T_tool_eef = T_tool_eef_from_calib
            print(f"T_tool_eef derived from T_eef_spoon (no FP registration needed).")

            # Still run FP registration once if we need to seed est.track_one for
            # the fallback visualization path (only when est was loaded).
            if est is not None:
                print(f"\nRegistering tool ('{args.tool_prompt}') for FP viz only...")
                rgb, depth = capture(pipe, align, depth_scale)
                mask = segment(gdino, sam_pred, rgb, args.tool_prompt,
                               args.box_threshold, args.text_threshold, args.device)
                if mask.sum() >= 100:
                    est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask,
                                 iteration=args.est_refine_iter)
                    print("  FP registered (viz only).")
        else:
            # Fall back to FP-based T_tool_eef (original behaviour)
            print(f"\nRegistering tool ('{args.tool_prompt}') for FP tracking...")
            rgb, depth = capture(pipe, align, depth_scale)
            mask = segment(gdino, sam_pred, rgb, args.tool_prompt,
                           args.box_threshold, args.text_threshold, args.device)
            if mask.sum() < 100:
                raise RuntimeError("Tool segmentation failed — adjust --tool_prompt "
                                   "or camera framing and retry.")
            pose_cam = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask,
                                     iteration=args.est_refine_iter)
            print("  FP registration done.")

            if need_registration:
                T_task_tool0 = tf_cam2world @ pose_cam.astype(np.float64)
                eef_pose0    = get_cartesian_pose(api)
                T_base_eef0  = kinova_pose_to_matrix(eef_pose0)
                T_base_tool0 = T_base_task @ T_task_tool0
                T_tool_eef   = np.linalg.inv(T_base_tool0) @ T_base_eef0
                os.makedirs(os.path.dirname(args.tool_eef_cache) or '.', exist_ok=True)
                np.save(args.tool_eef_cache, T_tool_eef)
                print(f"Calibrated and cached T_tool_eef -> {args.tool_eef_cache}")
            else:
                T_tool_eef = np.load(args.tool_eef_cache).astype(np.float64)
                print(f"Loaded cached T_tool_eef from {args.tool_eef_cache}")

        # ── Receding-horizon control loop ────────────────────────────────────
        # Design: one outer iteration = one control step (dt = 1/frequency).
        # obs_buffer is a rolling deque updated every step — so both obs frames
        # passed to inference are captured during robot motion, not while stopped.
        # Inference runs in a background thread kicked off when the action deque
        # drops to exec_steps//2 remaining, so new actions are ready before the
        # deque empties (zero-pause receding horizon).
        dt           = 1.0 / args.frequency
        max_rot_step = np.radians(args.max_rot_step_deg)

        # Rolling obs buffer: always the last n_obs_steps frames
        obs_buffer = collections.deque(maxlen=n_obs_steps)

        # Action deque: predicted tool poses (base frame) consumed 1-per-step
        action_deque = collections.deque()

        # Async inference state (thread-safe via lock)
        _infer_lock    = threading.Lock()
        _infer_running = [False]
        _infer_result  = [None]   # list of (pred_horizon,4,4) poses once ready

        def _infer_thread(buf_snap):
            view_tensors = [v[0] for v in buf_snap]
            proprio_list = [v[1] for v in buf_snap]
            _, poses = predict_action_sequence(
                model, normalizer, noise_scheduler, view_tensors, proprio_list, device)
            with _infer_lock:
                _infer_result[0]  = poses     # full pred_horizon poses
                _infer_running[0] = False

        # T_base_eef_cur tracks the last clamped command (for the clamp-step check).
        T_base_eef_cur = kinova_pose_to_matrix(get_cartesian_pose(api))

        # Last known set of predicted poses for visualisation (updated on each inference)
        latest_poses_base = None

        step = 0
        paused = False
        _fps_t0    = time.time()
        _fps_count = 0
        _fps_disp  = 0.0
        print("\nStarting control loop (rolling-obs async-inference).")
        print("P = pause/resume | Q = quit\n")
        prev_cmd_xyz_base = None
        cmd_xyz_base = None
        while True:
            t_loop_start = time.time()
            prev_cmd_xyz_base = cmd_xyz_base  # save last iteration's command before resetting
            cmd_xyz_base = None  # will be set when a command is sent this iteration

            # ── Key handling (always, even when paused) ───────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQ pressed — stopping.")
                break
            if key == ord('p'):
                paused = not paused
                if paused:
                    action_deque.clear()
                    if args.execute:
                        api.EraseAllTrajectories()
                    print(f"\n{'─'*50}")
                    print("  *** PAUSED — arm holding position ***")
                    print("  Press P to resume.")
                    print(f"{'─'*50}")
                else:
                    obs_buffer.clear()        # flush stale obs so next infer uses fresh frames
                    latest_poses_base = None
                    print("\n  *** RESUMED ***\n")

            # ── 1. Capture + get tool pose ────────────────────────────────────
            rgb, depth = capture(pipe, align, depth_scale)
            q_deg = get_joint_angles_deg(api)

            if T_eef_spoon is not None:
                # Joint-angle FK for proprio — matches the gold-mesh position in
                # deploy_viz and the actual physical spoon, unlike GetCartesianPosition
                # which can have a systematic Z offset on the Kinova Jaco2.
                T_base_eef_fk = joint_angles_to_eef(q_deg)
                T_base_tool   = T_base_eef_fk @ T_eef_spoon @ to_origin
                pose_cam      = None
            else:
                # Original FP-based tracking
                pose_cam    = est.track_one(rgb=rgb, depth=depth, K=K,
                                            iteration=args.track_refine_iter)
                T_base_tool = T_base_task_prop @ tf_cam2world @ pose_cam.astype(np.float64)
            # Use T_base_task_prop (joint-FK calibration) so that task-frame proprio
            # matches the FoundationPose-tracked training observations.
            T_task_tool = np.linalg.inv(T_base_task_prop) @ T_base_tool

            # ── 2. Mask + encode obs (both cameras if dual-cam) ──────────────
            robot_mask = unet_mask(unet, rgb, args.unet_threshold, device)
            masked_rgb = apply_mask(rgb, robot_mask)
            if not paused:
                cv2.imwrite(os.path.join(args.output_dir, f'{step:06d}_masked.jpg'),
                            cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2BGR))

            if pipe_other is not None:
                rgb_other, _     = capture(pipe_other, align_other, depth_scale_other)
                other_mask       = unet_mask(unet, rgb_other, args.unet_threshold, device)
                masked_rgb_other = apply_mask(rgb_other, other_mask)
            else:
                masked_rgb_other = None

            from PIL import Image
            img_t_main = transform(Image.fromarray(masked_rgb))
            if masked_rgb_other is not None:
                img_t = torch.stack([img_t_main,
                                     transform(Image.fromarray(masked_rgb_other))])
            else:
                img_t = img_t_main.unsqueeze(0)

            # Proprio in the frame the policy was trained on (from checkpoint)
            T_proprio = T_task_tool if action_frame == 'task' else T_base_tool
            proprio = normalizer.normalize(pose_matrix_to_9d(T_proprio))
            obs_buffer.append((img_t, proprio))   # rolling — always kept fresh

            T_base_eef_now = kinova_pose_to_matrix(get_cartesian_pose(api))

            if not paused:
                # ── 3. Collect completed inference result ─────────────────────
                with _infer_lock:
                    if _infer_result[0] is not None:
                        latest_poses_base = _infer_result[0]
                        action_deque.clear()
                        action_deque.extend(latest_poses_base[:args.exec_steps])
                        _infer_result[0] = None

                # ── 4. Kick off next inference when deque is low ──────────────
                with _infer_lock:
                    kick = (len(obs_buffer) == n_obs_steps
                            and not _infer_running[0]
                            and len(action_deque) <= 2)
                    if kick:
                        _infer_running[0] = True

                if kick:
                    buf_snap = list(obs_buffer)
                    threading.Thread(target=_infer_thread, args=(buf_snap,),
                                     daemon=True).start()

                # ── 5. Execute one action from deque ──────────────────────────
                _action_sent = False
                if action_deque:
                    T_pred = action_deque.popleft().astype(np.float64)
                    if action_frame == 'task':
                        T_base_tool_pred = T_base_task @ T_pred
                    else:
                        T_base_tool_pred = T_pred
                    T_base_eef_pred    = T_base_tool_pred @ T_tool_eef
                    T_base_eef_clamped = clamp_pose_step(
                        T_base_eef_cur, T_base_eef_pred,
                        args.max_trans_step, max_rot_step)
                    T_base_eef_cur = T_base_eef_clamped
                    t_pred  = T_base_eef_pred[:3, 3]
                    t_clamp = T_base_eef_clamped[:3, 3]
                    print(f"  step {step} [{len(action_deque)} remain]: "
                          f"eef pred=({t_pred[0]*100:.1f},{t_pred[1]*100:.1f},{t_pred[2]*100:.1f}) cm "
                          f"clamped=({t_clamp[0]*100:.1f},{t_clamp[1]*100:.1f},{t_clamp[2]*100:.1f}) cm")
                    if args.execute:
                        send_cartesian_pose(api, matrix_to_kinova_pose(T_base_eef_clamped),
                                            trans_speed=args.arm_trans_speed)
                        cmd_xyz_base = T_base_eef_clamped[:3, 3].copy()
                    _action_sent = True
                else:
                    print(f"  step {step}: waiting for inference...")

            # ── 6. Write state for 3D plotter (deploy_viz.py reads this) ────────
            if not paused:
                if latest_poses_base is not None:
                    poses_eef_pred  = np.stack([p.astype(np.float64) @ T_tool_eef
                                                 for p in latest_poses_base])
                    poses_tool_pred = np.stack([p.astype(np.float64)
                                                 for p in latest_poses_base])
                else:
                    poses_eef_pred  = np.zeros((0, 4, 4))
                    poses_tool_pred = np.zeros((0, 4, 4))
                np.savez('/tmp/deploy_state_tmp.npz',
                         q_deg=q_deg,
                         T_base_eef=T_base_eef_now,
                         T_base_tool=T_base_tool,
                         poses_tool_pred=poses_tool_pred,
                         poses_eef_pred=poses_eef_pred,
                         step=np.array([step]))
                os.replace('/tmp/deploy_state_tmp.npz', '/tmp/deploy_state.npz')

            # ── 7. Visualisation ─────────────────────────────────────────────
            # T_base_tool is already in OBB frame; bbox corners are also in OBB frame.
            # Project to camera using inv(T_base_task) to go base→task, then tf_world2cam task→cam.
            # Apply inv_to_origin to match the mesh frame used by predicted-action axes
            ob_in_cam = tf_world2cam @ np.linalg.inv(T_base_task) @ T_base_tool @ inv_to_origin
            vis = rgb.copy()
            vis = draw_axes_simple(vis, ob_in_cam, K, scale=0.10)
            center_pose = ob_in_cam

            if robot_mask.sum() > 0:
                red = np.zeros_like(vis)
                red[:, :, 0] = 255
                m = robot_mask.astype(bool)
                vis[m] = cv2.addWeighted(vis, 0.5, red, 0.5, 0)[m]

            if latest_poses_base is not None:
                if action_frame == 'task':
                    # policy output is already in task/world frame → directly to cam
                    cam_poses_pred = [tf_world2cam @ p.astype(np.float64) @ inv_to_origin
                                      for p in latest_poses_base]
                else:
                    cam_poses_pred = [tf_world2cam @ np.linalg.inv(T_base_task)
                                      @ p.astype(np.float64) @ inv_to_origin
                                      for p in latest_poses_base]
                vis = draw_axes_with_horizon(vis, cam_poses_pred, K, scale=0.04)

            t_task = T_task_tool[:3, 3]
            _fps_count += 1
            _fps_elapsed = time.time() - _fps_t0
            if _fps_elapsed >= 1.0:
                _fps_disp  = _fps_count / _fps_elapsed
                _fps_count = 0
                _fps_t0    = time.time()
            cv2.putText(vis,
                        f"step {step}  {len(action_deque)} queued  "
                        f"{'INFER' if _infer_running[0] else ''}  "
                        f"{'EXECUTE' if args.execute else 'DRY RUN'}  {_fps_disp:.1f} Hz",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
            cv2.putText(vis, f"proprio(task) xyz= {t_task[0]*100:.1f},{t_task[1]*100:.1f},{t_task[2]*100:.1f} cm"
                            f"  train=[{-1.8:.1f}~{19.8:.1f}, {-32.2:.1f}~{-0.7:.1f}, {7.5:.1f}~{20.8:.1f}]",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            if latest_poses_base is not None:
                p0 = latest_poses_base[0]
                cv2.putText(vis, f"pred[0](task) xyz= {p0[0,3]*100:.1f},{p0[1,3]*100:.1f},{p0[2,3]*100:.1f} cm",
                            (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
            # Console: print proprio vs pred every 10 steps for comparison
            if step % 10 == 0:
                from scipy.spatial.transform import Rotation as _R
                r_task = _R.from_matrix(T_task_tool[:3,:3]).as_euler('xyz', degrees=True)
                print(f"  [diag] proprio T_task_tool: xyz=({t_task[0]*100:.1f},{t_task[1]*100:.1f},{t_task[2]*100:.1f})cm  "
                      f"euler=({r_task[0]:.1f},{r_task[1]:.1f},{r_task[2]:.1f})deg")
                print(f"         train pos ranges: x[-1.8,19.8] y[-32.2,-0.7] z[7.5,20.8] cm")
                if latest_poses_base is not None:
                    p0 = latest_poses_base[0]
                    r_pred = _R.from_matrix(p0[:3,:3]).as_euler('xyz', degrees=True)
                    print(f"         pred[0] T_task_tool: xyz=({p0[0,3]*100:.1f},{p0[1,3]*100:.1f},{p0[2,3]*100:.1f})cm  "
                          f"euler=({r_pred[0]:.1f},{r_pred[1]:.1f},{r_pred[2]:.1f})deg")
            if paused:
                h, w = vis.shape[:2]
                overlay = vis.copy()
                cv2.rectangle(overlay, (0, h//2 - 40), (w, h//2 + 40), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)
                cv2.putText(vis, "PAUSED  (press P to resume)",
                            (w//2 - 220, h//2 + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 80, 255), 3)

            # Policy view: dual-cam → show cam0 | cam1; single-cam → show full | crop
            pv_left  = cv2.resize(cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2BGR), (424, 480))
            if masked_rgb_other is not None:
                pv_right = cv2.resize(cv2.cvtColor(masked_rgb_other, cv2.COLOR_RGB2BGR), (424, 480))
                cv2.putText(pv_left,  f"cam{args.track_cam}",  (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(pv_right, f"cam{other_cam}",       (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                crop_np  = cv2.resize(cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2BGR),
                                      (image_size, image_size))
                pad      = (image_size - crop_size) // 2
                crop_np  = crop_np[pad:pad+crop_size, pad:pad+crop_size]
                pv_right = cv2.resize(crop_np, (424, 480), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Policy View", np.concatenate([pv_left, pv_right], axis=1))
            cv2.imshow("Deploy", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            if not paused:
                cv2.imwrite(os.path.join(args.output_dir, f'{step:06d}.jpg'),
                            cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

            # ── 8. Wait for arm convergence or sleep to maintain control frequency ──
            if args.wait_convergence and cmd_xyz_base is not None and args.execute:
                # Scale timeout with commanded move distance: allow ~arm_speed cm/s
                # to actually reach the target, with a floor of convergence_timeout.
                cmd_dist = np.linalg.norm(cmd_xyz_base - prev_cmd_xyz_base) \
                    if prev_cmd_xyz_base is not None else 0.0
                # Use commanded speed if set, otherwise fall back to observed ~3 cm/s default
                arm_speed = args.arm_trans_speed if args.arm_trans_speed > 0 else 0.03
                adaptive_timeout = max(args.convergence_timeout,
                                       cmd_dist / arm_speed + 0.2)
                t_wait = time.time()
                converged = False
                while time.time() - t_wait < adaptive_timeout:
                    actual_xyz = get_cartesian_pose(api)[:3]
                    err = np.linalg.norm(actual_xyz - cmd_xyz_base)
                    if err < args.convergence_threshold:
                        converged = True
                        break
                    time.sleep(0.02)
                else:
                    actual_xyz = get_cartesian_pose(api)[:3]
                    err = np.linalg.norm(actual_xyz - cmd_xyz_base)
                t_conv = time.time() - t_wait
                status = "OK" if converged else "TIMEOUT"
                print(f"    convergence [{status}] err={err*100:.1f}cm "
                      f"in {t_conv*1000:.0f}ms (timeout={adaptive_timeout*1000:.0f}ms)")
            else:
                elapsed = time.time() - t_loop_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)

            # ── 9. Step-mode gate: block until SPACE before next action ──────────
            if args.step_mode and not paused and _action_sent:
                _vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                h_v, w_v = _vis_bgr.shape[:2]
                _bar = _vis_bgr.copy()
                cv2.rectangle(_bar, (0, h_v - 58), (w_v, h_v), (0, 0, 0), -1)
                cv2.addWeighted(_bar, 0.65, _vis_bgr, 0.35, 0, _vis_bgr)
                cv2.putText(_vis_bgr,
                            "STEP MODE  |  SPACE = next action   P = pause   Q = quit",
                            (10, h_v - 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.62, (50, 255, 120), 2)
                cv2.imshow("Deploy", _vis_bgr)
                print("  [STEP] Waiting for SPACE ...", end='', flush=True)
                _quit_step = False
                while True:
                    _k = cv2.waitKey(20) & 0xFF
                    if _k == ord('q'):
                        _quit_step = True
                        break
                    if _k == ord('p'):
                        paused = True
                        action_deque.clear()
                        if args.execute:
                            api.EraseAllTrajectories()
                        print('\n  *** PAUSED — press P to resume ***', flush=True)
                        break
                    if _k == ord(' '):
                        print(' GO.', flush=True)
                        break
                if _quit_step:
                    print("\nQ pressed — stopping.")
                    break

            if not paused:
                step += 1
                if args.max_steps and step >= args.max_steps:
                    print(f"\nReached --max_steps={args.max_steps} — stopping.")
                    break

    finally:
        if args.execute:
            api.EraseAllTrajectories()
        api.CloseAPI()
        pipe.stop()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == '__main__':
    main()
