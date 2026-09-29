"""
Step 7 variant — like 07_deploy_force.py (hybrid position+force admittance
control), but the admittance correction uses the FULL 6D WRENCH
[Fx,Fy,Fz,Mx,My,Mz] instead of just the 3D force [Fx,Fy,Fz]. Does NOT modify
07_deploy_force.py or 07_deploy.py -- this is a full standalone copy with the
rotational-admittance machinery layered in on top.

WHY A COPY, NOT A FLAG ON 07_deploy_force.py: same reasoning documented in
07_deploy_force.py's own docstring for why IT is a copy of 07_deploy.py --
keep the already-validated, hardware-tested translation-only hybrid path
completely untouched for future evaluation runs, and iterate on the new
(EXPERIMENTAL, not yet hardware-validated) rotational term here.

BACKGROUND -- why moments were being discarded: contact_detector.
torque_to_wrench(q, tau) already returns the FULL 6D wrench (via
pinv(J^T) @ tau, or the damped-least-squares equivalent) -- it was never
computationally expensive to get Mx,My,Mz, they were simply sliced off with
`[:3]` in 07_deploy_force.py's read_F_raw and never used again. Likewise
analyze_replay_full.py already saves external_moment_xyz alongside
external_force_xyz in replay_full_forces.npz, but 05_train_replay_force.py
only ever loads external_force_xyz (FORCE_DIM=3) -- no checkpoint in this
project predicts a future moment. See FORCE_AWARE_POLICY_REPORT.txt section
on "the full 6D wrench" for the fuller writeup.

WHAT'S NEW HERE:
  1. read_F_raw returns the full 6D wrench (drops the old `[:3]` slice) and
     is renamed read_wrench_raw.
  2. The live force pipeline (tare + causal Butterworth filter) runs on all
     6 channels instead of 3.
  3. A NEW rotational admittance term, gated by --admittance_moments
     (default OFF, requires --admittance):

         rot_correction = clip_magnitude((M_live - m_desired) / K_rot, max_rad)
         T_base_eef_pred[:3,:3] = Exp(rot_correction) @ T_base_eef_pred[:3,:3]

     applied BEFORE clamp_pose_step, exactly mirroring the translation
     term's ordering -- so the existing --max_rot_step_deg safety clamp
     still bounds the final commanded rotation step regardless of what this
     correction computes, the same "cannot bypass the safeguard by
     construction" argument 07_deploy_force.py's docstring makes for
     translation.

  IMPORTANT LIMITATION, stated plainly: unlike the translation term (whose
  f_desired comes from the policy's own predicted future force -- a
  time-varying, learned reference), there is no learned m_desired anywhere
  in this project -- no checkpoint predicts moments. m_desired is therefore
  FIXED AT ZERO (--m_desired, default [0,0,0]): this regulates the live
  moment toward zero ("don't twist/bind") rather than tracking any
  demonstration-derived moment profile. This is a materially different, and
  weaker, kind of reference than the translation term gets. Treat this as
  an experimental "resist unwanted torque buildup" damper, not a learned
  rotational admittance in the same sense as the translation term.

  This is why --admittance_moments defaults OFF and is a separate flag from
  --admittance: the translation term is hardware-validated (see
  POLICY_EXPERIMENTS.md's 7-trial comparison); this rotational term is NOT
  -- do a dry run, then a short/conservative real test, before trusting it.

Usage (dry run, translation-only hybrid, same as 07_deploy_force.py):
    python 07_deploy_force_6d.py \\
        --checkpoint data/checkpoints/PastaTransfer_force_replay_force_ep1-30/policy_epoch0100.pt \\
        --no_arm_mask \\
        --robot_extrinsics data/robot_extrinsics_stick_corrected_zmeasured.npy \\
        --robot_extrinsics_proprio data/robot_extrinsics_stick_corrected_zmeasured.npy \\
        --mesh newspoon1.obj --tool_prompt "spoon" --track_cam 1 \\
        --init_episode_dir data/episodes/PastaTransfer_force/021 --init_frame 0 \\
        --max_trans_step 0.04 --wait_convergence --admittance

Usage (dry run, full 6D wrench admittance -- translation + rotation):
    ... --admittance --admittance_moments --K 200 --K_rot 50 --wait_convergence

Usage (real motion, full 6D):
    ... --admittance --admittance_moments --wait_convergence --execute
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
sys.path.insert(0, PIPELINE_DIR)
from calibrate_firmware_gravity import apply_saved_gravity_params

from policy_common import pose_matrix_to_9d
from test_policy import (load_model, make_transform, predict_action_sequence,
                         predict_action_sequence_flow, predict_action_sequence_act,
                         _project, draw_axes_with_horizon, draw_axes_simple)
from contact_detector import (torque_to_wrench, gravity_regressor, compute_jacobian,
                              full_dynamics_regressor, VelocityDifferentiator)
from fit_gravity_residual_nn import GravityResidualNet, predict as predict_gravity_residual
from scipy.signal import butter, sosfilt, sosfilt_zi
from scipy.spatial.transform import Rotation

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
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    T[:3,  3] = xyz
    return T


def joint_angles_to_eef(q_deg: np.ndarray) -> np.ndarray:
    q = np.deg2rad(q_deg)
    T = np.eye(4, dtype=np.float64)
    for (xyz, rpy), qi in zip(_FK_JOINT_PARAMS, q):
        Tj = np.eye(4, dtype=np.float64)
        Tj[:3, :3] = Rotation.from_euler('z', float(qi)).as_matrix()
        T = T @ _make_fk_T(xyz, rpy) @ Tj
    return T @ _make_fk_T(*_FK_EEF_PARAMS)


# ════════════════════════════════════════════════════════════════════════════
# Kinova Jaco2 USB SDK bindings (CARTESIAN_POSITION trajectory control +
# gravity-free torque read, needed for the live force pipeline)
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
        ("GetAngularForceGravityFree", ctypes.c_int),   # live force pipeline
        ("GetAngularVelocity",   ctypes.c_int),          # for qddot estimation -- mass/Coriolis correction
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
    grav_ok = apply_saved_gravity_params(api)
    print(f'Firmware gravity params (data/gravity_params.npy) reapplied: '
          f'{"OK" if grav_ok else "not applied — see message above"}')
    return api


def get_joint_angles_deg(api) -> np.ndarray:
    pos = AngularPosition()
    api.GetAngularPosition(ctypes.byref(pos))
    a = pos.Actuators
    return np.array([a.Actuator1, a.Actuator2, a.Actuator3,
                     a.Actuator4, a.Actuator5, a.Actuator6], dtype=np.float64)


def get_qdot_deg(api) -> np.ndarray:
    """Joint angular velocity (deg/s) -- needed to estimate qddot for the
    mass/Coriolis dynamics regressor correction (contact_detector.
    full_dynamics_regressor), matching analyze_replay_full.py's training-label
    pipeline."""
    pos = AngularPosition()
    api.GetAngularVelocity(ctypes.byref(pos))
    a = pos.Actuators
    return np.array([a.Actuator1, a.Actuator2, a.Actuator3,
                     a.Actuator4, a.Actuator5, a.Actuator6], dtype=np.float64)


def get_tau_gf(api) -> np.ndarray:
    """Firmware gravity-free joint torque (Nm) -- same reader as
    deploy_streaming.py/calibrate_firmware_gravity.py."""
    pos = AngularPosition()
    api.GetAngularForceGravityFree(ctypes.byref(pos))
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
        tp.Limitations.speedParameter1 = float(trans_speed)
        tp.Limitations.speedParameter2 = float(rot_speed)
    api.SendBasicTrajectory(tp)


def kinova_pose_to_matrix(xyz_theta: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_euler('XYZ', xyz_theta[3:]).as_matrix()
    T[:3, 3]  = xyz_theta[:3]
    return T


def matrix_to_kinova_pose(T: np.ndarray) -> np.ndarray:
    theta = Rotation.from_matrix(T[:3, :3]).as_euler('XYZ')
    return np.concatenate([T[:3, 3], theta])


# ════════════════════════════════════════════════════════════════════════════
# Live force pipeline (full gravity+mass/Coriolis chain, FULL 6D WRENCH)
# ════════════════════════════════════════════════════════════════════════════

class OnlineButterworth:
    """Causal 2nd-order low-pass with persistent state -- duplicated from
    deploy_streaming.py/07_deploy_force.py rather than imported, same
    reasoning: each deployment script is meant to be readable/runnable
    standalone. One instance per scalar channel."""
    def __init__(self, cutoff_hz: float, fs_hz: float, order: int = 2):
        self.sos = butter(order, cutoff_hz, btype='low', fs=fs_hz, output='sos')
        self._zi = None

    def update(self, x: float) -> float:
        if self._zi is None:
            self._zi = sosfilt_zi(self.sos) * x
        y, self._zi = sosfilt(self.sos, [x], zi=self._zi)
        return float(y[0])


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
# Safety: per-step pose clamping (unchanged from 07_deploy.py/07_deploy_force.py)
# ════════════════════════════════════════════════════════════════════════════

def clamp_pose_step(T_current: np.ndarray, T_target: np.ndarray,
                     max_trans: float, max_rot_rad: float) -> np.ndarray:
    from scipy.spatial.transform import Slerp

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
    p.add_argument('--unet_checkpoint', default=None,
                   help='UNet checkpoint for masking the robot arm out of the observation image. '
                        'Required unless --no_arm_mask is passed.')
    p.add_argument('--unet_threshold', type=float, default=0.5,
                   help='Sigmoid threshold for UNet arm mask')
    p.add_argument('--no_arm_mask', action='store_true',
                   help='Feed the RAW (unmasked) camera image to the policy. Use this for '
                        'replay-image-trained checkpoints (05_train_replay.py/'
                        '05_train_replay_force.py), which is what this script is meant for.')
    p.add_argument('--task_frame', default=None,
                   help='tf_world2cam from 00_calibrate.py. Defaults to '
                        'data/cam_extrinsics.npy for cam0, data/cam1_extrinsics.npy for cam1.')
    p.add_argument('--robot_extrinsics', default='data/robot_extrinsics.npy',
                   help='T_base_task from 06_calibrate_robot.py.')
    p.add_argument('--robot_extrinsics_proprio', default=None,
                   help='Optional separate T_base_task for proprioception (joint-FK calibration).')
    p.add_argument('--track_cam', type=int, default=0)
    p.add_argument('--camera', type=int, default=None, help='Alias for --track_cam (deprecated)')
    p.add_argument('--box_threshold',  type=float, default=0.3)
    p.add_argument('--text_threshold', type=float, default=0.25)
    p.add_argument('--est_refine_iter',   type=int, default=5)
    p.add_argument('--track_refine_iter', type=int, default=2)
    p.add_argument('--tool_eef_cache', default='data/T_tool_eef.npy')
    p.add_argument('--recalibrate_tool_eef', action='store_true')
    p.add_argument('--T_eef_spoon', default='data/T_eef_spoon.npy')
    p.add_argument('--init_episode_dir', default=None)
    p.add_argument('--init_frame', type=int, default=0)
    p.add_argument('--frequency', type=float, default=10.0,
                   help='Control loop rate (Hz). Also the live force filter\'s sample rate.')
    p.add_argument('--exec_steps', type=int, default=8)
    p.add_argument('--max_steps', type=int, default=0)
    p.add_argument('--max_trans_step', type=float, default=0.02)
    p.add_argument('--max_rot_step_deg', type=float, default=10.0)
    p.add_argument('--arm_trans_speed', type=float, default=0.0)
    p.add_argument('--step_mode', action='store_true')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--wait_convergence', action='store_true')
    p.add_argument('--convergence_threshold', type=float, default=0.010)
    p.add_argument('--convergence_timeout', type=float, default=0.5)
    p.add_argument('--output_dir', default='/tmp/policy_deploy_force_6d')
    p.add_argument('--device', default='cuda')
    # ── Hybrid position+force (translation -- same as 07_deploy_force.py) ──────
    p.add_argument('--admittance', action='store_true',
                   help='Enable the translational force-admittance correction (requires a '
                        'predicts_force checkpoint). Without this: pure position control, both '
                        'corrections always 0 -- lets the same script/checkpoint serve as a '
                        'position-only baseline.')
    p.add_argument('--K', type=float, default=200.0, help='Admittance stiffness, N/m.')
    p.add_argument('--f_max_correction_cm', type=float, default=2.0,
                   help='Cap on the translational admittance correction magnitude (cm), '
                        'vector-norm-capped (not per-axis).')
    p.add_argument('--force_cutoff_hz', type=float, default=None,
                   help='Live force/moment low-pass cutoff (Hz). Defaults to the checkpoint\'s '
                        'own force_cutoff_hz (how its training labels were filtered) if present, '
                        'else 2.0.')
    p.add_argument('--tare_duration', type=float, default=1.0,
                   help='Seconds of wrench samples to average at startup as the resting-bias tare.')
    p.add_argument('--force_safety_threshold', type=float, default=15.0,
                   help='If live |F| exceeds this (N), zero the TRANSLATIONAL correction this '
                        'step (not the whole command) rather than trust it.')
    p.add_argument('--damping', type=float, default=0.05,
                   help='Tikhonov damping for the torque->wrench pinv (contact_detector.py). '
                        'Applies to the whole 6D solve, force and moment rows alike.')
    p.add_argument('--dynamics_pi', default='data/dynamics_residual_pi.npy',
                   help='Mass/Coriolis regressor parameters (contact_detector.'
                        'full_dynamics_regressor), same file analyze_replay_full.py used to '
                        'build the training-label force -- keeps live force on the same '
                        'correction chain as what the model was trained on.')
    p.add_argument('--qddot_smoothing', type=float, default=0.3,
                   help='EMA smoothing for the live qddot estimate (contact_detector.'
                        'VelocityDifferentiator) -- matches analyze_replay_full.py\'s default.')
    # ── NEW: rotational admittance from the full 6D wrench ─────────────────────
    p.add_argument('--admittance_moments', action='store_true',
                   help='EXPERIMENTAL, not yet hardware-validated. Enable a rotational '
                        'admittance correction from the live moment [Mx,My,Mz]. Requires '
                        '--admittance. Unlike the translation term, there is no learned '
                        'm_desired anywhere in this project (no checkpoint predicts moments) -- '
                        'this regulates the live moment toward --m_desired (default zero), not '
                        'a demonstration-derived reference. See module docstring.')
    p.add_argument('--K_rot', type=float, default=50.0,
                   help='Rotational admittance stiffness, N*m/rad.')
    p.add_argument('--m_max_correction_deg', type=float, default=3.0,
                   help='Cap on the rotational admittance correction magnitude (degrees), '
                        'vector-norm-capped -- kept small/conservative since this term is '
                        'experimental.')
    p.add_argument('--m_desired', type=float, nargs=3, default=[0.0, 0.0, 0.0],
                   help='Fixed target moment [Mx,My,Mz] in N*m, base frame (default: zero -- '
                        '"resist twisting", not a learned reference).')
    p.add_argument('--moment_safety_threshold', type=float, default=5.0,
                   help='If live |M| exceeds this (N*m), zero the ROTATIONAL correction this '
                        'step (not the whole command, not the translational correction).')
    return p.parse_args()


def main():
    args = parse_args()
    import torch

    if not args.no_arm_mask and args.unet_checkpoint is None:
        raise SystemExit('--unet_checkpoint is required unless --no_arm_mask is passed.')

    if args.camera is not None:
        args.track_cam = args.camera
    if args.task_frame is None:
        args.task_frame = ('data/cam_extrinsics.npy' if args.track_cam == 0
                           else f'data/cam{args.track_cam}_extrinsics.npy')

    if args.admittance_moments and not args.admittance:
        raise SystemExit('--admittance_moments requires --admittance (rotational admittance is '
                         'layered on top of the translational term, not a standalone mode).')

    os.makedirs(args.output_dir, exist_ok=True)

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
    train_method = ckpt.get('train_method', 'ddpm')
    predicts_force = ckpt.get('predicts_force', False)
    # force_in_proprio: whether the OBSERVATION includes force (proprio_dim=12),
    # independent of whether the ACTION target also includes force
    # (predicts_force). The conditioning-only variant (05_train_replay_force.py
    # --no_predict_force) has force_in_proprio=True, predicts_force=False --
    # gating proprio construction on predicts_force alone would silently build
    # a 9D proprio for that checkpoint, a shape mismatch against its 12D-proprio
    # model. Old checkpoints predating this field don't have it stored; default
    # to matching predicts_force, exactly correct for them.
    force_in_proprio = ckpt.get('force_in_proprio', predicts_force)
    force_dim      = ckpt.get('force_dim', 3)
    force_cutoff_hz = args.force_cutoff_hz if args.force_cutoff_hz is not None \
        else ckpt.get('force_cutoff_hz', 2.0)
    noise_scheduler = None
    flow_time_scale, flow_ode_steps = None, None
    if train_method == 'ddpm':
        noise_scheduler = ckpt['noise_scheduler']
        noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps)
    elif train_method == 'flow_matching':
        flow_time_scale = ckpt.get('time_scale', 999.0)
        flow_ode_steps  = ckpt.get('ode_steps', 50)
    elif train_method == 'act':
        pass
    else:
        raise SystemExit(f"Unsupported train_method '{train_method}'.")
    transform = make_transform(image_size, crop_size)
    print(f"Model: obs_steps={n_obs_steps}  n_views={n_views}  action_horizon={model.action_horizon}  "
          f"train_method={train_method}  proprio_dim={model.proprio_dim}  action_dim={model.action_dim}  "
          f"image={image_size}x{image_size} -> crop {crop_size}x{crop_size}  action_frame={action_frame}")
    if force_in_proprio:
        print(f"  force_in_proprio=True (force_dim={force_dim})  predicts_force={predicts_force}  "
              f"{'admittance ENABLED' if args.admittance else 'admittance disabled (--admittance to turn on)'}  "
              f"{'MOMENTS ENABLED (m_desired=' + str(args.m_desired) + ')' if args.admittance_moments else 'moments disabled (--admittance_moments to turn on)'}  "
              f"force_cutoff_hz={force_cutoff_hz}")
    if args.admittance and not predicts_force:
        raise SystemExit("--admittance requires a predicts_force checkpoint "
                         "(this one has predicts_force=False/absent) -- a conditioning-only "
                         "checkpoint has no predicted force to use as the admittance reference.")

    # ── Load UNet arm segmentor ──────────────────────────────────────────────
    unet = None
    if args.no_arm_mask:
        print("--no_arm_mask set: feeding the RAW camera image to the policy.")
    else:
        print(f"Loading UNet arm segmentor: {args.unet_checkpoint}")
        unet = load_unet(args.unet_checkpoint, device)
        print(f"  threshold={args.unet_threshold}")

    # ── Load mesh for to_origin ───────────────────────────────────────────────
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

    # ── T_eef_spoon: FK-based proprio ─────────────────────────────────────────
    eef_spoon_path = os.path.join(PIPELINE_DIR, args.T_eef_spoon)
    T_eef_spoon = None
    if os.path.exists(eef_spoon_path):
        T_eef_spoon = np.load(eef_spoon_path).astype(np.float64)
        T_tool_eef_from_calib = inv_to_origin @ np.linalg.inv(T_eef_spoon)
        t = T_eef_spoon[:3, 3]
        print(f"Loaded T_eef_spoon from {eef_spoon_path} "
              f"(t={t[0]*100:.1f},{t[1]*100:.1f},{t[2]*100:.1f} cm)")
        print("  → FK-based proprio mode: FoundationPose NOT used per-step.")

    need_fp = T_eef_spoon is None
    need_registration = (not os.path.exists(args.tool_eef_cache)
                         or args.recalibrate_tool_eef)

    est = gdino = sam_pred = None
    if need_fp:
        print("Loading GroundedSAM + FoundationPose...")
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
        # ── T_tool_eef ─────────────────────────────────────────────────────────
        if T_eef_spoon is not None:
            T_tool_eef = T_tool_eef_from_calib
            print(f"T_tool_eef derived from T_eef_spoon (no FP registration needed).")
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

        # ── Optional pre-positioning ───────────────────────────────────────────
        if args.init_episode_dir:
            init_pose_path = os.path.join(args.init_episode_dir, 'augmented', 'tool_poses_base.npz')
            init_poses = np.load(init_pose_path)
            init_key = str(args.init_frame)
            if init_key not in init_poses:
                avail = sorted(init_poses.keys(), key=int)
                raise SystemExit(f"--init_frame {args.init_frame} not found in {init_pose_path} "
                                 f"(have {avail[0]}..{avail[-1]})")
            T_base_tool_init = init_poses[init_key].astype(np.float64)
            T_base_eef_init  = T_base_tool_init @ T_tool_eef
            T_base_eef_now   = kinova_pose_to_matrix(get_cartesian_pose(api))
            init_dist = np.linalg.norm(T_base_eef_init[:3, 3] - T_base_eef_now[:3, 3])
            print(f"\nPre-positioning to {args.init_episode_dir} frame {args.init_frame} "
                  f"({init_dist*100:.1f} cm from current pose)...")
            if args.execute:
                send_cartesian_pose(api, matrix_to_kinova_pose(T_base_eef_init),
                                    trans_speed=args.arm_trans_speed)
                arm_speed = args.arm_trans_speed if args.arm_trans_speed > 0 else 0.03
                init_timeout = max(args.convergence_timeout, init_dist / arm_speed + 1.0)
                t_wait, converged = time.time(), False
                while time.time() - t_wait < init_timeout:
                    actual = get_cartesian_pose(api)[:3]
                    if np.linalg.norm(actual - T_base_eef_init[:3, 3]) < args.convergence_threshold:
                        converged = True
                        break
                    time.sleep(0.05)
                print(f"  Pre-position {'converged' if converged else 'TIMED OUT'} "
                      f"in {time.time()-t_wait:.1f}s")
            else:
                print("  (dry run — not moving; pass --execute to actually pre-position)")

        # ── Live wrench pipeline setup: ALWAYS runs, regardless of predicts_force
        # (reading live force/moment is useful diagnostic info even for a plain
        # 9D position-only checkpoint with no admittance correction). ──────────
        gravity_phi = np.load('data/gravity_phi_task_only.npy')
        _nn_ckpt = torch.load('data/gravity_residual_nn.pt', weights_only=False)
        gravity_nn = GravityResidualNet()
        gravity_nn.load_state_dict(_nn_ckpt['state_dict'])
        gravity_nn.eval()
        g_x_mean, g_x_std = _nn_ckpt['x_mean'], _nn_ckpt['x_std']
        dynamics_pi = np.load(args.dynamics_pi)
        qdiff = VelocityDifferentiator(smoothing=args.qddot_smoothing)

        # Full gravity + mass/Coriolis correction -- matches analyze_replay_full.py's
        # 3-stage chain EXACTLY (firmware -> +gravity regressor+NN -> +mass/Coriolis
        # regressor), same as 07_deploy_force.py. The ONE difference from that
        # script: no `[:3]` slice here -- torque_to_wrench already returns the
        # full 6D [Fx,Fy,Fz,Mx,My,Mz] wrench, previously discarded past this point.
        def read_wrench_raw(q_deg, qdot_deg, t):
            gf = get_tau_gf(api)
            g_res_lin = gravity_regressor(q_deg) @ gravity_phi
            g_res_nn  = predict_gravity_residual(gravity_nn, g_x_mean, g_x_std, q_deg)
            tau_after_gravity = gf - g_res_lin - g_res_nn
            qddot_deg = qdiff.update(qdot_deg, t)
            dyn_pred = full_dynamics_regressor(q_deg, qdot_deg, qddot_deg) @ dynamics_pi
            tau_final = tau_after_gravity - dyn_pred
            return torque_to_wrench(q_deg, tau_final, damping=args.damping)   # full (6,)

        print(f"\nTaring: capturing baseline for {args.tare_duration:.1f}s -- leave the arm untouched")
        n_tare = max(1, int(args.tare_duration * args.frequency))
        tare_samples = []
        for _ in range(n_tare):
            q0 = get_joint_angles_deg(api)
            qdot0 = get_qdot_deg(api)
            tare_samples.append(read_wrench_raw(q0, qdot0, time.time()))
        tare_offset = np.mean(tare_samples, axis=0)   # (6,)
        print(f"  tare offset F (N):   [{tare_offset[0]:+.3f}, {tare_offset[1]:+.3f}, {tare_offset[2]:+.3f}]")
        print(f"  tare offset M (N·m): [{tare_offset[3]:+.3f}, {tare_offset[4]:+.3f}, {tare_offset[5]:+.3f}]")

        wrench_filters = [OnlineButterworth(force_cutoff_hz, args.frequency) for _ in range(6)]

        def read_live_wrench_filtered():
            """Live, tared, causally-filtered [Fx,Fy,Fz,Mx,My,Mz] (base frame,
            EEF origin) -- same force convention 05_train_replay_force.py's
            training labels use; the moment half has no training-label analogue
            (see module docstring)."""
            q_deg = get_joint_angles_deg(api)
            qdot_deg = get_qdot_deg(api)
            W_raw = read_wrench_raw(q_deg, qdot_deg, time.time()) - tare_offset
            return np.array([wrench_filters[i].update(W_raw[i]) for i in range(6)])

        m_desired = np.array(args.m_desired, dtype=np.float64)

        # ── Receding-horizon control loop ────────────────────────────────────
        dt           = 1.0 / args.frequency
        max_rot_step = np.radians(args.max_rot_step_deg)
        m_max_correction_rad = np.radians(args.m_max_correction_deg)

        obs_buffer = collections.deque(maxlen=n_obs_steps)
        # Action deque: (T_pred (4,4) tool pose base frame, f_pred (3,) desired
        # force or zeros if not predicts_force) tuples, consumed 1-per-step.
        action_deque = collections.deque()

        _infer_lock    = threading.Lock()
        _infer_running = [False]
        _infer_result  = [None]

        def _infer_thread(buf_snap):
            view_tensors = [v[0] for v in buf_snap]
            proprio_list = [v[1] for v in buf_snap]
            if train_method == 'flow_matching':
                actions_raw, poses = predict_action_sequence_flow(
                    model, normalizer, view_tensors, proprio_list, device,
                    time_scale=flow_time_scale, ode_steps=flow_ode_steps)
            elif train_method == 'act':
                actions_raw, poses = predict_action_sequence_act(
                    model, normalizer, view_tensors, proprio_list, device)
            else:
                actions_raw, poses = predict_action_sequence(
                    model, normalizer, noise_scheduler, view_tensors, proprio_list, device)
            if predicts_force:
                forces_pred = actions_raw[:, 9:9 + force_dim]   # already denormalized (Newtons)
            else:
                forces_pred = np.zeros((len(poses), 3))
            with _infer_lock:
                _infer_result[0]  = list(zip(poses, forces_pred))
                _infer_running[0] = False

        T_base_eef_cur = kinova_pose_to_matrix(get_cartesian_pose(api))
        latest_actions_base = None   # list of (pose, force) tuples, for viz

        step = 0
        paused = False
        _fps_t0    = time.time()
        _fps_count = 0
        _fps_disp  = 0.0
        print("\nStarting control loop (rolling-obs async-inference).")
        print("P = pause/resume | Q = quit\n")
        prev_cmd_xyz_base = None
        cmd_xyz_base = None
        correction = np.zeros(3)       # translation correction (m, base frame)
        rot_correction = np.zeros(3)   # rotation correction (rotvec, rad, base frame)
        wrench_live = np.zeros(6)
        F_live = wrench_live[:3]
        M_live = wrench_live[3:]
        # NaN placeholders: only meaningful when action_deque has an action to
        # execute this tick -- stay NaN on "waiting for inference"/paused ticks,
        # so a force-vs-position-priority analysis can distinguish "no action
        # this tick" from a genuine zero correction.
        pos_cur_before = np.full(3, np.nan)
        pos_pred_raw   = np.full(3, np.nan)
        pos_clamped    = np.full(3, np.nan)
        t_run_start = time.time()
        log = {'t': [], 'F_live': [], 'M_live': [], 'q_deg': [], 'pos_cur_before': [],
              'pos_pred_raw': [], 'correction': [], 'rot_correction_deg': [], 'pos_clamped': []}
        while True:
            t_loop_start = time.time()
            prev_cmd_xyz_base = cmd_xyz_base
            cmd_xyz_base = None

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
                    obs_buffer.clear()
                    latest_actions_base = None
                    print("\n  *** RESUMED ***\n")

            # ── 1. Capture + get tool pose ────────────────────────────────────
            rgb, depth = capture(pipe, align, depth_scale)
            q_deg = get_joint_angles_deg(api)

            if T_eef_spoon is not None:
                T_base_eef_fk = joint_angles_to_eef(q_deg)
                T_base_tool   = T_base_eef_fk @ T_eef_spoon @ to_origin
                pose_cam      = None
            else:
                pose_cam    = est.track_one(rgb=rgb, depth=depth, K=K,
                                            iteration=args.track_refine_iter)
                T_base_tool = T_base_task_prop @ tf_cam2world @ pose_cam.astype(np.float64)
            T_task_tool = np.linalg.inv(T_base_task_prop) @ T_base_tool

            # ── 2. Mask + encode obs ──────────────────────────────────────────
            if args.no_arm_mask:
                robot_mask = np.zeros(rgb.shape[:2], dtype=bool)
                masked_rgb = rgb
            else:
                robot_mask = unet_mask(unet, rgb, args.unet_threshold, device)
                masked_rgb = apply_mask(rgb, robot_mask)
            if not paused:
                cv2.imwrite(os.path.join(args.output_dir, f'{step:06d}_masked.jpg'),
                            cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2BGR))

            if pipe_other is not None:
                rgb_other, _ = capture(pipe_other, align_other, depth_scale_other)
                if args.no_arm_mask:
                    masked_rgb_other = rgb_other
                else:
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

            # Live wrench reading (once per loop iteration -- used for BOTH the
            # past-force proprio feature -- force half only -- and both
            # admittance corrections).
            wrench_live = read_live_wrench_filtered()
            F_live = wrench_live[:3]
            M_live = wrench_live[3:]

            T_proprio = T_task_tool if action_frame == 'task' else T_base_tool
            proprio_9d = pose_matrix_to_9d(T_proprio)
            proprio_raw = np.concatenate([proprio_9d, F_live]) if force_in_proprio else proprio_9d
            proprio = normalizer.normalize(proprio_raw)
            obs_buffer.append((img_t, proprio))

            T_base_eef_now = kinova_pose_to_matrix(get_cartesian_pose(api))

            if not paused:
                # ── 3. Collect completed inference result ─────────────────────
                with _infer_lock:
                    if _infer_result[0] is not None:
                        latest_actions_base = _infer_result[0]
                        action_deque.clear()
                        action_deque.extend(latest_actions_base[:args.exec_steps])
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
                correction = np.zeros(3)
                rot_correction = np.zeros(3)
                if action_deque:
                    T_pred, f_desired = action_deque.popleft()
                    T_pred = T_pred.astype(np.float64)
                    if action_frame == 'task':
                        T_base_tool_pred = T_base_task @ T_pred
                    else:
                        T_base_tool_pred = T_pred
                    T_base_eef_pred = T_base_tool_pred @ T_tool_eef
                    pos_cur_before = T_base_eef_cur[:3, 3].copy()
                    pos_pred_raw = T_base_eef_pred[:3, 3].copy()   # BEFORE admittance correction

                    # ── Hybrid position+force: both corrections BEFORE
                    # clamping, so the existing max_trans_step/max_rot_step
                    # safety clamp still bounds the final commanded step
                    # regardless of what either correction computed. ─────────
                    if args.admittance and predicts_force:
                        F_mag = float(np.linalg.norm(F_live))
                        if F_mag > args.force_safety_threshold:
                            print(f"  [SAFETY] |F|={F_mag:.1f}N > threshold "
                                  f"{args.force_safety_threshold:.1f}N -- zeroing translational "
                                  f"correction this step (position target unaffected)")
                        else:
                            correction = (F_live - f_desired) / args.K
                            cap = args.f_max_correction_cm / 100
                            mag = np.linalg.norm(correction)
                            if mag > cap:
                                correction = correction * (cap / mag)
                        T_base_eef_pred[:3, 3] += correction

                        # ── NEW: rotational admittance from the live moment.
                        # m_desired is FIXED (default zero -- no checkpoint
                        # predicts moments, see module docstring), unlike
                        # f_desired above which is the policy's own live
                        # prediction. Composed as a base-frame perturbation
                        # (R_new = Exp(rot_correction) @ R_pred), matching the
                        # base-frame convention torque_to_wrench already uses
                        # for M_live. ─────────────────────────────────────────
                        if args.admittance_moments:
                            M_mag = float(np.linalg.norm(M_live))
                            if M_mag > args.moment_safety_threshold:
                                print(f"  [SAFETY] |M|={M_mag:.1f}N·m > threshold "
                                      f"{args.moment_safety_threshold:.1f}N·m -- zeroing "
                                      f"rotational correction this step")
                            else:
                                rot_correction = (M_live - m_desired) / args.K_rot
                                rmag = np.linalg.norm(rot_correction)
                                if rmag > m_max_correction_rad:
                                    rot_correction = rot_correction * (m_max_correction_rad / rmag)
                            T_base_eef_pred[:3, :3] = (
                                Rotation.from_rotvec(rot_correction).as_matrix()
                                @ T_base_eef_pred[:3, :3])

                    T_base_eef_clamped = clamp_pose_step(
                        T_base_eef_cur, T_base_eef_pred,
                        args.max_trans_step, max_rot_step)
                    T_base_eef_cur = T_base_eef_clamped
                    t_pred  = T_base_eef_pred[:3, 3]
                    t_clamp = T_base_eef_clamped[:3, 3]
                    pos_clamped = t_clamp.copy()
                    force_str = (f"  F_live=({F_live[0]:+.2f},{F_live[1]:+.2f},{F_live[2]:+.2f})N "
                                f"|F_live|={np.linalg.norm(F_live):.1f}N "
                                f"M_live=({M_live[0]:+.2f},{M_live[1]:+.2f},{M_live[2]:+.2f})N·m "
                                f"|M_live|={np.linalg.norm(M_live):.1f}N·m")
                    if predicts_force:
                        force_str += (f"  f_desired=({f_desired[0]:.1f},{f_desired[1]:.1f},{f_desired[2]:.1f})N  "
                                     f"corr={np.linalg.norm(correction)*100:.2f}cm")
                        if args.admittance_moments:
                            force_str += f"  rot_corr={np.degrees(np.linalg.norm(rot_correction)):.2f}deg"
                    print(f"  step {step} [{len(action_deque)} remain]: "
                          f"eef pred=({t_pred[0]*100:.1f},{t_pred[1]*100:.1f},{t_pred[2]*100:.1f}) cm "
                          f"clamped=({t_clamp[0]*100:.1f},{t_clamp[1]*100:.1f},{t_clamp[2]*100:.1f}) cm"
                          f"{force_str}")
                    if args.execute:
                        send_cartesian_pose(api, matrix_to_kinova_pose(T_base_eef_clamped),
                                            trans_speed=args.arm_trans_speed)
                        cmd_xyz_base = T_base_eef_clamped[:3, 3].copy()
                    _action_sent = True
                else:
                    print(f"  step {step}: waiting for inference...")

            # ── 6. Write state for 3D plotter ─────────────────────────────────
            if not paused:
                if latest_actions_base is not None:
                    latest_poses_base = [p for p, f in latest_actions_base]
                    poses_eef_pred  = np.stack([p.astype(np.float64) @ T_tool_eef
                                                 for p in latest_poses_base])
                    poses_tool_pred = np.stack([p.astype(np.float64)
                                                 for p in latest_poses_base])
                else:
                    latest_poses_base = None
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
            else:
                latest_poses_base = [p for p, f in latest_actions_base] if latest_actions_base else None

            # ── 7. Visualisation ─────────────────────────────────────────────
            ob_in_cam = tf_world2cam @ np.linalg.inv(T_base_task) @ T_base_tool @ inv_to_origin
            vis = rgb.copy()
            vis = draw_axes_simple(vis, ob_in_cam, K, scale=0.10)

            if robot_mask.sum() > 0:
                red = np.zeros_like(vis)
                red[:, :, 0] = 255
                m = robot_mask.astype(bool)
                vis[m] = cv2.addWeighted(vis, 0.5, red, 0.5, 0)[m]

            if latest_poses_base is not None:
                if action_frame == 'task':
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
            cv2.putText(vis, f"proprio(task) xyz= {t_task[0]*100:.1f},{t_task[1]*100:.1f},{t_task[2]*100:.1f} cm",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            force_hud = f"F=({F_live[0]:+.1f},{F_live[1]:+.1f},{F_live[2]:+.1f})N |F|={np.linalg.norm(F_live):.1f}N"
            if predicts_force:
                force_hud += (f"  corr={np.linalg.norm(correction)*100:.2f}cm  "
                             f"{'ADMITTANCE ON' if args.admittance else 'admittance off'}")
            cv2.putText(vis, force_hud, (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 180), 1)
            moment_hud = f"M=({M_live[0]:+.1f},{M_live[1]:+.1f},{M_live[2]:+.1f})N·m |M|={np.linalg.norm(M_live):.1f}N·m"
            if predicts_force:
                moment_hud += (f"  rot_corr={np.degrees(np.linalg.norm(rot_correction)):.2f}deg  "
                              f"{'MOMENTS ON' if args.admittance_moments else 'moments off'}")
            cv2.putText(vis, moment_hud, (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 255), 1)
            if paused:
                h, w = vis.shape[:2]
                overlay = vis.copy()
                cv2.rectangle(overlay, (0, h//2 - 40), (w, h//2 + 40), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)
                cv2.putText(vis, "PAUSED  (press P to resume)",
                            (w//2 - 220, h//2 + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 80, 255), 3)

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

            # Single unified log point, every tick regardless of paused state --
            # runs after any position-command update this tick, so pos_pred_raw/
            # correction/pos_clamped reflect what actually just happened, not a
            # stale value from an earlier tick.
            log['t'].append(t_loop_start - t_run_start)
            log['F_live'].append(F_live.copy())
            log['M_live'].append(M_live.copy())
            log['q_deg'].append(q_deg.copy())
            log['pos_cur_before'].append(pos_cur_before.copy())
            log['pos_pred_raw'].append(pos_pred_raw.copy())
            log['correction'].append(correction.copy())
            log['rot_correction_deg'].append(np.degrees(rot_correction.copy()))
            log['pos_clamped'].append(pos_clamped.copy())

            # ── 8. Wait for arm convergence or sleep ──────────────────────────
            if args.wait_convergence and cmd_xyz_base is not None and args.execute:
                cmd_dist = np.linalg.norm(cmd_xyz_base - prev_cmd_xyz_base) \
                    if prev_cmd_xyz_base is not None else 0.0
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

            # ── 9. Step-mode gate ──────────────────────────────────────────────
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
        if log['t']:
            # Timestamped filename -- avoids the same silent-overwrite bug
            # 07_deploy_force.py fixed (two runs against the same --output_dir
            # previously clobbered each other's log).
            log_path = os.path.join(args.output_dir,
                                    f'force_log_6d_{time.strftime("%Y%m%d_%H%M%S")}.npz')
            np.savez(log_path, t=np.array(log['t']),
                     F_live=np.array(log['F_live']),
                     M_live=np.array(log['M_live']),
                     q_deg=np.array(log['q_deg']),
                     pos_cur_before=np.array(log['pos_cur_before']),
                     pos_pred_raw=np.array(log['pos_pred_raw']),
                     correction=np.array(log['correction']),
                     rot_correction_deg=np.array(log['rot_correction_deg']),
                     pos_clamped=np.array(log['pos_clamped']))
            print(f"Live force+moment log saved -> {log_path}")
        print("Done.")


if __name__ == '__main__':
    main()
