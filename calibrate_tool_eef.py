#!/usr/bin/env python3
"""
Robust multi-pose calibration of T_tool_eef (rigid transform from the
FoundationPose tool-mesh-native-origin frame to the robot EEF frame).

This does NOT touch 06_calibrate_robot.py or its T_base_task output — it
CONSUMES that calibration (data/robot_extrinsics.npy) plus the camera
extrinsics (data/cam_extrinsics.npy), exactly like check_tool_eef_error.py's
single-shot 'C' capture:

    T_base_tool_fp(i) = T_base_task @ inv(tf_world2cam) @ T_cam_tool(i)
    T_tool_eef(i)      = inv(T_base_tool_fp(i)) @ T_base_eef(i)

Since T_base_task and tf_world2cam are already fixed/known, T_tool_eef(i) is
a full closed-form estimate from a SINGLE pose (no AX=XB solve needed) — the
single-shot version in check_tool_eef_error.py is just noisy because it's
one FoundationPose frame. This script collects that estimate at MANY poses
(you jog the spoon around with the joystick — vary position AND orientation)
and robustly averages them:

  - each "capture" (press C) itself averages a short burst
    (--samples_per_capture frames, held-still) to cut single-frame FP jitter
  - across captures: translation via mean, rotation via the SVD/chordal
    projection of the mean rotation matrix (proper handling of SO(3), no
    quaternion sign ambiguity)
  - one outlier-rejection pass (per-sample deviation from the first-pass
    robust estimate) before the final average, so one bad FP registration
    doesn't skew the result

CAVEAT — T_base_task may itself be a few cm off (06_calibrate_robot.py's
4-touch-point procedure has real touch-precision error). Holding it fixed
means any such bias leaks into the T_tool_eef estimate. Since a TRANSLATION
error in T_base_task, conjugated by different tool orientations, rotates
into different apparent directions (R @ Δt) rather than staying fixed,
diverse-orientation captures mostly average it out of T_tool_eef — but they
don't fix the underlying bias, which still throws off every deployed replay
that goes through T_base_task.

So this script ALSO runs a joint refinement (classic robot-world/hand-eye
calibration: solve for T_base_task correction AND T_tool_eef simultaneously
from the same captured data), warm-started from the loaded T_base_task and
the closed-form T_tool_eef. This needs rotations about >=2 non-parallel axes
across your captures to be well-conditioned (the same "vary orientation"
guidance already on screen). It reports the detected T_base_task correction
magnitude and never overwrites robot_extrinsics.npy — the refined pair is
saved to separate *_refined.npy files for you to adopt (via --robot_extrinsics
on downstream scripts) only if you choose to.

Usage:
    python calibrate_tool_eef.py --mesh spoon.obj --tool_prompt "spoon"
    python calibrate_tool_eef.py --mesh newspoon1.obj --tool_prompt "spoon" \\
        --output data/T_tool_eef.npy --min_samples 15

Controls (live camera window):
    SPACE  re-register the tool (if tracking is lost / lighting changed)
    C      capture a sample at the CURRENT pose (hold the arm still ~1s)
    X      discard the LAST capture (use this if the console flags it as an
           outlier vs. prior captures, or if the burst-consistency warning fired)
    F      finish early and solve with samples collected so far
    Q      quit without saving

After every C, a banner at the BOTTOM of the camera window (not the console,
which FoundationPose spams continuously and makes unreadable) shows this
capture's residual against the running average of prior captures, and warns
if the ~1s burst itself wasn't internally consistent (a sign FoundationPose
lost/snapped mid-capture) — red banner = something's wrong, press X
immediately to discard and retake that pose, rather than only discovering it
after all 20 are in and the final residuals turn out to be nonsense.

After it saves data/T_tool_eef.npy (backing up any previous file first),
validate live with:
    python check_tool_eef_error.py --mesh spoon.obj --tool_prompt "spoon"
"""

import os, sys, time, csv, argparse, shutil
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

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

import ctypes

# ── Kinova SDK (same boilerplate as check_tool_eef_error.py / 07_deploy.py) ──

NO_ERROR_KINOVA   = 1
SERIAL_LENGTH     = 20
MAX_KINOVA_DEVICE = 20

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
                   ("StopControlAPI", ctypes.c_int), ("GetCartesianPosition", ctypes.c_int)]:
        getattr(api, fn).restype = rt
    return api


def connect_robot_readonly():
    """READ-ONLY connect: no SetCartesianControl, no motion command ever sent.
    Safe to run while you jog the arm with the joystick, same idea as
    diag_live_forces.py's connect(api, control=False)."""
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
    print(f"Connected (read-only): {devices[0].Model.decode()} (serial {devices[0].SerialNumber.decode()})")
    return api


def get_eef_pose(api, euler_conv='XYZ') -> np.ndarray:
    """Returns (4,4) T_base_eef."""
    pos = CartesianPosition()
    api.GetCartesianPosition(ctypes.byref(pos))
    c = pos.Coordinates
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler(euler_conv, [c.ThetaX, c.ThetaY, c.ThetaZ]).as_matrix()
    T[:3,  3] = [c.X, c.Y, c.Z]
    return T


# ── Camera (same as check_tool_eef_error.py) ─────────────────────────────────

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


# ── GroundedSAM segmentation (same as check_tool_eef_error.py) ───────────────

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


# ── Visualization: mesh-native-origin XYZ axes overlay ───────────────────────

def _project_pt(pt3d, K):
    u = int(K[0, 0] * pt3d[0] / pt3d[2] + K[0, 2])
    v = int(K[1, 1] * pt3d[1] / pt3d[2] + K[1, 2])
    return (u, v)


def draw_axes(img, T_cam, K, length=0.08, thickness=3,
              x_color=(0, 0, 255), y_color=(0, 255, 0), z_color=(255, 0, 0), label=None):
    """Draws the XYZ axes of T_cam (mesh-native origin, i.e. exactly the frame
    T_tool_eef is defined relative to — NOT the OBB-centered box frame used
    for draw_posed_3d_box) onto img (BGR). Lets you visually catch
    FoundationPose axis swaps/flips before they get baked into a calibration."""
    origin = T_cam[:3, 3]
    if origin[2] <= 0.01:
        return img
    o_px = _project_pt(origin, K)
    for axis_idx, color in enumerate([x_color, y_color, z_color]):
        tip = origin + T_cam[:3, axis_idx] * length
        if tip[2] <= 0.01:
            continue
        t_px = _project_pt(tip, K)
        cv2.line(img, o_px, t_px, color, thickness, cv2.LINE_AA)
        cv2.circle(img, t_px, max(3, thickness), color, -1, cv2.LINE_AA)
    cv2.circle(img, o_px, 5, (255, 255, 255), -1, cv2.LINE_AA)
    if label:
        cv2.putText(img, label, (o_px[0] + 7, o_px[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(img, label, (o_px[0] + 7, o_px[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


# ── Robust SO(3)/SE(3) averaging ──────────────────────────────────────────────

def average_rotations(Rs):
    """Chordal L2 mean: project mean(R_i) back onto SO(3) via SVD. Avoids
    quaternion sign-flip bookkeeping and is well-defined for any spread of
    inputs (as long as they're not near-antipodal, which won't happen for a
    physical tool-eef offset)."""
    M = np.mean(np.stack(Rs, axis=0), axis=0)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def average_poses(Ts):
    """Mean translation + chordal-mean rotation over a list of 4x4 poses."""
    Rs = [T[:3, :3] for T in Ts]
    ts = [T[:3, 3] for T in Ts]
    T_avg = np.eye(4)
    T_avg[:3, :3] = average_rotations(Rs)
    T_avg[:3, 3] = np.mean(np.stack(ts, axis=0), axis=0)
    return T_avg


def tool_eef_to_eef_spoon(T_tool_eef, inv_to_origin):
    """
    Converts our T_tool_eef (mesh-native-origin tool frame -> EEF) into the
    T_eef_spoon convention 07_deploy.py/replay_episode.py actually load by
    default (they derive T_tool_eef = inv_to_origin @ inv(T_eef_spoon) —
    this is that relation solved for T_eef_spoon), so this calibration can
    feed the existing deployment pipeline's fast FK-only proprio path
    without any changes to those scripts.
    """
    return np.linalg.inv(T_tool_eef) @ inv_to_origin


def pose_residual(T_a, T_b):
    """(trans_m, rot_deg) between two 4x4 poses."""
    trans = np.linalg.norm(T_a[:3, 3] - T_b[:3, 3])
    R_rel = T_a[:3, :3].T @ T_b[:3, :3]
    rot   = np.degrees(Rotation.from_matrix(R_rel).magnitude())
    return trans, rot


def solve_tool_eef(candidates, trans_outlier_m, rot_outlier_deg):
    """
    candidates: list of 4x4 T_tool_eef point-estimates, one per captured pose.
    Returns (T_tool_eef_final, diagnostics dict).
    """
    n = len(candidates)
    T0 = average_poses(candidates)   # first-pass robust estimate (all samples)
    residuals = [pose_residual(T0, T) for T in candidates]
    inlier_mask = [(t <= trans_outlier_m and r <= rot_outlier_deg) for t, r in residuals]
    n_inliers = sum(inlier_mask)

    if n_inliers < 3:
        print(f"  WARNING: only {n_inliers}/{n} samples passed outlier rejection "
              f"(thresholds {trans_outlier_m*100:.1f}cm / {rot_outlier_deg:.1f}deg) — "
              f"using ALL samples instead (thresholds too tight or data too noisy).")
        inliers = candidates
        inlier_mask = [True] * n
    else:
        inliers = [T for T, keep in zip(candidates, inlier_mask) if keep]

    T_final = average_poses(inliers)
    final_residuals = [pose_residual(T_final, T) for T in inliers]
    trans_res = np.array([r[0] for r in final_residuals])
    rot_res   = np.array([r[1] for r in final_residuals])

    diag = dict(
        n_total=n, n_inliers=len(inliers), n_outliers=n - len(inliers),
        inlier_mask=inlier_mask, first_pass_residuals=residuals,
        trans_res_mean_cm=float(trans_res.mean() * 100), trans_res_max_cm=float(trans_res.max() * 100),
        rot_res_mean_deg=float(rot_res.mean()), rot_res_max_deg=float(rot_res.max()),
    )
    return T_final, diag


# ── Joint refinement of T_base_task + T_tool_eef (robot-world/hand-eye) ──────

def _se3_delta(rvec, t):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rvec).as_matrix()
    T[:3, 3] = t
    return T


def _joint_residuals(p, B_list, T_base_eef_list, T_base_task_init, T_tool_eef_init):
    T_base_task_p = T_base_task_init @ _se3_delta(p[0:3], p[3:6])
    T_tool_eef_p  = T_tool_eef_init  @ _se3_delta(p[6:9], p[9:12])
    res = np.empty(6 * len(B_list))
    for i, (B_i, T_base_eef_i) in enumerate(zip(B_list, T_base_eef_list)):
        pred = T_base_task_p @ B_i @ T_tool_eef_p
        err  = np.linalg.inv(T_base_eef_i) @ pred
        res[6*i:6*i+3] = err[:3, 3]
        res[6*i+3:6*i+6] = Rotation.from_matrix(err[:3, :3]).as_rotvec()
    return res


def refine_base_task_and_tool_eef(B_list, T_base_eef_list, T_base_task_init, T_tool_eef_init):
    """
    Jointly refines T_base_task and T_tool_eef so that, for every captured
    pose i:  T_base_task @ B_list[i] @ T_tool_eef ~= T_base_eef_list[i]
    (B_list[i] = tf_cam2world @ T_cam_tool_avg[i], the vision-only tool pose
    in task frame — independent of T_base_task).

    Local nonlinear refinement (scipy least_squares) around the loaded/
    closed-form initial guesses — appropriate since the expected error is
    small (a few cm / a few deg), well within the basin of convergence.
    Returns (T_base_task_refined, T_tool_eef_refined, diagnostics dict).
    """
    from scipy.optimize import least_squares

    res_before = _joint_residuals(np.zeros(12), B_list, T_base_eef_list,
                                  T_base_task_init, T_tool_eef_init)
    result = least_squares(_joint_residuals, x0=np.zeros(12), method='lm',
                          args=(B_list, T_base_eef_list, T_base_task_init, T_tool_eef_init))
    p_opt = result.x
    res_after = _joint_residuals(p_opt, B_list, T_base_eef_list,
                                 T_base_task_init, T_tool_eef_init)

    T_base_task_refined = T_base_task_init @ _se3_delta(p_opt[0:3], p_opt[3:6])
    T_tool_eef_refined  = T_tool_eef_init  @ _se3_delta(p_opt[6:9], p_opt[9:12])

    base_task_dtrans_cm = float(np.linalg.norm(p_opt[3:6]) * 100)
    base_task_drot_deg  = float(np.degrees(np.linalg.norm(p_opt[0:3])))

    def _rms(res):
        res = res.reshape(-1, 6)
        return float(np.sqrt((res[:, :3]**2).sum(axis=1)).mean() * 100), \
               float(np.degrees(np.linalg.norm(res[:, 3:6], axis=1)).mean())

    trans_rms_before, rot_rms_before = _rms(res_before)
    trans_rms_after,  rot_rms_after  = _rms(res_after)

    diag = dict(
        base_task_dtrans_cm=base_task_dtrans_cm, base_task_drot_deg=base_task_drot_deg,
        trans_rms_before_cm=trans_rms_before, rot_rms_before_deg=rot_rms_before,
        trans_rms_after_cm=trans_rms_after, rot_rms_after_deg=rot_rms_after,
        n_samples=len(B_list),
    )
    return T_base_task_refined, T_tool_eef_refined, diag


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mesh',        required=True, help='Tool mesh (.obj)')
    p.add_argument('--tool_prompt', required=True, help='GroundedSAM prompt, e.g. "spoon"')
    p.add_argument('--task_frame',       default='data/cam_extrinsics.npy')
    p.add_argument('--robot_extrinsics', default='data/robot_extrinsics.npy')
    p.add_argument('--output',           default='data/T_tool_eef.npy')
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--box_threshold',  type=float, default=0.3)
    p.add_argument('--text_threshold', type=float, default=0.25)
    p.add_argument('--est_refine_iter',   type=int, default=5)
    p.add_argument('--track_refine_iter', type=int, default=2)
    p.add_argument('--samples_per_capture', type=int, default=15,
                   help='Frames averaged per "C" press — hold the arm still while these are taken')
    p.add_argument('--min_samples', type=int, default=15,
                   help='Warn (not block) if you try to solve with fewer captured poses than this')
    p.add_argument('--trans_outlier_cm', type=float, default=2.0)
    p.add_argument('--rot_outlier_deg',  type=float, default=3.0)
    p.add_argument('--euler_conv', default='XYZ')
    p.add_argument('--output_csv', default='/tmp/tool_eef_calibration_samples.csv')
    p.add_argument('--device', default='cuda')
    p.add_argument('--no_refine_base_task', action='store_true',
                   help='Skip the joint T_base_task+T_tool_eef refinement (closed-form T_tool_eef only)')
    p.add_argument('--output_base_task_refined', default='data/robot_extrinsics_refined.npy')
    p.add_argument('--output_tool_eef_refined',  default='data/T_tool_eef_refined.npy')
    p.add_argument('--update_eef_spoon', action='store_true',
                   help='Also OVERWRITE data/T_eef_spoon.npy (backed up first) with this calibration, '
                        'converted to that convention — this is the file 07_deploy.py/replay_episode.py '
                        'actually use by default for the fast FK-only proprio path. Without this flag, '
                        'the equivalent value is only written to a sidecar file for you to adopt manually.')
    p.add_argument('--output_eef_spoon_sidecar', default='data/T_eef_spoon_from_calib.npy')
    p.add_argument('--burst_trans_warn_cm', type=float, default=5.0,
                   help='Warn if any single frame in a capture burst deviates from the burst '
                        'average by more than this (translation). Pure FP self-consistency check, '
                        'purely informational unless MOST of the burst exceeds it -- ordinary '
                        'per-frame jitter is smoothed out by the burst average anyway.')
    p.add_argument('--burst_rot_warn_deg', type=float, default=15.0,
                   help='Same, for rotation. Elongated/thin tools (e.g. a spoon handle) have a '
                        'weakly-constrained roll axis, so several degrees of jitter at rest is '
                        'normal, especially at a fast --track_refine_iter -- not a sign tracking '
                        'is actually failing.')
    p.add_argument('--capture_outlier_trans_cm', type=float, default=5.0,
                   help='Flag a capture as an outlier vs. the running average of prior captures '
                        'beyond this translation deviation.')
    p.add_argument('--capture_outlier_rot_deg', type=float, default=8.0,
                   help='Same, for rotation.')
    return p.parse_args()


def main():
    args = parse_args()

    tf_world2cam = np.load(args.task_frame).astype(np.float64)
    T_base_task  = np.load(args.robot_extrinsics).astype(np.float64)
    tf_cam2world = np.linalg.inv(tf_world2cam)
    print("Loaded task/camera + robot-base calibration.")

    cv2.namedWindow("Tool-EEF Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tool-EEF Calibration", 848, 480)

    gdino, sam_pred = load_gdino_sam(args.device)

    import trimesh
    loaded = trimesh.load(args.mesh)
    mesh = (trimesh.util.concatenate(list(loaded.geometry.values()))
            if isinstance(loaded, trimesh.Scene) else loaded)
    if mesh.bounding_box.extents.max() > 0.5:
        mesh.apply_scale(0.01)
        print("Mesh rescaled x0.01 (cm -> m)")
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    inv_to_origin = np.linalg.inv(to_origin)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    print("Loading FoundationPose...")
    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
    import nvdiffrast.torch as dr
    from Utils import draw_posed_3d_box
    scorer  = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx   = dr.RasterizeCudaContext()
    os.makedirs('/tmp/fp_calib_debug', exist_ok=True)
    est = FoundationPose(
        model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
        scorer=scorer, refiner=refiner, glctx=glctx,
        debug_dir='/tmp/fp_calib_debug', debug=0,
    )
    print("FoundationPose ready.")

    pipe, align, K, depth_scale = start_realsense(args.camera)
    api = connect_robot_readonly()

    def register_tool(rgb, depth):
        print(f"\nDetecting tool ('{args.tool_prompt}')...")
        mask = segment_tool(gdino, sam_pred, rgb, args.tool_prompt,
                            args.box_threshold, args.text_threshold, args.device)
        if mask.sum() < 100:
            print("  Tool not detected — reposition and press SPACE.")
            return None
        pose = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask,
                            iteration=args.est_refine_iter)
        print("  Registered.")
        return pose

    print(f"\nRegistering tool — press SPACE to re-register, C to capture, F to finish, Q to quit.\n"
         f"Move the arm with the joystick between captures — vary POSITION and ORIENTATION.\n")
    rgb, depth = capture(pipe, align, depth_scale)
    pose_cam = register_tool(rgb, depth)
    initialized = pose_cam is not None

    candidates = []       # T_tool_eef point-estimate per captured pose
    sample_log = []       # (T_base_eef_burst_avg, T_cam_tool_burst_avg) for CSV/debug
    status_lines = []     # on-screen feedback after each capture (FoundationPose
    status_color = (0, 255, 0)   # spams the console, so this is the only reliable place to see it)

    try:
        while True:
            rgb, depth = capture(pipe, align, depth_scale)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\nQuit — nothing saved.")
                return

            if key == ord(' '):
                pose_cam = register_tool(rgb, depth)
                initialized = pose_cam is not None
                continue

            if key == ord('f'):
                break

            if not initialized:
                vis_bgr = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
                cv2.putText(vis_bgr, "Registration failed — press SPACE to retry",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
                cv2.imshow("Tool-EEF Calibration", vis_bgr)
                continue

            pose_cam = est.track_one(rgb=rgb, depth=depth, K=K, iteration=args.track_refine_iter)
            T_cam_tool = pose_cam.astype(np.float64)

            if key == ord('x') and candidates:
                candidates.pop()
                sample_log.pop()
                status_lines = [f"DISCARDED last capture. Total now: {len(candidates)}"]
                status_color = (0, 165, 255)
                print(f"\n[discarded last capture] total now: {len(candidates)}")
                continue

            if key == ord('c'):
                print(f"\n[capture {len(candidates)+1}] holding for {args.samples_per_capture} frames...")
                burst_tool, burst_eef = [T_cam_tool.copy()], [get_eef_pose(api, args.euler_conv)]
                for burst_i in range(args.samples_per_capture - 1):
                    rgb_b, depth_b = capture(pipe, align, depth_scale)
                    pose_b = est.track_one(rgb=rgb_b, depth=depth_b, K=K, iteration=args.track_refine_iter)
                    burst_tool.append(pose_b.astype(np.float64))
                    burst_eef.append(get_eef_pose(api, args.euler_conv))
                    # Render EVERY burst frame -- previously this loop only pumped
                    # cv2.waitKey with no imshow, so the window sat frozen on the
                    # instant you pressed C and any mid-burst tracking glitch was
                    # completely invisible. Now you can actually watch it happen.
                    vis_burst = cv2.cvtColor(rgb_b.copy(), cv2.COLOR_RGB2BGR)
                    vis_burst = draw_posed_3d_box(K, img=vis_burst,
                                                  ob_in_cam=pose_b.astype(np.float64) @ np.linalg.inv(to_origin),
                                                  bbox=bbox)
                    draw_axes(vis_burst, pose_b.astype(np.float64), K, length=0.08, thickness=3)
                    cv2.putText(vis_burst, f"CAPTURING frame {burst_i+2}/{args.samples_per_capture} — hold still",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(vis_burst, f"CAPTURING frame {burst_i+2}/{args.samples_per_capture} — hold still",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.imshow("Tool-EEF Calibration", vis_burst)
                    cv2.waitKey(1)
                T_cam_tool_avg = average_poses(burst_tool)
                T_base_eef_avg = average_poses(burst_eef)

                # Per-frame breakdown (not just the max) -- tells us whether it's
                # one transient glitch or the whole burst disagreeing.
                per_frame = [pose_residual(T_cam_tool_avg, T) for T in burst_tool]
                n_bad_frames = sum(1 for t, r in per_frame
                                  if t * 100 > args.burst_trans_warn_cm or r > args.burst_rot_warn_deg)
                worst_i = int(np.argmax([t for t, r in per_frame]))
                print(f"  per-frame vs burst average: {n_bad_frames}/{len(per_frame)} frames over threshold; "
                     f"worst = frame {worst_i} (trans={per_frame[worst_i][0]*100:.2f}cm "
                     f"rot={per_frame[worst_i][1]:.2f}deg)")

                new_status = [f"CAPTURE #{len(candidates)+1}"]
                is_bad = False

                # Catch mid-burst tracking loss/drift: if the ~1s burst itself isn't
                # self-consistent, FoundationPose likely lost/snapped mid-capture.
                burst_trans_max = max(t for t, r in per_frame) * 100
                burst_rot_max   = max(r for t, r in per_frame)
                new_status.append(f"burst spread: trans_max={burst_trans_max:.2f}cm "
                                  f"rot_max={burst_rot_max:.2f}deg ({n_bad_frames}/{len(per_frame)} "
                                  f"frames over threshold, worst=frame {worst_i})")
                majority_bad = n_bad_frames > len(per_frame) / 2
                if burst_trans_max > args.burst_trans_warn_cm or burst_rot_max > args.burst_rot_warn_deg:
                    if majority_bad:
                        is_bad = True
                        new_status.append("BURST UNSTABLE -- FP likely lost/snapped mid-capture.")
                        new_status.append("SPACE to re-register, then X to discard + retake.")
                    else:
                        new_status.append(f"{n_bad_frames} frame(s) glitched (transient) -- "
                                          f"dropping and recomputing from the rest.")
                    print(f"  WARNING: burst itself was inconsistent (internal spread "
                         f"trans_max={burst_trans_max:.2f}cm rot_max={burst_rot_max:.2f}deg, "
                         f"{n_bad_frames}/{len(per_frame)} frames over threshold) — "
                         f"{'FoundationPose likely lost/snapped mid-capture' if majority_bad else 'a transient glitch, recovering from the remaining frames'}.")

                # If it's a MINORITY of frames glitching (not the whole burst), drop
                # them and recompute the average from the clean frames instead of
                # just warning and baking the glitch into the capture anyway.
                if 0 < n_bad_frames < len(per_frame) and not majority_bad:
                    good_tool = [T for T, (t, r) in zip(burst_tool, per_frame)
                                if not (t * 100 > args.burst_trans_warn_cm or r > args.burst_rot_warn_deg)]
                    good_eef  = [E for E, (t, r) in zip(burst_eef, per_frame)
                                if not (t * 100 > args.burst_trans_warn_cm or r > args.burst_rot_warn_deg)]
                    T_cam_tool_avg = average_poses(good_tool)
                    T_base_eef_avg = average_poses(good_eef)
                    new_status.append(f"-> recomputed average from {len(good_tool)} clean frames "
                                      f"(dropped {n_bad_frames} glitched)")
                    print(f"  Recomputed burst average from {len(good_tool)}/{len(per_frame)} "
                         f"clean frames (dropped the glitched one(s)).")

                T_base_tool_fp = T_base_task @ tf_cam2world @ T_cam_tool_avg
                T_tool_eef_i = np.linalg.inv(T_base_tool_fp) @ T_base_eef_avg

                # Catch cross-capture drift/ambiguity: compare against the running
                # average of prior captures BEFORE adding this one.
                if len(candidates) >= 1:
                    running_avg = average_poses(candidates)
                    t_res, r_res = pose_residual(running_avg, T_tool_eef_i)
                    is_outlier = (t_res * 100 > args.capture_outlier_trans_cm
                                 or r_res > args.capture_outlier_rot_deg)
                    new_status.append(f"vs. {len(candidates)} prior: trans={t_res*100:.2f}cm "
                                      f"rot={r_res:.2f}deg")
                    if is_outlier:
                        is_bad = True
                        new_status.append("OUTLIER vs. prior captures -- press X to discard")
                    print(f"  vs. running average of {len(candidates)} prior capture(s): "
                         f"trans={t_res*100:.2f}cm rot={r_res:.2f}deg"
                         + (" *** OUTLIER ***" if is_outlier else ""))

                candidates.append(T_tool_eef_i)
                sample_log.append((T_base_eef_avg, T_cam_tool_avg))
                new_status.append(f"total captured: {len(candidates)} (recommended >= {args.min_samples})")
                status_lines = new_status
                status_color = (0, 60, 255) if is_bad else (0, 255, 0)
                print(f"  captured. total so far: {len(candidates)} "
                     f"(need >= {args.min_samples} recommended, spanning diverse orientations)")

            vis_bgr = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
            center_pose = T_cam_tool @ np.linalg.inv(to_origin)
            vis_bgr = draw_posed_3d_box(K, img=vis_bgr, ob_in_cam=center_pose, bbox=bbox)
            draw_axes(vis_bgr, T_cam_tool, K, length=0.08, thickness=3, label="spoon")
            lines = [
                f"Captured: {len(candidates)} / recommended >= {args.min_samples}",
                "SPACE=re-register  C=capture (hold still)  X=discard last  F=finish  Q=quit",
                "Vary POSITION and ORIENTATION between captures",
                "Axes: X=red Y=green Z=blue (mesh-native origin, same frame as T_tool_eef)",
            ]
            for i, line in enumerate(lines):
                y = 25 + i * 24
                cv2.putText(vis_bgr, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
                cv2.putText(vis_bgr, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

            # Last-capture feedback, on-screen -- FoundationPose spams the console,
            # so this banner is the only reliable place to actually see this.
            if status_lines:
                banner_h = 30 + 26 * len(status_lines)
                overlay = vis_bgr.copy()
                cv2.rectangle(overlay, (0, vis_bgr.shape[0] - banner_h),
                              (vis_bgr.shape[1], vis_bgr.shape[0]), (0, 0, 0), -1)
                vis_bgr = cv2.addWeighted(overlay, 0.65, vis_bgr, 0.35, 0)
                for i, line in enumerate(status_lines):
                    y = vis_bgr.shape[0] - banner_h + 24 + i * 26
                    cv2.putText(vis_bgr, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(vis_bgr, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, status_color, 1, cv2.LINE_AA)

            cv2.imshow("Tool-EEF Calibration", vis_bgr)

    finally:
        api.CloseAPI()
        pipe.stop()
        cv2.destroyAllWindows()

    if len(candidates) < 3:
        print(f"\nOnly {len(candidates)} sample(s) captured — need at least 3 to solve. Nothing saved.")
        return
    if len(candidates) < args.min_samples:
        print(f"\nWARNING: only {len(candidates)} samples (recommended >= {args.min_samples}) — "
             f"result may be noisy / poorly conditioned in orientation.")

    T_tool_eef_final, diag = solve_tool_eef(
        candidates, args.trans_outlier_cm / 100.0, args.rot_outlier_deg)

    print(f"\n{'='*70}")
    print(f"Solved T_tool_eef from {diag['n_total']} samples "
         f"({diag['n_inliers']} inliers, {diag['n_outliers']} rejected as outliers)")
    print(f"Inlier residual vs. final estimate: "
         f"trans mean={diag['trans_res_mean_cm']:.2f}cm max={diag['trans_res_max_cm']:.2f}cm | "
         f"rot mean={diag['rot_res_mean_deg']:.2f}deg max={diag['rot_res_max_deg']:.2f}deg")
    print(f"T_tool_eef:\n{T_tool_eef_final.round(4)}")
    print(f"{'='*70}")

    if not args.no_refine_base_task:
        inlier_B      = [tf_cam2world @ sample_log[i][1] for i, T_i in enumerate(candidates) if diag['inlier_mask'][i]]
        inlier_eef    = [sample_log[i][0] for i, T_i in enumerate(candidates) if diag['inlier_mask'][i]]
        if len(inlier_B) < 8:
            print(f"\n(Skipping joint T_base_task refinement — only {len(inlier_B)} inlier samples, "
                 f"need >= 8 diverse-orientation poses to condition it reasonably.)")
        else:
            T_base_task_refined, T_tool_eef_refined, rdiag = refine_base_task_and_tool_eef(
                inlier_B, inlier_eef, T_base_task, T_tool_eef_final)
            print(f"\n{'='*70}")
            print(f"Joint refinement of T_base_task + T_tool_eef ({rdiag['n_samples']} inlier samples)")
            print(f"Detected T_base_task correction: "
                 f"trans={rdiag['base_task_dtrans_cm']:.2f}cm  rot={rdiag['base_task_drot_deg']:.2f}deg")
            print(f"Residual RMS  fixed-T_base_task: trans={rdiag['trans_rms_before_cm']:.2f}cm "
                 f"rot={rdiag['rot_rms_before_deg']:.2f}deg")
            print(f"Residual RMS  jointly-refined:   trans={rdiag['trans_rms_after_cm']:.2f}cm "
                 f"rot={rdiag['rot_rms_after_deg']:.2f}deg")
            print(f"(robot_extrinsics.npy and T_tool_eef.npy are left untouched — this pair is saved "
                 f"separately; adopt it by pointing downstream scripts' --robot_extrinsics / tool_eef "
                 f"cache at the *_refined.npy files below, if the residual improvement looks worthwhile)")
            print(f"{'='*70}")

            base_out = os.path.join(PIPELINE_DIR, args.output_base_task_refined)
            tool_out = os.path.join(PIPELINE_DIR, args.output_tool_eef_refined)
            np.save(base_out, T_base_task_refined)
            np.save(tool_out, T_tool_eef_refined)
            print(f"Saved refined pair -> {base_out}\n                       {tool_out}")
            print(f"NOTE: {args.output_eef_spoon_sidecar} / --update_eef_spoon below are derived from "
                 f"the CLOSED-FORM T_tool_eef (paired with your current, unrefined robot_extrinsics.npy) "
                 f"— NOT from this refined pair. If you adopt the refined robot_extrinsics.npy for "
                 f"deployment, T_eef_spoon must be re-derived to match it (ask, rather than mixing "
                 f"a refined T_tool_eef with the old robot_extrinsics.npy).")

    # ── T_eef_spoon sidecar — feeds 07_deploy.py/replay_episode.py's fast FK-only
    # proprio path, which by default prefers T_eef_spoon.npy over T_tool_eef.npy ──
    T_eef_spoon_calib = tool_eef_to_eef_spoon(T_tool_eef_final, inv_to_origin)
    sidecar_path = os.path.join(PIPELINE_DIR, args.output_eef_spoon_sidecar)
    np.save(sidecar_path, T_eef_spoon_calib)
    print(f"\nSaved T_eef_spoon-equivalent sidecar -> {sidecar_path}")
    if args.update_eef_spoon:
        live_eef_spoon = os.path.join(PIPELINE_DIR, 'data', 'T_eef_spoon.npy')
        if os.path.exists(live_eef_spoon):
            backup = live_eef_spoon.replace('.npy', f'_backup_{time.strftime("%Y%m%d_%H%M%S")}.npy')
            shutil.copy2(live_eef_spoon, backup)
            print(f"Backed up previous T_eef_spoon.npy -> {backup}")
        np.save(live_eef_spoon, T_eef_spoon_calib)
        print(f"Updated {live_eef_spoon} — 07_deploy.py/replay_episode.py will now use this calibration.")
    else:
        print(f"(--update_eef_spoon not passed — data/T_eef_spoon.npy, which 07_deploy.py/"
             f"replay_episode.py actually load by default, is UNCHANGED. Copy the sidecar over "
             f"it, or rerun with --update_eef_spoon, once you're happy with the residuals above.)")

    out_path = os.path.join(PIPELINE_DIR, args.output) if not os.path.isabs(args.output) else args.output
    if os.path.exists(out_path):
        backup_path = out_path.replace('.npy', f'_backup_{time.strftime("%Y%m%d_%H%M%S")}.npy')
        shutil.copy2(out_path, backup_path)
        print(f"Backed up previous file -> {backup_path}")
    np.save(out_path, T_tool_eef_final)
    meta_path = out_path.replace('.npy', '_meta.npz')
    np.savez(meta_path, date=time.strftime('%Y-%m-%d %H:%M:%S'), mesh=args.mesh,
             tool_prompt=args.tool_prompt, n_total=diag['n_total'], n_inliers=diag['n_inliers'],
             trans_res_mean_cm=diag['trans_res_mean_cm'], trans_res_max_cm=diag['trans_res_max_cm'],
             rot_res_mean_deg=diag['rot_res_mean_deg'], rot_res_max_deg=diag['rot_res_max_deg'])
    print(f"Saved T_tool_eef -> {out_path}")
    print(f"Saved metadata   -> {meta_path}")

    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample', 'is_inlier', 'trans_res_to_final_cm', 'rot_res_to_final_deg'])
        inlier_iter = iter(diag['inlier_mask'])
        for i, T_i in enumerate(candidates):
            t_res, r_res = pose_residual(T_tool_eef_final, T_i)
            writer.writerow([i, diag['inlier_mask'][i], round(t_res * 100, 3), round(r_res, 3)])
    print(f"Saved per-sample residuals -> {args.output_csv}")

    print(f"\nValidate live with:\n"
         f"  python check_tool_eef_error.py --mesh {args.mesh} --tool_prompt \"{args.tool_prompt}\" "
         f"--tool_eef_cache {args.output}")
    if not args.no_refine_base_task:
        print(f"Or validate the jointly-refined pair with:\n"
             f"  python check_tool_eef_error.py --mesh {args.mesh} --tool_prompt \"{args.tool_prompt}\" "
             f"--tool_eef_cache {args.output_tool_eef_refined} --robot_extrinsics {args.output_base_task_refined}")


if __name__ == '__main__':
    main()
