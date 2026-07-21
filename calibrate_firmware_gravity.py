#!/usr/bin/env python3
"""
################################################################################
# UPDATE (2026-07-03): RE-RAN this script. Result this time was clean and
# consistent — mean ||F|| 21.4N -> 1.69N (89-94% reduction, all 8 poses), NO
# fault: joystick normal, J6 confirmed rotating freely both directions after.
# Param[15] (J6) fitted to -0.18 this time, in line with the other 15 values
# (vs. the first run's Param[15]=10.68 outlier, ~5x the next-largest param,
# which is the strongly-suspected cause of that run's J6 fault). data/
# gravity_params.npy / gravity_params_meta.npz now hold THIS good fit,
# overwriting the earlier bad one. Still worth a fresh eye each time this is
# re-run: check the fitted Param[15] (and the others) aren't a lone outlier
# before trusting it, and physically check joystick + J6 both directions
# after Step 3/4 regardless of how good the before/after numbers look — the
# first bad run also "looked fine" in the printed report.
#
# ORIGINAL RESULT (2026-07-02, first run, PARAMS SINCE OVERWRITTEN — kept here
# for context only): after applying that fit, the joystick showed fault
# lights and J6 could only rotate in ONE direction. Root cause suspected:
# this arm's torque sensors are almost certainly not calibrated via Kinova's
# Development Center tool (same conclusion independently reached from
# teach_by_hand.py's torque-control failures), which is this routine's own
# documented precondition — that may still be true in general, this second
# run's cleaner fit doesn't rule it out, it just didn't manifest as a fault
# this time. See project_force_sensing.md memory for the full timeline.
################################################################################

Fix the Kinova Jaco2 firmware gravity compensator at the source, using the
SDK's built-in gravity parameter estimation routine (RunGravityZEstimationSequence),
instead of (or before) fitting a software regressor for the residual.

READ THIS BEFORE RUNNING STEP 3 (the actual estimation call)
  RunGravityZEstimationSequence has NO documentation in the SDK headers at all
  (verified below — this script prints exactly what it found). The only real
  documentation exists in the reference kinova-ros driver's docstring for its
  wrapper (runCOMParameterEstimation), which says, verbatim:

    "The arm must be in Trajectory-Position mode before to launch the
    procedure. Before using this procedure, you should make sure that the
    torque sensors are well calibrated. [...] When the program is launched,
    the robot will execute a trajectory. The user must remain alert and turn
    off the robot if something wrong occurs (for example if the robot
    collides with an object)."

  Two things follow from that:
  1. This is an AUTONOMOUS, UNDOCUMENTED, BLOCKING trajectory this script
     cannot preview, clamp, or abort in software once started — unlike every
     other motion command in this pipeline (dry-run defaults, step clamps,
     convergence polling), this is a single opaque SDK call. Kinova's own
     stated safety mechanism is "be ready to power the arm off."
  2. It documents CALIBRATED TORQUE SENSORS as a precondition. We have
     first-hand evidence from this session (teach_by_hand.py: torque mode
     engaged per every return code and GetTrajectoryTorqueMode() readback,
     yet the arm stayed rigid and unbackdrivable) that this arm's torque
     sensors are very likely NOT properly calibrated via Kinova's Development
     Center tool (which this USB SDK does not expose). That means this
     routine's own documented precondition is probably unmet on this
     hardware — it may run, move the arm, and still produce a poor fit.

  Given that, this script still builds and runs everything you asked for, but
  Step 3 requires an explicit, informed Enter-press confirmation, printed
  right after the warning above.

WHAT THIS SCRIPT DOES (steps, per spec)
  1. SDK discovery — connect, and scan+print exactly what the headers say
     about gravity-related functions (no guessing: read first, print, then act).
  2. Pre-calibration baseline — 8 poses (clamped to real joint limits from
     j2s6s300.xacro), settle-checked (qdot < 1 deg/s for >= 1.5s), 200-sample
     averaged GetAngularForceGravityFree -> ||F|| via contact_detector.torque_to_wrench
     (imported, not reimplemented). Saved to data/gravity_baseline_before.npy.
  3. Gravity estimation — RunGravityZEstimationSequence, then SetGravityOptimalZParam
     + SetGravityType(OPTIMAL) to activate it. Gated behind the warning above.
  4. Post-calibration — identical 8-pose measurement, data/gravity_baseline_after.npy.
  5. Before/after report + verdict (thresholds: <0.5N sufficient, 0.5-1.0N
     meaningful-but-consider-regressor, >1.0N regressor-needed).
  6. Persistence — NOT documented locally whether params survive a power
     cycle (see Step 6 output). Saved unconditionally to data/gravity_params.npy
     with a loader (apply_saved_gravity_params) you can call at the start of
     any session regardless of which case turns out to be true.

Does NOT touch contact_detector.py (per spec — that's a separate step if this
firmware fix leaves meaningful residual). Does NOT use the moving/EMA baseline
anywhere. mode='joint' from contact_detector is not needed here — this script
reports ||F|| via torque_to_wrench directly, matching the spec's table format.

USAGE
  python calibrate_firmware_gravity.py
  python calibrate_firmware_gravity.py --robot_type 7   # override auto-detected ROBOT_TYPE
"""
import os, sys, ctypes, time, argparse, re
import numpy as np

_LIB_DIR = os.path.join(os.path.expanduser('~/working_dir/kinovaDrivers'), 'sdk', 'lib')
_INCLUDE_DIR = os.path.join(os.path.expanduser('~/working_dir/kinovaDrivers'), 'sdk', 'include')
if _LIB_DIR not in os.environ.get('LD_LIBRARY_PATH', '').split(':'):
    os.environ['LD_LIBRARY_PATH'] = _LIB_DIR + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    os.execv(sys.executable, [sys.executable] + sys.argv)

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)
from contact_detector import torque_to_wrench

LIB_PATH      = os.path.join(_LIB_DIR, 'USBCommandLayerUbuntu.so')
COMM_LIB_PATH = os.path.join(_LIB_DIR, 'USBCommLayerUbuntu.so')
HEADER_PATH   = os.path.join(_INCLUDE_DIR, 'Kinova.API.USBCommandLayerUbuntu.h')

NO_ERROR_KINOVA, SERIAL_LENGTH, MAX_KINOVA_DEVICE = 1, 20, 20
ANGULAR_POSITION, NOMOVEMENT_HAND = 2, 0

# GRAVITY_TYPE enum (KinovaTypes.h): MANUAL_INPUT=0, OPTIMAL=1 ("via an automatic routine")
GRAVITY_MANUAL_INPUT, GRAVITY_OPTIMAL = 0, 1
# Sizes, verbatim from KinovaTypes.h / Kinova.API.USBCommandLayerUbuntu.h #defines
OPTIMAL_Z_PARAM_SIZE = 16     # RunGravityZEstimationSequence's raw output (double[16])
GRAVITY_PARAM_SIZE   = 42     # SetGravityOptimalZParam's expected input (float[42])
# ROBOT_TYPE enum (KinovaTypes.h) — SPHERICAL_6DOF_SERVICE=7 confirmed this session by
# matching the connected device's printed Model string ("Spherical 6DOF Serv").
ROBOT_TYPE_MAP = {
    'JACOV1_ASSISTIVE': 0, 'MICO_6DOF_SERVICE': 1, 'MICO_4DOF_SERVICE': 2,
    'JACOV2_6DOF_SERVICE': 3, 'JACOV2_4DOF_SERVICE': 4, 'MICO_6DOF_ASSISTIVE': 5,
    'JACOV2_6DOF_ASSISTIVE': 6, 'SPHERICAL_6DOF_SERVICE': 7, 'SPHERICAL_7DOF_SERVICE': 8,
}

# Real per-joint limits. J1, J4, J6 are "continuous" joints (no hard limit,
# cumulative angle — matches observed readings like q4=408.4 deg this
# session). J2, J3, J5 are "revolute" with real hard limits.
#
# J5 CORRECTED (2026-07-02, from real hardware + independent user-supplied
# spec): kinova-ros's j2s6s300.xacro documents J5's range as 30-330deg, but
# that does NOT match this physical arm. Two independent calibrate_gravity_
# residual.py aborts (targets 301.4deg and 303.9deg, arrival errors 6.6deg
# and 9.2deg) both resolve to the same actual mechanical stop, ~294.7-294.8deg
# (target minus reported error, consistent to within 0.1deg across both) —
# user confirmed by hand this is a real physical limit, not a fault. User
# separately supplied an official-spec range of 65-295deg for J5, which
# matches the empirically-derived upper bound almost exactly (295 vs
# ~294.7-294.8) — strong independent corroboration. Using 293/65 (~2deg
# margin below the stated hard limits) rather than the exact edges.
#
# J2/J3: user also supplied "+-47deg from vertical" (J2) and "+-74.5deg from
# fully extended" (J3), but these reference a physical configuration ("vertical"
# / "fully extended") whose raw-angle offset is NOT YET RECONCILED against
# GetAngularPosition()'s convention -- the natural guess (reference = raw
# 180deg) is directly CONTRADICTED by already-successful poses this session
# (J2 commanded to 230.8, 231.1, 229.0, 225.5deg all converged cleanly, all
# exceeding a hypothetical 227deg upper bound under that guess). Left as the
# URDF values until the reference offset is confirmed -- do not guess.
JOINT_LIMITS_DEG = {
    1: None,            # continuous
    2: (47.0, 313.0),   # from URDF -- NOT yet reconciled with user-supplied "+-47deg from vertical"
    3: (19.0, 341.0),   # from URDF -- NOT yet reconciled with user-supplied "+-74.5deg from fully extended"
    4: None,            # continuous
    5: (65.0, 293.0),   # corrected from real hardware + corroborating user-supplied spec, see above
    6: None,            # continuous
}
# Rated velocity limits, same source: J1-3 = 36 deg/s, J4-6 = 48 deg/s.
JOINT_VEL_LIMIT_DPS = {1: 36.0, 2: 36.0, 3: 36.0, 4: 48.0, 5: 48.0, 6: 48.0}
SAFE_SPEED_DPS = 15.0   # default commanded speed — well under half the rated limits,
                        # matches the same conservative default used in
                        # test_residual_repeatability.py's commanded mode.

# NOTE: the original spec's poses (all with J5=180 deg) sit essentially exactly
# on the spherical-wrist singularity (J4/J6 axes align there) -- verified via
# contact_detector.compute_jacobian: cond(J) ~1e17-1e18 at every one of them.
# torque_to_wrench's pinv(J^T) blows those up into meaningless multi-million-
# Newton "forces" (confirmed on real hardware via test_residual_repeatability.py
# before this fix). Replaced with poses generated from this arm's actual
# observed working configurations this session, filtered to cond(J) ~8-9 (see
# test_residual_repeatability.py's data/repeat_poses.npy generation) — these
# are the SAME poses already verified REPEATABLE on hardware (peak-to-peak
# 0.04-0.45 N across 5 runs), so Step 2/4 here will produce sane numbers.
TEST_POSES_DEG = np.array([
    [180.5, 217.7,  64.7, 329.4, 253.2, 222.9],
    [161.9, 215.6,  56.2, 308.1, 261.0, 251.9],
    [168.2, 207.0,  57.7, 299.4, 260.1, 229.6],
    [168.1, 213.0,  80.9, 329.0, 259.5, 232.2],
    [164.5, 223.1,  53.7, 325.5, 259.8, 243.0],
    [185.7, 206.6,  74.5, 321.5, 248.7, 249.7],
    [149.1, 206.7,  68.6, 389.3, 238.4, 119.7],
    [186.7, 211.4,  82.8, 319.0, 245.6, 217.9],
], dtype=np.float64)


# ════════════════════════════════════════════════════════════════════════════
# ctypes structs — exact pattern from the existing codebase
# ════════════════════════════════════════════════════════════════════════════

class KinovaDevice(ctypes.Structure):
    _fields_ = [('SerialNumber', ctypes.c_char * SERIAL_LENGTH),
                ('Model', ctypes.c_char * SERIAL_LENGTH),
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

class CartesianInfo(ctypes.Structure):
    _fields_ = [('X', ctypes.c_float), ('Y', ctypes.c_float), ('Z', ctypes.c_float),
                ('ThetaX', ctypes.c_float), ('ThetaY', ctypes.c_float), ('ThetaZ', ctypes.c_float)]

class UserPosition(ctypes.Structure):
    _fields_ = [('Type', ctypes.c_int), ('Delay', ctypes.c_float),
                ('CartesianPosition', CartesianInfo), ('Actuators', AngularInfo),
                ('HandMode', ctypes.c_int), ('Fingers', FingersPosition)]

class Limitation(ctypes.Structure):
    _fields_ = [('speedParameter1', ctypes.c_float), ('speedParameter2', ctypes.c_float),
                ('speedParameter3', ctypes.c_float), ('forceParameter1', ctypes.c_float),
                ('forceParameter2', ctypes.c_float), ('forceParameter3', ctypes.c_float),
                ('accelerationParameter1', ctypes.c_float), ('accelerationParameter2', ctypes.c_float),
                ('accelerationParameter3', ctypes.c_float)]

class TrajectoryPoint(ctypes.Structure):
    _fields_ = [('Position', UserPosition), ('LimitationsActive', ctypes.c_int),
                ('SynchroType', ctypes.c_int), ('Limitations', Limitation)]


def load_api():
    ctypes.CDLL(COMM_LIB_PATH, mode=ctypes.RTLD_GLOBAL)
    api = ctypes.CDLL(LIB_PATH)
    for fn in ('InitAPI', 'CloseAPI', 'RefresDevicesList', 'GetDevices', 'SetActiveDevice',
               'StartControlAPI', 'StopControlAPI', 'SetAngularControl',
               'GetAngularPosition', 'GetAngularVelocity', 'GetAngularForceGravityFree',
               'SendBasicTrajectory', 'EraseAllTrajectories',
               'RunGravityZEstimationSequence', 'SetGravityOptimalZParam', 'SetGravityType'):
        getattr(api, fn).restype = ctypes.c_int
    api.SendBasicTrajectory.argtypes = [TrajectoryPoint]
    api.SetGravityType.argtypes = [ctypes.c_int]
    api.SetGravityOptimalZParam.argtypes = [ctypes.c_float * GRAVITY_PARAM_SIZE]
    api.RunGravityZEstimationSequence.argtypes = [ctypes.c_int, ctypes.c_double * OPTIMAL_Z_PARAM_SIZE]
    return api


def ok(r):
    return r == NO_ERROR_KINOVA


def check(r, what, tolerate=()):
    """Print + abort on unexpected SDK error codes. `tolerate` mirrors the
    reference driver's own precedent (e.g. it accepts 2005 from the two
    SetGravity*Param calls specifically — not a guess, copied from
    kinova_comm.cpp's runCOMParameterEstimation)."""
    if r == NO_ERROR_KINOVA or r in tolerate:
        print(f'  {what} -> {r} OK' + (' (tolerated)' if r in tolerate else ''))
        return True
    print(f'\n[ABORT] {what} returned error code {r} (NO_ERROR is {NO_ERROR_KINOVA}).')
    return False


# ════════════════════════════════════════════════════════════════════════════
# Step 1 — SDK discovery: read headers, print what's found, then connect
# ════════════════════════════════════════════════════════════════════════════

def scan_headers_for_gravity():
    print('=' * 78)
    print(f'STEP 1a — scanning {HEADER_PATH} for gravity-related declarations')
    print('=' * 78)
    if not os.path.exists(HEADER_PATH):
        print(f'  Header not found at expected path: {HEADER_PATH}')
        return
    text = open(HEADER_PATH).read()
    pattern = re.compile(r'^\s*extern\s+"C".*Gravity\w*.*;', re.MULTILINE | re.IGNORECASE)
    matches = pattern.findall(text)
    if not matches:
        print('  No gravity-related function declarations found.')
    for m in matches:
        print(f'  {m.strip()}')
    # Also surface the ONE real doc comment that exists for this routine family —
    # it lives in the kinova-ros driver source, not the header itself.
    print('\n  NOTE: the header above has NO doc comments for these functions.')
    print('  The only real documentation found locally is in kinova-ros\'s')
    print('  kinova_comm.cpp (runCOMParameterEstimation docstring) — see the')
    print('  warning printed before Step 3.')
    print()


def connect(api, control: bool):
    r = api.InitAPI()
    if not ok(r):
        raise SystemExit(f'InitAPI() failed: {r}')
    api.RefresDevicesList()
    devs = (KinovaDevice * MAX_KINOVA_DEVICE)()
    err = ctypes.c_int(NO_ERROR_KINOVA)
    if api.GetDevices(devs, ctypes.byref(err)) == 0:
        api.CloseAPI(); raise SystemExit('No Kinova device found')
    api.SetActiveDevice(devs[0])
    model = devs[0].Model.decode()
    serial = devs[0].SerialNumber.decode()
    if control:
        api.StartControlAPI(); api.StopControlAPI(); api.StartControlAPI()
        # Kinova's documented precondition: "arm must be in Trajectory-Position
        # mode" before RunGravityZEstimationSequence. SetAngularControl puts
        # commands in angular (joint-space) position/trajectory mode.
        api.SetAngularControl()
    print(f'STEP 1b — Connected: {model} (serial {serial})'
          + ('  [ANGULAR/TRAJECTORY-POSITION MODE]' if control else '  [READ-ONLY]'))
    return model, serial


def detect_robot_type(model: str, override: int = None) -> int:
    if override is not None:
        print(f'  --robot_type override: {override}')
        return override
    m = model.lower()
    if 'spherical' in m and '6' in m:
        rt = ROBOT_TYPE_MAP['SPHERICAL_6DOF_SERVICE']
        print(f'  Auto-detected ROBOT_TYPE from Model="{model}" -> '
              f'SPHERICAL_6DOF_SERVICE ({rt})')
        return rt
    if 'spherical' in m and '7' in m:
        rt = ROBOT_TYPE_MAP['SPHERICAL_7DOF_SERVICE']
        print(f'  Auto-detected ROBOT_TYPE from Model="{model}" -> '
              f'SPHERICAL_7DOF_SERVICE ({rt})')
        return rt
    raise SystemExit(f'Could not auto-detect ROBOT_TYPE from Model="{model}". '
                     f'Pass --robot_type explicitly (see ROBOT_TYPE_MAP in this file).')


# ════════════════════════════════════════════════════════════════════════════
# Shared: motion, settle-check, sampling
# ════════════════════════════════════════════════════════════════════════════

def read6(api, fn):
    s = AngularPosition()
    getattr(api, fn)(ctypes.byref(s))
    a = s.Actuators
    return np.array([a.Actuator1, a.Actuator2, a.Actuator3,
                     a.Actuator4, a.Actuator5, a.Actuator6], np.float64)


def get_q(api):
    return read6(api, 'GetAngularPosition')


def get_qdot(api):
    return read6(api, 'GetAngularVelocity')


def get_tau_gf(api):
    return read6(api, 'GetAngularForceGravityFree')


def angle_diff_deg(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def clamp_pose(q_deg: np.ndarray) -> np.ndarray:
    """Clamp each joint to its real hard limit (JOINT_LIMITS_DEG). J1/J4/J6 are
    continuous (None) and left unclamped. Prints a warning per joint clamped."""
    out = q_deg.copy()
    for i in range(6):
        lim = JOINT_LIMITS_DEG[i + 1]
        if lim is None:
            continue
        lo, hi = lim
        if not (lo <= out[i] <= hi):
            clamped = float(np.clip(out[i], lo, hi))
            print(f'  WARNING: J{i+1}={out[i]:.1f} deg outside limit [{lo},{hi}] '
                 f'— clamping to {clamped:.1f}')
            out[i] = clamped
    return out


def send_angular(api, q_deg, speed_dps):
    tp = TrajectoryPoint()
    ctypes.memset(ctypes.byref(tp), 0, ctypes.sizeof(tp))
    tp.Position.Type = ANGULAR_POSITION
    tp.Position.HandMode = NOMOVEMENT_HAND
    for i in range(6):
        setattr(tp.Position.Actuators, f'Actuator{i+1}', float(q_deg[i]))
    if speed_dps > 0:
        tp.LimitationsActive = 1
        tp.Limitations.speedParameter1 = float(speed_dps)
        tp.Limitations.speedParameter2 = float(speed_dps)
    api.SendBasicTrajectory(tp)


def move_and_settle(api, target_deg, speed_dps=SAFE_SPEED_DPS,
                    converge_deg=1.0, converge_timeout=25.0,
                    settle_vel_dps=1.0, settle_hold_s=1.5, settle_timeout=10.0):
    """Send an angular position command, wait for arrival, then require
    joint velocity to stay below settle_vel_dps for settle_hold_s continuous
    seconds (per spec) before returning."""
    send_angular(api, target_deg, speed_dps)
    tw = time.time()
    while time.time() - tw < converge_timeout:
        if np.max(np.abs(angle_diff_deg(get_q(api), target_deg))) < converge_deg:
            break
        time.sleep(0.05)
    arrived_err = np.max(np.abs(angle_diff_deg(get_q(api), target_deg)))

    t_settle_start = time.time()
    hold_start = None
    while time.time() - t_settle_start < settle_timeout:
        v = np.max(np.abs(get_qdot(api)))
        if v < settle_vel_dps:
            if hold_start is None:
                hold_start = time.time()
            elif time.time() - hold_start >= settle_hold_s:
                return arrived_err, True
        else:
            hold_start = None
        time.sleep(0.05)
    return arrived_err, False   # settle timed out


def capture_visit(api, n_samples=200, hz=100.0):
    taus = []
    dt = 1.0 / hz
    for _ in range(n_samples):
        taus.append(get_tau_gf(api))
        time.sleep(dt)
    return np.stack(taus).mean(axis=0)


# ════════════════════════════════════════════════════════════════════════════
# Step 2 / Step 4 — 8-pose baseline measurement (shared)
# ════════════════════════════════════════════════════════════════════════════

def measure_baseline(api, label: str, out_path: str):
    print('\n' + '=' * 78)
    print(f'{label} — measuring at {len(TEST_POSES_DEG)} poses')
    print('=' * 78)
    records = []
    for i, raw_pose in enumerate(TEST_POSES_DEG):
        pose = clamp_pose(raw_pose)
        print(f'\n[pose {i}] target: {np.round(pose, 1)}')
        err, settled = move_and_settle(api, pose)
        if not settled:
            print(f'  WARNING: velocity never settled below 1 deg/s for 1.5s — '
                 f'sampling anyway, treat this pose\'s reading with caution.')
        tau_gf = capture_visit(api, n_samples=200, hz=100.0)
        q_actual = get_q(api)
        F = float(np.linalg.norm(torque_to_wrench(q_actual, tau_gf)[:3]))
        print(f'  arrived (max err {err:.2f} deg)   ||F|| = {F:.3f} N')
        records.append(dict(pose_id=i, q=q_actual, tau_gf=tau_gf, Fnorm=F))

    print(f'\n{"pose":>4} {"joint angles (deg)":>42} {"||F|| (N)":>10}')
    for r in records:
        print(f'{r["pose_id"]:>4} {np.array2string(r["q"], precision=1):>42} {r["Fnorm"]:>10.3f}')

    np.save(out_path, np.array(records, dtype=object))
    print(f'\nSaved -> {out_path}')
    return records


# ════════════════════════════════════════════════════════════════════════════
# Step 3 — gravity estimation
# ════════════════════════════════════════════════════════════════════════════

GRAVITY_ESTIMATION_WARNING = """
################################################################################
STEP 3 — RunGravityZEstimationSequence: READ THIS BEFORE CONTINUING
################################################################################

Verbatim from kinova-ros's kinova_comm.cpp (the only real documentation found
locally for this routine — the SDK header itself has NO comment for it):

  "The arm must be in Trajectory-Position mode before to launch the
  procedure. Before using this procedure, you should make sure that the
  torque sensors are well calibrated. This procedure is explained in the
  user guide and in the Advanced Specification Guide. When the program is
  launched, the robot will execute a trajectory. The user must remain alert
  and turn off the robot if something wrong occurs (for example if the robot
  collides with an object)."

TWO THINGS THIS SCRIPT CANNOT DO ANYTHING ABOUT:

  1. This call moves the arm AUTONOMOUSLY through an undocumented trajectory.
     It is a single blocking SDK call — this script cannot preview, clamp, or
     abort it once started, unlike every other motion command in this
     pipeline. Kinova's own stated safety mechanism is "be ready to power the
     arm off," not a software abort.

  2. This session already found strong evidence (teach_by_hand.py) that this
     arm's torque sensors are very likely NOT properly calibrated via
     Kinova's Development Center tool (torque mode engaged per every return
     code and the GetTrajectoryTorqueMode() readback, yet the arm stayed
     rigid). That directly conflicts with this routine's documented
     precondition ("make sure the torque sensors are well calibrated"). It
     may still run and move the arm, but the fitted parameters may not be
     good — you will not know until Step 5's before/after report.

Make sure the arm's full range of motion for the 8 test poses is clear of
obstacles and people before continuing.
################################################################################
"""


def run_gravity_estimation(api, robot_type: int):
    print(GRAVITY_ESTIMATION_WARNING)
    input('Press Enter to confirm the arm is clear and start the estimation sequence '
         '(Ctrl+C to abort)... ')

    params16 = (ctypes.c_double * OPTIMAL_Z_PARAM_SIZE)()
    print('\nCalling RunGravityZEstimationSequence — this blocks until the arm\'s '
         'internal trajectory finishes. Duration is undocumented; please wait...')
    t0 = time.time()
    r = api.RunGravityZEstimationSequence(robot_type, params16)
    elapsed = time.time() - t0
    print(f'RunGravityZEstimationSequence returned after {elapsed:.1f}s -> {r}')
    if not check(r, 'RunGravityZEstimationSequence'):
        return None

    fitted = np.array(list(params16), dtype=np.float64)
    print(f'Fitted OptimalzParam (16 values): {np.round(fitted, 4)}')

    # Reference driver's exact pattern (kinova_comm.cpp::runCOMParameterEstimation):
    # SetGravityOptimalZParam expects the FULL 42-element GRAVITY_PARAM_SIZE buffer,
    # not the 16-element estimation output directly (size mismatch — this is NOT a
    # guess, copied from the reference implementation). Zero-pad the remaining 26.
    com_params42 = (ctypes.c_float * GRAVITY_PARAM_SIZE)()
    for i in range(GRAVITY_PARAM_SIZE):
        com_params42[i] = 0.0
    for i in range(OPTIMAL_Z_PARAM_SIZE):
        com_params42[i] = float(fitted[i])

    # Reference driver tolerates result code 2005 specifically for this call —
    # copied precedent, not a guess.
    r = api.SetGravityOptimalZParam(com_params42)
    if not check(r, 'SetGravityOptimalZParam', tolerate=(2005,)):
        return None

    r = api.SetGravityType(GRAVITY_OPTIMAL)
    if not check(r, 'SetGravityType(OPTIMAL)'):
        return None

    # Reference driver docs: output also written to "ParametersOptimal_Z.txt" in
    # "the program folder" — check cwd for it as independent confirmation.
    for candidate in ('ParametersOptimal_Z.txt',
                      os.path.join(PIPELINE_DIR, 'ParametersOptimal_Z.txt')):
        if os.path.exists(candidate):
            print(f'Found {candidate} (written by the SDK routine itself) — '
                 f'independent confirmation it actually ran.')

    return fitted, com_params42


# ════════════════════════════════════════════════════════════════════════════
# Step 5 — report
# ════════════════════════════════════════════════════════════════════════════

def report(before, after):
    print('\n' + '=' * 78)
    print('STEP 5 — BEFORE / AFTER REPORT')
    print('=' * 78)
    print(f'{"pose":>4} {"||F|| before (N)":>17} {"||F|| after (N)":>16} '
          f'{"reduction (N)":>14} {"reduction (%)":>14}')
    reductions = []
    for b, a in zip(before, after):
        db = b['Fnorm']; da = a['Fnorm']
        red = db - da
        pct = (red / db * 100.0) if db > 1e-9 else 0.0
        reductions.append(red)
        print(f'{b["pose_id"]:>4} {db:>17.3f} {da:>16.3f} {red:>14.3f} {pct:>13.1f}%')

    mean_before = float(np.mean([b['Fnorm'] for b in before]))
    mean_after  = float(np.mean([a['Fnorm'] for a in after]))
    print(f'\nMean ||F|| before: {mean_before:.3f} N')
    print(f'Mean ||F|| after : {mean_after:.3f} N')

    if mean_after < 0.5:
        verdict = 'Firmware calibration sufficient — regressor likely unnecessary.'
    elif mean_after <= 1.0:
        verdict = 'Meaningful improvement — consider regressor for remaining residual.'
    else:
        verdict = 'Limited improvement — regressor will be needed.'
    print(f'\nVERDICT: {verdict}')
    return mean_before, mean_after, verdict


# ════════════════════════════════════════════════════════════════════════════
# Step 6 — persistence
# ════════════════════════════════════════════════════════════════════════════

PERSISTENCE_NOTE = """
STEP 6 — persistence across power cycles
  NOT documented in any locally available header or driver source — the
  kinova-ros driver docstring references an external "user guide" and
  "Advanced Specification Guide" this repo does not have. We cannot claim a
  definitive answer without a real power-cycle test.

  Safest assumption: treat these as SESSION parameters that may need
  reapplication. Saved to data/gravity_params.npy either way (cheap
  insurance), and apply_saved_gravity_params() below reapplies them via
  SetGravityOptimalZParam + SetGravityType(OPTIMAL) at the start of any
  session — harmless to call even if they DID persist.

  Suggested manual check: power-cycle the arm, then re-run just Step 2's
  measurement at one pose. If ||F|| is still low (matches this session's
  "after"), parameters persisted. If it's back near the "before" level,
  they didn't — call apply_saved_gravity_params() at the start of every
  session that needs the calibrated model.
"""


def save_gravity_params(com_params42, mean_before, mean_after, robot_type, path):
    arr = np.array(list(com_params42), dtype=np.float32)
    np.save(path, arr)
    meta_path = path.replace('.npy', '_meta.npz')
    np.savez(meta_path, date=time.strftime('%Y-%m-%d %H:%M:%S'),
             mean_before=mean_before, mean_after=mean_after, robot_type=robot_type)
    print(f'Saved fitted 42-element gravity params -> {path}')
    print(f'Saved metadata -> {meta_path}')


def apply_saved_gravity_params(api, path=os.path.join(PIPELINE_DIR, 'data', 'gravity_params.npy')):
    """Call at the start of any session to reapply a previously fitted
    firmware gravity model. Harmless to call even if parameters already
    persisted across a power cycle.

    CONFIRMED (2026-07-02): parameters do NOT persist across whatever cleared
    the J6 fault (power cycle / joystick fault-reset) — ||F|| at a reference
    pose was measured back at the pre-calibration ~16.8N until this was
    called again. This function must run at the start of every session that
    needs the calibrated model."""
    if not os.path.exists(path):
        print(f'  apply_saved_gravity_params: no saved params at {path} — skipping '
             f'(run calibrate_firmware_gravity.py first to produce one).')
        return False
    arr = np.load(path).astype(np.float32)
    buf = (ctypes.c_float * GRAVITY_PARAM_SIZE)(*arr)
    r1 = api.SetGravityOptimalZParam(buf)
    # SetGravityOptimalZParam normally returns 2005 on this hardware (same
    # tolerance as run_gravity_estimation's check(..., tolerate=(2005,)) —
    # a strict ok(r1) here previously reported FAILED even when the call
    # demonstrably worked (residual dropped from 16.8N to 1.85N regardless).
    r2 = api.SetGravityType(GRAVITY_OPTIMAL)
    return (ok(r1) or r1 == 2005) and ok(r2)


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--robot_type', type=int, default=None,
                   help='Override auto-detected ROBOT_TYPE enum value (see ROBOT_TYPE_MAP).')
    p.add_argument('--before_path', default='data/gravity_baseline_before.npy')
    p.add_argument('--after_path', default='data/gravity_baseline_after.npy')
    p.add_argument('--params_path', default='data/gravity_params.npy')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.join(PIPELINE_DIR, 'data'), exist_ok=True)
    before_path = os.path.join(PIPELINE_DIR, args.before_path)
    after_path  = os.path.join(PIPELINE_DIR, args.after_path)
    params_path = os.path.join(PIPELINE_DIR, args.params_path)

    scan_headers_for_gravity()

    api = load_api()
    try:
        model, serial = connect(api, control=True)
        robot_type = detect_robot_type(model, args.robot_type)

        before = measure_baseline(api, 'STEP 2 — PRE-CALIBRATION BASELINE', before_path)

        result = run_gravity_estimation(api, robot_type)
        if result is None:
            print('\nGravity estimation failed or was aborted — stopping before Step 4.')
            return
        fitted16, com_params42 = result

        after = measure_baseline(api, 'STEP 4 — POST-CALIBRATION MEASUREMENT', after_path)

        mean_before, mean_after, verdict = report(before, after)

        print(PERSISTENCE_NOTE)
        save_gravity_params(com_params42, mean_before, mean_after, robot_type, params_path)

    finally:
        api.EraseAllTrajectories()
        api.CloseAPI()
        print('\nAPI closed.')


if __name__ == '__main__':
    main()
