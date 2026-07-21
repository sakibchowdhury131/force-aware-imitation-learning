#!/usr/bin/env python3
"""
Kinesthetic teaching for the Kinova Jaco2 — puts the arm into direct torque
control with only a small velocity-damping command (no active positioning
torque beyond gravity compensation + damping), so you can physically grab the
tool/gripper and move the arm by hand. Records the joint trajectory while
you do so.

HOW IT WORKS
  Kinova's firmware compensates gravity automatically in torque mode (same
  estimate behind GetAngularForceGravityFree). Commanding torque = 0 means
  "hold up against gravity, no extra push" -- so the arm just floats and can
  be pushed around. We command a small DAMPING torque (tau = -damping * qdot)
  instead of a literal zero, so it doesn't feel floaty/oscillatory when let go.
  This is NOT a spring back to a setpoint -- there's no restoring force, so
  the arm stays wherever your hand leaves it.

SAFETY
  - Read the whole docstring before running this on real hardware.
  - The gravity model does NOT know about the attached tool's mass unless you
    pass --payload_mass/--payload_com -- expect the arm to sag or resist in
    the tool's direction otherwise (we measured ~10.5 N*m of unmodeled bias
    on J6 earlier via calibrate_contact_baseline.py -- same root cause here).
  - HOLD THE TOOL/GRIPPER FIRMLY before torque mode engages -- there's a
    brief settling period and the arm may shift slightly on the mode switch.
  - The torque command loop runs in ONE dedicated background thread (the
    Kinova SDK is not documented thread-safe for concurrent calls -- the
    official kinova-ros driver wraps every single call in a mutex). If that
    loop ever throws, its `finally` immediately switches back to POSITION
    control before the thread exits -- but there's no substitute for keeping
    a hand ready and the e-stop within reach, especially the first run.
  - --max_command_torque is a software safety clamp independent of firmware
    limits, applied every tick, in addition to --safety_factor.
  - Always exits back to POSITION control mode, even on Ctrl+C or a crash.

USAGE
  # Just try it, no payload compensation, defaults
  python teach_by_hand.py --task pokeTask

  # With the spoon's mass roughly compensated (weigh it first!)
  python teach_by_hand.py --task pokeTask --payload_mass 0.15 --payload_com 0 0 0.05

Interactive commands (typed + Enter, while the arm is compliant):
  r         toggle recording on/off
  s         save the current recording as a new episode, start a fresh buffer
  q         quit -- switches back to POSITION control and exits
"""
import os, sys, ctypes, time, threading, argparse, json
import numpy as np

_LIB_DIR = os.path.join(os.path.expanduser('~/working_dir/kinovaDrivers'), 'sdk', 'lib')
if _LIB_DIR not in os.environ.get('LD_LIBRARY_PATH', '').split(':'):
    os.environ['LD_LIBRARY_PATH'] = _LIB_DIR + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    os.execv(sys.executable, [sys.executable] + sys.argv)

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)
from calibrate_firmware_gravity import apply_saved_gravity_params

LIB_PATH      = os.path.join(_LIB_DIR, 'USBCommandLayerUbuntu.so')
COMM_LIB_PATH = os.path.join(_LIB_DIR, 'USBCommLayerUbuntu.so')
NO_ERROR_KINOVA   = 1
SERIAL_LENGTH     = 20
MAX_KINOVA_DEVICE = 20

# GENERALCONTROL_TYPE enum (KinovaTypes.h): POSITION=0, TORQUE=1
POSITION_MODE = 0
TORQUE_MODE   = 1
# TORQUECONTROL_TYPE enum (KinovaTypes.h): DIRECTTORQUE obeys exactly what we
# command; IMPEDANCEANGULAR/IMPEDANCECARTESIAN instead add a virtual
# spring/damper around wherever the arm was when torque mode engaged, which
# would feel like it's still holding position — NOT what a damping-only
# hand-guiding command wants. The reference kinova-ros driver never calls
# SetTorqueControlType at all, so we can't assume the firmware's power-on
# default is DIRECTTORQUE — set it explicitly.
DIRECTTORQUE = 0
# GRAVITY_TYPE enum (KinovaTypes.h): OPTIMAL selects the arm's own
# automatically-identified gravity model (same one behind
# GetAngularForceGravityFree) rather than MANUAL_INPUT (which needs an
# explicit 42-float per-link mass/COM table we don't have). The reference
# driver calls SetGravityType(OPTIMAL) unconditionally, early in startup,
# before anything torque-related — our script skipped this entirely.
GRAVITY_OPTIMAL = 1
# SendAngularTorqueCommand's C signature is float Command[COMMAND_SIZE] with
# COMMAND_SIZE=70 -- a large shared buffer type reused across several SDK
# command functions. Only indices [0:6] are meaningful for our 6-DOF arm.
COMMAND_ARRAY_SIZE = 70


class KinovaDevice(ctypes.Structure):
    _fields_ = [('SerialNumber', ctypes.c_char * SERIAL_LENGTH),
                ('Model',        ctypes.c_char * SERIAL_LENGTH),
                ('VersionMajor', ctypes.c_int), ('VersionMinor',   ctypes.c_int),
                ('VersionRelease', ctypes.c_int), ('DeviceType',   ctypes.c_int),
                ('DeviceID',     ctypes.c_int)]

class AngularInfo(ctypes.Structure):
    _fields_ = [(f'Actuator{i}', ctypes.c_float) for i in range(1, 8)]

class FingersPosition(ctypes.Structure):
    _fields_ = [('Finger1', ctypes.c_float), ('Finger2', ctypes.c_float),
                ('Finger3', ctypes.c_float)]

class AngularPosition(ctypes.Structure):
    _fields_ = [('Actuators', AngularInfo), ('Fingers', FingersPosition)]


def load_api():
    ctypes.CDLL(COMM_LIB_PATH, mode=ctypes.RTLD_GLOBAL)
    api = ctypes.CDLL(LIB_PATH)
    for fn in ('InitAPI', 'CloseAPI', 'RefresDevicesList', 'GetDevices', 'SetActiveDevice',
               'StartControlAPI', 'StopControlAPI', 'GetAngularPosition', 'GetAngularVelocity',
               'GetAngularForceGravityFree', 'SetTorqueSafetyFactor', 'SwitchTrajectoryTorque',
               'SendAngularTorqueCommand', 'SetGravityPayload', 'SetTorqueControlType',
               'GetTrajectoryTorqueMode', 'SetGravityType'):
        getattr(api, fn).restype = ctypes.c_int
    api.SwitchTrajectoryTorque.argtypes  = [ctypes.c_int]
    api.SetTorqueControlType.argtypes    = [ctypes.c_int]
    api.SetGravityType.argtypes          = [ctypes.c_int]
    api.SetTorqueSafetyFactor.argtypes   = [ctypes.c_float]
    api.SendAngularTorqueCommand.argtypes = [ctypes.c_float * COMMAND_ARRAY_SIZE]
    api.SetGravityPayload.argtypes       = [ctypes.c_float * 4]
    api.GetTrajectoryTorqueMode.argtypes = [ctypes.POINTER(ctypes.c_int)]
    return api


def ok(result):
    return result == NO_ERROR_KINOVA


def connect(api):
    print('Loading Kinova USB API...')
    r = api.InitAPI()
    if not ok(r):
        raise RuntimeError(f'InitAPI() = {r}')
    api.RefresDevicesList()
    devices = (KinovaDevice * MAX_KINOVA_DEVICE)()
    err = ctypes.c_int(NO_ERROR_KINOVA)
    n = api.GetDevices(devices, ctypes.byref(err))
    if n == 0:
        api.CloseAPI()
        raise RuntimeError('No Kinova device found')
    api.SetActiveDevice(devices[0])
    api.StartControlAPI()
    api.StopControlAPI()
    api.StartControlAPI()
    print(f'Connected: {devices[0].Model.decode()} ({devices[0].SerialNumber.decode()})')


def get_joint_angles_deg(api) -> np.ndarray:
    pos = AngularPosition()
    api.GetAngularPosition(ctypes.byref(pos))
    a = pos.Actuators
    return np.array([a.Actuator1, a.Actuator2, a.Actuator3,
                     a.Actuator4, a.Actuator5, a.Actuator6], dtype=np.float64)


def get_joint_velocity_deg(api) -> np.ndarray:
    pos = AngularPosition()
    api.GetAngularVelocity(ctypes.byref(pos))
    a = pos.Actuators
    return np.array([a.Actuator1, a.Actuator2, a.Actuator3,
                     a.Actuator4, a.Actuator5, a.Actuator6], dtype=np.float64)


def get_gravity_free_torque(api) -> np.ndarray:
    g = AngularPosition()
    api.GetAngularForceGravityFree(ctypes.byref(g))
    a = g.Actuators
    return np.array([a.Actuator1, a.Actuator2, a.Actuator3,
                     a.Actuator4, a.Actuator5, a.Actuator6], dtype=np.float64)


class SharedState:
    """All cross-thread state, guarded by one lock. The background torque
    loop is the ONLY thread that touches the Kinova API."""
    def __init__(self):
        self.lock = threading.Lock()
        self.recording = False
        self.should_quit = False
        self.buffer = []       # list of (t, q_deg, qdot_deg, tau_cmd, tau_gf)
        self.error = None
        self.last_loop_t = None


def torque_loop(api, state: SharedState, args):
    """Runs continuously once torque mode is engaged. Sends tau = -damping*qdot
    (software-clamped), ramped in over the first `ramp_seconds` to soften the
    position->torque mode transition. ALWAYS reverts to POSITION mode on the
    way out, success or failure, since leaving the arm in torque mode with no
    more commands coming is undefined."""
    dt = 1.0 / args.hz
    t0 = time.time()
    cmd_buf = (ctypes.c_float * COMMAND_ARRAY_SIZE)()
    try:
        while True:
            with state.lock:
                if state.should_quit:
                    break
            t_start = time.time()
            elapsed_since_start = t_start - t0
            ramp = min(1.0, elapsed_since_start / max(args.ramp_seconds, 1e-6))

            qdot_deg = get_joint_velocity_deg(api)
            q_deg    = get_joint_angles_deg(api)
            qdot_rad = np.deg2rad(qdot_deg)

            tau_cmd = -args.damping * ramp * qdot_rad
            tau_cmd = np.clip(tau_cmd, -args.max_command_torque, args.max_command_torque)

            for i in range(COMMAND_ARRAY_SIZE):
                cmd_buf[i] = 0.0
            for i in range(6):
                cmd_buf[i] = float(tau_cmd[i])
            api.SendAngularTorqueCommand(cmd_buf)

            with state.lock:
                state.last_loop_t = time.time()
                if state.recording:
                    tau_gf = get_gravity_free_torque(api)
                    state.buffer.append((elapsed_since_start, q_deg.copy(),
                                         qdot_deg.copy(), tau_cmd.copy(), tau_gf.copy()))

            sleep_left = dt - (time.time() - t_start)
            if sleep_left > 0:
                time.sleep(sleep_left)
    except Exception as e:
        with state.lock:
            state.error = repr(e)
            state.should_quit = True
    finally:
        try:
            api.SwitchTrajectoryTorque(POSITION_MODE)
            print('\n[torque_loop] Reverted to POSITION control mode.')
        except Exception as e:
            print(f'\n[torque_loop] WARNING: failed to revert to POSITION mode: {e!r}')


def save_episode(state: SharedState, out_root: str, episode_idx: int, args) -> bool:
    with state.lock:
        if not state.buffer:
            print('Nothing recorded yet — press "r" to start recording first.')
            return False
        buf = state.buffer
        state.buffer = []
        state.recording = False

    ep_dir = os.path.join(out_root, f'{episode_idx:03d}')
    os.makedirs(ep_dir, exist_ok=True)
    t_arr      = np.array([b[0] for b in buf])
    q_arr      = np.stack([b[1] for b in buf])
    qdot_arr   = np.stack([b[2] for b in buf])
    tau_cmd_arr = np.stack([b[3] for b in buf])
    tau_gf_arr  = np.stack([b[4] for b in buf])

    np.savez(os.path.join(ep_dir, 'joint_trajectory.npz'),
             t=t_arr, q_deg=q_arr, qdot_deg=qdot_arr,
             tau_cmd=tau_cmd_arr, tau_gravity_free=tau_gf_arr)
    with open(os.path.join(ep_dir, 'meta.json'), 'w') as f:
        json.dump({'task': args.task, 'episode': f'{episode_idx:03d}',
                   'n_samples': len(buf), 'duration_sec': float(t_arr[-1] - t_arr[0]),
                   'hz_target': args.hz, 'damping': args.damping,
                   'max_command_torque': args.max_command_torque,
                   'safety_factor': args.safety_factor,
                   'payload_mass': args.payload_mass, 'payload_com': args.payload_com},
                  f, indent=2)
    print(f'Saved {len(buf)} samples ({t_arr[-1]-t_arr[0]:.1f}s) -> {ep_dir}/joint_trajectory.npz')
    return True


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', default='demo', help='Task name, used in the output path')
    p.add_argument('--output_dir', default=None,
                   help='Defaults to data/kinesthetic/<task>/')
    p.add_argument('--hz', type=float, default=100.0,
                   help='Target torque-command loop rate (Hz). Kinova torque control '
                        'wants a steady, fairly fast command stream (default 100 Hz).')
    p.add_argument('--damping', type=float, default=0.5,
                   help='N*m per rad/s of joint velocity — resistive-only, no spring-back '
                        'to a setpoint. Increase if it feels floaty/oscillatory when let go, '
                        'decrease if it feels too sticky to move.')
    p.add_argument('--max_command_torque', type=float, default=3.0,
                   help='Software safety clamp (N*m) applied to every joint every tick, '
                        'independent of firmware limits.')
    p.add_argument('--safety_factor', type=float, default=0.3,
                   help='SetTorqueSafetyFactor (0-1). CORRECTED (2026-07-02, from the '
                        'official Kinova SDK User Guide, Torque Console section): 0 = '
                        'MAXIMUM safety (auto-reverts to trajectory/position mode as soon '
                        'as actuator velocity exceeds a very low threshold), 1 = MINIMAL/NO '
                        'safety (never auto-reverts). The doc explicitly warns: "Do not set '
                        'Safety Factor to 1 unless you are sure the robot is in a '
                        'collision-free environment... validate the torque readings and '
                        'gravity-free torques before setting Safety Factor to 1, or else '
                        'the robot could start moving very fast without being stopped." '
                        'This script previously defaulted to 1.0 based on an incorrect '
                        'assumption (that kinova-ros\'s constructor default meant "the '
                        'standard/safe value") -- it does not; kinova-ros just always sets '
                        'it, independent of whether 1.0 is a good idea for a given session. '
                        '0.3 leaves a real safety net while still tolerating normal hand-'
                        'guiding velocity; raise only after torque sensors are validated '
                        '(e.g. via Development Center\'s "torque zero").')
    p.add_argument('--ramp_seconds', type=float, default=0.5,
                   help='Ramp the damping command in linearly over this many seconds '
                        'after engaging torque mode, to soften the mode-switch transition.')
    p.add_argument('--payload_mass', type=float, default=0.0,
                   help='Attached tool mass (kg) for SetGravityPayload — improves gravity '
                        'compensation accuracy. Leave at 0 to skip (expect some sag/resistance '
                        'in the tool direction from the unmodeled mass otherwise).')
    p.add_argument('--payload_com', type=float, nargs=3, default=[0.0, 0.0, 0.0],
                   help='Tool center-of-mass offset (m) [x y z] in the EEF frame, for '
                        '--payload_mass. Only used if --payload_mass > 0.')
    return p.parse_args()


def main():
    args = parse_args()
    out_root = args.output_dir or os.path.join('data', 'kinesthetic', args.task)
    os.makedirs(out_root, exist_ok=True)
    episode_idx = len([d for d in os.listdir(out_root)
                       if os.path.isdir(os.path.join(out_root, d))])

    api = load_api()
    connect(api)

    # apply_saved_gravity_params sets BOTH the fitted OptimalZ params AND
    # SetGravityType(OPTIMAL) — previously this only did the latter, meaning
    # OPTIMAL mode ran against whatever Z params happened to already be
    # resident in firmware (usually stale/default), not the fitted model.
    # This is likely relevant to this script's own torque-mode-refusal
    # history: the SDK's documented reason for refusing torque mode is
    # "measured torques and computed gravity torques are too different."
    grav_ok = apply_saved_gravity_params(api)
    status = 'OK' if grav_ok else 'not applied — falling back to plain SetGravityType(OPTIMAL)'
    print(f'Firmware gravity params (data/gravity_params.npy) reapplied: {status}')
    if not grav_ok:
        r = api.SetGravityType(GRAVITY_OPTIMAL)
        print(f'SetGravityType(OPTIMAL) -> {r} {"OK" if ok(r) else "FAILED"}')

    q0 = get_joint_angles_deg(api)
    print(f'\nCurrent joint angles (deg): {np.round(q0, 1)}')
    print('Make sure this is a safe pose (away from joint limits / near-singular '
          'configurations) before continuing.\n')

    r = api.SetTorqueSafetyFactor(args.safety_factor)
    print(f'SetTorqueSafetyFactor({args.safety_factor}) -> {r} {"OK" if ok(r) else "FAILED"}')

    r = api.SetTorqueControlType(DIRECTTORQUE)
    print(f'SetTorqueControlType(DIRECTTORQUE) -> {r} {"OK" if ok(r) else "FAILED"}')

    if args.payload_mass > 0:
        payload = (ctypes.c_float * 4)(args.payload_mass, *args.payload_com)
        r = api.SetGravityPayload(payload)
        print(f'SetGravityPayload(mass={args.payload_mass}, com={args.payload_com}) '
              f'-> {r} {"OK" if ok(r) else "FAILED"}')
    else:
        print('No --payload_mass set — expect some sag/resistance toward the attached '
              'tool (unmodeled mass in the gravity model; we measured ~10.5 N*m of this '
              'on J6 earlier via calibrate_contact_baseline.py).')

    print(f'\nEngaging TORQUE control in 3 seconds — HOLD THE TOOL/GRIPPER NOW.')
    for s in (3, 2, 1):
        print(f'  {s}...')
        time.sleep(1.0)

    r = api.SwitchTrajectoryTorque(TORQUE_MODE)
    if not ok(r):
        print(f'\nSwitchTrajectoryTorque(TORQUE) FAILED — return code {r} (NO_ERROR is {NO_ERROR_KINOVA}).')
        print('The arm is still in POSITION control (this is exactly why it would feel rigid '
              'and not move by hand). Common causes: the arm has never had its torque '
              'sensors calibrated via Kinova\'s Development Center tool, or a prior torque-mode '
              'session did not clean up properly and the firmware is in a bad state. '
              'Cycling power on the arm and retrying is a reasonable first thing to check.')
        api.CloseAPI()
        return

    mode_readback = ctypes.c_int(-1)
    api.GetTrajectoryTorqueMode(ctypes.byref(mode_readback))
    print(f'SwitchTrajectoryTorque(TORQUE) -> {r} OK. GetTrajectoryTorqueMode() reads back '
          f'{mode_readback.value} (expect {TORQUE_MODE}=TORQUE).')
    if mode_readback.value != TORQUE_MODE:
        print(f'  WARNING: readback does not match — the arm may not actually be in torque '
              f'control despite the OK return code.')
    print(f'Arm should be compliant now (ramping in over {args.ramp_seconds}s) '
          f'— try moving it by hand.\n')

    state = SharedState()
    thread = threading.Thread(target=torque_loop, args=(api, state, args), daemon=True)
    thread.start()

    print('Commands: r = toggle recording | s = save episode | q = quit\n')
    try:
        while True:
            with state.lock:
                if state.should_quit:
                    if state.error:
                        print(f'\n[ERROR] torque loop stopped: {state.error}')
                    break
            cmd = input('> ').strip().lower()
            if cmd == 'r':
                with state.lock:
                    state.recording = not state.recording
                    status = 'STARTED' if state.recording else 'PAUSED'
                    n = len(state.buffer)
                print(f'Recording {status}. ({n} samples buffered)')
            elif cmd == 's':
                if save_episode(state, out_root, episode_idx, args):
                    episode_idx += 1
            elif cmd == 'q':
                break
            else:
                print('commands: r = toggle recording | s = save episode | q = quit')
    finally:
        with state.lock:
            state.should_quit = True
        thread.join(timeout=2.0)
        try:
            api.SwitchTrajectoryTorque(POSITION_MODE)
        except Exception:
            pass
        api.CloseAPI()
        print('API closed. Back in POSITION control mode.')


if __name__ == '__main__':
    main()
