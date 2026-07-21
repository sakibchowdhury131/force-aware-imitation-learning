#!/usr/bin/env python3
"""
Live contact detector for the Kinova Jaco2 — flags when the arm touches
something hard, using ContactDetector (contact_detector.py) on top of the
gravity-free joint torque -> EEF force mapping (Jacobian transpose).

At startup it collects a short baseline (arm must be stationary, NOT touching
anything) to cancel out residual gravity-compensation bias, then live-plots
the contact-force magnitude with the enter/exit thresholds, shading the
timeline red while contact is detected, and printing ENTER/EXIT events.

Usage:
    python diag_contact.py
    python diag_contact.py --threshold 8.0       # EEF force threshold (N)
    python diag_contact.py --mode joint          # trigger on raw joint torque instead
    python diag_contact.py --calib_seconds 3.0   # longer baseline capture
    python diag_contact.py --compensate_dynamics # also subtract M(q)qddot + C(q,qdot)qdot
                                                  # so moving (not just touching) doesn't
                                                  # look like contact — see contact_detector.py

Press Q or close window to quit.
"""
import os, sys, ctypes, time, argparse, collections
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ── Kinova SDK bootstrap ───────────────────────────────────────────────────────
_LIB_DIR = os.path.join(os.path.expanduser('~/working_dir/kinovaDrivers'), 'sdk', 'lib')
if _LIB_DIR not in os.environ.get('LD_LIBRARY_PATH', '').split(':'):
    os.environ['LD_LIBRARY_PATH'] = _LIB_DIR + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    os.execv(sys.executable, [sys.executable] + sys.argv)

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)
from contact_detector import ContactDetector, VelocityDifferentiator
from calibrate_firmware_gravity import apply_saved_gravity_params

LIB_PATH      = os.path.join(_LIB_DIR, 'USBCommandLayerUbuntu.so')
COMM_LIB_PATH = os.path.join(_LIB_DIR, 'USBCommLayerUbuntu.so')
NO_ERROR_KINOVA   = 1
SERIAL_LENGTH     = 20
MAX_KINOVA_DEVICE = 20


class KinovaDevice(ctypes.Structure):
    _fields_ = [('SerialNumber', ctypes.c_char * SERIAL_LENGTH),
                ('Model',        ctypes.c_char * SERIAL_LENGTH),
                ('VersionMajor', ctypes.c_int), ('VersionMinor',   ctypes.c_int),
                ('VersionRelease', ctypes.c_int), ('DeviceType',   ctypes.c_int),
                ('DeviceID',     ctypes.c_int)]

# Real SDK struct has 7 actuator slots regardless of 6-DOF arm — matches
# 07_deploy.py / replay_episode.py / diag_joint_torques.py.
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
    for fn in ('InitAPI', 'CloseAPI', 'RefresDevicesList', 'GetDevices',
               'SetActiveDevice', 'GetAngularForce', 'GetAngularForceGravityFree',
               'GetAngularPosition', 'GetAngularVelocity'):
        getattr(api, fn).restype = ctypes.c_int
    return api


def get_joint_angles_deg(api) -> np.ndarray:
    pos = AngularPosition()
    api.GetAngularPosition(ctypes.byref(pos))
    a = pos.Actuators
    return np.array([a.Actuator1, a.Actuator2, a.Actuator3,
                     a.Actuator4, a.Actuator5, a.Actuator6], dtype=np.float64)


def get_joint_velocity_deg(api) -> np.ndarray:
    """GetAngularVelocity reuses the AngularPosition struct layout to report
    joint angular velocity (deg/s) instead of position."""
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--hz', type=float, default=25.0, help='Poll rate in Hz (default 25)')
    p.add_argument('--window', type=float, default=15.0, help='Seconds of history shown')
    p.add_argument('--mode', choices=['force', 'joint'], default='force',
                   help="'force' = Jacobian-mapped EEF contact force (N); "
                        "'joint' = max raw joint torque (N·m) — use if the arm "
                        "spends time near a singularity where the Jacobian is ill-conditioned.")
    p.add_argument('--threshold', type=float, default=5.0,
                   help='Contact-enter threshold (N for force mode, N·m for joint mode). '
                        'Tune this to your setup — start here and watch the plot.')
    p.add_argument('--exit_ratio', type=float, default=0.6,
                   help='exit_threshold = threshold * exit_ratio (hysteresis, avoids chattering)')
    p.add_argument('--debounce', type=int, default=3,
                   help='Consecutive samples required before flipping contact state')
    p.add_argument('--calib_seconds', type=float, default=1.5,
                   help='Baseline capture duration at startup — keep the arm still and '
                        'NOT touching anything during this window. Ignored if '
                        '--baseline_file exists (unless --recalibrate is also passed).')
    p.add_argument('--baseline_file', default='data/contact_baseline.npy',
                   help='Saved baseline from calibrate_contact_baseline.py. Loaded '
                        'automatically if it exists, skipping the live capture below.')
    p.add_argument('--recalibrate', action='store_true',
                   help='Ignore --baseline_file and do a fresh live capture instead.')
    p.add_argument('--baseline_mode', choices=['fixed', 'moving'], default='fixed',
                   help="'fixed' (default) = subtract the single calibrated bias forever — "
                        "simple, but pose-dependent residual creeps back in away from the "
                        "calibration pose. 'moving' = continuously track a slow exponential "
                        "moving average instead, so pose-related drift is absorbed "
                        "automatically and only sudden spikes (real contact) register.")
    p.add_argument('--moving_time_constant', type=float, default=2.0,
                   help='Seconds — how slowly the moving-average baseline adapts. Only used '
                        'with --baseline_mode moving. Must be slower than a real contact '
                        'transient but fast enough to track genuine pose-change drift.')
    p.add_argument('--compensate_dynamics', action='store_true',
                   help='Also subtract M(q)*qddot + C(q,qdot)*qdot (rigid-body motion '
                        'torque, via RNEA) before thresholding, so moving the arm itself '
                        'does not look like contact. qdot comes from GetAngularVelocity; '
                        'qddot is estimated by differentiating it (see VelocityDifferentiator '
                        'in contact_detector.py) since the SDK does not expose it directly.')
    p.add_argument('--qddot_smoothing', type=float, default=0.5,
                   help='Exponential-smoothing factor (0-1) for the estimated qddot. '
                        'Higher = smoother but laggier. Only used with --compensate_dynamics.')
    return p.parse_args()


def main():
    args = parse_args()

    api = load_api()
    if api.InitAPI() != NO_ERROR_KINOVA:
        raise RuntimeError('Kinova InitAPI failed')
    api.RefresDevicesList()
    devices = (KinovaDevice * MAX_KINOVA_DEVICE)()
    err = ctypes.c_int(NO_ERROR_KINOVA)
    n = api.GetDevices(devices, ctypes.byref(err))
    if n == 0:
        api.CloseAPI()
        raise RuntimeError('No Kinova device found')
    api.SetActiveDevice(devices[0])
    print(f'Connected: {devices[0].Model.decode()} ({devices[0].SerialNumber.decode()})')
    grav_ok = apply_saved_gravity_params(api)
    print(f'Firmware gravity params (data/gravity_params.npy) reapplied: '
          f'{"OK" if grav_ok else "not applied — see message above"}')

    detector = ContactDetector(enter_threshold=args.threshold, exit_ratio=args.exit_ratio,
                               debounce=args.debounce, mode=args.mode,
                               compensate_dynamics=args.compensate_dynamics,
                               baseline_mode=args.baseline_mode,
                               moving_time_constant=args.moving_time_constant)
    qddot_est = VelocityDifferentiator(smoothing=args.qddot_smoothing)

    baseline_path = os.path.join(PIPELINE_DIR, args.baseline_file)
    if os.path.exists(baseline_path) and not args.recalibrate:
        detector.set_bias(np.load(baseline_path))
        print(f'\nLoaded saved baseline from {baseline_path} '
              f'(pass --recalibrate to capture a fresh one instead)')
    else:
        print(f'\nCalibrating baseline for {args.calib_seconds}s — '
              f'DO NOT TOUCH THE ARM, keep it still...')
        calib_samples = []
        t_calib = time.time()
        while time.time() - t_calib < args.calib_seconds:
            calib_samples.append(get_gravity_free_torque(api))
            time.sleep(1.0 / args.hz)
        detector.calibrate(np.stack(calib_samples))
    # Warm up the acceleration estimator too (needs one velocity sample before
    # it can produce a finite-difference estimate on the next update()).
    qddot_est.update(get_joint_velocity_deg(api), time.time())
    print(f'Baseline bias (N·m): {np.round(detector.bias, 3)}')
    print(f'Mode={args.mode}  baseline_mode={args.baseline_mode}'
          + (f' (tau={args.moving_time_constant}s)' if args.baseline_mode == 'moving' else '')
          + f'  enter_threshold={args.threshold}  '
          f'exit_threshold={detector.exit_threshold:.2f}  debounce={args.debounce}  '
          f'compensate_dynamics={args.compensate_dynamics}')
    print('Close window or press Q to quit.\n')

    maxlen = int(args.hz * args.window)
    t_buf   = collections.deque(maxlen=maxlen)
    mag_buf = collections.deque(maxlen=maxlen)
    contact_buf = collections.deque(maxlen=maxlen)
    t0 = time.time()

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#111111')
    unit = 'N' if args.mode == 'force' else 'N·m'
    ax.set_title(f'Contact Detector — {args.mode} mode', fontsize=13,
                fontweight='bold', color='#eeeeee')
    ax.set_xlabel('Time (s)'); ax.set_ylabel(f'Contact magnitude ({unit})')
    ax.tick_params(colors='#cccccc')
    ax.xaxis.label.set_color('#cccccc'); ax.yaxis.label.set_color('#cccccc')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')

    ax.axhline(args.threshold, color='#ff4d4d', linewidth=1.0, linestyle='--',
              label=f'enter ({args.threshold:.1f})')
    ax.axhline(detector.exit_threshold, color='#ffa64d', linewidth=1.0, linestyle=':',
              label=f'exit ({detector.exit_threshold:.1f})')
    line_mag, = ax.plot([], [], color='#42d4f4', linewidth=1.5, label='magnitude')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.4,
              labelcolor='#eeeeee', facecolor='#222222')

    status_txt = fig.text(0.5, 0.93, '', ha='center', fontsize=16, fontweight='bold')
    contact_spans = []   # list of axvspan artists, redrawn each frame

    dt = 1.0 / args.hz
    last_poll = [time.time()]

    def update(_frame):
        now = time.time()
        if now - last_poll[0] >= dt:
            last_poll[0] = now
            tau_gf = get_gravity_free_torque(api)
            q_deg  = get_joint_angles_deg(api)
            if args.compensate_dynamics:
                qdot_deg  = get_joint_velocity_deg(api)
                qddot_deg = qddot_est.update(qdot_deg, now)
            else:
                qdot_deg = qddot_deg = None
            in_contact, mag, _F, changed = detector.update(
                tau_gf, q_deg, qdot_deg, qddot_deg, t=now)
            t_buf.append(now - t0)
            mag_buf.append(mag)
            contact_buf.append(in_contact)
            if changed:
                state = 'CONTACT' if in_contact else 'clear'
                print(f'  [{t_buf[-1]:7.2f}s] {state:8s}  magnitude={mag:.2f} {unit}')

        if len(t_buf) < 2:
            return []

        t_arr   = np.array(t_buf)
        mag_arr = np.array(mag_buf)
        contact_arr = np.array(contact_buf)

        line_mag.set_data(t_arr, mag_arr)
        ax.relim(); ax.autoscale_view(scalex=True, scaley=True)
        x_max = t_arr[-1]
        x_min = max(0.0, x_max - args.window)
        ax.set_xlim(x_min, x_max + 0.1)

        # Redraw contact shading (cheap enough at this sample rate/window)
        for span in contact_spans:
            span.remove()
        contact_spans.clear()
        in_span = False
        span_start = None
        for i, c in enumerate(contact_arr):
            if c and not in_span:
                span_start = t_arr[i]; in_span = True
            elif not c and in_span:
                contact_spans.append(ax.axvspan(span_start, t_arr[i], color='red', alpha=0.15))
                in_span = False
        if in_span:
            contact_spans.append(ax.axvspan(span_start, t_arr[-1], color='red', alpha=0.15))

        if detector.in_contact:
            status_txt.set_text('CONTACT')
            status_txt.set_color('#ff4d4d')
        else:
            status_txt.set_text('clear')
            status_txt.set_color('#42d4f4')

        return [line_mag, status_txt] + contact_spans

    ani = animation.FuncAnimation(fig, update, interval=max(20, int(dt * 1000)),
                                  blit=False, cache_frame_data=False)

    def on_key(event):
        if event.key in ('q', 'Q'):
            plt.close('all')
    fig.canvas.mpl_connect('key_press_event', on_key)

    try:
        plt.tight_layout(rect=[0, 0, 1, 0.90])
        plt.show()
    finally:
        api.CloseAPI()
        print('API closed.')


if __name__ == '__main__':
    main()
