#!/usr/bin/env python3
"""
Real-time plot of Kinova Jaco2 joint torque sensor readings, plus the
corresponding end-effector force/torque wrench computed via the Jacobian
transpose (F = pinv(J^T) @ tau), expressed in the robot base frame.

Plots both raw torque (GetAngularForce) and gravity-compensated torque
(GetAngularForceGravityFree) so you can see external loads clearly, one
subplot per joint (left block) and per EEF wrench component (right block).

Usage:
    python diag_joint_torques.py
    python diag_joint_torques.py --gravity_free   # show only gravity-free channel
    python diag_joint_torques.py --hz 50          # poll rate (default 25 Hz)
    python diag_joint_torques.py --window 10      # seconds of history (default 8)

Press Q or close window to quit.
"""
import os, sys, ctypes, time, argparse, collections
import numpy as np
from scipy.spatial.transform import Rotation
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
from calibrate_firmware_gravity import apply_saved_gravity_params

LIB_PATH      = os.path.join(_LIB_DIR, 'USBCommandLayerUbuntu.so')
COMM_LIB_PATH = os.path.join(_LIB_DIR, 'USBCommLayerUbuntu.so')
NO_ERROR_KINOVA = 1
SERIAL_LENGTH   = 20
MAX_KINOVA_DEVICE = 20

# ── ctypes structs ─────────────────────────────────────────────────────────────
class KinovaDevice(ctypes.Structure):
    _fields_ = [('SerialNumber', ctypes.c_char * SERIAL_LENGTH),
                ('Model',        ctypes.c_char * SERIAL_LENGTH),
                ('VersionMajor', ctypes.c_int), ('VersionMinor',   ctypes.c_int),
                ('VersionRelease', ctypes.c_int), ('DeviceType',   ctypes.c_int),
                ('DeviceID',     ctypes.c_int)]

# NOTE: the real SDK struct has 7 actuator slots (up to 7-DOF arms) regardless
# of this being a 6-DOF Jaco2 — matches 07_deploy.py / replay_episode.py.
# (Previously this was declared with only 6 fields here, which under-sizes
# the buffer GetAngularForce/GetAngularPosition actually write into.)
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
               'GetAngularPosition'):
        getattr(api, fn).restype = ctypes.c_int
    return api


def get_joint_angles_deg(api) -> np.ndarray:
    pos = AngularPosition()
    api.GetAngularPosition(ctypes.byref(pos))
    a = pos.Actuators
    return np.array([a.Actuator1, a.Actuator2, a.Actuator3,
                     a.Actuator4, a.Actuator5, a.Actuator6], dtype=np.float64)


def read_torques(api, buf_raw, buf_gf):
    r = AngularPosition()
    g = AngularPosition()
    api.GetAngularForce(ctypes.byref(r))
    api.GetAngularForceGravityFree(ctypes.byref(g))
    raw = np.array([getattr(r.Actuators, f'Actuator{i}') for i in range(1, 7)], np.float32)
    gf  = np.array([getattr(g.Actuators, f'Actuator{i}') for i in range(1, 7)], np.float32)
    buf_raw.append(raw)
    buf_gf.append(gf)


# ════════════════════════════════════════════════════════════════════════════
# Forward kinematics + geometric Jacobian
# Same DH chain as 07_deploy.py / deploy_viz.py — kept self-contained here to
# match the rest of this pipeline's per-script convention.
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


def fk_frames(q_deg: np.ndarray):
    """Returns (pre_frames, T_eef): pre_frames[i] is the joint-i frame BEFORE
    its own rotation is applied (origin + Z axis = joint i's rotation axis in
    base frame) — exactly what's needed to build the geometric Jacobian."""
    q = np.deg2rad(q_deg)
    T = np.eye(4, dtype=np.float64)
    pre_frames = []
    for (xyz, rpy), qi in zip(_FK_JOINT_PARAMS, q):
        T_pre = T @ _make_fk_T(xyz, rpy)
        pre_frames.append(T_pre)
        Tj = np.eye(4, dtype=np.float64)
        Tj[:3, :3] = Rotation.from_euler('z', float(qi)).as_matrix()
        T = T_pre @ Tj
    T_eef = T @ _make_fk_T(*_FK_EEF_PARAMS)
    return pre_frames, T_eef


def compute_jacobian(q_deg: np.ndarray) -> np.ndarray:
    """6x6 geometric Jacobian (rows 0-2 linear, rows 3-5 angular; base frame).
    All joints are revolute, so column i = [z_i x (p_eef - p_i); z_i]."""
    pre_frames, T_eef = fk_frames(q_deg)
    p_eef = T_eef[:3, 3]
    J = np.zeros((6, 6))
    for i, T_pre in enumerate(pre_frames):
        z_i = T_pre[:3, 2]
        p_i = T_pre[:3, 3]
        J[:3, i] = np.cross(z_i, p_eef - p_i)
        J[3:, i] = z_i
    return J


def torque_to_wrench(q_deg: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Static-equilibrium relation tau = J^T @ F  =>  F = pinv(J^T) @ tau.
    Returns [Fx,Fy,Fz,Mx,My,Mz] at the EEF origin, in the robot base frame.
    Uses pinv (not inv) so this degrades gracefully near singularities."""
    J = compute_jacobian(q_deg)
    return np.linalg.pinv(J.T) @ tau


# ── plotting ───────────────────────────────────────────────────────────────────
JOINT_LABELS  = [f'J{i}' for i in range(1, 7)]
WRENCH_LABELS = ['Fx (N)', 'Fy (N)', 'Fz (N)', 'Mx (N·m)', 'My (N·m)', 'Mz (N·m)']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--hz',          type=float, default=25.0,
                   help='Poll rate in Hz (default 25)')
    p.add_argument('--window',      type=float, default=8.0,
                   help='Seconds of history to display (default 8)')
    p.add_argument('--gravity_free', action='store_true',
                   help='Show only gravity-compensated channel (cleaner for external loads)')
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
    print('Close window or press Q to quit.\n')

    maxlen = int(args.hz * args.window)
    buf_raw    = collections.deque(maxlen=maxlen)
    buf_gf     = collections.deque(maxlen=maxlen)
    buf_wr_raw = collections.deque(maxlen=maxlen)   # EEF wrench from raw torque
    buf_wr_gf  = collections.deque(maxlen=maxlen)   # EEF wrench from gravity-free torque
    t_buf      = collections.deque(maxlen=maxlen)
    t0         = time.time()

    def poll():
        read_torques(api, buf_raw, buf_gf)
        q_deg = get_joint_angles_deg(api)
        buf_wr_raw.append(torque_to_wrench(q_deg, buf_raw[-1].astype(np.float64)))
        buf_wr_gf.append(torque_to_wrench(q_deg, buf_gf[-1].astype(np.float64)))

    # Pre-fill with one sample so arrays are never empty
    poll()
    t_buf.append(0.0)

    RAW_COLOR = '#f58231'
    GF_COLOR  = '#42d4f4'

    # Left block: one subplot per joint torque (3x2). Right block: one subplot
    # per EEF wrench component (3x2), computed via the Jacobian transpose.
    fig, axes_grid = plt.subplots(3, 4, figsize=(20, 9), sharex=True)
    joint_axes  = axes_grid[:, :2].flatten(order='F')   # J1..J6
    wrench_axes = axes_grid[:, 2:].flatten(order='F')   # Fx,Fy,Fz,Mx,My,Mz
    fig.suptitle('Kinova Jaco2 — Joint Torques & EEF Wrench (Jacobian^T)',
                fontsize=13, fontweight='bold')

    def _style_axis(ax, title, ylabel, is_bottom_row):
        ax.set_title(title, fontsize=10, color='#eeeeee')
        ax.set_ylabel(ylabel)
        if is_bottom_row:
            ax.set_xlabel('Time (s)')
        ax.axhline(0, color='white', linewidth=0.5, alpha=0.4)
        ax.set_facecolor('#111111')
        ax.tick_params(colors='#cccccc')
        ax.yaxis.label.set_color('#cccccc')
        ax.xaxis.label.set_color('#cccccc')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')
    fig.patch.set_facecolor('#1a1a1a')

    def _make_lines(ax):
        d = {}
        if not args.gravity_free:
            ln_raw, = ax.plot([], [], color=RAW_COLOR, linewidth=1.2, label='raw')
            d['raw'] = ln_raw
        ln_gf, = ax.plot([], [], color=GF_COLOR, linewidth=1.4, label='gravity-free')
        d['gf'] = ln_gf
        ax.legend(loc='upper left', fontsize=7, framealpha=0.4,
                  labelcolor='#eeeeee', facecolor='#222222')
        return d

    def _make_readouts(ax):
        ro = {}
        if not args.gravity_free:
            ro['raw'] = ax.text(0.98, 0.90, '', transform=ax.transAxes,
                                color=RAW_COLOR, fontsize=8, va='top', ha='right',
                                fontfamily='monospace')
        ro['gf'] = ax.text(0.98, 0.78, '', transform=ax.transAxes,
                           color=GF_COLOR, fontsize=8, va='top', ha='right',
                           fontfamily='monospace')
        return ro

    joint_lines = []
    joint_ro    = []
    for j, (ax, lbl) in enumerate(zip(joint_axes, JOINT_LABELS)):
        _style_axis(ax, lbl, 'Torque (N·m)', is_bottom_row=(j % 3 == 2))
        joint_lines.append(_make_lines(ax))
        joint_ro.append(_make_readouts(ax))

    wrench_lines = []
    wrench_ro    = []
    for j, (ax, lbl) in enumerate(zip(wrench_axes, WRENCH_LABELS)):
        _style_axis(ax, lbl, lbl.split(' ')[1].strip('()'), is_bottom_row=(j % 3 == 2))
        wrench_lines.append(_make_lines(ax))
        wrench_ro.append(_make_readouts(ax))

    dt = 1.0 / args.hz
    last_poll = [time.time()]

    def update(_frame):
        now = time.time()
        if now - last_poll[0] >= dt:
            last_poll[0] = now
            poll()
            t_buf.append(now - t0)

        if len(t_buf) < 2:
            return []

        t_arr      = np.array(t_buf)
        raw_arr    = np.array(buf_raw)      # (N, 6) joint torque
        gf_arr     = np.array(buf_gf)       # (N, 6) joint torque
        wr_raw_arr = np.array(buf_wr_raw)   # (N, 6) EEF wrench
        wr_gf_arr  = np.array(buf_wr_gf)    # (N, 6) EEF wrench

        updated = []

        def _draw(ax, lines_d, ro_d, raw_col, gf_col, j):
            if 'raw' in lines_d:
                lines_d['raw'].set_data(t_arr, raw_col[:, j])
                ro_d['raw'].set_text(f'raw: {raw_col[-1, j]:+7.2f}')
                updated.append(lines_d['raw']); updated.append(ro_d['raw'])
            lines_d['gf'].set_data(t_arr, gf_col[:, j])
            ro_d['gf'].set_text(f'gf:  {gf_col[-1, j]:+7.2f}')
            updated.append(lines_d['gf']); updated.append(ro_d['gf'])
            ax.relim()
            ax.autoscale_view(scalex=True, scaley=True)
            x_max = t_arr[-1]
            ax.set_xlim(max(0.0, x_max - args.window), x_max + 0.1)

        for j, ax in enumerate(joint_axes):
            _draw(ax, joint_lines[j], joint_ro[j], raw_arr, gf_arr, j)
        for j, ax in enumerate(wrench_axes):
            _draw(ax, wrench_lines[j], wrench_ro[j], wr_raw_arr, wr_gf_arr, j)

        return updated

    ani = animation.FuncAnimation(fig, update, interval=max(20, int(dt * 1000)),
                                  blit=False, cache_frame_data=False)

    def on_key(event):
        if event.key in ('q', 'Q'):
            plt.close('all')

    fig.canvas.mpl_connect('key_press_event', on_key)

    try:
        plt.tight_layout(rect=[0, 0, 0.99, 0.95])
        plt.show()
    finally:
        api.CloseAPI()
        print('API closed.')


if __name__ == '__main__':
    main()
