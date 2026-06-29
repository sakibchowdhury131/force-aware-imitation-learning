#!/usr/bin/env python3
"""
Real-time plot of Kinova Jaco2 joint torque sensor readings.

Plots both raw torque (GetAngularForce) and gravity-compensated torque
(GetAngularForceGravityFree) so you can see external loads clearly.

Usage:
    python diag_joint_torques.py
    python diag_joint_torques.py --gravity_free   # show only gravity-free channel
    python diag_joint_torques.py --hz 50          # poll rate (default 25 Hz)
    python diag_joint_torques.py --window 10      # seconds of history (default 8)

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

class AngularInfo(ctypes.Structure):
    _fields_ = [(f'Actuator{i}', ctypes.c_float) for i in range(1, 7)]

class FingersPosition(ctypes.Structure):
    _fields_ = [('Finger1', ctypes.c_float), ('Finger2', ctypes.c_float),
                ('Finger3', ctypes.c_float)]

class AngularPosition(ctypes.Structure):
    _fields_ = [('Actuators', AngularInfo), ('Fingers', FingersPosition)]


def load_api():
    ctypes.CDLL(COMM_LIB_PATH, mode=ctypes.RTLD_GLOBAL)
    api = ctypes.CDLL(LIB_PATH)
    for fn in ('InitAPI', 'CloseAPI', 'RefresDevicesList', 'GetDevices',
               'SetActiveDevice', 'GetAngularForce', 'GetAngularForceGravityFree'):
        getattr(api, fn).restype = ctypes.c_int
    return api


def read_torques(api, buf_raw, buf_gf):
    r = AngularPosition()
    g = AngularPosition()
    api.GetAngularForce(ctypes.byref(r))
    api.GetAngularForceGravityFree(ctypes.byref(g))
    raw = np.array([getattr(r.Actuators, f'Actuator{i}') for i in range(1, 7)], np.float32)
    gf  = np.array([getattr(g.Actuators, f'Actuator{i}') for i in range(1, 7)], np.float32)
    buf_raw.append(raw)
    buf_gf.append(gf)


# ── plotting ───────────────────────────────────────────────────────────────────
JOINT_COLORS = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4']
JOINT_LABELS = [f'J{i}' for i in range(1, 7)]


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
    print('Close window or press Q to quit.\n')

    maxlen = int(args.hz * args.window)
    buf_raw = collections.deque(maxlen=maxlen)
    buf_gf  = collections.deque(maxlen=maxlen)
    t_buf   = collections.deque(maxlen=maxlen)
    t0      = time.time()

    # Pre-fill with one sample so arrays are never empty
    read_torques(api, buf_raw, buf_gf)
    t_buf.append(0.0)

    n_rows = 1 if args.gravity_free else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 4 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]
    fig.suptitle('Kinova Jaco2 — Joint Torques (N·m)', fontsize=13, fontweight='bold')

    titles = (['Gravity-free torque (external load)'] if args.gravity_free
              else ['Raw torque (GetAngularForce)',
                    'Gravity-free torque (GetAngularForceGravityFree)'])

    lines = []
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=10)
        ax.set_ylabel('Torque (N·m)')
        ax.set_xlabel('Time (s)')
        ax.axhline(0, color='white', linewidth=0.5, alpha=0.4)
        ax.set_facecolor('#111111')
        fig.patch.set_facecolor('#1a1a1a')
        ax.tick_params(colors='#cccccc')
        ax.yaxis.label.set_color('#cccccc')
        ax.xaxis.label.set_color('#cccccc')
        ax.title.set_color('#eeeeee')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')
        row_lines = []
        for j, (col, lbl) in enumerate(zip(JOINT_COLORS, JOINT_LABELS)):
            ln, = ax.plot([], [], color=col, linewidth=1.4, label=lbl)
            row_lines.append(ln)
        ax.legend(loc='upper left', fontsize=8, framealpha=0.4,
                  labelcolor='#eeeeee', facecolor='#222222')
        lines.append(row_lines)

    # Text annotations: current value readout at right edge
    readouts = []
    for ax, row_lines in zip(axes, lines):
        row_ro = []
        for j, (col, ln) in enumerate(zip(JOINT_COLORS, row_lines)):
            txt = ax.text(1.001, 0.85 - j * 0.14, '', transform=ax.transAxes,
                          color=col, fontsize=7.5, va='center', ha='left',
                          fontfamily='monospace')
            row_ro.append(txt)
        readouts.append(row_ro)

    dt = 1.0 / args.hz
    last_poll = [time.time()]

    def update(_frame):
        now = time.time()
        if now - last_poll[0] >= dt:
            last_poll[0] = now
            read_torques(api, buf_raw, buf_gf)
            t_buf.append(now - t0)

        if len(t_buf) < 2:
            return []

        t_arr = np.array(t_buf)
        raw_arr = np.array(buf_raw)   # (N, 6)
        gf_arr  = np.array(buf_gf)    # (N, 6)

        bufs = ([gf_arr] if args.gravity_free else [raw_arr, gf_arr])

        updated = []
        for ax, row_lines, row_bufs, row_ro in zip(axes, lines, bufs, readouts):
            for j, (ln, txt) in enumerate(zip(row_lines, row_ro)):
                ln.set_data(t_arr, row_bufs[:, j])
                txt.set_text(f'J{j+1}: {row_bufs[-1, j]:+7.2f} N·m')
                updated.append(ln)
                updated.append(txt)
            ax.relim()
            ax.autoscale_view(scalex=True, scaley=True)
            # Keep x axis sliding
            x_max = t_arr[-1]
            x_min = max(0.0, x_max - args.window)
            ax.set_xlim(x_min, x_max + 0.1)

        return updated

    ani = animation.FuncAnimation(fig, update, interval=max(20, int(dt * 1000)),
                                  blit=False, cache_frame_data=False)

    def on_key(event):
        if event.key in ('q', 'Q'):
            plt.close('all')

    fig.canvas.mpl_connect('key_press_event', on_key)

    try:
        plt.tight_layout(rect=[0, 0, 0.97, 0.95])
        plt.show()
    finally:
        api.CloseAPI()
        print('API closed.')


if __name__ == '__main__':
    main()
