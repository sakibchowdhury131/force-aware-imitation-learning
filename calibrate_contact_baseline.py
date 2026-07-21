#!/usr/bin/env python3
"""
Capture and save the gravity-free torque baseline used by ContactDetector /
diag_contact.py — a persistent alternative to recalibrating fresh every time
diag_contact.py starts.

Recalibrate whenever the attached tool changes (the bias includes whatever
mass the gripper is currently holding) — the residual gravity-compensation
error is tool- and pose-specific (see contact_detector.py docstring).

Read-only: this script never sends a motion command, only polls the torque
sensors. Keep the arm STATIONARY and NOT touching anything during capture.

Usage:
    python calibrate_contact_baseline.py
    python calibrate_contact_baseline.py --duration 3.0 --output data/contact_baseline.npy
"""
import os, sys, ctypes, time, argparse
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
    for fn in ('InitAPI', 'CloseAPI', 'RefresDevicesList', 'GetDevices',
               'SetActiveDevice', 'GetAngularForceGravityFree', 'GetAngularPosition'):
        getattr(api, fn).restype = ctypes.c_int
    return api


def get_gravity_free_torque(api) -> np.ndarray:
    g = AngularPosition()
    api.GetAngularForceGravityFree(ctypes.byref(g))
    a = g.Actuators
    return np.array([a.Actuator1, a.Actuator2, a.Actuator3,
                     a.Actuator4, a.Actuator5, a.Actuator6], dtype=np.float64)


def get_joint_angles_deg(api) -> np.ndarray:
    pos = AngularPosition()
    api.GetAngularPosition(ctypes.byref(pos))
    a = pos.Actuators
    return np.array([a.Actuator1, a.Actuator2, a.Actuator3,
                     a.Actuator4, a.Actuator5, a.Actuator6], dtype=np.float64)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--duration', type=float, default=2.0,
                   help='Capture duration in seconds (default 2.0)')
    p.add_argument('--hz', type=float, default=25.0, help='Poll rate (default 25 Hz)')
    p.add_argument('--output', default='data/contact_baseline.npy')
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

    try:
        q_deg = get_joint_angles_deg(api)
        print(f'Pose at calibration (deg): {np.round(q_deg, 1)}')
        print(f'Capturing {args.duration}s of gravity-free torque — '
              f'arm must be STATIONARY and NOT touching anything (read-only, no motion sent)...')
        samples = []
        t0 = time.time()
        while time.time() - t0 < args.duration:
            samples.append(get_gravity_free_torque(api))
            time.sleep(1.0 / args.hz)
        samples = np.stack(samples)
        bias = samples.mean(axis=0)
        std  = samples.std(axis=0)

        print(f'\nBias  (N·m): {np.round(bias, 4)}')
        print(f'Noise (std, N·m): {np.round(std, 4)}')

        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        np.save(out_path, bias)
        print(f'\nSaved baseline -> {out_path}')
        print('Load it in diag_contact.py with --baseline_file (or its default path).')
    finally:
        api.CloseAPI()
        print('API closed.')


if __name__ == '__main__':
    main()
