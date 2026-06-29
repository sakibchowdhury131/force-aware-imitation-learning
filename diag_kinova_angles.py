"""
Quick Kinova angle diagnostic — polls GetCartesianPosition at 5Hz and prints
the raw ThetaX/Y/Z values alongside the rotation matrix computed under several
conventions, so we can see which one tracks physical wrist rotation correctly.
"""
import os, sys, ctypes, time
import numpy as np
from scipy.spatial.transform import Rotation

_LIB_DIR = os.path.join(os.path.expanduser('~/working_dir/kinovaDrivers'), 'sdk', 'lib')
if _LIB_DIR not in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
    os.environ["LD_LIBRARY_PATH"] = _LIB_DIR + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.execv(sys.executable, [sys.executable] + sys.argv)

LIB_PATH      = os.path.join(_LIB_DIR, "USBCommandLayerUbuntu.so")
COMM_LIB_PATH = os.path.join(_LIB_DIR, "USBCommLayerUbuntu.so")
NO_ERROR_KINOVA = 1
SERIAL_LENGTH   = 20
MAX_KINOVA_DEVICE = 20

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
                   ("SetActiveDevice", ctypes.c_int), ("GetCartesianPosition", ctypes.c_int)]:
        getattr(api, fn).restype = rt
    return api

def rotvec_angle(R):
    return np.degrees(Rotation.from_matrix(R).magnitude())

CONVENTIONS = ['xyz', 'XYZ', 'ZYX', 'zyx']

def angles_to_R(conv, tx, ty, tz):
    if conv == 'ZYX':
        # test with reversed arg order too
        return Rotation.from_euler('ZYX', [tz, ty, tx]).as_matrix()
    return Rotation.from_euler(conv, [tx, ty, tz]).as_matrix()

api = load_api()
if api.InitAPI() != NO_ERROR_KINOVA:
    raise RuntimeError("Kinova InitAPI failed")
api.RefresDevicesList()
devices = (KinovaDevice * MAX_KINOVA_DEVICE)()
err = ctypes.c_int(NO_ERROR_KINOVA)
n = api.GetDevices(devices, ctypes.byref(err))
if n == 0:
    raise RuntimeError("No Kinova device found")
api.SetActiveDevice(devices[0])
print(f"Connected: {devices[0].Model.decode()} ({devices[0].SerialNumber.decode()})\n")

print("Polling at 5 Hz for 30 seconds.")
print("Please HOLD STILL for 3s, then rotate the WRIST continuously.\n")
print(f"{'t':>5}  {'TX':>8} {'TY':>8} {'TZ':>8}  "
      + "  ".join(f"{'|'+c+'|':>8}" for c in CONVENTIONS))
print("-" * 90)

pos = CartesianPosition()
t0  = time.time()
R_prev = {c: None for c in CONVENTIONS}

try:
    while time.time() - t0 < 30:
        api.GetCartesianPosition(ctypes.byref(pos))
        c  = pos.Coordinates
        tx, ty, tz = c.ThetaX, c.ThetaY, c.ThetaZ
        t  = time.time() - t0

        cols = []
        for conv in CONVENTIONS:
            R = angles_to_R(conv, tx, ty, tz)
            ang = rotvec_angle(R)
            if R_prev[conv] is not None:
                delta = rotvec_angle(R_prev[conv].T @ R)
                cols.append(f"{ang:6.1f}({delta:+5.1f}°)")
            else:
                cols.append(f"{ang:6.1f}(  ---)")
            R_prev[conv] = R

        print(f"{t:5.1f}  {tx:+8.3f} {ty:+8.3f} {tz:+8.3f}  " + "  ".join(cols))
        time.sleep(0.2)
finally:
    api.CloseAPI()
