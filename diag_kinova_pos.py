"""Print raw Kinova XYZ + Theta and project FK into camera frame."""
import os, sys, ctypes
import numpy as np
from scipy.spatial.transform import Rotation

_LIB_DIR = os.path.join(os.path.expanduser('~/working_dir/kinovaDrivers'), 'sdk', 'lib')
if _LIB_DIR not in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
    os.environ["LD_LIBRARY_PATH"] = _LIB_DIR + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.execv(sys.executable, [sys.executable] + sys.argv)

LIB_PATH      = os.path.join(_LIB_DIR, "USBCommandLayerUbuntu.so")
COMM_LIB_PATH = os.path.join(_LIB_DIR, "USBCommLayerUbuntu.so")

class KinovaDevice(ctypes.Structure):
    _fields_ = [("SerialNumber", ctypes.c_char * 20), ("Model", ctypes.c_char * 20),
                ("VersionMajor", ctypes.c_int), ("VersionMinor", ctypes.c_int),
                ("VersionRelease", ctypes.c_int), ("DeviceType", ctypes.c_int),
                ("DeviceID", ctypes.c_int)]
class CartesianInfo(ctypes.Structure):
    _fields_ = [("X", ctypes.c_float), ("Y", ctypes.c_float), ("Z", ctypes.c_float),
                ("ThetaX", ctypes.c_float), ("ThetaY", ctypes.c_float), ("ThetaZ", ctypes.c_float)]
class FingersPosition(ctypes.Structure):
    _fields_ = [("Finger1", ctypes.c_float), ("Finger2", ctypes.c_float), ("Finger3", ctypes.c_float)]
class CartesianPosition(ctypes.Structure):
    _fields_ = [("Coordinates", CartesianInfo), ("Fingers", FingersPosition)]

ctypes.CDLL(COMM_LIB_PATH, mode=ctypes.RTLD_GLOBAL)
api = ctypes.CDLL(LIB_PATH)
for fn in ["InitAPI", "RefresDevicesList", "GetDevices", "SetActiveDevice",
           "GetCartesianPosition", "CloseAPI"]:
    getattr(api, fn).restype = ctypes.c_int

api.InitAPI()
api.RefresDevicesList()
devices = (KinovaDevice * 20)()
err = ctypes.c_int(1)
n = api.GetDevices(devices, ctypes.byref(err))
print(f"Found {n} device(s): {devices[0].Model.decode()}")
api.SetActiveDevice(devices[0])

pos = CartesianPosition()
api.GetCartesianPosition(ctypes.byref(pos))
c = pos.Coordinates
print(f"\nRaw Kinova EEF pose:")
print(f"  Position : X={c.X:.4f} m  Y={c.Y:.4f} m  Z={c.Z:.4f} m")
print(f"  Orientation: TX={c.ThetaX:.4f} rad  TY={c.ThetaY:.4f} rad  TZ={c.ThetaZ:.4f} rad")

# Load calibration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
tf_world2cam = np.load(os.path.join(SCRIPT_DIR, 'data/cam_extrinsics.npy')).astype(np.float64)
T_base_task  = np.load(os.path.join(SCRIPT_DIR, 'data/robot_extrinsics.npy')).astype(np.float64)
K = np.array([[606, 0, 424], [0, 606, 240], [0, 0, 1]])  # approx RealSense 848x480

print(f"\nCalibration:")
print(f"  T_base_task:\n{T_base_task.round(4)}")

print("\nProjected FK origin (EEF) in camera frame for convention 'xyz':")
T_base_eef = np.eye(4)
T_base_eef[:3,:3] = Rotation.from_euler('xyz', [c.ThetaX, c.ThetaY, c.ThetaZ]).as_matrix()
T_base_eef[:3, 3] = [c.X, c.Y, c.Z]

T_cam_eef = tf_world2cam @ np.linalg.inv(T_base_task) @ T_base_eef
p = T_cam_eef[:3, 3]
print(f"  3D cam pos: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) m   depth={p[2]:.3f} m")
if p[2] > 0:
    u = int(K[0,0] * p[0] / p[2] + K[0,2])
    v = int(K[1,1] * p[1] / p[2] + K[1,2])
    print(f"  Pixel (approx, 848x480): u={u}  v={v}  (center=424,240)")
else:
    print("  Behind camera (depth <= 0) — something is wrong")

# Also try negated Z (in case Kinova Z is up but we think it's down)
print("\nIf we NEGATE the Kinova Z (test if Z convention is flipped):")
T_base_eef2 = T_base_eef.copy()
T_base_eef2[2, 3] = -c.Z
T_cam_eef2 = tf_world2cam @ np.linalg.inv(T_base_task) @ T_base_eef2
p2 = T_cam_eef2[:3, 3]
print(f"  3D cam pos: ({p2[0]:.3f}, {p2[1]:.3f}, {p2[2]:.3f}) m   depth={p2[2]:.3f} m")
if p2[2] > 0:
    u2 = int(K[0,0] * p2[0] / p2[2] + K[0,2])
    v2 = int(K[1,1] * p2[1] / p2[2] + K[1,2])
    print(f"  Pixel (approx, 848x480): u={u2}  v={v2}  (center=424,240)")

api.CloseAPI()
print("\nDone.")
