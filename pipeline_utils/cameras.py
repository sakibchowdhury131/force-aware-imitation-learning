"""RealSense camera utilities — works with any number of cameras."""
import numpy as np
import pyrealsense2 as rs


def get_connected_serials():
    ctx = rs.context()
    return [d.get_info(rs.camera_info.serial_number) for d in ctx.devices]


class RealSenseCamera:
    def __init__(self, serial: str, resolution=(848, 480), fps=30):
        self.serial = serial
        self.W, self.H = resolution
        self.fps = fps
        self._pipe        = None
        self._profile     = None
        self._align       = None
        self._depth_scale = 0.001  # default; overwritten on start()

    def start(self):
        self._pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.W, self.H, rs.format.rgb8,  self.fps)
        cfg.enable_stream(rs.stream.depth, self.W, self.H, rs.format.z16,   self.fps)
        self._profile = self._pipe.start(cfg)

        color_sensor = self._profile.get_device().first_color_sensor()
        color_sensor.set_option(rs.option.enable_auto_exposure, 1)
        color_sensor.set_option(rs.option.enable_auto_white_balance, 1)

        depth_sensor = self._profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()  # metres per unit

        self._align = rs.align(rs.stream.color)

    def stop(self):
        if self._pipe:
            self._pipe.stop()
            self._pipe = None

    def get_intrinsics(self):
        intr = (self._profile.get_stream(rs.stream.color)
                .as_video_stream_profile().get_intrinsics())
        K = np.array([[intr.fx, 0, intr.ppx],
                      [0, intr.fy, intr.ppy],
                      [0, 0, 1]], dtype=np.float64)
        D = np.array(intr.coeffs, dtype=np.float64)
        return K, D

    def grab_rgbd(self) -> tuple:
        """Returns (rgb HxWx3 uint8, depth HxW float32 metres, aligned)."""
        frames    = self._align.process(self._pipe.wait_for_frames(timeout_ms=3000))
        rgb       = np.asanyarray(frames.get_color_frame().get_data())
        depth_raw = np.asanyarray(frames.get_depth_frame().get_data())
        depth_m   = depth_raw.astype(np.float32) * self._depth_scale
        return rgb, depth_m

    def grab_rgb(self) -> np.ndarray:
        """Returns (H, W, 3) uint8 RGB (depth discarded)."""
        rgb, _ = self.grab_rgbd()
        return rgb

    def warmup(self, n=90):
        for _ in range(n):
            self._pipe.wait_for_frames()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


class MultiCamera:
    def __init__(self, serials=None, resolution=(848, 480), fps=30):
        if serials is None:
            serials = get_connected_serials()
        if not serials:
            raise RuntimeError("No RealSense cameras found.")
        self.cameras = [RealSenseCamera(sn, resolution, fps) for sn in serials]
        self.serials = serials

    def start(self):
        for cam in self.cameras:
            cam.start()

    def stop(self):
        for cam in self.cameras:
            cam.stop()

    def warmup(self, n=90):
        for cam in self.cameras:
            cam.warmup(n)

    def grab_all(self) -> list:
        """Returns list of (H, W, 3) uint8 RGB frames, one per camera."""
        return [cam.grab_rgb() for cam in self.cameras]

    def grab_all_rgbd(self) -> list:
        """Returns list of (rgb, depth_metres) tuples, one per camera."""
        return [cam.grab_rgbd() for cam in self.cameras]

    def get_all_intrinsics(self):
        return [cam.get_intrinsics() for cam in self.cameras]

    def __len__(self):
        return len(self.cameras)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
