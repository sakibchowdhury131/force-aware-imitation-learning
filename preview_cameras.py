"""
Live preview of all connected RealSense cameras — no recording, just for
positioning the cameras before calibration / data collection.

Usage:
  python preview_cameras.py
  (press Q or Esc to quit)
"""

import cv2
import numpy as np
from pipeline_utils.cameras import MultiCamera, get_connected_serials


def main():
    serials = get_connected_serials()
    if not serials:
        print("ERROR: No RealSense cameras detected.")
        return
    print(f"Found {len(serials)} camera(s): {serials}")

    with MultiCamera(serials, resolution=(848, 480), fps=30) as mc:
        print("Warming up...")
        mc.warmup(30)
        print("Streaming. Press Q or Esc to quit.")

        for i in range(len(mc)):
            cv2.namedWindow(f"cam{i} ({serials[i]})", cv2.WINDOW_NORMAL)

        while True:
            frames = mc.grab_all()
            for i, rgb in enumerate(frames):
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow(f"cam{i} ({serials[i]})", bgr)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
