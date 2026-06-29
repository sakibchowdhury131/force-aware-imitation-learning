"""
Collect raw RGB frames for training a UNet (robot-arm/gripper vs. held-tool
segmentation). Just dumps frames from all connected RealSense cameras while
you move the camera(s) around the robot (and move the arm with the joystick).

A live preview window is shown so you can frame the shot; frames are saved
automatically at --fps regardless of what's on screen.

Usage:
    python collect_unet_data.py --session 001 --duration 300 --fps 5
    (press Q or Esc in the preview window to stop early)

Output layout:
    data/unet_seg/<session>/
        cam0/  cam1/  ...   ← JPEG frames (000000.jpg, ...)
        meta.json
"""

import os, sys, json, time, argparse
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from pipeline_utils.cameras import MultiCamera, get_connected_serials


PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--session',  default='001', help='Session ID string (output subdir name)')
    p.add_argument('--duration', type=float, default=300.0, help='Recording duration in seconds')
    p.add_argument('--fps',      type=float, default=5.0,  help='Frames per second to save')
    p.add_argument('--width',    type=int,   default=848)
    p.add_argument('--height',   type=int,   default=480)
    p.add_argument('--no_preview', action='store_true', help='Disable live preview window')
    p.add_argument('--out', default=os.path.join(PIPELINE_DIR, 'data', 'unet_seg'))
    return p.parse_args()


def main():
    args = parse_args()

    session_dir = os.path.join(args.out, args.session)
    os.makedirs(session_dir, exist_ok=True)

    serials = get_connected_serials()
    if not serials:
        print("ERROR: No RealSense cameras detected.")
        return
    print(f"Found {len(serials)} camera(s): {serials}")

    cam_dirs = []
    for i in range(len(serials)):
        d = os.path.join(session_dir, f'cam{i}')
        os.makedirs(d, exist_ok=True)
        cam_dirs.append(d)

    with MultiCamera(serials, resolution=(args.width, args.height), fps=30) as mc:
        print("Warming up cameras (3s for auto-exposure to settle)...")
        mc.warmup(90)

        if not args.no_preview:
            for i in range(len(serials)):
                cv2.namedWindow(f"cam{i} ({serials[i]})", cv2.WINDOW_NORMAL)

        period = 1.0 / args.fps
        print(f"\nRecording up to {args.duration:.0f}s at {args.fps} fps "
              f"(~{int(args.duration * args.fps)} frames/cam).")
        print("Move the camera(s) around the robot and move the arm with the joystick.")
        if not args.no_preview:
            print("Press Q or Esc in the preview window to stop early.\n")

        saved = 0
        t_start = time.time()
        next_save = t_start
        stop = False

        while not stop:
            now = time.time()
            elapsed = now - t_start
            if elapsed >= args.duration:
                break

            frames = mc.grab_all()

            if now >= next_save:
                tag = f'{saved:06d}'
                for rgb, cam_dir in zip(frames, cam_dirs):
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(cam_dir, f'{tag}.jpg'), bgr,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1
                next_save += period
                if saved % 10 == 0:
                    print(f"  {elapsed:.1f}s / {args.duration:.0f}s  ({saved} frames)")

            if not args.no_preview:
                for i, rgb in enumerate(frames):
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    cv2.imshow(f"cam{i} ({serials[i]})", bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    stop = True

    if not args.no_preview:
        cv2.destroyAllWindows()

    meta = {
        'session': args.session,
        'serials': serials,
        'save_fps': args.fps,
        'resolution': [args.width, args.height],
        'n_frames': saved,
        'duration_sec': time.time() - t_start,
        'recorded_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(session_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved {saved} frame(s) per camera to '{session_dir}/'")
    print(f"Cameras: {[f'cam{i}' for i in range(len(serials))]}")


if __name__ == '__main__':
    main()
