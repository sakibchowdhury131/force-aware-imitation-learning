"""
Step 1 — Record a demonstration episode.

Usage:
    python 01_record.py --task hammer --episode 001 --duration 15

Output layout:
    data/episodes/<task>/<episode>/
        cam0/  cam1/  ...  camN/          ← JPEG frames (000000.jpg, ...)
        cam0_depth/  cam1_depth/  ...     ← 16-bit PNG depth (000000.png, mm units)
        meta.json                         ← camera serials, fps, frame count, intrinsics
"""

import os, sys, json, time, argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from pipeline_utils.cameras import MultiCamera, get_connected_serials


PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--task',     default='task',  help='Task name (e.g. hammer)')
    p.add_argument('--episode',  default='001',   help='Episode ID string')
    p.add_argument('--duration', type=float, default=15.0, help='Recording duration in seconds')
    p.add_argument('--fps',      type=int,   default=30)
    p.add_argument('--width',    type=int,   default=848)
    p.add_argument('--height',   type=int,   default=480)
    p.add_argument('--out',      default=os.path.join(PIPELINE_DIR, 'data', 'episodes'))
    return p.parse_args()


def main():
    args = parse_args()

    episode_dir = os.path.join(args.out, args.task, args.episode)
    os.makedirs(episode_dir, exist_ok=True)

    serials = get_connected_serials()
    print(f"Found {len(serials)} camera(s): {serials}")

    cam_dirs   = []
    depth_dirs = []
    for i in range(len(serials)):
        d = os.path.join(episode_dir, f'cam{i}')
        os.makedirs(d, exist_ok=True)
        cam_dirs.append(d)
        dd = os.path.join(episode_dir, f'cam{i}_depth')
        os.makedirs(dd, exist_ok=True)
        depth_dirs.append(dd)

    with MultiCamera(serials, resolution=(args.width, args.height), fps=args.fps) as mc:
        print("Warming up cameras (3s for auto-exposure to settle)...")
        mc.warmup(90)

        # Sanity-check brightness after warmup
        test_frames = mc.grab_all()
        for i, f in enumerate(test_frames):
            mean_brightness = f.mean()
            print(f"  cam{i} brightness after warmup: {mean_brightness:.1f}/255", end="")
            if mean_brightness < 30:
                print("  WARNING: very dark — check lighting or re-run")
            else:
                print("  OK")

        intrinsics = []
        for cam in mc.cameras:
            K, D = cam.get_intrinsics()
            intrinsics.append({'K': K.tolist(), 'D': D.tolist()})

        n_frames = int(args.duration * args.fps)
        print(f"\nRecording {args.duration:.0f}s ({n_frames} frames).")
        print("Perform the task now...\n")

        saved = 0
        t_start = time.time()
        for i in range(n_frames):
            rgbd_frames = mc.grab_all_rgbd()
            tag = f'{i:06d}'
            for cam_i, ((rgb, depth_m), cam_dir, depth_dir) in enumerate(
                    zip(rgbd_frames, cam_dirs, depth_dirs)):
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(cam_dir, f'{tag}.jpg'), bgr,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                depth_mm = (depth_m * 1000.0).clip(0, 65535).astype(np.uint16)
                cv2.imwrite(os.path.join(depth_dir, f'{tag}.png'), depth_mm)
            saved += 1
            if i % args.fps == 0:
                elapsed = time.time() - t_start
                print(f"  {elapsed:.1f}s / {args.duration:.0f}s  ({saved} frames)")

    meta = {
        'task': args.task,
        'episode': args.episode,
        'serials': serials,
        'fps': args.fps,
        'resolution': [args.width, args.height],
        'n_frames': saved,
        'duration_sec': args.duration,
        'intrinsics': intrinsics,
        'has_depth': True,
        'depth_scale': 0.001,
        'recorded_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(episode_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved {saved} frames to '{episode_dir}/'")
    print(f"Cameras: {[f'cam{i}' for i in range(len(serials))]}")
    print(f"Depth:   {[f'cam{i}_depth' for i in range(len(serials))]}")
    print(f"Next: python 02_augment.py --episode_dir {episode_dir}")


if __name__ == '__main__':
    main()
