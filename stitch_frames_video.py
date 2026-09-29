#!/usr/bin/env python3
"""
Stitch a directory of JPEG frames into an .mp4 video (via OpenCV's own
VideoWriter -- no ffmpeg required). Generic frame-sequence stitcher, not
specific to any one episode/camera.

Standalone, additive script.

Usage:
    python stitch_frames_video.py --frames_dir data/episodes/PastaTransfer_force/046/cam0 \\
        --save_path data/episodes/PastaTransfer_force/046/cam0_demo.mp4 --fps 30
"""
import os, glob, argparse
import cv2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--frames_dir', required=True)
    p.add_argument('--save_path', required=True)
    p.add_argument('--fps', type=float, default=30.0)
    p.add_argument('--label', default=None, help='Optional text burned into the top-left corner')
    p.add_argument('--pattern', default='*.jpg', help='Glob pattern within --frames_dir')
    return p.parse_args()


def main():
    args = parse_args()
    files = sorted(glob.glob(os.path.join(args.frames_dir, args.pattern)))
    if not files:
        raise SystemExit(f"No .jpg frames found in {args.frames_dir}")

    sample = cv2.imread(files[0])
    h, w = sample.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(args.save_path, fourcc, args.fps, (w, h))

    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        if args.label:
            cv2.putText(img, args.label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, args.label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(img)
    vw.release()

    print(f"Saved -> {args.save_path}  ({len(files)} frames @ {args.fps}fps = "
          f"{len(files)/args.fps:.1f}s, {w}x{h}, {os.path.getsize(args.save_path)/1e6:.1f} MB)")


if __name__ == '__main__':
    main()
