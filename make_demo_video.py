#!/usr/bin/env python3
"""
Builds a demo video from replay_episode.py --capture_camera frames across
multiple episodes (e.g. for a GitHub README/demo reel).

Uses cv2.VideoWriter directly (no ffmpeg dependency, since ffmpeg isn't
installed on this machine).

Two layouts:
  --layout grid       (default) all episodes play SIMULTANEOUSLY, tiled into
                       a grid (2x2 for 4 episodes) -- one frame of output
                       contains one frame from EACH episode, side by side.
                       Episodes with fewer frames freeze on their last frame
                       until the longest episode finishes.
  --layout sequential  episodes play one after another in a single frame.

Grid mode can also overlay a picture-in-picture inset in each cell's
bottom-right corner, showing the ORIGINAL HUMAN DEMO frame (from
<episode_dir>/cam{N}/, the raw recording) at the exact same frame_id as the
main robot-replay frame -- side by side human demo vs. robot replay.

Usage:
    # 2x2 grid, all 4 episodes running at once
    python make_demo_video.py --task PastaTransfer_force --episodes 001 002 003 004

    # with a human-demo picture-in-picture inset (cam0 of the original recording)
    python make_demo_video.py --task PastaTransfer_force --episodes 001 002 003 004 \\
        --pip_camera 0

    # one after another instead
    python make_demo_video.py --task PastaTransfer_force --episodes 001 002 003 004 \\
        --layout sequential
"""
import os, argparse, glob, re
import numpy as np
import cv2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', required=True)
    p.add_argument('--episodes', nargs='+', required=True, help='e.g. 001 002 003 004')
    p.add_argument('--camera', type=int, default=1,
                   help='Which replay camera to use for the main view (default: 1)')
    p.add_argument('--fps', type=float, default=15.0, help='Output video frame rate')
    p.add_argument('--output', default=None, help='Default: media/demo_<task>.mp4')
    p.add_argument('--no_label', action='store_true', help='Skip the episode-number text overlay')
    p.add_argument('--episodes_root', default='data/episodes')
    p.add_argument('--layout', choices=['grid', 'sequential'], default='grid')
    p.add_argument('--cols', type=int, default=None,
                   help='Grid columns (default: ceil(sqrt(n_episodes)), e.g. 2 for 4 episodes)')
    p.add_argument('--cell_width', type=int, default=None,
                   help='Grid mode: width per cell in pixels (default: keeps total width '
                        '<= one source frame width, i.e. 848 // cols)')
    p.add_argument('--pip_camera', type=int, default=None,
                   help='Grid mode: add a small picture-in-picture inset in each cell\'s '
                        'bottom-right corner, showing the ORIGINAL HUMAN DEMO frame from this '
                        'camera (<episode_dir>/cam{N}/), matched to the same frame_id as the '
                        'main robot-replay frame. Default: no PiP.')
    p.add_argument('--pip_scale', type=float, default=0.35,
                   help='PiP inset width as a fraction of the cell width (default 0.35)')
    return p.parse_args()


def _frame_id(path):
    return int(os.path.splitext(os.path.basename(path))[0].split('_')[0])


def load_replay_frames(args, ep, camera=None):
    """Returns (cam_dir, [(frame_id, path), ...]) from the REPLAY capture."""
    camera = args.camera if camera is None else camera
    cam_dir = os.path.join(args.episodes_root, args.task, ep, 'replay', f'cam{camera}')
    paths = sorted(glob.glob(os.path.join(cam_dir, '*.jpg')), key=_frame_id)
    return cam_dir, [(_frame_id(p), p) for p in paths]


def load_demo_frame_dict(args, ep, camera):
    """Returns (cam_dir, {frame_id: path}) from the ORIGINAL human-demo recording
    (01_record.py's saved frames -- NOT the replay capture)."""
    cam_dir = os.path.join(args.episodes_root, args.task, ep, f'cam{camera}')
    paths = glob.glob(os.path.join(cam_dir, '*.jpg'))
    return cam_dir, {_frame_id(p): p for p in paths}


def overlay_pip(cell, pip_img, cell_w, cell_h, scale, margin=6):
    pip_h_src, pip_w_src = pip_img.shape[:2]
    pip_w = max(1, int(cell_w * scale))
    pip_h = int(pip_w * pip_h_src / pip_w_src)
    pip_resized = cv2.resize(pip_img, (pip_w, pip_h))

    x1, y1 = cell_w - pip_w - margin, cell_h - pip_h - margin
    x2, y2 = x1 + pip_w, y1 + pip_h
    # White border so the inset reads clearly against a busy background
    cv2.rectangle(cell, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), -1)
    cell[y1:y2, x1:x2] = pip_resized
    return cell


def label_frame(img, label):
    cv2.putText(img, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 255, 0), 1, cv2.LINE_AA)
    return img


def run_sequential(args, out_path):
    writer = None
    total_frames = 0
    for ep in args.episodes:
        cam_dir, frames = load_replay_frames(args, ep)
        if not frames:
            print(f'WARNING: no frames found in {cam_dir} -- skipping episode {ep}')
            continue
        print(f'Episode {ep}: {len(frames)} frames from {cam_dir}')
        for _, fp in frames:
            img = cv2.imread(fp)
            if img is None:
                continue
            if writer is None:
                h, w = img.shape[:2]
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h))
                print(f'Video: {w}x{h} @ {args.fps}fps -> {out_path}')
            if not args.no_label:
                label_frame(img, f'Episode {ep}')
            writer.write(img)
            total_frames += 1
    return writer, total_frames


def run_grid(args, out_path):
    n = len(args.episodes)
    cols = args.cols or int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    print(f'Grid layout: {rows} rows x {cols} cols for {n} episodes')

    episode_frames = []   # list of (ep, [(frame_id, path)], demo_frame_dict)
    max_len = 0
    for ep in args.episodes:
        cam_dir, frames = load_replay_frames(args, ep)
        if not frames:
            print(f'WARNING: no frames found in {cam_dir} -- skipping episode {ep}')
            continue
        print(f'Episode {ep}: {len(frames)} frames from {cam_dir}')
        demo_dict = {}
        if args.pip_camera is not None:
            demo_dir, demo_dict = load_demo_frame_dict(args, ep, args.pip_camera)
            if not demo_dict:
                print(f'  WARNING: --pip_camera {args.pip_camera} requested but no demo frames in {demo_dir}')
        episode_frames.append((ep, frames, demo_dict))
        max_len = max(max_len, len(frames))

    if not episode_frames:
        return None, 0

    sample = cv2.imread(episode_frames[0][1][0][1])
    src_h, src_w = sample.shape[:2]
    cell_w = args.cell_width or (src_w // cols)
    cell_h = int(cell_w * src_h / src_w)
    out_w, out_h = cell_w * cols, cell_h * rows
    print(f'Cell size: {cell_w}x{cell_h}  ->  output canvas: {out_w}x{out_h}')

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (out_w, out_h))
    print(f'Video: {out_w}x{out_h} @ {args.fps}fps -> {out_path}')

    # Preload/cache: read each episode's frame lazily as we go, holding the
    # last successfully-read frame for episodes shorter than max_len.
    last_frame = [None] * len(episode_frames)
    last_pip_frame = [None] * len(episode_frames)

    for i in range(max_len):
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        for idx, (ep, frames, demo_dict) in enumerate(episode_frames):
            if i < len(frames):
                frame_id, fp = frames[i]
                img = cv2.imread(fp)
                if img is not None:
                    last_frame[idx] = img

                if demo_dict:
                    demo_path = demo_dict.get(frame_id)
                    if demo_path is not None:
                        demo_img = cv2.imread(demo_path)
                        if demo_img is not None:
                            last_pip_frame[idx] = demo_img

            img = last_frame[idx]
            if img is None:
                continue

            cell = cv2.resize(img, (cell_w, cell_h))
            if not args.no_label:
                label_frame(cell, f'Episode {ep}')
            if args.pip_camera is not None and last_pip_frame[idx] is not None:
                overlay_pip(cell, last_pip_frame[idx], cell_w, cell_h, args.pip_scale)
            r, c = divmod(idx, cols)
            canvas[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = cell
        writer.write(canvas)

    return writer, max_len


def main():
    args = parse_args()
    out_path = args.output or os.path.join('media', f'demo_{args.task}.mp4')
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    if args.layout == 'grid':
        writer, total_frames = run_grid(args, out_path)
    else:
        writer, total_frames = run_sequential(args, out_path)

    if writer is None:
        print('ERROR: no frames written -- nothing to save.')
        return

    writer.release()
    duration = total_frames / args.fps
    print(f'\nSaved {total_frames} frames ({duration:.1f}s) -> {out_path}')


if __name__ == '__main__':
    main()
