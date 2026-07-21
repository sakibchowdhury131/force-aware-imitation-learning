#!/usr/bin/env python3
"""
replay_live_display.py
======================
Offline counterpart to `live_novel_views.py`: replays a recorded episode through the
exact live-display pipeline (same `infer_pair` + `build_canvas`) and writes the
composited window to an MP4 — useful for inspecting what the live tool would show on
a capture, without cameras attached.

Faithful timing (default, --mode live)
--------------------------------------
The live tool runs three threads: the real feeds + display refresh at the camera
rate (~30 fps) while a single inference thread produces a new novel-view set only at
the NoPoSplat rate (~2 fps), holding the previous set in between. This replay
reproduces that: the `cam0|cam1` tiles update every output frame, but the novel
tiles are recomputed only every ~`source_fps / infer_fps` frames (using the measured
inference time) and held otherwise. So the novel views visibly update ~2x/sec while
the real feeds move smoothly — matching the bottom-bar `infer` rate.

`--mode dense` instead infers every frame (novels update at the full output rate);
smoother but not representative of the live cadence.

Runs in the `noposplat` conda env (needs gsplat, hydra, the NoPoSplat backend repo).

Example
-------
    conda run -n noposplat python replay_live_display.py 001 002 \\
        --out live_display_videos --no_antialias
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NOPOSPLAT_ROOT = "/home/rhuang/Stevens/NoPoSplat/NoPoSplat"

sys.path.insert(0, PIPELINE_DIR)
from pipeline_utils import noposplat_core as R
from live_novel_views import build_canvas, infer_pair


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('episodes', nargs='+',
                   help='Episode dirs (each with cam0/, cam1/, meta.json). Relative '
                        'paths are resolved against this pipeline directory.')
    p.add_argument('--out', default='live_display_videos',
                   help='Output folder for the MP4s (created if missing).')
    p.add_argument('--mode', choices=['live', 'dense'], default='live',
                   help="'live' (default): novels refresh at the measured inference rate "
                        "(faithful). 'dense': infer every frame (smoother, not realistic).")
    # ---- backends ----
    p.add_argument('--noposplat_root', default=DEFAULT_NOPOSPLAT_ROOT,
                   help='Path to a clone of https://github.com/cvg/NoPoSplat.')
    p.add_argument('--checkpoint', default=None,
                   help='Path to a NoPoSplat .ckpt; if omitted, downloads from HF.')
    p.add_argument('--res', type=int, default=512, choices=[256, 512])
    p.add_argument('--device', default='cuda')
    # ---- geometry / views (match live_novel_views.py defaults) ----
    p.add_argument('--cam_pair', nargs=2, type=int, default=[0, 1])
    p.add_argument('--input_mode', choices=['aspect', 'letterbox'], default='letterbox')
    p.add_argument('--input_long_side', default='768')
    p.add_argument('--min_short_side', type=int, default=256)
    p.add_argument('--num_novel_views', type=int, default=6)
    p.add_argument('--view_seed', type=int, default=0)
    p.add_argument('--no_antialias', action='store_true')
    p.add_argument('--edge_trim', type=int, default=10)
    p.add_argument('--max_depth_pct', type=float, default=99.0)
    p.add_argument('--max_scale_pct', type=float, default=99.5)
    p.add_argument('--max_anisotropy', type=float, default=30.0)
    p.add_argument('--alpha_thresh', type=float, default=0.1)
    p.add_argument('--crop_pad', type=int, default=4)
    p.add_argument('--max_crop_frac', type=float, default=0.25)
    p.add_argument('--no_crop', action='store_true', help='Disable the black-border crop.')
    # ---- output video ----
    p.add_argument('--source_fps', type=float, default=None,
                   help='Capture fps to simulate (default: meta.json fps).')
    p.add_argument('--out_fps', type=float, default=None,
                   help='Playback fps of the written video (default: source fps).')
    p.add_argument('--max_frames', type=int, default=0, help='Cap frames (0 = all).')
    p.add_argument('--tile_w', type=int, default=320)
    p.add_argument('--grid_cols', type=int, default=4)
    return p.parse_args()


def episode_setup(ep_dir, args, device):
    """Build the per-episode geometry/intrinsics/context once (mirrors the setup in
    live_novel_views.main). Returns a dict of everything infer_pair needs."""
    meta = json.load(open(os.path.join(ep_dir, 'meta.json')))
    ow, oh = R._orig_size(meta, (args.res, args.res))
    c0, c1 = args.cam_pair
    K0_full, K1_full, _ = R.full_intrinsics(meta, c0, c1, ow, oh, 60.0)

    if args.input_mode == 'aspect':
        long_side = max(ow, oh) if str(args.input_long_side).lower() == 'native' \
            else int(args.input_long_side)
        geo = R.aspect_geometry(ow, oh, long_side, min_short=args.min_short_side)
    else:
        geo = R.letterbox_geometry(ow, oh, args.res)
    Win, Hin = geo['Win'], geo['Hin']

    K0n = R.normalize_K(R.input_K(K0_full, geo), Win, Hin)
    K1n = R.normalize_K(R.input_K(K1_full, geo), Win, Hin)
    am, npix = R.input_active_mask(geo)
    am = R.erode_active_mask(am, Hin, Win, args.edge_trim)

    return {
        'meta': meta, 'ow': ow, 'oh': oh, 'c0': c0, 'c1': c1,
        'geo': geo, 'N_per_view': Win * Hin,
        'K0_full': K0_full, 'K1_full': K1_full,
        'intr_ctx': torch.tensor(np.stack([K0n, K1n])[None], device=device, dtype=torch.float32),
        'K0_render_t': torch.tensor(K0_full, device=device, dtype=torch.float32),
        'active_t': torch.tensor(am, device=device),
        'cam1_native_pix': npix[am],
        'n_frames': int(meta.get('n_frames') or
                        len(os.listdir(os.path.join(ep_dir, f'cam{c0}')))),
        'source_fps': float(args.source_fps or meta.get('fps', 30)),
        'task': meta.get('task', os.path.basename(ep_dir.rstrip('/'))),
    }


def run_episode(ep_dir, args, encoder, device):
    s = episode_setup(ep_dir, args, device)
    c0, c1 = s['c0'], s['c1']
    nf = s['n_frames'] if args.max_frames <= 0 else min(s['n_frames'], args.max_frames)
    fps = s['source_fps']
    out_fps = float(args.out_fps or fps)
    crop_on = not args.no_crop
    name = os.path.basename(ep_dir.rstrip('/'))

    def read(i):
        p0 = os.path.join(ep_dir, f'cam{c0}', f'{i:06d}.jpg')
        p1 = os.path.join(ep_dir, f'cam{c1}', f'{i:06d}.jpg')
        return R.read_rgb(p0), R.read_rgb(p1)

    def infer(rgb0, rgb1, nonce):
        return infer_pair(R, encoder, rgb0, rgb1, s['geo'], s['N_per_view'],
                          s['intr_ctx'], s['active_t'], s['cam1_native_pix'],
                          s['K0_full'], s['K1_full'], s['K0_render_t'],
                          s['ow'], s['oh'], args, crop_on, device, nonce)

    # --- pass 1: build the inference timeline ---
    # In 'live' mode the inference thread always works on the latest frame and a
    # result becomes available after the measured inference time (~round(dt*fps)
    # source frames later); in 'dense' mode every frame is inferred.
    timeline = []          # (available_from_frame, novels, infer_dt)
    last = []
    nonce = 0
    i = 0
    t_all = time.time()
    while i < nf:
        rgb0, rgb1 = read(i)
        t0 = time.time()
        try:
            novels, _, _, _, _, _ = infer(rgb0, rgb1, nonce)
            last = novels
        except RuntimeError:                  # PnP failed -> reuse previous novels
            novels = last
        dt = time.time() - t0
        nonce += 1
        span = 1 if args.mode == 'dense' else max(1, int(round(dt * fps)))
        timeline.append((i + span, novels, dt))
        i += span
        print(f"  {name} [{args.mode}] inferred frame {i if i<=nf else nf}/{nf} "
              f"({1.0/dt:4.1f} fps)", end='\r')
    print()

    # --- pass 2: render every output frame (real feed fresh, novels held) ---
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f'{name}_live_display.mp4')
    writer = None
    ti = 0                                    # index into timeline
    active = 0                                # currently-displayed result
    for j in range(nf):
        while ti < len(timeline) and timeline[ti][0] <= j:
            active = ti
            ti += 1
        _, novels, dt = timeline[active]      # forward-fills the first result to frame 0
        rgb0, rgb1 = read(j)
        status = [
            (f'disp {out_fps:4.1f} fps | infer {1.0/dt:4.1f} fps | crop '
             f'{"ON" if crop_on else "OFF"} | view_seed={args.view_seed} | mode={args.mode}',
             (200, 200, 200)),
            (f'[replay]  {s["task"]} ({name})  frame {j+1}/{nf}', (40, 210, 210)),
        ]
        canvas = build_canvas(rgb0, rgb1, novels, args.tile_w, args.grid_cols, status, False)
        bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        if writer is None:
            h, w = bgr.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                     out_fps, (w, h))
        writer.write(bgr)
    writer.release()
    print(f"{name}: wrote {out_path} | {nf} frames @ {out_fps:g}fps | "
          f"{len(timeline)} inferences | {time.time()-t_all:.0f}s")


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("WARNING: CUDA not available -> CPU (gsplat needs CUDA; very slow).")

    ckpt = args.checkpoint
    if ckpt is None:
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download(repo_id=R.HF_REPO,
                               filename=R.CKPT_512 if args.res == 512 else R.CKPT_256)
    else:
        ckpt = os.path.expanduser(ckpt)
    print("Loading NoPoSplat encoder once...")
    encoder = R.load_noposplat_encoder(
        os.path.abspath(os.path.expanduser(args.noposplat_root)), ckpt, args.res, device)

    for ep in args.episodes:
        ep_dir = ep if os.path.isabs(ep) else os.path.join(PIPELINE_DIR, ep)
        if not os.path.isdir(ep_dir):
            print(f"[skip] {ep}: not a directory ({ep_dir})")
            continue
        run_episode(ep_dir, args, encoder, device)
    print("Done.")


if __name__ == '__main__':
    main()
