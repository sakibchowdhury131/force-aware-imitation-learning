#!/usr/bin/env python3
"""
live_novel_views.py
===================
Real-time novel-view preview from two live RealSense cameras using NoPoSplat.

This is the interactive, real-time counterpart to `02_augment.py`. Where step 2
runs MASt3R + per-frame 3D-Gaussian *training* (slow, seconds/frame), NoPoSplat is
a single feed-forward pass, so the same augmentation viewpoints can be previewed
live at interactive rates. Point the two cameras at the scene, watch the novel
views update, and only hit record once the coverage looks good -- collapsing the
record -> augment loop into one fast, visual step.

What it shows (the "replicate augmentation" preview)
----------------------------------------------------
For every inferred frame pair (cam0/cam1) it reproduces exactly what `02_augment`
/ `render_novel_views.py` would emit: NoPoSplat Gaussians -> cam1 pose via PnP ->
`num_novel_views` Beta-sampled Slerp viewpoints -> alpha-cropped renders. The
viewpoint RNG is seeded by ``--view_seed`` + a per-frame nonce, so the novel
viewpoints VARY every frame -- just like the offline augmentation -- for both the
live display and the recorded data. The single render per frame feeds both, so what
you see is what gets saved. Press ``r`` to jump to a different random sequence.

Recording (start / pause / stop)
--------------------------------
Press ``s`` to start recording, ``p`` to pause/resume, ``x`` to stop and finalize.
Capture and augmentation are decoupled, matching the offline pipeline's data model:
the raw RGB-D stream is saved DENSELY at the capture rate (what step 4's temporal
tracking needs), while the novel-view augmentation is saved SPARSELY at the
inference rate (throttled by ``--record_fps``), each augmented frame tagged with the
raw index it came from. The result is a drop-in replacement for `01_record.py` +
`02_augment.py`::

    <out>/<task>/<episode>/
        cam0/ cam1/                 raw native frames        (000000.jpg, ...)
        cam0_depth/ cam1_depth/     16-bit depth (mm)        (000000.png, ...)
        meta.json                   serials, fps, intrinsics, n_frames
        augmented/
            real/   {idx}_cam0.jpg, {idx}_cam1.jpg
            novel/  {idx}_novel{kk}.jpg
            novel_cameras.npz        per-view crop-aware intrinsics + poses

So after recording you can jump straight to `03_segment.py`.

Hotkeys
-------
    s  start recording          p  pause / resume recording
    x  stop & finalize episode   r  shift viewpoint sequence
    c  toggle crop on/off        h  toggle help overlay
    q / ESC  quit

Environment
-----------
Runs entirely in the `noposplat` conda env (needs gsplat, hydra, the NoPoSplat
repo as a backend, and pyrealsense2). Requires a CUDA GPU and >= 2 RealSense
cameras connected.

Example
-------
    conda run -n noposplat python live_novel_views.py \
        --task hammer --episode 001 \
        --num_novel_views 6 --input_mode letterbox --no_antialias
"""

import argparse
import json
import os
import sys
import threading
import time

import cv2
import numpy as np
import torch

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NOPOSPLAT_ROOT = "/home/rhuang/Stevens/NoPoSplat/NoPoSplat"

sys.path.insert(0, PIPELINE_DIR)
from pipeline_utils.cameras import MultiCamera, get_connected_serials
from pipeline_utils import noposplat_core as R   # vendored NoPoSplat functions


class Latest:
    """A single-slot, thread-safe mailbox holding only the most recent value.
    The capture thread overwrites frames faster than the consumer reads them; we
    always want the newest, so old values are simply dropped (no queue backlog)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._v = None

    def put(self, v):
        with self._lock:
            self._v = v

    def get(self):
        with self._lock:
            return self._v


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # ---- backends ----
    p.add_argument('--noposplat_root', default=DEFAULT_NOPOSPLAT_ROOT,
                   help='Path to a clone of https://github.com/cvg/NoPoSplat.')
    p.add_argument('--checkpoint', default=None,
                   help='Path to a NoPoSplat .ckpt; if omitted, downloads from HF.')
    p.add_argument('--res', type=int, default=512, choices=[256, 512],
                   help='Checkpoint resolution (512 generalises better, 256 is faster).')
    p.add_argument('--device', default='cuda')
    # ---- cameras ----
    p.add_argument('--serials', nargs='*', default=None,
                   help='RealSense serials to use (default: auto-detect all).')
    p.add_argument('--cam_pair', nargs=2, type=int, default=[0, 1],
                   help='Which two camera indices feed NoPoSplat.')
    p.add_argument('--width', type=int, default=848)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--warmup', type=int, default=30,
                   help='Frames to drop so auto-exposure settles before streaming.')
    # ---- input geometry (mirrors render_novel_views.py) ----
    p.add_argument('--input_mode', choices=['aspect', 'letterbox'], default='letterbox')
    p.add_argument('--input_long_side', default='768')
    p.add_argument('--min_short_side', type=int, default=256)
    # ---- novel views ----
    p.add_argument('--num_novel_views', type=int, default=6,
                   help='Must be even (cam0-biased + cam1-biased). Lower = faster live.')
    p.add_argument('--view_seed', type=int, default=0,
                   help='Base seed for the per-frame Beta-sampled viewpoints (the actual '
                        'seed is view_seed + frame nonce, so viewpoints vary every frame). '
                        "Key 'r' shifts it to jump to a different random sequence.")
    p.add_argument('--infer_every', type=int, default=1,
                   help='(legacy; ignored) inference now runs in its own thread on the '
                        'latest captured frame, so the live feed is never gated by it.')
    # ---- render / cleanup (same knobs/defaults as render_novel_views.py) ----
    p.add_argument('--no_antialias', action='store_true',
                   help="gsplat 'classic' rasterization (crisper) vs 'antialiased'.")
    p.add_argument('--edge_trim', type=int, default=10)
    p.add_argument('--max_depth_pct', type=float, default=99.0)
    p.add_argument('--max_scale_pct', type=float, default=99.5)
    p.add_argument('--max_anisotropy', type=float, default=30.0)
    p.add_argument('--alpha_thresh', type=float, default=0.1)
    p.add_argument('--crop_pad', type=int, default=4)
    p.add_argument('--max_crop_frac', type=float, default=0.25)
    p.add_argument('--no_crop', action='store_true', help='Start with cropping off.')
    # ---- recording output ----
    p.add_argument('--task', default='task', help='Task name (episode grouping).')
    p.add_argument('--episode', default='001', help='Episode ID string.')
    p.add_argument('--out', default=os.path.join(PIPELINE_DIR, 'data', 'episodes'))
    p.add_argument('--record_fps', type=float, default=None,
                   help='Throttle the AUGMENTED (novel-view) frames to this rate (e.g. 2 '
                        '~= sample_every=15). Raw RGB-D is always saved densely at the '
                        'capture rate. Default: augment every inferred frame.')
    # ---- display ----
    p.add_argument('--tile_w', type=int, default=320, help='Per-tile display width (px).')
    p.add_argument('--grid_cols', type=int, default=4, help='Columns in the novel grid.')
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Display helpers                                                              #
# --------------------------------------------------------------------------- #
def _fit(img_rgb, tw):
    """Resize an RGB tile to width `tw`, preserving aspect ratio."""
    h, w = img_rgb.shape[:2]
    th = max(1, int(round(tw * h / w)))
    return cv2.resize(img_rgb, (tw, th), interpolation=cv2.INTER_AREA)


def _label(img_rgb, text, color=(40, 220, 40)):
    """Draw a small caption box top-left (in place, RGB)."""
    cv2.rectangle(img_rgb, (0, 0), (len(text) * 9 + 8, 18), (0, 0, 0), -1)
    cv2.putText(img_rgb, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                color, 1, cv2.LINE_AA)
    return img_rgb


def _hstack(tiles, gap=4, bg=30):
    """Horizontally concat equal-width tiles, padding heights to the tallest."""
    h = max(t.shape[0] for t in tiles)
    out = []
    for t in tiles:
        if t.shape[0] < h:
            pad = np.full((h - t.shape[0], t.shape[1], 3), bg, np.uint8)
            t = np.vstack([t, pad])
        out.append(t)
        out.append(np.full((h, gap, 3), bg, np.uint8))
    return np.hstack(out[:-1]) if out else np.zeros((1, 1, 3), np.uint8)


def _vstack(rows, gap=4, bg=30):
    """Vertically concat rows, padding widths to the widest."""
    w = max(r.shape[1] for r in rows)
    out = []
    for r in rows:
        if r.shape[1] < w:
            pad = np.full((r.shape[0], w - r.shape[1], 3), bg, np.uint8)
            r = np.hstack([r, pad])
        out.append(r)
        out.append(np.full((gap, w, 3), bg, np.uint8))
    return np.vstack(out[:-1]) if out else np.zeros((1, 1, 3), np.uint8)


def build_canvas(real0, real1, novels, tile_w, grid_cols, status_lines, help_on):
    """Compose the live window: top row of the two real feeds, then a grid of the
    novel views, then a status/help footer. Inputs are RGB uint8."""
    top = _hstack([_label(_fit(real0.copy(), tile_w), 'cam0 (real)'),
                   _label(_fit(real1.copy(), tile_w), 'cam1 (real)')])

    tiles = [_label(_fit(n.copy(), tile_w), f'novel{k:02d}')
             for k, n in enumerate(novels)] if novels else []
    grid_rows = []
    for i in range(0, len(tiles), grid_cols):
        grid_rows.append(_hstack(tiles[i:i + grid_cols]))

    sections = [top] + grid_rows
    canvas = _vstack(sections)

    # footer
    W = canvas.shape[1]
    foot_h = 22 * len(status_lines) + 12
    footer = np.full((foot_h, W, 3), 20, np.uint8)
    for i, (txt, col) in enumerate(status_lines):
        cv2.putText(footer, txt, (8, 20 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, col, 1, cv2.LINE_AA)
    canvas = np.vstack([canvas, footer])

    if help_on:
        help_txt = ["s start   p pause/resume   x stop & save",
                    "r shift view seq   c toggle crop   h help   q quit"]
        hb = np.full((22 * len(help_txt) + 12, W, 3), 50, np.uint8)
        for i, t in enumerate(help_txt):
            cv2.putText(hb, t, (8, 20 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (230, 230, 230), 1, cv2.LINE_AA)
        canvas = np.vstack([canvas, hb])
    return canvas


# --------------------------------------------------------------------------- #
# Recorder (start / pause / stop state machine)                               #
# --------------------------------------------------------------------------- #
class Recorder:
    """Writes a pipeline-compatible episode incrementally, with raw capture and the
    novel-view augmentation DECOUPLED (matching 01_record + 02_augment):

      * RAW frames (cam{i}/, cam{i}_depth/) are saved DENSELY at the capture rate by
        the capture thread -- the dense RGB-D stream step 4's temporal tracking needs.
      * AUGMENTATION (augmented/real, augmented/novel, novel_cameras.npz) is saved
        SPARSELY at the inference rate by the inference thread, throttled to
        <= record_fps. Each augmented frame is tagged with the raw index it came
        from, so augmented tags are a subset of the dense raw indices.

    Thread-safe: capture thread calls save_raw(), inference thread calls
    save_augmentation(), main thread calls start()/pause()/stop(). An internal lock
    guards state + the shared index/counters/dict; the slow cv2.imwrite()s run
    outside the lock so the two threads' disk I/O don't serialize.

    States: idle -> recording <-> paused -> (stop) finalize."""

    def __init__(self, out_root, task, episode, serials, fps, resolution,
                 intrinsics, cam_pair, record_fps=None):
        self.base = os.path.join(out_root, task, episode)
        self.task, self.episode = task, episode
        self.serials, self.fps = serials, fps
        self.resolution = list(resolution)
        self.intrinsics = intrinsics           # list of {'K':..,'D':..} per camera
        self.cam_pair = cam_pair
        # throttle AUGMENTED frames to <= record_fps (None/<=0 = every inferred frame)
        self.min_interval = 1.0 / record_fps if record_fps and record_fps > 0 else 0.0
        self.record_fps = record_fps
        self._last_aug_t = 0.0
        self.state = 'idle'                     # 'idle' | 'recording' | 'paused'
        self.raw_idx = 0                        # dense raw frame counter (capture)
        self.aug_count = 0                      # sparse augmented frame counter
        self.novel_cams = {}
        self._dirs_ready = False
        self._lock = threading.Lock()

    @property
    def active(self):
        return self.state == 'recording'

    def _ensure_dirs(self):
        if self._dirs_ready:
            return
        # pick a non-clobbering directory if one already exists
        if os.path.exists(self.base) and os.listdir(self.base):
            suffix = time.strftime('%H%M%S')
            self.base = f"{self.base}_{suffix}"
        self.cam_dirs, self.depth_dirs = [], []
        for i in range(len(self.serials)):
            d = os.path.join(self.base, f'cam{i}')
            dd = os.path.join(self.base, f'cam{i}_depth')
            os.makedirs(d, exist_ok=True)
            os.makedirs(dd, exist_ok=True)
            self.cam_dirs.append(d)
            self.depth_dirs.append(dd)
        self.aug = os.path.join(self.base, 'augmented')
        self.real_dir = os.path.join(self.aug, 'real')
        self.novel_dir = os.path.join(self.aug, 'novel')
        os.makedirs(self.real_dir, exist_ok=True)
        os.makedirs(self.novel_dir, exist_ok=True)
        self._dirs_ready = True
        print(f"[rec] writing episode -> {self.base}")

    def start(self):
        with self._lock:
            if self.state == 'idle':
                self._ensure_dirs()
                self.state = 'recording'
                print("[rec] START")
            elif self.state == 'paused':
                self.state = 'recording'
                print("[rec] RESUME")

    def pause(self):
        with self._lock:
            if self.state == 'recording':
                self.state = 'paused'
                print(f"[rec] PAUSE @ raw {self.raw_idx} / aug {self.aug_count}")
                return
            resume = self.state == 'paused'
        if resume:
            self.start()

    def save_raw(self, rgbs, depths):
        """Dense raw save (capture thread). Returns the assigned frame index if it was
        saved (recording active), else None. The index is reserved under the lock; the
        imwrites happen outside it."""
        with self._lock:
            if self.state != 'recording':
                return None
            idx = self.raw_idx
            self.raw_idx += 1
            cam_dirs, depth_dirs = self.cam_dirs, self.depth_dirs
        tag = f'{idx:06d}'
        for i, (rgb, depth) in enumerate(zip(rgbs, depths)):
            cv2.imwrite(os.path.join(cam_dirs[i], f'{tag}.jpg'),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            if depth is not None:
                depth_mm = (depth * 1000.0).clip(0, 65535).astype(np.uint16)
                cv2.imwrite(os.path.join(depth_dirs[i], f'{tag}.png'), depth_mm)
        return idx

    def save_augmentation(self, idx, rgbs, novel_imgs, view_Ks, novel_w2c, cam_w2c):
        """Sparse augmentation save (inference thread), tagged with the raw index `idx`
        the frame came from. Throttled to <= record_fps; frames arriving too soon are
        dropped (-> a sparse subset of the dense raw frames, like sample_every)."""
        c0, c1 = self.cam_pair
        with self._lock:
            if self.state != 'recording':
                return
            now = time.time()
            if self.min_interval > 0.0 and (now - self._last_aug_t) < self.min_interval:
                return
            self._last_aug_t = now
            real_dir, novel_dir = self.real_dir, self.novel_dir
        tag = f'{idx:06d}'
        # augmented/real (the cam pair fed to NoPoSplat; same pixels as raw `idx`)
        cv2.imwrite(os.path.join(real_dir, f'{tag}_cam{c0}.jpg'),
                    cv2.cvtColor(rgbs[c0], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(real_dir, f'{tag}_cam{c1}.jpg'),
                    cv2.cvtColor(rgbs[c1], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        # augmented/novel
        for k, nv in enumerate(novel_imgs):
            cv2.imwrite(os.path.join(novel_dir, f'{tag}_novel{k:02d}.jpg'),
                        cv2.cvtColor(nv, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        with self._lock:
            self.novel_cams[f'{tag}_cam{c0}_w2c'] = cam_w2c[0]
            self.novel_cams[f'{tag}_cam{c1}_w2c'] = cam_w2c[1]
            self.novel_cams[f'{tag}_novel_w2c'] = novel_w2c
            self.novel_cams[f'{tag}_novel_K'] = np.stack(view_Ks)
            self.aug_count += 1

    def stop(self):
        """Finalize: write meta.json + novel_cameras.npz. Returns the episode dir."""
        with self._lock:
            if self.state == 'idle' or not self._dirs_ready:
                print("[rec] nothing recorded.")
                self.state = 'idle'
                return None
            meta = {
                'task': self.task, 'episode': self.episode, 'serials': self.serials,
                'fps': self.fps, 'resolution': self.resolution,
                'n_frames': self.raw_idx,             # dense raw count
                'n_augmented': self.aug_count,        # sparse augmented count
                'intrinsics': self.intrinsics, 'has_depth': True, 'depth_scale': 0.001,
                'cam_pair': self.cam_pair, 'source': 'live_novel_views.py',
                'record_fps': self.record_fps,        # augmentation throttle (null = infer rate)
                'recorded_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            }
            with open(os.path.join(self.base, 'meta.json'), 'w') as f:
                json.dump(meta, f, indent=2)
            np.savez(os.path.join(self.aug, 'novel_cameras.npz'), **self.novel_cams)
            print(f"[rec] STOP -> {self.raw_idx} raw / {self.aug_count} augmented "
                  f"frames to {self.base}")
            print(f"[rec] next: python 03_segment.py --episode_dir {self.base}")
            out = self.base
            # reset so a subsequent start writes a fresh episode dir
            self.state = 'idle'
            self.raw_idx = 0
            self.aug_count = 0
            self.novel_cams = {}
            self._dirs_ready = False
            self._last_aug_t = 0.0
        return out


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    if args.num_novel_views % 2 != 0:
        raise ValueError("--num_novel_views must be even")

    device = args.device if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("WARNING: CUDA not available -> CPU (gsplat needs CUDA; will be very slow).")

    # --- cameras ---
    serials = args.serials or get_connected_serials()
    if len(serials) < 2:
        raise RuntimeError(f"Need >= 2 RealSense cameras; found {len(serials)}: {serials}")
    c0, c1 = args.cam_pair
    if max(c0, c1) >= len(serials):
        raise RuntimeError(f"--cam_pair {args.cam_pair} out of range for {len(serials)} cams")
    print(f"Cameras ({len(serials)}): {serials} | NoPoSplat pair: cam{c0}+cam{c1}")

    mc = MultiCamera(serials, resolution=(args.width, args.height), fps=args.fps)
    mc.start()
    try:
        print(f"Warming up ({args.warmup} frames)...")
        mc.warmup(args.warmup)

        # --- per-camera intrinsics straight from the RealSense (native pixels) ---
        intr_list = []           # for meta.json (all cameras)
        Ks = []
        for cam in mc.cameras:
            K, D = cam.get_intrinsics()
            intr_list.append({'K': K.tolist(), 'D': D.tolist()})
            Ks.append(K)
        K0_full, K1_full = Ks[c0], Ks[c1]
        ow, oh = args.width, args.height

        # --- input geometry (built once; cameras are fixed) ---
        if args.input_mode == 'aspect':
            long_side = max(ow, oh) if str(args.input_long_side).lower() == 'native' \
                else int(args.input_long_side)
            geo = R.aspect_geometry(ow, oh, long_side, min_short=args.min_short_side)
        else:
            geo = R.letterbox_geometry(ow, oh, args.res)
        Win, Hin = geo["Win"], geo["Hin"]
        N_per_view = Win * Hin

        K0_in_norm = R.normalize_K(R.input_K(K0_full, geo), Win, Hin)
        K1_in_norm = R.normalize_K(R.input_K(K1_full, geo), Win, Hin)
        active_mask_np, native_pix_np = R.input_active_mask(geo)
        active_mask_np = R.erode_active_mask(active_mask_np, Hin, Win, args.edge_trim)

        intr_ctx = torch.tensor(np.stack([K0_in_norm, K1_in_norm])[None],
                                device=device, dtype=torch.float32)
        K0_render_t = torch.tensor(K0_full, device=device, dtype=torch.float32)
        active_t = torch.tensor(active_mask_np, device=device)
        cam1_native_pix = native_pix_np[active_mask_np]

        # --- model (download/load once) ---
        ckpt_path = args.checkpoint
        if ckpt_path is None:
            from huggingface_hub import hf_hub_download
            fname = R.CKPT_512 if args.res == 512 else R.CKPT_256
            print(f"Loading NoPoSplat checkpoint {fname} from HF...")
            ckpt_path = hf_hub_download(repo_id=R.HF_REPO, filename=fname)
        else:
            ckpt_path = os.path.expanduser(ckpt_path)
        encoder = R.load_noposplat_encoder(
            os.path.abspath(os.path.expanduser(args.noposplat_root)),
            ckpt_path, args.res, device)
        print("Model ready. Streaming -- press 'h' for help, 'q' to quit.\n")

        recorder = Recorder(args.out, args.task, args.episode, serials, args.fps,
                            (ow, oh), intr_list, [c0, c1], record_fps=args.record_fps)

        frames = Latest()       # newest (rgbs, seq, raw_idx) from the capture thread
        results = Latest()      # newest inference dict for the display thread
        shared = {'crop_on': not args.no_crop}   # toggled live by the main thread
        stop_event = threading.Event()

        # --- capture thread: newest frame + DENSE raw save at the capture rate ---
        def capture_loop():
            seq = 0
            while not stop_event.is_set():
                try:
                    rgbd = mc.grab_all_rgbd()
                except Exception as e:           # transient USB hiccup -> retry
                    print(f"[capture] {e}")
                    continue
                rgbs = [r for r, _ in rgbd]
                depths = [d for _, d in rgbd]
                seq += 1
                # dense raw save (no-op unless recording); returns this frame's index
                raw_idx = recorder.save_raw(rgbs, depths)
                frames.put((rgbs, seq, raw_idx))

        # --- inference thread: NoPoSplat on the latest frame + SPARSE augmentation ---
        def infer_loop():
            last_seq = -1
            infer_count = 0                      # nonce -> fresh viewpoints per frame
            while not stop_event.is_set():
                fr = frames.get()
                if fr is None:
                    time.sleep(0.005); continue
                rgbs, seq, raw_idx = fr
                if seq == last_seq:              # no new frame -> don't redo the GPU work
                    time.sleep(0.002); continue
                last_seq = seq
                t = time.time()
                try:
                    out = infer_pair(
                        R, encoder, rgbs[c0], rgbs[c1], geo, N_per_view, intr_ctx,
                        active_t, cam1_native_pix, K0_full, K1_full, K0_render_t,
                        ow, oh, args, shared['crop_on'], device, infer_count)
                except RuntimeError:             # PnP can fail on degenerate frames
                    continue
                infer_count += 1
                novels, view_Ks, novel_w2c, cam0_w2c, cam1_w2c, n_gauss = out
                results.put({'novels': novels, 'Ks': view_Ks, 'novel_w2c': novel_w2c,
                             'cam': (cam0_w2c, cam1_w2c), 'n_gauss': n_gauss,
                             'infer_dt': time.time() - t})
                # augment under the raw index this frame was saved with (throttled);
                # raw_idx is None when not recording, so nothing is written then
                if raw_idx is not None:
                    recorder.save_augmentation(raw_idx, rgbs, novels, view_Ks,
                                               novel_w2c, (cam0_w2c, cam1_w2c))

        cap_t = threading.Thread(target=capture_loop, daemon=True)
        inf_t = threading.Thread(target=infer_loop, daemon=True)
        cap_t.start()
        inf_t.start()

        win = 'live novel views (NoPoSplat)'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        help_on = False
        disp_fps = None
        t_prev = time.time()

        # --- main thread: display + keys at camera rate (never gated by inference) ---
        while True:
            now = time.time()
            disp_dt = max(now - t_prev, 1e-6)
            t_prev = now
            disp_fps = 1.0 / disp_dt if disp_fps is None else \
                0.9 * disp_fps + 0.1 * (1.0 / disp_dt)

            fr = frames.get()
            if fr is None:                       # cameras not streaming yet
                if (cv2.waitKey(30) & 0xFF) in (ord('q'), 27):
                    break
                continue
            rgbs, _, _ = fr
            res = results.get() or {}
            novels = res.get('novels', [])

            infer_dt = res.get('infer_dt')
            ifps = f'{1.0 / infer_dt:4.1f}' if infer_dt else '  - '
            state_col = {'recording': (40, 40, 240), 'paused': (40, 200, 240),
                         'idle': (180, 180, 180)}[recorder.state]
            rec_rate = f' @ {args.record_fps:g}fps' if args.record_fps else ''
            counts = f'raw {recorder.raw_idx} / aug {recorder.aug_count}{rec_rate}'
            rec_txt = {'recording': f'[REC]  {counts}',
                       'paused': f'[PAUSED]  {counts}',
                       'idle': 'idle (press s to record)'}[recorder.state]
            status = [
                (f'disp {disp_fps:4.1f} fps | infer {ifps} fps | '
                 f'{res.get("n_gauss", "-")} gaussians | '
                 f'crop {"ON" if shared["crop_on"] else "OFF"} | view_seed={args.view_seed}',
                 (200, 200, 200)),
                (rec_txt, state_col),
            ]
            canvas = build_canvas(rgbs[c0], rgbs[c1], novels, args.tile_w,
                                  args.grid_cols, status, help_on)
            cv2.imshow(win, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                recorder.start()
            elif key == ord('p'):
                recorder.pause()
            elif key == ord('x'):
                recorder.stop()
            elif key == ord('r'):
                args.view_seed += 1
                print(f"[view] shift viewpoint sequence -> view_seed={args.view_seed}")
            elif key == ord('c'):
                shared['crop_on'] = not shared['crop_on']
                print(f"[view] crop {'ON' if shared['crop_on'] else 'OFF'}")
            elif key == ord('h'):
                help_on = not help_on

        # --- shutdown: stop threads, then finalize any in-progress recording ---
        stop_event.set()
        inf_t.join(timeout=3.0)
        cap_t.join(timeout=3.0)
        if recorder.state != 'idle':
            recorder.stop()
    finally:
        mc.stop()
        cv2.destroyAllWindows()
        print("Stopped.")


def infer_pair(R, encoder, rgb0, rgb1, geo, N_per_view, intr_ctx, active_t,
               cam1_native_pix, K0_full, K1_full, K0_render_t, ow, oh, args,
               crop_on, device, view_nonce=0):
    """One NoPoSplat forward + Beta-sampled novel-view render for a live frame pair.
    Mirrors the per-frame body of render_novel_views.main(). The viewpoint RNG is
    seeded by (args.view_seed + view_nonce); the inference thread passes a per-frame
    nonce so the novel viewpoints VARY every frame (like the offline augmentation),
    for both the live display and the recorded data. It is deterministic given the
    nonce, and key 'r' shifts args.view_seed to jump to a different random sequence.
    The single render here feeds both display and recording, so they always match."""
    img0 = R.load_input_tensor(rgb0, geo, device)
    img1 = R.load_input_tensor(rgb1, geo, device)
    images = torch.stack([img0, img1])[None]               # (1,2,3,Hin,Win)
    with torch.no_grad():
        gauss = encoder({"image": images, "intrinsics": intr_ctx}, global_step=0)

    g0 = R.extract_per_view(gauss, 0, N_per_view)
    g1 = R.extract_per_view(gauss, 1, N_per_view)
    g0 = {k: v[active_t] for k, v in g0.items()}
    g1 = {k: v[active_t] for k, v in g1.items()}
    keep0 = R.gaussian_keep_mask(g0, args.max_depth_pct, args.max_scale_pct, args.max_anisotropy)
    keep1 = R.gaussian_keep_mask(g1, args.max_depth_pct, args.max_scale_pct, args.max_anisotropy)
    g0 = {k: v[keep0] for k, v in g0.items()}
    g1 = {k: v[keep1] for k, v in g1.items()}
    cam1_pix = cam1_native_pix[keep1.detach().cpu().numpy()]

    gaussians = {k: torch.cat([g0[k], g1[k]], dim=0) for k in g0}
    cam0_w2c = np.eye(4)
    obj = g1['means'].detach().cpu().numpy()
    cam1_w2c = R.solve_pnp_w2c(obj, cam1_pix, K1_full)      # may raise RuntimeError

    np.random.seed(args.view_seed + view_nonce)            # fresh viewpoints per frame
    novel_w2c = R.generate_novel_view_w2c(cam0_w2c, cam1_w2c, args.num_novel_views)

    novels, view_Ks = [], []
    for k in range(args.num_novel_views):
        w2c_t = torch.tensor(novel_w2c[k], device=device, dtype=torch.float32)
        rgb_t, alpha_t = R.render_view_alpha(gaussians, w2c_t, K0_render_t,
                                             oh, ow, antialias=not args.no_antialias)
        rgb = (rgb_t.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        if not crop_on:
            out_rgb, Kv = rgb, K0_full.astype(np.float64).copy()
        else:
            a = alpha_t.detach().cpu().numpy()
            l, t, r, b = R.compute_crop(a, args.alpha_thresh, args.crop_pad, args.max_crop_frac)
            cw, ch = r - l, b - t
            out_rgb = cv2.resize(rgb[t:b, l:r], (ow, oh), interpolation=cv2.INTER_LINEAR)
            Kv = R.crop_rescale_K(K0_full, l, t, cw, ch, ow, oh)
        novels.append(out_rgb)
        view_Ks.append(Kv)

    return novels, view_Ks, novel_w2c, cam0_w2c, cam1_w2c, gaussians['means'].shape[0]


if __name__ == '__main__':
    main()
