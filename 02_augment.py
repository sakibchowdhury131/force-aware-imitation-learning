"""
Step 2 — Novel view synthesis via MASt3R + 3D Gaussian Splatting.

For each sampled timestep:
  1. Load the synchronized frame pair from the two fixed cameras.
  2. Run MASt3R reconstruction → camera poses + point cloud.
  3. Train 3D Gaussians on the two camera views.
  4. Render K novel views from Slerp-interpolated camera poses.

Usage (single episode):
    python 02_augment.py --episode_dir data/episodes/hammer/001

Usage (all episodes in a task):
    python 02_augment.py --task_dir data/episodes/hammer

Output (per episode):
    data/episodes/<task>/<episode>/
        augmented/
            real/              ← copies of the sampled real frames
            novel/             ← novel view JPEG renders (XXXXXX_novelYY.jpg)
            novel_cameras.npz  ← per-frame camera matrices for step 4
"""

import os, sys, json, time, tempfile, argparse, shutil
import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import starster
from pipeline_utils.geometry import generate_novel_view_w2c


MODEL_NAME   = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--episode_dir',
                     help='Single episode, e.g. data/episodes/hammer/001')
    src.add_argument('--task_dir',
                     help='Task directory to process all episodes, e.g. data/episodes/hammer')
    p.add_argument('--num_novel_views', type=int, default=6)
    p.add_argument('--gs_iters_pruning',   type=int, default=500)
    p.add_argument('--gs_iters_fine',      type=int, default=100)
    p.add_argument('--sample_every',       type=int, default=15,
                   help='Process every Nth frame (15 = 2fps from 30fps)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--cam_pair', nargs=2, type=int, default=[0, 1],
                   help='Which two camera indices to use for reconstruction')
    p.add_argument('--skip_done', action='store_true',
                   help='Skip episodes that already have novel_cameras.npz')
    return p.parse_args()


MAST3R_SIZE = 512  # matches original Tool-as-Interface mast3r_size


def load_frame_pair(episode_dir, frame_idx, cam_ids):
    imgs, paths = [], []
    for cam_id in cam_ids:
        p = os.path.join(episode_dir, f'cam{cam_id}', f'{frame_idx:06d}.jpg')
        imgs.append(starster.load_image(p, size=MAST3R_SIZE))
        paths.append(p)
    return imgs, paths


def render_novel_views(gs, scene, num_novel_views, device):
    """
    Returns (renders, novel_w2c_np, K_novel_np, cam0_w2c_np, cam1_w2c_np).

    renders       — list of (H, W, 3) uint8 RGB arrays
    novel_w2c_np  — (N, 4, 4) float64 world-to-camera for each novel view
    K_novel_np    — (N, 3, 3) float64 intrinsics for each novel view
    cam0_w2c_np   — (4, 4) float64 world-to-camera for cam0
    cam1_w2c_np   — (4, 4) float64 world-to-camera for cam1
    """
    if num_novel_views % 2 != 0:
        raise ValueError("num_novel_views must be even (split equally between cam0/cam1 bias)")

    w2c = scene.w2c().to(device)          # (2, 4, 4)
    K   = scene.intrinsics().to(device)   # (2, 3, 3)

    novel_w2c = generate_novel_view_w2c(
        w2c[0], w2c[1], num_matrices=num_novel_views
    ).to(device)   # (N, 4, 4)

    rand_idx = torch.randint(0, 2, (num_novel_views,), device=device)
    K_novel  = K[rand_idx]   # (N, 3, 3)

    renders = []
    with torch.no_grad():
        for i in range(num_novel_views):
            out    = gs.render_views(novel_w2c[i:i+1], K_novel[i:i+1],
                                     MAST3R_SIZE, MAST3R_SIZE)
            img_np = (out[0][0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            renders.append(img_np)

    return (renders,
            novel_w2c.cpu().double().numpy(),
            K_novel.cpu().double().numpy(),
            w2c[0].cpu().double().numpy(),
            w2c[1].cpu().double().numpy())


def process_episode(episode_dir, model, args):
    """Process a single episode. model must already be loaded."""
    meta_path = os.path.join(episode_dir, 'meta.json')
    if not os.path.exists(meta_path):
        print(f"  Skipping {episode_dir} — no meta.json")
        return False

    with open(meta_path) as f:
        meta = json.load(f)

    cam_ids   = args.cam_pair
    aug_dir   = os.path.join(episode_dir, 'augmented')
    real_dir  = os.path.join(aug_dir, 'real')
    novel_dir = os.path.join(aug_dir, 'novel')

    if args.skip_done and os.path.exists(os.path.join(aug_dir, 'novel_cameras.npz')):
        print(f"  Skipping {episode_dir} — already done (novel_cameras.npz exists)")
        return False

    os.makedirs(real_dir,  exist_ok=True)
    os.makedirs(novel_dir, exist_ok=True)

    n_frames      = meta['n_frames']
    frame_indices = list(range(0, n_frames, args.sample_every))
    print(f"  {n_frames} frames  →  processing {len(frame_indices)} "
          f"(every {args.sample_every})  |  cam{cam_ids[0]}+cam{cam_ids[1]}  "
          f"|  {args.num_novel_views} novel views each")

    novel_cams = {}
    ep_t = time.time()

    for step, fi in enumerate(frame_indices):
        t0 = time.time()
        print(f"  [{step+1}/{len(frame_indices)}] frame {fi:06d}", end="", flush=True)

        imgs, paths = load_frame_pair(episode_dir, fi, cam_ids)

        for cam_id in cam_ids:
            src = os.path.join(episode_dir, f'cam{cam_id}', f'{fi:06d}.jpg')
            dst = os.path.join(real_dir, f'{fi:06d}_cam{cam_id}.jpg')
            shutil.copy2(src, dst)

        tmpdir_obj = tempfile.TemporaryDirectory()
        try:
            scene = starster.reconstruct_scene(model, imgs, paths, args.device,
                                               tmpdir=tmpdir_obj.name)
            gs = starster.GSTrainer(scene, device=args.device)
            gs.run_optimization(args.gs_iters_pruning, enable_pruning=True,  verbose=False)
            gs.run_optimization(args.gs_iters_fine,    enable_pruning=False, verbose=False)

            renders, novel_w2c_np, K_novel_np, cam0_w2c_np, cam1_w2c_np = render_novel_views(
                gs, scene, args.num_novel_views, args.device)
        finally:
            tmpdir_obj.cleanup()

        for k, rgb in enumerate(renders):
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(novel_dir, f'{fi:06d}_novel{k:02d}.jpg'),
                        bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        tag = f'{fi:06d}'
        novel_cams[f'{tag}_cam{cam_ids[0]}_w2c'] = cam0_w2c_np
        novel_cams[f'{tag}_cam{cam_ids[1]}_w2c'] = cam1_w2c_np
        novel_cams[f'{tag}_novel_w2c']            = novel_w2c_np
        novel_cams[f'{tag}_novel_K']              = K_novel_np

        elapsed = time.time() - t0
        done    = step + 1
        eta     = (time.time() - ep_t) / done * (len(frame_indices) - done)
        print(f"  {elapsed:.1f}s/frame  ETA {eta:.0f}s")

    np.savez(os.path.join(aug_dir, 'novel_cameras.npz'), **novel_cams)
    total_novel = len(frame_indices) * args.num_novel_views
    print(f"  Saved {total_novel} novel views + novel_cameras.npz  "
          f"({time.time()-ep_t:.0f}s total)")
    return True


def collect_episodes(task_dir):
    """Return sorted list of episode directories (subdirs containing meta.json)."""
    episodes = []
    for name in sorted(os.listdir(task_dir)):
        path = os.path.join(task_dir, name)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, 'meta.json')):
            episodes.append(path)
    return episodes


def main():
    args = parse_args()

    if args.episode_dir:
        episode_dirs = [args.episode_dir]
    else:
        episode_dirs = collect_episodes(args.task_dir)
        if not episode_dirs:
            print(f"No episodes found under {args.task_dir} (looking for subdirs with meta.json)")
            return
        print(f"Found {len(episode_dirs)} episode(s) under {args.task_dir}:")
        for ep in episode_dirs:
            print(f"  {os.path.basename(ep)}")

    print("\nLoading MASt3R model (once for all episodes)...")
    model = starster.Mast3rModel.from_pretrained(MODEL_NAME).to(args.device).eval()

    total_t  = time.time()
    n_done   = 0
    n_skip   = 0

    for i, episode_dir in enumerate(episode_dirs):
        ep_name = os.path.relpath(episode_dir, args.task_dir) if args.task_dir else episode_dir
        print(f"\n{'='*60}")
        print(f"Episode {i+1}/{len(episode_dirs)}: {ep_name}")
        print('='*60)
        ok = process_episode(episode_dir, model, args)
        if ok:
            n_done += 1
        else:
            n_skip += 1

    print(f"\nAll done in {time.time()-total_t:.0f}s  "
          f"({n_done} processed, {n_skip} skipped)")

    if args.task_dir:
        print(f"\nNext: run 03_segment.py and 04_track.py for each episode, or loop:")
        print(f"  for ep in {args.task_dir}/*/; do")
        print(f"      python 03_segment.py --episode_dir $ep")
        print(f"  done")
    else:
        print(f"Next: python 03_segment.py --episode_dir {args.episode_dir}")


if __name__ == '__main__':
    main()
