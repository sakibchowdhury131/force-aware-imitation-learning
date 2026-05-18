"""
Step 2 — Novel view synthesis via MASt3R + 3D Gaussian Splatting.

For each sampled timestep:
  1. Load the synchronized frame pair from the two fixed cameras.
  2. Run MASt3R reconstruction → camera poses + point cloud.
  3. Train 3D Gaussians on the two camera views.
  4. Render K novel views from Slerp-interpolated camera poses.

Usage:
    python 02_augment.py --episode_dir data/episodes/hammer/001

Output:
    data/episodes/<task>/<episode>/
        augmented/
            real/   ← copies of the original camera frames used
            novel/  ← novel view JPEG renders (frame_XXXXXX_novelYY.jpg)
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
    p.add_argument('--episode_dir', required=True,
                   help='Absolute path, e.g. /path/to/pipeline/data/episodes/hammer/001')
    p.add_argument('--num_novel_views', type=int, default=6)
    p.add_argument('--gs_iters_pruning',   type=int, default=500)
    p.add_argument('--gs_iters_fine',      type=int, default=100)
    p.add_argument('--sample_every',       type=int, default=15,
                   help='Process every Nth frame (15 = 2fps from 30fps)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--cam_pair', nargs=2, type=int, default=[0, 1],
                   help='Which two camera indices to use for reconstruction')
    return p.parse_args()


MAST3R_SIZE = 512  # matches original Tool-as-Interface mast3r_size


def load_frame_pair(episode_dir, frame_idx, cam_ids):
    """Load a synchronized frame pair from disk. Returns list of tensors + paths."""
    imgs, paths = [], []
    for cam_id in cam_ids:
        p = os.path.join(episode_dir, f'cam{cam_id}', f'{frame_idx:06d}.jpg')
        imgs.append(starster.load_image(p, size=MAST3R_SIZE))
        paths.append(p)
    return imgs, paths


def reconstruct_and_train(imgs, paths, device, gs_iters_pruning, gs_iters_fine):
    """
    Run MASt3R reconstruction + Gaussian training.
    Returns (gs_trainer, scene, W, H, tmpdir_obj).
    Caller must call tmpdir_obj.cleanup() after all scene access is done.
    """
    tmpdir_obj = tempfile.TemporaryDirectory()
    scene = starster.reconstruct_scene(None, imgs, paths, device,
                                       tmpdir=tmpdir_obj.name)

    H, W = scene.sparse_ga.imgs[0].shape[:2]

    gs = starster.GSTrainer(scene, device=device)
    gs.run_optimization(gs_iters_pruning, enable_pruning=True,  verbose=False)
    gs.run_optimization(gs_iters_fine,    enable_pruning=False, verbose=False)

    return gs, scene, W, H, tmpdir_obj


def render_novel_views(gs, scene, num_novel_views, device):
    """
    Generate novel view w2c matrices using the paper's Slerp + Beta method,
    render them, and return (renders, novel_w2c_np, K_novel_np, cam0_w2c_np).

    renders       — list of (H, W, 3) uint8 RGB arrays
    novel_w2c_np  — (N, 4, 4) float64 world-to-camera for each novel view
    K_novel_np    — (N, 3, 3) float64 intrinsics for each novel view
    cam0_w2c_np   — (4, 4) float64 world-to-camera for cam0
    """
    if num_novel_views % 2 != 0:
        raise ValueError("num_novel_views must be even (split equally between cam0/cam1 bias)")

    w2c = scene.w2c().to(device)   # (2, 4, 4)
    K   = scene.intrinsics().to(device)  # (2, 3, 3)

    novel_w2c = generate_novel_view_w2c(
        w2c[0], w2c[1], num_matrices=num_novel_views
    ).to(device)   # (N, 4, 4)

    # Randomly select cam0 or cam1 intrinsics per novel view (matches original)
    rand_idx  = torch.randint(0, 2, (num_novel_views,), device=device)
    K_novel   = K[rand_idx]   # (N, 3, 3)

    renders = []
    with torch.no_grad():
        for i in range(num_novel_views):
            out   = gs.render_views(novel_w2c[i:i+1], K_novel[i:i+1],
                                    MAST3R_SIZE, MAST3R_SIZE)
            img_t = out[0][0]   # (H, W, 3)
            img_np = (img_t.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            renders.append(img_np)

    return (renders,
            novel_w2c.cpu().double().numpy(),
            K_novel.cpu().double().numpy(),
            w2c[0].cpu().double().numpy(),
            w2c[1].cpu().double().numpy())


def main():
    args = parse_args()
    episode_dir = args.episode_dir

    with open(os.path.join(episode_dir, 'meta.json')) as f:
        meta = json.load(f)

    n_frames = meta['n_frames']
    cam_ids  = args.cam_pair

    aug_dir   = os.path.join(episode_dir, 'augmented')
    real_dir  = os.path.join(aug_dir, 'real')
    novel_dir = os.path.join(aug_dir, 'novel')
    os.makedirs(real_dir,  exist_ok=True)
    os.makedirs(novel_dir, exist_ok=True)

    frame_indices = list(range(0, n_frames, args.sample_every))
    print(f"Episode: {n_frames} frames  →  processing {len(frame_indices)} frames "
          f"(every {args.sample_every})")
    print(f"Camera pair: cam{cam_ids[0]} + cam{cam_ids[1]}")
    print(f"Novel views per frame: {args.num_novel_views}")

    # Load model once
    print("\nLoading MASt3R model...")
    model = starster.Mast3rModel.from_pretrained(MODEL_NAME).to(args.device).eval()

    total_t = time.time()
    novel_cams = {}   # will be saved to novel_cameras.npz for step 4

    for step, fi in enumerate(frame_indices):
        t0 = time.time()
        print(f"\n[{step+1}/{len(frame_indices)}] frame {fi:06d}")

        # Load frame pair
        imgs, paths = load_frame_pair(episode_dir, fi, cam_ids)

        # Copy real frames to augmented/real/
        for cam_id in cam_ids:
            src = os.path.join(episode_dir, f'cam{cam_id}', f'{fi:06d}.jpg')
            dst = os.path.join(real_dir, f'{fi:06d}_cam{cam_id}.jpg')
            import shutil; shutil.copy2(src, dst)

        # Reconstruct + train (model passed via closure)
        tmpdir_obj = tempfile.TemporaryDirectory()
        try:
            scene = starster.reconstruct_scene(model, imgs, paths, args.device,
                                               tmpdir=tmpdir_obj.name)
            gs    = starster.GSTrainer(scene, device=args.device)
            gs.run_optimization(args.gs_iters_pruning, enable_pruning=True,  verbose=False)
            gs.run_optimization(args.gs_iters_fine,    enable_pruning=False, verbose=False)

            renders, novel_w2c_np, K_novel_np, cam0_w2c_np, cam1_w2c_np = render_novel_views(
                gs, scene, args.num_novel_views, args.device)
        finally:
            tmpdir_obj.cleanup()

        # Save novel views
        for k, rgb in enumerate(renders):
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            out_path = os.path.join(novel_dir, f'{fi:06d}_novel{k:02d}.jpg')
            cv2.imwrite(out_path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Accumulate camera matrices for step 4 temporal tracking
        tag = f'{fi:06d}'
        novel_cams[f'{tag}_cam{cam_ids[0]}_w2c'] = cam0_w2c_np   # (4, 4)
        novel_cams[f'{tag}_cam{cam_ids[1]}_w2c'] = cam1_w2c_np   # (4, 4)
        novel_cams[f'{tag}_novel_w2c']            = novel_w2c_np  # (N, 4, 4)
        novel_cams[f'{tag}_novel_K']              = K_novel_np    # (N, 3, 3)

        elapsed = time.time() - t0
        done    = step + 1
        eta     = (time.time() - total_t) / done * (len(frame_indices) - done)
        print(f"  {args.num_novel_views} novel views  |  {elapsed:.1f}s/frame  |  ETA {eta:.0f}s")

    # Save per-frame novel-view camera matrices for step 4
    np.savez(os.path.join(aug_dir, 'novel_cameras.npz'), **novel_cams)
    print(f"\nSaved novel_cameras.npz ({len(frame_indices)} frames × 3 arrays)")

    total_novel = len(frame_indices) * args.num_novel_views
    print(f"\nDone in {time.time()-total_t:.0f}s")
    print(f"Output: {aug_dir}/")
    print(f"  real/   — {len(frame_indices) * len(cam_ids)} frames")
    print(f"  novel/  — {total_novel} novel views")
    print(f"Next: python 03_segment.py --episode_dir {episode_dir}")


if __name__ == '__main__':
    main()
