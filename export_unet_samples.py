"""
Export the exact training and validation sample pairs used by train_unet_seg.py
to a browsable folder for manual inspection.

Each output image is a side-by-side panel:
    left  — original frame
    right — original with GT mask overlaid in green

Usage:
    python export_unet_samples.py --task_dir data/episodes/pastaTransfer \
                                  --output_dir data/unet_samples \
                                  [--n_train 2000] [--n_val 500]
"""

import os, sys, argparse, random
import numpy as np
import cv2

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)

from train_unet_seg import episode_split


def make_panel(orig_bgr, masked_bgr):
    """Side-by-side: original | GT mask overlay (green)."""
    orig_rgb   = cv2.cvtColor(orig_bgr,   cv2.COLOR_BGR2RGB)
    masked_rgb = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB)

    gt_mask = np.any(orig_rgb.astype(np.int16) != masked_rgb.astype(np.int16), axis=2)

    overlay = orig_rgb.copy()
    green = np.zeros_like(overlay)
    green[..., 1] = 255
    overlay[gt_mask] = (0.5 * green[gt_mask] + 0.5 * overlay[gt_mask]).astype(np.uint8)

    panel = np.concatenate([orig_rgb, overlay], axis=1)
    return cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)


def export_split(pairs, out_dir, label):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Exporting {len(pairs)} {label} samples → {out_dir}/")
    for i, (orig_path, masked_path) in enumerate(pairs):
        orig_bgr   = cv2.imread(orig_path)
        masked_bgr = cv2.imread(masked_path)
        if orig_bgr is None or masked_bgr is None:
            print(f"  WARNING: missing file at index {i}, skipping")
            continue
        panel = make_panel(orig_bgr, masked_bgr)
        # Filename encodes source so user can trace back to episode
        ep   = orig_path.split(os.sep)[-4]   # episode id
        src  = orig_path.split(os.sep)[-2]   # real or novel
        base = os.path.splitext(os.path.basename(orig_path))[0]
        fname = f"{i:05d}_{ep}_{src}_{base}.jpg"
        cv2.imwrite(os.path.join(out_dir, fname), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(pairs)}")
    print(f"  Done.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--task_dir',   required=True)
    p.add_argument('--output_dir', default=os.path.join(PIPELINE_DIR, 'data', 'unet_samples'))
    p.add_argument('--n_train',    type=int, default=2000)
    p.add_argument('--n_val',      type=int, default=500)
    return p.parse_args()


def main():
    args = parse_args()

    print("Collecting pairs (same split as train_unet_seg.py)...")
    train_pairs, val_pairs, eval_pairs, eval_eps = episode_split(args.task_dir)
    print(f"  Full split — train={len(train_pairs)}, val={len(val_pairs)}, eval={len(eval_pairs)}")
    print(f"  Eval episodes (not exported): {[os.path.basename(e) for e in eval_eps]}")

    # Same subsampling as training (seed=42)
    rng = random.Random(42)
    if args.n_train < len(train_pairs):
        train_pairs = rng.sample(train_pairs, args.n_train)
    if args.n_val < len(val_pairs):
        val_pairs = rng.sample(val_pairs, args.n_val)

    export_split(train_pairs, os.path.join(args.output_dir, 'train'), 'train')
    export_split(val_pairs,   os.path.join(args.output_dir, 'val'),   'val')

    print(f"\nDone. Browse samples in:")
    print(f"  {args.output_dir}/train/  ({args.n_train} images)")
    print(f"  {args.output_dir}/val/    ({args.n_val} images)")
    print(f"  Left half = original,  right half = green mask overlay")


if __name__ == '__main__':
    main()
