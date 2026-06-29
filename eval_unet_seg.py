"""
Evaluate a trained UNet segmentation model on the held-out eval set.

Reports per-image and aggregate metrics (IoU, Dice, precision, recall) and
saves visual samples (input | GT mask overlay | predicted mask overlay) so
you can spot-check quality.

Usage:
    python eval_unet_seg.py \
        --checkpoint data/checkpoints/unet_seg/unet_best.pt \
        --task_dir   data/episodes/pastaTransfer \
        --output_dir /tmp/unet_eval

    # Or supply explicit eval pairs (saved by train_unet_seg.py)
    python eval_unet_seg.py \
        --checkpoint  data/checkpoints/unet_seg/unet_best.pt \
        --eval_pairs  data/checkpoints/unet_seg/eval_pairs.json \
        --output_dir  /tmp/unet_eval

Output:
    /tmp/unet_eval/
        metrics.json          — aggregate + per-image numbers
        samples/              — visual comparison images (up to --n_samples)
"""

import os, sys, json, argparse, random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)

from train_unet_seg import ResNetUNet, IMG_H, IMG_W, MEAN, STD, episode_split


def load_model(checkpoint_path, device):
    model = ResNetUNet(pretrained=False).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


def predict_mask(model, img_rgb, threshold, device):
    oh, ow = img_rgb.shape[:2]
    t = TF.normalize(
        TF.to_tensor(Image.fromarray(img_rgb).resize((IMG_W, IMG_H), Image.BILINEAR)),
        MEAN, STD
    ).unsqueeze(0).to(device)
    with torch.no_grad():
        if device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logit = model(t)
        else:
            logit = model(t)
    prob = logit.sigmoid().squeeze()
    prob_full = F.interpolate(
        prob.float().unsqueeze(0).unsqueeze(0),
        size=(oh, ow), mode='bilinear', align_corners=False
    ).squeeze().cpu().numpy()
    return prob_full > threshold, prob_full


def mask_metrics(pred, gt):
    """Returns dict of IoU, Dice, precision, recall for a single binary mask pair."""
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    union = tp + fp + fn
    iou  = float(tp) / float(union + 1e-6)
    dice = 2.0 * float(tp) / float(2 * tp + fp + fn + 1e-6)
    prec = float(tp) / float(tp + fp + 1e-6)
    rec  = float(tp) / float(tp + fn + 1e-6)
    return dict(iou=iou, dice=dice, precision=prec, recall=rec)


def overlay(img_rgb, mask, color=(255, 0, 0), alpha=0.5):
    out = img_rgb.copy()
    col = np.zeros_like(out)
    col[..., :] = color
    out[mask] = (alpha * col[mask] + (1 - alpha) * out[mask]).astype(np.uint8)
    return out


def make_sample_image(img_rgb, gt_mask, pred_mask):
    """Three-panel: input | GT overlay | prediction overlay."""
    gt_vis   = overlay(img_rgb, gt_mask,   color=(0, 255, 0))   # green = GT
    pred_vis = overlay(img_rgb, pred_mask, color=(255, 0, 0))   # red   = pred
    panel = np.concatenate([img_rgb, gt_vis, pred_vis], axis=1)
    return cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',  required=True)
    p.add_argument('--task_dir',    default=None,
                   help='Task dir (uses same episode split as training)')
    p.add_argument('--eval_pairs',  default=None,
                   help='eval_pairs.json saved by train_unet_seg.py (overrides --task_dir)')
    p.add_argument('--output_dir',  default='/tmp/unet_eval')
    p.add_argument('--threshold',   type=float, default=0.5)
    p.add_argument('--n_samples',   type=int,   default=30,
                   help='Number of visual sample images to save')
    p.add_argument('--device',      default='cuda')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    samples_dir = os.path.join(args.output_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)

    # ── Load eval pairs ───────────────────────────────────────────────────
    if args.eval_pairs:
        with open(args.eval_pairs) as f:
            data = json.load(f)
        eval_pairs = data['eval_pairs']
        eval_eps   = data.get('eval_episodes', [])
        print(f"Loaded {len(eval_pairs)} eval pairs from {args.eval_pairs}")
        if eval_eps:
            print(f"  Eval episodes: {[os.path.basename(e) for e in eval_eps]}")
    elif args.task_dir:
        _, _, eval_pairs, eval_eps = episode_split(args.task_dir)
        print(f"Episode split → {len(eval_pairs)} eval pairs")
        print(f"  Eval episodes: {[os.path.basename(e) for e in eval_eps]}")
    else:
        raise ValueError("Provide --task_dir or --eval_pairs")

    if not eval_pairs:
        raise RuntimeError("No eval pairs found.")

    # ── Load model ────────────────────────────────────────────────────────
    print(f"\nLoading UNet from {args.checkpoint}...")
    model = load_model(args.checkpoint, device)

    # ── Evaluate ──────────────────────────────────────────────────────────
    rng = random.Random(0)
    sample_indices = set(rng.sample(range(len(eval_pairs)), min(args.n_samples, len(eval_pairs))))

    all_metrics = []
    print(f"\nEvaluating {len(eval_pairs)} pairs...")
    for i, (orig_path, masked_path) in enumerate(eval_pairs):
        orig   = cv2.cvtColor(cv2.imread(orig_path),   cv2.COLOR_BGR2RGB)
        masked = cv2.cvtColor(cv2.imread(masked_path), cv2.COLOR_BGR2RGB)

        # Ground-truth mask from pixel difference
        gt_mask = np.any(orig.astype(np.int16) != masked.astype(np.int16), axis=2)

        pred_mask, pred_prob = predict_mask(model, orig, args.threshold, device)

        m = mask_metrics(pred_mask, gt_mask)
        m['path'] = orig_path
        m['gt_positive_frac'] = float(gt_mask.mean())
        all_metrics.append(m)

        if i in sample_indices:
            vis = make_sample_image(orig, gt_mask, pred_mask)
            fname = f"{i:05d}_{os.path.splitext(os.path.basename(orig_path))[0]}.jpg"
            cv2.imwrite(os.path.join(samples_dir, fname), vis,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(eval_pairs)}", flush=True)

    # ── Aggregate ─────────────────────────────────────────────────────────
    keys = ['iou', 'dice', 'precision', 'recall']
    agg  = {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}

    # Separate stats for frames with GT mask vs. empty frames
    nonempty = [m for m in all_metrics if m['gt_positive_frac'] > 0.001]
    empty    = [m for m in all_metrics if m['gt_positive_frac'] <= 0.001]

    agg_nonempty = {k: float(np.mean([m[k] for m in nonempty])) for k in keys} if nonempty else {}
    agg_empty_fp = float(np.mean([m['precision'] for m in empty])) if empty else None

    print(f"\n{'─'*50}")
    print(f"{'Metric':<12}  {'All':>8}  {'With arm':>10}")
    print(f"{'─'*50}")
    for k in keys:
        all_v  = agg[k]
        arm_v  = agg_nonempty.get(k, float('nan'))
        print(f"{k:<12}  {all_v:>8.4f}  {arm_v:>10.4f}")
    print(f"{'─'*50}")
    print(f"Total pairs: {len(all_metrics)}  "
          f"(with arm: {len(nonempty)},  empty: {len(empty)})")
    if agg_empty_fp is not None:
        print(f"False positive rate on empty frames: {1 - agg_empty_fp:.4f}")

    results = {
        'checkpoint': args.checkpoint,
        'threshold':  args.threshold,
        'n_pairs':    len(all_metrics),
        'n_with_arm': len(nonempty),
        'n_empty':    len(empty),
        'aggregate':  agg,
        'aggregate_with_arm': agg_nonempty,
        'per_image':  all_metrics,
    }
    out_json = os.path.join(args.output_dir, 'metrics.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nMetrics saved → {out_json}")
    print(f"Samples saved → {samples_dir}/  ({len(sample_indices)} images)")
    print(f"  (green = GT arm,  red = predicted arm)")


if __name__ == '__main__':
    main()
