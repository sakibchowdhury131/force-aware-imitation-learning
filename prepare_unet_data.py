"""
Prepare UNet training data by running GroundedSAM2 on randomly selected
frames. Saves clean binary mask PNGs alongside the source images.

Samples from both real camera frames (augmented/real/) and novel views
(augmented/novel/), then splits into train/val sets.

Usage:
    python prepare_unet_data.py \
        --task_dir data/episodes/pastaTransfer2 \
        --output_dir data/unet_hand_data \
        --n_real 350 --n_novel 50 --val_split 0.1

Output:
    data/unet_hand_data/
        train/
            00000_img.jpg
            00000_mask.png    ← binary mask (white = hand)
            ...
        val/
            ...
        viz/
            train/  ← side-by-side: original | red mask overlay (for manual review)
            val/
        meta.json
"""

import os, sys, json, argparse, random
import numpy as np
import cv2
import torch
from PIL import Image

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)

GDINO_WEIGHTS   = os.path.join(PIPELINE_DIR, 'checkpoints', 'groundingdino_swint_ogc.pth')
SAM2_CHECKPOINT = os.path.join(PIPELINE_DIR, 'checkpoints', 'sam2.1_hiera_large.pt')
SAM2_MODEL_CFG  = 'configs/sam2.1/sam2.1_hiera_l.yaml'

import groundingdino
GDINO_CONFIG = os.path.join(os.path.dirname(groundingdino.__file__),
                             'config', 'GroundingDINO_SwinT_OGC.py')

EVAL_EPISODES = set()   # optionally hold out episodes from training


# ── GroundedSAM2 (inlined from 03_segment.py) ────────────────────────────

def load_models(device):
    from groundingdino.util.inference import load_model as load_gdino
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    gdino = load_gdino(GDINO_CONFIG, GDINO_WEIGHTS)
    gdino = gdino.to(device).eval()
    sam2  = build_sam2(SAM2_MODEL_CFG, SAM2_CHECKPOINT, device=device)
    return gdino, SAM2ImagePredictor(sam2)


def preprocess_caption(caption):
    result = caption.lower().strip()
    return result if result.endswith('.') else result + '.'


def gdino_transform():
    import groundingdino.datasets.transforms as GT
    return GT.Compose([
        GT.RandomResize([800], max_size=1333),
        GT.ToTensor(),
        GT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_masks_batch(gdino, sam2_predictor, images_rgb, prompt,
                    box_thresh, text_thresh, device):
    B = len(images_rgb)
    H, W = images_rgb[0].shape[:2]
    gd_t = gdino_transform()

    gd_tensors = [gd_t(Image.fromarray(im), None)[0] for im in images_rgb]
    gd_batch   = torch.stack(gd_tensors).to(device)
    caption    = preprocess_caption(prompt)

    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        outputs = gdino(gd_batch, captions=[caption] * B)
    logits_b = outputs['pred_logits'].cpu().sigmoid()
    boxes_b  = outputs['pred_boxes'].cpu()

    per_image_boxes = []
    for i in range(B):
        keep = logits_b[i].max(dim=1)[0] > box_thresh
        boxes = boxes_b[i][keep]
        if len(boxes) == 0:
            per_image_boxes.append(None)
            continue
        xyxy = boxes.clone()
        xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * W
        xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * H
        xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * W
        xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * H
        per_image_boxes.append(xyxy.numpy())

    sam2_predictor.set_image_batch(images_rgb)
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        masks_batch, _, _ = sam2_predictor.predict_batch(
            box_batch=per_image_boxes, multimask_output=False)

    masks_out = []
    for i in range(B):
        if per_image_boxes[i] is None or len(masks_batch[i]) == 0:
            masks_out.append(np.zeros((H, W), dtype=bool))
            continue
        m = masks_batch[i]
        if m.ndim == 4:
            m = m.squeeze(1)
        masks_out.append(np.any(m > 0, axis=0))

    return masks_out


# ── Data collection ───────────────────────────────────────────────────────

def _episode_dirs(task_dir):
    return sorted(
        os.path.join(task_dir, n)
        for n in os.listdir(task_dir)
        if os.path.isdir(os.path.join(task_dir, n))
        and os.path.exists(os.path.join(task_dir, n, 'meta.json'))
        and n not in EVAL_EPISODES
    )


def collect_real_paths(task_dir):
    """Collect all real camera frame paths (augmented/real/) from all episodes."""
    paths = []
    for ep in _episode_dirs(task_dir):
        real_dir = os.path.join(ep, 'augmented', 'real')
        if not os.path.isdir(real_dir):
            continue
        for fname in sorted(os.listdir(real_dir)):
            if fname.endswith('.jpg'):
                paths.append(os.path.join(real_dir, fname))
    return paths


def collect_novel_paths(task_dir):
    """Collect all novel view paths (augmented/novel/) from all episodes."""
    paths = []
    for ep in _episode_dirs(task_dir):
        novel_dir = os.path.join(ep, 'augmented', 'novel')
        if not os.path.isdir(novel_dir):
            continue
        for fname in sorted(os.listdir(novel_dir)):
            if fname.endswith('.jpg'):
                paths.append(os.path.join(novel_dir, fname))
    return paths


# ── Visualization ─────────────────────────────────────────────────────────

def make_viz(img_rgb, mask):
    overlay = img_rgb.copy()
    red = np.zeros_like(overlay)
    red[..., 0] = 255
    overlay[mask] = (0.5 * red[mask] + 0.5 * overlay[mask]).astype(np.uint8)
    panel = np.concatenate([img_rgb, overlay], axis=1)
    return cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)


# ── Main ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--task_dir',   required=True)
    p.add_argument('--output_dir', default=os.path.join(PIPELINE_DIR, 'data', 'unet_hand_data'))
    p.add_argument('--n_real',     type=int, default=350,
                   help='Number of real camera frames to sample (augmented/real/)')
    p.add_argument('--n_novel',    type=int, default=50,
                   help='Number of novel view frames to sample (augmented/novel/)')
    p.add_argument('--val_split',  type=float, default=0.1,
                   help='Fraction of total samples held out for validation (default 0.1 = 10%%)')
    p.add_argument('--prompt',     default='human hand . human arm . person')
    p.add_argument('--box_threshold',  type=float, default=0.3)
    p.add_argument('--text_threshold', type=float, default=0.25)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--device',     default='cuda')
    return p.parse_args()


def process_split(paths, split_dir, viz_dir, gdino, sam2_predictor, args, device, global_offset=0):
    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(viz_dir,   exist_ok=True)
    n = len(paths)
    print(f"  Processing {n} images...")

    for start in range(0, n, args.batch_size):
        batch_paths = paths[start:start + args.batch_size]
        images_rgb  = []
        for p in batch_paths:
            bgr = cv2.imread(p)
            images_rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        masks = get_masks_batch(gdino, sam2_predictor, images_rgb,
                                args.prompt, args.box_threshold,
                                args.text_threshold, device)

        for j, (src_path, img_rgb, mask) in enumerate(zip(batch_paths, images_rgb, masks)):
            idx  = global_offset + start + j
            # encode source type (real/novel) and episode in filename for traceability
            parts = src_path.split(os.sep)
            ep    = parts[-4]
            src   = parts[-2]   # 'real' or 'novel'
            base  = os.path.splitext(os.path.basename(src_path))[0]
            stem  = f"{idx:05d}_{ep}_{src}_{base}"

            cv2.imwrite(os.path.join(split_dir, f"{stem}_img.jpg"),
                        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(os.path.join(split_dir, f"{stem}_mask.png"),
                        (mask.astype(np.uint8) * 255))
            cv2.imwrite(os.path.join(viz_dir, f"{stem}.jpg"),
                        make_viz(img_rgb, mask),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])

        done = min(start + args.batch_size, n)
        if done % 50 == 0 or done == n:
            print(f"    {done}/{n}", flush=True)


def main():
    args   = parse_args()
    device = torch.device(args.device)
    rng    = random.Random(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Collect source paths ──────────────────────────────────────────────
    print("Collecting real camera frame paths...")
    real_paths = collect_real_paths(args.task_dir)
    print(f"  Found {len(real_paths)} real frames")

    print("Collecting novel view paths...")
    novel_paths = collect_novel_paths(args.task_dir)
    print(f"  Found {len(novel_paths)} novel view frames")

    if len(real_paths) < args.n_real:
        raise RuntimeError(f"Only {len(real_paths)} real frames available, need {args.n_real}")
    if len(novel_paths) < args.n_novel:
        raise RuntimeError(f"Only {len(novel_paths)} novel frames available, need {args.n_novel}")

    sampled_real  = rng.sample(real_paths,  args.n_real)
    sampled_novel = rng.sample(novel_paths, args.n_novel)
    all_selected  = sampled_real + sampled_novel
    rng.shuffle(all_selected)

    total   = len(all_selected)
    n_val   = max(1, int(round(total * args.val_split)))
    n_train = total - n_val
    train_paths = all_selected[:n_train]
    val_paths   = all_selected[n_train:]

    print(f"\nSampled {total} images total  ({args.n_real} real + {args.n_novel} novel)")
    print(f"  Train: {n_train}  |  Val: {n_val}  (split={args.val_split})")

    # ── Run GroundedSAM2 ──────────────────────────────────────────────────
    print(f"\nLoading GroundedSAM2...")
    gdino, sam2_predictor = load_models(device)
    print(f"  Prompt: \"{args.prompt}\"")

    print(f"\n── Train ({n_train} images) ──")
    process_split(train_paths,
                  os.path.join(args.output_dir, 'train'),
                  os.path.join(args.output_dir, 'viz', 'train'),
                  gdino, sam2_predictor, args, device,
                  global_offset=0)

    print(f"\n── Val ({n_val} images) ──")
    process_split(val_paths,
                  os.path.join(args.output_dir, 'val'),
                  os.path.join(args.output_dir, 'viz', 'val'),
                  gdino, sam2_predictor, args, device,
                  global_offset=n_train)

    meta = {
        'task_dir':        args.task_dir,
        'prompt':          args.prompt,
        'box_threshold':   args.box_threshold,
        'text_threshold':  args.text_threshold,
        'n_real':          args.n_real,
        'n_novel':         args.n_novel,
        'val_split':       args.val_split,
        'n_train':         n_train,
        'n_val':           n_val,
        'seed':            args.seed,
        'train_sources':   train_paths,
        'val_sources':     val_paths,
    }
    with open(os.path.join(args.output_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone.")
    print(f"  Images + masks : {args.output_dir}/train/   {args.output_dir}/val/")
    print(f"  Visualizations : {args.output_dir}/viz/train/   {args.output_dir}/viz/val/")
    print(f"  (left = original, right = red mask overlay — delete bad viz files then retrain)")
    print(f"\nNext step — after reviewing viz/ and removing bad samples:")
    print(f"  python train_unet_seg.py --data_dir {args.output_dir} --output_dir data/checkpoints/unet_hand")


if __name__ == '__main__':
    main()
