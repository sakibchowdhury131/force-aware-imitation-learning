"""
Run GroundedSAM2 on camera frames from the first N episodes of a task,
producing three review folders:

    <output_dir>/
        raw/    ← original JPEG frames  ({stem}.jpg)
        masks/  ← binary PNG masks      ({stem}.png)  white=hand, black=background
        vis/    ← side-by-side overlay  ({stem}_vis.jpg)  original | red mask overlay

Stem format:  {episode}_{cam}_{frame}   e.g.  001_cam0_000042

Workflow:
    1. Run this script.
    2. Open <output_dir>/vis/ in a file browser.
    3. DELETE any vis file where the mask looks wrong.
    4. Run clean_and_merge.py to remove the bad raw/mask pairs and build the
       combined training dataset (merging with existing UNet datasets).

Usage:
    python generate_hand_masks.py \\
        --task_dir data/episodes/pastaTransfer4 \\
        --n_episodes 5 \\
        --output_dir data/pastaTransfer4_hand_review
"""

import os, sys, argparse
import numpy as np
import cv2
import torch
from PIL import Image

PIPELINE_DIR    = os.path.dirname(os.path.abspath(__file__))
GDINO_WEIGHTS   = os.path.join(PIPELINE_DIR, 'checkpoints', 'groundingdino_swint_ogc.pth')
SAM2_CHECKPOINT = os.path.join(PIPELINE_DIR, 'checkpoints', 'sam2.1_hiera_large.pt')
SAM2_MODEL_CFG  = 'configs/sam2.1/sam2.1_hiera_l.yaml'

import groundingdino
GDINO_CONFIG = os.path.join(os.path.dirname(groundingdino.__file__),
                             'config', 'GroundingDINO_SwinT_OGC.py')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--task_dir',    required=True,
                   help='Task directory containing episode subdirs')
    p.add_argument('--n_episodes',  type=int, default=5,
                   help='Number of episodes to process (first N, default 5)')
    p.add_argument('--output_dir',  default=None,
                   help='Output directory for raw/masks/vis. '
                        'Default: <task_dir>/../<task_name>_hand_review')
    p.add_argument('--prompt',          default='human hand . human arm . person')
    p.add_argument('--box_threshold',   type=float, default=0.3)
    p.add_argument('--text_threshold',  type=float, default=0.25)
    p.add_argument('--batch_size',      type=int,   default=2)
    p.add_argument('--skip_done',       action='store_true',
                   help='Skip frames whose vis file already exists')
    p.add_argument('--device',          default='cuda')
    return p.parse_args()


# ── GroundedSAM2 ─────────────────────────────────────────────────────────────

def load_models(device):
    from groundingdino.util.inference import load_model as load_gdino
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    print("  Loading GroundingDINO...")
    gdino = load_gdino(GDINO_CONFIG, GDINO_WEIGHTS).to(device).eval()
    print("  Loading SAM2...")
    sam2  = build_sam2(SAM2_MODEL_CFG, SAM2_CHECKPOINT, device=device)
    return gdino, SAM2ImagePredictor(sam2)


def gdino_transform():
    import groundingdino.datasets.transforms as GT
    return GT.Compose([
        GT.RandomResize([800], max_size=1333),
        GT.ToTensor(),
        GT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_masks_batch(gdino, sam2_pred, images_rgb, prompt, box_thr, text_thr, device):
    B   = len(images_rgb)
    H, W = images_rgb[0].shape[:2]
    gd_t = gdino_transform()
    caption = prompt.lower().strip()
    if not caption.endswith('.'):
        caption += '.'

    gd_tensors = [gd_t(Image.fromarray(im), None)[0] for im in images_rgb]
    gd_batch   = torch.stack(gd_tensors).to(device)
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        outputs  = gdino(gd_batch, captions=[caption] * B)
    logits_b = outputs['pred_logits'].cpu().sigmoid()
    boxes_b  = outputs['pred_boxes'].cpu()

    per_image_boxes = []
    for i in range(B):
        keep = logits_b[i].max(dim=1)[0] > box_thr
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

    sam2_pred.set_image_batch(images_rgb)
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        masks_batch, _, _ = sam2_pred.predict_batch(
            box_batch=per_image_boxes, multimask_output=False)

    out = []
    for i in range(B):
        if per_image_boxes[i] is None or len(masks_batch[i]) == 0:
            out.append(np.zeros((H, W), dtype=bool))
            continue
        m = masks_batch[i]
        if m.ndim == 4:
            m = m.squeeze(1)
        out.append(np.any(m > 0, axis=0))
    return out


def make_vis(img_rgb, mask):
    overlay = img_rgb.copy()
    overlay[mask, 0] = np.clip(overlay[mask, 0].astype(np.int16) // 2 + 128, 0, 255).astype(np.uint8)
    overlay[mask, 1] = (overlay[mask, 1] // 2).astype(np.uint8)
    overlay[mask, 2] = (overlay[mask, 2] // 2).astype(np.uint8)
    panel = np.concatenate([img_rgb, overlay], axis=1)
    return cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device(args.device)

    task_name = os.path.basename(os.path.abspath(args.task_dir))
    if args.output_dir is None:
        args.output_dir = os.path.join(PIPELINE_DIR, 'data',
                                       f'{task_name}_hand_review')

    raw_dir  = os.path.join(args.output_dir, 'raw')
    mask_dir = os.path.join(args.output_dir, 'masks')
    vis_dir  = os.path.join(args.output_dir, 'vis')
    for d in (raw_dir, mask_dir, vis_dir):
        os.makedirs(d, exist_ok=True)

    # Collect first N episode dirs
    all_eps = sorted(
        os.path.join(args.task_dir, n)
        for n in os.listdir(args.task_dir)
        if os.path.isdir(os.path.join(args.task_dir, n))
        and os.path.exists(os.path.join(args.task_dir, n, 'meta.json'))
    )
    episodes = all_eps[:args.n_episodes]
    if not episodes:
        print(f"No episodes found in {args.task_dir}")
        return
    print(f"Processing {len(episodes)} episode(s): "
          f"{[os.path.basename(e) for e in episodes]}")

    # Build full frame list: (stem, path)
    frame_list = []
    for ep_dir in episodes:
        ep_name = os.path.basename(ep_dir)
        cam_dirs = sorted(
            d for d in os.listdir(ep_dir)
            if d.startswith('cam') and not d.endswith('_depth')
            and os.path.isdir(os.path.join(ep_dir, d))
        )
        for cam in cam_dirs:
            cam_path = os.path.join(ep_dir, cam)
            for fname in sorted(f for f in os.listdir(cam_path) if f.endswith('.jpg')):
                frame_id = os.path.splitext(fname)[0]
                stem     = f'{ep_name}_{cam}_{frame_id}'
                frame_list.append((stem, os.path.join(cam_path, fname)))

    if args.skip_done:
        existing = {os.path.splitext(f)[0].replace('_vis', '')
                    for f in os.listdir(vis_dir) if f.endswith('_vis.jpg')}
        before = len(frame_list)
        frame_list = [(s, p) for s, p in frame_list if s not in existing]
        print(f"Skipping {before - len(frame_list)} already-done frames.")

    total = len(frame_list)
    print(f"\n{total} frames to process across {len(episodes)} episodes.")
    print(f"Loading GroundedSAM2...")
    gdino, sam2_pred = load_models(device)
    print(f"  Prompt: \"{args.prompt}\"\n")

    done = 0
    for start in range(0, total, args.batch_size):
        batch = frame_list[start:start + args.batch_size]
        images_rgb = []
        for _, path in batch:
            bgr = cv2.imread(path)
            images_rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        masks = get_masks_batch(gdino, sam2_pred, images_rgb,
                                args.prompt, args.box_threshold,
                                args.text_threshold, device)

        for (stem, src_path), img_rgb, mask in zip(batch, images_rgb, masks):
            # raw
            cv2.imwrite(os.path.join(raw_dir, f'{stem}.jpg'),
                        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            # mask (binary 0/255 PNG)
            cv2.imwrite(os.path.join(mask_dir, f'{stem}.png'),
                        mask.astype(np.uint8) * 255)
            # visualization
            cv2.imwrite(os.path.join(vis_dir, f'{stem}_vis.jpg'),
                        make_vis(img_rgb, mask),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])

        done += len(batch)
        if done % 100 == 0 or done == total:
            print(f"  {done}/{total} frames processed", flush=True)

    print(f"\nDone. Output in {args.output_dir}/")
    print(f"  raw/   — {total} images")
    print(f"  masks/ — {total} masks")
    print(f"  vis/   — {total} visualizations")
    print(f"\nNext steps:")
    print(f"  1. Open {vis_dir}/ and DELETE any file where the mask looks wrong.")
    print(f"  2. Run:")
    print(f"       python clean_and_merge.py \\")
    print(f"           --review_dir {args.output_dir} \\")
    print(f"           --existing_dirs data/unet_hand_data data/unet_training_data \\")
    print(f"           --output_dir data/unet_combined \\")
    print(f"           --val_split 0.1")
    print(f"  3. Then train:")
    print(f"       python train_unet_seg.py \\")
    print(f"           --data_dir data/unet_combined \\")
    print(f"           --output_dir human_hand_segmentation_UNET \\")
    print(f"           --resume human_hand_segmentation_UNET/unet_best.pt")


if __name__ == '__main__':
    main()
