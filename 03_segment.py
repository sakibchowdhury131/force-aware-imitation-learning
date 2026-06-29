"""
Step 3 — Mask out human arms/hands using GroundedSAM2.

Runs on both the real camera frames and the novel view renders.
Masked pixels are set to black. Images are processed in batches
(GroundingDINO + SAM2 image encoder run on a whole batch at once)
for speed.

Usage (single episode):
    python 03_segment.py --episode_dir data/episodes/hammer/001

Usage (all episodes in a task):
    python 03_segment.py --task_dir data/episodes/hammer

Output (per episode):
    data/episodes/<task>/<episode>/
        augmented/
            masked_real/   ← real frames with hands/arms blacked out
            masked_novel/  ← novel views with hands/arms blacked out
"""

import os, sys, argparse
import numpy as np
import cv2
import torch
from PIL import Image

PIPELINE_DIR   = os.path.dirname(os.path.abspath(__file__))
GDINO_WEIGHTS  = os.path.join(PIPELINE_DIR, 'checkpoints', 'groundingdino_swint_ogc.pth')
SAM2_CHECKPOINT = os.path.join(PIPELINE_DIR, 'checkpoints', 'sam2.1_hiera_large.pt')
SAM2_MODEL_CFG  = 'configs/sam2.1/sam2.1_hiera_l.yaml'

# groundingdino is pip-installed (groundingdino-py); config ships with the package
import groundingdino
GDINO_CONFIG = os.path.join(os.path.dirname(groundingdino.__file__),
                             'config', 'GroundingDINO_SwinT_OGC.py')


def load_models(device):
    from groundingdino.util.inference import load_model as load_gdino
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    gdino = load_gdino(GDINO_CONFIG, GDINO_WEIGHTS)
    gdino = gdino.to(device).eval()

    sam2 = build_sam2(SAM2_MODEL_CFG, SAM2_CHECKPOINT, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2)

    return gdino, sam2_predictor


def preprocess_caption(caption: str) -> str:
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
    """Returns a list of binary masks (H, W), one per input image."""
    B = len(images_rgb)
    H, W = images_rgb[0].shape[:2]
    gd_t = gdino_transform()

    # --- GroundingDINO, batched forward pass ---
    gd_tensors = [gd_t(Image.fromarray(im), None)[0] for im in images_rgb]
    gd_batch = torch.stack(gd_tensors).to(device)
    caption = preprocess_caption(prompt)
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        outputs = gdino(gd_batch, captions=[caption] * B)
    logits_b = outputs['pred_logits'].cpu().sigmoid()  # (B, nq, 256)
    boxes_b  = outputs['pred_boxes'].cpu()             # (B, nq, 4)

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

    # --- SAM2, batched image encoder + batched box-prompted decoding ---
    sam2_predictor.set_image_batch(images_rgb)
    # Images with no detections get box=None (rather than an empty array),
    # since SAM2's mask decoder asserts image/token batch sizes match and
    # an empty box array yields a 0-sized token batch.
    box_batch = per_image_boxes
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        masks_batch, _, _ = sam2_predictor.predict_batch(
            box_batch=box_batch,
            multimask_output=False,
        )

    masks_out = []
    for i in range(B):
        if per_image_boxes[i] is None or len(masks_batch[i]) == 0:
            masks_out.append(np.zeros((H, W), dtype=bool))
            continue
        # masks_batch[i] shape: (N, 1, H, W) or (N, H, W)
        m = masks_batch[i]
        if m.ndim == 4:
            m = m.squeeze(1)
        masks_out.append(np.any(m > 0, axis=0))

    return masks_out


def apply_mask(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Black out masked regions."""
    out = img_rgb.copy()
    out[mask] = 0
    return out


def process_dir(src_dir, dst_dir, mask_fn, batch_size):
    """mask_fn(images_rgb) -> list of bool masks (H,W)"""
    os.makedirs(dst_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(src_dir) if f.endswith('.jpg'))
    print(f"  Processing {len(files)} images from {os.path.basename(src_dir)}/ "
          f"(batch size {batch_size})")

    for start in range(0, len(files), batch_size):
        batch_files = files[start:start + batch_size]
        images_rgb = [cv2.cvtColor(cv2.imread(os.path.join(src_dir, fname)),
                                   cv2.COLOR_BGR2RGB)
                      for fname in batch_files]

        masks = mask_fn(images_rgb)

        for fname, img_rgb, mask in zip(batch_files, images_rgb, masks):
            out_rgb = apply_mask(img_rgb, mask)
            cv2.imwrite(os.path.join(dst_dir, fname),
                        cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])


def process_episode(episode_dir, mask_fn, args):
    """Mask real (and optionally novel) frames for one episode. Returns True if work was done."""
    aug_dir   = os.path.join(episode_dir, 'augmented')
    real_dst  = os.path.join(aug_dir, 'masked_real')
    novel_dst = os.path.join(aug_dir, 'masked_novel')

    if args.cam_only:
        # Read directly from episode_dir/cam0/, cam1/, etc. — no step 2 needed.
        # Output: augmented/masked_real/XXXXXX_camN.jpg  (same naming as step 2 path)
        if args.skip_done and os.path.isdir(real_dst):
            print(f"  Skipping {episode_dir} — already done (masked_real exists)")
            return False
        import re, json
        with open(os.path.join(episode_dir, 'meta.json')) as f:
            meta = json.load(f)
        # intrinsics is a list indexed by camera position
        n_cams = len(meta.get('intrinsics', []))
        cam_indices = list(range(n_cams))
        if not cam_indices:
            print(f"  Skipping {episode_dir} — no cameras found in meta.json")
            return False
        os.makedirs(real_dst, exist_ok=True)
        for cam_i in cam_indices:
            cam_dir = os.path.join(episode_dir, f'cam{cam_i}')
            if not os.path.isdir(cam_dir):
                continue
            files = sorted(f for f in os.listdir(cam_dir) if f.endswith('.jpg'))
            print(f"  cam{cam_i}: {len(files)} frames → masked_real/")
            for start in range(0, len(files), args.batch_size):
                batch_files = files[start:start + args.batch_size]
                images_rgb = [cv2.cvtColor(cv2.imread(os.path.join(cam_dir, fn)),
                                           cv2.COLOR_BGR2RGB)
                              for fn in batch_files]
                masks = mask_fn(images_rgb)
                for fn, img_rgb, mask in zip(batch_files, images_rgb, masks):
                    fid = os.path.splitext(fn)[0]   # e.g. "000042"
                    out_name = f'{fid}_cam{cam_i}.jpg'
                    out_rgb = apply_mask(img_rgb, mask)
                    cv2.imwrite(os.path.join(real_dst, out_name),
                                cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
        return True

    # Default: read from augmented/real/ and augmented/novel/ (step 2 output)
    real_src  = os.path.join(aug_dir, 'real')
    novel_src = os.path.join(aug_dir, 'novel')

    if not os.path.isdir(real_src) or not os.path.isdir(novel_src):
        print(f"  Skipping {episode_dir} — augmented/real/ or novel/ not found (run step 2 first)")
        return False

    if args.skip_done and os.path.isdir(real_dst) and os.path.isdir(novel_dst):
        print(f"  Skipping {episode_dir} — already done")
        return False

    process_dir(real_src,  real_dst,  mask_fn, args.batch_size)
    process_dir(novel_src, novel_dst, mask_fn, args.batch_size)
    return True


def collect_episodes(task_dir):
    return [os.path.join(task_dir, name)
            for name in sorted(os.listdir(task_dir))
            if os.path.isdir(os.path.join(task_dir, name))
            and os.path.exists(os.path.join(task_dir, name, 'meta.json'))]


def parse_args():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--episode_dir', help='Single episode to process')
    src.add_argument('--task_dir',    help='Task directory; processes all episodes')
    p.add_argument('--prompt', default='human hand . human arm . person')
    p.add_argument('--box_threshold',  type=float, default=0.3)
    p.add_argument('--text_threshold', type=float, default=0.25)
    p.add_argument('--batch_size', type=int, default=8,
                   help='Number of images per GroundingDINO/SAM batch')
    p.add_argument('--skip_done', action='store_true',
                   help='Skip episodes that already have masked_real/ and masked_novel/')
    p.add_argument('--cam_only', action='store_true',
                   help='Read directly from episode cam0/,cam1/,... dirs (no step 2 needed). '
                        'Writes augmented/masked_real/XXXXXX_camN.jpg. '
                        'Skips novel view masking entirely.')
    p.add_argument('--device', default='cuda')
    p.add_argument('--unet_checkpoint', default=None,
                   help='Path to UNet checkpoint (train_unet_seg.py output). '
                        'If provided, uses UNet instead of GroundedSAM2 — much faster.')
    p.add_argument('--unet_threshold', type=float, default=0.5,
                   help='Sigmoid threshold for UNet predictions (default: 0.5)')
    return p.parse_args()


def main():
    args = parse_args()

    if args.episode_dir:
        episode_dirs = [args.episode_dir]
    else:
        episode_dirs = collect_episodes(args.task_dir)
        if not episode_dirs:
            print(f"No episodes found under {args.task_dir}")
            return
        print(f"Found {len(episode_dirs)} episode(s) under {args.task_dir}")

    # ── Build mask function ───────────────────────────────────────────────
    device = torch.device(args.device)

    if args.unet_checkpoint:
        import sys
        sys.path.insert(0, PIPELINE_DIR)
        from train_unet_seg import ResNetUNet, IMG_H, IMG_W, MEAN, STD
        import torch.nn.functional as F
        import torchvision.transforms.functional as TF
        from PIL import Image as PILImage

        print(f"Loading UNet from {args.unet_checkpoint} ...")
        model = ResNetUNet(pretrained=False).to(device)
        ckpt  = torch.load(args.unet_checkpoint, map_location=device)
        model.load_state_dict(ckpt['model'])
        model.eval()

        thresh = args.unet_threshold

        def mask_fn(images_rgb):
            tensors = [TF.normalize(
                           TF.to_tensor(PILImage.fromarray(im).resize((IMG_W, IMG_H), PILImage.BILINEAR)),
                           MEAN, STD)
                       for im in images_rgb]
            batch = torch.stack(tensors).to(device)
            with torch.no_grad():
                if device.type == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        logits = model(batch)
                else:
                    logits = model(batch)
            masks = []
            for logit, im in zip(logits, images_rgb):
                oh, ow = im.shape[:2]
                prob = F.interpolate(
                    logit.sigmoid().float().unsqueeze(0),
                    size=(oh, ow), mode='bilinear', align_corners=False
                ).squeeze().cpu().numpy()
                masks.append(prob > thresh)
            return masks

        print(f"  UNet mode  (threshold={thresh},  batch_size={args.batch_size})")

    else:
        print("Loading GroundingDINO + SAM2 (once for all episodes)...")
        gdino, sam2_predictor = load_models(args.device)

        def mask_fn(images_rgb):
            # Group by resolution (torch.stack requires identical sizes)
            groups = {}
            for idx, im in enumerate(images_rgb):
                groups.setdefault(im.shape[:2], []).append(idx)
            masks = [None] * len(images_rgb)
            for shape, idxs in groups.items():
                sub = [images_rgb[i] for i in idxs]
                sub_masks = get_masks_batch(gdino, sam2_predictor, sub,
                                            args.prompt, args.box_threshold,
                                            args.text_threshold, device)
                for i, m in zip(idxs, sub_masks):
                    masks[i] = m
            return masks

        print(f"  GroundedSAM2 mode  (prompt=\"{args.prompt}\")")

    # ── Process episodes ──────────────────────────────────────────────────
    n_done = n_skip = 0
    for i, episode_dir in enumerate(episode_dirs):
        ep_name = os.path.relpath(episode_dir, args.task_dir) if args.task_dir else episode_dir
        print(f"\n{'='*60}")
        print(f"Episode {i+1}/{len(episode_dirs)}: {ep_name}")
        print('='*60)
        ok = process_episode(episode_dir, mask_fn, args)
        if ok:
            n_done += 1
        else:
            n_skip += 1

    print(f"\nDone ({n_done} processed, {n_skip} skipped)")

    if args.task_dir:
        print(f"Next: python 04_track.py --task_dir {args.task_dir} --mesh <mesh.obj> --tool_prompt \"<prompt>\"")
    else:
        print(f"Next: python 04_track.py --episode_dir {args.episode_dir} --mesh <mesh.obj> --tool_prompt \"<prompt>\"")


if __name__ == '__main__':
    main()
