"""
Step 3 — Mask out human arms/hands using GroundedSAM.

Runs on both the real camera frames and the novel view renders.
Masked pixels are set to black.

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

PIPELINE_DIR  = os.path.dirname(os.path.abspath(__file__))
GDINO_WEIGHTS = os.path.join(PIPELINE_DIR, 'checkpoints', 'groundingdino_swint_ogc.pth')
SAM_WEIGHTS   = os.path.join(PIPELINE_DIR, 'checkpoints', 'sam_vit_h_4b8939.pth')

# groundingdino is pip-installed (groundingdino-py); config ships with the package
import groundingdino
GDINO_CONFIG = os.path.join(os.path.dirname(groundingdino.__file__),
                             'config', 'GroundingDINO_SwinT_OGC.py')


def load_models(device):
    from groundingdino.util.inference import load_model as load_gdino
    from segment_anything import sam_model_registry, SamPredictor

    gdino = load_gdino(GDINO_CONFIG, GDINO_WEIGHTS)
    gdino = gdino.to(device).eval()

    sam   = sam_model_registry['vit_h'](checkpoint=SAM_WEIGHTS).to(device)
    predictor = SamPredictor(sam)

    return gdino, predictor


def get_masks(gdino, sam_predictor, img_rgb: np.ndarray, prompt: str,
              box_thresh=0.3, text_thresh=0.25, device='cuda'):
    """Returns binary mask (H, W) where True = region to mask out."""
    from groundingdino.util.inference import predict
    import torchvision.transforms as T

    H, W = img_rgb.shape[:2]

    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img_pil = Image.fromarray(img_rgb)
    img_t   = transform(img_pil).to(device)

    with torch.no_grad():
        boxes, logits, phrases = predict(
            model=gdino, image=img_t, caption=prompt,
            box_threshold=box_thresh, text_threshold=text_thresh,
            device=device,
        )

    if boxes is None or len(boxes) == 0:
        return np.zeros((H, W), dtype=bool)

    # Convert boxes from cx,cy,w,h normalised → xyxy pixel
    boxes_xyxy = boxes.clone()
    boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * W
    boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * H
    boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * W
    boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * H

    sam_predictor.set_image(img_rgb)
    masks_all = np.zeros((H, W), dtype=bool)

    for box in boxes_xyxy:
        masks, _, _ = sam_predictor.predict(
            box=box.cpu().numpy(),
            multimask_output=False,
        )
        masks_all |= masks[0].astype(bool)

    return masks_all


def apply_mask(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Black out masked regions."""
    out = img_rgb.copy()
    out[mask] = 0
    return out


def process_dir(src_dir, dst_dir, gdino, sam_pred, prompt, box_thresh, text_thresh, device):
    os.makedirs(dst_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(src_dir) if f.endswith('.jpg'))
    print(f"  Processing {len(files)} images from {os.path.basename(src_dir)}/")
    for fname in files:
        img_bgr = cv2.imread(os.path.join(src_dir, fname))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask    = get_masks(gdino, sam_pred, img_rgb, prompt,
                            box_thresh, text_thresh, device)
        out_rgb = apply_mask(img_rgb, mask)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(dst_dir, fname), out_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])


def process_episode(episode_dir, gdino, sam_pred, args):
    """Mask real and novel frames for one episode. Returns True if work was done."""
    aug_dir     = os.path.join(episode_dir, 'augmented')
    real_src    = os.path.join(aug_dir, 'real')
    novel_src   = os.path.join(aug_dir, 'novel')
    real_dst    = os.path.join(aug_dir, 'masked_real')
    novel_dst   = os.path.join(aug_dir, 'masked_novel')

    if not os.path.isdir(real_src) or not os.path.isdir(novel_src):
        print(f"  Skipping {episode_dir} — augmented/real/ or novel/ not found (run step 2 first)")
        return False

    if args.skip_done and os.path.isdir(real_dst) and os.path.isdir(novel_dst):
        print(f"  Skipping {episode_dir} — already done")
        return False

    process_dir(real_src,  real_dst,  gdino, sam_pred, args.prompt,
                args.box_threshold, args.text_threshold, args.device)
    process_dir(novel_src, novel_dst, gdino, sam_pred, args.prompt,
                args.box_threshold, args.text_threshold, args.device)
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
    p.add_argument('--skip_done', action='store_true',
                   help='Skip episodes that already have masked_real/ and masked_novel/')
    p.add_argument('--device', default='cuda')
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

    print("Loading GroundingDINO + SAM (once for all episodes)...")
    gdino, sam_pred = load_models(args.device)

    n_done = n_skip = 0
    for i, episode_dir in enumerate(episode_dirs):
        ep_name = os.path.relpath(episode_dir, args.task_dir) if args.task_dir else episode_dir
        print(f"\n{'='*60}")
        print(f"Episode {i+1}/{len(episode_dirs)}: {ep_name}")
        print('='*60)
        ok = process_episode(episode_dir, gdino, sam_pred, args)
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
