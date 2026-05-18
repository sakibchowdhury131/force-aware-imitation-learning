"""
Step 3 — Mask out human arms/hands using GroundedSAM.

Runs on both the real camera frames and the novel view renders.
Masked pixels are set to black (or optionally replaced with background).

Usage:
    python 03_segment.py --episode_dir data/episodes/hammer/001 \
        --prompt "human hand . human arm"

Output:
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--episode_dir', required=True)
    p.add_argument('--prompt', default='human hand . human arm . person')
    p.add_argument('--box_threshold',  type=float, default=0.3)
    p.add_argument('--text_threshold', type=float, default=0.25)
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def main():
    args = parse_args()
    aug_dir = os.path.join(args.episode_dir, 'augmented')

    print("Loading GroundingDINO + SAM...")
    gdino, sam_pred = load_models(args.device)

    process_dir(
        src_dir=os.path.join(aug_dir, 'real'),
        dst_dir=os.path.join(aug_dir, 'masked_real'),
        gdino=gdino, sam_pred=sam_pred,
        prompt=args.prompt,
        box_thresh=args.box_threshold,
        text_thresh=args.text_threshold,
        device=args.device,
    )

    process_dir(
        src_dir=os.path.join(aug_dir, 'novel'),
        dst_dir=os.path.join(aug_dir, 'masked_novel'),
        gdino=gdino, sam_pred=sam_pred,
        prompt=args.prompt,
        box_thresh=args.box_threshold,
        text_thresh=args.text_threshold,
        device=args.device,
    )

    print(f"\nMasked frames saved to {aug_dir}/masked_real/ and masked_novel/")
    print(f"Next: python 04_track.py --episode_dir {args.episode_dir}")


if __name__ == '__main__':
    main()
