"""
Generate metric depth maps for an episode camera using Depth Anything V2 (indoor model).

Output: one .npy file per image saved next to it, e.g.
    cam0/000000.jpg  →  cam0_depth/000000.npy   (float32, metres)

Usage:
    python generate_depth.py --image_dir data/episodes/hammer/001/cam0
"""

import os, sys, glob, argparse
import numpy as np
import cv2
import torch
from tqdm import tqdm

DA2_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'Depth-Anything-V2', 'metric_depth')
CKPT    = os.path.join(DA2_DIR, 'checkpoints',
                       'depth_anything_v2_metric_hypersim_vitl.pth')

sys.path.insert(0, DA2_DIR)
from depth_anything_v2.dpt import DepthAnythingV2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--image_dir', required=True, help='Directory of RGB images')
    p.add_argument('--encoder',   default='vitl', choices=['vits', 'vitb', 'vitl'])
    p.add_argument('--max_depth', type=float, default=20.0,
                   help='Max depth in metres (20 for indoor)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main():
    args = parse_args()

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    print(f"Loading Depth Anything V2 ({args.encoder}, indoor) ...")
    model = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    model.load_state_dict(torch.load(CKPT, map_location='cpu'))
    model = model.to(args.device).eval()
    print("Model loaded.")

    images = sorted(glob.glob(os.path.join(args.image_dir, '*.jpg')) +
                    glob.glob(os.path.join(args.image_dir, '*.png')))
    if not images:
        raise RuntimeError(f"No images found in {args.image_dir}")

    out_dir = args.image_dir.rstrip('/') + '_depth'
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving depth maps to: {out_dir}")

    with torch.no_grad():
        for img_path in tqdm(images):
            img_bgr = cv2.imread(img_path)
            depth   = model.infer_image(img_bgr)  # float32 HxW in metres
            stem    = os.path.splitext(os.path.basename(img_path))[0]
            np.save(os.path.join(out_dir, f'{stem}.npy'), depth.astype(np.float32))

    print(f"Done. {len(images)} depth maps saved to {out_dir}/")


if __name__ == '__main__':
    main()
