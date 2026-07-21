#!/usr/bin/env python3
"""
Run the trained UNet on all 342 images and produce a side-by-side video:
  left  = original image
  right = original + colour-coded segmentation mask overlay

Output: /home/sakib/unet_seg/training/results/segmentation_demo.mp4
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# ── Config ────────────────────────────────────────────────────────────────────

ROOT        = Path("/home/sakib/unet_seg")
CKPT        = ROOT / "training/checkpoints/best_model.pth"
OUT_VIDEO   = ROOT / "training/results/segmentation_demo.mp4"

IMG_FOLDERS = [
    ROOT / "001/cam0",
    ROOT / "001/cam1",
    ROOT / "002/cam0",
    ROOT / "002/cam1",
    ROOT / "003/jaco2_web",
]

MODEL_H, MODEL_W = 288, 512      # inference resolution
VIDEO_W, VIDEO_H = 1280, 360    # output frame: two 640×360 panels side by side
FPS             = 6              # slow enough to read individual frames
MASK_COLOR      = (0, 200, 255)  # BGR: amber/orange overlay
MASK_ALPHA      = 0.45           # opacity of mask overlay

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model():
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=3, classes=1).to(DEVICE)
    ckpt = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint — best val IoU: {ckpt['val_iou']:.4f}  "
          f"Dice: {ckpt['val_dice']:.4f}  (epoch {ckpt['epoch']})")
    return model


transform = Compose([
    Resize(MODEL_H, MODEL_W),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


@torch.no_grad()
def predict_mask(model, bgr_img):
    """Return a float32 probability mask at model resolution."""
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    aug = transform(image=rgb)
    x   = aug["image"].float().unsqueeze(0).to(DEVICE)
    prob = torch.sigmoid(model(x)).squeeze().cpu().numpy()  # [H, W] 0-1
    return prob


def make_frame(orig_bgr, prob_mask):
    """
    Build one 1280×360 frame:
      left  = resized original
      right = original + coloured mask overlay + contour
    """
    panel_w, panel_h = VIDEO_W // 2, VIDEO_H

    # Resize original to panel size
    orig = cv2.resize(orig_bgr, (panel_w, panel_h))

    # Upscale mask to panel size
    mask_full = cv2.resize(prob_mask, (panel_w, panel_h))

    # Compute IoU-style confidence for this image
    binary = (mask_full > 0.5).astype(np.uint8)
    coverage = binary.mean() * 100           # % of frame covered

    # Colour overlay
    overlay = orig.copy()
    colour_layer = np.full_like(orig, MASK_COLOR, dtype=np.uint8)
    blended = cv2.addWeighted(colour_layer, MASK_ALPHA, orig, 1 - MASK_ALPHA, 0)
    overlay[binary == 1] = blended[binary == 1]

    # Draw contours for crispness
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 100), 2)

    # Confidence heatmap strip (thin bar at bottom)
    heat = cv2.applyColorMap(
        (mask_full * 255).astype(np.uint8), cv2.COLORMAP_JET)
    strip_h = 8
    heat_strip = cv2.resize(heat, (panel_w, strip_h))
    overlay[-strip_h:] = heat_strip

    # Labels
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    cv2.putText(orig,    "Original",            (10, 25), font, fs, (255,255,255), th+1)
    cv2.putText(orig,    "Original",            (10, 25), font, fs, (20,20,20),   th)
    cv2.putText(overlay, "Segmentation",        (10, 25), font, fs, (255,255,255), th+1)
    cv2.putText(overlay, "Segmentation",        (10, 25), font, fs, (20,20,20),   th)
    cv2.putText(overlay, f"arm: {coverage:.1f}%", (10, 48), font, fs, (0,220,255), th)

    return np.concatenate([orig, overlay], axis=1)


def label_frame(frame, folder_name, img_name, idx, total):
    """Add a thin header bar with metadata."""
    bar = np.zeros((28, frame.shape[1], 3), dtype=np.uint8)
    bar[:] = (40, 40, 40)
    text = f"[{idx:3d}/{total}]  {folder_name}/{img_name}"
    cv2.putText(bar, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return np.vstack([bar, frame])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    model = load_model()

    # Collect & sort all images
    image_paths = []
    for folder in IMG_FOLDERS:
        if folder.exists():
            imgs = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.jpeg")) \
                 + sorted(folder.glob("*.png"))
            image_paths.extend(imgs)

    total = len(image_paths)
    print(f"Processing {total} images → {OUT_VIDEO}")

    frame_h = VIDEO_H + 28   # +28 for header bar
    writer = cv2.VideoWriter(
        str(OUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (VIDEO_W, frame_h),
    )
    if not writer.isOpened():
        sys.exit("VideoWriter failed to open. Check FFmpeg support.")

    for i, img_path in enumerate(image_paths, 1):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"  Skip (unreadable): {img_path.name}")
            continue

        prob = predict_mask(model, bgr)
        frame = make_frame(bgr, prob)
        frame = label_frame(frame, img_path.parent.name, img_path.name, i, total)
        writer.write(frame)

        if i % 50 == 0 or i == total:
            print(f"  {i}/{total} frames written")

    writer.release()
    size_mb = OUT_VIDEO.stat().st_size / 1e6
    print(f"\nDone!  {OUT_VIDEO}  ({size_mb:.1f} MB, {total} frames @ {FPS} fps)")
    print(f"Duration: ~{total/FPS:.0f}s  |  Play with: vlc {OUT_VIDEO}")


if __name__ == "__main__":
    main()
