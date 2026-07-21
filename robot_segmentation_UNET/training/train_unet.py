#!/usr/bin/env python3
"""
UNet training script for robotic arm segmentation.
Reads Label Studio polygon annotations, converts to binary masks, trains UNet.

Usage:
    python3 train_unet.py
"""

import json
import os
import random
import sys
import time
from pathlib import Path

import albumentations as A
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

import segmentation_models_pytorch as smp

# ── Configuration ─────────────────────────────────────────────────────────────

EXPORT_JSON   = Path("/home/sakib/unet_seg/annotations_export.json")
IMAGE_ROOT    = Path("/home/sakib")          # document root (same as LS setting)
MASK_DIR      = Path("/home/sakib/unet_seg/training/masks")
CKPT_DIR      = Path("/home/sakib/unet_seg/training/checkpoints")
RESULTS_DIR   = Path("/home/sakib/unet_seg/training/results")

IMG_H, IMG_W  = 288, 512      # resize target (same aspect as 480×848, div by 32)
BATCH_SIZE    = 8
NUM_EPOCHS    = 80
LR            = 3e-4
VAL_SPLIT     = 10             # number of images held out for validation
SEED          = 42
ENCODER       = "resnet34"
ENCODER_WEIGHTS = "imagenet"  # pretrained encoder weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ── Step 1: Parse annotations & build masks ───────────────────────────────────

def ls_path_to_local(ls_path: str) -> Path:
    """Convert /data/local-files/?d=unet_seg/... to /home/sakib/unet_seg/..."""
    if "?d=" in ls_path:
        rel = ls_path.split("?d=", 1)[1]
        return IMAGE_ROOT / rel
    return IMAGE_ROOT / ls_path.lstrip("/")


def polygon_to_mask(points_pct, orig_w, orig_h, out_w=None, out_h=None):
    """
    Convert Label Studio polygon (percentage coords) to a binary mask.
    points_pct: list of [x_pct, y_pct]
    Returns uint8 mask (0/255) at (out_h, out_w) or (orig_h, orig_w).
    """
    out_w = out_w or orig_w
    out_h = out_h or orig_h
    pts = np.array([[p[0] / 100 * orig_w, p[1] / 100 * orig_h] for p in points_pct],
                   dtype=np.float32)
    mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
    if (out_w, out_h) != (orig_w, orig_h):
        mask = cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return mask


def prepare_masks(export_json: Path, mask_dir: Path):
    """Parse export JSON and save binary mask PNGs. Returns list of (img_path, mask_path)."""
    mask_dir.mkdir(parents=True, exist_ok=True)
    with open(export_json) as f:
        tasks = json.load(f)

    pairs = []
    skipped = 0
    for task in tasks:
        img_path = ls_path_to_local(task["data"]["image"])
        if not img_path.exists():
            skipped += 1
            continue

        results = []
        for ann in task.get("annotations", []):
            for r in ann.get("result", []):
                if r["type"] == "polygonlabels" and "points" in r["value"]:
                    results.append(r)

        if not results:
            skipped += 1
            continue

        # Use original image dimensions from annotation metadata
        orig_w = results[0]["original_width"]
        orig_h = results[0]["original_height"]

        # Merge all polygon regions into one binary mask
        combined = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for r in results:
            m = polygon_to_mask(r["value"]["points"], orig_w, orig_h)
            combined = np.maximum(combined, m)

        mask_name = img_path.stem + "__" + img_path.parent.name + ".png"
        mask_path = mask_dir / mask_name
        cv2.imwrite(str(mask_path), combined)
        pairs.append((img_path, mask_path))

    print(f"Prepared {len(pairs)} image-mask pairs (skipped {skipped})")
    return pairs


# ── Step 2: Dataset ───────────────────────────────────────────────────────────

def make_transforms(is_train: bool):
    if is_train:
        return A.Compose([
            A.Resize(IMG_H, IMG_W),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.6),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(IMG_H, IMG_W),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


class ArmSegDataset(Dataset):
    def __init__(self, pairs, transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask  = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        aug   = self.transform(image=image, mask=mask)
        img_t = aug["image"].float()
        msk_t = (aug["mask"] > 127).float().unsqueeze(0)  # [1, H, W]
        return img_t, msk_t


# ── Step 3: Loss ──────────────────────────────────────────────────────────────

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        smooth = 1e-5
        inter = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = 1 - (2 * inter + smooth) / (union + smooth)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice.mean()


# ── Step 4: Metrics ───────────────────────────────────────────────────────────

def compute_iou(logits, targets, threshold=0.5):
    preds = (torch.sigmoid(logits) > threshold).float()
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - inter
    iou = (inter + 1e-5) / (union + 1e-5)
    return iou.mean().item()


def compute_dice(logits, targets, threshold=0.5):
    preds = (torch.sigmoid(logits) > threshold).float()
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * inter + 1e-5) / (union + 1e-5)
    return dice.mean().item()


# ── Step 5: Training loop ─────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(imgs)
            loss   = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss, total_iou, total_dice = 0.0, 0.0, 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        logits = model(imgs)
        total_loss += criterion(logits, masks).item()
        total_iou  += compute_iou(logits, masks)
        total_dice += compute_dice(logits, masks)
    n = len(loader)
    return total_loss / n, total_iou / n, total_dice / n


# ── Step 6: Visualize predictions ─────────────────────────────────────────────

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

@torch.no_grad()
def save_val_predictions(model, val_pairs, epoch, results_dir):
    model.eval()
    transform = make_transforms(is_train=False)
    fig, axes = plt.subplots(len(val_pairs), 3, figsize=(12, 4 * len(val_pairs)))
    if len(val_pairs) == 1:
        axes = [axes]

    for i, (img_path, mask_path) in enumerate(val_pairs):
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask  = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        aug   = transform(image=image, mask=mask)
        img_t = aug["image"].float().unsqueeze(0).to(DEVICE)
        gt    = (aug["mask"] > 127).cpu().numpy()

        logit = model(img_t)
        pred  = (torch.sigmoid(logit) > 0.5).squeeze().cpu().numpy()

        # denormalize for display
        disp = aug["image"].permute(1, 2, 0).cpu().numpy()
        disp = (disp * IMAGENET_STD + IMAGENET_MEAN).clip(0, 1)

        axes[i][0].imshow(disp);            axes[i][0].set_title("Image");      axes[i][0].axis("off")
        axes[i][1].imshow(gt, cmap="gray"); axes[i][1].set_title("GT Mask");    axes[i][1].axis("off")
        axes[i][2].imshow(pred, cmap="gray"); axes[i][2].set_title("Predicted"); axes[i][2].axis("off")

    plt.suptitle(f"Epoch {epoch} — Validation Predictions", fontsize=14)
    plt.tight_layout()
    out = results_dir / f"val_predictions_epoch{epoch:03d}.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Saved predictions → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 1. Prepare masks
    print("\n── Preparing masks ──")
    pairs = prepare_masks(EXPORT_JSON, MASK_DIR)

    # 2. Train / val split
    random.shuffle(pairs)
    val_pairs   = pairs[:VAL_SPLIT]
    train_pairs = pairs[VAL_SPLIT:]
    print(f"Train: {len(train_pairs)}  |  Val: {len(val_pairs)}")

    # 3. Datasets & loaders
    train_ds = ArmSegDataset(train_pairs, make_transforms(is_train=True))
    val_ds   = ArmSegDataset(val_pairs,   make_transforms(is_train=False))
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)

    # 4. Model
    print(f"\n── Building UNet ({ENCODER}, pretrained={ENCODER_WEIGHTS}) ──")
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=1,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {total_params:.1f}M")

    # 5. Loss, optimizer, scheduler
    criterion = BCEDiceLoss(bce_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler    = torch.cuda.amp.GradScaler()

    # 6. Training loop
    print(f"\n── Training for {NUM_EPOCHS} epochs ──")
    best_iou    = 0.0
    history     = {"train_loss": [], "val_loss": [], "val_iou": [], "val_dice": []}
    viz_epochs  = {1, NUM_EPOCHS // 2, NUM_EPOCHS}  # epochs to save visualizations

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_dl, optimizer, criterion, scaler)
        val_loss, val_iou, val_dice = validate(model, val_dl, criterion)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)
        history["val_dice"].append(val_dice)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
              f"loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
              f"IoU {val_iou:.4f} | Dice {val_dice:.4f} | {elapsed:.1f}s")

        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_iou": val_iou,
                "val_dice": val_dice,
            }, CKPT_DIR / "best_model.pth")
            print(f"  ★ New best IoU: {best_iou:.4f} — saved checkpoint")

        # Save periodic visualizations
        if epoch in viz_epochs:
            save_val_predictions(model, val_pairs[:5], epoch, RESULTS_DIR)

    # 7. Final plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, NUM_EPOCHS + 1)

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"],   label="val")
    axes[0].set_title("Loss (BCE + Dice)"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs, history["val_iou"])
    axes[1].set_title("Validation IoU"); axes[1].set_xlabel("Epoch"); axes[1].grid(True)

    axes[2].plot(epochs, history["val_dice"])
    axes[2].set_title("Validation Dice"); axes[2].set_xlabel("Epoch"); axes[2].grid(True)

    plt.suptitle(f"UNet Training — Best IoU: {best_iou:.4f}", fontsize=13)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_curves.png", dpi=120, bbox_inches="tight")
    plt.close()

    # 8. Final validation with best model
    print("\n── Final evaluation with best model ──")
    ckpt = torch.load(CKPT_DIR / "best_model.pth", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    _, final_iou, final_dice = validate(model, val_dl, criterion)
    save_val_predictions(model, val_pairs, ckpt["epoch"], RESULTS_DIR)
    print(f"\nBest model (epoch {ckpt['epoch']}):")
    print(f"  IoU:  {final_iou:.4f}")
    print(f"  Dice: {final_dice:.4f}")
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Best checkpoint:  {CKPT_DIR}/best_model.pth")


if __name__ == "__main__":
    main()
