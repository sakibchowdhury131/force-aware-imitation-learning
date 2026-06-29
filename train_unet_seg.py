"""
Train a lightweight ResNet-18 UNet to segment human arms/hands from RGB frames.
Distills GroundedSAM2 predictions (slow) into a fast UNet (~10 ms/frame).

Binary ground-truth masks are recovered from (original, blacked-out) JPEG pairs:
any pixel set to [0,0,0] by 03_segment.py is treated as the positive class.

Usage:
    python train_unet_seg.py --task_dir data/episodes/pastaTransfer \
                             --output_dir data/checkpoints/unet_seg

    # resume
    python train_unet_seg.py --task_dir data/episodes/pastaTransfer \
                             --output_dir data/checkpoints/unet_seg \
                             --resume data/checkpoints/unet_seg/unet_best.pt
"""

import os, sys, argparse, random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms.functional as TF
from PIL import Image

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input size fed to the network (must be divisible by 32)
IMG_H, IMG_W = 256, 448

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# ── Architecture ──────────────────────────────────────────────────────────

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ResNetUNet(nn.Module):
    """ResNet-18 encoder + UNet decoder → binary segmentation logits."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        enc = models.resnet18(weights=weights)

        self.enc0 = nn.Sequential(enc.conv1, enc.bn1, enc.relu)  # 64, /2
        self.pool  = enc.maxpool                                   # /4
        self.enc1  = enc.layer1   # 64,  /4
        self.enc2  = enc.layer2   # 128, /8
        self.enc3  = enc.layer3   # 256, /16
        self.enc4  = enc.layer4   # 512, /32

        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128,  64,  64)
        self.up1 = UpBlock( 64,  64,  64)
        self.head = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d  = self.up4(e4, e3)
        d  = self.up3(d,  e2)
        d  = self.up2(d,  e1)
        d  = self.up1(d,  e0)
        return self.head(d)   # (B, 1, H, W) raw logits


# ── Loss ──────────────────────────────────────────────────────────────────

def dice_loss(logits, target, eps=1e-6):
    pred  = logits.sigmoid()
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return (1 - (2 * inter + eps) / (union + eps)).mean()

def seg_loss(logits, target):
    return 0.5 * F.binary_cross_entropy_with_logits(logits, target) \
         + 0.5 * dice_loss(logits, target)


# ── Dataset ───────────────────────────────────────────────────────────────

class SegPairDataset(Dataset):
    """
    Loads (img, mask) pairs from a flat directory produced by prepare_unet_data.py.
    Each sample is a {stem}_img.jpg + {stem}_mask.png pair.
    mask.png is a binary PNG (white = arm, black = background).
    """

    def __init__(self, data_dir, augment=True):
        self.augment  = augment
        stems = sorted({f.replace('_img.jpg', '')
                        for f in os.listdir(data_dir)
                        if f.endswith('_img.jpg')})
        self.img_paths  = [os.path.join(data_dir, s + '_img.jpg')  for s in stems]
        self.mask_paths = [os.path.join(data_dir, s + '_mask.png') for s in stems]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        orig = cv2.cvtColor(cv2.imread(self.img_paths[idx]), cv2.COLOR_BGR2RGB)
        mask_gray = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask_gray > 127).astype(np.float32)

        orig_pil = Image.fromarray(orig)
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode='L')

        orig_pil = orig_pil.resize((IMG_W, IMG_H), Image.BILINEAR)
        mask_pil = mask_pil.resize((IMG_W, IMG_H), Image.NEAREST)

        if self.augment:
            if random.random() > 0.5:
                orig_pil = TF.hflip(orig_pil)
                mask_pil = TF.hflip(mask_pil)
            orig_pil = TF.adjust_brightness(orig_pil, 1 + random.uniform(-0.3, 0.3))
            orig_pil = TF.adjust_contrast(orig_pil,   1 + random.uniform(-0.3, 0.3))
            orig_pil = TF.adjust_saturation(orig_pil, 1 + random.uniform(-0.3, 0.3))

        img_t  = TF.normalize(TF.to_tensor(orig_pil), MEAN, STD)        # (3, H, W)
        mask_t = torch.from_numpy(np.array(mask_pil) / 255.0).float().unsqueeze(0)  # (1, H, W)

        return img_t, mask_t


def collect_pairs_for_episode(ep_dir):
    pairs = []
    aug = os.path.join(ep_dir, 'augmented')
    for src, dst in [('real', 'masked_real'), ('novel', 'masked_novel')]:
        src_dir = os.path.join(aug, src)
        dst_dir = os.path.join(aug, dst)
        if not os.path.isdir(src_dir) or not os.path.isdir(dst_dir):
            continue
        masked_set = set(os.listdir(dst_dir))
        for fname in sorted(os.listdir(src_dir)):
            if fname.endswith('.jpg') and fname in masked_set:
                pairs.append((os.path.join(src_dir, fname),
                              os.path.join(dst_dir, fname)))
    return pairs


def collect_pairs(task_dir):
    ep_dirs = sorted(
        os.path.join(task_dir, n)
        for n in os.listdir(task_dir)
        if os.path.isdir(os.path.join(task_dir, n))
        and os.path.exists(os.path.join(task_dir, n, 'meta.json'))
    )
    pairs = []
    for ep in ep_dirs:
        pairs.extend(collect_pairs_for_episode(ep))
    return pairs


def episode_split(task_dir, eval_frac=0.15, val_frac=0.10, seed=42):
    """
    Split at the episode level so eval/val episodes are fully held out.
    Returns (train_pairs, val_pairs, eval_pairs, eval_episode_dirs).
    """
    ep_dirs = sorted(
        os.path.join(task_dir, n)
        for n in os.listdir(task_dir)
        if os.path.isdir(os.path.join(task_dir, n))
        and os.path.exists(os.path.join(task_dir, n, 'meta.json'))
    )
    rng = random.Random(seed)
    shuffled = ep_dirs[:]
    rng.shuffle(shuffled)

    n_eval = max(1, int(len(shuffled) * eval_frac))
    n_val  = max(1, int(len(shuffled) * val_frac))

    eval_eps  = shuffled[:n_eval]
    val_eps   = shuffled[n_eval:n_eval + n_val]
    train_eps = shuffled[n_eval + n_val:]

    train_pairs = []
    for ep in train_eps:
        train_pairs.extend(collect_pairs_for_episode(ep))
    val_pairs = []
    for ep in val_eps:
        val_pairs.extend(collect_pairs_for_episode(ep))
    eval_pairs = []
    for ep in eval_eps:
        eval_pairs.extend(collect_pairs_for_episode(ep))

    return train_pairs, val_pairs, eval_pairs, eval_eps


# ── Training ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',   required=True,
                   help='Directory with train/ and val/ subdirs (output of prepare_unet_data.py)')
    p.add_argument('--output_dir', default=os.path.join(PIPELINE_DIR, 'data', 'checkpoints', 'unet_seg'))
    p.add_argument('--epochs',     type=int,   default=100)
    p.add_argument('--batch_size', type=int,   default=16)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--workers',    type=int,   default=4)
    p.add_argument('--device',     default='cuda')
    p.add_argument('--resume',     default=None, help='Path to checkpoint to resume from')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    train_ds = SegPairDataset(os.path.join(args.data_dir, 'train'), augment=True)
    val_ds   = SegPairDataset(os.path.join(args.data_dir, 'val'),   augment=False)
    print(f"Dataset: {len(train_ds)} train / {len(val_ds)} val  (from {args.data_dir})")

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"No samples found in {args.data_dir}/train/ or /val/")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.workers, pin_memory=True)

    model     = ResNetUNet(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_val    = float('inf')

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_val    = ckpt.get('best_val', float('inf'))
        print(f"Resumed from epoch {ckpt['epoch']},  best_val={best_val:.4f}")

    for epoch in range(start_epoch, args.epochs):
        # Train
        model.train()
        tr_loss = 0.0
        for imgs, masks in train_dl:
            imgs, masks = imgs.to(device), masks.to(device)
            loss = seg_loss(model(imgs), masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(imgs)
        tr_loss /= len(train_ds)

        # Val
        model.eval()
        vl_loss = 0.0
        tp_sum = fp_sum = fn_sum = 0
        with torch.no_grad():
            for imgs, masks in val_dl:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                vl_loss += seg_loss(logits, masks).item() * len(imgs)
                pred = (logits.sigmoid() > 0.5)
                gt   = masks > 0.5
                tp_sum += (pred & gt).sum().item()
                fp_sum += (pred & ~gt).sum().item()
                fn_sum += (~pred & gt).sum().item()
        vl_loss /= max(1, len(val_ds))
        scheduler.step()

        iou  = tp_sum / max(tp_sum + fp_sum + fn_sum, 1)
        dice = 2 * tp_sum / max(2 * tp_sum + fp_sum + fn_sum, 1)
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:4d}/{args.epochs}  "
              f"train={tr_loss:.4f}  val={vl_loss:.4f}  "
              f"IoU={iou:.4f}  Dice={dice:.4f}  lr={lr_now:.2e}")

        if vl_loss < best_val:
            best_val = vl_loss
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'best_val': best_val},
                       os.path.join(args.output_dir, 'unet_best.pt'))
            print(f"  -> best checkpoint (val={best_val:.4f})")

    torch.save({'epoch': args.epochs - 1, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'best_val': best_val},
               os.path.join(args.output_dir, 'unet_final.pt'))
    print(f"\nDone.  Best val loss: {best_val:.4f}")
    print(f"Checkpoints: {args.output_dir}/unet_best.pt  unet_final.pt")


if __name__ == '__main__':
    main()
