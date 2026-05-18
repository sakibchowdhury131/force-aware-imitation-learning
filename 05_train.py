"""
Step 5 — Train a diffusion policy on the augmented data.

Observation: RGB image (masked novel view, resized to 96×96)
Action:      6DOF tool pose as 6D rotation representation + translation (9D total)

Architecture: DDPM with UNet backbone (Chi et al. 2023)

Usage:
    python 05_train.py --data_dir data/episodes/hammer \
        --output_dir data/checkpoints/hammer

Requires:
    pip install diffusers accelerate
"""

import os, sys, glob, argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
from diffusers import DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_cosine_schedule_with_warmup


# ── Rotation utilities ────────────────────────────────────────────────────────

def matrix_to_6d(R: np.ndarray) -> np.ndarray:
    """Convert (3,3) rotation matrix to 6D representation (first two columns)."""
    return R[:, :2].T.reshape(6)  # (6,)


def pose_to_action(T: np.ndarray) -> np.ndarray:
    """Convert (4,4) pose matrix → 9D action [6d_rotation, translation]."""
    R = T[:3, :3]
    t = T[:3, 3]
    return np.concatenate([matrix_to_6d(R), t])  # (9,)


# ── Dataset ───────────────────────────────────────────────────────────────────

class AugmentedEpisodeDataset(Dataset):
    def __init__(self, data_dir: str, obs_horizon: int = 2,
                 action_horizon: int = 8, image_size: int = 96):
        """
        Loads from all episodes under data_dir/.
        Each episode: augmented/masked_novel/*.jpg  +  augmented/tool_poses.npz
        """
        self.obs_horizon    = obs_horizon
        self.action_horizon = action_horizon
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        # Build flat list of (img_path, action_9d) across all episodes
        self.samples = []
        for ep_dir in sorted(glob.glob(os.path.join(data_dir, '*'))):
            pose_file = os.path.join(ep_dir, 'augmented', 'tool_poses.npz')
            novel_dir = os.path.join(ep_dir, 'augmented', 'masked_novel')
            if not os.path.exists(pose_file) or not os.path.exists(novel_dir):
                continue
            poses = np.load(pose_file)
            for frame_id_str in sorted(poses.keys(), key=int):
                frame_id   = int(frame_id_str)
                frame_poses = poses[frame_id_str]  # (N_novel, 4, 4)
                for k, T in enumerate(frame_poses):
                    img_path = os.path.join(novel_dir,
                                            f'{frame_id:06d}_novel{k:02d}.jpg')
                    if os.path.exists(img_path):
                        action = pose_to_action(T).astype(np.float32)
                        self.samples.append((img_path, action))

        if not self.samples:
            raise RuntimeError(f"No samples found under {data_dir}. "
                               "Run steps 2–4 first.")

        print(f"Dataset: {len(self.samples)} (image, action) pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, action = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)   # (3, H, W)
        return img, torch.from_numpy(action)


# ── Model ─────────────────────────────────────────────────────────────────────

class DiffusionPolicyNet(nn.Module):
    """
    Minimal DDPM-based policy:
      - Vision encoder: ResNet-18 (frozen backbone, fine-tuned head)
      - Noise prediction: lightweight MLP diffusion model over action sequence
    """

    def __init__(self, action_dim: int = 9, obs_dim: int = 512,
                 action_horizon: int = 8, num_train_timesteps: int = 100):
        super().__init__()
        import torchvision.models as tvm
        backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])  # → (B, 512, 1, 1)

        self.action_horizon = action_horizon
        self.action_dim     = action_dim
        flat_action = action_dim * action_horizon

        self.noise_pred = nn.Sequential(
            nn.Linear(obs_dim + flat_action + 1, 512),  # +1 for timestep embedding
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, flat_action),
        )

    def encode_obs(self, imgs: torch.Tensor) -> torch.Tensor:
        """imgs: (B, 3, H, W) → (B, 512)"""
        feats = self.encoder(imgs).squeeze(-1).squeeze(-1)
        return feats

    def forward(self, noisy_actions: torch.Tensor, timesteps: torch.Tensor,
                obs_emb: torch.Tensor) -> torch.Tensor:
        """
        noisy_actions: (B, action_horizon * action_dim)
        timesteps:     (B,)
        obs_emb:       (B, obs_dim)
        Returns predicted noise: (B, action_horizon * action_dim)
        """
        t_emb = timesteps.float().unsqueeze(-1) / 100.0
        x = torch.cat([obs_emb, noisy_actions, t_emb], dim=-1)
        return self.noise_pred(x)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    dataset    = AugmentedEpisodeDataset(args.data_dir,
                                         image_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=4, pin_memory=True)

    model = DiffusionPolicyNet(
        action_dim=9,
        action_horizon=args.action_horizon,
    ).to(device)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon',
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_sched  = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,
        num_training_steps=args.num_epochs * len(dataloader),
    )

    print(f"Training for {args.num_epochs} epochs, {len(dataloader)} steps/epoch")

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        for imgs, actions in dataloader:
            imgs    = imgs.to(device)        # (B, 3, H, W)
            actions = actions.to(device)     # (B, 9)

            # Repeat action across horizon (simplified: same action for all steps)
            actions_seq = actions.unsqueeze(1).repeat(1, args.action_horizon, 1)
            actions_flat = actions_seq.reshape(len(imgs), -1)  # (B, action_horizon*9)

            # Forward diffusion
            noise     = torch.randn_like(actions_flat)
            timesteps = torch.randint(0, 100, (len(imgs),), device=device).long()
            noisy     = noise_scheduler.add_noise(actions_flat, noise, timesteps)

            # Predict noise
            with torch.no_grad():
                obs_emb = model.encode_obs(imgs)
            pred_noise = model(noisy, timesteps, obs_emb)

            loss = F.mse_loss(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_sched.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{args.num_epochs}]  loss={avg:.4f}")

        if (epoch + 1) % 50 == 0:
            ckpt = os.path.join(args.output_dir, f'policy_epoch{epoch+1}.pt')
            torch.save({'epoch': epoch+1, 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'noise_scheduler': noise_scheduler,
                        'action_horizon': args.action_horizon}, ckpt)
            print(f"  Saved checkpoint: {ckpt}")

    final_ckpt = os.path.join(args.output_dir, 'policy_final.pt')
    torch.save({'epoch': args.num_epochs, 'model': model.state_dict(),
                'noise_scheduler': noise_scheduler,
                'action_horizon': args.action_horizon}, final_ckpt)
    print(f"\nTraining complete. Final checkpoint: {final_ckpt}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',      required=True, help='Dir containing episode subdirs')
    p.add_argument('--output_dir',    default='data/checkpoints')
    p.add_argument('--num_epochs',    type=int,   default=300)
    p.add_argument('--batch_size',    type=int,   default=64)
    p.add_argument('--lr',            type=float, default=1e-4)
    p.add_argument('--image_size',    type=int,   default=96)
    p.add_argument('--action_horizon',type=int,   default=8)
    p.add_argument('--device',        default='cuda')
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
