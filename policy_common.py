"""
Shared definitions for the diffusion policy used by 05_train.py (training)
and test_policy.py (inference) — kept in one place so the two stay in sync.

Observation (mirrors the original Tool-as-Interface paper, Sec. 3 Problem
Statement): the robot's observation O^r is a SINGLE-view RGB image plus
proprioceptive data x^r ∈ SE(3) — N_VIEWS=1, PROPRIO_DIM=9 (tracked tool
pose, standing in for the end-effector pose for human demos).

Each human demo frame O^h = {I_v1, I_v2} (two real cameras) is augmented via
3D Gaussian-splat novel-view synthesis into a pool of candidate single-view
images — real cam0, real cam1, and the rendered masked novel views — all
sharing the same task-space action label (the action is viewpoint-invariant).
During TRAINING, one image is randomly drawn from this pool per obs step
(treating each candidate as if it were its own "episode" recording of the
same trajectory — the paper's novel-view data augmentation). At INFERENCE,
the deployed single real camera (cam0) is used.
"""

import os, glob, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_VIEWS     = 1   # single-view RGB observation (matches O^r in the paper)
PROPRIO_DIM = 9


# ── Rotation / pose utilities ─────────────────────────────────────────────────

def matrix_to_6d(R: np.ndarray) -> np.ndarray:
    """(3,3) rotation matrix → 6D (first two columns, row-major)."""
    return R[:, :2].T.reshape(6)


def pose_matrix_to_9d(T: np.ndarray) -> np.ndarray:
    """(4,4) pose → 9D [tx, ty, tz, rot6d] — translation first, matches original."""
    return np.concatenate([T[:3, 3], matrix_to_6d(T[:3, :3])]).astype(np.float32)


def rot6d_to_matrix(r6d: np.ndarray) -> np.ndarray:
    """6D → (3,3) via Gram-Schmidt. Inverse of matrix_to_6d."""
    a1, a2 = r6d[:3], r6d[3:]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def action_9d_to_pose(a: np.ndarray) -> np.ndarray:
    """9D action [tx, ty, tz, rot6d] → (4,4) pose."""
    T = np.eye(4, dtype=np.float64)
    T[:3, 3]  = a[:3]
    T[:3, :3] = rot6d_to_matrix(a[3:])
    return T


# ── Normaliser ────────────────────────────────────────────────────────────────

class MaxAbsNormalizer:
    """
    Scales each action/pose dimension independently so that the max absolute
    value maps to 1. Matches the original paper's normalizer_from_stat() logic.
    Used for both action targets and proprioceptive tool-pose observations,
    since they share the same 9D [tx,ty,tz,rot6d] representation.
    """
    def __init__(self, actions: np.ndarray = None):
        if actions is not None:
            max_abs = np.maximum(actions.max(axis=0), np.abs(actions.min(axis=0)))
            max_abs = np.where(max_abs < 1e-8, 1.0, max_abs)
            self.scale  = (1.0 / max_abs).astype(np.float32)   # (action_dim,)
            self.offset = np.zeros_like(self.scale)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return x * self.scale + self.offset

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.offset) / self.scale

    def state_dict(self):
        return {'scale': self.scale, 'offset': self.offset}

    @classmethod
    def from_state_dict(cls, d):
        obj = cls()
        obj.scale  = d['scale']
        obj.offset = d['offset']
        return obj


# ── Observation view gathering ──────────────────────────────────────────────

def gather_obs_pool(aug_dir: str, fid, cam0_only: bool = False,
                    track_cam: int = 0) -> list:
    """
    Returns the pool of candidate images for one obs frame.

    pool[0] is always the deployment camera (track_cam).

    Falls back to raw cam dirs (episode_dir/camN/) when masked_real/ doesn't
    exist yet so training can proceed on raw (unmasked) images.

    cam0_only=True: pool = [track_cam image only].
    cam0_only=False: pool = [track_cam, other_cams..., novel_views...]
    """
    real_dir    = os.path.join(aug_dir, 'masked_real')
    episode_dir = os.path.dirname(aug_dir)

    def _find_cam(cam_i):
        p = os.path.join(real_dir, f'{int(fid):06d}_cam{cam_i}.jpg')
        if os.path.exists(p):
            return p
        p = os.path.join(episode_dir, f'cam{cam_i}', f'{int(fid):06d}.jpg')
        return p if os.path.exists(p) else None

    tc_path = _find_cam(track_cam)
    if tc_path is None:
        return []

    if cam0_only:
        return [tc_path]

    all_masked = sorted(glob.glob(os.path.join(real_dir, f'{int(fid):06d}_cam*.jpg')))
    other_real = [p for p in all_masked if p != tc_path]
    if not other_real:
        import re
        for cam_dir in sorted(glob.glob(os.path.join(episode_dir, 'cam[0-9]*'))):
            if '_depth' in cam_dir:
                continue
            m = re.search(r'cam(\d+)$', os.path.basename(cam_dir))
            if not m:
                continue
            cam_i = int(m.group(1))
            if cam_i == track_cam:
                continue
            p = os.path.join(cam_dir, f'{int(fid):06d}.jpg')
            if os.path.exists(p):
                other_real.append(p)

    novel_dir   = os.path.join(aug_dir, 'masked_novel')
    novel_paths = sorted(glob.glob(os.path.join(novel_dir, f'{int(fid):06d}_novel*.jpg')))
    return [tc_path] + other_real + novel_paths


def select_obs_views(pool: list, n_views: int, training: bool) -> list:
    """
    Returns exactly n_views image paths from pool for one obs step.

    n_views=1: randomly drawn from pool if training (view augmentation),
               else pool[0] (deployment camera).
    n_views>1: always pool[:n_views] — track_cam at [0], others at [1..n-1].
               Deterministic: the model always sees the same camera bundle.
    """
    if n_views == 1:
        return [np.random.choice(pool) if training else pool[0]]
    return pool[:n_views]


def sample_obs_view(pool: list, training: bool) -> str:
    """Legacy single-view wrapper."""
    return np.random.choice(pool) if training else pool[0]


def gather_obs_views(aug_dir: str, fid) -> list:
    """Inference-time view gathering — single real cam0 masked image (N_VIEWS=1)."""
    pool = gather_obs_pool(aug_dir, fid)
    if not pool:
        return None
    return [pool[0]]


# ── UNet-1D noise predictor ───────────────────────────────────────────────────

class _SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=x.device) / (half - 1)
        )
        emb = x.float().unsqueeze(1) * freqs.unsqueeze(0)   # (B, half)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)     # (B, dim)


class _ResConv1dBlock(nn.Module):
    """Conv1d residual block: GroupNorm → Mish → Conv1d, with FiLM conditioning."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 5, cond_dim: int = 256):
        super().__init__()
        pad = kernel // 2
        self.block1   = nn.Sequential(
            nn.GroupNorm(8, in_ch),  nn.Mish(),
            nn.Conv1d(in_ch,  out_ch, kernel, padding=pad))
        self.block2   = nn.Sequential(
            nn.GroupNorm(8, out_ch), nn.Mish(),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad))
        self.film     = nn.Linear(cond_dim, out_ch * 2)
        self.res_conv = (nn.Conv1d(in_ch, out_ch, 1)
                         if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h            = self.block1(x)
        scale, shift = self.film(cond).unsqueeze(-1).chunk(2, dim=1)
        h            = h * (1 + scale) + shift
        return self.block2(h) + self.res_conv(x)


class ConditionalUNet1D(nn.Module):
    """
    Temporal U-Net noise predictor (Chi et al. 2023, Diffusion Policy).

    Treats the action sequence as a 1-D temporal signal so the network can
    model dependencies between consecutive action steps — unlike the flat MLP
    which has no notion of temporal order.

    Architecture:
      input_proj  → [encoder blocks + downsampling] → bottleneck
                  → [upsampling + skip concat + decoder blocks] → output_head

    Global conditioning (timestep + obs embedding) enters every residual block
    via FiLM (feature-wise linear modulation: scale + shift).
    """
    def __init__(self, action_dim: int, action_horizon: int,
                 obs_dim: int = 512,
                 down_dims: tuple = (128, 256, 512),
                 kernel: int = 5):
        super().__init__()
        self.action_dim     = action_dim
        self.action_horizon = action_horizon

        # ── Global conditioning ───────────────────────────────────────────────
        t_dim = down_dims[0]
        self.time_mlp = nn.Sequential(
            _SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 4), nn.Mish(),
            nn.Linear(t_dim * 4, t_dim),
        )
        self.obs_proj = nn.Linear(obs_dim, t_dim)
        cond_dim      = t_dim * 2          # cat([time_emb, obs_proj])

        # ── Input projection ─────────────────────────────────────────────────
        self.input_proj = nn.Conv1d(action_dim, down_dims[0], 1)

        # ── Encoder ──────────────────────────────────────────────────────────
        self.down_blocks  = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        in_ch = down_dims[0]
        for i, out_ch in enumerate(down_dims):
            self.down_blocks.append(nn.ModuleList([
                _ResConv1dBlock(in_ch,  out_ch, kernel, cond_dim),
                _ResConv1dBlock(out_ch, out_ch, kernel, cond_dim),
            ]))
            is_last = (i == len(down_dims) - 1)
            self.downsamplers.append(
                nn.Identity() if is_last
                else nn.Conv1d(out_ch, out_ch, 3, stride=2, padding=1))
            in_ch = out_ch

        # ── Bottleneck ────────────────────────────────────────────────────────
        bot = down_dims[-1]
        self.mid1 = _ResConv1dBlock(bot, bot, kernel, cond_dim)
        self.mid2 = _ResConv1dBlock(bot, bot, kernel, cond_dim)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.up_blocks   = nn.ModuleList()
        self.upsamplers  = nn.ModuleList()
        up_dims = list(reversed(down_dims))
        for i in range(len(up_dims) - 1):
            in_ch   = up_dims[i]
            out_ch  = up_dims[i + 1]
            skip_ch = up_dims[i + 1]     # encoder stage at this resolution has out_ch=skip_ch
            self.upsamplers.append(
                nn.ConvTranspose1d(in_ch, in_ch, 4, stride=2, padding=1))
            self.up_blocks.append(nn.ModuleList([
                _ResConv1dBlock(in_ch + skip_ch, out_ch, kernel, cond_dim),
                _ResConv1dBlock(out_ch,           out_ch, kernel, cond_dim),
            ]))

        # ── Output head ───────────────────────────────────────────────────────
        self.out_norm = nn.GroupNorm(8, down_dims[0])
        self.out_conv = nn.Conv1d(down_dims[0], action_dim, 1)

    def forward(self, noisy_actions_flat: torch.Tensor,
                timesteps: torch.Tensor,
                obs_emb: torch.Tensor) -> torch.Tensor:
        B = noisy_actions_flat.shape[0]

        # Global conditioning vector
        cond = torch.cat([self.time_mlp(timesteps),
                          self.obs_proj(obs_emb)], dim=-1)   # (B, cond_dim)

        # (B, T*D) → (B, D, T)
        x = (noisy_actions_flat
             .reshape(B, self.action_horizon, self.action_dim)
             .permute(0, 2, 1))
        x = self.input_proj(x)

        # Encoder — save skip connections
        skips = []
        for (b1, b2), ds in zip(self.down_blocks, self.downsamplers):
            x = b1(x, cond)
            x = b2(x, cond)
            skips.append(x)
            x = ds(x)

        # Bottleneck
        x = self.mid1(x, cond)
        x = self.mid2(x, cond)

        # Decoder — skip connections from encoder (reversed, excluding last)
        for (b1, b2), us, skip in zip(
                self.up_blocks, self.upsamplers, reversed(skips[:-1])):
            x = us(x)
            x = torch.cat([x, skip], dim=1)
            x = b1(x, cond)
            x = b2(x, cond)

        x = F.mish(self.out_norm(x))
        x = self.out_conv(x)                          # (B, action_dim, T)
        return x.permute(0, 2, 1).reshape(B, -1)      # (B, T*action_dim)


# ── Model ─────────────────────────────────────────────────────────────────────

class DiffusionPolicyNet(nn.Module):
    """
    DDPM-based diffusion policy.

    Observation encoder (unchanged from MLP version):
      ResNet-18 encodes each (obs_step, view) image independently, mean-pools
      across views, concatenates proprioception, projects to obs_dim.

    Noise predictor (upgraded from MLP to UNet-1D):
      ConditionalUNet1D treats the action sequence as a 1-D temporal signal
      with skip connections across resolutions, matching Chi et al. 2023.
    """

    def __init__(self, action_dim: int = 9, obs_dim: int = 512,
                 action_horizon: int = 8, n_obs_steps: int = 2,
                 n_views: int = N_VIEWS, proprio_dim: int = PROPRIO_DIM,
                 pretrained: bool = True,
                 unet_dims: tuple = (128, 256, 512),
                 unet_kernel: int = 5):
        super().__init__()
        import torchvision.models as tvm

        self.n_obs_steps    = n_obs_steps
        self.n_views        = n_views
        self.proprio_dim    = proprio_dim
        self.action_horizon = action_horizon
        self.action_dim     = action_dim

        # ── Image encoder: shared ResNet-18 ──────────────────────────────────
        weights  = tvm.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tvm.resnet18(weights=weights)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])  # (B,512,1,1)

        # Concatenate across views (not mean-pool) so each camera keeps its identity.
        # per_step_dim = obs_dim * n_views + proprio_dim
        per_step_dim = obs_dim * n_views + proprio_dim
        self.obs_proj = nn.Sequential(
            nn.Linear(per_step_dim * n_obs_steps, obs_dim),
            nn.SiLU(),
        )

        # ── UNet-1D noise predictor ───────────────────────────────────────────
        self.unet = ConditionalUNet1D(
            action_dim=action_dim,
            action_horizon=action_horizon,
            obs_dim=obs_dim,
            down_dims=unet_dims,
            kernel=unet_kernel,
        )

    def encode_obs(self, imgs: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """
        imgs:    (B, n_obs_steps, n_views, 3, H, W)
        proprio: (B, n_obs_steps, proprio_dim)
        Returns: (B, obs_dim)
        """
        B, T, V, C, H, W = imgs.shape
        feats = self.encoder(imgs.reshape(B * T * V, C, H, W)).flatten(1)  # (B*T*V, obs_dim)
        feats = feats.reshape(B, T, -1)                                     # (B, T, V*obs_dim) — concat views
        feats = torch.cat([feats, proprio], dim=-1)                         # (B, T, V*obs_dim+proprio_dim)
        return self.obs_proj(feats.reshape(B, -1))

    def forward(self, noisy_actions: torch.Tensor, timesteps: torch.Tensor,
                obs_emb: torch.Tensor) -> torch.Tensor:
        """
        noisy_actions: (B, action_horizon * action_dim)
        timesteps:     (B,)
        obs_emb:       (B, obs_dim)
        Returns predicted noise: (B, action_horizon * action_dim)
        """
        return self.unet(noisy_actions, timesteps, obs_emb)
