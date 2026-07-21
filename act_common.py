"""
ACT (Action Chunking Transformer) model definitions, adapted from Zhao et al.
2023 "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware".

Not diffusion-based: a single forward pass through a transformer encoder/
decoder predicts the full action_horizon chunk directly (DETR-style learned
queries), instead of iteratively denoising. Trained as a conditional VAE:
  - CVAEEncoder sees the ground-truth action chunk (+ proprio) and produces a
    latent style variable z (only available at train time -- teacher forcing
    the "how" of an otherwise multi-modal demonstration).
  - The main encoder/decoder conditions on z, proprio, and image tokens to
    predict the action chunk.
  - At inference, z is fixed to zero (no ground truth to encode from) --
    matches the original ACT implementation's deterministic-eval convention.

Loss = L1 reconstruction + kl_weight * KL(q(z|proprio,actions) || N(0,I)).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_2d_pos_embed(h: int, w: int, dim: int, device) -> torch.Tensor:
    """Fixed 2D sinusoidal positional embedding (DETR-style). Returns (h*w, dim)."""
    assert dim % 4 == 0, "hidden_dim must be divisible by 4 for 2D sinusoidal pos embed"
    d4 = dim // 4
    freqs = torch.exp(-math.log(10000) * torch.arange(d4, device=device).float() / d4)
    y_pos = torch.arange(h, device=device).float()
    x_pos = torch.arange(w, device=device).float()
    pe_y = y_pos[:, None] * freqs[None, :]              # (h, d4)
    pe_x = x_pos[:, None] * freqs[None, :]              # (w, d4)
    pe_y = torch.cat([pe_y.sin(), pe_y.cos()], dim=-1)  # (h, d4*2)
    pe_x = torch.cat([pe_x.sin(), pe_x.cos()], dim=-1)  # (w, d4*2)
    pe = torch.cat([
        pe_y[:, None, :].expand(h, w, -1),
        pe_x[None, :, :].expand(h, w, -1),
    ], dim=-1)                                          # (h, w, dim)
    return pe.reshape(h * w, dim)


class CVAEEncoder(nn.Module):
    """[CLS, proprio, action_1..action_H] -> (mu, logvar) for latent z."""
    def __init__(self, action_dim: int, proprio_dim: int, hidden_dim: int,
                 latent_dim: int, n_heads: int, n_layers: int, action_horizon: int):
        super().__init__()
        self.cls_token     = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.proprio_proj  = nn.Linear(proprio_dim, hidden_dim)
        self.action_proj   = nn.Linear(action_dim, hidden_dim)
        self.pos_embed     = nn.Parameter(torch.randn(1, action_horizon + 2, hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(hidden_dim, n_heads, hidden_dim * 4,
                                           dropout=0.1, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head    = nn.Linear(hidden_dim, latent_dim * 2)

    def forward(self, proprio_last: torch.Tensor, actions: torch.Tensor):
        B = proprio_last.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        p   = self.proprio_proj(proprio_last).unsqueeze(1)
        a   = self.action_proj(actions)
        x   = torch.cat([cls, p, a], dim=1) + self.pos_embed[:, :2 + actions.shape[1]]
        out = self.encoder(x)
        mu, logvar = self.head(out[:, 0]).chunk(2, dim=-1)
        return mu, logvar


class ACTPolicy(nn.Module):
    def __init__(self, action_dim: int = 9, proprio_dim: int = 9, action_horizon: int = 16,
                 n_obs_steps: int = 2, n_views: int = 2, hidden_dim: int = 256,
                 latent_dim: int = 32, n_heads: int = 8, n_enc_layers: int = 4,
                 n_dec_layers: int = 7, pretrained: bool = True):
        super().__init__()
        import torchvision.models as tvm

        self.action_horizon = action_horizon
        self.action_dim     = action_dim
        self.n_obs_steps    = n_obs_steps
        self.n_views        = n_views
        self.hidden_dim     = hidden_dim
        self.latent_dim     = latent_dim

        # ── Image backbone: keep the spatial feature map (no global pool) ──────
        weights  = tvm.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tvm.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # (B,512,h,w)
        self.img_proj = nn.Conv2d(512, hidden_dim, 1)

        self.proprio_proj = nn.Linear(proprio_dim, hidden_dim)
        self.latent_proj  = nn.Linear(latent_dim, hidden_dim)
        self.cvae_encoder = CVAEEncoder(action_dim, proprio_dim, hidden_dim,
                                        latent_dim, n_heads, 4, action_horizon)

        # [z, proprio_1..proprio_T] learned positional embeddings
        self.extra_pos_embed = nn.Parameter(torch.randn(1, 1 + n_obs_steps, hidden_dim) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(hidden_dim, n_heads, hidden_dim * 4,
                                               dropout=0.1, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_enc_layers)

        self.query_embed = nn.Parameter(torch.randn(1, action_horizon, hidden_dim) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(hidden_dim, n_heads, hidden_dim * 4,
                                               dropout=0.1, activation='gelu', batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, n_dec_layers)

        self.action_head = nn.Linear(hidden_dim, action_dim)

    def encode_images(self, imgs: torch.Tensor) -> torch.Tensor:
        """imgs: (B, n_obs_steps, n_views, 3, H, W) -> (B, T*V*h*w, hidden_dim)"""
        B, T, V, C, H, W = imgs.shape
        feat = self.backbone(imgs.reshape(B * T * V, C, H, W))   # (B*T*V, 512, h, w)
        feat = self.img_proj(feat)                                # (B*T*V, D, h, w)
        _, D, h, w = feat.shape
        pos = sinusoidal_2d_pos_embed(h, w, D, feat.device)       # (h*w, D)
        feat = feat.flatten(2).permute(0, 2, 1) + pos[None]       # (B*T*V, h*w, D)
        return feat.reshape(B, T * V * h * w, D)

    def forward(self, imgs: torch.Tensor, proprio: torch.Tensor,
                actions: torch.Tensor = None):
        """
        imgs:    (B, n_obs_steps, n_views, 3, H, W)
        proprio: (B, n_obs_steps, proprio_dim)
        actions: (B, action_horizon, action_dim), or None at inference (z=0)
        Returns: pred_actions (B, action_horizon, action_dim), mu, logvar
        """
        B = imgs.shape[0]
        img_tokens = self.encode_images(imgs)

        if actions is not None:
            mu, logvar = self.cvae_encoder(proprio[:, -1], actions)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        else:
            mu = logvar = None
            z = torch.zeros(B, self.latent_dim, device=imgs.device)

        z_tok = self.latent_proj(z).unsqueeze(1)                  # (B, 1, D)
        p_tok = self.proprio_proj(proprio)                        # (B, n_obs_steps, D)
        extra = torch.cat([z_tok, p_tok], dim=1) + self.extra_pos_embed
        memory = self.encoder(torch.cat([extra, img_tokens], dim=1))

        queries = self.query_embed.expand(B, -1, -1)
        decoded = self.decoder(queries, memory)
        pred_actions = self.action_head(decoded)
        return pred_actions, mu, logvar


def act_loss(pred_actions: torch.Tensor, gt_actions: torch.Tensor,
            mu: torch.Tensor, logvar: torch.Tensor, kl_weight: float = 10.0):
    l1 = F.l1_loss(pred_actions, gt_actions)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return l1 + kl_weight * kl, l1, kl
