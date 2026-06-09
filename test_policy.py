"""
Run inference with a trained diffusion policy.

Two modes:
  --images      One image per obs_step (oldest → newest). Prints the predicted
                action sequence. If only one image is given it is repeated for all
                obs_steps (convenient for quick sanity checks).
  --episode_dir Run on every temporal window of n_obs_steps consecutive frames in
                an episode, save overlaid images and/or a trajectory plot.

Usage:
    # Single window (n_obs_steps=2: provide two consecutive masked novel views)
    python test_policy.py --checkpoint data/checkpoints/hammer/policy_final.pt \
        --images frame_t-1.jpg frame_t.jpg

    # Full episode — axes overlay on real cam0 images (no mesh needed)
    python test_policy.py --checkpoint data/checkpoints/hammer/policy_final.pt \
        --episode_dir data/episodes/hammer/001 \
        --overlay --output_dir /tmp/policy_test

    # Full episode — trajectory plot vs ground truth
    python test_policy.py --checkpoint data/checkpoints/hammer/policy_final.pt \
        --episode_dir data/episodes/hammer/001 \
        --plot_trajectory --output_dir /tmp/policy_test

    # Full episode — mesh overlay (requires FoundationPose + trimesh)
    python test_policy.py --checkpoint data/checkpoints/hammer/policy_final.pt \
        --episode_dir data/episodes/hammer/001 \
        --mesh hammer.obj --output_dir /tmp/policy_test
"""

import os, sys, glob, argparse, json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import cv2


# ── Re-create model (must match 05_train.py) ─────────────────────────────────

class DiffusionPolicyNet(nn.Module):
    def __init__(self, action_dim=9, obs_dim=512, action_horizon=8, n_obs_steps=2):
        super().__init__()
        import torchvision.models as tvm
        backbone = tvm.resnet18(weights=None)
        self.encoder        = nn.Sequential(*list(backbone.children())[:-1])
        self.n_obs_steps    = n_obs_steps
        self.action_horizon = action_horizon
        self.action_dim     = action_dim

        self.obs_proj = nn.Sequential(
            nn.Linear(obs_dim * n_obs_steps, obs_dim),
            nn.SiLU(),
        )
        self.time_emb = nn.Sequential(
            nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64),
        )

        hidden      = 512
        flat_action = action_dim * action_horizon
        self.net = nn.ModuleList([
            nn.Linear(flat_action + 64, hidden),
            nn.Linear(hidden, hidden),
            nn.Linear(hidden, hidden),
            nn.Linear(hidden, flat_action),
        ])
        self.film = nn.ModuleList([
            nn.Linear(obs_dim, hidden * 2) for _ in range(3)
        ])
        self.act = nn.SiLU()

    def encode_obs(self, imgs: torch.Tensor) -> torch.Tensor:
        """imgs: (B, n_obs_steps*3, H, W) → (B, obs_dim)"""
        frame_imgs = imgs.chunk(self.n_obs_steps, dim=1)
        embs = [self.encoder(f).squeeze(-1).squeeze(-1) for f in frame_imgs]
        return self.obs_proj(torch.cat(embs, dim=-1))

    def forward(self, noisy_actions, timesteps, obs_emb):
        t_emb = self.time_emb(timesteps.float().unsqueeze(-1) / 100.0)
        x = torch.cat([noisy_actions, t_emb], dim=-1)
        for i, layer in enumerate(self.net[:-1]):
            x = layer(x)
            film_out = self.film[i](obs_emb)
            scale, shift = film_out.chunk(2, dim=-1)
            x = x * (1 + scale) + shift
            x = self.act(x)
        return self.net[-1](x)


# ── Normaliser ────────────────────────────────────────────────────────────────

class MaxAbsNormalizer:
    def __init__(self, state_dict):
        self.scale  = state_dict['scale']
        self.offset = state_dict['offset']

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.offset) / self.scale


# ── Rotation / pose utilities ─────────────────────────────────────────────────

def rot6d_to_matrix(r6d: np.ndarray) -> np.ndarray:
    """6D → (3,3) rotation matrix via Gram-Schmidt."""
    a1, a2 = r6d[:3], r6d[3:]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def action_9d_to_pose(a: np.ndarray) -> np.ndarray:
    """9D [tx, ty, tz, rot6d] → 4×4 pose matrix."""
    T = np.eye(4, dtype=np.float64)
    T[:3, 3]  = a[:3]
    T[:3, :3] = rot6d_to_matrix(a[3:])
    return T


# ── Inference ─────────────────────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    n_obs_steps    = ckpt.get('n_obs_steps',    2)
    action_horizon = ckpt['action_horizon']
    model = DiffusionPolicyNet(
        action_dim=9,
        action_horizon=action_horizon,
        n_obs_steps=n_obs_steps,
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    normalizer = MaxAbsNormalizer(ckpt['normalizer'])
    return model, normalizer, ckpt


def make_transform(image_size, crop_size):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


@torch.no_grad()
def predict_action_sequence(model, normalizer, noise_scheduler, img_tensors, device):
    """
    DDPM reverse diffusion over a temporal window.

    img_tensors: list[n_obs_steps] of (3, H, W) tensors, oldest → newest
    Returns:
      actions: (action_horizon, 9) float64 — denormalized [tx, ty, tz, rot6d]
      poses:   list[action_horizon] of 4×4 np.ndarray
    """
    action_dim     = model.action_dim
    action_horizon = model.action_horizon
    flat           = action_dim * action_horizon

    obs_img = torch.cat(img_tensors, dim=0).unsqueeze(0).to(device)  # (1, n_obs*3, H, W)
    obs_emb = model.encode_obs(obs_img)

    x = torch.randn(1, flat, device=device)
    noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps)
    for t in noise_scheduler.timesteps:
        t_batch    = torch.tensor([t], device=device).long()
        noise_pred = model(x, t_batch, obs_emb)
        x          = noise_scheduler.step(noise_pred, t, x).prev_sample

    actions_norm = x[0].cpu().numpy().reshape(action_horizon, action_dim)
    actions      = normalizer.denormalize(actions_norm)
    poses        = [action_9d_to_pose(a) for a in actions]
    return actions, poses


# ── Visualisation ─────────────────────────────────────────────────────────────

def _project(pts_3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    """pts_3d: (N, 3) in camera frame → (N, 2) pixel coords (no distortion)."""
    h = (K @ pts_3d.T).T          # (N, 3)
    return (h[:, :2] / h[:, 2:3]).astype(int)


def draw_axes_simple(img_rgb: np.ndarray, pose: np.ndarray, K: np.ndarray,
                     scale: float = 0.08) -> np.ndarray:
    """
    Draw XYZ coordinate axes of a pose on an image.
    pose is in the same camera frame as K (i.e. cam0 pose on cam0 image).
    scale: axis length in metres.
    Returns annotated copy (RGB).
    """
    o  = pose[:3, 3]
    ax = [o + pose[:3, k] * scale for k in range(3)]
    pts = _project(np.array([o, ax[0], ax[1], ax[2]]), K)

    if np.any(pts < -5000) or np.any(pts > 50000):
        return img_rgb.copy()   # degenerate projection, skip

    vis = img_rgb.copy()
    origin = tuple(pts[0])
    colors = [(255, 60, 60), (60, 220, 60), (60, 100, 255)]   # X=red Y=green Z=blue
    labels = ['X', 'Y', 'Z']
    for k in range(3):
        tip = tuple(pts[k + 1])
        cv2.arrowedLine(vis, origin, tip, colors[k], 2, tipLength=0.25)
        cv2.putText(vis, labels[k], tip, cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, colors[k], 1, cv2.LINE_AA)

    t = pose[:3, 3]
    cv2.putText(vis,
                f"x={t[0]*100:.1f}  y={t[1]*100:.1f}  z={t[2]*100:.1f} cm",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
    return vis


def try_mesh_overlay(img_rgb, pose, K, mesh_path):
    """Draw 3D bounding box overlay via FoundationPose utils. Returns None if unavailable."""
    try:
        import trimesh
        PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
        THIRD_PARTY  = os.path.join(PIPELINE_DIR, '..', 'Tool_as_Interface', 'third_party')
        FP_DIR       = os.path.join(THIRD_PARTY, 'FoundationPose')
        sys.path.insert(0, FP_DIR)
        sys.path.insert(0, THIRD_PARTY)
        from Utils import draw_posed_3d_box, draw_xyz_axis

        loaded = trimesh.load(mesh_path)
        mesh   = (trimesh.util.concatenate(list(loaded.geometry.values()))
                  if isinstance(loaded, trimesh.Scene) else loaded)
        if mesh.bounding_box.extents.max() > 0.5:
            mesh.apply_scale(0.01)
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox        = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
        center_pose = pose @ np.linalg.inv(to_origin)

        vis = draw_posed_3d_box(K, img=img_rgb.copy(), ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.08,
                            K=K, thickness=3, transparency=0, is_input_rgb=True)
        t = pose[:3, 3]
        cv2.putText(vis, f"x={t[0]*100:.1f} y={t[1]*100:.1f} z={t[2]*100:.1f} cm",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        return vis
    except Exception as e:
        print(f"  [mesh overlay skipped: {e}]")
        return None


def _rotmat_to_euler(R: np.ndarray) -> np.ndarray:
    """(3,3) rotation matrix → [roll, pitch, yaw] in degrees (XYZ extrinsic)."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_euler('xyz', degrees=True)


def _rot6d_to_euler(r6d: np.ndarray) -> np.ndarray:
    """9D action [tx,ty,tz,rot6d] → Euler angles in degrees."""
    return _rotmat_to_euler(rot6d_to_matrix(r6d[3:]))


def save_trajectory_plot(out_dir: str, pred_poses: dict, gt_poses: dict = None,
                         all_seqs: dict = None):
    """
    Plot translation (tx/ty/tz in cm) and rotation (roll/pitch/yaw in degrees)
    over time — 6 subplots total.

    pred_poses:  {fid_str → (4,4)}  — step+1 predictions
    gt_poses:    {fid_str → (4,4)}  — ground-truth cam0 poses
    all_seqs:    {fid_str → (action_horizon, 9)}  — full horizon [tx,ty,tz,rot6d]

    Each predicted step k is plotted at x = window_index + k so it aligns with
    the GT frame it is predicting.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    frame_ids = sorted(pred_poses.keys())
    n         = len(frame_ids)

    # ── Extract step+1 translation and rotation ───────────────────────────────
    pred_t = np.array([pred_poses[fid][:3, 3]      for fid in frame_ids]) * 100.0
    pred_r = np.array([_rotmat_to_euler(pred_poses[fid][:3, :3]) for fid in frame_ids])

    gt_xs, gt_t, gt_r = [], None, None
    if gt_poses is not None:
        gt_fids = [fid for fid in frame_ids if fid in gt_poses]
        if gt_fids:
            gt_xs = [frame_ids.index(fid) for fid in gt_fids]
            gt_t  = np.array([gt_poses[fid][:3, 3]           for fid in gt_fids]) * 100.0
            gt_r  = np.array([_rotmat_to_euler(gt_poses[fid][:3, :3]) for fid in gt_fids])

    # ── Build per-horizon-step arrays ─────────────────────────────────────────
    # Each entry: (xs, translations_cm (N,3), euler_deg (N,3))
    horizon_steps = None
    if all_seqs is not None:
        H = next(iter(all_seqs.values())).shape[0]
        horizon_steps = []
        for k in range(H):
            xs  = np.arange(n) + (k + 1)
            t_k = np.array([all_seqs[fid][k, :3] for fid in frame_ids]) * 100.0
            r_k = np.array([_rot6d_to_euler(all_seqs[fid][k]) for fid in frame_ids])
            horizon_steps.append((xs, t_k, r_k))

    # ── Layout: 6 rows (translation top, rotation bottom) ────────────────────
    fig, axes = plt.subplots(6, 1, figsize=(14, 16), sharex=True)

    t_labels = ['tx (cm)',    'ty (cm)',    'tz (cm)']
    r_labels = ['roll (°)',   'pitch (°)', 'yaw (°)']
    t_colors = ['#e05555',    '#44aa44',   '#4488dd']
    r_colors = ['#cc6600',    '#9933cc',   '#009999']

    def _plot_row(ax, i, values_step1, gt_values, color, ylabel, is_rotation=False):
        xs_s1 = np.arange(n) + 1

        # GT
        if gt_values is not None and len(gt_xs):
            ax.plot(gt_xs, gt_values[:, i], color='black', linewidth=1.8,
                    linestyle='--', marker='x', markersize=3,
                    label='ground truth', zorder=5)

        if horizon_steps is not None:
            H = len(horizon_steps)
            # Fan: steps 2 … H-1
            for k in range(1, H - 1):
                xs, t_k, r_k = horizon_steps[k]
                data = r_k if is_rotation else t_k
                alpha = 0.08 + 0.12 * (H - 1 - k) / max(H - 2, 1)
                ax.plot(xs, data[:, i], color=color, linewidth=0.8,
                        alpha=alpha, zorder=2)
            # Step+1 (solid)
            xs1, t1, r1 = horizon_steps[0]
            data1 = r1 if is_rotation else t1
            ax.plot(xs1, data1[:, i], color=color, linewidth=1.8,
                    marker='o', markersize=2, label='step+1', zorder=4)
            # Step+H (dotted)
            xsH, tH, rH = horizon_steps[-1]
            dataH = rH if is_rotation else tH
            ax.plot(xsH, dataH[:, i], color=color, linewidth=1.2,
                    linestyle=':', label=f'step+{H}', zorder=3)
        else:
            ax.plot(xs_s1, values_step1[:, i], color=color, linewidth=1.5,
                    marker='o', markersize=2, label='step+1', zorder=4)

        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    # Translation rows
    for i in range(3):
        _plot_row(axes[i], i, pred_t, gt_t, t_colors[i], t_labels[i], is_rotation=False)

    # Divider between translation and rotation sections
    axes[2].spines['bottom'].set_linewidth(2)

    # Rotation rows
    for i in range(3):
        _plot_row(axes[3 + i], i, pred_r, gt_r, r_colors[i], r_labels[i], is_rotation=True)

    axes[-1].set_xlabel('Frame index (GT-aligned — each step shifted by its horizon offset)',
                        fontsize=9)

    title = 'Policy prediction horizon — cam0 frame'
    if horizon_steps:
        title += f'  (fan = steps 1–{len(horizon_steps)}, dotted = step+{len(horizon_steps)})'
    axes[0].set_title(title, fontsize=11)

    # Section labels
    fig.text(0.005, 0.72, 'TRANSLATION', fontsize=8, color='gray',
             rotation=90, va='center', fontweight='bold')
    fig.text(0.005, 0.27, 'ROTATION', fontsize=8, color='gray',
             rotation=90, va='center', fontweight='bold')

    plt.tight_layout(rect=[0.015, 0, 1, 1])

    out_path = os.path.join(out_dir, 'trajectory.png')
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved trajectory plot → {out_path}")


# ── Modes ─────────────────────────────────────────────────────────────────────

def run_single(args, model, normalizer, noise_scheduler, transform, device, n_obs_steps):
    img_paths = args.images
    if len(img_paths) == 1 and n_obs_steps > 1:
        print(f"  [note: repeating single image for all {n_obs_steps} obs steps]")
        img_paths = img_paths * n_obs_steps
    if len(img_paths) != n_obs_steps:
        raise ValueError(f"Expected {n_obs_steps} image(s), got {len(img_paths)}")

    img_tensors = [transform(Image.open(p).convert('RGB')) for p in img_paths]
    img_rgb     = np.array(Image.open(img_paths[-1]).convert('RGB'))

    actions, poses = predict_action_sequence(
        model, normalizer, noise_scheduler, img_tensors, device)

    print(f"\nPredicted action sequence ({len(actions)} future steps):")
    for k, a in enumerate(actions):
        print(f"  step {k+1:2d}: tx={a[0]*100:.1f}  ty={a[1]*100:.1f}  tz={a[2]*100:.1f} cm")

    print(f"\nFirst predicted pose 4×4 (metres):")
    print(poses[0].round(4))

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float64)
        if args.mesh:
            vis = try_mesh_overlay(img_rgb, poses[0], K, args.mesh)
        elif args.overlay:
            vis = draw_axes_simple(img_rgb, poses[0], K)
        else:
            vis = None
        if vis is not None:
            out = os.path.join(args.output_dir, 'prediction.jpg')
            cv2.imwrite(out, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            print(f"\nSaved overlay → {out}")


def run_episode(args, model, normalizer, noise_scheduler, transform, device, n_obs_steps):
    aug_dir   = os.path.join(args.episode_dir, 'augmented')
    novel_dir = os.path.join(aug_dir, 'masked_novel')
    real_dir  = os.path.join(aug_dir, 'masked_real')     # cam0 images, hands removed
    cam0_dir  = os.path.join(args.episode_dir, 'cam0')   # raw cam0 images fallback
    out_dir   = args.output_dir or os.path.join(aug_dir, 'policy_predictions')
    os.makedirs(out_dir, exist_ok=True)

    # Group masked novel view images by frame_id (filename: {fid:06d}_novel{k}.jpg)
    frame_map = {}  # fid_str → sorted list of novel view paths
    for p in sorted(glob.glob(os.path.join(novel_dir, '*.jpg'))):
        stem  = os.path.splitext(os.path.basename(p))[0]
        parts = stem.split('_novel')
        if len(parts) != 2:
            continue
        frame_map.setdefault(parts[0], []).append(p)
    for fid in frame_map:
        frame_map[fid].sort()

    frame_ids = sorted(frame_map.keys())
    n_frames  = len(frame_ids)

    if n_frames < n_obs_steps:
        print(f"Not enough frames ({n_frames}) for n_obs_steps={n_obs_steps}")
        return

    # Intrinsics — use cam0 K at full resolution for overlay on real images
    K_cam0   = None
    meta_path = os.path.join(args.episode_dir, 'meta.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        K_cam0 = np.array(meta['intrinsics'][0]['K'], dtype=np.float64)

    # Ground-truth poses for trajectory comparison
    gt_poses = None
    gt_path  = os.path.join(aug_dir, 'tool_poses_cam0.npz')
    if args.plot_trajectory and os.path.exists(gt_path):
        raw = dict(np.load(gt_path))
        gt_poses = {f'{int(k):06d}': v for k, v in raw.items()}
        print(f"  Loaded {len(gt_poses)} ground-truth poses from {gt_path}")

    want_overlay = args.overlay or args.mesh
    all_poses    = {}
    all_seqs     = {}
    n_windows    = n_frames - n_obs_steps + 1

    print(f"Running inference on {n_windows} windows → {out_dir}/")

    for i in range(n_obs_steps - 1, n_frames):
        obs_fids    = frame_ids[i - n_obs_steps + 1 : i + 1]
        img_tensors = [transform(Image.open(frame_map[fid][0]).convert('RGB'))
                       for fid in obs_fids]

        actions, poses = predict_action_sequence(
            model, normalizer, noise_scheduler, img_tensors, device)

        fid_str = frame_ids[i]
        all_poses[fid_str] = poses[0].astype(np.float32)
        all_seqs[fid_str]  = actions.astype(np.float32)

        t = poses[0][:3, 3]
        print(f"  frame {fid_str}  next: x={t[0]*100:.1f} y={t[1]*100:.1f} z={t[2]*100:.1f} cm")

        if want_overlay and K_cam0 is not None:
            # Prefer real cam0 image — pose is in cam0 frame so projection is exact
            cam0_img_path = (os.path.join(real_dir, f'{fid_str}.jpg')
                             if os.path.isdir(real_dir)
                             else os.path.join(cam0_dir,  f'{fid_str}.jpg'))
            if not os.path.exists(cam0_img_path):
                # Last resort: novel view (approximate; pose frame differs)
                cam0_img_path = frame_map[fid_str][0]

            img_rgb = np.array(Image.open(cam0_img_path).convert('RGB'))

            if args.mesh:
                vis = try_mesh_overlay(img_rgb, poses[0], K_cam0, args.mesh)
                if vis is None:
                    vis = draw_axes_simple(img_rgb, poses[0], K_cam0)
            else:
                vis = draw_axes_simple(img_rgb, poses[0], K_cam0)

            if vis is not None:
                cv2.imwrite(os.path.join(out_dir, f'{fid_str}_pred.jpg'),
                            cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    np.savez(os.path.join(out_dir, 'predicted_poses.npz'),       **all_poses)
    np.savez(os.path.join(out_dir, 'predicted_action_seqs.npz'), **all_seqs)
    print(f"\nSaved {len(all_poses)} windows → {out_dir}/")

    if want_overlay:
        print(f"View overlays:  eog {out_dir}/*_pred.jpg")

    if args.plot_trajectory:
        save_trajectory_plot(out_dir, all_poses, gt_poses, all_seqs)


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, help='Path to .pt checkpoint')
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--images',      nargs='+',
                     help='One image per obs_step (oldest → newest). '
                          'One image is OK — repeated for all obs steps.')
    src.add_argument('--episode_dir', help='Episode directory; sliding window over masked_novel/')
    p.add_argument('--overlay',          action='store_true',
                   help='Draw XYZ axes on real cam0 images (no mesh required)')
    p.add_argument('--plot_trajectory',  action='store_true',
                   help='Save tx/ty/tz trajectory plot; includes GT if tool_poses_cam0.npz exists')
    p.add_argument('--mesh',       default=None, help='Tool mesh for 3D bbox overlay (.obj/.ply)')
    p.add_argument('--output_dir', default=None, help='Where to save outputs')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    print(f"Loading checkpoint: {args.checkpoint}")
    model, normalizer, ckpt = load_model(args.checkpoint, device)

    image_size      = ckpt.get('image_size', 128)
    crop_size       = ckpt.get('crop_size',  115)
    n_obs_steps     = ckpt.get('n_obs_steps', 2)
    action_horizon  = ckpt['action_horizon']
    noise_scheduler = ckpt['noise_scheduler']
    noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps)

    args.crop_size = crop_size
    transform = make_transform(image_size, crop_size)

    print(f"Model: obs_steps={n_obs_steps}  action_horizon={action_horizon}  "
          f"image={image_size}×{image_size} → crop {crop_size}×{crop_size}  "
          f"device={args.device}")

    if args.images:
        run_single(args, model, normalizer, noise_scheduler, transform, device, n_obs_steps)
    else:
        run_episode(args, model, normalizer, noise_scheduler, transform, device, n_obs_steps)


if __name__ == '__main__':
    main()
