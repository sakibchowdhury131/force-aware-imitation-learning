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
from PIL import Image
import cv2

from policy_common import (N_VIEWS, PROPRIO_DIM, pose_matrix_to_9d, rot6d_to_matrix,
                            action_9d_to_pose, MaxAbsNormalizer, DiffusionPolicyNet,
                            gather_obs_views)


# ── Inference ─────────────────────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    n_obs_steps    = ckpt.get('n_obs_steps',    2)
    n_views        = ckpt.get('n_views',        N_VIEWS)
    proprio_dim    = ckpt.get('proprio_dim',    PROPRIO_DIM)
    action_dim     = ckpt.get('action_dim',     9)   # >9 for force-conditioned checkpoints (pose9+force3)
    action_horizon = ckpt['action_horizon']

    # train_method absent -> 'ddpm'/'flow_matching' both reuse DiffusionPolicyNet
    # (only the sampling procedure differs, see predict_action_sequence_flow).
    # 'act' is a genuinely different architecture (act_common.ACTPolicy).
    if ckpt.get('train_method') == 'act':
        from act_common import ACTPolicy
        model = ACTPolicy(
            action_dim=action_dim,
            proprio_dim=proprio_dim,
            action_horizon=action_horizon,
            n_obs_steps=n_obs_steps,
            n_views=n_views,
            hidden_dim=ckpt.get('hidden_dim', 256),
            latent_dim=ckpt.get('latent_dim', 32),
            n_heads=ckpt.get('n_heads', 8),
            n_enc_layers=ckpt.get('n_enc_layers', 4),
            n_dec_layers=ckpt.get('n_dec_layers', 7),
            pretrained=False,
        ).to(device)
    else:
        unet_dims   = ckpt.get('unet_dims',   (256, 512, 1024))
        unet_kernel = ckpt.get('unet_kernel', 5)
        model = DiffusionPolicyNet(
            action_dim=action_dim,
            action_horizon=action_horizon,
            n_obs_steps=n_obs_steps,
            n_views=n_views,
            proprio_dim=proprio_dim,
            pretrained=False,
            unet_dims=tuple(unet_dims),
            unet_kernel=unet_kernel,
        ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    normalizer = MaxAbsNormalizer.from_state_dict(ckpt['normalizer'])
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
def predict_action_sequence(model, normalizer, noise_scheduler, view_tensors, proprio, device):
    """
    DDPM reverse diffusion over a temporal window.

    view_tensors: list[n_obs_steps] of (N_VIEWS, 3, H, W) tensors, oldest → newest
    proprio:      list[n_obs_steps] of (proprio_dim,) arrays — normalised tool pose
    Returns:
      actions: (action_horizon, 9) float64 — denormalized [tx, ty, tz, rot6d]
      poses:   list[action_horizon] of 4×4 np.ndarray
    """
    action_dim     = model.action_dim
    action_horizon = model.action_horizon
    flat           = action_dim * action_horizon

    obs_imgs  = torch.stack(view_tensors).unsqueeze(0).to(device)  # (1, n_obs_steps, N_VIEWS, 3, H, W)
    proprio_t = torch.from_numpy(np.stack(proprio).astype(np.float32)).unsqueeze(0).to(device)
    obs_emb   = model.encode_obs(obs_imgs, proprio_t)

    x = torch.randn(1, flat, device=device)
    noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps)
    for t in noise_scheduler.timesteps:
        t_batch    = torch.tensor([t], device=device).long()
        noise_pred = model(x, t_batch, obs_emb)
        x          = noise_scheduler.step(noise_pred, t, x).prev_sample

    actions_norm = x[0].cpu().numpy().reshape(action_horizon, action_dim)
    actions      = normalizer.denormalize(actions_norm)
    poses        = [action_9d_to_pose(a[:9]) for a in actions]   # a[:9] -- extra dims (e.g. force) ignored here
    return actions, poses


@torch.no_grad()
def predict_action_sequence_flow(model, normalizer, view_tensors, proprio, device,
                                 time_scale=999.0, ode_steps=50):
    """
    Flow-matching counterpart to predict_action_sequence — same model
    (DiffusionPolicyNet reused as a velocity field, see 05_train_flow.py),
    but samples by Euler-integrating dx/dt = v_theta(x_t, t*time_scale, obs)
    from x0 ~ N(0,I) at t=0 to t=1, instead of DDPM reverse diffusion.

    view_tensors / proprio: same shapes as predict_action_sequence.
    Returns: actions (action_horizon, 9) denormalized, poses (list of 4x4).
    """
    action_dim     = model.action_dim
    action_horizon = model.action_horizon
    flat           = action_dim * action_horizon

    obs_imgs  = torch.stack(view_tensors).unsqueeze(0).to(device)
    proprio_t = torch.from_numpy(np.stack(proprio).astype(np.float32)).unsqueeze(0).to(device)
    obs_emb   = model.encode_obs(obs_imgs, proprio_t)

    x  = torch.randn(1, flat, device=device)
    dt = 1.0 / ode_steps
    for i in range(ode_steps):
        t = torch.full((1,), i * dt, device=device)
        v = model(x, t * time_scale, obs_emb)
        x = x + v * dt

    actions_norm = x[0].cpu().numpy().reshape(action_horizon, action_dim)
    actions      = normalizer.denormalize(actions_norm)
    poses        = [action_9d_to_pose(a[:9]) for a in actions]   # a[:9] -- extra dims (e.g. force) ignored here
    return actions, poses


@torch.no_grad()
def predict_action_sequence_act(model, normalizer, view_tensors, proprio, device):
    """
    ACT counterpart to predict_action_sequence -- single transformer forward
    pass predicts the whole action chunk directly (no iterative sampling).
    actions=None at inference -> CVAE latent z is fixed to zero (see
    act_common.ACTPolicy.forward), matching the original ACT eval convention.

    view_tensors / proprio: same shapes as predict_action_sequence.
    Returns: actions (action_horizon, 9) denormalized, poses (list of 4x4).
    """
    obs_imgs  = torch.stack(view_tensors).unsqueeze(0).to(device)
    proprio_t = torch.from_numpy(np.stack(proprio).astype(np.float32)).unsqueeze(0).to(device)
    pred_actions, _, _ = model(obs_imgs, proprio_t, actions=None)

    actions_norm = pred_actions[0].cpu().numpy()
    actions      = normalizer.denormalize(actions_norm)
    poses        = [action_9d_to_pose(a[:9]) for a in actions]   # a[:9] -- extra dims (e.g. force) ignored here
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


def draw_axes_with_horizon(img_rgb: np.ndarray, cam_poses: list, K: np.ndarray,
                            scale: float = 0.06) -> np.ndarray:
    """
    Draw step+1 XYZ axes plus a fading dot trail for the full prediction horizon.
    cam_poses: list of (4,4) poses already in camera frame, oldest=step+1 first.
    """
    vis = img_rgb.copy()

    # Draw horizon trail (steps 2..H) as fading yellow dots
    H = len(cam_poses)
    for k in range(H - 1, -1, -1):
        p = cam_poses[k]
        if p[:3, 3][2] <= 0:
            continue
        pts = _project(p[:3, 3].reshape(1, 3), K)
        if np.any(pts < -2000) or np.any(pts > 20000):
            continue
        alpha = 0.3 + 0.7 * (H - k) / H   # step+1 brightest
        radius = max(3, 8 - k)
        color = (int(255 * alpha), int(200 * alpha), 0)
        cv2.circle(vis, tuple(pts[0]), radius, color, -1, cv2.LINE_AA)

    # Draw full XYZ axes at step+1
    pose = cam_poses[0]
    if pose[:3, 3][2] > 0:
        o  = pose[:3, 3]
        ax = [o + pose[:3, k] * scale for k in range(3)]
        pts = _project(np.array([o, ax[0], ax[1], ax[2]]), K)
        if not (np.any(pts < -5000) or np.any(pts > 50000)):
            origin = tuple(pts[0])
            colors = [(255, 60, 60), (60, 220, 60), (60, 100, 255)]
            labels = ['X', 'Y', 'Z']
            for k in range(3):
                tip = tuple(pts[k + 1])
                cv2.arrowedLine(vis, origin, tip, colors[k], 2, tipLength=0.25)
                cv2.putText(vis, labels[k], tip, cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, colors[k], 1, cv2.LINE_AA)

    t = cam_poses[0][:3, 3]
    cv2.putText(vis,
                f"x={t[0]*100:.1f}  y={t[1]*100:.1f}  z={t[2]*100:.1f} cm",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
    return vis


def _compile_video(img_paths: list, out_dir: str, fps: int = 10):
    if not img_paths:
        return
    first = cv2.imread(img_paths[0])
    if first is None:
        return
    h, w = first.shape[:2]
    out_path = os.path.join(out_dir, 'overlay_video.mp4')
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for p in img_paths:
        frame = cv2.imread(p)
        if frame is not None:
            writer.write(frame)
    writer.release()
    print(f"Saved video → {out_path}")


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

    title = 'Policy prediction horizon — task frame'
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

def run_single(args, model, normalizer, noise_scheduler, transform, device,
               n_obs_steps, n_views, proprio_dim):
    img_paths = args.images
    if len(img_paths) == 1 and n_obs_steps > 1:
        print(f"  [note: repeating single image for all {n_obs_steps} obs steps]")
        img_paths = img_paths * n_obs_steps
    if len(img_paths) != n_obs_steps:
        raise ValueError(f"Expected {n_obs_steps} image(s), got {len(img_paths)}")

    print(f"  [note: --images is a quick sanity check — proprioception is zeroed. "
          f"For a faithful run, use --episode_dir]")

    view_tensors = []
    for p in img_paths:
        t = transform(Image.open(p).convert('RGB'))
        view_tensors.append(t.unsqueeze(0))   # (N_VIEWS=1, 3, H, W)
    proprio = [np.zeros(proprio_dim, dtype=np.float32) for _ in range(n_obs_steps)]

    img_rgb = np.array(Image.open(img_paths[-1]).convert('RGB'))

    actions, poses = predict_action_sequence(
        model, normalizer, noise_scheduler, view_tensors, proprio, device)

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


def run_episode(args, model, normalizer, noise_scheduler, transform, device,
                n_obs_steps, n_views, proprio_dim):
    aug_dir  = os.path.join(args.episode_dir, 'augmented')
    real_dir = os.path.join(aug_dir, 'masked_real')      # cam0/cam1 images, hands removed
    cam0_dir = os.path.join(args.episode_dir, 'cam0')    # raw cam0 images fallback
    out_dir  = args.output_dir or os.path.join(aug_dir, 'policy_predictions')
    os.makedirs(out_dir, exist_ok=True)

    # Tool poses define the frame set — same frames used during training,
    # each paired with N_VIEWS images (real cam0/cam1 + rendered novel views)
    base_gt_path = os.path.join(aug_dir, 'tool_poses_base.npz')
    task_gt_path = os.path.join(aug_dir, 'tool_poses_task.npz')
    cam0_gt_path = os.path.join(aug_dir, 'tool_poses_cam0.npz')
    if os.path.exists(base_gt_path):
        gt_path = base_gt_path
    elif os.path.exists(task_gt_path):
        gt_path = task_gt_path
    else:
        gt_path = cam0_gt_path
    if not os.path.exists(gt_path):
        print(f"No tool poses found under {aug_dir} "
              f"(need tool_poses_base.npz, tool_poses_task.npz or tool_poses_cam0.npz)")
        return

    poses_raw  = dict(np.load(gt_path))
    frame_keys = sorted(poses_raw.keys(), key=int)
    n_frames   = len(frame_keys)
    frame      = 'base' if gt_path == base_gt_path else ('task' if gt_path == task_gt_path else 'cam0')
    print(f"  Loaded {n_frames} tool poses ({frame} frame) from {gt_path}")

    gt_poses = {f'{int(k):06d}': v for k, v in poses_raw.items()} if args.plot_trajectory else None

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

    # Camera extrinsics for task→cam reprojection (and base→task if frame == 'base')
    tf_world2cam = None
    if getattr(args, 'task_frame', None) and os.path.exists(args.task_frame):
        tf_world2cam = np.load(args.task_frame).astype(np.float64)
        print(f"  Loaded camera extrinsics from {args.task_frame}")
    elif args.overlay:
        print("  [warn] --overlay without --task_frame: predictions in task frame, "
              "projection will be wrong unless poses are already in cam0 frame")

    T_task_base = None  # inv(T_base_task): base -> task frame
    if frame == 'base':
        if getattr(args, 'robot_extrinsics', None) and os.path.exists(args.robot_extrinsics):
            T_base_task = np.load(args.robot_extrinsics).astype(np.float64)
            T_task_base = np.linalg.inv(T_base_task)
            print(f"  Loaded robot extrinsics from {args.robot_extrinsics}")
        elif args.overlay:
            print("  [warn] --overlay with base-frame poses but no --robot_extrinsics: "
                  "projection will be wrong")

    want_overlay = args.overlay or args.mesh
    all_poses    = {}
    all_seqs     = {}
    n_windows    = n_frames - n_obs_steps + 1

    print(f"Running inference on {n_windows} windows → {out_dir}/")

    for i in range(n_obs_steps - 1, n_frames):
        obs_keys = frame_keys[i - n_obs_steps + 1 : i + 1]

        view_tensors, proprio, skip = [], [], False
        for k in obs_keys:
            paths = gather_obs_views(aug_dir, k)
            if paths is None:
                print(f"  [skip] frame {int(k):06d}: no masked images found")
                skip = True
                break
            view_tensors.append(torch.stack(
                [transform(Image.open(p).convert('RGB')) for p in paths]))
            proprio.append(normalizer.normalize(pose_matrix_to_9d(poses_raw[k])))
        if skip:
            continue

        actions, poses = predict_action_sequence(
            model, normalizer, noise_scheduler, view_tensors, proprio, device)

        fid_str = f'{int(frame_keys[i]):06d}'
        all_poses[fid_str] = poses[0].astype(np.float32)
        all_seqs[fid_str]  = actions.astype(np.float32)

        t = poses[0][:3, 3]
        print(f"  frame {fid_str}  next: x={t[0]*100:.1f} y={t[1]*100:.1f} z={t[2]*100:.1f} cm")

        if want_overlay and K_cam0 is not None:
            # Prefer hand-removed cam0 image (masked_real uses {fid}_cam0.jpg naming)
            cam0_img_path = None
            for candidate in [
                os.path.join(real_dir, f'{fid_str}_cam0.jpg'),   # masked_real naming
                os.path.join(real_dir, f'{fid_str}.jpg'),         # alternate naming
                os.path.join(cam0_dir,  f'{fid_str}.jpg'),        # raw cam0
            ]:
                if os.path.exists(candidate):
                    cam0_img_path = candidate
                    break
            if cam0_img_path is None:
                cam0_img_path = gather_obs_views(aug_dir, obs_keys[-1])[0]  # last resort

            img_rgb = np.array(Image.open(cam0_img_path).convert('RGB'))

            # Convert predicted poses → camera frame for correct projection
            # (base -> task -> cam, or task -> cam if not base-frame)
            if frame == 'base' and T_task_base is not None and tf_world2cam is not None:
                cam_poses = [tf_world2cam @ T_task_base @ p.astype(np.float64) for p in poses]
            elif tf_world2cam is not None:
                cam_poses = [tf_world2cam @ p.astype(np.float64) for p in poses]
            else:
                cam_poses = [p.astype(np.float64) for p in poses]

            if args.mesh:
                vis = try_mesh_overlay(img_rgb, cam_poses[0], K_cam0, args.mesh)
                if vis is None:
                    vis = draw_axes_with_horizon(img_rgb, cam_poses, K_cam0)
            else:
                vis = draw_axes_with_horizon(img_rgb, cam_poses, K_cam0)

            if vis is not None:
                cv2.imwrite(os.path.join(out_dir, f'{fid_str}_pred.jpg'),
                            cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    np.savez(os.path.join(out_dir, 'predicted_poses.npz'),       **all_poses)
    np.savez(os.path.join(out_dir, 'predicted_action_seqs.npz'), **all_seqs)
    print(f"\nSaved {len(all_poses)} windows → {out_dir}/")

    if want_overlay:
        overlay_imgs = sorted(glob.glob(os.path.join(out_dir, '*_pred.jpg')))
        if overlay_imgs:
            _compile_video(overlay_imgs, out_dir)
        else:
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
                   help='Draw XYZ axes + horizon trail on real cam0 images')
    p.add_argument('--plot_trajectory',  action='store_true',
                   help='Save tx/ty/tz trajectory plot; includes GT if tool_poses_task.npz exists')
    p.add_argument('--task_frame', default=None,
                   help='Path to cam_extrinsics.npy — converts task-frame predictions to '
                        'cam0 frame for correct image projection (required for --overlay)')
    p.add_argument('--robot_extrinsics', default=None,
                   help='Path to robot_extrinsics.npy (T_base_task) — required for --overlay '
                        'when poses are in robot-base frame (tool_poses_base.npz)')
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
    n_views         = ckpt.get('n_views',     N_VIEWS)
    proprio_dim     = ckpt.get('proprio_dim', PROPRIO_DIM)
    action_horizon  = ckpt['action_horizon']
    noise_scheduler = ckpt['noise_scheduler']
    noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps)

    args.crop_size = crop_size
    transform = make_transform(image_size, crop_size)

    print(f"Model: obs_steps={n_obs_steps}  n_views={n_views}  proprio_dim={proprio_dim}  "
          f"action_horizon={action_horizon}  "
          f"image={image_size}×{image_size} → crop {crop_size}×{crop_size}  "
          f"device={args.device}")

    if args.images:
        run_single(args, model, normalizer, noise_scheduler, transform, device,
                   n_obs_steps, n_views, proprio_dim)
    else:
        run_episode(args, model, normalizer, noise_scheduler, transform, device,
                    n_obs_steps, n_views, proprio_dim)


if __name__ == '__main__':
    main()
