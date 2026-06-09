"""
Visualize a full episode as a side-by-side video:
  Left  — real cam0 image with predicted pose axes overlaid
  Right — 3D point cloud (from RGBD depth) + predicted and GT trajectories

The predicted trajectory builds up frame by frame so you can see the policy
"running" through the episode autonomously.

Usage:
    python visualize_episode_3d.py \
        --episode_dir data/episodes/spoonFood/001 \
        --output /tmp/episode_3d.mp4

    # Adjust 3D viewpoint
    python visualize_episode_3d.py \
        --episode_dir data/episodes/spoonFood/001 \
        --elev 25 --azim -45 \
        --output /tmp/episode_3d.mp4

Requirements: opencv-python, matplotlib, numpy (no extra deps beyond the pipeline)
"""

import os, json, glob, argparse
import numpy as np
import cv2
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 — registers 3d projection


# ── Point cloud helpers ───────────────────────────────────────────────────────

def load_depth(path: str, scale: float) -> np.ndarray:
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return d.astype(np.float32) * scale if d is not None else None


def unproject(depth: np.ndarray, K: np.ndarray,
              rgb: np.ndarray = None,
              stride: int = 15,
              min_depth: float = 0.1,
              max_depth: float = 2.5):
    """
    Depth image → 3D points in camera frame.
    stride: pixel step (higher = fewer points, faster)
    Returns pts (N,3) XYZ, colors (N,3) float32 [0,1] or None.
    """
    H, W = depth.shape
    vs, us = np.mgrid[0:H:stride, 0:W:stride]
    z = depth[vs, us]
    valid = (z > min_depth) & (z < max_depth)
    us, vs, z = us[valid], vs[valid], z[valid]

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pts = np.stack([(us - cx) * z / fx,
                    (vs - cy) * z / fy,
                    z], axis=1)

    colors = None
    if rgb is not None:
        colors = rgb[vs, us].astype(np.float32) / 255.0

    return pts, colors


def find_cam_image(fid: str, real_dir: str, cam0_dir: str) -> str:
    """Try multiple naming conventions; return the first path that exists."""
    candidates = [
        os.path.join(real_dir,  f'{fid}_cam0.jpg'),   # masked_real convention
        os.path.join(real_dir,  f'{fid}.jpg'),
        os.path.join(cam0_dir,  f'{fid}.jpg'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def build_background_cloud(frame_ids, episode_dir, aug_dir, K,
                            depth_scale, stride, max_depth, n_bg_frames=12):
    """Sample ~n_bg_frames depth frames to form a static background cloud."""
    cam0_dir  = os.path.join(episode_dir, 'cam0')
    depth_dir = os.path.join(episode_dir, 'cam0_depth')
    real_dir  = os.path.join(aug_dir, 'masked_real')

    if not os.path.isdir(depth_dir):
        print("  [no cam0_depth/ found — skipping point cloud]")
        return None, None

    step = max(1, len(frame_ids) // n_bg_frames)
    pts_list, col_list = [], []

    for fid in frame_ids[::step]:
        depth_path = os.path.join(depth_dir, f'{fid}.png')
        if not os.path.exists(depth_path):
            continue
        depth = load_depth(depth_path, depth_scale)
        if depth is None:
            continue

        rgb_path = find_cam_image(fid, real_dir, cam0_dir)
        rgb = (np.array(Image.open(rgb_path).convert('RGB'))
               if rgb_path else None)

        pts, cols = unproject(depth, K, rgb, stride=stride, max_depth=max_depth)
        if len(pts):
            pts_list.append(pts)
            if cols is not None:
                col_list.append(cols)

    if not pts_list:
        return None, None

    bg_pts    = np.vstack(pts_list)
    bg_colors = np.vstack(col_list) if len(col_list) == len(pts_list) else None

    # Cap at 30k points
    if len(bg_pts) > 30_000:
        idx = np.random.choice(len(bg_pts), 30_000, replace=False)
        bg_pts    = bg_pts[idx]
        bg_colors = bg_colors[idx] if bg_colors is not None else None

    print(f"  Background cloud: {len(bg_pts):,} points from "
          f"{len(pts_list)} frames")
    return bg_pts, bg_colors


# ── Image overlay ─────────────────────────────────────────────────────────────

def draw_axes_bgr(img_bgr: np.ndarray, pose: np.ndarray,
                  K: np.ndarray, scale: float = 0.06) -> np.ndarray:
    """Draw XYZ axes on a BGR image. Returns annotated copy."""
    img = img_bgr.copy()
    o   = pose[:3, 3]
    pts3d = np.array([o,
                      o + pose[:3, 0] * scale,
                      o + pose[:3, 1] * scale,
                      o + pose[:3, 2] * scale])
    h = (K @ pts3d.T).T
    if np.any(h[:, 2] <= 0):
        return img
    p2d = (h[:, :2] / h[:, 2:3]).astype(int)
    orig = tuple(p2d[0])
    for k, color in enumerate([(60, 60, 255), (60, 220, 60), (255, 60, 60)]):
        cv2.arrowedLine(img, orig, tuple(p2d[k + 1]), color, 2, tipLength=0.25)
    t = pose[:3, 3]
    cv2.putText(img,
                f"x={t[0]*100:.1f} y={t[1]*100:.1f} z={t[2]*100:.1f} cm",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return img


# ── 3D render ─────────────────────────────────────────────────────────────────

# matplotlib renders in X/Z/-Y so the scene looks "upright" in camera convention
_X = lambda t: t[:, 0]
_Y = lambda t: t[:, 2]       # camera Z → plot Y (depth)
_Z = lambda t: -t[:, 1]      # camera -Y → plot Z (up)


def render_3d(bg_pts, bg_colors,
              pred_traj, gt_traj,
              current_idx: int,
              elev: float, azim: float,
              figsize=(6, 6), dpi=100) -> np.ndarray:
    """
    Render one frame of the 3D trajectory view.
    Returns (H, W, 3) uint8 RGB.
    pred_traj: list of (3,) translations (all frames)
    gt_traj:   list of (3,) translations (all GT frames, may differ in count)
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax  = fig.add_subplot(111, projection='3d')

    # Background point cloud
    if bg_pts is not None:
        c = bg_colors if bg_colors is not None else 'lightgray'
        ax.scatter(_X(bg_pts), _Y(bg_pts), _Z(bg_pts),
                   c=c, s=0.5, alpha=0.5, linewidths=0, depthshade=True)

    # Full GT trajectory (faint dashed)
    if gt_traj:
        gt = np.array(gt_traj)
        ax.plot(_X(gt), _Y(gt), _Z(gt),
                color='black', linewidth=1.0, linestyle='--', alpha=0.5, label='GT')

    # Predicted trajectory up to current frame (builds up)
    if current_idx >= 0 and pred_traj:
        pt = np.array(pred_traj[:current_idx + 1])
        ax.plot(_X(pt), _Y(pt), _Z(pt),
                color='red', linewidth=2.0, label='predicted')
        # Current position marker
        ax.scatter([_X(pt)[-1]], [_Y(pt)[-1]], [_Z(pt)[-1]],
                   color='red', s=80, zorder=6, depthshade=False)

    ax.set_xlabel('X (m)', fontsize=7)
    ax.set_ylabel('Z depth (m)', fontsize=7)
    ax.set_zlabel('-Y up (m)', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=elev, azim=azim)
    ax.legend(fontsize=7, loc='upper left')
    ax.set_title('Tool trajectory — cam0 frame', fontsize=9)

    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return buf


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--episode_dir', required=True,
                   help='Episode directory (e.g. data/episodes/spoonFood/001)')
    p.add_argument('--predictions', default=None,
                   help='predicted_poses.npz (default: augmented/policy_predictions/)')
    p.add_argument('--output', default=None,
                   help='Output .mp4 path (default: augmented/policy_predictions/episode_3d.mp4)')
    p.add_argument('--fps',              type=int,   default=10)
    p.add_argument('--depth_stride',     type=int,   default=15,
                   help='Pixel stride for point cloud subsampling (higher = fewer points)')
    p.add_argument('--max_depth',        type=float, default=2.5,
                   help='Maximum depth in metres to include in point cloud')
    p.add_argument('--elev',             type=float, default=20,
                   help='3D plot elevation angle (degrees)')
    p.add_argument('--azim',             type=float, default=-60,
                   help='3D plot azimuth angle (degrees)')
    p.add_argument('--panel_height',     type=int,   default=480,
                   help='Height of each video panel in pixels')
    return p.parse_args()


def main():
    args    = parse_args()
    ep_dir  = args.episode_dir
    aug_dir = os.path.join(ep_dir, 'augmented')

    pred_path = (args.predictions
                 or os.path.join(aug_dir, 'policy_predictions', 'predicted_poses.npz'))
    out_path  = (args.output
                 or os.path.join(aug_dir, 'policy_predictions', 'episode_3d.avi'))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if not os.path.exists(pred_path):
        raise FileNotFoundError(
            f"Predictions not found: {pred_path}\n"
            "Run test_policy.py --episode_dir first.")

    # ── Load meta ─────────────────────────────────────────────────────────────
    with open(os.path.join(ep_dir, 'meta.json')) as f:
        meta = json.load(f)
    K           = np.array(meta['intrinsics'][0]['K'], dtype=np.float64)
    depth_scale = meta.get('depth_scale', 0.001)

    # ── Load poses ────────────────────────────────────────────────────────────
    raw_pred = dict(np.load(pred_path))
    pred_poses = {f'{int(k):06d}': v for k, v in raw_pred.items()}

    gt_poses = {}
    gt_path  = os.path.join(aug_dir, 'tool_poses_cam0.npz')
    if os.path.exists(gt_path):
        raw_gt = dict(np.load(gt_path))
        gt_poses = {f'{int(k):06d}': v for k, v in raw_gt.items()}
        print(f"Loaded {len(gt_poses)} GT poses")
    else:
        print("No tool_poses_cam0.npz found — GT trajectory will not be shown")

    frame_ids = sorted(pred_poses.keys())
    print(f"Episode: {len(frame_ids)} prediction windows")

    pred_traj = [pred_poses[fid][:3, 3] for fid in frame_ids]
    gt_traj   = [gt_poses[fid][:3, 3]
                 for fid in sorted(gt_poses.keys()) if fid in gt_poses]

    # ── Build background point cloud ──────────────────────────────────────────
    print("Building background point cloud...")
    bg_pts, bg_colors = build_background_cloud(
        frame_ids, ep_dir, aug_dir, K, depth_scale,
        stride=args.depth_stride, max_depth=args.max_depth)

    # ── Determine layout ──────────────────────────────────────────────────────
    cam0_dir = os.path.join(ep_dir, 'cam0')
    real_dir = os.path.join(aug_dir, 'masked_real')

    sample_cam_path = find_cam_image(frame_ids[0], real_dir, cam0_dir)
    sample_cam = cv2.imread(sample_cam_path) if sample_cam_path else None
    orig_h, orig_w = sample_cam.shape[:2] if sample_cam is not None else (480, 640)

    ph  = args.panel_height
    cam_w = int(orig_w * ph / orig_h)

    # 3D panel: square at same height
    p3d_w = ph

    total_w = cam_w + p3d_w
    fourcc  = cv2.VideoWriter_fourcc(*'XVID')
    writer  = cv2.VideoWriter(out_path, fourcc, args.fps, (total_w, ph))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open {out_path} — try a different --output path")

    # ── Render frames ─────────────────────────────────────────────────────────
    print(f"Rendering {len(frame_ids)} frames  →  {out_path}")
    for i, fid in enumerate(frame_ids):
        # Left: cam0 image with axes
        cam_path = find_cam_image(fid, real_dir, cam0_dir)
        if cam_path:
            cam_bgr = cv2.imread(cam_path)
        else:
            cam_bgr = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

        cam_bgr = draw_axes_bgr(cam_bgr, pred_poses[fid], K)
        cam_bgr = cv2.resize(cam_bgr, (cam_w, ph))

        # Frame counter
        cv2.putText(cam_bgr, f"frame {i+1}/{len(frame_ids)}",
                    (8, ph - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1, cv2.LINE_AA)

        # Right: 3D trajectory
        frame_3d_rgb = render_3d(
            bg_pts, bg_colors, pred_traj, gt_traj, i,
            elev=args.elev, azim=args.azim, figsize=(p3d_w / 100, ph / 100))
        frame_3d_bgr = cv2.cvtColor(
            cv2.resize(frame_3d_rgb, (p3d_w, ph)), cv2.COLOR_RGB2BGR)

        writer.write(np.hstack([cam_bgr, frame_3d_bgr]))

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  {i+1}/{len(frame_ids)}")

    writer.release()
    print(f"\nDone → {out_path}")
    print(f"Play:  vlc {out_path}  or  ffplay {out_path}")


if __name__ == '__main__':
    main()
