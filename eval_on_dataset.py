"""
Run the trained policy on recorded dataset episodes and produce overlay videos.

For each episode:
  - Feeds masked cam0 images + tool_poses_base proprioception to the policy
  - Overlays predicted future tool positions (action horizon) on the unmasked frame
  - Saves a side-by-side video: left = unmasked frame, right = masked policy input

Usage:
    python eval_on_dataset.py \\
        --checkpoint data/checkpoints/pastaTransfer2/policy_epoch3100.pt \\
        --data_dir   data/episodes/pastaTransfer2 \\
        --episodes   001 002 003 \\
        --output_dir /tmp/eval_videos
"""

import os, sys, glob, argparse, json
import numpy as np
import cv2
import torch
from PIL import Image

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)

from policy_common import (pose_matrix_to_9d, action_9d_to_pose,
                            MaxAbsNormalizer, DiffusionPolicyNet,
                            sample_obs_view)
from torchvision import transforms


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = DiffusionPolicyNet(
        action_dim=9,
        action_horizon=ckpt['action_horizon'],
        n_obs_steps=ckpt['n_obs_steps'],
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    normalizer = MaxAbsNormalizer.from_state_dict(ckpt['normalizer'])
    return model, normalizer, ckpt


def make_transform(image_size, crop_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


@torch.no_grad()
def predict(model, normalizer, noise_scheduler, view_tensors, proprio_list, device):
    """Run DDPM denoising and return predicted (action_horizon, 9) poses."""
    n_obs  = len(view_tensors)
    obs_imgs = torch.stack([
        view_tensors[i]                        # (1, 3, H, W) = (N_VIEWS=1, 3, H, W)
        for i in range(n_obs)
    ]).unsqueeze(0).to(device)                 # (1, n_obs, N_VIEWS, 3, H, W)

    proprio_t = torch.from_numpy(
        np.stack(proprio_list).astype(np.float32)
    ).unsqueeze(0).to(device)                  # (1, n_obs, 9)

    obs_emb    = model.encode_obs(obs_imgs, proprio_t)
    action_dim = model.action_horizon * 9
    noisy      = torch.randn(1, action_dim, device=device)

    noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps)
    for t in noise_scheduler.timesteps:
        pred   = model(noisy, t.unsqueeze(0).to(device), obs_emb)
        noisy  = noise_scheduler.step(pred, t, noisy).prev_sample

    actions_norm = noisy[0].cpu().numpy().reshape(model.action_horizon, 9)
    actions      = normalizer.denormalize(actions_norm)
    poses        = [action_9d_to_pose(a) for a in actions]
    return poses   # list of (4,4) in base frame


# ── Projection helpers ────────────────────────────────────────────────────────

def project_to_pixel(T_base_tool, tf_world2cam, T_task_base, K):
    """Project tool origin (base frame) → (u, v) pixel in cam0.

    Chain mirrors 04c_to_base_frame.py:
        T_base_tool = T_base_task @ inv(tf_world2cam) @ cam0_pose
    so the inverse is:
        cam0_pose   = tf_world2cam @ T_task_base @ T_base_tool
    """
    T_cam_tool = tf_world2cam @ T_task_base @ T_base_tool
    xyz = T_cam_tool[:3, 3]
    if xyz[2] <= 0:
        return None
    u = K[0, 0] * xyz[0] / xyz[2] + K[0, 2]
    v = K[1, 1] * xyz[1] / xyz[2] + K[1, 2]
    return int(round(u)), int(round(v))


def draw_predictions(frame, poses_base, tf_world2cam, T_task_base, K):
    """Draw predicted action-horizon dots on frame (fading yellow → green)."""
    H = len(poses_base)
    vis = frame.copy()
    for k in range(H - 1, -1, -1):
        pix = project_to_pixel(poses_base[k], tf_world2cam, T_task_base, K)
        if pix is None:
            continue
        u, v = pix
        if not (0 <= u < vis.shape[1] and 0 <= v < vis.shape[0]):
            continue
        alpha  = 0.3 + 0.7 * (H - k) / H
        radius = max(4, 12 - k)
        color  = (0, int(255 * alpha), int(255 * (1 - alpha * 0.5)))
        cv2.circle(vis, (u, v), radius, color, -1, cv2.LINE_AA)
        cv2.circle(vis, (u, v), radius, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def draw_actual(frame, T_base_tool, tf_world2cam, T_task_base, K):
    """Draw actual tool position as a blue dot."""
    pix = project_to_pixel(T_base_tool, tf_world2cam, T_task_base, K)
    if pix is None:
        return frame
    u, v = pix
    vis = frame.copy()
    if 0 <= u < vis.shape[1] and 0 <= v < vis.shape[0]:
        cv2.circle(vis, (u, v), 8, (255, 80, 0), -1, cv2.LINE_AA)
        cv2.circle(vis, (u, v), 8, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


# ── Per-episode evaluation ────────────────────────────────────────────────────

def eval_episode(ep_dir, model, normalizer, noise_scheduler, transform,
                 tf_world2cam, T_task_base, K, n_obs_steps, device, output_dir):
    ep_name  = os.path.basename(ep_dir)
    aug_dir  = os.path.join(ep_dir, 'augmented')
    pose_path = os.path.join(aug_dir, 'tool_poses_base.npz')
    cam_path  = os.path.join(aug_dir, 'novel_cameras.npz')

    if not os.path.exists(pose_path):
        print(f"  [{ep_name}] No tool_poses_base.npz — skipping")
        return

    poses_raw  = dict(np.load(pose_path))           # str(fid) → (4,4)
    frame_ids  = sorted(poses_raw.keys(), key=int)
    n_frames   = len(frame_ids)

    if n_frames < n_obs_steps + 1:
        print(f"  [{ep_name}] Too few frames — skipping")
        return

    out_path = os.path.join(output_dir, f'{ep_name}.mp4')
    H_out, W_out = 480, 848 * 2   # side-by-side: left=overlay, right=masked input

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw     = cv2.VideoWriter(out_path, fourcc, 10, (W_out, H_out))

    obs_buffer = []   # (img_tensor, proprio_9d_normalized)
    print(f"  [{ep_name}] {n_frames} frames → {out_path}")

    for i, fid in enumerate(frame_ids):
        fid_int = int(fid)

        # Load unmasked frame for overlay background
        raw_path = os.path.join(aug_dir, 'real', f'{fid_int:06d}_cam0.jpg')
        if not os.path.exists(raw_path):
            raw_path = os.path.join(ep_dir, 'cam0', f'{fid_int:06d}.jpg')
        if not os.path.exists(raw_path):
            continue
        raw_bgr = cv2.imread(raw_path)
        if raw_bgr is None:
            continue

        # Load masked frame for policy input
        masked_path = os.path.join(aug_dir, 'masked_real', f'{fid_int:06d}_cam0.jpg')
        if not os.path.exists(masked_path):
            continue
        masked_pil = Image.open(masked_path).convert('RGB')

        # Proprioception
        T_base_tool = poses_raw[fid].astype(np.float64)
        proprio_raw  = pose_matrix_to_9d(T_base_tool)
        proprio_norm = normalizer.normalize(proprio_raw.reshape(1, 9))[0]

        img_t = transform(masked_pil).unsqueeze(0)   # (1,3,H,W)
        obs_buffer.append((img_t, proprio_norm))
        if len(obs_buffer) > n_obs_steps:
            obs_buffer.pop(0)

        # Run policy when buffer is full
        poses_pred = None
        if len(obs_buffer) == n_obs_steps:
            view_tensors = [v[0] for v in obs_buffer]    # each (1,3,H,W)
            proprio_list = [v[1] for v in obs_buffer]
            try:
                poses_pred = predict(model, normalizer, noise_scheduler,
                                     view_tensors, proprio_list, device)
            except Exception as e:
                print(f"    inference error at frame {fid_int}: {e}")

        # Build overlay frame
        overlay = draw_actual(raw_bgr, T_base_tool, tf_world2cam, T_task_base, K)
        if poses_pred is not None:
            overlay = draw_predictions(overlay, poses_pred, tf_world2cam, T_task_base, K)

        # Add legend
        cv2.putText(overlay, f"ep {ep_name}  frame {fid_int:03d}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(overlay, "BLUE = actual  YELLOW/GREEN = predicted",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Right panel: masked input resized to match
        masked_bgr = cv2.cvtColor(np.array(masked_pil), cv2.COLOR_RGB2BGR)
        masked_bgr = cv2.resize(masked_bgr, (848, 480))
        overlay    = cv2.resize(overlay, (848, 480))

        frame_out = np.concatenate([overlay, masked_bgr], axis=1)
        vw.write(frame_out)

    vw.release()
    print(f"  [{ep_name}] saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data_dir',   default='data/episodes/pastaTransfer2')
    p.add_argument('--episodes',   nargs='+', default=None,
                   help='Episode names to eval (e.g. 001 002). Default: first 3.')
    p.add_argument('--output_dir', default='/tmp/eval_videos')
    p.add_argument('--robot_extrinsics', default='data/robot_extrinsics.npy')
    p.add_argument('--device',     default='cuda')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"Loading policy: {args.checkpoint}")
    model, normalizer, ckpt = load_model(args.checkpoint, device)
    image_size   = ckpt.get('image_size', 128)
    crop_size    = ckpt.get('crop_size', 115)
    n_obs_steps  = ckpt.get('n_obs_steps', 2)
    noise_scheduler = ckpt['noise_scheduler']
    noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps)
    transform = make_transform(image_size, crop_size)

    T_base_task   = np.load(os.path.join(PIPELINE_DIR, args.robot_extrinsics)).astype(np.float64)
    T_task_base   = np.linalg.inv(T_base_task)
    tf_world2cam  = np.load(os.path.join(PIPELINE_DIR, 'data', 'cam_extrinsics.npy')).astype(np.float64)

    # Find episodes
    all_eps = sorted(glob.glob(os.path.join(PIPELINE_DIR, args.data_dir, '*')))
    if args.episodes:
        all_eps = [os.path.join(PIPELINE_DIR, args.data_dir, e) for e in args.episodes]
    else:
        all_eps = all_eps[:3]

    for ep_dir in all_eps:
        if not os.path.isdir(ep_dir):
            print(f"  Skipping {ep_dir} — not a directory")
            continue
        # Load cam0 intrinsics from meta.json
        meta_path = os.path.join(ep_dir, 'meta.json')
        if not os.path.exists(meta_path):
            print(f"  Skipping {ep_dir} — no meta.json")
            continue
        meta = json.load(open(meta_path))
        K = np.array(meta['intrinsics'][0]['K'], dtype=np.float64)

        eval_episode(ep_dir, model, normalizer, noise_scheduler, transform,
                     tf_world2cam, T_task_base, K, n_obs_steps, device, args.output_dir)

    print(f"\nDone. Videos saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
