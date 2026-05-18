"""
Visualize tool poses — overlay 3D bounding box and axes on images.

Three modes (mutually exclusive flags):
  (default)    masked_novel/ images  +  tool_poses.npz
  --use_novel  novel/ images         +  tool_poses.npz      (unmasked, better context)
  --use_real   cam0/ real images     +  tool_poses_cam0.npz (pose in real camera frame)

Usage:
    python visualize_poses.py --episode_dir data/episodes/hammer/001 --mesh Hammer.obj
    python visualize_poses.py --episode_dir data/episodes/hammer/001 --mesh Hammer.obj --use_novel
    python visualize_poses.py --episode_dir data/episodes/hammer/001 --mesh Hammer.obj --use_real
"""

import os, sys, argparse, json
import numpy as np
import cv2
import trimesh

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
THIRD_PARTY  = os.path.join(PIPELINE_DIR, '..', 'Tool_as_Interface', 'third_party')
FP_DIR       = os.path.join(THIRD_PARTY, 'FoundationPose')
sys.path.insert(0, FP_DIR)
sys.path.insert(0, THIRD_PARTY)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--episode_dir', required=True)
    p.add_argument('--mesh',        required=True, help='Tool mesh (.obj or .ply)')
    p.add_argument('--output_dir',  default=None,  help='Where to save (default: augmented/viz_poses/)')
    p.add_argument('--axis_scale',  type=float, default=0.08, help='Axis length in metres')
    p.add_argument('--camera',      type=int,   default=0,
                   help='Camera index used in step 4 (only affects --use_real, default: 0)')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--use_novel', action='store_true',
                      help='Overlay on unmasked novel/ images instead of masked_novel/')
    mode.add_argument('--use_real',  action='store_true',
                      help='Overlay on real camN/ images using tool_poses_camN.npz')
    return p.parse_args()


def load_mesh(mesh_path):
    loaded = trimesh.load(mesh_path)
    mesh = (trimesh.util.concatenate(list(loaded.geometry.values()))
            if isinstance(loaded, trimesh.Scene) else loaded)
    if mesh.bounding_box.extents.max() > 0.5:
        mesh.apply_scale(0.01)
    return mesh


def draw_one(img_rgb, pose, K, to_origin, bbox, axis_scale, label):
    from Utils import draw_posed_3d_box, draw_xyz_axis
    center_pose = pose @ np.linalg.inv(to_origin)
    vis = draw_posed_3d_box(K, img=img_rgb, ob_in_cam=center_pose, bbox=bbox)
    vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=axis_scale,
                        K=K, thickness=3, transparency=0, is_input_rgb=True)
    t = pose[:3, 3]
    cv2.putText(vis, f"{label}  x={t[0]*100:.1f} y={t[1]*100:.1f} z={t[2]*100:.1f} cm",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return vis


def main():
    args    = parse_args()
    aug_dir = os.path.join(args.episode_dir, 'augmented')
    out_dir = args.output_dir or os.path.join(aug_dir, 'viz_poses')
    os.makedirs(out_dir, exist_ok=True)

    meta   = json.load(open(os.path.join(args.episode_dir, 'meta.json')))
    K_cam0 = np.array(meta['intrinsics'][0]['K'], dtype=np.float64)
    orig_w, orig_h = meta['resolution']

    # K scaled to 512×512 (novel view resolution)
    K_novel_fallback = K_cam0.copy()
    K_novel_fallback[0, 0] *= 512 / orig_w;  K_novel_fallback[0, 2] *= 512 / orig_w
    K_novel_fallback[1, 1] *= 512 / orig_h;  K_novel_fallback[1, 2] *= 512 / orig_h

    mesh = load_mesh(args.mesh)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    # ── Real cam0 mode ────────────────────────────────────────────────────────
    if args.use_real:
        cam_idx    = args.camera
        poses_path = os.path.join(aug_dir, f'tool_poses_cam{cam_idx}.npz')
        if not os.path.exists(poses_path):
            print(f"ERROR: {poses_path} not found.")
            print(f"  Re-run step 4 with --camera {cam_idx} in temporal mode to generate it.")
            return
        poses   = dict(np.load(poses_path))   # str(frame_id) → (4, 4)
        K_use   = np.array(meta['intrinsics'][cam_idx]['K'], dtype=np.float64)
        cam_dir = os.path.join(args.episode_dir, f'cam{cam_idx}')
        total   = 0
        for frame_key, pose in sorted(poses.items(), key=lambda x: int(x[0])):
            frame_id = int(frame_key)
            img_path = os.path.join(cam_dir, f'{frame_id:06d}.jpg')
            if not os.path.exists(img_path):
                continue
            img_rgb  = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            vis      = draw_one(img_rgb, pose, K_use, to_origin, bbox,
                                args.axis_scale, f"frame {frame_id} [cam{cam_idx}]")
            out_path = os.path.join(out_dir, f'{frame_id:06d}_cam{cam_idx}_pose.jpg')
            cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            total += 1
        print(f"Saved {total} annotated cam{cam_idx} images → {out_dir}/")
        print(f"View:  eog {out_dir}/*_cam{cam_idx}_pose.jpg")
        return

    # ── Novel view mode (masked or unmasked) ──────────────────────────────────
    poses_path = os.path.join(aug_dir, 'tool_poses.npz')
    if not os.path.exists(poses_path):
        print(f"ERROR: {poses_path} not found — run step 4 first.")
        return

    poses      = dict(np.load(poses_path))
    cams_path  = os.path.join(aug_dir, 'novel_cameras.npz')
    novel_cams = dict(np.load(cams_path)) if os.path.exists(cams_path) else {}
    img_dir    = os.path.join(aug_dir, 'novel' if args.use_novel else 'masked_novel')

    total = 0
    for frame_key, pose_batch in sorted(poses.items(), key=lambda x: int(x[0])):
        frame_id = int(frame_key)
        tag      = f'{frame_id:06d}'
        for k, pose in enumerate(pose_batch):
            fname    = f'{tag}_novel{k:02d}.jpg'
            img_path = os.path.join(img_dir, fname)
            if not os.path.exists(img_path):
                continue
            img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            K_key = f'{tag}_novel_K'
            K = novel_cams[K_key][k].astype(np.float64) if K_key in novel_cams else K_novel_fallback
            vis = draw_one(img_rgb, pose, K, to_origin, bbox,
                           args.axis_scale, f"frame {frame_id}  view {k}")
            out_path = os.path.join(out_dir, f'{tag}_novel{k:02d}_pose.jpg')
            cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            total += 1

    src = 'novel' if args.use_novel else 'masked_novel'
    print(f"Saved {total} annotated {src}/ images → {out_dir}/")
    print(f"View:  eog {out_dir}/*.jpg")


if __name__ == '__main__':
    main()
