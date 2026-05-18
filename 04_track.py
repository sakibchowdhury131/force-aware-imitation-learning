"""
Step 4 — Estimate tool pose using FoundationPose.

Two modes depending on available data:

  TEMPORAL (preferred, requires real depth from step 1):
    Segments tool on the first cam0 frame, registers with FoundationPose, then
    tracks 6DOF pose through the sampled cam0 sequence.  For each timestep the
    tracked cam0 pose is transformed into every novel-view frame using the
    camera matrices saved by step 2.

    Requires:
      data/episodes/<task>/<ep>/cam0_depth/     (recorded by 01_record.py)
      data/episodes/<task>/<ep>/augmented/novel_cameras.npz  (saved by 02_augment.py)

  FALLBACK (per-view registration, no real depth needed):
    Segments and registers every masked novel view independently using either
    DA2 monocular depth maps or a constant depth plane.

    Pre-compute depth with:
      python generate_depth.py --image_dir .../augmented/masked_novel

Usage:
    python 04_track.py --episode_dir data/episodes/hammer/001 \\
        --tool_prompt "hammer" --mesh Hammer.obj

Output:
    data/episodes/<task>/<episode>/augmented/tool_poses.npz
        dict: str(frame_id) → (N_novel, 4, 4) pose matrices
"""

import os, sys, argparse, json
import numpy as np
import cv2
import torch
import trimesh
from PIL import Image
from tqdm import tqdm

PIPELINE_DIR  = os.path.dirname(os.path.abspath(__file__))
THIRD_PARTY   = os.path.join(PIPELINE_DIR, '..', 'Tool_as_Interface', 'third_party')
FP_DIR        = os.path.join(THIRD_PARTY, 'FoundationPose')
GDINO_WEIGHTS = os.path.join(PIPELINE_DIR, 'checkpoints', 'groundingdino_swint_ogc.pth')
SAM_WEIGHTS   = os.path.join(PIPELINE_DIR, 'checkpoints', 'sam_vit_h_4b8939.pth')

sys.path.insert(0, FP_DIR)
sys.path.insert(0, THIRD_PARTY)

import groundingdino
GDINO_CONFIG = os.path.join(os.path.dirname(groundingdino.__file__),
                             'config', 'GroundingDINO_SwinT_OGC.py')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--episode_dir', required=True)
    p.add_argument('--tool_prompt', default='hammer')
    p.add_argument('--mesh',        required=True, help='Path to tool mesh (.obj or .ply)')
    p.add_argument('--camera',      type=int, default=0,
                   help='Camera index to use for real RGBD tracking (default: 0)')
    p.add_argument('--box_threshold',    type=float, default=0.3)
    p.add_argument('--text_threshold',   type=float, default=0.25)
    p.add_argument('--est_refine_iter',  type=int,   default=5)
    p.add_argument('--track_refine_iter',type=int,   default=2)
    p.add_argument('--depth_const',      type=float, default=0.5,
                   help='Fallback depth (m) when DA2 maps are absent')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def load_gdino_sam(device):
    import torchvision.transforms as T
    from groundingdino.util.inference import load_model as load_gdino
    from segment_anything import sam_model_registry, SamPredictor
    print("  Loading GroundingDINO...")
    gdino = load_gdino(GDINO_CONFIG, GDINO_WEIGHTS).to(device).eval()
    print("  Loading SAM...")
    sam = sam_model_registry['vit_h'](checkpoint=SAM_WEIGHTS).to(device)
    return gdino, SamPredictor(sam)


def segment_tool(gdino, sam_pred, img_rgb, prompt, box_thr, text_thr, device):
    import torchvision.transforms as T
    from groundingdino.util.inference import predict
    H, W = img_rgb.shape[:2]
    transform = T.Compose([
        T.Resize(800), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img_t = transform(Image.fromarray(img_rgb)).to(device)
    with torch.no_grad():
        boxes, _, _ = predict(gdino, img_t, prompt, box_thr, text_thr, device=device)

    if boxes is None or len(boxes) == 0:
        return np.zeros((H, W), dtype=np.uint8)

    boxes_px = boxes.clone()
    boxes_px[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * W
    boxes_px[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * H
    boxes_px[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * W
    boxes_px[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * H

    sam_pred.set_image(img_rgb)
    mask_all = np.zeros((H, W), dtype=bool)
    for box in boxes_px:
        m, _, _ = sam_pred.predict(box=box.cpu().numpy(), multimask_output=False)
        mask_all |= m[0].astype(bool)
    return mask_all.astype(np.uint8)


def load_mesh(mesh_path):
    loaded = trimesh.load(mesh_path)
    mesh = (trimesh.util.concatenate(list(loaded.geometry.values()))
            if isinstance(loaded, trimesh.Scene) else loaded)
    max_extent = mesh.bounding_box.extents.max()
    if max_extent > 0.5:
        print(f"  Rescaling mesh cm→m (max extent was {max_extent:.3f})")
        mesh.apply_scale(0.01)
    print(f"  Mesh diameter: {mesh.bounding_box.extents.max():.4f} m")
    return mesh


def build_estimator(mesh):
    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
    import nvdiffrast.torch as dr
    scorer  = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx   = dr.RasterizeCudaContext()
    os.makedirs('/tmp/fp_step4_debug', exist_ok=True)
    return FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        glctx=glctx,
        debug_dir='/tmp/fp_step4_debug',
        debug=0,
    )


def temporal_track(args, meta, aug_dir, est, gdino, sam_pred):
    """
    Register on camN frame 0, track through sequence, transform each tracked
    pose into all novel-view frames using novel_cameras.npz.
    """
    cam_idx     = args.camera
    cam0_dir    = os.path.join(args.episode_dir, f'cam{cam_idx}')
    depth_dir   = os.path.join(args.episode_dir, f'cam{cam_idx}_depth')
    depth_scale = meta.get('depth_scale', 0.001)
    K_cam0      = np.array(meta['intrinsics'][cam_idx]['K'], dtype=np.float64)

    novel_cams_path = os.path.join(aug_dir, 'novel_cameras.npz')
    novel_cams = dict(np.load(novel_cams_path))

    cam_key_suffix = f'_cam{cam_idx}_w2c'
    frame_ids = sorted({int(k.split('_')[0])
                        for k in novel_cams if k.endswith(cam_key_suffix)})
    if not frame_ids:
        raise RuntimeError(
            f"novel_cameras.npz has no '{cam_key_suffix}' entries — "
            f"re-run 02_augment.py (which now saves both cameras).")
    print(f"Temporal tracking: {len(frame_ids)} frames in cam{cam_idx} space")

    all_poses     = {}
    cam0_poses    = {}   # raw FoundationPose output in cam0 frame
    pose = None

    for idx, fi in enumerate(tqdm(frame_ids, unit='frame')):
        rgb_path   = os.path.join(cam0_dir,  f'{fi:06d}.jpg')
        depth_path = os.path.join(depth_dir, f'{fi:06d}.png')

        rgb_bgr = cv2.imread(rgb_path)
        rgb     = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        depth   = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) * depth_scale

        if idx == 0:
            print("\n  Segmenting tool on first frame...")
            mask = segment_tool(gdino, sam_pred, rgb, args.tool_prompt,
                                 args.box_threshold, args.text_threshold, args.device)
            if mask.sum() < 100:
                print("  WARNING: segmentation failed on frame 0 — all poses will be identity")
            else:
                print(f"  Mask: {mask.sum()} px  |  Registering initial pose...")
            pose = est.register(K=K_cam0, rgb=rgb, depth=depth,
                                 ob_mask=mask, iteration=args.est_refine_iter)
            if pose is not None:
                t = pose[:3, 3]
                print(f"  Registered: x={t[0]*100:.1f} y={t[1]*100:.1f} z={t[2]*100:.1f} cm")
        else:
            pose = est.track_one(rgb=rgb, depth=depth, K=K_cam0,
                                  iteration=args.track_refine_iter)

        if pose is None:
            pose = np.eye(4, dtype=np.float32)

        cam0_poses[fi] = pose.astype(np.float32)   # save raw cam0 pose

        # Transform camN pose → each novel-view frame
        tag           = f'{fi:06d}'
        cam0_w2c      = novel_cams[f'{tag}_cam{cam_idx}_w2c']  # (4, 4)
        novel_w2c     = novel_cams[f'{tag}_novel_w2c']          # (N, 4, 4)
        cam0_to_world = np.linalg.inv(cam0_w2c)

        poses = []
        for k in range(len(novel_w2c)):
            pose_novel_k = novel_w2c[k] @ cam0_to_world @ pose
            poses.append(pose_novel_k.astype(np.float32))
        all_poses[fi] = np.stack(poses)   # (N_novel, 4, 4)

    return all_poses, cam0_poses


def fallback_track(args, meta, aug_dir, est, gdino, sam_pred):
    """
    Per-view independent registration on masked novel view images.
    Uses DA2 depth maps if present, otherwise a constant depth plane.
    """
    novel_dir = os.path.join(aug_dir, 'masked_novel')
    depth_dir = novel_dir + '_depth'
    has_depth = os.path.isdir(depth_dir)
    print(f"Fallback (per-view): depth={'DA2 from ' + depth_dir if has_depth else f'constant {args.depth_const}m'}")
    if not has_depth:
        print("  Tip: run `python generate_depth.py --image_dir` on the novel dir for better results")

    # Scale cam0 K to the 512×512 render resolution
    K_orig = np.array(meta['intrinsics'][0]['K'], dtype=np.float64)
    orig_w, orig_h = meta['resolution']
    render_size = 512
    K = K_orig.copy()
    K[0, 0] *= render_size / orig_w;  K[0, 2] *= render_size / orig_w
    K[1, 1] *= render_size / orig_h;  K[1, 2] *= render_size / orig_h

    def load_depth(img_path, shape):
        if has_depth:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            npy  = os.path.join(depth_dir, f'{stem}.npy')
            if os.path.exists(npy):
                return np.load(npy)
        return np.full(shape[:2], args.depth_const, dtype=np.float32)

    files = sorted(f for f in os.listdir(novel_dir) if f.endswith('.jpg'))
    frames: dict = {}
    for fname in files:
        frame_id = int(fname.split('_')[0])
        frames.setdefault(frame_id, []).append(fname)

    print(f"Estimating poses for {len(frames)} frame groups ({len(files)} images total)...")
    all_poses = {}

    for frame_id, fnames in tqdm(sorted(frames.items()), unit='frame'):
        poses = []
        for fname in sorted(fnames):
            img_path = os.path.join(novel_dir, fname)
            img_bgr  = cv2.imread(img_path)
            img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            depth    = load_depth(img_path, img_rgb.shape)

            mask = segment_tool(gdino, sam_pred, img_rgb, args.tool_prompt,
                                 args.box_threshold, args.text_threshold, args.device)

            if mask.sum() < 100:
                poses.append(np.eye(4, dtype=np.float32))
                continue

            pose = est.register(K=K, rgb=img_rgb, depth=depth,
                                 ob_mask=mask, iteration=args.est_refine_iter)
            poses.append(pose.astype(np.float32) if pose is not None
                         else np.eye(4, dtype=np.float32))

        all_poses[frame_id] = np.stack(poses)   # (N_novel, 4, 4)

    return all_poses


def main():
    args = parse_args()
    aug_dir = os.path.join(args.episode_dir, 'augmented')

    with open(os.path.join(args.episode_dir, 'meta.json')) as f:
        meta = json.load(f)

    # Decide tracking mode
    cam_depth_dir   = os.path.join(args.episode_dir, f'cam{args.camera}_depth')
    novel_cams_path = os.path.join(aug_dir, 'novel_cameras.npz')
    use_temporal    = os.path.isdir(cam_depth_dir) and os.path.exists(novel_cams_path)

    if use_temporal:
        print(f"Mode: TEMPORAL TRACKING  (cam{args.camera} RGBD + novel_cameras.npz)")
    else:
        print("Mode: FALLBACK PER-VIEW REGISTRATION")
        if not os.path.isdir(cam_depth_dir):
            print(f"  (no {cam_depth_dir} — re-record with updated 01_record.py for temporal mode)")
        if not os.path.exists(novel_cams_path):
            print(f"  (no novel_cameras.npz — re-run 02_augment.py for temporal mode)")

    print("\nLoading GroundingDINO + SAM...")
    gdino, sam_pred = load_gdino_sam(args.device)

    print("Loading mesh...")
    mesh = load_mesh(args.mesh)

    print("Loading FoundationPose...")
    est = build_estimator(mesh)
    print("  FoundationPose ready.")

    if use_temporal:
        all_poses, cam0_poses = temporal_track(args, meta, aug_dir, est, gdino, sam_pred)
        cam0_out = os.path.join(aug_dir, f'tool_poses_cam{args.camera}.npz')
        np.savez(cam0_out, **{str(k): v for k, v in cam0_poses.items()})
        print(f"Saved cam{args.camera} poses → {cam0_out}")
    else:
        all_poses = fallback_track(args, meta, aug_dir, est, gdino, sam_pred)

    out_path = os.path.join(aug_dir, 'tool_poses.npz')
    np.savez(out_path, **{str(k): v for k, v in all_poses.items()})
    print(f"Saved {len(all_poses)} frame pose groups → {out_path}")
    print(f"Next: python 05_train.py --data_dir {os.path.dirname(args.episode_dir)} "
          f"--output_dir data/checkpoints/{meta.get('task','task')}")


if __name__ == '__main__':
    main()
