"""
Step 4 — Estimate tool pose using FoundationPose.

Two modes depending on available data:

  TEMPORAL (preferred, requires real depth from step 1):
    Segments tool on the first camN frame, registers with FoundationPose, then
    tracks 6DOF pose through the sampled camN sequence.  For each timestep the
    tracked pose is transformed into every novel-view frame using the camera
    matrices saved by step 2.

    Requires:
      data/episodes/<task>/<ep>/camN_depth/     (recorded by 01_record.py)
      data/episodes/<task>/<ep>/augmented/novel_cameras.npz  (saved by 02_augment_noposplat.py)

  FALLBACK (per-view registration, no real depth needed):
    Segments and registers every masked novel view independently using either
    DA2 monocular depth maps or a constant depth plane.

    Pre-compute depth with:
      python generate_depth.py --image_dir .../augmented/masked_novel

Usage (single episode):
    python 04_track.py --episode_dir data/episodes/hammer/001 \\
        --tool_prompt "hammer" --mesh Hammer.obj

Usage (all episodes in a task):
    python 04_track.py --task_dir data/episodes/hammer \\
        --tool_prompt "hammer" --mesh Hammer.obj

Output (per episode):
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
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--episode_dir', help='Single episode to process')
    src.add_argument('--task_dir',    help='Task directory; processes all episodes')
    p.add_argument('--tool_prompt', default='hammer')
    p.add_argument('--mesh',        required=True, help='Path to tool mesh (.obj or .ply)')
    p.add_argument('--track_cam',   type=int, default=0,
                   help='Camera index to use for real RGBD tracking (default: 0)')
    p.add_argument('--camera',      type=int, default=None,
                   help='Alias for --track_cam (deprecated)')
    p.add_argument('--box_threshold',    type=float, default=0.3)
    p.add_argument('--text_threshold',   type=float, default=0.25)
    p.add_argument('--est_refine_iter',  type=int,   default=5)
    p.add_argument('--track_refine_iter',type=int,   default=2)
    p.add_argument('--depth_const',      type=float, default=0.5,
                   help='Fallback depth (m) when DA2 maps are absent')
    p.add_argument('--skip_done', action='store_true',
                   help='Skip episodes that already have tool_poses.npz')
    p.add_argument('--use_masked', action='store_true',
                   help='Use hand-masked RGB frames (augmented/masked_real/) instead of '
                        'raw camN/ frames for tracking. Depth is still loaded from '
                        'camN_depth/ unmasked. Requires step 3 (--cam_only) to have run.')
    p.add_argument('--task_frame', default=None,
                   help='Path to cam_extrinsics.npy (from 00_calibrate.py). '
                        'Defaults to data/cam_extrinsics.npy for cam0, '
                        'data/cam1_extrinsics.npy for cam1, etc.')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    if args.camera is not None:
        args.track_cam = args.camera
    args.camera = args.track_cam   # internal alias used throughout
    if args.task_frame is None:
        args.task_frame = ('data/cam_extrinsics.npy' if args.track_cam == 0
                           else f'data/cam{args.track_cam}_extrinsics.npy')
    return args


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
    Register on camN frame 0, track through sequence.
    If novel_cameras.npz exists, also transforms poses to novel-view frames.
    If not (e.g. step 2 hasn't run yet), skips that step — still saves
    cam-frame and task-frame poses which are all that training needs.
    """
    cam_idx     = args.camera
    cam0_dir    = os.path.join(args.episode_dir, f'cam{cam_idx}')
    depth_dir   = os.path.join(args.episode_dir, f'cam{cam_idx}_depth')
    depth_scale = meta.get('depth_scale', 0.001)
    K_cam0      = np.array(meta['intrinsics'][cam_idx]['K'], dtype=np.float64)

    novel_cams_path = os.path.join(aug_dir, 'novel_cameras.npz')
    have_novel_cams = os.path.exists(novel_cams_path)

    if have_novel_cams:
        novel_cams     = dict(np.load(novel_cams_path))
        cam_key_suffix = f'_cam{cam_idx}_w2c'
        frame_ids = sorted({int(k.split('_')[0])
                            for k in novel_cams if k.endswith(cam_key_suffix)})
        if not frame_ids:
            raise RuntimeError(
                f"novel_cameras.npz has no '{cam_key_suffix}' entries — "
                f"re-run 02_augment_noposplat.py (which saves both cameras).")
    else:
        # No novel views yet — derive frame list directly from cam directory
        novel_cams = None
        frame_ids  = sorted(
            int(os.path.splitext(f)[0])
            for f in os.listdir(cam0_dir) if f.endswith('.jpg'))

    masked_real_dir = os.path.join(aug_dir, 'masked_real')
    use_masked = args.use_masked and os.path.isdir(masked_real_dir)
    if args.use_masked and not use_masked:
        print("  WARNING: --use_masked requested but masked_real/ not found — using raw cam frames")

    print(f"Temporal tracking: {len(frame_ids)} frames in cam{cam_idx} space"
          + ("" if have_novel_cams else "  (no novel_cameras.npz — cam/task poses only)")
          + ("  [RGB from masked_real/]" if use_masked else ""))

    all_poses     = {}
    cam0_poses    = {}   # raw FoundationPose output in cam0 frame
    pose = None

    for idx, fi in enumerate(tqdm(frame_ids, unit='frame')):
        # RGB: use hand-masked frames if --use_masked, otherwise raw cam frames
        if use_masked:
            rgb_path = os.path.join(masked_real_dir, f'{fi:06d}_cam{cam_idx}.jpg')
        else:
            rgb_path = os.path.join(cam0_dir, f'{fi:06d}.jpg')
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

        cam0_poses[fi] = pose.astype(np.float32)   # save raw cam pose

        # Transform camN pose → each novel-view frame (only if step 2 has run)
        if have_novel_cams:
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


def collect_episodes(task_dir):
    return [os.path.join(task_dir, name)
            for name in sorted(os.listdir(task_dir))
            if os.path.isdir(os.path.join(task_dir, name))
            and os.path.exists(os.path.join(task_dir, name, 'meta.json'))]


def process_episode(episode_dir, args, est, gdino, sam_pred):
    """Track tool pose for one episode. Returns True if work was done."""
    aug_dir = os.path.join(episode_dir, 'augmented')

    if not os.path.exists(os.path.join(episode_dir, 'meta.json')):
        print(f"  Skipping {episode_dir} — no meta.json")
        return False

    out_path = os.path.join(aug_dir, 'tool_poses.npz')
    if args.skip_done and os.path.exists(out_path):
        print(f"  Skipping {episode_dir} — already done (tool_poses.npz exists)")
        return False

    with open(os.path.join(episode_dir, 'meta.json')) as f:
        meta = json.load(f)

    cam_depth_dir   = os.path.join(episode_dir, f'cam{args.camera}_depth')
    novel_cams_path = os.path.join(aug_dir, 'novel_cameras.npz')
    use_temporal    = os.path.isdir(cam_depth_dir)   # depth is all we need

    if use_temporal:
        has_novel = os.path.exists(novel_cams_path)
        print(f"  Mode: TEMPORAL  (cam{args.camera} RGBD"
              + ("  + novel_cameras.npz" if has_novel else "  — no novel views yet") + ")")
    else:
        print("  Mode: FALLBACK PER-VIEW REGISTRATION")
        print(f"    (no cam{args.camera}_depth — re-record for temporal mode)")

    # Swap in the episode_dir for helpers that read args.episode_dir
    orig_ep = args.episode_dir
    args.episode_dir = episode_dir

    if use_temporal:
        all_poses, cam_poses = temporal_track(args, meta, aug_dir, est, gdino, sam_pred)
        cam_out = os.path.join(aug_dir, f'tool_poses_cam{args.camera}.npz')
        np.savez(cam_out, **{str(k): v for k, v in cam_poses.items()})
        print(f"  Saved cam{args.camera} poses → {cam_out}")

        # Transform to task/world frame if calibration is available
        if args.task_frame:
            tf_world2cam = np.load(args.task_frame)            # (4,4) W2C
            tf_cam2world = np.linalg.inv(tf_world2cam)         # C2W = T^task_camera
            task_poses = {k: (tf_cam2world @ v.astype(np.float64)).astype(np.float32)
                          for k, v in cam_poses.items()}
            task_out = os.path.join(aug_dir, 'tool_poses_task.npz')
            np.savez(task_out, **{str(k): v for k, v in task_poses.items()})
            print(f"  Saved task-frame poses → {task_out}")
    else:
        all_poses = fallback_track(args, meta, aug_dir, est, gdino, sam_pred)

    args.episode_dir = orig_ep

    np.savez(out_path, **{str(k): v for k, v in all_poses.items()})
    print(f"  Saved {len(all_poses)} frame pose groups → {out_path}")
    return True


def main():
    args = parse_args()

    if args.episode_dir:
        episode_dirs = [args.episode_dir]
    else:
        episode_dirs = collect_episodes(args.task_dir)
        if not episode_dirs:
            print(f"No episodes found under {args.task_dir}")
            return
        print(f"Found {len(episode_dirs)} episode(s) under {args.task_dir}")

    print("\nLoading GroundingDINO + SAM...")
    gdino, sam_pred = load_gdino_sam(args.device)

    print("Loading mesh...")
    mesh = load_mesh(args.mesh)

    print("Loading FoundationPose...")
    est = build_estimator(mesh)
    print("  FoundationPose ready.")

    n_done = n_skip = 0
    for i, episode_dir in enumerate(episode_dirs):
        ep_name = os.path.relpath(episode_dir, args.task_dir) if args.task_dir else episode_dir
        print(f"\n{'='*60}")
        print(f"Episode {i+1}/{len(episode_dirs)}: {ep_name}")
        print('='*60)
        ok = process_episode(episode_dir, args, est, gdino, sam_pred)
        if ok:
            n_done += 1
        else:
            n_skip += 1

    print(f"\nDone ({n_done} processed, {n_skip} skipped)")

    task_dir = args.task_dir or os.path.dirname(args.episode_dir)
    task_name = os.path.basename(task_dir)
    print(f"Next: python 05_train.py --data_dir {task_dir} "
          f"--output_dir data/checkpoints/{task_name}")


if __name__ == '__main__':
    main()
