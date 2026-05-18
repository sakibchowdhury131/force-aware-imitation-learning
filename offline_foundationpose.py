"""
Offline FoundationPose on pre-recorded RGB images.

Segments the tool on the first frame with GroundedSAM, then tracks its 6DOF pose
through the sequence. Saves annotated frames to --output_dir instead of a window.

Usage:
    python offline_foundationpose.py \
        --image_dir data/episodes/hammer/001/cam0 \
        --meta      data/episodes/hammer/001/meta.json \
        --mesh      Hammer.obj \
        --tool_prompt "hammer" \
        --init_frame 0 \
        --output_dir /tmp/fp_results

Depth maps (from generate_depth.py) are loaded automatically if the
<image_dir>_depth/ directory exists, otherwise a constant 0.5m plane is used.
"""

import os, sys, glob, argparse, json
import numpy as np
import cv2
import torch
import trimesh
from PIL import Image
import torchvision.transforms as T
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
    p.add_argument('--image_dir',    required=True)
    p.add_argument('--meta',         required=True)
    p.add_argument('--mesh',         required=True)
    p.add_argument('--cam_idx',      type=int,   default=0)
    p.add_argument('--tool_prompt',  default='hammer')
    p.add_argument('--init_frame',   type=int,   default=0,
                   help='Frame index to use for initial segmentation + registration')
    p.add_argument('--num_frames',   type=int,   default=50,
                   help='How many frames to track after init (0 = all)')
    p.add_argument('--box_threshold',  type=float, default=0.3)
    p.add_argument('--text_threshold', type=float, default=0.25)
    p.add_argument('--est_refine_iter',   type=int, default=5)
    p.add_argument('--track_refine_iter', type=int, default=2)
    p.add_argument('--depth_const',  type=float, default=0.5)
    p.add_argument('--output_dir',   default='/tmp/fp_results')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def load_gdino_sam(device):
    from groundingdino.util.inference import load_model as load_gdino
    from segment_anything import sam_model_registry, SamPredictor
    print("Loading GroundingDINO...")
    gdino = load_gdino(GDINO_CONFIG, GDINO_WEIGHTS).to(device).eval()
    print("Loading SAM...")
    sam = sam_model_registry['vit_h'](checkpoint=SAM_WEIGHTS).to(device)
    return gdino, SamPredictor(sam)


def segment(gdino, sam_pred, img_rgb, prompt, box_thr, text_thr, device):
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
        print("  [SAM] No boxes detected")
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

    print(f"  [SAM] Mask: {mask_all.sum()} pixels ({100*mask_all.mean():.1f}%)")
    return mask_all.astype(np.uint8)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.meta) as f:
        meta = json.load(f)
    K = np.array(meta['intrinsics'][args.cam_idx]['K'], dtype=np.float64)

    images = sorted(glob.glob(os.path.join(args.image_dir, '*.jpg')) +
                    glob.glob(os.path.join(args.image_dir, '*.png')))
    if not images:
        raise RuntimeError(f"No images found in {args.image_dir}")
    print(f"Found {len(images)} images")

    depth_dir = args.image_dir.rstrip('/') + '_depth'
    has_depth = os.path.isdir(depth_dir)
    print(f"Depth: {'DA2 maps from ' + depth_dir if has_depth else f'constant {args.depth_const}m'}")

    def load_depth(img_path, shape):
        if has_depth:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            npy  = os.path.join(depth_dir, f'{stem}.npy')
            if os.path.exists(npy):
                return np.load(npy)
        return np.full(shape[:2], args.depth_const, dtype=np.float32)

    # Load SAM + GroundingDINO BEFORE nvdiffrast to avoid CUDA deadlock
    gdino, sam_pred = load_gdino_sam(args.device)

    print(f"Loading mesh: {args.mesh}")
    loaded = trimesh.load(args.mesh)
    if isinstance(loaded, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
    else:
        mesh = loaded

    # Auto-detect cm meshes (max extent > 0.5 implies cm, not m) and rescale to metres
    max_extent = mesh.bounding_box.extents.max()
    if max_extent > 0.5:
        print(f"  Mesh extents suggest cm units (max={max_extent:.3f}) — rescaling ×0.01 to metres")
        mesh.apply_scale(0.01)
    print(f"  Mesh diameter: {mesh.bounding_box.extents.max():.4f} m")

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    # Load FoundationPose AFTER SAM (nvdiffrast must come last)
    print("Loading FoundationPose...")
    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
    import nvdiffrast.torch as dr
    from Utils import draw_posed_3d_box, draw_xyz_axis

    scorer  = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx   = dr.RasterizeCudaContext()
    os.makedirs('/tmp/fp_offline_debug', exist_ok=True)
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        glctx=glctx,
        debug_dir='/tmp/fp_offline_debug',
        debug=0,
    )
    print("FoundationPose ready.")

    # --- Initialize on init_frame ---
    init_img_path = images[args.init_frame]
    init_bgr      = cv2.imread(init_img_path)
    init_rgb      = cv2.cvtColor(init_bgr, cv2.COLOR_BGR2RGB)
    init_depth    = load_depth(init_img_path, init_rgb.shape)

    print(f"\nSegmenting frame {args.init_frame}: {os.path.basename(init_img_path)}")
    mask = segment(gdino, sam_pred, init_rgb, args.tool_prompt,
                   args.box_threshold, args.text_threshold, args.device)

    if mask.sum() < 100:
        print("ERROR: Segmentation failed on init frame. Try --init_frame or --tool_prompt.")
        return

    # Save mask visualisation
    mask_vis = init_rgb.copy()
    mask_vis[mask.astype(bool)] = (mask_vis[mask.astype(bool)] * 0.4 +
                                    np.array([0, 255, 0]) * 0.6).astype(np.uint8)
    cv2.imwrite(os.path.join(args.output_dir, 'init_mask.jpg'),
                cv2.cvtColor(mask_vis, cv2.COLOR_RGB2BGR))
    print(f"  Mask saved → {args.output_dir}/init_mask.jpg")

    print("Registering initial pose...")
    pose = est.register(K=K, rgb=init_rgb, depth=init_depth,
                        ob_mask=mask, iteration=args.est_refine_iter)
    if pose is None:
        print("ERROR: Registration failed.")
        return
    print(f"  Registered. t={pose[:3,3]*100} cm")

    # --- Track through subsequent frames ---
    start = args.init_frame
    end   = len(images) if args.num_frames == 0 else min(start + args.num_frames, len(images))
    track_frames = images[start:end]

    print(f"\nTracking {len(track_frames)} frames → {args.output_dir}/")

    for i, img_path in enumerate(tqdm(track_frames)):
        bgr   = cv2.imread(img_path)
        rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = load_depth(img_path, rgb.shape)

        if i == 0:
            current_pose = pose  # already registered
        else:
            current_pose = est.track_one(rgb=rgb, depth=depth, K=K,
                                          iteration=args.track_refine_iter)

        vis = rgb.copy()
        if current_pose is not None:
            center_pose = current_pose @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(K, img=vis, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K,
                                thickness=3, transparency=0, is_input_rgb=True)
            t = current_pose[:3, 3]
            cv2.putText(vis, f"x={t[0]*100:.1f} y={t[1]*100:.1f} z={t[2]*100:.1f} cm",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        stem = os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(os.path.join(args.output_dir, f'{stem}_fp.jpg'),
                    cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    print(f"\nDone. Results saved to {args.output_dir}/")
    print(f"  View with:  eog {args.output_dir}/*.jpg")


if __name__ == '__main__':
    main()
