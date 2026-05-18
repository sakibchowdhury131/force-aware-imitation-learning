"""
Live FoundationPose demo using a RealSense D435i camera.

Segments the tool on the first frame using GroundedSAM, then tracks its
6DOF pose in real-time using FoundationPose.

Usage:
    python live_foundationpose.py --mesh Hammer.obj --tool_prompt "hammer"
    python live_foundationpose.py --mesh Paddle.obj --tool_prompt "table tennis paddle" --camera 1

Press:
    SPACE  — re-initialize pose (re-segment + register on current frame)
    Q      — quit
"""

import os, sys, argparse
import numpy as np
import cv2
import torch
import trimesh
import pyrealsense2 as rs
from PIL import Image
# torchvision.transforms and groundingdino are imported inside functions —
# importing them at module level deadlocks cv2.namedWindow on Linux/NVIDIA (Qt5 conflict).

PIPELINE_DIR   = os.path.dirname(os.path.abspath(__file__))
THIRD_PARTY    = os.path.join(PIPELINE_DIR, '..', 'Tool_as_Interface', 'third_party')
FP_DIR         = os.path.join(THIRD_PARTY, 'FoundationPose')
GDINO_WEIGHTS  = os.path.join(PIPELINE_DIR, 'checkpoints', 'groundingdino_swint_ogc.pth')
SAM_WEIGHTS    = os.path.join(PIPELINE_DIR, 'checkpoints', 'sam_vit_h_4b8939.pth')

# FP_DIR for estimater.py/datareader.py/Utils.py; THIRD_PARTY for `from FoundationPose.X import *`
sys.path.insert(0, FP_DIR)
sys.path.insert(0, THIRD_PARTY)

# groundingdino is imported inside load_gdino_sam() — importing it at module
# level causes cv2.namedWindow to deadlock on Linux/NVIDIA (Qt5 + GDINO conflict).


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mesh', required=True, help='Path to tool mesh (.obj or .ply)')
    p.add_argument('--tool_prompt', default='hammer', help='Text prompt for segmentation')
    p.add_argument('--camera', type=int, default=0, help='Camera index (0=first, 1=second, ...)')
    p.add_argument('--box_threshold',  type=float, default=0.3)
    p.add_argument('--text_threshold', type=float, default=0.25)
    p.add_argument('--est_refine_iter',   type=int, default=5)
    p.add_argument('--track_refine_iter', type=int, default=2)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def load_gdino_sam(device):
    import groundingdino
    gdino_config = os.path.join(os.path.dirname(groundingdino.__file__),
                                'config', 'GroundingDINO_SwinT_OGC.py')
    from groundingdino.util.inference import load_model as load_gdino
    from segment_anything import sam_model_registry, SamPredictor
    print("Loading GroundingDINO...")
    gdino = load_gdino(gdino_config, GDINO_WEIGHTS).to(device).eval()
    print("Loading SAM...")
    sam = sam_model_registry['vit_h'](checkpoint=SAM_WEIGHTS).to(device)
    return gdino, SamPredictor(sam)


def segment(gdino, sam_pred, img_rgb, prompt, box_thr, text_thr, device):
    import torchvision.transforms as T
    from groundingdino.util.inference import predict
    H, W = img_rgb.shape[:2]
    transform = T.Compose([
        T.Resize(800), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img_t = transform(Image.fromarray(img_rgb)).to(device)

    with torch.no_grad():
        boxes, logits, _ = predict(gdino, img_t, prompt, box_thr, text_thr, device=device)

    if boxes is None or len(boxes) == 0:
        print("  [SAM] No boxes detected — try a different prompt or threshold")
        return np.zeros((H, W), dtype=np.uint8)

    # Convert cx,cy,w,h (normalized) → x1,y1,x2,y2 (pixel)
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


def start_realsense(camera_index=0):
    """Start a single RealSense pipeline with aligned depth."""
    ctx     = rs.context()
    devices = ctx.query_devices()
    n       = len(devices)
    if n == 0:
        raise RuntimeError("No RealSense devices found.")
    if camera_index >= n:
        raise RuntimeError(f"--camera {camera_index} requested but only {n} device(s) connected.")
    serial = devices[camera_index].get_info(rs.camera_info.serial_number)
    print(f"Using camera {camera_index}: serial {serial}  ({n} device(s) connected)")

    pipe = rs.pipeline()
    cfg  = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)
    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    profile = pipe.start(cfg)

    # Intrinsics
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0, 0, 1]], dtype=np.float64)

    # Depth scale (converts raw uint16 → meters)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale  = depth_sensor.get_depth_scale()

    # Auto-exposure
    sensor = profile.get_device().first_color_sensor()
    sensor.set_option(rs.option.enable_auto_exposure, 1)
    sensor.set_option(rs.option.enable_auto_white_balance, 1)

    align = rs.align(rs.stream.color)

    print("Warming up camera (90 frames)...")
    for _ in range(90):
        pipe.wait_for_frames()

    return pipe, align, K, depth_scale


def capture(pipe, align, depth_scale):
    """Returns (rgb uint8 HxWx3, depth float32 HxW in metres)."""
    frames  = align.process(pipe.wait_for_frames(timeout_ms=3000))
    color   = np.asanyarray(frames.get_color_frame().get_data())   # HxWx3 RGB
    depth_raw = np.asanyarray(frames.get_depth_frame().get_data()) # HxW uint16
    depth   = depth_raw.astype(np.float32) * depth_scale           # metres
    return color, depth


def loading_screen(msg, line=1):
    """Show a status message in the window during model loading."""
    canvas = np.zeros((480, 848, 3), dtype=np.uint8)
    cv2.putText(canvas, "FoundationPose Live — Loading...",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    cv2.putText(canvas, msg,
                (20, 50 + line * 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 100), 2)
    cv2.imshow("FoundationPose Live", canvas)
    cv2.waitKey(1)


def main():
    args = parse_args()

    # Create cv2 window FIRST — OpenGL context must exist before CUDA/nvdiffrast.
    # groundingdino must NOT be imported before this point (causes Qt5 deadlock).
    cv2.namedWindow("FoundationPose Live", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("FoundationPose Live", 848, 480)
    loading_screen("Starting up...")

    # Load GroundingDINO + SAM before nvdiffrast/FoundationPose to avoid CUDA deadlock.
    loading_screen("Loading GroundingDINO + SAM  (30-60s)...")
    gdino, sam_pred = load_gdino_sam(args.device)

    loading_screen("Loading mesh...", line=2)
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

    # Load FoundationPose AFTER SAM (nvdiffrast CUDA context must come last)
    loading_screen("Loading FoundationPose...", line=3)
    print("Loading FoundationPose...")
    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
    import nvdiffrast.torch as dr
    from Utils import draw_posed_3d_box, draw_xyz_axis

    scorer  = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx   = dr.RasterizeCudaContext()
    os.makedirs('/tmp/fp_live_debug', exist_ok=True)
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        glctx=glctx,
        debug_dir='/tmp/fp_live_debug',
        debug=0,
    )
    print("FoundationPose ready.")

    loading_screen("Starting camera...", line=4)
    pipe, align, K, depth_scale = start_realsense(args.camera)

    pose = None
    initialized = False

    print("\nPress SPACE to initialize/re-initialize, Q to quit.")

    try:
        while True:
            rgb, depth = capture(pipe, align, depth_scale)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            if key == ord(' ') or not initialized:
                print("\nSegmenting tool...")
                mask = segment(gdino, sam_pred, rgb, args.tool_prompt,
                               args.box_threshold, args.text_threshold, args.device)
                if mask.sum() < 100:
                    print("  Segmentation failed — hold tool in view and press SPACE again")
                    vis = rgb.copy()
                    cv2.putText(vis, "Segmentation failed — press SPACE", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    cv2.imshow("FoundationPose Live", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                    continue

                # Show mask overlay briefly
                overlay = rgb.copy()
                overlay[mask.astype(bool)] = (overlay[mask.astype(bool)] * 0.4 +
                                               np.array([0, 255, 0]) * 0.6).astype(np.uint8)
                cv2.imshow("FoundationPose Live", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                cv2.waitKey(500)

                print("Registering initial pose...")
                pose = est.register(
                    K=K,
                    rgb=rgb,
                    depth=depth,
                    ob_mask=mask,
                    iteration=args.est_refine_iter,
                )
                initialized = (pose is not None)
                if initialized:
                    print("  Pose registered.")
                else:
                    print("  Registration failed.")
                continue

            if initialized and pose is not None:
                pose = est.track_one(
                    rgb=rgb,
                    depth=depth,
                    K=K,
                    iteration=args.track_refine_iter,
                )

            vis = rgb.copy()
            if initialized and pose is not None:
                center_pose = pose @ np.linalg.inv(to_origin)
                vis = draw_posed_3d_box(K, img=vis, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K,
                                    thickness=3, transparency=0, is_input_rgb=True)
                # Print translation in top-left
                t = pose[:3, 3]
                txt = f"x={t[0]*100:.1f}  y={t[1]*100:.1f}  z={t[2]*100:.1f} cm"
                cv2.putText(vis, txt, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(vis, "Press SPACE to initialize", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 0), 2)

            cv2.imshow("FoundationPose Live", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    finally:
        pipe.stop()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == '__main__':
    main()
