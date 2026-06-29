"""
Step 1 — Record a demonstration episode.

Usage (plain, no preview):
    python 01_record.py --task pastaTransfer2 --episode 001 --duration 15

Usage (with FoundationPose preview):
    python 01_record.py --task pastaTransfer2 --episode 001 --duration 15 \\
        --mesh spoon.obj --tool_prompt "spoon"

A camera preview window is always shown during recording.
If --mesh is supplied, FoundationPose tracks the tool and overlays the 3D
bounding box + axes on cam0.  Registration happens on the table before
recording starts — press SPACE to re-register, ENTER to begin recording.

Output layout:
    data/episodes/<task>/<episode>/
        cam0/  cam1/  ...  camN/          ← JPEG frames (000000.jpg, ...)
        cam0_depth/  cam1_depth/  ...     ← 16-bit PNG depth (000000.png, mm units)
        meta.json                         ← camera serials, fps, frame count, intrinsics
"""

import os, sys, json, time, argparse, queue, threading
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from pipeline_utils.cameras import MultiCamera, get_connected_serials

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
THIRD_PARTY  = os.path.join(PIPELINE_DIR, '..', 'Tool_as_Interface', 'third_party')
FP_DIR       = os.path.join(THIRD_PARTY, 'FoundationPose')
GDINO_WEIGHTS = os.path.join(PIPELINE_DIR, 'checkpoints', 'groundingdino_swint_ogc.pth')
SAM_WEIGHTS   = os.path.join(PIPELINE_DIR, 'checkpoints', 'sam_vit_h_4b8939.pth')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--task',     default='task',  help='Task name (e.g. pastaTransfer2)')
    p.add_argument('--episode',  default='001',   help='Episode ID string')
    p.add_argument('--duration', type=float, default=15.0, help='Recording duration in seconds')
    p.add_argument('--fps',      type=int,   default=30)
    p.add_argument('--width',    type=int,   default=848)
    p.add_argument('--height',   type=int,   default=480)
    p.add_argument('--out',      default=os.path.join(PIPELINE_DIR, 'data', 'episodes'))
    # FoundationPose overlay (optional)
    p.add_argument('--mesh',        default=None, help='Tool mesh (.obj) — enables FP overlay')
    p.add_argument('--tool_prompt', default='tool', help='GroundedSAM prompt for the tool')
    p.add_argument('--box_threshold',  type=float, default=0.3)
    p.add_argument('--text_threshold', type=float, default=0.25)
    p.add_argument('--est_refine_iter',   type=int, default=5)
    p.add_argument('--track_refine_iter', type=int, default=2)
    p.add_argument('--preview_fps', type=int, default=5,
                   help='How often to run FP tracking for the preview (Hz). '
                        'Frames are always saved at --fps regardless.')
    p.add_argument('--track_cam', type=int, default=0,
                   help='Camera index to use for FP tracking preview (default: 0). '
                        'All cameras are always recorded; this only affects the live overlay.')
    p.add_argument('--device', default='cuda')
    return p.parse_args()


# ── FoundationPose helpers ───────────────────────────────────────────────────

def load_gdino_sam(device):
    import groundingdino
    gdino_config = os.path.join(os.path.dirname(groundingdino.__file__),
                                'config', 'GroundingDINO_SwinT_OGC.py')
    from groundingdino.util.inference import load_model as load_gdino
    from segment_anything import sam_model_registry, SamPredictor
    print("  Loading GroundingDINO...")
    gdino = load_gdino(gdino_config, GDINO_WEIGHTS).to(device).eval()
    print("  Loading SAM...")
    sam = sam_model_registry['vit_h'](checkpoint=SAM_WEIGHTS).to(device)
    return gdino, SamPredictor(sam)


def segment_tool(gdino, sam_pred, img_rgb, prompt, box_thr, text_thr, device):
    import torch
    import torchvision.transforms as T   # inside function — avoids cv2 Qt5 deadlock
    from groundingdino.util.inference import predict
    from PIL import Image
    H, W = img_rgb.shape[:2]
    transform = T.Compose([T.Resize(800), T.ToTensor(),
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img_t = transform(Image.fromarray(img_rgb)).to(device)
    with torch.no_grad():
        boxes, logits, _ = predict(gdino, img_t, prompt, box_thr, text_thr, device=device)
    if boxes is None or len(boxes) == 0:
        return np.zeros((H, W), dtype=np.uint8)
    boxes_px = boxes.clone()
    boxes_px[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * W
    boxes_px[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * H
    boxes_px[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * W
    boxes_px[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * H
    sam_pred.set_image(img_rgb)
    mask = np.zeros((H, W), dtype=bool)
    for box in boxes_px:
        m, _, _ = sam_pred.predict(box=box.cpu().numpy(), multimask_output=False)
        mask |= m[0].astype(bool)
    return mask.astype(np.uint8)


def load_fp(mesh_path, device):
    import trimesh, sys as _sys
    _sys.path.insert(0, FP_DIR)
    _sys.path.insert(0, THIRD_PARTY)
    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
    import nvdiffrast.torch as dr
    from Utils import draw_posed_3d_box, draw_xyz_axis

    loaded = trimesh.load(mesh_path)
    mesh = (trimesh.util.concatenate(list(loaded.geometry.values()))
            if isinstance(loaded, trimesh.Scene) else loaded)
    if mesh.bounding_box.extents.max() > 0.5:
        mesh.apply_scale(0.01)
        print("  Mesh rescaled ×0.01 (cm → m)")

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    scorer  = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx   = dr.RasterizeCudaContext()
    os.makedirs('/tmp/fp_record_debug', exist_ok=True)
    est = FoundationPose(
        model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
        scorer=scorer, refiner=refiner, glctx=glctx,
        debug_dir='/tmp/fp_record_debug', debug=0,
    )
    return est, to_origin, bbox, draw_posed_3d_box, draw_xyz_axis


def draw_overlay(vis_bgr, pose, K, to_origin, bbox,
                 draw_posed_3d_box, draw_xyz_axis):
    """Draw FP bounding box + axes onto vis_bgr (in-place, returns modified)."""
    center_pose = pose @ np.linalg.inv(to_origin)
    vis_bgr = draw_posed_3d_box(K, img=vis_bgr, ob_in_cam=center_pose, bbox=bbox)
    vis_bgr = draw_xyz_axis(vis_bgr, ob_in_cam=center_pose, scale=0.08, K=K,
                            thickness=3, transparency=0, is_input_rgb=False)
    return vis_bgr


def put_text(img, text, pos, color=(0, 255, 0), scale=0.65, thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    episode_dir = os.path.join(args.out, args.task, args.episode)
    os.makedirs(episode_dir, exist_ok=True)

    serials = get_connected_serials()
    print(f"Found {len(serials)} camera(s): {serials}")

    cam_dirs, depth_dirs = [], []
    for i in range(len(serials)):
        d = os.path.join(episode_dir, f'cam{i}')
        os.makedirs(d, exist_ok=True)
        cam_dirs.append(d)
        dd = os.path.join(episode_dir, f'cam{i}_depth')
        os.makedirs(dd, exist_ok=True)
        depth_dirs.append(dd)

    use_fp = args.mesh is not None

    # Create cv2 window FIRST — must exist before any CUDA model is loaded
    cv2.namedWindow("Recording", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Recording", 848, 480)

    fp_est = to_origin = bbox = draw_posed_3d_box = draw_xyz_axis_fn = None
    gdino = sam_pred = None
    pose = None
    fp_initialized = False

    if use_fp:
        # Load SAM before FoundationPose/nvdiffrast (CUDA deadlock order)
        print("\nLoading GroundedSAM + FoundationPose for preview overlay...")
        sys.path.insert(0, FP_DIR); sys.path.insert(0, THIRD_PARTY)
        gdino, sam_pred = load_gdino_sam(args.device)
        fp_est, to_origin, bbox, draw_posed_3d_box, draw_xyz_axis_fn = \
            load_fp(args.mesh, args.device)
        print("FoundationPose ready.\n")

    with MultiCamera(serials, resolution=(args.width, args.height), fps=args.fps) as mc:
        print("Warming up cameras (3s)...")
        mc.warmup(90)

        # Sanity-check brightness
        test_frames = mc.grab_all()
        for i, f in enumerate(test_frames):
            mean_brightness = f.mean()
            ok = mean_brightness >= 30
            print(f"  cam{i} brightness: {mean_brightness:.1f}/255  "
                  f"{'OK' if ok else 'WARNING: very dark'}")

        intrinsics = []
        for cam in mc.cameras:
            K_cam, D_cam = cam.get_intrinsics()
            intrinsics.append({'K': K_cam.tolist(), 'D': D_cam.tolist()})
        tc = min(args.track_cam, len(intrinsics) - 1)
        K0 = np.array(intrinsics[tc]['K'])  # track_cam intrinsics for FP projection

        # ── Pre-recording phase: preview + optional FP registration ──────────
        print("\n" + ("─" * 60))
        if use_fp:
            print("PRE-RECORDING:  Place the tool on the table and press SPACE")
            print("                to register FoundationPose.")
            print("                Press ENTER when ready to start recording.")
        else:
            print("PRE-RECORDING:  Check camera view, then press ENTER to record.")
        print("─" * 60 + "\n")

        while True:
            rgbd_frames = mc.grab_all_rgbd()
            rgb0, depth0 = rgbd_frames[0]
            rgb_tc, depth_tc = rgbd_frames[tc]   # track_cam frames for FP
            vis = cv2.cvtColor(rgb_tc.copy(), cv2.COLOR_RGB2BGR)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Aborted.")
                cv2.destroyAllWindows()
                return

            if use_fp:
                if key == ord(' '):
                    print("Segmenting tool...")
                    mask = segment_tool(gdino, sam_pred, rgb_tc, args.tool_prompt,
                                        args.box_threshold, args.text_threshold,
                                        args.device)
                    if mask.sum() < 100:
                        print("  Not detected — reposition tool and press SPACE again.")
                        fp_initialized = False
                    else:
                        pose = fp_est.register(K=K0, rgb=rgb_tc, depth=depth_tc,
                                               ob_mask=mask,
                                               iteration=args.est_refine_iter)
                        fp_initialized = True
                        print("  Registered.")

                if fp_initialized and pose is not None:
                    pose = fp_est.track_one(rgb=rgb_tc, depth=depth_tc, K=K0,
                                            iteration=args.track_refine_iter)
                    vis = draw_overlay(vis, pose, K0, to_origin, bbox,
                                       draw_posed_3d_box, draw_xyz_axis_fn)
                    put_text(vis, "SPACE=re-register  ENTER=start recording  Q=quit",
                             (10, vis.shape[0] - 12), color=(200, 200, 200), scale=0.5)
                else:
                    put_text(vis, "Press SPACE to register tool",
                             (20, 40), color=(0, 200, 255))
                    put_text(vis, "ENTER=start without tracking  Q=quit",
                             (10, vis.shape[0] - 12), color=(200, 200, 200), scale=0.5)
            else:
                put_text(vis, "Press ENTER to start recording  Q=quit",
                         (10, vis.shape[0] - 12), color=(200, 200, 200), scale=0.5)

            cv2.imshow("Recording", vis)

            if key in (13, ord('\r'), ord('\n')):   # ENTER
                break

        # ── Recording loop ───────────────────────────────────────────────────
        n_frames = int(args.duration * args.fps)
        print(f"Recording {args.duration:.0f}s ({n_frames} frames). Perform the task now!\n")

        saved = 0
        t_start = time.time()

        # Background writer: capture loop enqueues raw frames; this thread
        # handles all disk I/O so the camera loop is never blocked by writes.
        write_queue = queue.Queue(maxsize=120)   # ~4s buffer at 30fps

        def _writer():
            while True:
                item = write_queue.get()
                if item is None:   # sentinel — stop
                    break
                tag, rgbd = item
                for cam_i, ((rgb, depth_m), cam_dir, depth_dir) in enumerate(
                        zip(rgbd, cam_dirs, depth_dirs)):
                    cv2.imwrite(os.path.join(cam_dir, f'{tag}.jpg'),
                                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    depth_mm = (depth_m * 1000.0).clip(0, 65535).astype(np.uint16)
                    cv2.imwrite(os.path.join(depth_dir, f'{tag}.png'), depth_mm)
                write_queue.task_done()

        writer_thread = threading.Thread(target=_writer, daemon=True)
        writer_thread.start()

        # FP runs every fp_every frames so it doesn't throttle the capture loop.
        # At 30fps, fp_every=6 → ~5fps tracking updates; preview refreshes every frame.
        fp_every   = max(1, args.fps // args.preview_fps)
        cached_vis = None   # last FP-overlaid frame for display between FP calls

        for i in range(n_frames):
            rgbd_frames = mc.grab_all_rgbd()
            tag = f'{i:06d}'

            # Enqueue for async disk write — never blocks the capture loop
            write_queue.put((tag, rgbd_frames))
            saved += 1

            # Preview: run FP every fp_every frames on track_cam, reuse overlay otherwise
            rgb_tc, depth_tc = rgbd_frames[tc]
            vis = cv2.cvtColor(rgb_tc.copy(), cv2.COLOR_RGB2BGR)

            if use_fp and fp_initialized and pose is not None:
                if i % fp_every == 0:
                    try:
                        pose = fp_est.track_one(rgb=rgb_tc, depth=depth_tc, K=K0,
                                                iteration=args.track_refine_iter)
                        cached_vis = draw_overlay(vis.copy(), pose, K0, to_origin,
                                                  bbox, draw_posed_3d_box,
                                                  draw_xyz_axis_fn)
                    except Exception:
                        pass
                if cached_vis is not None:
                    vis = cached_vis.copy()

            elapsed = time.time() - t_start
            actual_fps = saved / elapsed if elapsed > 0 else 0
            put_text(vis, f"REC  {elapsed:.1f}s / {args.duration:.0f}s  "
                          f"frame {saved}/{n_frames}  {actual_fps:.1f}fps",
                     (10, 30), color=(0, 60, 255))
            hint = "SPACE=re-register  Q=stop early" if (use_fp and fp_initialized) \
                   else "Q=stop early"
            put_text(vis, hint, (10, vis.shape[0] - 12),
                     color=(180, 180, 180), scale=0.5)

            cv2.imshow("Recording", vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Stopped early by user.")
                break

            if use_fp and key == ord(' ') and fp_initialized:
                print(f"  Re-registering at frame {i}...")
                mask = segment_tool(gdino, sam_pred, rgb_tc, args.tool_prompt,
                                    args.box_threshold, args.text_threshold,
                                    args.device)
                if mask.sum() >= 100:
                    pose = fp_est.register(K=K0, rgb=rgb_tc, depth=depth_tc,
                                           ob_mask=mask,
                                           iteration=args.est_refine_iter)
                    cached_vis = None
                    print("  Re-registered.")

            if i % args.fps == 0:
                print(f"  {elapsed:.1f}s / {args.duration:.0f}s  "
                      f"({saved} frames, {actual_fps:.1f} fps)")

    cv2.destroyAllWindows()

    # Flush remaining frames to disk before writing meta.json
    print("Flushing remaining frames to disk...", end='', flush=True)
    write_queue.put(None)   # sentinel
    writer_thread.join()
    print(" done.")

    meta = {
        'task': args.task,
        'episode': args.episode,
        'serials': serials,
        'fps': args.fps,
        'resolution': [args.width, args.height],
        'n_frames': saved,
        'duration_sec': args.duration,
        'intrinsics': intrinsics,
        'has_depth': True,
        'depth_scale': 0.001,
        'recorded_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(episode_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved {saved} frames to '{episode_dir}/'")
    print(f"Next: python 02_augment_noposplat.py --episode_dir {episode_dir} "
          f"--noposplat_root ~/working_dir/NoPoSplat --input_mode letterbox "
          f"--no_antialias --sample_every 1 --num_novel_views 6")


if __name__ == '__main__':
    main()
