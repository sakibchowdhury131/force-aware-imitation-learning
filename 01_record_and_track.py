"""
Record an episode (same layout as 01_record.py) and simultaneously track the
tool with FoundationPose in a background thread, so steps 1 + 4 + 4c are done
in a single pass.

Usage:
    python 01_record_and_track.py \\
        --task pastaTransfer4 --episode 001 --duration 15 \\
        --mesh spoon.obj --tool_prompt "spoon" \\
        --track_cam 0 --subsample 3

Controls:
    SPACE  — segment + register tool (pre-recording) or queue re-registration (recording)
    ENTER  — start recording
    Q      — quit / stop early

Output:
    data/episodes/<task>/<episode>/
        cam0/  cam1/  ...           <- JPEG frames (all cameras, full fps)
        cam0_depth/  ...            <- 16-bit PNG depth (mm)
        meta.json
        augmented/
            tool_poses_cam{N}.npz   <- FP poses in cam frame     (every --subsample frame)
            tool_poses_task.npz     <- poses in task/world frame  (requires --task_frame)
            tool_poses_base.npz     <- poses in robot-base frame  (requires --robot_extrinsics)
            track_vis/              <- per-frame overlay JPEGs
            track_vis.mp4           <- overlay video at 10 fps
"""

import os, sys, json, time, argparse, queue, threading
import numpy as np
import cv2
import torch

PIPELINE_DIR  = os.path.dirname(os.path.abspath(__file__))
THIRD_PARTY   = os.path.join(PIPELINE_DIR, '..', 'Tool_as_Interface', 'third_party')
FP_DIR        = os.path.join(THIRD_PARTY, 'FoundationPose')
GDINO_WEIGHTS = os.path.join(PIPELINE_DIR, 'checkpoints', 'groundingdino_swint_ogc.pth')
SAM_WEIGHTS   = os.path.join(PIPELINE_DIR, 'checkpoints', 'sam_vit_h_4b8939.pth')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--task',     required=True,    help='Task name, e.g. pastaTransfer4')
    p.add_argument('--episode',  default='001',    help='Episode ID string')
    p.add_argument('--duration', type=float, default=15.0, help='Recording duration (s)')
    p.add_argument('--fps',      type=int,   default=30)
    p.add_argument('--width',    type=int,   default=848)
    p.add_argument('--height',   type=int,   default=480)
    p.add_argument('--out', default=os.path.join(PIPELINE_DIR, 'data', 'episodes'))
    # Tracking
    p.add_argument('--mesh',        required=True,   help='Tool mesh (.obj/.ply)')
    p.add_argument('--tool_prompt', default='spoon', help='GroundedSAM text prompt')
    p.add_argument('--track_cam',   type=int, default=0,
                   help='Camera index used for FP tracking (default: 0)')
    p.add_argument('--subsample',   type=int, default=3,
                   help='Track every N-th frame (default 3 → 10 Hz @ 30 fps). '
                        'Must match --subsample in 05_train.py.')
    p.add_argument('--box_threshold',     type=float, default=0.3)
    p.add_argument('--text_threshold',    type=float, default=0.25)
    p.add_argument('--est_refine_iter',   type=int,   default=5)
    p.add_argument('--track_refine_iter', type=int,   default=2)
    p.add_argument('--task_frame', default=None,
                   help='cam_extrinsics.npy from 00_calibrate.py. '
                        'Defaults to data/cam_extrinsics.npy (cam0) or '
                        'data/cam{N}_extrinsics.npy.')
    p.add_argument('--robot_extrinsics', default='data/robot_extrinsics.npy',
                   help='T_base_task from 06_calibrate_robot.py (for base-frame poses)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


# ── Model helpers ─────────────────────────────────────────────────────────────

def load_gdino_sam(device):
    import groundingdino
    from groundingdino.util.inference import load_model as load_gdino
    from segment_anything import sam_model_registry, SamPredictor
    gdino_config = os.path.join(os.path.dirname(groundingdino.__file__),
                                'config', 'GroundingDINO_SwinT_OGC.py')
    print("  Loading GroundingDINO...")
    gdino = load_gdino(gdino_config, GDINO_WEIGHTS).to(device).eval()
    print("  Loading SAM...")
    sam = sam_model_registry['vit_h'](checkpoint=SAM_WEIGHTS).to(device)
    return gdino, SamPredictor(sam)


def segment_tool(gdino, sam_pred, img_rgb, prompt, box_thr, text_thr, device):
    import torchvision.transforms as T   # deferred import — avoids cv2/Qt5 deadlock
    from groundingdino.util.inference import predict
    from PIL import Image
    H, W = img_rgb.shape[:2]
    transform = T.Compose([T.Resize(800), T.ToTensor(),
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
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
    mask = np.zeros((H, W), dtype=bool)
    for box in boxes_px:
        m, _, _ = sam_pred.predict(box=box.cpu().numpy(), multimask_output=False)
        mask |= m[0].astype(bool)
    return mask.astype(np.uint8)


def load_fp(mesh_path):
    import trimesh
    sys.path.insert(0, FP_DIR)
    sys.path.insert(0, THIRD_PARTY)
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
    os.makedirs('/tmp/fp_rt_debug', exist_ok=True)
    est = FoundationPose(
        model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
        scorer=scorer, refiner=refiner, glctx=glctx,
        debug_dir='/tmp/fp_rt_debug', debug=0,
    )
    return est, to_origin, bbox, draw_posed_3d_box, draw_xyz_axis


def draw_overlay(vis_bgr, pose, K, to_origin, bbox, draw_posed_3d_box_fn, draw_xyz_axis_fn):
    center_pose = pose @ np.linalg.inv(to_origin)
    vis_bgr = draw_posed_3d_box_fn(K, img=vis_bgr, ob_in_cam=center_pose, bbox=bbox)
    vis_bgr = draw_xyz_axis_fn(vis_bgr, ob_in_cam=center_pose, scale=0.08, K=K,
                               thickness=3, transparency=0, is_input_rgb=False)
    return vis_bgr


def put_text(img, text, pos, color=(0, 255, 0), scale=0.65, thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


# ── Post-recording visualization ──────────────────────────────────────────────

def save_vis_video(episode_dir, cam_idx, K, fp_poses, to_origin, bbox,
                   draw_posed_3d_box_fn, draw_xyz_axis_fn, aug_dir, vis_fps=10):
    cam_dir = os.path.join(episode_dir, f'cam{cam_idx}')
    vis_dir = os.path.join(aug_dir, 'track_vis')
    os.makedirs(vis_dir, exist_ok=True)

    sorted_frames = sorted(fp_poses.keys())
    sample = cv2.imread(os.path.join(cam_dir, f'{sorted_frames[0]:06d}.jpg'))
    if sample is None:
        print("WARNING: could not read sample frame — skipping visualization")
        return
    h, w = sample.shape[:2]

    video_path = os.path.join(aug_dir, 'track_vis.mp4')
    vw = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), vis_fps, (w, h))

    print(f"Saving visualization ({len(sorted_frames)} frames)...", end='', flush=True)
    for fi in sorted_frames:
        img_path = os.path.join(cam_dir, f'{fi:06d}.jpg')
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        try:
            frame = draw_overlay(frame, fp_poses[fi], K, to_origin, bbox,
                                 draw_posed_3d_box_fn, draw_xyz_axis_fn)
        except Exception:
            pass
        cv2.imwrite(os.path.join(vis_dir, f'{fi:06d}.jpg'), frame)
        vw.write(frame)
    vw.release()
    print(f" done → {video_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    sys.path.insert(0, PIPELINE_DIR)

    if args.task_frame is None:
        args.task_frame = ('data/cam_extrinsics.npy' if args.track_cam == 0
                           else f'data/cam{args.track_cam}_extrinsics.npy')

    episode_dir = os.path.join(args.out, args.task, args.episode)
    aug_dir     = os.path.join(episode_dir, 'augmented')
    os.makedirs(aug_dir, exist_ok=True)

    from pipeline_utils.cameras import MultiCamera, get_connected_serials
    serials = get_connected_serials()
    print(f"Found {len(serials)} camera(s): {serials}")

    cam_dirs, depth_dirs = [], []
    for i in range(len(serials)):
        d = os.path.join(episode_dir, f'cam{i}');   os.makedirs(d, exist_ok=True);  cam_dirs.append(d)
        dd = os.path.join(episode_dir, f'cam{i}_depth'); os.makedirs(dd, exist_ok=True); depth_dirs.append(dd)

    # Open cv2 window FIRST — must exist before any CUDA model is loaded
    cv2.namedWindow("Record+Track", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Record+Track", 848, 480)

    # Load SAM before FoundationPose/nvdiffrast (avoids CUDA deadlock on Linux/NVIDIA)
    print("\nLoading GroundedSAM + FoundationPose...")
    gdino, sam_pred = load_gdino_sam(args.device)
    fp_est, to_origin, bbox, draw_posed_3d_box, draw_xyz_axis_fn = load_fp(args.mesh)
    print("Models ready.\n")

    with MultiCamera(serials, resolution=(args.width, args.height), fps=args.fps) as mc:
        print("Warming up cameras (3s)...")
        mc.warmup(90)

        intrinsics = []
        for cam in mc.cameras:
            K_cam, D_cam = cam.get_intrinsics()
            intrinsics.append({'K': K_cam.tolist(), 'D': D_cam.tolist()})
        tc   = min(args.track_cam, len(intrinsics) - 1)
        K_tc = np.array(intrinsics[tc]['K'])

        fp_initialized = False
        pose           = None

        # ── Pre-recording: register tool ─────────────────────────────────────
        print("─" * 60)
        print("PRE-RECORDING:")
        print("  SPACE — segment + register tool")
        print("  ENTER — start recording  (can start without registration)")
        print("  Q     — quit")
        print("─" * 60 + "\n")

        while True:
            rgbd_frames       = mc.grab_all_rgbd()
            rgb_tc, depth_tc  = rgbd_frames[tc]
            vis = cv2.cvtColor(rgb_tc.copy(), cv2.COLOR_RGB2BGR)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                cv2.destroyAllWindows()
                return

            if key == ord(' '):
                print("Segmenting...", end='', flush=True)
                mask = segment_tool(gdino, sam_pred, rgb_tc, args.tool_prompt,
                                    args.box_threshold, args.text_threshold, args.device)
                if mask.sum() < 100:
                    print(" not detected — reposition and try again.")
                    fp_initialized = False
                else:
                    pose = fp_est.register(K=K_tc, rgb=rgb_tc, depth=depth_tc,
                                           ob_mask=mask, iteration=args.est_refine_iter)
                    fp_initialized = True
                    print(f" registered ({mask.sum()} px).")

            if fp_initialized and pose is not None:
                try:
                    pose = fp_est.track_one(rgb=rgb_tc, depth=depth_tc, K=K_tc,
                                            iteration=args.track_refine_iter)
                    vis  = draw_overlay(vis, pose, K_tc, to_origin, bbox,
                                        draw_posed_3d_box, draw_xyz_axis_fn)
                except Exception:
                    pass
                put_text(vis, "SPACE=re-register  ENTER=start  Q=quit",
                         (10, vis.shape[0] - 12), color=(200, 200, 200), scale=0.5)
            else:
                put_text(vis, "Press SPACE to register tool", (20, 40), color=(0, 200, 255))
                put_text(vis, "ENTER=start without tracking  Q=quit",
                         (10, vis.shape[0] - 12), color=(200, 200, 200), scale=0.5)

            cv2.imshow("Record+Track", vis)
            if key in (13, ord('\r'), ord('\n')):
                break

        # ── Queues and background threads ─────────────────────────────────────
        n_frames    = int(args.duration * args.fps)
        write_queue = queue.Queue(maxsize=120)
        fp_queue    = queue.Queue()

        # fp_poses written exclusively by tracker thread; read by main after join
        fp_poses    = {}
        latest_pose = [pose.copy() if (fp_initialized and pose is not None) else None]

        def _writer():
            while True:
                item = write_queue.get()
                if item is None:
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

        fp_done_evt = threading.Event()

        def _tracker():
            last_pose    = (pose.copy() if (fp_initialized and pose is not None)
                            else np.eye(4, dtype=np.float32))
            needs_init   = not fp_initialized

            while True:
                item = fp_queue.get()
                if item is None:
                    fp_done_evt.set()
                    break
                item_type, frame_idx, rgb, depth = item
                try:
                    if item_type == 'register' or needs_init:
                        action = 'Re-registering' if item_type == 'register' else 'Auto-registering'
                        print(f"\n  [FP] {action} at frame {frame_idx}...", end='', flush=True)
                        mask = segment_tool(gdino, sam_pred, rgb, args.tool_prompt,
                                            args.box_threshold, args.text_threshold, args.device)
                        if mask.sum() >= 100:
                            p = fp_est.register(K=K_tc, rgb=rgb, depth=depth,
                                                ob_mask=mask, iteration=args.est_refine_iter)
                            if p is not None:
                                last_pose  = p
                                needs_init = False
                                print(f" done.")
                        else:
                            print(f" segmentation failed.")
                    else:
                        p = fp_est.track_one(rgb=rgb, depth=depth, K=K_tc,
                                             iteration=args.track_refine_iter)
                        if p is not None:
                            last_pose = p
                except Exception as e:
                    print(f"\n  [FP] frame {frame_idx} error: {e}")

                fp_poses[frame_idx] = last_pose.astype(np.float32).copy()
                latest_pose[0]      = fp_poses[frame_idx]
                fp_queue.task_done()

        writer_thread  = threading.Thread(target=_writer,  daemon=True)
        tracker_thread = threading.Thread(target=_tracker, daemon=True)
        writer_thread.start()
        tracker_thread.start()

        # ── Recording loop ────────────────────────────────────────────────────
        saved   = 0
        t_start = time.time()
        print(f"Recording {args.duration:.0f}s ({n_frames} frames). Perform the task!\n")

        for i in range(n_frames):
            rgbd_frames      = mc.grab_all_rgbd()
            tag              = f'{i:06d}'
            write_queue.put((tag, rgbd_frames))
            saved += 1

            rgb_tc, depth_tc = rgbd_frames[tc]

            if i % args.subsample == 0:
                fp_queue.put(('track', i, rgb_tc.copy(), depth_tc.copy()))

            # Display using the most recent pose from the tracker thread
            vis = cv2.cvtColor(rgb_tc.copy(), cv2.COLOR_RGB2BGR)
            lp  = latest_pose[0]
            if lp is not None:
                try:
                    vis = draw_overlay(vis, lp, K_tc, to_origin, bbox,
                                       draw_posed_3d_box, draw_xyz_axis_fn)
                except Exception:
                    pass

            elapsed    = time.time() - t_start
            actual_fps = saved / elapsed if elapsed > 0 else 0
            put_text(vis, f"REC  {elapsed:.1f}s/{args.duration:.0f}s  "
                          f"frame {saved}/{n_frames}  {actual_fps:.1f}fps  "
                          f"FP queue: {fp_queue.qsize()}",
                     (10, 30), color=(0, 60, 255))
            put_text(vis, "SPACE=re-register (async)  Q=stop early",
                     (10, vis.shape[0] - 12), color=(180, 180, 180), scale=0.5)

            cv2.imshow("Record+Track", vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Stopped early by user.")
                break

            if key == ord(' '):
                print(f"  Re-registration queued at frame {i}.")
                fp_queue.put(('register', i, rgb_tc.copy(), depth_tc.copy()))

            if i % args.fps == 0 and i > 0:
                print(f"  {elapsed:.1f}s/{args.duration:.0f}s  "
                      f"({saved} frames, {actual_fps:.1f}fps, FP queue: {fp_queue.qsize()})")

    cv2.destroyAllWindows()

    # Flush disk writer
    print("Flushing frames to disk...", end='', flush=True)
    write_queue.put(None)
    writer_thread.join()
    print(" done.")

    # Drain FP tracker (may have a queue built up if FP was slower than subsample rate)
    remaining = fp_queue.qsize()
    if remaining > 0:
        print(f"Processing {remaining} remaining FP frames (may take a moment)...")
    fp_queue.put(None)
    fp_done_evt.wait()
    tracker_thread.join()

    # ── Save poses ────────────────────────────────────────────────────────────
    if not fp_poses:
        print("WARNING: no FP poses saved (tool was never registered).")
    else:
        sorted_items = sorted(fp_poses.items())

        # Cam-frame poses
        cam_out = os.path.join(aug_dir, f'tool_poses_cam{tc}.npz')
        np.savez(cam_out, **{str(k): v for k, v in sorted_items})
        print(f"Saved {len(fp_poses)} cam-frame poses → {cam_out}")

        # Task/world-frame poses
        if os.path.exists(args.task_frame):
            tf_world2cam = np.load(args.task_frame)
            tf_cam2world = np.linalg.inv(tf_world2cam)
            task_poses   = {k: (tf_cam2world @ v.astype(np.float64)).astype(np.float32)
                            for k, v in fp_poses.items()}
            task_out = os.path.join(aug_dir, 'tool_poses_task.npz')
            np.savez(task_out, **{str(k): v for k, v in sorted(task_poses.items())})
            print(f"Saved task-frame poses     → {task_out}")

            # Robot-base-frame poses (requires robot_extrinsics.npy)
            if os.path.exists(args.robot_extrinsics):
                T_base_task = np.load(args.robot_extrinsics).astype(np.float64)
                base_poses  = {k: (T_base_task @ v.astype(np.float64)).astype(np.float32)
                               for k, v in task_poses.items()}
                base_out = os.path.join(aug_dir, 'tool_poses_base.npz')
                np.savez(base_out, **{str(k): v for k, v in sorted(base_poses.items())})
                print(f"Saved robot-base poses     → {base_out}")
            else:
                print(f"NOTE: {args.robot_extrinsics} not found — skipping base-frame poses")
        else:
            print(f"NOTE: {args.task_frame} not found — skipping task/base-frame poses")

        # Visualization video
        save_vis_video(episode_dir, tc, K_tc, fp_poses,
                       to_origin, bbox, draw_posed_3d_box, draw_xyz_axis_fn, aug_dir)

    # ── meta.json (identical format to 01_record.py) ─────────────────────────
    meta = {
        'task':         args.task,
        'episode':      args.episode,
        'serials':      serials,
        'fps':          args.fps,
        'resolution':   [args.width, args.height],
        'n_frames':     saved,
        'duration_sec': args.duration,
        'intrinsics':   intrinsics,
        'has_depth':    True,
        'depth_scale':  0.001,
        'recorded_at':  time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(episode_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    n_tracked = len(fp_poses)
    print(f"\nDone: {saved} frames + {n_tracked} tracked poses → {episode_dir}/")
    if n_tracked:
        print(f"Visualization: {os.path.join(aug_dir, 'track_vis.mp4')}")
    print(f"\nNext:")
    print(f"  python 05_train.py --data_dir data/episodes/{args.task} "
          f"--track_cam {tc} --subsample {args.subsample}")


if __name__ == '__main__':
    main()
