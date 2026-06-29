"""
Step 4b — Transform existing cam0-frame tool poses into the task/world frame.

Reads tool_poses_cam0.npz (saved by 04_track.py) and the world-to-cam0
extrinsics from 00_calibrate.py, and writes tool_poses_task.npz alongside it.
Does not re-run FoundationPose.

Usage (single episode):
    python 04b_to_task_frame.py --episode_dir data/episodes/pastaTransfer/001 \\
        --task_frame data/cam_extrinsics.npy

Usage (all episodes in a task):
    python 04b_to_task_frame.py --task_dir data/episodes/pastaTransfer \\
        --task_frame data/cam_extrinsics.npy

Output (per episode):
    data/episodes/<task>/<episode>/augmented/tool_poses_task.npz
        dict: str(frame_id) → (4, 4) pose matrix in task/world frame
"""

import os, argparse
import numpy as np


def collect_episodes(task_dir):
    return [os.path.join(task_dir, name)
            for name in sorted(os.listdir(task_dir))
            if os.path.isdir(os.path.join(task_dir, name))
            and os.path.exists(os.path.join(task_dir, name, 'meta.json'))]


def process_episode(episode_dir, tf_cam2world, skip_done):
    aug_dir = os.path.join(episode_dir, 'augmented')
    cam0_path = os.path.join(aug_dir, 'tool_poses_cam0.npz')
    out_path  = os.path.join(aug_dir, 'tool_poses_task.npz')

    if not os.path.exists(cam0_path):
        print(f"  Skipping {episode_dir} — no tool_poses_cam0.npz")
        return False
    if skip_done and os.path.exists(out_path):
        print(f"  Skipping {episode_dir} — already done (tool_poses_task.npz exists)")
        return False

    cam_poses = dict(np.load(cam0_path))
    task_poses = {k: (tf_cam2world @ v.astype(np.float64)).astype(np.float32)
                   for k, v in cam_poses.items()}
    np.savez(out_path, **{str(k): v for k, v in task_poses.items()})
    print(f"  Saved {len(task_poses)} task-frame poses -> {out_path}")
    return True


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--episode_dir', help='Single episode to process')
    src.add_argument('--task_dir',    help='Task directory; processes all episodes')
    p.add_argument('--task_frame', required=True,
                   help='Path to cam_extrinsics.npy (world-to-cam0, 4x4) from 00_calibrate.py')
    p.add_argument('--skip_done', action='store_true',
                   help='Skip episodes that already have tool_poses_task.npz')
    args = p.parse_args()

    tf_world2cam = np.load(args.task_frame)
    tf_cam2world = np.linalg.inv(tf_world2cam)

    if args.episode_dir:
        episode_dirs = [args.episode_dir]
    else:
        episode_dirs = collect_episodes(args.task_dir)
        if not episode_dirs:
            print(f"No episodes found under {args.task_dir}")
            return
        print(f"Found {len(episode_dirs)} episode(s) under {args.task_dir}")

    n_done = n_skip = 0
    for episode_dir in episode_dirs:
        ep_name = os.path.relpath(episode_dir, args.task_dir) if args.task_dir else episode_dir
        print(f"Episode {ep_name}")
        ok = process_episode(episode_dir, tf_cam2world, args.skip_done)
        if ok:
            n_done += 1
        else:
            n_skip += 1

    print(f"\nDone ({n_done} processed, {n_skip} skipped)")


if __name__ == '__main__':
    main()
