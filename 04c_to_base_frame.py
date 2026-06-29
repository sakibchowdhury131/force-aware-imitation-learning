"""
Step 4c — Transform task/world-frame tool poses into the robot-base frame.

Reads tool_poses_task.npz (saved by 04b_to_task_frame.py) and T_base_task
from 06_calibrate_robot.py, and writes tool_poses_base.npz alongside it.
Does not re-run FoundationPose.

Training on robot-base-frame poses matches the original Tool-as-Interface
approach: the policy's proprioception/action live in a frame that is fixed
relative to the robot (not the camera), so camera recalibration after data
collection cannot introduce a train/deploy frame mismatch.

Usage (single episode):
    python 04c_to_base_frame.py --episode_dir data/episodes/pastaTransfer/001 \\
        --robot_extrinsics data/robot_extrinsics.npy

Usage (all episodes in a task):
    python 04c_to_base_frame.py --task_dir data/episodes/pastaTransfer \\
        --robot_extrinsics data/robot_extrinsics.npy

Output (per episode):
    data/episodes/<task>/<episode>/augmented/tool_poses_base.npz
        dict: str(frame_id) -> (4, 4) pose matrix in robot-base frame
"""

import os, argparse
import numpy as np


def collect_episodes(task_dir):
    return [os.path.join(task_dir, name)
            for name in sorted(os.listdir(task_dir))
            if os.path.isdir(os.path.join(task_dir, name))
            and os.path.exists(os.path.join(task_dir, name, 'meta.json'))]


def process_episode(episode_dir, T_base_task, skip_done):
    aug_dir   = os.path.join(episode_dir, 'augmented')
    task_path = os.path.join(aug_dir, 'tool_poses_task.npz')
    out_path  = os.path.join(aug_dir, 'tool_poses_base.npz')

    if not os.path.exists(task_path):
        print(f"  Skipping {episode_dir} — no tool_poses_task.npz")
        return False
    if skip_done and os.path.exists(out_path):
        print(f"  Skipping {episode_dir} — already done (tool_poses_base.npz exists)")
        return False

    task_poses = dict(np.load(task_path))
    base_poses = {k: (T_base_task @ v.astype(np.float64)).astype(np.float32)
                   for k, v in task_poses.items()}
    np.savez(out_path, **{str(k): v for k, v in base_poses.items()})
    print(f"  Saved {len(base_poses)} base-frame poses -> {out_path}")
    return True


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--episode_dir', help='Single episode to process')
    src.add_argument('--task_dir',    help='Task directory; processes all episodes')
    p.add_argument('--robot_extrinsics', default='data/robot_extrinsics.npy',
                   help='Path to T_base_task (4x4) from 06_calibrate_robot.py')
    p.add_argument('--skip_done', action='store_true',
                   help='Skip episodes that already have tool_poses_base.npz')
    args = p.parse_args()

    T_base_task = np.load(args.robot_extrinsics).astype(np.float64)

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
        ok = process_episode(episode_dir, T_base_task, args.skip_done)
        if ok:
            n_done += 1
        else:
            n_skip += 1

    print(f"\nDone ({n_done} processed, {n_skip} skipped)")


if __name__ == '__main__':
    main()
