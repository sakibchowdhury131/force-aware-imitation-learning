#!/usr/bin/env python3
"""
Populate <episode_dir>/replay/augmented/real/{fid:06d}_camN.jpg from the raw
replay capture <episode_dir>/replay/camN/{fid:06d}.jpg.

WHY: 02_augment_noposplat.py produces this same augmented/real/ naming, but
also runs the full NoPoSplat encoder per frame to render novel views. When
training dual-cam with no novel views (--n_views 2, cam0_only=False), the
novel views are never used -- ReplayImageWindowDataset only requires
replay/augmented/real/ to exist for n_views > 1 (see 05_train_replay.py). So
this script does just the copy/rename, skipping NoPoSplat entirely (pure
filesystem op, no GPU).

replay_episode.py names raw capture files by the ORIGINAL demo's frame_id
(e.g. 000000.jpg, 000003.jpg, ... for --subsample 3), matching the keys in
augmented/tool_poses_task.npz -- so a straight copy preserves alignment.

Usage:
    python populate_replay_real.py --task_dir data/episodes/PastaTransfer_force
    python populate_replay_real.py --episode_dir data/episodes/PastaTransfer_force/021
"""
import os, sys, glob, argparse, shutil


def process_episode(ep_dir, cams, skip_done):
    replay_dir = os.path.join(ep_dir, 'replay')
    if not os.path.isdir(replay_dir):
        print(f"  Skipping {ep_dir} — no replay/ dir")
        return False

    real_dst = os.path.join(replay_dir, 'augmented', 'real')
    if skip_done and os.path.isdir(real_dst) and os.listdir(real_dst):
        print(f"  Skipping {ep_dir} — already populated")
        return False

    os.makedirs(real_dst, exist_ok=True)
    n_copied = 0
    for cam in cams:
        cam_dir = os.path.join(replay_dir, f'cam{cam}')
        if not os.path.isdir(cam_dir):
            continue
        for src in sorted(glob.glob(os.path.join(cam_dir, '*.jpg'))):
            fid = os.path.splitext(os.path.basename(src))[0]   # e.g. "000021"
            dst = os.path.join(real_dst, f'{fid}_cam{cam}.jpg')
            shutil.copyfile(src, dst)
            n_copied += 1
    print(f"  {os.path.basename(ep_dir)}: copied {n_copied} frames -> replay/augmented/real/")
    return True


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--episode_dir', help='Single episode directory')
    src.add_argument('--task_dir',    help='Task directory; processes all episodes')
    p.add_argument('--cams', type=int, nargs='+', default=[0, 1])
    p.add_argument('--skip_done', action='store_true', default=True)
    p.add_argument('--no_skip_done', dest='skip_done', action='store_false')
    args = p.parse_args()

    if args.episode_dir:
        ep_dirs = [args.episode_dir]
    else:
        ep_dirs = sorted(d for d in glob.glob(os.path.join(args.task_dir, '*')) if os.path.isdir(d))
        print(f"Found {len(ep_dirs)} episode(s) under {args.task_dir}")

    for ep_dir in ep_dirs:
        process_episode(ep_dir, args.cams, args.skip_done)


if __name__ == '__main__':
    main()
