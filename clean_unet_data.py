"""
Remove mislabeled training samples after manual review of the viz/ folder.

Workflow:
    1. Run prepare_unet_data.py  → generates train/, val/, viz/train/, viz/val/
    2. Open viz/train/ and viz/val/ in a file browser
    3. DELETE any viz image where the mask looks wrong
    4. Run this script to remove the corresponding _img.jpg + _mask.png pairs

Usage:
    python clean_unet_data.py --data_dir data/unet_hand_data

    # dry run first (shows what would be deleted without deleting):
    python clean_unet_data.py --data_dir data/unet_hand_data --dry_run
"""

import os, sys, argparse


def clean_split(split_dir, viz_dir, dry_run):
    if not os.path.isdir(split_dir):
        print(f"  Split dir not found: {split_dir}")
        return 0, 0

    # stems present in viz (these are KEPT)
    viz_stems = set()
    if os.path.isdir(viz_dir):
        for f in os.listdir(viz_dir):
            if f.endswith('.jpg'):
                viz_stems.add(os.path.splitext(f)[0])

    # stems present in the data dir
    data_stems = set()
    for f in os.listdir(split_dir):
        if f.endswith('_img.jpg'):
            data_stems.add(f[:-len('_img.jpg')])

    removed_stems = data_stems - viz_stems
    n_removed = 0

    for stem in sorted(removed_stems):
        for suffix in ('_img.jpg', '_mask.png'):
            path = os.path.join(split_dir, stem + suffix)
            if os.path.exists(path):
                if dry_run:
                    print(f"  [dry] would remove: {path}")
                else:
                    os.remove(path)
                n_removed += 1

    return len(removed_stems), n_removed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True,
                   help='Output dir from prepare_unet_data.py')
    p.add_argument('--dry_run', action='store_true',
                   help='Print what would be deleted without actually deleting')
    return p.parse_args()


def main():
    args = parse_args()

    for split in ('train', 'val'):
        split_dir = os.path.join(args.data_dir, split)
        viz_dir   = os.path.join(args.data_dir, 'viz', split)
        print(f"\n── {split} ──")
        n_stems, n_files = clean_split(split_dir, viz_dir, args.dry_run)
        action = "Would remove" if args.dry_run else "Removed"
        print(f"  {action} {n_stems} samples ({n_files} files)")

    if args.dry_run:
        print("\nDry run complete. Re-run without --dry_run to apply.")
    else:
        print(f"\nDone. Now re-train:")
        print(f"  python train_unet_seg.py --data_dir {args.data_dir} "
              f"--output_dir data/checkpoints/unet_hand")


if __name__ == '__main__':
    main()
