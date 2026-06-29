"""
After manual review of generate_hand_masks.py output:
  1. Remove raw/mask pairs whose vis file was deleted.
  2. Merge the remaining new pairs with existing UNet datasets.
  3. Write a combined train/val split ready for train_unet_seg.py.

Usage:
    # Dry run first (shows what will be deleted / kept):
    python clean_and_merge.py \\
        --review_dir data/pastaTransfer4_hand_review \\
        --existing_dirs data/unet_hand_data data/unet_training_data \\
        --output_dir data/unet_combined \\
        --dry_run

    # Apply:
    python clean_and_merge.py \\
        --review_dir data/pastaTransfer4_hand_review \\
        --existing_dirs data/unet_hand_data data/unet_training_data \\
        --output_dir data/unet_combined \\
        --val_split 0.1
"""

import os, sys, argparse, random, shutil

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--review_dir',    required=True,
                   help='Output dir from generate_hand_masks.py (has raw/, masks/, vis/)')
    p.add_argument('--existing_dirs', nargs='*', default=[],
                   help='Existing UNet dataset dirs (each has train/ and val/ with '
                        '*_img.jpg + *_mask.png pairs)')
    p.add_argument('--output_dir',    default=os.path.join(PIPELINE_DIR, 'data', 'unet_combined'))
    p.add_argument('--val_split',     type=float, default=0.1,
                   help='Fraction of new data to hold out for validation (default 0.1)')
    p.add_argument('--seed',          type=int, default=42)
    p.add_argument('--dry_run',       action='store_true',
                   help='Print what would happen without writing anything')
    return p.parse_args()


def collect_review_pairs(review_dir):
    """Return list of (stem, raw_path, mask_path) for all pairs whose vis file still exists."""
    vis_dir  = os.path.join(review_dir, 'vis')
    raw_dir  = os.path.join(review_dir, 'raw')
    mask_dir = os.path.join(review_dir, 'masks')

    # stems with vis still present
    kept_stems = set()
    for f in os.listdir(vis_dir):
        if f.endswith('_vis.jpg'):
            kept_stems.add(f[:-len('_vis.jpg')])

    # stems with raw still present
    all_stems = set()
    for f in os.listdir(raw_dir):
        if f.endswith('.jpg'):
            all_stems.add(f[:-len('.jpg')])

    removed = all_stems - kept_stems
    kept    = all_stems & kept_stems

    print(f"Review dir: {review_dir}")
    print(f"  Total raw frames  : {len(all_stems)}")
    print(f"  Kept (vis present): {len(kept)}")
    print(f"  Removed (vis deleted): {len(removed)}")

    pairs = []
    for stem in sorted(kept):
        raw_path  = os.path.join(raw_dir,  f'{stem}.jpg')
        mask_path = os.path.join(mask_dir, f'{stem}.png')
        if os.path.exists(raw_path) and os.path.exists(mask_path):
            pairs.append((stem, raw_path, mask_path))
        else:
            print(f"  WARNING: missing raw or mask for {stem} — skipping")

    return pairs


def collect_existing_pairs(data_dir):
    """Collect (stem, img_path, mask_path) from an existing train/val dataset dir."""
    pairs = []
    for split in ('train', 'val'):
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            continue
        stems = sorted({f[:-len('_img.jpg')]
                        for f in os.listdir(split_dir) if f.endswith('_img.jpg')})
        for stem in stems:
            img_path  = os.path.join(split_dir, f'{stem}_img.jpg')
            mask_path = os.path.join(split_dir, f'{stem}_mask.png')
            if os.path.exists(img_path) and os.path.exists(mask_path):
                pairs.append((stem, img_path, mask_path))
    return pairs


def write_split(pairs, split_dir, label, dry_run, global_offset=0):
    if not dry_run:
        os.makedirs(split_dir, exist_ok=True)
    print(f"  {label}: {len(pairs)} samples → {split_dir}")
    for i, (stem, img_src, mask_src) in enumerate(pairs):
        dst_stem  = f'{global_offset + i:05d}_{stem}'
        img_dst   = os.path.join(split_dir, f'{dst_stem}_img.jpg')
        mask_dst  = os.path.join(split_dir, f'{dst_stem}_mask.png')
        if not dry_run:
            shutil.copy2(img_src,  img_dst)
            shutil.copy2(mask_src, mask_dst)


def main():
    args = parse_args()
    rng  = random.Random(args.seed)

    # 1. Collect new pairs from the review dir
    new_pairs = collect_review_pairs(args.review_dir)
    print()

    # 2. Collect existing pairs
    existing_pairs = []
    for d in args.existing_dirs:
        if not os.path.isdir(d):
            print(f"WARNING: existing dir not found: {d} — skipping")
            continue
        ep = collect_existing_pairs(d)
        print(f"Existing dataset {d}: {len(ep)} pairs")
        existing_pairs.extend(ep)
    print()

    # 3. Split new pairs into train/val
    rng.shuffle(new_pairs)
    n_val_new   = max(1, int(round(len(new_pairs) * args.val_split)))
    n_train_new = len(new_pairs) - n_val_new
    new_train   = new_pairs[:n_train_new]
    new_val     = new_pairs[n_train_new:]

    # 4. All existing pairs go to train (they were already split during their creation)
    #    and we re-shuffle for the combined dataset
    combined_train = existing_pairs + new_train
    combined_val   = new_val
    rng.shuffle(combined_train)

    print(f"Combined dataset:")
    print(f"  train: {len(existing_pairs)} existing + {len(new_train)} new = {len(combined_train)}")
    print(f"  val  : {len(combined_val)} new")

    if args.dry_run:
        print("\nDry run — no files written. Re-run without --dry_run to apply.")
        return

    train_dir = os.path.join(args.output_dir, 'train')
    val_dir   = os.path.join(args.output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir,   exist_ok=True)

    print(f"\nWriting combined dataset to {args.output_dir}/")
    write_split(combined_train, train_dir, 'train', dry_run=False, global_offset=0)
    write_split(combined_val,   val_dir,   'val',   dry_run=False, global_offset=len(combined_train))

    print(f"\nDone. To train:")
    print(f"  python train_unet_seg.py \\")
    print(f"      --data_dir {args.output_dir} \\")
    print(f"      --output_dir human_hand_segmentation_UNET \\")
    print(f"      --resume human_hand_segmentation_UNET/unet_best.pt")


if __name__ == '__main__':
    main()
