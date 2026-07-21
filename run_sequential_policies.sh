#!/usr/bin/env bash
# Runs the 6 PastaTransfer_force policy experiments one at a time (full GPU
# per job, no contention) instead of concurrently. Each command blocks until
# its training finishes (early stopping via --patience, default 500) before
# the next one starts.
set -uo pipefail
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ti

EPS21_37="021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037"

echo "=== [1/6] Replay images, all 37 episodes ==="
python3 05_train_replay.py \
  --data_dir data/episodes/PastaTransfer_force \
  --output_dir data/checkpoints/PastaTransfer_force_replay_dualcam_h16_all37 \
  --n_views 2 --track_cam 1 --action_horizon 16 --subsample 3 --action_frame task \
  --val_episodes 037

echo "=== [2/6] Replay images, episodes 21-37 ==="
python3 05_train_replay.py \
  --data_dir data/episodes/PastaTransfer_force \
  --output_dir data/checkpoints/PastaTransfer_force_replay_dualcam_h16_ep21-37 \
  --n_views 2 --track_cam 1 --action_horizon 16 --subsample 3 --action_frame task \
  --include_episodes $EPS21_37 --val_episodes 037

echo "=== [3/6] Masked human-demo images, all 37 episodes ==="
python3 05_train.py \
  --data_dir data/episodes/PastaTransfer_force \
  --output_dir data/checkpoints/PastaTransfer_force_masked_dualcam_h16_all37 \
  --n_views 2 --track_cam 1 --action_horizon 16 --subsample 3 --action_frame task

echo "=== [4/6] Masked human-demo images, episodes 20-37 ==="
python3 05_train.py \
  --data_dir data/episodes/PastaTransfer_force_ep20-37 \
  --output_dir data/checkpoints/PastaTransfer_force_masked_dualcam_h16_ep20-37 \
  --n_views 2 --track_cam 1 --action_horizon 16 --subsample 3 --action_frame task

echo "=== [5/6] Flow matching, masked demo images, all 37 episodes ==="
python3 05_train_flow.py \
  --data_dir data/episodes/PastaTransfer_force \
  --output_dir data/checkpoints/PastaTransfer_force_flow_dualcam_h16_all37 \
  --n_views 2 --track_cam 1 --action_horizon 16 --subsample 3 --action_frame task

echo "=== [6/6] ACT, masked demo images, all 37 episodes ==="
python3 05_train_act.py \
  --data_dir data/episodes/PastaTransfer_force \
  --output_dir data/checkpoints/PastaTransfer_force_act_dualcam_h16_all37 \
  --n_views 2 --track_cam 1 --action_horizon 16 --subsample 3 --action_frame task

echo "=== ALL 6 POLICIES DONE ==="
