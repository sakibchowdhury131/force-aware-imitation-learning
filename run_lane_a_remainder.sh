#!/usr/bin/env bash
# Lane A remainder: policy 3 (PID $1) is already running standalone (orphaned
# from the original sequential wrapper, which was killed so policy 5 could
# start concurrently in Lane B). Wait for policy 3 to exit, then run policy 4.
set -uo pipefail
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ti

POLICY3_PID="$1"
echo "Lane A: waiting for policy 3 (PID $POLICY3_PID) to finish..."
while kill -0 "$POLICY3_PID" 2>/dev/null; do
  sleep 30
done
echo "Lane A: policy 3 finished. Starting policy 4."

echo "=== [4/6] Masked human-demo images, episodes 20-37 ==="
python3 05_train.py \
  --data_dir data/episodes/PastaTransfer_force_ep20-37 \
  --output_dir data/checkpoints/PastaTransfer_force_masked_dualcam_h16_ep20-37 \
  --n_views 2 --track_cam 1 --action_horizon 16 --subsample 3 --action_frame task

echo "=== Lane A done ==="
