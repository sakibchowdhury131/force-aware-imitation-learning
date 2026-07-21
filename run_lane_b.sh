#!/usr/bin/env bash
# Lane B: policy 5 (flow matching) then policy 6 (ACT), run concurrently
# alongside Lane A (policy 3 -> policy 4). Both lanes share masked-demo data
# already prepared, so no dependency between lanes.
set -uo pipefail
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ti

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

echo "=== Lane B done ==="
