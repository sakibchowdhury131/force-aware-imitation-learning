#!/usr/bin/env bash
# Waits for policy 4 (PID $1, running alone to avoid the 4-way GPU/CPU
# contention that was slowing everything down) to finish, then launches
# policies 3, 5, and 6 together (back to 3-way concurrency, which performed
# reasonably in the earlier run).
set -uo pipefail
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ti

POLICY4_PID="$1"
echo "Waiting for policy 4 (PID $POLICY4_PID) to finish..."
while kill -0 "$POLICY4_PID" 2>/dev/null; do
  sleep 30
done
echo "Policy 4 finished. Launching policies 3, 5, 6."

SCRATCH="/tmp/claude-1000/-home-sakib-working-dir-tool-as-interface-pipeline/24529643-4b9c-4e0a-a7ab-fcd6d16c65b3/scratchpad"

nohup python3 05_train.py \
  --data_dir data/episodes/PastaTransfer_force \
  --output_dir data/checkpoints/PastaTransfer_force_masked_dualcam_h16_all37 \
  --n_views 2 --track_cam 1 --action_horizon 16 --subsample 3 --action_frame task \
  > "$SCRATCH/policy3_train_v3.log" 2>&1 &
disown
echo "policy 3 launched, PID $!"

nohup python3 05_train_flow.py \
  --data_dir data/episodes/PastaTransfer_force \
  --output_dir data/checkpoints/PastaTransfer_force_flow_dualcam_h16_all37 \
  --n_views 2 --track_cam 1 --action_horizon 16 --subsample 3 --action_frame task \
  > "$SCRATCH/policy5_train_v3.log" 2>&1 &
disown
echo "policy 5 launched, PID $!"

nohup python3 05_train_act.py \
  --data_dir data/episodes/PastaTransfer_force \
  --output_dir data/checkpoints/PastaTransfer_force_act_dualcam_h16_all37 \
  --n_views 2 --track_cam 1 --action_horizon 16 --subsample 3 --action_frame task \
  > "$SCRATCH/policy6_train_v3.log" 2>&1 &
disown
echo "policy 6 launched, PID $!"

echo "=== run_after_policy4.sh done launching 3/5/6 ==="
