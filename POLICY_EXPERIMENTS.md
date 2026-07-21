# PastaTransfer_force — Policy Experiments (37-episode batch)

Tracking doc for the six policies trained on the expanded 37-episode dataset
(original 20 + 17 new episodes collected 2026-07-10). All use dual-camera
observations (`--n_views 2`, cam1 + other, no novel views), `action_horizon=16`,
`action_frame=task`, `--track_cam 1`, `--subsample 3`. Trained **sequentially**
(one at a time, full GPU per job) after an earlier concurrent run of all six
was found to be too GPU-contended (~11GB/12.3GB, each job 10-20x slower than
running alone) and was cancelled.

Data prep done once, shared by all six:
- `populate_replay_real.py` — copied raw replay captures into the
  `replay/augmented/real/` naming `05_train_replay.py` expects, for the 17
  new episodes (021-037). No NoPoSplat needed since no novel views.
- `03_segment.py --cam_only --unet_checkpoint human_hand_segmentation_UNET/unet_best.pt
  --unet_checkpoint2 robot_segmentation_UNET/training/checkpoints/best_model.pth
  --unet2_cams 1` — human-hand masking on all 37 episodes' raw demo frames
  (never done for this task before, not even the original 20), **plus** a
  second UNet masking the robot's stationary base/motor housing, which is
  partially visible on cam1 only. See "Masking bug" finding below — this was
  caught and fixed mid-run, requiring a full re-mask + retrain of policies
  3/4/5/6.

## Policies

| # | Name | Method | Data source | Episodes | Script | Output dir |
|---|------|--------|-------------|----------|--------|-------------|
| 1 | replay_dualcam_h16_all37 | DDPM diffusion | Replay images (robot's own gripper) | all 37 (36 train + 037 val) | `05_train_replay.py` | `data/checkpoints/PastaTransfer_force_replay_dualcam_h16_all37` |
| 2 | replay_dualcam_h16_ep21-37 | DDPM diffusion | Replay images | 021-037 only (16 train + 037 val) | `05_train_replay.py` | `data/checkpoints/PastaTransfer_force_replay_dualcam_h16_ep21-37` |
| 3 | masked_dualcam_h16_all37 | DDPM diffusion | Masked human-demo images | all 37 | `05_train.py` (unmodified) | `data/checkpoints/PastaTransfer_force_masked_dualcam_h16_all37` |
| 4 | masked_dualcam_h16_ep20-37 | DDPM diffusion | Masked human-demo images | 020-037 (symlinked subset) | `05_train.py` (unmodified) | `data/checkpoints/PastaTransfer_force_masked_dualcam_h16_ep20-37` |
| 5 | flow_dualcam_h16_all37 | Flow matching | Masked human-demo images | all 37 | `05_train_flow.py` (new) | `data/checkpoints/PastaTransfer_force_flow_dualcam_h16_all37` |
| 6 | act_dualcam_h16_all37 | ACT (transformer, CVAE) | Masked human-demo images | all 37 | `05_train_act.py` (new) | `data/checkpoints/PastaTransfer_force_act_dualcam_h16_all37` |

Policies 1-4 and untouched `05_train.py`/`05_train_replay.py` all preserve
backward compatibility with every previously trained checkpoint — no existing
script behavior changed for callers that don't pass the new optional flags
(`--include_episodes`).

## Status

| # | Status | Best epoch | Notes |
|---|--------|-----------|-------|
| 1 | **done** | early-stopped @ 578, best val_ema=0.0620 | ~9.3 sec/epoch. Val gap turned positive ~epoch 70-80 and climbed steadily after — overfitting on the 36-episode train split, patience=500 triggered cleanly. Unaffected by the masking bug below (replay-based, no masked images involved). |
| 2 | **done** | early-stopped @ 597, best val_ema=0.0706 | ~9.3 sec/epoch. Worse best-val than policy 1, consistent with training on fewer episodes (16 vs 36). Same overfitting pattern. Unaffected by the masking bug. |
| 3 | **stopped for testing** @ epoch 2600/3050 (checkpoint `policy_epoch2600.pt`), loss ema~0.0006 | — | Manually stopped (not early-stopped/completed) to test the batch of checkpoints before deciding whether to keep training. No resume support yet in `05_train.py` -- continuing later means either retraining from scratch or building a checkpoint-load wrapper first. |
| 4 | **done** | full 3050 epochs, final ema=0.0005 | Solo/2-way rate settled at ~11.5 sec/epoch after initial startup overhead (17.4s/epoch first 10 epochs). No validation split, so this is train-loss only -- very low loss on 18 episodes is at least consistent with the model having enough capacity to fit that smaller set closely; not itself evidence of good generalization. |
| 5 | **stopped for testing** @ epoch 1100/3050 (checkpoint `policy_epoch1100.pt`), loss ema~0.011 | — | Manually stopped alongside 3 and 6. Same resume caveat as policy 3 -- `05_train_flow.py` has no checkpoint-load path yet either. |
| 6 | **stopped for testing** @ epoch 1100/3050 (checkpoint `policy_epoch1100.pt`), loss ema~0.005 | — | Manually stopped. **KL was still exactly 0.0000 through epoch 1150** before stopping -- posterior collapse confirmed as the real, persistent pattern for this run (held for 1000+ epochs), not noise. CVAE latent likely unused; worth a lower --kl_weight or KL warmup if this policy underperforms in testing. |

**Next step (per user request):** test policies 1/2/4 (fully done) and the partially-trained 3/5/6 checkpoints above before deciding whether to resume training any of them further from their current checkpoints.

Run order/orchestration: started as `run_sequential_policies.sh` (fully
sequential). Policy 3 turned out to be data-loading-bound rather than
GPU-bound (GPU util bursting 41-99%, never pegged; only ~5/32 CPU cores in
use), so switched to running all four masked-data policies (3/4/5/6)
concurrently as standalone background jobs instead, once GPU/CPU headroom was
confirmed (GPU settled around 8/12.3GB with all four running, well short of
saturating either GPU memory or the 32 CPU cores).

## Deployment (2026-07-12)

All six checkpoints deployed live on the real robot via `07_deploy.py`, one at
a time, each pre-positioned to episode 021's frame-0 pose first (see
"Pre-positioning" below) so every policy starts from the same known,
in-distribution pose regardless of wherever the arm was last left.

| # | Deploy masking flag | Result |
|---|---------------------|--------|
| 1 | `--no_arm_mask` | Initially looked like the old "stalls near pasta box" failure — turned out to be a real control-loop bug (see "Open-loop drift" below), not the model. **Works correctly** once `--wait_convergence` was added. |
| 2 | `--no_arm_mask` | **Works correctly** (same fix as policy 1). |
| 3 | `--unet_checkpoint robot_segmentation_UNET/.../best_model.pth` | **Failed** — could not scoop the pasta reliably, and did not transfer to the plate even when scooping partially succeeded. Checkpoint was only trained to epoch 2600/3050 (manually stopped, not early-stopped/converged) — worth retesting after further training before concluding the *architecture* is the problem. |
| 4 | `--unet_checkpoint robot_segmentation_UNET/.../best_model.pth` | **Works correctly.** Fully-converged checkpoint (3050/3050 epochs), trained on the 18 newest episodes only. |
| 5 | `--unet_checkpoint robot_segmentation_UNET/.../best_model.pth` | **Works correctly.** First-ever live test of the flow-matching deploy path (see below) — no issues. |
| 6 | `--unet_checkpoint robot_segmentation_UNET/.../best_model.pth` | **Works correctly.** First-ever live test of the ACT deploy path (see below) — no issues, despite the posterior-collapse finding from training (KL≈0 the whole run). Worth remembering this policy is likely behaving close to a deterministic regressor rather than using its CVAE latent for anything. |

### Deployment commands (exact, as run)

Common to all six: `--robot_extrinsics`/`--robot_extrinsics_proprio` both point
at the corrected calibration, `--mesh newspoon1.obj --tool_prompt "spoon"`,
`--track_cam 1 --frequency 10`, pre-positioned to episode 021 frame 0, full
16-step execution per inference cycle, and `--wait_convergence` (required —
see "control loop never reconciled" finding below). Only the checkpoint and
the arm-masking flag change per policy.

**Policy 1** (replay, all 37):
```bash
python3 07_deploy.py \
  --checkpoint data/checkpoints/PastaTransfer_force_replay_dualcam_h16_all37/policy_final.pt \
  --no_arm_mask \
  --robot_extrinsics data/robot_extrinsics_stick_corrected_zmeasured.npy \
  --robot_extrinsics_proprio data/robot_extrinsics_stick_corrected_zmeasured.npy \
  --mesh newspoon1.obj --tool_prompt "spoon" --track_cam 1 --frequency 10 \
  --init_episode_dir data/episodes/PastaTransfer_force/021 --init_frame 0 \
  --exec_steps 16 --wait_convergence --execute
```

**Policy 2** (replay, episodes 21-37):
```bash
python3 07_deploy.py \
  --checkpoint data/checkpoints/PastaTransfer_force_replay_dualcam_h16_ep21-37/policy_final.pt \
  --no_arm_mask \
  --robot_extrinsics data/robot_extrinsics_stick_corrected_zmeasured.npy \
  --robot_extrinsics_proprio data/robot_extrinsics_stick_corrected_zmeasured.npy \
  --mesh newspoon1.obj --tool_prompt "spoon" --track_cam 1 --frequency 10 \
  --init_episode_dir data/episodes/PastaTransfer_force/021 --init_frame 0 \
  --exec_steps 16 --wait_convergence --execute
```

**Policy 3** (masked demo, all 37, epoch 2600/3050 — the one that failed):
```bash
python3 07_deploy.py \
  --checkpoint data/checkpoints/PastaTransfer_force_masked_dualcam_h16_all37/policy_epoch2600.pt \
  --unet_checkpoint robot_segmentation_UNET/training/checkpoints/best_model.pth \
  --robot_extrinsics data/robot_extrinsics_stick_corrected_zmeasured.npy \
  --robot_extrinsics_proprio data/robot_extrinsics_stick_corrected_zmeasured.npy \
  --mesh newspoon1.obj --tool_prompt "spoon" --track_cam 1 --frequency 10 \
  --init_episode_dir data/episodes/PastaTransfer_force/021 --init_frame 0 \
  --exec_steps 16 --wait_convergence --execute
```

**Policy 4** (masked demo, episodes 20-37, fully converged):
```bash
python3 07_deploy.py \
  --checkpoint data/checkpoints/PastaTransfer_force_masked_dualcam_h16_ep20-37/policy_final.pt \
  --unet_checkpoint robot_segmentation_UNET/training/checkpoints/best_model.pth \
  --robot_extrinsics data/robot_extrinsics_stick_corrected_zmeasured.npy \
  --robot_extrinsics_proprio data/robot_extrinsics_stick_corrected_zmeasured.npy \
  --mesh newspoon1.obj --tool_prompt "spoon" --track_cam 1 --frequency 10 \
  --init_episode_dir data/episodes/PastaTransfer_force/021 --init_frame 0 \
  --exec_steps 16 --wait_convergence --execute
```

**Policy 5** (flow matching, epoch 1100/3050):
```bash
python3 07_deploy.py \
  --checkpoint data/checkpoints/PastaTransfer_force_flow_dualcam_h16_all37/policy_epoch1100.pt \
  --unet_checkpoint robot_segmentation_UNET/training/checkpoints/best_model.pth \
  --robot_extrinsics data/robot_extrinsics_stick_corrected_zmeasured.npy \
  --robot_extrinsics_proprio data/robot_extrinsics_stick_corrected_zmeasured.npy \
  --mesh newspoon1.obj --tool_prompt "spoon" --track_cam 1 --frequency 10 \
  --init_episode_dir data/episodes/PastaTransfer_force/021 --init_frame 0 \
  --exec_steps 16 --wait_convergence --execute
```

**Policy 6** (ACT, epoch 1100/3050):
```bash
python3 07_deploy.py \
  --checkpoint data/checkpoints/PastaTransfer_force_act_dualcam_h16_all37/policy_epoch1100.pt \
  --unet_checkpoint robot_segmentation_UNET/training/checkpoints/best_model.pth \
  --robot_extrinsics data/robot_extrinsics_stick_corrected_zmeasured.npy \
  --robot_extrinsics_proprio data/robot_extrinsics_stick_corrected_zmeasured.npy \
  --mesh newspoon1.obj --tool_prompt "spoon" --track_cam 1 --frequency 10 \
  --init_episode_dir data/episodes/PastaTransfer_force/021 --init_frame 0 \
  --exec_steps 16 --wait_convergence --execute
```

Deploy masking convention established: policies 1/2 (replay-based, trained on
*unmasked* images showing the robot's own gripper) use `--no_arm_mask` at
deploy time — masking live would be a train/deploy mismatch. Policies 3-6
(masked-demo-based, trained on images with the *human hand* masked out) need
the opposite: live masking of the robot's own gripper, so the visual
distribution matches training (no visible task-performing agent, human or
robot). `07_deploy.py`'s existing `--unet_checkpoint`/`load_unet`/`unet_mask`
path turned out to already be built for exactly this — it loads a
`segmentation_models_pytorch` resnet34 Unet at 288x512 with ImageNet
normalization, i.e. `robot_segmentation_UNET`'s exact architecture/preprocessing
— so no code changes were needed there, just pointing `--unet_checkpoint` at
`robot_segmentation_UNET/training/checkpoints/best_model.pth`.

### New: `--init_episode_dir` / `--init_frame` pre-positioning (`07_deploy.py`)

Added a one-time, unclamped pre-positioning move at deploy startup — loads
`<init_episode_dir>/augmented/tool_poses_base.npz[str(init_frame)]`, converts
to an EEF target via the already-computed `T_tool_eef` (same convention used
everywhere else in the file), and sends+waits for convergence before the
policy control loop begins. Mirrors `replay_episode.py`'s existing
pre-position-to-frame[0] step. Purely additive (new optional args, `None` by
default — skips entirely if not passed), so no change to prior deploy
behavior. Used with `--init_episode_dir data/episodes/PastaTransfer_force/021
--init_frame 0` for every policy in this batch, since episode 021 falls
within all six policies' training distributions.

### Found + fixed: control loop never reconciled its "current position" against reality

While debugging why policy 1 initially looked like it was predicting motion
toward the plate but the physical arm never got there: traced the full
pred -> clamp -> send path in `07_deploy.py` and confirmed the *sent* command
always exactly matches the printed "clamped" value (no bug there). The real
issue: `T_base_eef_cur` (the baseline each new step's clamp is computed
relative to) was only ever updated as `T_base_eef_cur = T_base_eef_clamped`
(line ~943) — i.e. the software *assumes* every commanded step is reached
instantly, and never re-reads `get_cartesian_pose(api)` to confirm. The
actual measured pose (`T_base_eef_now`, computed every loop iteration) was
only ever used for the 3D-viz state file, never fed back into the clamp math.
If the physical arm can't keep up with the commanded pace (up to 2cm per
100ms step by default, faster than the arm's default speed), the software's
belief of "where the arm is" silently drifts ahead of reality, and each new
clamp gets computed relative to that increasingly-wrong assumption.

Not fixed in code — resolved by using the existing `--wait_convergence` flag
(polls `get_cartesian_pose` and blocks until the arm actually reaches each
target before sending the next command), which the deploy commands up to this
point had not been using. All six policies deployed with `--wait_convergence`
from policy 1 onward and none showed this symptom again. A more robust
code-level fix (reconciling `T_base_eef_cur` with `T_base_eef_now` every
iteration regardless of `--wait_convergence`) is still open if this recurs.

### New: deploy-time inference support for flow matching and ACT

Neither `05_train_flow.py` nor `05_train_act.py` checkpoints could be
deployed before this — `07_deploy.py`/`test_policy.py`'s `load_model` always
constructed a `DiffusionPolicyNet` and unconditionally expected
`ckpt['noise_scheduler']` for DDPM sampling, which flow-matching checkpoints
don't have and ACT checkpoints (different architecture entirely) can't use at
all. Added, all additive/backward-compatible via `ckpt.get('train_method',
'ddpm')` branching (absent -> `'ddpm'`, so every pre-existing checkpoint from
policies 1-4 takes the exact same path as before, zero behavior change):
  - `test_policy.predict_action_sequence_flow` — Euler-integrates
    dx/dt = v_theta(x_t, t*time_scale, obs) from t=0 to 1, reusing the same
    `DiffusionPolicyNet` as a velocity field (mirrors `05_train_flow.py`'s
    training objective).
  - `test_policy.predict_action_sequence_act` — single transformer forward
    pass via `act_common.ACTPolicy`, `actions=None` at inference (CVAE latent
    z fixed to zero, matching the original ACT eval convention).
  - `load_model` now branches on `ckpt.get('train_method') == 'act'` to
    construct `ACTPolicy` instead of `DiffusionPolicyNet`.
  - `07_deploy.py`'s inference call site branches on the same field to call
    the right predict function.
Both were offline-smoke-tested against real episode data (episode 021, frame
0) before the live test — sane, in-training-range predictions, no crashes —
then confirmed working live on the first attempt for both policies 5 and 6.

## Findings & Notes

### Masking bug: robot base was only partially masked (found + fixed 2026-07-11)

Policies 3-6 all train on `augmented/masked_real/` images produced by
`03_segment.py`. The initial run only used the human-hand UNet
(`human_hand_segmentation_UNET`), which masks the human demonstrator's
hand/arm. It missed that **cam1 also partially shows the robot's stationary
base/motor housing** in the corner of frame — never masked, so it was
present (unmasked) in every training image sourced from cam1.

Fix: added `--unet_checkpoint2` / `--unet2_cams` to `03_segment.py`
(additive-only change, default behavior unchanged for any caller not passing
these flags). `--unet_checkpoint2` loads a second UNet — `robot_segmentation_UNET`
uses a completely different architecture (`segmentation_models_pytorch`
ResNet34-Unet, not the pipeline's own `train_unet_seg.ResNetUNet`), so it
needed its own preprocessing/loading path. `--unet2_cams` (default `[1]`)
restricts the second mask to cam1 only, since cam0 never shows the robot and
unioning it there would just add false-positive blackouts for nothing. Ran
`03_segment.py --cam_only` again on all 37 episodes to regenerate
`masked_real/` with both masks combined; verified visually (zoomed crop) that
the robot base is now correctly blacked out on cam1 while cam0 is untouched.

**Side effect**: while testing the fix on episode 001's live files (which
policies 3 and 5 were actively reading mid-training), a dataloader worker in
each hit `OSError: image file is truncated` from reading a JPEG mid-overwrite
— a race condition from re-masking data that was being actively consumed by
training. Not a real loss since 3/4/5/6 all needed a full retrain on the
corrected masks anyway; policies 4 and 6 were killed manually for the same
reason once the masking bug was confirmed.

---

### Background: why this batch exists

Earlier deployment attempts (single-cam and dual-cam, `action_horizon=8`,
20-episode dataset) showed the robot correctly approaching the pasta box and
scooping, then stalling — converging to a near-static predicted pose instead
of continuing to the plate. Offline teacher-forced evaluation
(`eval_replay_offline.py`) showed the trained policy *could* predict forward
motion through that same phase when fed the exact training-distribution
images, ruling out "the model never learned this part of the task" as the
sole explanation. Most likely cause: closed-loop distribution shift — small
real-world execution errors compounding until the live post-scoop image looks
sufficiently unlike anything in the 20-episode training set that the
diffusion sampler defaults to a "stay put" prediction (a known multimodal
mode-averaging failure). This batch tests two levers at once: more/varied
data (37 episodes vs 20, plus an ablation training on only the newest 17) and
alternative architectures/objectives (flow matching, ACT) that may be more
robust to this kind of distribution shift than DDPM diffusion.

---

## Streaming Deployment (`deploy_streaming.py`, 2026-07-12)

Separate deployment path from `07_deploy.py`'s blocking predict→clamp→wait
loop: two threads (slow policy-inference loop + fast 100Hz control loop)
streaming continuous `CARTESIAN_VELOCITY` commands, with a clamped-cubic-spline
feedforward tracker, proportional position/rotation feedback to convert the
spline target into a velocity command, and an optional force-admittance
correction term. `07_deploy.py`/`test_policy.py` are imported/reused
unmodified — every existing position-based deploy command still works exactly
as before. See `benchmark_velocity_control.py` and `benchmark_force_admittance.py`
for the standalone hardware-capability checks done before wiring this in
(100Hz `CARTESIAN_VELOCITY` streaming confirmed viable; force/admittance
signal needs a per-run tare — see below).

Checkpoint used for all live testing so far:
`data/checkpoints/PastaTransfer_force_masked_dualcam_h16_ep20-37/policy_final.pt`
(policy 4 from the table above — the fully-converged masked/ep20-37 checkpoint).
Pre-positioned to episode 021 frame 0 for every run, same convention as the
`07_deploy.py` batch.

### Found + fixed: rotation control diverged, not just lagged (2026-07-12)

The first several live `--execute` runs all showed the same symptom regardless
of any translation-side fix (velocity cap, spline min-gap filtering,
ground-truth position anchoring — see code comments in `deploy_streaming.py`
for that earlier round of fixes): the robot "drifted to a random location."
Translation-side logs looked bounded and sane throughout, which was the
confusing part — turned out rotation was never logged at all, so every
earlier diagnosis was blind to it.

Added `rot_err_deg`/`w_cmd`/`w_mag_precap`/`R_actual_quat`/`R_target_quat`
logging, then ran the policy live (short duration, conservative
`--max_rot_speed`, user holding the physical stop button) specifically to
capture this. Result: `rot_err_deg` grew **monotonically** for the entire 6s
run (0° → 42°, never once decreasing) while `w_mag_precap` stayed saturated
80% of the time — a diverging control loop, not a lagging one.

Root cause: the rotation law `w_cmd = kp_rot * (R_target * R_actual.inv()).as_rotvec()`
is the textbook-correct proportional law *if* the firmware expects angular
velocity in the **spatial/base frame** (`Omega = vee(Ṙ·R⁻¹)`, exactly what
Kinova's own `kinova-ros` driver source comments claim). Empirically this
Jaco2's legacy USB SDK behaves as though `CARTESIAN_VELOCITY`'s rotational
fields want **body-frame** angular velocity instead. Fix: swap to
`R_err = R_actual.inv() * R_target` (body-frame law). Re-ran the identical
test immediately after — `rot_err_deg` became bounded and self-correcting
(oscillates in a 0-20° band, mean ~6°, `w_mag_precap` dropped from mean 0.52
to 0.16 rad/s) — confirmed across three separate runs afterward (6s/8s/28s,
increasing speed caps each time), no recurrence of the divergence.

**Takeaway for any future velocity-mode rotation work on this hardware**: do
not trust the SDK/driver source comments for angular-velocity frame
convention — verify empirically. A monotonically-growing (not oscillating)
error that never once recovers, even briefly, is the signature of a
sign/frame mismatch, not a speed/gain tuning problem.

### Found: translation speed cap was the wrong kind of conservative

Separately from the rotation bug, `--max_speed` (originally defaulted to a
very conservative 0.05 m/s while first ruling out a translation-side runaway)
turned out to just be slower than the policy's natural pace: at 0.05 m/s the
controller wanted a median ~9-13cm/s and was saturated 85-87% of every run's
steps. Not dangerous (every `v_cmd` stayed correctly bounded at the cap,
`qdot` stayed low) but meant the arm could never actually catch up to the
spline target — tracking error grows every chunk transition, partially
recovers, repeats. Raised to 0.10-0.14 m/s across the later runs; saturation
dropped to 33-38%. Likely still has headroom to raise further once comfortable
with how a run at 0.14 looks physically.

### Live test results (chronological, all with the rotation fix where noted)

| Run | `--max_speed` | `--max_rot_speed` | Duration | Rotation fix? | Result |
|---|---|---|---|---|---|
| 1 | 0.03 | 0.5 (default) | 8s | No | Translation bounded/sane; robot drifted to a random location (rotation divergence, undiagnosed at the time) |
| 2 | 0.05 | 0.2 | 6s | No (diagnostic run) | Confirmed `rot_err_deg` 0°→42° monotonic divergence |
| 3 | 0.05 | 0.2 | 6s | **Yes** | `rot_err_deg` bounded, mean 6.35°, max 17.97° |
| 4 | 0.05 | 0.5 (normal cap) | 8s | Yes | `rot_err_deg` mean 6.19°, `w_mag_precap` never hit the cap. Physically: moved fine, entered pasta box, scooped, run ended mid-task (duration limit) |
| 5 | 0.10 | 0.5 | 18s | Yes | Physically: scooped and lifted, approached the bowl, did not drop pasta before duration ended |
| 6 | 0.14 | 0.5 | 28s | Yes | Physically: hovered over the pasta box without progressing — no controller-side cause found in the log (buffer only dry at startup, one brief force-freeze at t=4.5s) — looks like a policy-competency stall, not a controls bug |

Full episode 021 is 450 frames @ 30fps = 15s of demonstrated task time;
because the controller still runs somewhat behind the policy's implied pace
(see saturation finding above), wall-clock run duration needs to exceed 15s
by a comfortable margin to see the full task play out live.

### Added: live camera preview (`"Deploy"` + `"Policy View"` windows)

`deploy_streaming.py` originally ran fully headless (no `07_deploy.py`-style
`imshow` at all). Added the same two windows back, reusing `test_policy.py`'s
`draw_axes_simple`/`draw_axes_with_horizon` unmodified: `"Deploy"` shows the
current tool pose + predicted-horizon trail projected onto the raw camera
frame plus a HUD line (step, EXECUTE/DRY RUN, live force/`v_precap`/
`rot_err`/tracking-error numbers pulled from the fast loop); `"Policy View"`
shows the masked camera feed(s) the policy actually sees. New `--task_frame`
arg (same default-resolution convention as `07_deploy.py`:
`data/cam_extrinsics.npy` or `data/cam<N>_extrinsics.npy`).

Implementation note: all `cv2` GUI calls (window creation + `imshow`/
`waitKey`) are confined to the slow-loop thread only, not split across
threads — OpenCV's Linux GUI backends aren't reliably thread-safe across
different threads, and the fast/control loop runs on the main thread, so
window creation happening in `main()` while `imshow` happened in the
slow-loop thread (the first draft) risked a crash/hang. `q` in either window
now sets `shared.stop`, which both loops check, for a clean quit.

**Not yet live-tested**: robot was powered off before a run with camera
frames could be done (`InitAPI()` failed with `ERROR_NO_DEVICE_FOUND`/1015 —
`lsusb` confirmed the Jaco2 wasn't enumerating, both RealSense cameras were
fine). Module load-tested only (imports resolve, compiles clean). First step
next session: a dry run to confirm the preview windows render correctly
before doing anything with `--execute`.

### Commands (exact, as run)

Common to every run below: `--checkpoint
data/checkpoints/PastaTransfer_force_masked_dualcam_h16_ep20-37/policy_final.pt`,
`--robot_extrinsics`/`--robot_extrinsics_proprio` both point at the corrected
calibration, `--mesh newspoon1.obj --tool_prompt "spoon" --track_cam 1`,
`--init_episode_dir data/episodes/PastaTransfer_force_ep20-37/021 --init_frame 0`.
No `--no_arm_mask` (this checkpoint was trained on masked images, so masking
must stay on at deploy time too, unlike policies 1/2 in the `07_deploy.py`
batch above).

**Dry run** (no motion — always do this first for any new checkpoint/episode/frame):
```bash
python3 deploy_streaming.py \
  --checkpoint data/checkpoints/PastaTransfer_force_masked_dualcam_h16_ep20-37/policy_final.pt \
  --robot_extrinsics data/robot_extrinsics_stick_corrected_zmeasured.npy \
  --robot_extrinsics_proprio data/robot_extrinsics_stick_corrected_zmeasured.npy \
  --mesh newspoon1.obj --tool_prompt "spoon" --track_cam 1 \
  --init_episode_dir data/episodes/PastaTransfer_force_ep20-37/021 --init_frame 0 \
  --duration 20
```

**Latest known-good real-motion run** (28s, current best settings post rotation fix):
```bash
python3 deploy_streaming.py \
  --checkpoint data/checkpoints/PastaTransfer_force_masked_dualcam_h16_ep20-37/policy_final.pt \
  --robot_extrinsics data/robot_extrinsics_stick_corrected_zmeasured.npy \
  --robot_extrinsics_proprio data/robot_extrinsics_stick_corrected_zmeasured.npy \
  --mesh newspoon1.obj --tool_prompt "spoon" --track_cam 1 \
  --init_episode_dir data/episodes/PastaTransfer_force_ep20-37/021 --init_frame 0 \
  --max_speed 0.14 --max_rot_speed 0.5 --duration 28 --execute
```

Not yet tried live: `--admittance --K 200 --f_desired 0` (force-compliance
condition — every run so far has used stiff position tracking only, `corr=0`
throughout).

### Open items

- Camera preview windows added this session, not yet confirmed live (robot
  was powered off — see above).
- Policy stalling short of task completion (hovering over the pasta box in
  the longest run so far) looks like a model-competency issue per the log,
  not a controls bug, but only one data point — worth a few more longer runs
  to see if it's consistent.
- `--max_speed` likely still has headroom above 0.14 given saturation was
  still 33-38% in the most recent runs; raise incrementally and recheck the
  `v_mag_precap`/`v_capped` stats each time.
- `cond(J)=6.94e+17` singularity warning appears inconsistently right after
  pre-positioning to episode 021 frame 0 across different runs (present in
  some, absent in others) — traced one occurrence to a log-array
  zero-initialization artifact rather than a real pose issue, but this should
  be double-checked directly (print `cond(J)` right after pre-positioning
  completes, independent of the log) if it keeps showing up.

---

## Force-conditioned policy + hybrid position/force deployment (2026-07-12/13)

Separate track from the streaming-controller work above: instead of pure
position control, predict force alongside pose and use the prediction as a
time-varying admittance reference on top of `07_deploy.py`'s proven blocking
control loop. Motivation: pure position control already works (see the six
policies above), so this isn't about fixing a failure — it's about
robustness to variation (pasta pile shape/amount, calibration drift) that
none of those six runs had to face, on the theory that "push until you feel
this much resistance" generalizes better than "go to this exact xyz" for a
deformable, granular contact task.

### Training: `05_train_replay_force.py` (new, `05_train.py`/`05_train_replay.py` untouched)

Extends the replay-image dataset (`05_train_replay.py`) to 12D
`[pose9, force3]` for both observation (past force) and action target
(future force, predicted jointly with pose over the same `action_horizon`).
Force labels come from each episode's **dense** replay force log
(`replay/replay_full_forces.npz`, ~47Hz — not the sparse ~150-sample
per-frame log, too coarse to filter at 2Hz), tared against the first 1.0s,
low-pass filtered at **2Hz**, matched to each training frame by timestamp via
`replay/torque_log.npz`'s frame→time mapping. Loss: per-dimension weighted
noise-prediction MSE, pose dims weighted 1.0, force dims weighted 0.2 (force
is a noisier, more contact-geometry-sensitive signal across demonstrations
than pose — stays auxiliary, actions dominate).

`policy_common.py`'s `DiffusionPolicyNet`/`MaxAbsNormalizer` needed zero
changes (already dimension-generic via `action_dim`/`proprio_dim`
constructor params) — only the dataset/training script changed.

**Full training run:** all 18 episodes with replay force logs turned out to
be 36 of `PastaTransfer_force`'s 37 episodes (001-019 never had
`--capture_camera` replay; the "ep20-37" naming from the earlier policy
batch was a training-data-subset choice, not a force-log-availability
boundary — broader coverage than expected). Episode 037 held out for
validation, matching the existing replay-policy convention. Early-stopped at
epoch 574/3050, best `val_ema=0.0548` (~epoch 80) — same overfitting pattern
already documented for the position-only policies above, nothing new.
Checkpoint: `data/checkpoints/PastaTransfer_force_replay_force/policy_final.pt`.

```bash
python3 05_train_replay_force.py \
  --data_dir data/episodes/PastaTransfer_force \
  --output_dir data/checkpoints/PastaTransfer_force_replay_force \
  --subsample 3 --action_frame task --track_cam 1 --n_views 2 \
  --val_episodes 037
```

### Deployment: `07_deploy_force.py` (new, `07_deploy.py` untouched)

Full copy of `07_deploy.py`'s blocking/`--wait_convergence` control loop,
unchanged, plus: a live force pipeline (tare + causal 2Hz Butterworth, same
chain as `deploy_streaming.py`) feeding past force into the 12D proprio
observation, and `--admittance` (off by default) which turns the predicted
future force into `f_desired(t)` — correction =
vector-magnitude-capped `K⁻¹(F_live − f_desired)`, added to the predicted
position **before** `clamp_pose_step`'s existing safety clamp, so it
inherits the same per-step safety bound a bad position prediction would.
Works with ordinary 9D checkpoints too (correction forced to zero), so the
same script A/Bs position-only vs. hybrid.

Two small, backward-compatible fixes were needed in `test_policy.py` (every
existing 9D checkpoint unaffected, defaults preserve exact old behavior):
`load_model` now reads `action_dim` from the checkpoint instead of
hardcoding 9, and `action_9d_to_pose(a)` → `action_9d_to_pose(a[:9])` — the
unfixed version would silently feed the 3 trailing force values into the
6D-rotation decoder (`rot6d_to_matrix` expects exactly 6 numbers),
corrupting the rotation matrix. Caught via an offline smoke test (real
episode-021 images through the real checkpoint) before ever touching the
robot — confirmed broken before the fix (non-orthonormal rotation matrix)
and correct after (`R @ R.T ≈ I`, `det(R) ≈ 1.0`).

### Found + fixed: admittance correction saturating against the position-clamp budget

First `--execute --admittance` run: robot scooped successfully, then
hovered over the pasta box indefinitely instead of proceeding to the bowl,
despite the policy-view overlay showing the model predicting motion toward
the bowl. `corr=` in the console log was pinned at the `--f_max_correction_cm`
cap (2.00cm) on >95% of all steps from step 9 through the end of a 150-step
run — not an occasional nudge, continuous saturation.

Root cause: `--max_trans_step` (the per-step position-clamp safety bound)
and `--f_max_correction_cm` (the admittance correction cap) both defaulted
to the same 2cm. Since the correction is added to the target *before*
`clamp_pose_step` runs, a saturated correction alone could consume the
entire per-step motion budget, leaving no room for the model's actual
predicted progress to survive the clamp — `clamp_pose_step` scales the
combined (progress + correction) vector down to fit in 2cm regardless of
how much of that vector was "useful" motion. Position visibly froze in a
narrow band (Z stuck at 6-8cm, no further descent-then-lift) for the entire
second half of the run.

Fix (tested, confirmed): raised `--max_trans_step` to 0.04 (double the
correction cap), keeping the correction-before-clamp ordering (so total
per-step motion is still hard-bounded, just by a larger number, and the
safety semantics are unchanged in kind). Re-ran the identical scenario —
position trajectory now visibly transitions through multiple regions
instead of freezing (a real ~12cm/~5cm/~8cm shift in X/Y/Z around steps
105-134 of the 150-step run, then stabilizing in a new region near the
bowl) — **task completed successfully**: scoop, transfer, and drop into the
bowl, live on the robot, using the predicted-force admittance reference.

**Known-good command** (first fully successful hybrid position+force run):
```bash
python3 07_deploy_force.py \
  --checkpoint data/checkpoints/PastaTransfer_force_replay_force/policy_final.pt \
  --no_arm_mask \
  --robot_extrinsics data/robot_extrinsics_stick_corrected_zmeasured.npy \
  --robot_extrinsics_proprio data/robot_extrinsics_stick_corrected_zmeasured.npy \
  --mesh newspoon1.obj --tool_prompt "spoon" --track_cam 1 \
  --init_episode_dir data/episodes/PastaTransfer_force/021 --init_frame 0 \
  --max_trans_step 0.04 \
  --wait_convergence --admittance --execute
```

### Open items

- `--f_max_correction_cm` (still 2cm) hasn't itself been re-tuned — only
  `--max_trans_step` was raised to give it headroom. Worth testing whether a
  smaller correction cap (e.g. 0.5-1cm, a gentler nudge) plus the original
  2cm `--max_trans_step` gives similar or better results without needing a
  larger position-clamp bound.
- Possible secondary contributor, not yet isolated: the causal
  (`OnlineButterworth`, real-time, phase-lagged) live filter vs. the
  non-causal (`filtfilt`, zero-phase) filter used for training labels —
  ruled out as the *primary* cause this round (the clamp-budget saturation
  fully explained the observed behavior), but not independently verified to
  contribute nothing.
- Only tested against episode 021's pre-positioned starting pose so far —
  worth trying other episodes/frames to see how well the force reference
  generalizes.
- No comparison yet against the position-only baseline (`--admittance` off)
  on this exact 12D checkpoint under the same `--max_trans_step 0.04` —
  worth checking whether the raised clamp alone (without admittance) would
  have also completed the task, to isolate how much the force correction
  itself contributed versus just having a larger motion budget per step.

---

## Quantitative evaluation, drift investigation, and a real pipeline bug found (2026-07-13)

### Offline force-prediction accuracy: episode 037 vs. 020

Built `eval_force_prediction.py` (reuses `05_train_replay_force.py`'s dataset
+ `test_policy.py`'s real DDPM inference, no robot needed) to compare
predicted vs. actual recorded force on a held-out episode, against a naive
"assume force stays constant" baseline.

| Episode | Model MAE | Baseline MAE | Result |
|---|---|---|---|
| 037 (held out during the first `PastaTransfer_force_replay_force` training run) | 1.754N | 0.955N | Model **worse** than baseline by 83.8%, worst on Fx (2.946 vs 0.696) |
| 020 (was in that run's *training* set — not a clean generalization test) | 0.393N | 0.866N | Model **beats** baseline by 54.6%, wins on every axis and every horizon step |

The 037 result wasn't just "noisier" — model error was flat and bad from
horizon step 1 onward (not degrading gracefully with distance into the
future), the signature of a systematic mismatch rather than a hard-to-predict-far-ahead problem.

### Found: episodes 031-037 are a high-drift outlier cluster, traced to a session boundary

Checked "drift" (mean force in the last 3s of an episode's recording minus
the first 3s) across all 37 episodes with replay force logs:

| Episodes | Drift magnitude |
|---|---|
| 001-030 | mean 1.78N, median 1.57N, range [0.53, 3.39] |
| **031-037** | **mean 5.12N, median 5.37N, range [3.24, 6.55] — 2-3x higher** |

Cross-referenced recording timestamps: 025-030 recorded 15:15-15:48
(2026-07-10), an 11-minute gap, then 031-037 recorded 15:59-16:47 — same day,
same session. The drift jump lines up almost exactly with that gap,
consistent with a session-cumulative effect (most likely thermal — motor/
joint heating over a continuous multi-hour session) rather than random
per-episode noise. Episode 037 — the episode used for validation in the
first training run — sits in this anomalous cluster, which plausibly
explains its poor generalization result above more than any fundamental
flaw in the approach.

**Live check, same day**: deployed the existing (position-only,
non-force) `PastaTransfer_force_masked_dualcam_h16_ep20-37` checkpoint live
and measured the *current* live force drift over a verified, complete
95.3s run: **1.36N** — squarely in the 001-030 "normal" range, nowhere near
the 031-037 cluster. Confirms the anomaly was specific to that one
2026-07-10 session, not a persistent property of this robot's current
calibration state.

(Note: an early version of this comparison was accidentally lost — a
position-only run's `force_log.npz` got silently overwritten by a
same-named hybrid run's log, since both used the same default
`--output_dir`. Fixed by timestamping the filename
(`force_log_<timestamp>.npz`) in `07_deploy_force.py`, so this can't recur.)

### New training variant: force as conditioning-only (no force in the output)

Extended `05_train_replay_force.py` (still the same file — additive,
backward-compatible flag, not a new copy) with `--no_predict_force`: force
still feeds the observation (proprio stays 12D), but the action target is
pure 9D pose — the model is never asked to predict future force, only
better positions. Isolates "does force-conditioning improve pose
prediction" from "can this model also predict force" (the thing that failed
on episode 037). Checkpoint metadata now stores `force_in_proprio`
separately from `predicts_force`, so `07_deploy_force.py` can build the
right-shaped proprio observation regardless of which variant is loaded
(previously it incorrectly inferred this from `predicts_force` alone, which
would have broken this variant's shape).

Two more real bugs caught by an offline smoke test before ever touching the
robot: `policy_common.MaxAbsNormalizer.normalize`/`denormalize` didn't
handle a 9D action being scaled against a 12D-fit normalizer (shape
mismatch) — fixed generically (auto-slices to match, no-op when dimensions
already agree, so every existing call site is unaffected).

### Retrained both variants on episodes 001-030 only (excluding the high-drift cluster)

`PastaTransfer_force_replay_force_ep1-30` (combined, predicts force) and
`PastaTransfer_force_cond_only_ep1-30` (conditioning-only), both with
episode 020 held out for validation, both trained concurrently (GPU
headroom was fine — 29 episodes each is much less contended than the
earlier 4-6-concurrent-job problem). Same overfitting shape as every other
run this project: val loss bottomed around epoch 100-110 for both (combined
0.0422, cond-only 0.0449) then rose — manually stopped at epoch ~290 once
this was confirmed, `policy_epoch0100.pt` is the best-available saved
checkpoint for both (checkpoints saved every 100 epochs; epoch 100 is very
close to the true minimum for both).

### Found + fixed: live force was missing an entire correction stage vs. training labels

While reading `analyze_replay_full.py` (which builds `replay_full_forces.npz`,
the source of every training label used today) to answer a user question:
its `external_force_xyz` field is the **third** of three correction stages —
firmware → +gravity (regressor+NN) → **+mass/Coriolis dynamics regressor**
(`full_dynamics_regressor(q,qdot,qddot) @ dynamics_residual_pi`). But
`07_deploy_force.py`'s live `read_F_raw` only ever did the first two stages —
copied from `deploy_streaming.py`'s 100Hz loop, which explicitly left the
dynamics regressor out for being too slow (confirmed ~26Hz max, 39ms/call).
That constraint was never re-checked for this script's 10Hz loop (100ms
budget — the regressor uses under 40% of one tick, comfortably fine).

Net effect: **every live force reading all day, in every script, was on a
different correction chain than what the model was actually trained on** —
training labels were fully dynamics-compensated, live reads were not.
Fixed: added `get_qdot_deg`/`GetAngularVelocity` binding, a persistent
`contact_detector.VelocityDifferentiator` for live qddot estimation, and the
missing regressor subtraction, matching `analyze_replay_full.py`'s chain
exactly. New `--dynamics_pi`/`--qddot_smoothing` args (defaults match the
training-label pipeline).

### Read Modern Robotics (Lynch & Park) Section 11.5, Force Control — implementability assessment

Section 11.5's actual force-control law commands **joint torque** directly
(`τ = g̃(θ) + Jᵀ(θ)[Fd + Kfp·Fe + Kfi·∫Fe dt − Kdamp·V]`), requiring reliable
torque-control mode. Not implementable here — direct torque control on this
Jaco2 was already found unreliable earlier in this project (API reports
success but control fails; needed a power cycle), so we deliberately stay
in position control for all force work. What we've built all day
(`x_cmd = x_spline + K⁻¹·Fe`, staying in position-control mode) is the
textbook's own prescribed alternative for exactly this situation — Section
11.7, Impedance/Admittance Control, defined as `Y(s)=X(s)/F(s)` (force in,
position out), literally the structure we already had. Concrete, doable
improvement identified but not yet implemented: our correction is
proportional-only; adding an integral term (`Kfi·∫Fe dt`, reduces
steady-state force error) and a velocity-damping term (`−Kdamp·V`, prevents
runaway correction when there's nothing to push against) would bring it
closer to the book's refined law, within the same position-control-only
constraint.

### Live re-test with the fixed pipeline + new (episodes 1-30) checkpoints

All three configurations ran cleanly (no crashes) with the corrected force
pipeline:
- Conditioning-only, `--exec_steps 8`: task completed successfully.
- Conditioning-only, `--exec_steps 16` (full horizon executed before
  replanning, vs. the usual receding-horizon 8): task completed, "felt about
  the same" — no obvious difference from the shorter execution horizon.
- Combined/hybrid, `--admittance --max_trans_step 0.04`: task completed,
  but **subjectively more forceful/aggressive during the task** than the
  earlier (pre-fix) hybrid run.

Investigated the "more aggressive" report directly from the logs (`t`,
`F_live`, `q_deg` — position reconstructed via forward kinematics from
`q_deg`, `07_deploy_force.py`'s own FK chain): mean `|F|` was actually
similar to before (2.77N vs. 2.70N), but max jumped to 26.2N (vs. 13.7N
before, and vs. episode 021's own reference max of 10.25N). Traced the peak
to steps 18-24 (t=4.8-7.2s): Z drops from 19cm to 12.6cm (scoop/box-entry
descent) with strongly negative Fz throughout, rising and falling smoothly
over ~2.4s — the "real push" signature `analyze_replay_full.py` itself
describes (smooth rise-peak-fall over hundreds of ms to seconds), not an
isolated noisy sample. Only 3.5% of the run's steps exceed the reference
episode's own peak, 5% exceed its mean — the elevated force is localized to
the genuine scoop-contact event, not a run-wide effect. Best current
interpretation: the fix is working as intended — the old, dynamics-
contaminated pipeline likely *under-reported* this same contact event,
since the contamination scales with velocity/acceleration and this is
precisely the fastest-moving part of the task.

### Repeated hybrid trials (4 total) + matched position-only control -- genuine benefit found

Ran the hybrid config (`ep1-30` combined checkpoint, `--admittance
--max_trans_step 0.04`, episode 021 init) 3 more times (4 total including
the one above), then the **identical** checkpoint/settings with
`--admittance` **off** as a matched control -- the one comparison missing
from every earlier attempt today.

| Condition | \|F\| mean | \|F\| max | drift | path length |
|---|---|---|---|---|
| Position-only (admittance off) | 2.18N | **4.46N** | 2.27N | 103.6cm |
| Hybrid (4 trials, admittance on) | 2.57 ± 0.16N | **15.68 ± 6.34N** | 2.10 ± 0.29N | 138.3 ± 5.6cm |
| Reference (episode 021 demo) | 5.70N | 10.25N | 3.43N | -- |

Two things stand out. First, **reproducibility**: across the 4 hybrid
trials, `|F|mean` (±0.16N, ~6%) and path length (±5.6cm, ~4%) are tight --
this is a stable, repeatable controller, not a one-off fluke (trial0's 26N
peak was actually the outlier of the four; trials 1-3 peaked at 9.7-14.8N,
much closer to the reference's own 10.25N).

Second, and more importantly: **with everything else held identical, the
one variable that changed (admittance on/off) produced a real, large
effect** -- peak contact force was 3-6x higher with admittance on, and the
position-only run *never* exceeded 4.46N the whole run (well below even
the smallest hybrid trial's peak, and well below the human demonstration's
own 10.25N peak). Mean force only rose modestly (~18%), but peak force and
path length (~33% longer) both point the same direction: the position-only
policy looks like it's systematically under-committing to contact --
moving through the scoop without pushing hard enough to match the
demonstrated force profile -- while the admittance correction consistently
drives real contact forces into the same range as the human reference.

This is the first clean evidence this session that the force correction is
doing more than "complete the task while feeling different" -- it's
measurably closer to the demonstrated contact behavior than position-only
control, using the same checkpoint and the same non-force settings
otherwise.

Ran 2 more position-only control trials (3 total) to put the same
confidence interval on that side as the hybrid side already has:

| Condition | \|F\| mean | \|F\| max | path length |
|---|---|---|---|
| Position-only (3 trials) | 2.17 ± 0.16N | **5.09 ± 0.58N** (range 4.46-5.87) | 109.5 ± 5.0cm (range 104-116) |
| Hybrid (4 trials) | 2.57 ± 0.16N | **15.68 ± 6.34N** (range 9.66-26.21) | 138.3 ± 5.6cm (range 133-148) |
| Reference (demo) | 5.70N | 10.25N | -- |

With 7 total trials, the peak-force and path-length ranges **don't overlap
at all** between conditions -- position-only never exceeds 5.9N across 3
runs, hybrid never drops below 9.7N across 4. Same for path length (104-116cm
vs. 133-148cm). This is now solid, low-variance evidence on both sides, not
a single noisy comparison -- the admittance correction reliably produces
higher peak contact force (closer to the 10.25N human-demonstrated
reference) and a longer effective task path than position-only control,
holding the checkpoint and every other setting identical.

### Open items

- The mass/Coriolis fix changes what "expected/reference force" means for
  every earlier comparison this session that used the old (gravity-only)
  live pipeline -- those numbers aren't directly comparable to anything
  measured after this fix.
- Still untested: the conditioning-only variant (`PastaTransfer_force_cond_only_ep1-30`)
  against a true 9D-no-force baseline, to isolate whether force-as-input
  alone (no admittance at all) already improves position predictions, the
  original motivation for building that variant.

### Sparse-pasta test: does admittance help scoop when little pasta is left?

Motivated by the peak-force finding above -- if position-only reliably
under-commits to contact, it might be more likely to glide over sparse
remaining pasta rather than press down enough to gather it. Tried to test
this directly; took a few iterations to find the right amount (first two
attempts left the box essentially empty -- no pasta for *either* controller
to contact, not an interesting test; force stayed near baseline and
correction rarely saturated in that condition, consistent with "nothing to
push against" rather than telling us anything about the hypothesis).

With a genuinely sparse-but-nonzero amount (confirmed by live force
readings showing real contact, up to ~3.8N, not just noise):
- **Position-only**: 1 trial, failed to scoop.
- **Hybrid**: 2 trials (box topped up slightly before these, then repeated
  without further changes), **both succeeded** -- scooped and transferred.

Also captured, for the first hybrid sparse-pasta trial, a direct
force-vs-position decomposition (new logging added to `07_deploy_force.py`:
`pos_cur_before`/`pos_pred_raw`/`correction`/`pos_clamped` per tick, since
the correction magnitude wasn't being saved before, only printed): the
force correction contributed **~39% of the combined per-step signal**
(median 41%), was the *dominant* contributor (bigger than the position
policy's own intended step) on **22% of ticks**, and only hit the hard 2cm
cap 8.3% of the time in this run -- much less saturated than the
normal-pasta hybrid trials, suggesting the predicted force reference
tracked live force more closely here.

**Caveat, stated plainly**: this is N=1 vs N=2, and the box wasn't held
under identical conditions across the comparison (refilled slightly
between the position-only failure and the hybrid attempts) -- not a clean,
controlled result the way the 7-trial normal-pasta comparison was. Treat it
as a suggestive, directionally-consistent data point, not proof. Worth
repeating with the pasta amount held fixed across both conditions if a
firmer answer is wanted.

### 50% pasta test -- result flips, and traced to a specific stuck-at-low-Z failure mode

Ran the matched comparison again at a fixed, deliberately-measured 50%
pasta level (topped up from the sparse test): **position-only succeeded
twice (2/2)**, **hybrid failed twice (2/2)** -- the opposite result from
the sparse-pasta test above. Checked both failed hybrid logs directly
(position via FK from `q_deg`, plus the new `correction`/`pos_pred_raw`
logging):

Both failed trials show the same signature: the arm descends into the box
normally (Z: ~23cm -> 6-8cm), then **gets stuck oscillating at that low Z
for the rest of the run** (never climbs back out), with the correction
magnitude frequently pinned at the 2cm cap during the stuck phase. Compare
to the successful sparse-pasta hybrid trial, which shows a clean transition
-- down to ~6cm, then decisively back up to ~20cm and staying there (the
lift-and-carry phase). The saturated correction during the stuck phase
suggests the admittance term is actively fighting the transition out of
contact, not just failing to help.

**Root cause, confirmed directly with the user**: episodes 001-030 (the
entire training set for this checkpoint) were recorded with a **constant
pasta fill level** -- there is no fill-level variation anywhere in the
training data for the model to have learned a force-vs-fill-level
relationship from. `f_desired` is therefore a fixed reference tied to
whatever force followed each point in the trajectory during training,
with no mechanism to adapt to how much material is actually in the box on
a given run. Under that lens, the inconsistent pattern across pasta levels
(sparse: hybrid wins 2/2 vs position-only 0/1; 50%: position-only wins 2/2
vs hybrid 0/2) is exactly the expected shape of **uncontrolled
out-of-distribution extrapolation**, not evidence the mechanism is
unreliable in general. Sometimes the fixed reference happens to
approximately match the actual contact dynamics (helps, or is neutral);
sometimes it doesn't (the correction fights the natural trajectory instead
of assisting it, producing the stuck-at-low-Z pattern).

**Implication**: today's varying-pasta-amount tests do not actually test
whether force-conditioning *generalizes better* to physical variation --
they test what an untrained-for extrapolation looks like, which is
inherently unpredictable. To properly test the original hypothesis (the
motivation for this whole line of work -- "force should generalize better
than vision to variation vision can't see"), training data would need to
span multiple fill levels, so the model has an actual chance to learn how
force should relate to how much material is present, rather than replaying
one fixed trajectory's force regardless of what's really in the box. The
clean, controlled result that still stands from today is the *full-pasta*
7-trial comparison above -- hybrid reliably generates higher peak
force/more path length than position-only when conditions match training.
Whether that translates to better task success under conditions the model
was never shown remains open, and this data suggests it can go either way.

All 24 logs from today's evaluation (both this section and the ones above)
copied to `deploymentRuns/2026-07-13_force_conditioning_eval/` for future
reference, with a `README.md` indexing which checkpoint/mode/pasta-level/
outcome each file corresponds to.

### Confirmatory re-test at 50% pasta: does hybrid still generate more force?

One more matched pair (position-only, then hybrid) at the fixed 50% level,
specifically to check whether the *force-generation* difference (not task
success) still holds outside the training distribution:

| Condition (50% pasta) | \|F\|mean | \|F\|max | path length |
|---|---|---|---|
| Position-only | 2.38N | **7.91N** | 121.3cm |
| Hybrid | 2.60N | **7.33N** | 171.9cm |

The peak-force gap that was completely reliable at full pasta (hybrid
9.7-26.2N vs. position-only 4.5-5.9N, zero overlap across 7 trials) is
**gone** here -- position-only's peak is actually marginally higher than
hybrid's this time, and mean force is nearly identical. Only the
path-length effect survives (hybrid still ~42% longer, consistent with the
full-pasta finding).

This directly confirms the training-data explanation above: the "hybrid
generates more force" result is conditional on matching what the model was
trained on (constant, full fill level), not a general property of the
mechanism. Outside that distribution, the force-generation advantage
disappears along with task-success reliability. **Bottom line for this
whole line of investigation**: force-conditioning-driven admittance control
is real, learnable from human demonstrations, and produces a measurable,
reliable effect (peak force, path length) when deployed under matching
conditions -- but nothing tested today shows it generalizing *better* than
position-only to conditions the training data didn't include. Testing that
properly would require training data that itself varies the thing being
generalized over (pasta fill level, here).

### Execution horizon (`--exec_steps`) also modulates the effect size, back at full pasta

With the box refilled to full, ran matched pairs at `--exec_steps 16`
(execute the full 16-step/1.6s predicted horizon before replanning,
instead of the default 8) to see if the full-pasta force-generation effect
depends on this setting too.

| exec_steps | Position-only \|F\|max | Hybrid \|F\|max | Path (hybrid vs. pos-only) |
|---|---|---|---|
| 8 (original 7-trial set) | 5.09 +- 0.58N (range 4.46-5.87) | 15.68 +- 6.34N (range 9.66-26.21) | 138.3 vs. 109.5cm |
| 16 (n=3 each) | 6.90 +- 0.37N | 11.15 +- 4.24N (range 7.5-17.1) | 136.6 vs. 101.8cm |

The effect survives at exec_steps=16 -- hybrid still higher on every trial
-- but the gap shrinks substantially (~3x separation at exec_steps=8 down
to ~1.6x at exec_steps=16), and hybrid's peak force becomes noticeably
noisier (std 4.24 vs 0.37 for position-only). Plausible mechanism:
exec_steps=16 replans half as often, so there are fewer "fresh chunk"
transition moments -- and if large corrections concentrate specifically at
those transitions (consistent with earlier deploy_streaming.py findings
this project), fewer transitions per unit time means fewer/smaller spikes.

**Then re-ran exec_steps=8 once more (back to the original setting) to
double check**: position-only \|F\|max=7.16N, hybrid \|F\|max=11.54N --
hybrid still ahead (~1.6x), but position-only's own peak here is *higher*
than any of its original 3 exec_steps=8 trials (4.46-5.87N), narrowing
today's gap similarly to the exec_steps=16 result even though this run used
the exec_steps=8 default. This suggests at least part of the narrowing
across today's later trials may be a **session-drift effect** (the same
kind of cumulative calibration/thermal drift found earlier in the project
for episodes 031-037) rather than purely an exec_steps effect -- we ran
several dozen trials over a few hours, and this is exactly the kind of
signature that showed up before. Not confirmed, just the most likely
explanation given today's other findings.

**Status at end of day**: the core result (hybrid generates measurably
more peak force and moves through more of the task, matching training
conditions) held up across every full-pasta trial run today, all session,
regardless of exec_steps -- but the *magnitude* of that gap varied
noticeably across the session, likely due to some combination of
exec_steps and session-level drift. Continuing tomorrow; open items above
(fill-level training data, more trials to separate exec_steps from
session-drift, conditioning-only-vs-no-force baseline) are all still on
the table. All logs from today (36 total) are archived in
`deploymentRuns/2026-07-13_force_conditioning_eval/` with a full README index.

### Identified but not yet implemented: our admittance law is stiffness-only, missing inertia/damping

`07_deploy_force.py`'s correction (`correction = (F_live - f_desired) / K`)
implements only the stiffness (`k`) term of the full second-order impedance
relationship from *Modern Robotics* Eq. 11.62 (`m*x_ddot + b*x_dot + k*x = f`)
-- no inertia (`m`) or damping (`b`) term anywhere in the law. It's a
memoryless, purely proportional map from force error to position
displacement, recomputed independently every tick.

This is a plausible explanation (not confirmed) for the "stuck at low Z,
oscillating, correction pinned at the cap" failure signature found in the
50%-pasta hybrid trials above -- a proportional-only (undamped) feedback
loop hunting around a reference it can't quite satisfy, rather than
settling, is a textbook symptom of missing damping. It also directly
connects back to Section 11.5's own refined force law, which adds a
`-Kdamp*V` term for exactly this reason ("if there is nothing to push
against, it will accelerate in a failing attempt..."), something we noted
as a "concrete, doable improvement" when we first read that section but
never followed up on.

**Next step, not yet done**: add a velocity-damping term to the correction
(`correction = (F_live - f_desired)/K - Kdamp*V_actual`, using Cartesian
velocity already derivable from consecutive position reads). Inertia is a
separate, more involved addition (would need the correction to have actual
second-order dynamics of its own, not just filtering the F_live
measurement) -- lower priority, worth revisiting only if damping alone
doesn't resolve the oscillation pattern.
