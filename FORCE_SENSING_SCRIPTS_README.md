# Force-Sensing Scripts — How to Use Them

Practical usage guide for every script built for gravity/dynamics
calibration and external-force sensing on the Kinova Jaco2. For *what we
found* (numbers, conclusions, which approach won), see
`FORCE_SENSING_FINDINGS.md` — this file is about *how to run things*.

## The pipeline, in one picture

```
raw torque (GetAngularForce)
  -> GetAngularForceGravityFree            [firmware gravity matrix]
  -> - gravity_regressor(q) @ phi          [software gravity regressor]
  -> - gravity_residual_nn(q)              [gravity residual NN]
  -> - full_dynamics_regressor(q,qdot,qddot) @ pi   [mass/Coriolis regressor]
  -> tau_ext  ->  torque_to_wrench()  ->  F_ext (the external force you want)
```

Every "live" or "analysis" tool below computes some prefix of this chain.
The fitted numbers that make each stage work already exist on disk (see
"Data files" at the bottom) — you do **not** need to redo any calibration to
just go look at live forces. Skip straight to "Quick start" if that's all
you want.

---

## Quick start: "I just want to see live external force right now"

**Option A — while replaying a tracked episode:**
```bash
# terminal 1
python diag_replay_forces.py
# terminal 2 (starts sending motion once terminal 1 says "Found it")
python replay_episode.py --episode_dir data/episodes/pastaTransfer4/<NNN> --execute
```

**Option B — while hand-guiding with the joystick (no replay needed):**
```bash
python diag_live_forces.py
```
Then just move the arm with the joystick. This one connects directly and
never sends a motion command — safe to run anytime.

In both cases, watch the blue **"final estimate"** line. It should sit near
a ~1-3N noise floor with nothing touching the arm, and spike (a smooth
rise-peak-fall over ~0.3-0.5s, not a single jagged sample) when something
pushes on it.

---

## End-to-end recipe: recording a NEW task, then replaying with force logging

Worked example for a task called `PastaTransfer_force` — substitute your own
task name and mesh. Uses `data/robot_extrinsics_stick_corrected_zmeasured.npy`
(the Z-corrected robot-base calibration — see README.md Step 0b) and
`newspoon1.obj` throughout; swap in whatever's current for you.

### Fastest path: `record_and_replay_episode.py` (one command, per episode)

Chains everything below (record → auto-track → visualize → **pause for you to move the tool onto
the gripper** → base-frame convert → dry-run replay → **confirm** → real replay with
`--capture_camera` → calibrated force analysis + per-frame image tagging) into one command per
episode. Each stage is still the same underlying script, run as a subprocess — this is a
convenience wrapper, not new logic, and it aborts the whole run if any stage fails:

```bash
python record_and_replay_episode.py --task PastaTransfer_force \
    --mesh newspoon1.obj --tool_prompt "spoon"
```
Episode ID auto-increments from what's already under `data/episodes/PastaTransfer_force/` if you
don't pass `--episode`. Two human-in-the-loop pauses are built in on purpose: one after
recording+tracking (attach the tool to the gripper, press ENTER), one after the dry run (confirm
before the arm actually moves for real, or Ctrl+C to abort) — pass `--skip_confirm` to skip the
second one only once you trust the pipeline for a given setup. `--skip_record --episode NNN` reuses
an already-recorded/tracked episode and only runs the replay+force stages.

Read on for what each stage does individually, or if you want to run/debug them one at a time.

### 1. Record demos AND track them in the same command (`--auto_track`)

`01_record.py --auto_track` runs the accurate offline tracking pass (what used to be a separate
step 4) automatically right after each recording finishes, reusing the already-loaded FP/GDINO/SAM
models — one command per episode instead of record-then-track-then-visualize separately:

```bash
python 01_record.py --task PastaTransfer_force --episode 001 --duration 15 \
    --mesh newspoon1.obj --tool_prompt "spoon" --track_cam 1 --auto_track
```
SPACE to (re-)register on the table, ENTER to start recording, Q to stop early. Bump `--episode`
for each new one (002, 003, ...).

After recording, it automatically:
1. Tracks the spoon through the whole clip at **high quality** (`--auto_track_refine_iter 5`,
   `--auto_track_est_refine_iter 8` by default) — deliberately higher than the live-preview
   overlay's fast/cheap settings (`--track_refine_iter 2`), which exist only to keep up with 30fps
   capture and were found to cause severe drift if used for the saved trajectory on long clips.
2. **Always tracks RAW frames, never masked.** On `pastaScoop/001` we found the hand-segmentation
   UNet's mask clipped spoon geometry near the grip closely enough that FoundationPose froze
   mid-episode (right as the spoon dipped into the pasta box) and silently stayed frozen for the
   rest of the clip — despite looking fine in the preview. Raw frames + higher offline refinement
   fixed it completely. So `03_segment.py` is skipped entirely for auto-tracked episodes.
3. Saves `tool_poses_cam{N}.npz` / `tool_poses_task.npz` (same output as running `04_track.py`
   separately).
4. Runs `visualize_poses.py` automatically and tells you where to look
   (`.../augmented/viz_poses/*_cam{N}_pose.jpg`) — pass `--skip_auto_visualize` to skip this.

**Always spot-check a few episodes** before trusting the batch — look at frames spanning the
*whole* clip, not just the start. Check the box stays locked on the spoon all the way through
(including while it's in/near the pasta box), and that its distance from the robot base stays
physically reachable (~35-70cm for this workspace) — a value like >100cm, or a position that
never recovers after dipping into the box, means tracking silently failed for that stretch.

*(Prefer the old two-step flow, or need to re-track without re-recording? `04_track.py
--task_dir ... --track_refine_iter 5 --est_refine_iter 8 --skip_done` still works standalone,
batched over a whole task directory — same accuracy, just not fused into the recording step.)*

### 2. Convert to robot base frame (batch, once you've recorded/tracked some or all episodes)
```bash
python 04c_to_base_frame.py --task_dir data/episodes/PastaTransfer_force \
    --robot_extrinsics data/robot_extrinsics_stick_corrected_zmeasured.npy --skip_done
```

### 3. Replay on the robot + force-sense, one episode at a time (real hardware)

Dry run first (no `--execute`) to sanity-check the waypoint plan before moving the robot. Then,
two terminals for the real run:

```bash
# terminal 1 — sends the motion, auto-saves torque_log_dense.npz (raw force data) AND
# torque_log.npz (sparse, one sample per waypoint -- needed below for image/force tagging).
# --capture_camera saves a real RGB image from EVERY connected camera at every waypoint
# (data/.../replay/cam0/{frame_id}.jpg, cam1/{frame_id}.jpg, ...).
python replay_episode.py --episode_dir data/episodes/PastaTransfer_force/001 \
    --mesh newspoon1.obj \
    --T_eef_spoon data/T_eef_spoon.npy \
    --robot_extrinsics data/robot_extrinsics_stick_corrected_zmeasured.npy \
    --execute --capture_camera \
    --no_wait_convergence --speed_scale 0.3 --arm_trans_speed 0.05

# terminal 2 (optional) — live 3-tier external-force plot while it runs
python diag_replay_forces.py
```
`--no_wait_convergence --speed_scale 0.3 --arm_trans_speed 0.05` gives smooth continuous motion
instead of stop-start jitter at each waypoint — tune to taste. Default `--subsample 3` means only
every 3rd original frame becomes a waypoint (150 of 450) — pass `--subsample 1` to `replay_episode.py`
if you want an image + tagged force at every original frame instead.

### 4. Post-hoc calibrated force analysis, and tagging forces to image/waypoint instants

`replay_episode.py --execute` already saved the raw torque/velocity logs — this step turns them
into the actual calibrated external-force estimate (3-tier: firmware-only / +gravity / final):
```bash
python analyze_replay_full.py --episode_dir data/episodes/PastaTransfer_force/001
```
Saves three things per episode:
- `.../replay/replay_full_forces.png` — plot
- `.../replay/replay_full_forces.npz` — the full dense-rate data: `external_force_xyz` (N, the
  pure external force, base frame), `external_moment_xyz`, all three tiers' norms
- `.../replay/replay_forces_per_frame.npz` — **the same force data resampled at each waypoint's
  time instant**, tagged with `frame_id` matching `--capture_camera`'s saved image filenames
  exactly (`{frame_id:06d}.jpg`) — this is how you pair a specific image with the force at that
  same moment. Also prints mean/std/min/max per tier and the peak detected force.

To batch this over everything you've replayed so far:
```bash
for log in data/episodes/PastaTransfer_force/*/replay/torque_log_dense.npz; do
    python analyze_replay_full.py --log_path "$log" \
        --plot_path "$(dirname "$log")/replay_full_forces.png" \
        --output_npz "$(dirname "$log")/replay_full_forces.npz" \
        --output_per_frame_npz "$(dirname "$log")/replay_forces_per_frame.npz"
done
```

Repeat steps 3-4 for each episode you want to replay. Once steps 1-2 have produced
`tool_poses_base.npz` for enough episodes, that same `data/episodes/PastaTransfer_force/`
directory is what you'd point `05_train.py` at to train the diffusion policy.

### 5. Generate novel views from the REPLAY images (for training on replay instead of demo images)

```bash
for ep in 001 002 003 004; do
    cp data/episodes/PastaTransfer_force/$ep/meta.json \
       data/episodes/PastaTransfer_force/$ep/replay/meta.json

    python 02_augment_noposplat.py \
        --episode_dir data/episodes/PastaTransfer_force/$ep/replay \
        --noposplat_root ~/working_dir/NoPoSplat \
        --input_mode letterbox --no_antialias \
        --num_novel_views 6 --sample_every 3 --skip_done
done
```
`--sample_every` here must match the replay's `--subsample` (3 by default) — replay images only
exist at the frame_ids that became waypoints, not a dense sequence, so a mismatch means most
requested frames silently get skipped as "missing image." Output lands in
`<episode_dir>/replay/augmented/{real,novel,novel_cameras.npz}` — separate from the original
demo's own `augmented/` folder (which holds the tracked poses), so nothing collides.

### 6. Train on the replay images instead of the human-demo images

```bash
python 05_train_replay.py \
    --data_dir data/episodes/PastaTransfer_force \
    --output_dir data/checkpoints/PastaTransfer_force_replay \
    --track_cam 1 --subsample 3 --action_frame task --val_episodes 020
```
Replay images already show the robot's own gripper doing the task (matching what deployment's
live camera will see), so there's no human hand to mask out — this trains on the **unmasked**
`replay/augmented/real` + `novel` images. Action/proprio labels are unchanged (still from the
original demo's tracked trajectory — replay is the robot re-executing that exact trajectory, so
the pose label at a given frame_id is still correct). Doesn't modify `05_train.py` at all — its
`parse_args()`/model/checkpoint-saving are reused unchanged; only the training loop itself is
its own, since it additionally tracks validation loss (see next paragraph), which the shared
`train()` doesn't support.

**Validation split:** `--val_episodes 020` (or any episode ID(s)) holds that episode out of
training and reports validation loss alongside training loss every 10 epochs — that comparison is
the actual overfitting signal (training loss keeps dropping regardless; validation loss
plateauing/rising while training loss keeps falling means it's overfitting). With the current 20
recorded episodes (~2660 windows total across all of them, ~2GB of replay images), holding out 1
leaves ~2527 training windows and ~133 validation windows. The validation set reuses the training
set's fitted normalizer rather than fitting its own, so both losses stay on the same scale.
Early stopping switches from training-loss EMA to validation-loss EMA automatically once
`--val_episodes` is given.

**Deploy with matching, unmasked observations:**
```bash
python 07_deploy.py \
    --checkpoint data/checkpoints/PastaTransfer_force_replay/policy_final.pt \
    --no_arm_mask \
    --track_cam 1 --frequency 10 --exec_steps 16
```
`--no_arm_mask` feeds the raw live camera image to the policy instead of masking the robot arm
out — required here, since training was unmasked. Using `--no_arm_mask` with a policy trained by
the regular `05_train.py` (masked human-demo images), or omitting it with a `05_train_replay.py`
checkpoint, is a train/deploy mismatch.

---

## 1. Core module — `contact_detector.py`

Not run directly (except its self-test). Everything else imports from it.

```bash
python contact_detector.py   # self-test: FK/Jacobian/RNEA sanity checks, ~2s, no hardware
```

Key functions:
- `gravity_regressor(q)`, `rnea_no_gravity(q,qdot,qddot)` — nominal (uncalibrated) physics
- `rnea_full(q,qdot,qddot,link_params)`, `full_dynamics_regressor(q,qdot,qddot)` — the reparametrized, exactly-linear mass/Coriolis regressor (72 params)
- `torque_to_wrench(q,tau,damping=0.05)` — maps joint torque to an EEF wrench via the Jacobian transpose (damped near singularities)
- `recover_external_force(q,tau_gf,phi,...)` — the older, gravity-regressor-only clean-force function (superseded by the fuller pipeline above but still used internally in a few places)

---

## 2. Gravity calibration

Run in roughly this order if starting from scratch; **skip all of this** if
you just want to use the already-fitted files (see "Data files").

### 2.1 `calibrate_firmware_gravity.py` — firmware gravity matrix

```bash
python calibrate_firmware_gravity.py
```
Runs Kinova's own `RunGravityZEstimationSequence`. **This is the highest-risk
script in the whole toolkit** — it's a single opaque autonomous SDK call that
moves the arm for several minutes with no software abort available. Requires
an explicit Enter-press confirmation after printing a warning. Only re-run
this if the currently-saved fit (see below) is actually bad — it already
produced a clean, validated result once; there's no need to redo it.
Saves `data/gravity_params.npy` + `data/gravity_params_meta.npz`.

### 2.2 `upload_gravity_params.py` — reapply the firmware fit

```bash
python upload_gravity_params.py
```
The firmware fit does **not** survive a power cycle. This reapplies the
saved `data/gravity_params.npy`. Read/write only, no motion, safe anytime.
Already wired automatically into the startup of every script below — you
only need this standalone if you're writing a new script that doesn't
already call `apply_saved_gravity_params()`.

### 2.3 `collect_task_poses.py` — linear gravity regressor (recommended)

```bash
python collect_task_poses.py --task pastaTransfer4 --episodes 003 008 013 020 027 034 041 048 \
    --frames_per_episode 5 --execute
```
Fits `phi` (the linear `gravity_regressor(q) @ phi` correction) from REAL
task-episode poses — dramatically better and more consistent than the
random-pose alternative (`calibrate_gravity_residual.py`, kept for reference
only, not recommended). Moves the arm to each pose via Cartesian control,
averages torque. Accumulates into `--session_records_path` across multiple
runs (default `data/gravity_calibration_records_task_only.npz`). Re-fits and
saves `data/gravity_phi_task_only.npy` every time it's run.

### 2.4 `evaluate_task_poses.py` — held-out check

```bash
python evaluate_task_poses.py --task pastaTransfer4 --episodes 006 016 026 036 046 \
    --frames_per_episode 3 --phi data/gravity_phi_task_only.npy --execute
```
Pure evaluation — visits poses from episodes NOT in the training set, reports
before/after ||F||, never touches the training data or refits anything.

### 2.5 `fit_gravity_residual_nn.py` — gravity residual NN (best gravity result)

```bash
python fit_gravity_residual_nn.py
```
No hardware motion at all — reuses already-collected data (the 40 dedicated
poses + near-static stretches mined from dynamics-excitation episode logs,
see `extend_gravity_from_excitation.py`'s `extract_static_stretches`). Fits
a small NN, input `sin(q),cos(q)` only, on top of `gravity_phi_task_only.npy`.
This closes far more of the residual than re-fitting the linear model with
more data ever did — **run this after 2.3, always, it's essentially free**.
Saves `data/gravity_residual_nn.pt`.

### 2.6 `extend_gravity_from_excitation.py` — superseded, reference only

Tried extending the *linear* gravity fit with more poses mined from motion
logs. Only marginally helped (~5-7%). Superseded by 2.5. Kept because its
`extract_static_stretches()` helper is reused by `fit_gravity_residual_nn.py`.

---

## 3. Mass/Coriolis dynamics residual

### 3.1 `dynamics_dataset.py` — shared dataset builder (not run directly)

`build_dataset(episodes, task, phi_gravity, gravity_residual_fn=...)` builds
the target (`gravity-free - gravity(regressor+NN) - nominal RNEA`) and the
72-column `Y_full` regressor input for a list of episodes. Expensive
(~25ms/sample) — building a 30-episode training set takes ~15 min of pure
CPU time, no hardware. Imported by the fit/compare scripts below.

### 3.2 `fit_dynamics_regressor.py` — linear regressor (recommended for deployment)

```bash
python fit_dynamics_regressor.py
```
Fits the 72-parameter ridge regression on `full_dynamics_regressor`. Saves
`data/dynamics_residual_pi.npy`. **This is the one to actually use** — tied
with the NN in accuracy, far simpler and more stable.

### 3.3 `fit_dynamics_nn_constrained.py` — the correct NN (module, reference)

Not run standalone. `ConstrainedResidualNet`: two heads,
`M_res(q) @ qddot + C_res(q,qdot) @ qdot`, mathematically zero at rest by
construction. Use this one if you need the NN version for any reason —
**never** `fit_dynamics_nn.py` (next).

### 3.4 `fit_dynamics_nn.py` — deprecated, DO NOT USE

The original unconstrained NN (`[sin q, cos q, qdot, qddot] -> tau` directly).
Proven to leak gravity-model error into its output (a zero-velocity probe
showed it predicts ~1.5N even at synthetic rest). Kept only as a documented
"what not to do" reference — see `FORCE_SENSING_FINDINGS.md` §3.2.

### 3.5 `compare_dynamics_residual_v3.py` — full head-to-head comparison

```bash
python compare_dynamics_residual_v3.py
```
Builds train + held-out datasets, fits both the regressor (3.2) and the
constrained NN (3.3), evaluates both, prints a side-by-side report. This is
what produced the final numbers in `FORCE_SENSING_FINDINGS.md`. Edit
`TRAIN_EPISODES`/`HELDOUT_EPISODES` at the top of the file to change the
split. ~15-20 min, pure CPU, no hardware.

---

## 4. Collecting new excitation data (only needed to retrain the dynamics models)

```bash
python replay_episode.py --episode_dir data/episodes/pastaTransfer4/<NNN> \
    --no_wait_convergence --speed_scale 0.5 --arm_trans_speed 0.08 --execute
```
The dynamics regressor/NN need real velocity/acceleration to fit against —
the default quasi-static replay (see §5) barely moves. This is the setting
used to build every training/held-out episode in `FORCE_SENSING_FINDINGS.md`.
Real hardware time, ~20-25s per episode. Saves dense `(t,q,qdot,torque)` logs
to `<episode_dir>/replay/torque_log_dense.npz`.

---

## 5. Replay & live force-sensing tools

### 5.1 `replay_episode.py` — replay a tracked episode on the robot

```bash
python replay_episode.py --episode_dir data/episodes/pastaTransfer4/<NNN>          # dry run
python replay_episode.py --episode_dir data/episodes/pastaTransfer4/<NNN> --execute # real motion
```
Default mode is **quasi-static** (stops and settles at each of 100 waypoints,
~70s total) — deliberately slow, for clean contact-force readings, at the
cost of looking jittery/stop-start. Add `--no_wait_convergence --speed_scale 0.5
--arm_trans_speed 0.08` for **smooth continuous motion** (~20-25s total) —
looks natural, but carries more real inertial noise in the force reading.
Neither is "more correct" — pick based on what you're doing (clean force
demo vs. natural-looking motion vs. exciting the dynamics models).

Always saves `<out_dir>/torque_log.npz` (one sample per waypoint) and
`<out_dir>/torque_log_dense.npz` (continuous, dense samples throughout —
what the analysis scripts below actually use). Default `out_dir` is
`<episode_dir>/replay/`; override with `--output_dir` to avoid overwriting a
previous run of the same episode.

**`--capture_camera`** recaptures a real RGB frame from **every connected
camera** at each waypoint (fixed 2026-07-08 — it used to only capture
`--track_cam`, which no longer exists as a flag; matches `01_record.py`,
which also saves all cameras), saved to `<out_dir>/cam{N}/{frame_id:06d}.jpg`
per camera. `frame_id` matches the original recorded episode's frame
numbers and `replay_forces_per_frame.npz`'s `frame_id` (5.6), so you can
pair a specific image with the force at that same instant.

**This is raw data only** — `torque_log*.npz` stores `raw_torque` and
`gravity_free_torque` (the firmware's own gravity compensation), nothing
more. It does NOT apply the software gravity regressor, gravity-residual NN,
or dynamics regressor from this document — run `analyze_replay_full.py`
(5.6) afterward for the actual calibrated external-force estimate.

**Gravity params are reapplied on every connect, and this is now a hard
failure if it doesn't work** (fixed 2026-07-08): the Kinova does NOT retain
the calibrated gravity matrix across a power cycle (confirmed 2026-07-02),
so `connect()` calls `apply_saved_gravity_params()` and aborts with `sys.exit(1)`
if it fails — previously this only printed a warning and continued, which
would have silently corrupted `gravity_free_torque` (and therefore every
downstream force estimate) for the whole session. Same fix applied to
`diag_live_forces.py` (5.3).

### 5.2 `diag_replay_forces.py` — LIVE plot while `replay_episode.py` runs

```bash
# separate terminal from replay_episode.py, start this FIRST or in either order
python diag_replay_forces.py
```
Reads `/tmp/replay_state.npz` (written by `replay_episode.py`, at the dense
sample rate — updates many times/second, not once per waypoint). Plots
firmware-only / +gravity / final-estimate, three lines, live. Does **not**
connect to the arm itself — only one process can hold the USB connection,
and `replay_episode.py` has it.

**Design note (learned the hard way):** the heavy per-sample computation
(~25ms, mostly the 72-parameter dynamics regressor) runs in a background
thread that appends ONE atomic record per step to a single shared buffer.
Do not "simplify" this back to direct computation inside the matplotlib
animation callback — that's what caused the GUI to fall progressively
behind in earlier testing, and a version that split the record across
several deques instead of one caused an intermittent shape-mismatch crash
that silently froze the plot. Both are fixed; keep the single-buffer,
background-thread structure if you ever touch this file.

### 5.3 `diag_live_forces.py` — LIVE plot while you hand-guide with the joystick

```bash
python diag_live_forces.py
```
Same three-line plot, same threaded architecture, but connects to the arm
directly (read-only — `connect(api, control=False)`, never sends a motion
command) instead of reading a replay state file. Use this when you want to
push/move the arm by hand rather than running a scripted episode. Does not
save any log to disk — if you want a saved record to analyze afterward, ask
for logging to be added (a few lines) rather than assuming it's there.

### 5.4 `analyze_replay_forces.py` — post-hoc, 2-tier (older, simpler)

```bash
python analyze_replay_forces.py --episode_dir data/episodes/pastaTransfer4/<NNN>
```
firmware-only vs. gravity-regressor-only (no gravity-residual NN, no
dynamics correction). Useful for a quick before/after gravity-regressor
check; use `analyze_replay_full.py` (5.6) for the complete picture.

### 5.5 `analyze_replay_dynamics.py` — post-hoc, dynamics-only (older)

```bash
python analyze_replay_dynamics.py --episode_dir data/episodes/pastaTransfer4/<NNN>
```
static (gravity-free only) vs. dynamics-compensated (nominal RNEA only, not
the fitted regressor/NN). Predates the gravity-residual-NN and the
regressor-vs-NN work; mostly superseded by 5.6, kept for the specific
"how much does *nominal* RNEA matter" question.

### 5.6 `analyze_replay_full.py` — post-hoc, full 3-tier (recommended)

```bash
python analyze_replay_full.py --episode_dir data/episodes/pastaTransfer4/<NNN>
python analyze_replay_full.py --log_path data/episodes/.../replay_smooth/torque_log_dense.npz
```
The saved-log equivalent of `diag_replay_forces.py`'s live plot: firmware /
+gravity(regressor+NN) / final-estimate(+mass-Coriolis regressor). Prints
per-tier mean/std/min/max, reduction percentages, flags the peak sample
(check it looks like a real push — smooth rise-peak-fall over ~0.3-0.5s —
not an isolated spike), and saves a plot PNG next to the log.

**Also saves the actual computed force data** (added 2026-07-08 — previously
this script only plotted/printed, the calibrated force values were never
persisted): `<log_dir>/replay_full_forces.npz` with per-timestep `t`,
`external_force_xyz` (N — the pure external force at the EEF, base frame;
this is the number that matters), `external_moment_xyz` (N·m), and all
three tiers' `||F||` norms (`Fn_firmware_only`/`Fn_plus_gravity`/`Fn_final`)
plus `q_deg`/`qdot_deg` for reference. Pass `--output_npz ""` to skip saving.

**Also tags the force at each image/waypoint instant** (added 2026-07-08 —
for pairing with `replay_episode.py --capture_camera`'s saved images): if
`<log_dir>/torque_log.npz` (the sparse per-waypoint log, always saved
alongside the dense one) is present, saves `<log_dir>/replay_forces_per_frame.npz`
with `frame_id` (matches `--capture_camera`'s `{frame_id:06d}.jpg` filenames
exactly), `t`, `external_force_xyz`, `external_moment_xyz`, `Fn_final`, and
`match_dt` (how far off the matched dense sample's timestamp was — should be
a few ms; large values mean the dense log is too sparse to trust the tagging).
Uses the dense-computed force (which has continuous `qdot` for the dynamics
correction) resampled at the waypoint times, rather than recomputing a
cruder estimate directly from the sparse log. Pass `--output_per_frame_npz ""`
to skip. Remember: with the default `--subsample 3` in `replay_episode.py`,
you get 1/3 of the original recorded frames as waypoints (e.g. 150 of 450),
not all of them — use `--subsample 1` at replay time if you need every frame.

---

## 6. What to expect (sanity-check numbers)

With nothing touching the arm, the final-estimate tier should read
**~1-3N**, occasionally briefly higher during fast motion segments — this is
the established sensor/model noise floor for this arm, not a defect. A real
applied push looks like a clean ramp from that floor up to however hard you
push (we saw 30N+ for a firm push in testing) and back down over roughly
half a second — sharp, isolated single-sample spikes with no ramp are more
likely noise/a bad sample than real contact.

The regressor and gravity-residual NN were both fit on `pastaTransfer4`
workspace poses. Expect degraded (but not broken) accuracy far outside that
region — extend the training data (§2.3, §4) if you need better coverage
elsewhere.

---

## 7. Data files quick reference

| File | What it is | Produced by |
|---|---|---|
| `data/gravity_params.npy` | Firmware gravity fit | `calibrate_firmware_gravity.py` |
| `data/gravity_phi_task_only.npy` | Linear gravity regressor (trusted) | `collect_task_poses.py` |
| `data/gravity_residual_nn.pt` | Gravity residual NN (trusted, best gravity result) | `fit_gravity_residual_nn.py` |
| `data/dynamics_residual_pi.npy` | Mass/Coriolis linear regressor (trusted, recommended). Regenerated 2026-07-08 — the file on disk had silently been the earlier/stale fit, since `compare_dynamics_residual_v3.py` computes pi internally but never saved it back; fixed by re-running the same 30-episode fit and saving it properly. Stale version backed up as `dynamics_residual_pi_stale_backup_*.npy`. | `fit_dynamics_regressor.py` / `compare_dynamics_residual_v3.py` |
| `data/dynamics_residual_nn_constrained_v3.pt` | Mass/Coriolis constrained NN (reference — ties with regressor, not simpler) | `compare_dynamics_residual_v3.py` |
| `<episode_dir>/replay/torque_log_dense.npz` | **Raw** joint torques only, continuous/dense samples (`raw_torque`, `gravity_free_torque`, `q_deg`, `qdot_deg`) — no calibration applied | `replay_episode.py --execute` |
| `<episode_dir>/replay/torque_log.npz` | **Raw** joint torques, ONE sample per waypoint, with `frame_id` (matches original recorded episode's frame numbers) | `replay_episode.py --execute` |
| `<episode_dir>/replay/cam{N}/{frame_id}.jpg` | Real RGB recaptured at each waypoint, from EVERY connected camera (`cam0/`, `cam1/`, ...) | `replay_episode.py --execute --capture_camera` |
| `<episode_dir>/replay/replay_full_forces.npz` | The actual calibrated force data, dense rate: `external_force_xyz` (N, pure external force), `external_moment_xyz`, all 3 tiers' `\|\|F\|\|` norms | `analyze_replay_full.py` |
| `<episode_dir>/replay/replay_forces_per_frame.npz` | Same force data, resampled at each waypoint's time instant, tagged with `frame_id` — matches `--capture_camera`'s image filenames, for pairing images with forces | `analyze_replay_full.py` |

Note: `data/dynamics_residual_nn.pt` (deprecated unconstrained NN) and `data/gravity_phi.npy` /
`gravity_phi_task_only_v2.npy` (superseded intermediate fits) were removed during a 2026-07-08
cleanup — if you see references to them elsewhere, they no longer exist on disk.
