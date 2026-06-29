# Tool-as-Interface Pipeline

> Adapted from [Tool-as-Interface](https://github.com/Tool-as-Interface/Tool_as_Interface) (Chen et al., UIUC & Columbia).

A modular re-implementation of the Tool-as-Interface imitation learning pipeline. Provide any tool as a 3D mesh and a text prompt — no code changes required.

---

## Physical setup

| | |
|---|---|
| ![Full lab setup — Kinova Jaco2 arm with two RealSense cameras on tripods](media/setup_overview.jpg) | ![Workspace view — RealSense camera, corkboard task surface, pasta container and bowl](media/camera_setup.jpg) |
| *Full setup: Kinova Jaco2 6DOF arm, two Intel RealSense D435 cameras on tripods, corkboard task workspace* | *Workspace detail: task-frame corkboard, pasta source container, target bowl, and RealSense camera* |
| ![Robot gripper scooping pasta — close-up](media/robot_scooping_closeup.jpg) | ![Robot scooping pasta — wide shot](media/robot_scooping_wide.jpg) |
| *Close-up: Jaco2 gripper holding spoon, scooping elbow pasta from container* | *Wide view: robot executing the pastaTransfer task — scoop from box, transfer to bowl* |

**Hardware:**
- **Robot:** Kinova Jaco2 6DOF spherical-wrist arm (USB SDK)
- **Cameras:** 2× Intel RealSense D435 (848×480, 30fps RGBD)
- **Task:** pasta transfer — scoop elbow pasta from a container into a bowl

---

## Architecture

**Diffusion policy** (DDPM, 100 denoising steps):

| Component | Details |
|---|---|
| Image encoder | Shared ResNet-18 (ImageNet pretrained) |
| Multi-view fusion | Concatenate per-step features across `n_views` cameras, then project |
| Observation context | `n_obs_steps=2` consecutive timesteps, each with `n_views` images + proprioception |
| Noise predictor | Temporal **UNet-1D** (Chi et al. 2023) with GroupNorm + Mish + FiLM conditioning |
| Default UNet dims | `(128, 256, 512)` — ~30M total params |
| Action representation | 9D = 3D translation (m) + 6D rotation (first two cols of R, flattened) |
| Action horizon | 16 steps |
| Execution horizon | 8 steps (receding horizon, async inference) |

**Control frequency consistency** — this must be set consistently across training and deployment:

```
record_fps / subsample = deploy_hz

Examples:
  30 fps, subsample=3  →  10 Hz  (recommended baseline)
  30 fps, subsample=2  →  15 Hz
```

Set `--subsample` in step 5 and `--frequency` in step 7 to the same Hz value. The action horizon and execution horizon are measured in control steps, so 16 steps at 10 Hz = 1.6 s prediction window, 8 steps = 800 ms execution window.

---

## Pipeline overview

```
Step 0:   00_calibrate.py          — camera extrinsics (ChArUco board)
Step 0b:  06_calibrate_robot.py    — robot base frame relative to task frame
Step 0c:  check_tool_eef_error.py  — calibrate T_tool_eef (tool→EEF rigid offset)

Step 1:   01_record.py             — record demonstrations (all cameras, RGBD)
Step 2:   02_augment_noposplat.py  — novel-view augmentation [OPTIONAL — skip with --cam_only in step 3]
Step 3:   03_segment.py            — mask human hands/arms
Step 4:   04_track.py              — 6DOF tool tracking (FoundationPose)
Step 4b:  04b_to_task_frame.py     — [standalone] re-transform cam→task frame
Step 4c:  04c_to_base_frame.py     — [standalone] re-transform task→robot base frame
Step 5:   05_train.py              — train diffusion policy
Step 7:   07_deploy.py             — live inference on the real robot
```

**Preferred pose frame for training:** robot base frame (`tool_poses_base.npz`, produced by step 4c). Training falls back to task frame (`tool_poses_task.npz`) or cam frame (`tool_poses_cam{N}.npz`) if base frame is absent. Robot base frame is recommended because camera recalibration after data collection cannot introduce a train/deploy frame mismatch.

---

## Setup

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ti
```

### Installation

Expected workspace layout:
```
tool-as-interface/
    pipeline/                  ← this directory
    Tool_as_Interface/
        third_party/
            FoundationPose/
            Grounded-Segment-Anything/
    Depth-Anything-V2/         ← optional, for fallback depth maps
```

```bash
# 1. Create environment
conda create -n ti python=3.10 -y && conda activate ti

# 2. PyTorch (CUDA 11.8)
pip install torch==2.0.0+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.1+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# 3. Core dependencies
pip install pyrealsense2 groundingdino-py trimesh nvdiffrast open3d \
    opencv-contrib-python diffusers==0.20.1 accelerate pillow \
    numpy==1.26.4 scipy tqdm einops roma kornia timm \
    segmentation-models-pytorch plotly

# 4. segment_anything
pip install -e Tool_as_Interface/third_party/Grounded-Segment-Anything/segment_anything

# 5. FoundationPose
pip install -r Tool_as_Interface/third_party/FoundationPose/requirements.txt

# 6. NoPoSplat (step 2, optional)
git clone https://github.com/cvg/NoPoSplat ~/working_dir/NoPoSplat
pip install gsplat hydra-core omegaconf huggingface_hub "e3nn==0.5.1"
# Note: use e3nn==0.5.1 specifically — newer versions break the CUDA 11.8 environment
```

### Model checkpoints

All weights go in `pipeline/checkpoints/`.

```bash
# GroundingDINO
wget -P checkpoints \
    https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# SAM ViT-H
wget -P checkpoints \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

**FoundationPose** — download from [HuggingFace](https://huggingface.co/bowen-wen/FoundationPose):
```
checkpoints/foundation_pose_weights/
    2023-10-28-18-33-37/model_best.pth   ← scorer
    2024-01-11-20-02-45/model_best.pth   ← refiner
```

**Depth Anything V2** (optional, step 2 fallback):
`Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth`

---

## Full pipeline walkthrough

### Step 0 — Camera calibration (`00_calibrate.py`)

Place the ChArUco board flat on the table in the task workspace. Run once per camera, once per camera placement.

```bash
python 00_calibrate.py --camera 0 --output data/cam_extrinsics.npy
python 00_calibrate.py --camera 1 --output data/cam1_extrinsics.npy
```

| Argument | Default | Description |
|---|---|---|
| `--output` | `data/cam_extrinsics.npy` | Output path for the 4×4 world→cam matrix |
| `--camera` | `0` | RealSense camera index |
| `--n_stable` | `5` | Number of stable detections to average |

---

### Step 0b — Robot base frame calibration (`06_calibrate_robot.py`)

Jog the robot to four known board points and press SPACE at each. Run once per camera/robot placement.

```bash
# Standard calibration (uses GetCartesianPosition)
python 06_calibrate_robot.py --output data/robot_extrinsics.npy

# FK-based calibration (more accurate Z height — recommended for proprio)
python 06_calibrate_robot.py --use_joint_fk --output data/robot_extrinsics_jfk.npy
```

The `--use_joint_fk` mode reads joint angles via `GetAngularPosition` and runs the full DH FK chain to get the EEF position, bypassing `GetCartesianPosition` which can have a systematic ~4.5 cm Z offset on the Kinova Jaco2. Use the FK-calibrated file as `--robot_extrinsics_proprio` in step 7.

| Argument | Default | Description |
|---|---|---|
| `--output` | `data/robot_extrinsics.npy` | Output path for 4×4 T\_base\_task |
| `--camera` | `0` | RealSense camera index |
| `--use_joint_fk` | off | Use joint-angle FK instead of GCP for EEF height (more accurate Z) |

---

### Step 0c — T\_tool\_eef calibration (`check_tool_eef_error.py`)

Calibrates the rigid offset between tool frame (FoundationPose) and robot EEF frame. Run once per tool grasp configuration.

**Procedure:**
1. Place the tool on the table (not in the gripper yet).
2. Launch: `python check_tool_eef_error.py --mesh spoon.obj --tool_prompt "spoon"`
3. Verify the bounding box wraps the tool — press **SPACE** to re-register if needed.
4. Pick up the tool with the gripper.
5. Once gripped, press **C** to capture and save `T_tool_eef`.
6. Verify error < 2 cm / < 5°. Press **Q** to quit.

| Argument | Default | Description |
|---|---|---|
| `--mesh` | required | Tool mesh (.obj) |
| `--tool_prompt` | required | GroundedSAM prompt |
| `--task_frame` | `data/cam_extrinsics.npy` | Camera extrinsics |
| `--robot_extrinsics` | `data/robot_extrinsics.npy` | Robot extrinsics |
| `--tool_eef_cache` | `data/T_tool_eef.npy` | Save path |
| `--camera` | `0` | RealSense camera index |

---

### Step 1 — Record demonstrations (`01_record.py`)

```bash
python 01_record.py --task pastaTransfer --episode 001 --duration 15 \
    --mesh spoon.obj --tool_prompt "spoon"
```

| Key | Action |
|---|---|
| SPACE | Re-register FoundationPose on current frame |
| ENTER | Start recording |
| Q | Stop / abort |

| Argument | Default | Description |
|---|---|---|
| `--task` | `task` | Task name |
| `--episode` | `001` | Episode ID |
| `--duration` | `15.0` | Recording duration (seconds) |
| `--fps` | `30` | Frame rate |
| `--width` / `--height` | `848` / `480` | Resolution |
| `--mesh` | `None` | Tool mesh — enables FP overlay |
| `--tool_prompt` | `"tool"` | GroundedSAM prompt |

**Output:**
```
data/episodes/<task>/<episode>/
    cam0/           ← JPEG frames (000000.jpg …)
    cam1/
    cam0_depth/     ← 16-bit PNG depth (mm)
    cam1_depth/
    meta.json
```

---

### Step 2 — Novel-view augmentation (`02_augment_noposplat.py`) [OPTIONAL]

Skip this step if using `--cam_only` in step 3 (single/dual real camera only, no rendered views).

```bash
python 02_augment_noposplat.py \
    --task_dir data/episodes/pastaTransfer \
    --noposplat_root ~/working_dir/NoPoSplat \
    --num_novel_views 6 --sample_every 1 --skip_done
```

| Argument | Default | Description |
|---|---|---|
| `--task_dir` | — | Task directory |
| `--num_novel_views` | `6` | Novel views per timestep |
| `--sample_every` | `15` | Process every Nth frame |
| `--skip_done` | off | Skip episodes already done |

**Output per episode:** `augmented/real/`, `augmented/novel/`, `augmented/novel_cameras.npz`

---

### Step 3 — Mask human hands/arms (`03_segment.py`)

Two modes:

**A. Camera-only (no step 2 needed) — recommended:**
```bash
python 03_segment.py \
    --task_dir data/episodes/pastaTransfer \
    --cam_only \
    --unet_checkpoint human_hand_segmentation_UNET/unet_best.pt \
    --skip_done
```

**B. With novel views (requires step 2):**
```bash
python 03_segment.py --task_dir data/episodes/pastaTransfer --skip_done
```

| Argument | Default | Description |
|---|---|---|
| `--cam_only` | off | Read directly from `camN/` dirs; write `masked_real/XXXXXX_camN.jpg` |
| `--unet_checkpoint` | `None` | Use trained hand-segmentation UNet instead of GroundedSAM2 (much faster) |
| `--unet_threshold` | `0.5` | Sigmoid threshold for UNet predictions |
| `--prompt` | `"human hand . human arm . person"` | GroundingDINO prompt (when not using UNet) |
| `--skip_done` | off | Skip episodes with existing `masked_real/` |

**Output per episode:** `augmented/masked_real/XXXXXX_camN.jpg`

---

### Step 4 — Track tool pose (`04_track.py`)

```bash
python 04_track.py \
    --task_dir data/episodes/pastaTransfer \
    --mesh spoon.obj --tool_prompt "spoon" \
    --track_cam 1 \
    --use_masked \
    --skip_done
```

`--use_masked` feeds the hand-blacked-out frames (`masked_real/`) to FoundationPose for RGB input, while keeping raw depth. Recommended when the human hand occludes the tool during demonstrations.

| Argument | Default | Description |
|---|---|---|
| `--mesh` | required | Tool mesh (.obj or .ply) |
| `--tool_prompt` | `"hammer"` | Text prompt for initial segmentation |
| `--track_cam` | `0` | Camera index for RGBD tracking |
| `--use_masked` | off | Use `masked_real/` frames for RGB (depth stays raw) |
| `--est_refine_iter` | `5` | Initial registration iterations |
| `--track_refine_iter` | `2` | Per-frame tracking iterations |
| `--skip_done` | off | Skip episodes with existing `tool_poses.npz` |
| `--task_frame` | auto | Path to cam extrinsics. Defaults to `data/cam{N}_extrinsics.npy` |

**Output per episode:**
```
augmented/
    tool_poses_cam{N}.npz     ← tracked poses in camera frame
    tool_poses_task.npz       ← poses in task/world frame (cam extrinsics applied)
```

**Verify tracking quality:**
```bash
python visualize_poses.py \
    --episode_dir data/episodes/pastaTransfer/001 \
    --mesh spoon.obj --use_real --camera 1 --use_masked
```

---

### Step 4b / 4c — Optional standalone transforms

These are only needed if re-transforming existing poses without re-running step 4.

```bash
# cam → task frame
python 04b_to_task_frame.py --task_dir ... --task_frame data/cam1_extrinsics.npy

# task → robot base frame (RECOMMENDED before training)
python 04c_to_base_frame.py --task_dir ... --robot_extrinsics data/robot_extrinsics.npy
```

Step 4c output (`tool_poses_base.npz`) is preferred by step 5 because the robot base frame is fixed relative to the robot regardless of camera placement.

---

### Step 5 — Train diffusion policy (`05_train.py`)

```bash
python 05_train.py \
    --data_dir data/episodes/pastaTransfer \
    --output_dir data/checkpoints/pastaTransfer \
    --track_cam 1 \
    --n_views 2 \
    --action_horizon 16 \
    --subsample 3 \
    --num_epochs 3050
```

**Frame rate:** `--subsample` must satisfy `record_fps / subsample = deploy_hz`. With 30fps recording and 10Hz deployment, use `--subsample 3`.

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | required | Task directory |
| `--output_dir` | `data/checkpoints` | Checkpoint output |
| `--track_cam` | `0` | Camera used in step 4 (determines which cam is primary) |
| `--n_views` | `1` | Cameras per obs step. `1` = single cam (random pool). `2` = dual cam (always track_cam + one other) |
| `--subsample` | `1` | Use every Nth frame. Set to `record_fps / deploy_hz` |
| `--action_horizon` | `16` | Predicted action steps |
| `--n_obs_steps` | `2` | Observation context steps |
| `--unet_dims` | `128 256 512` | UNet-1D channel sizes per encoder stage |
| `--unet_kernel` | `5` | UNet-1D Conv1d kernel size |
| `--num_epochs` | `3050` | Training epochs |
| `--batch_size` | `32` | |
| `--lr` | `1e-4` | |
| `--image_size` | `128` | Resize before crop |
| `--crop_size` | `115` | Random crop size |
| `--checkpoint_every` | `100` | Save checkpoint every N epochs |

---

### Step 7 — Deploy on robot (`07_deploy.py`)

**Dry run (no robot motion — always start here):**
```bash
python 07_deploy.py \
    --checkpoint data/checkpoints/pastaTransfer/policy_final.pt \
    --unet_checkpoint robot_segmentation_UNET/training/checkpoints/best_model.pth \
    --track_cam 1 \
    --frequency 10 \
    --exec_steps 16
```

**Live execution (recommended full command):**
```bash
python 07_deploy.py \
    --checkpoint data/checkpoints/pastaTransfer4/policy_final.pt \
    --unet_checkpoint robot_segmentation_UNET/training/checkpoints/best_model.pth \
    --exec_steps 16 --frequency 10 \
    --robot_extrinsics_proprio data/robot_extrinsics_jfk.npy \
    --max_trans_step 0.05 --max_rot_step_deg 20 \
    --arm_trans_speed 0.20 \
    --execute --wait_convergence
```

**Step-by-step mode (manual inspection):**
```bash
python 07_deploy.py ... --execute --wait_convergence --step_mode
# SPACE = send next action, P = pause, Q = quit
```

**Async inference design:** inference runs in a background thread. It fires when ≤2 actions remain in the queue, giving `2 / frequency` seconds of runway (200 ms at 10 Hz) for the next inference to complete. Observations are captured continuously during robot motion — not from a stopped robot.

**`--wait_convergence` and arm speed:** The Kinova Jaco2 USB SDK defaults to a very slow internal speed (~3 cm/s effective). Pass `--arm_trans_speed 0.15–0.25` to explicitly command a faster speed (`LimitationsActive=1`). The convergence timeout scales adaptively with the commanded move distance.

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | required | `policy_final.pt` from step 5 |
| `--unet_checkpoint` | required | Robot arm UNet segmentor weights |
| `--track_cam` | `0` | Camera for FP tracking + primary obs |
| `--frequency` | `10.0` | **Must match `record_fps / subsample` from training** |
| `--exec_steps` | `8` | Actions per inference cycle (receding horizon) |
| `--execute` | off | Enable robot motion (dry-run by default) |
| `--task_frame` | auto | Cam extrinsics (`data/cam{N}_extrinsics.npy`) |
| `--robot_extrinsics` | `data/robot_extrinsics.npy` | T\_base\_task for action commands |
| `--robot_extrinsics_proprio` | same as above | Separate T\_base\_task for proprio, calibrated with `--use_joint_fk`. Use when GCP and FK disagree on EEF height |
| `--T_eef_spoon` | `data/T_eef_spoon.npy` | EEF→spoon rigid offset (from `calib_viz_3d.py`). When present, skips per-step FP tracking entirely |
| `--tool_eef_cache` | `data/T_tool_eef.npy` | Rigid tool→EEF offset (fallback when T\_eef\_spoon absent) |
| `--recalibrate_tool_eef` | off | Force FP re-registration at startup |
| `--max_trans_step` | `0.02` | Max translation per command (m) — safety clamp |
| `--max_rot_step_deg` | `10.0` | Max rotation per command (degrees) — safety clamp |
| `--arm_trans_speed` | `0.0` | Kinova translation speed (m/s). `0` = robot default (slow). Use `0.15–0.25` |
| `--wait_convergence` | off | Poll `GetCartesianPosition` after each command until EEF reaches target before sending next |
| `--convergence_threshold` | `0.010` | Position error threshold for `--wait_convergence` (m) |
| `--convergence_timeout` | `0.5` | Base timeout per step (s); scaled up adaptively with move distance |
| `--step_mode` | off | Block after each action until SPACE is pressed — useful for manual inspection |

---

### Step 7b — 3D visualisation companion (`deploy_viz.py`)

Run in a **separate terminal** alongside `07_deploy.py`. Reads `/tmp/deploy_state.npz` written each step.

```bash
python deploy_viz.py --mesh spoon.obj --T_eef_spoon data/T_eef_spoon.npy
```

Shows in the robot base frame:
- **Grey meshes** — live robot arm (FK from joint angles)
- **Gold mesh** — spoon (FK + T\_eef\_spoon)
- **Cyan axes / sphere** — current tool pose (proprio sent to policy)
- **Green→red spheres** — predicted action waypoints
- **Orange** — task/ChArUco frame; **Yellow** — camera frame

| Argument | Default | Description |
|---|---|---|
| `--mesh` | required | Tool mesh (.obj) |
| `--T_eef_spoon` | `data/T_eef_spoon.npy` | EEF→spoon calibration |
| `--cam_extrinsics` | `data/cam_extrinsics.npy` | Camera extrinsics |
| `--robot_extrinsics` | `data/robot_extrinsics.npy` | T\_base\_task |
| `--poll_hz` | `10.0` | State file polling rate |

---

## Human hand segmentation UNet

Used in step 3 (`--unet_checkpoint`) for fast hand masking. Must be retrained if camera placement changes.

```bash
# Prepare training data (runs GroundedSAM2 on random frames to generate masks)
python prepare_unet_data.py --task_dir data/episodes/pastaTransfer

# Optional: clean mislabeled samples after reviewing viz/
python clean_unet_data.py

# Train
python train_unet_seg.py

# Evaluate
python eval_unet_seg.py
```

Weights saved to `human_hand_segmentation_UNET/unet_best.pt`.

---

## Robot arm segmentation UNet

Used in step 7 (`--unet_checkpoint`) to mask the robot arm out of the observation image. Must be retrained if camera placement changes.

```bash
python collect_unet_data.py   # collect frames from live camera
# then train using robot_segmentation_UNET/training/train_unet.py
```

---

## Calibration file reference

| File | Created by | Used by |
|---|---|---|
| `data/cam_extrinsics.npy` | `00_calibrate.py` | `04_track.py`, `07_deploy.py`, `check_tool_eef_error.py` |
| `data/cam1_extrinsics.npy` | `00_calibrate.py --camera 1` | `04_track.py --track_cam 1`, `07_deploy.py --track_cam 1` |
| `data/robot_extrinsics.npy` | `06_calibrate_robot.py` | `04c_to_base_frame.py`, `07_deploy.py --robot_extrinsics` |
| `data/robot_extrinsics_jfk.npy` | `06_calibrate_robot.py --use_joint_fk` | `07_deploy.py --robot_extrinsics_proprio` |
| `data/T_eef_spoon.npy` | `calib_viz_3d.py` | `07_deploy.py`, `deploy_viz.py` |
| `data/T_tool_eef.npy` | `check_tool_eef_error.py` (C key) | `07_deploy.py` (fallback when T\_eef\_spoon absent) |

**When to redo calibration:**

| Event | Redo |
|---|---|
| Camera moved | `00_calibrate.py`, `06_calibrate_robot.py`, `check_tool_eef_error.py` |
| Robot base moved | `06_calibrate_robot.py` (both GCP and FK versions), `check_tool_eef_error.py` |
| Tool re-grasped | `calib_viz_3d.py` (T\_eef\_spoon), `check_tool_eef_error.py` |
| Camera image quality poor | Retrain hand / robot arm UNet |

---

## Data layout

```
data/episodes/<task>/<episode>/
    cam0/              ← raw RGB frames (000000.jpg …)
    cam1/
    cam0_depth/        ← aligned depth (000000.png, uint16 mm)
    cam1_depth/
    meta.json
    augmented/
        masked_real/            ← real frames with hands blacked out  (step 3 --cam_only)
        masked_novel/           ← novel views with hands blacked out   (step 3 default)
        real/                   ← sampled real frames                  (step 2)
        novel/                  ← novel view renders                   (step 2)
        novel_cameras.npz       ← per-frame camera matrices            (step 2)
        tool_poses_cam{N}.npz   ← tracked poses in camN frame          (step 4)
        tool_poses_task.npz     ← poses in task/world frame            (step 4)
        tool_poses_base.npz     ← poses in robot base frame            (step 4c) ← preferred for training
        viz_poses/              ← overlay images / videos              (visualize_poses.py)
```

---

## Utility scripts

| Script | Purpose |
|---|---|
| `preview_cameras.py` | Live preview of all connected RealSense cameras |
| `visualize_poses.py` | Overlay tracked poses on episode frames; encode video |
| `show_robot_frame.py` | Project robot base frame axes onto live camera view |
| `calib_viz_3d.py` | Calibrate `T_eef_spoon` (EEF→tool rigid offset) via 3D visualization |
| `kinova_fk_viz.py` | Live FK visualization — verify DH chain matches physical arm |
| `deploy_viz.py` | Live 3D view alongside `07_deploy.py`: arm mesh, spoon, predicted waypoints |
| `plot_deploy.py` | Live 2D plot of actual vs predicted trajectory during deployment |
| `plot_3d_deploy.py` | Live 3D plot of EEF pose + action horizon during deployment |
| `diag_joint_torques.py` | Real-time plot of all 6 joint torque sensors (raw + gravity-compensated) |
| `diag_kinova_angles.py` | Diagnostic: compare GetCartesianPosition Euler conventions |
| `diag_kinova_pos.py` | Diagnostic: live Cartesian position readout |
| `eval_on_dataset.py` | Run trained policy on recorded episodes; produce overlay videos |
| `check_tool_eef_error.py` | Calibrate and verify T\_tool\_eef (FoundationPose → EEF offset) |

---

## Known issues / gotchas

- **Torchvision / cv2 deadlock:** on Linux/NVIDIA, importing `torchvision.transforms` at module level deadlocks `cv2.namedWindow`. All torchvision imports in this pipeline are inside function bodies.
- **Open3D / CUDA init order:** create cv2 windows before loading any CUDA model (FP, SAM) to avoid process kill on Linux/NVIDIA.
- **FoundationPose occlusion:** when the tool is occluded by the demonstrator's hand (e.g. during scooping), temporal tracking drifts. Use `--use_masked` in step 4 to feed hand-blacked-out frames, or choose a camera angle where the tool face remains visible.
- **GroundedSAM arm contamination:** when the robot arm is holding the tool during initial registration, SAM often includes the arm in the mask. Always register on the table first (step 0c procedure).








## Quick reference — Inference

```bash
# Terminal 1: run policy on robot
python 07_deploy.py \
    --checkpoint data/checkpoints/pastaTransfer4/policy_final.pt \
    --unet_checkpoint robot_segmentation_UNET/training/checkpoints/best_model.pth \
    --exec_steps 16 --frequency 10 \
    --robot_extrinsics_proprio data/robot_extrinsics_jfk.npy \
    --max_trans_step 0.05 --max_rot_step_deg 20 \
    --arm_trans_speed 0.20 \
    --execute --wait_convergence

# Terminal 2: live 3D visualisation
python deploy_viz.py --mesh spoon.obj --T_eef_spoon data/T_eef_spoon.npy

# Step-by-step mode (add --step_mode; press SPACE to advance each action)
python 07_deploy.py ... --execute --wait_convergence --step_mode

# Joint torque monitor (separate terminal, useful during deployment)
python diag_joint_torques.py --gravity_free
```

