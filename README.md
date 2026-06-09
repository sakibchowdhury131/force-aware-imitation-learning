# Tool-as-Interface Pipeline

> **Adapted from the original [Tool-as-Interface](https://github.com/Tool-as-Interface/Tool_as_Interface) repository.**
>
> Chen, H., Zhu, C., Li, Y., & Driggs-Campbell, K.
> *Tool-as-Interface: Learning Robot Tool Use from Human Play through Imitation Learning.*
> University of Illinois Urbana-Champaign & Columbia University.
> [[GitHub]](https://github.com/Tool-as-Interface/Tool_as_Interface)

A clean, modular re-implementation of the Tool-as-Interface pipeline. While the original work
demonstrated the approach on a fixed set of tools (hammer, screwdriver, etc.) with a specific
robot setup, this implementation is designed to work with **any tool** — simply provide a 3D
mesh and a text prompt. The pipeline is organized as a series of self-contained scripts, each
with clear inputs and outputs, making it straightforward to swap in a new tool, camera
configuration, or episode without modifying any code.

Key improvements over the original:
- Works with any number of RealSense cameras (not fixed to two)
- Real depth recorded alongside RGB in step 1, enabling accurate temporal tracking in step 4
- Camera selection (`--camera`) available wherever real images are used
- Temporal FoundationPose tracking (register once, track through sequence) instead of
  per-frame independent registration
- Live demo script for real-time pose tracking with any tool
- Offline tracking script for post-hoc analysis on saved recordings
- Pose visualization script for inspection after step 4

## Pipeline overview

```
01_record.py  →  02_augment.py  →  03_segment.py  →  04_track.py  →  05_train.py
  (record)        (MASt3R+3DGS)     (GroundedSAM)    (FoundationPose)  (diffusion)
```

## Setup (activate environment)

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ti
```

---

## Installation

The workspace layout expected on disk:

```
tool-as-interface/
    pipeline/                  ← this directory
    mast3r/                    ← MASt3R repo (for reference; starster is pip-installed)
    Depth-Anything-V2/         ← DA2 repo
    Tool_as_Interface/
        third_party/
            FoundationPose/
            Grounded-Segment-Anything/
```

### 1. Create the conda environment

```bash
conda create -n ti python=3.10 -y
conda activate ti
```

### 2. Install PyTorch (CUDA 11.8)

```bash
pip install torch==2.0.0+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.1+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118
```

### 3. Install pip dependencies

```bash
pip install \
    pyrealsense2 \
    groundingdino-py \
    trimesh \
    nvdiffrast \
    open3d \
    opencv-contrib-python \
    diffusers==0.20.1 \
    accelerate \
    pillow \
    numpy==1.26.4 \
    scipy \
    tqdm \
    einops \
    roma \
    kornia \
    timm \
    starster
```

> `starster` is the pip package that wraps MASt3R + 3DGS for step 2.

### 4. Install segment_anything from source

```bash
pip install -e Tool_as_Interface/third_party/Grounded-Segment-Anything/segment_anything
```

### 5. Install FoundationPose dependencies

```bash
pip install -r Tool_as_Interface/third_party/FoundationPose/requirements.txt
```

### 6. Install Depth Anything V2 (for fallback depth, optional)

```bash
pip install -r Depth-Anything-V2/requirements.txt
```

### 7. Download model checkpoints

All weights go into `pipeline/checkpoints/`.

**GroundingDINO** (`checkpoints/groundingdino_swint_ogc.pth`):
```bash
wget -P pipeline/checkpoints \
    https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

**SAM ViT-H** (`checkpoints/sam_vit_h_4b8939.pth`):
```bash
wget -P pipeline/checkpoints \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

**FoundationPose** — download from the [official HuggingFace repo](https://huggingface.co/bowen-wen/FoundationPose)
and place under `checkpoints/foundation_pose_weights/`:
```
checkpoints/foundation_pose_weights/
    2023-10-28-18-33-37/model_best.pth   ← scorer
    2024-01-11-20-02-45/model_best.pth   ← refiner
```

**Depth Anything V2 indoor ViT-L** (only needed for `generate_depth.py`):
Download `depth_anything_v2_metric_hypersim_vitl.pth` from the
[HuggingFace model page](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large)
and place it at `Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth`.

---

## Steps

### Step 1 — Record a demonstration (`01_record.py`)

Records synchronized RGB frames and aligned depth from all connected RealSense cameras.

```bash
python 01_record.py --task hammer --episode 001 --duration 15
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--task` | `task` | Task name (used as directory name) |
| `--episode` | `001` | Episode ID string |
| `--duration` | `15.0` | Recording duration in seconds |
| `--fps` | `30` | Frame rate |
| `--width` / `--height` | `848` / `480` | Resolution per camera |
| `--out` | `data/episodes/` | Root output directory |

**Output:**
```
data/episodes/<task>/<episode>/
    cam0/           ← JPEG frames (000000.jpg, 000001.jpg, ...)
    cam1/
    cam0_depth/     ← 16-bit PNG depth maps (000000.png, mm units)
    cam1_depth/
    meta.json       ← serials, fps, resolution, frame count, intrinsics, depth_scale
```

`meta.json` includes `has_depth: true` and `depth_scale: 0.001` (mm → m conversion factor).

---

### Step 2 — Augment with novel views (`02_augment.py`)

For each sampled timestep: runs MASt3R stereo reconstruction on the camera pair,
trains 3D Gaussians, and renders novel views from Slerp-interpolated camera poses.
Also saves the per-frame camera matrices needed by step 4 temporal tracking.

Can process a single episode or all episodes in a task directory in one go.
The MASt3R model is loaded once and reused across all episodes.

```bash
# Single episode
python 02_augment.py --episode_dir data/episodes/hammer/001

# All episodes in a task (recommended)
python 02_augment.py --task_dir data/episodes/hammer

# Skip episodes already processed
python 02_augment.py --task_dir data/episodes/hammer --skip_done
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--episode_dir` | — | Single episode to process (mutually exclusive with `--task_dir`) |
| `--task_dir` | — | Task directory; processes all episode subdirs with a `meta.json` |
| `--skip_done` | off | Skip episodes that already have `novel_cameras.npz` |
| `--num_novel_views` | `6` | Novel views to render per timestep (must be even) |
| `--sample_every` | `15` | Process every Nth frame (15 = 2 fps from 30 fps) |
| `--gs_iters_pruning` | `500` | Gaussian splatting pruning iterations |
| `--gs_iters_fine` | `100` | Gaussian splatting fine-tuning iterations |
| `--cam_pair` | `0 1` | Which two camera indices to use for reconstruction |
| `--device` | `cuda` | |

**Output:**
```
augmented/
    real/                  ← copies of the sampled real frames
    novel/                 ← novel view renders (000000_novel00.jpg, ...)
    novel_cameras.npz      ← per-frame camera matrices for step 4
        {frame}_cam0_w2c   → (4, 4) world-to-camera for cam0
        {frame}_cam1_w2c   → (4, 4) world-to-camera for cam1
        {frame}_novel_w2c  → (N, 4, 4) world-to-camera for each novel view
        {frame}_novel_K    → (N, 3, 3) intrinsics for each novel view
```

---

### Step 3 — Mask human hands/arms (`03_segment.py`)

Uses GroundedSAM (GroundingDINO + SAM ViT-H) to detect and black out human body
parts in both real and novel view frames. Models are loaded once and reused across
all episodes.

```bash
# Single episode
python 03_segment.py --episode_dir data/episodes/hammer/001

# All episodes in a task
python 03_segment.py --task_dir data/episodes/hammer

# Skip episodes already processed
python 03_segment.py --task_dir data/episodes/hammer --skip_done
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--episode_dir` | — | Single episode (mutually exclusive with `--task_dir`) |
| `--task_dir` | — | Task directory; processes all episodes |
| `--skip_done` | off | Skip episodes that already have `masked_real/` and `masked_novel/` |
| `--prompt` | `"human hand . human arm . person"` | GroundingDINO text prompt |
| `--box_threshold` | `0.3` | Detection confidence threshold |
| `--text_threshold` | `0.25` | Text matching threshold |
| `--device` | `cuda` | |

**Output:**
```
augmented/
    masked_real/     ← real frames with hands/arms blacked out
    masked_novel/    ← novel views with hands/arms blacked out
```

---

### Step 4 — Track tool pose (`04_track.py`)

Estimates 6DOF tool pose using FoundationPose. Runs in one of two modes depending
on what data is available. Models (GroundingDINO, SAM, FoundationPose) are loaded
once and reused across all episodes.

```bash
# Single episode
python 04_track.py --episode_dir data/episodes/hammer/001 \
    --tool_prompt "hammer" --mesh Hammer.obj

# All episodes in a task
python 04_track.py --task_dir data/episodes/hammer \
    --tool_prompt "hammer" --mesh Hammer.obj

# Skip already-processed episodes
python 04_track.py --task_dir data/episodes/hammer \
    --tool_prompt "hammer" --mesh Hammer.obj --skip_done
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--episode_dir` | — | Single episode (mutually exclusive with `--task_dir`) |
| `--task_dir` | — | Task directory; processes all episodes |
| `--mesh` | required | Path to tool mesh (.obj or .ply) |
| `--tool_prompt` | `"hammer"` | Text prompt for initial segmentation |
| `--camera` | `0` | Which real camera to use for RGBD tracking |
| `--skip_done` | off | Skip episodes that already have `tool_poses.npz` |
| `--box_threshold` | `0.3` | |
| `--text_threshold` | `0.25` | |
| `--est_refine_iter` | `5` | Iterations for initial registration |
| `--track_refine_iter` | `2` | Iterations for per-frame tracking |
| `--depth_const` | `0.5` | Fallback depth (m) when no DA2 maps exist |
| `--device` | `cuda` | |

**Tracking modes** (auto-detected per episode):

- **TEMPORAL** (preferred): used when `camN_depth/` and `novel_cameras.npz` both exist.
  Segments the tool on frame 0, registers an initial pose, then calls `track_one()` on
  every subsequent frame. Each tracked cam pose is transformed into all novel-view frames
  via: `pose_novel = novel_w2c[k] @ inv(camN_w2c) @ pose_camN`.

- **FALLBACK**: used when depth or camera matrices are unavailable. Independently segments
  and registers every masked novel view using DA2 depth maps (if present) or a constant
  depth plane. Slower and less temporally consistent.

**Mesh notes:** meshes are auto-rescaled from cm to metres if max extent > 0.5.

**Output:**
```
augmented/
    tool_poses.npz         ← str(frame_id) → (N_novel, 4, 4) poses in each novel view
    tool_poses_cam0.npz    ← str(frame_id) → (4, 4) raw pose in camN frame (temporal only)
```

---

### Step 5 — Train diffusion policy (`05_train.py`)

Trains a DDPM-based policy on (novel view image → 6DOF tool pose) pairs.
Observation: 96×96 RGB image. Action: 9D = 6D rotation (first two columns of R) + 3D translation.

Already operates at task level — pass the task directory and it automatically
loads all episodes under it.

```bash
python 05_train.py --data_dir data/episodes/hammer \
    --output_dir data/checkpoints/hammer
```

**Image pipeline** (matches the original paper):
- Resize to **128×128**
- Random crop to **115×115** during training; center crop during eval
- ColorJitter augmentation (brightness, contrast, saturation ±0.2)

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | required | Task directory containing all episode subdirectories |
| `--output_dir` | `data/checkpoints` | Where to save checkpoints |
| `--num_epochs` | `3050` | Matches original paper |
| `--batch_size` | `32` | Matches original paper |
| `--lr` | `1e-4` | Learning rate |
| `--image_size` | `128` | Resize images to this size before cropping (original paper: 128) |
| `--crop_size` | `115` | Random crop size during training (original paper: 115) |
| `--action_horizon` | `8` | Action sequence length |
| `--checkpoint_every` | `100` | Save a checkpoint every N epochs |
| `--device` | `cuda` | |

Checkpoints saved every `--checkpoint_every` epochs and at the end as `policy_final.pt`.
Each checkpoint stores `image_size` and `crop_size` so inference can apply the correct transform.

---

### Test / inference (`test_policy.py`)

Runs a trained checkpoint on a temporal window of images or a full episode. No robot required.

**Inputs:** `n_obs_steps` consecutive masked novel views (stacked along channels, matching training)
**Output:** `action_horizon` future tool poses as 9D vectors `[tx, ty, tz, rot6d]`, decoded to 4×4 matrices

```bash
# Single window — provide n_obs_steps images oldest→newest (here n_obs_steps=2)
python test_policy.py \
    --checkpoint data/checkpoints/hammer/policy_final.pt \
    --images frame_t-1.jpg frame_t.jpg

# One image is OK too — repeated for all obs steps (quick sanity check)
python test_policy.py \
    --checkpoint data/checkpoints/hammer/policy_final.pt \
    --images data/episodes/hammer/001/augmented/masked_novel/000030_novel0.jpg

# Full episode — XYZ axes overlay on real cam0 images (no mesh needed)
python test_policy.py \
    --checkpoint data/checkpoints/hammer/policy_final.pt \
    --episode_dir data/episodes/hammer/001 \
    --overlay --output_dir /tmp/policy_test

# Full episode — trajectory plot (tx/ty/tz vs time, shown against GT if available)
python test_policy.py \
    --checkpoint data/checkpoints/hammer/policy_final.pt \
    --episode_dir data/episodes/hammer/001 \
    --plot_trajectory --output_dir /tmp/policy_test

# Full episode — 3D bounding-box overlay (requires trimesh + FoundationPose)
python test_policy.py \
    --checkpoint data/checkpoints/hammer/policy_final.pt \
    --episode_dir data/episodes/hammer/001 \
    --mesh hammer.obj --output_dir /tmp/policy_test
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | required | Path to `.pt` checkpoint from step 5 |
| `--images` | — | One or more image paths, oldest → newest (mutually exclusive with `--episode_dir`) |
| `--episode_dir` | — | Episode dir; sliding window over `augmented/masked_novel/` |
| `--overlay` | off | Draw RGB XYZ axes on real cam0 images — no mesh needed |
| `--plot_trajectory` | off | Save `trajectory.png` (tx/ty/tz vs frame); overlays GT if `tool_poses_cam0.npz` exists |
| `--mesh` | `None` | Tool mesh for 3D bounding-box overlay (optional; falls back to axes if unavailable) |
| `--output_dir` | `augmented/policy_predictions/` | Where to save output images / npz files |
| `--device` | `cuda` | |

> **Why overlay on cam0 images?**
> Predicted poses are in the **cam0 camera frame**, so drawing axes on the real cam0 image
> (from `augmented/masked_real/`) is geometrically exact. Novel-view images use a different
> virtual camera, so overlaying on them requires an extra frame transform.

**Inference pipeline:**
1. Stack `n_obs_steps` frames along channel dim → `(1, n_obs_steps×3, H, W)` (matches training)
2. Encode with shared ResNet-18 per frame → project to 512-dim obs embedding
3. Start from Gaussian noise in action space `(1, action_horizon × 9)`
4. Run 100 DDPM denoising steps conditioned on obs embedding (FiLM scale+shift per layer)
5. Denormalize with `MaxAbsNormalizer` loaded from the checkpoint
6. Reshape to `(action_horizon, 9)` — each step is `[tx, ty, tz, rot6d(6)]`

**Output files (episode mode):**
- `{fid}_pred.jpg` — cam0 image with predicted pose axes / bbox overlaid
- `predicted_poses.npz` — first predicted step per window: `{fid: (4,4)}` in cam0 frame (metres)
- `predicted_action_seqs.npz` — full `(action_horizon, 9)` sequence per window
- `trajectory.png` — translation plot (with `--plot_trajectory`)

All model hyper-parameters (`image_size`, `crop_size`, `n_obs_steps`, normalizer) are read
directly from the checkpoint — no need to pass them manually.

---

## Utility scripts

### `generate_depth.py` — DA2 monocular depth (fallback mode)

Generates metric depth maps using Depth Anything V2 (indoor ViT-L model).
Only needed for step 4 fallback mode when real RealSense depth is unavailable.

```bash
python generate_depth.py --image_dir data/episodes/hammer/001/cam0
```

Output: `cam0_depth/*.npy` (float32, metres).

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--image_dir` | required | Directory of RGB images |
| `--encoder` | `vitl` | Model size: `vits`, `vitb`, or `vitl` |
| `--max_depth` | `20.0` | Max depth in metres |

---

### `live_foundationpose.py` — Live pose tracking demo

Streams from a RealSense camera, segments the tool with GroundedSAM on demand,
then tracks 6DOF pose in real-time with FoundationPose.

```bash
python live_foundationpose.py --mesh Hammer.obj --tool_prompt "hammer"
python live_foundationpose.py --mesh Paddle.obj --tool_prompt "table tennis paddle" --camera 1
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--mesh` | required | Path to tool mesh (.obj or .ply) |
| `--tool_prompt` | `"hammer"` | Text prompt for segmentation |
| `--camera` | `0` | Camera index (0 = first connected, 1 = second, ...) |
| `--box_threshold` | `0.3` | |
| `--text_threshold` | `0.25` | |
| `--est_refine_iter` | `5` | |
| `--track_refine_iter` | `2` | |
| `--device` | `cuda` | |

**Controls:** `SPACE` — segment + re-initialize pose. `Q` — quit.

A loading screen is shown while models load (~2–3 min on first run).

> **Note:** Loading order matters on Linux/NVIDIA. The cv2 window is created first,
> then GroundingDINO + SAM, then FoundationPose/nvdiffrast.
> Do not import `torchvision.transforms` or `groundingdino` at module level — this
> deadlocks `cv2.namedWindow` due to a Qt5/X11 conflict.

---

### `offline_foundationpose.py` — Offline tracking on saved images

Same register-then-track approach as the live demo but on pre-recorded images.
Saves annotated JPEG frames instead of displaying a window.

```bash
python offline_foundationpose.py \
    --image_dir data/episodes/hammer/001/cam0 \
    --meta      data/episodes/hammer/001/meta.json \
    --mesh      Hammer.obj \
    --tool_prompt "hammer" \
    --output_dir /tmp/fp_results
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--image_dir` | required | Directory of frames to track |
| `--meta` | required | Path to `meta.json` for intrinsics |
| `--mesh` | required | |
| `--tool_prompt` | `"hammer"` | |
| `--cam_idx` | `0` | Which camera's intrinsics to use |
| `--init_frame` | `0` | Frame to segment + register on |
| `--num_frames` | `50` | Frames to track after init (0 = all) |
| `--depth_const` | `0.5` | Fallback depth if no `_depth/` dir |
| `--output_dir` | `/tmp/fp_results` | Where to save annotated frames |

Loads DA2 depth maps from `<image_dir>_depth/` automatically if present.

---

### `visualize_poses.py` — Overlay poses on images

Draws the 3D bounding box and XYZ axes on saved frames for inspection.

```bash
# Novel views (masked, default)
python visualize_poses.py --episode_dir data/episodes/hammer/001 --mesh Hammer.obj

# Novel views (unmasked, better visual context)
python visualize_poses.py --episode_dir data/episodes/hammer/001 --mesh Hammer.obj --use_novel

# Real camera frames (uses tool_poses_camN.npz from step 4)
python visualize_poses.py --episode_dir data/episodes/hammer/001 --mesh Hammer.obj --use_real
python visualize_poses.py --episode_dir data/episodes/hammer/001 --mesh Hammer.obj --use_real --camera 1
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--episode_dir` | required | |
| `--mesh` | required | |
| `--output_dir` | `augmented/viz_poses/` | |
| `--axis_scale` | `0.08` | Axis length in metres |
| `--camera` | `0` | Camera index (only affects `--use_real`) |
| `--use_novel` | off | Overlay on unmasked `novel/` instead of `masked_novel/` |
| `--use_real` | off | Overlay on real `camN/` images using `tool_poses_camN.npz` |

---

## Data layout

```
data/episodes/<task>/<episode>/
    cam0/              ← raw RGB frames (000000.jpg, ...)
    cam1/
    cam0_depth/        ← aligned depth frames (000000.png, uint16 mm)
    cam1_depth/
    meta.json          ← serials, fps, resolution, intrinsics, depth_scale
    augmented/
        real/          ← sampled real frames used for reconstruction
        novel/         ← novel view renders
        masked_real/   ← real frames with hands blacked out
        masked_novel/  ← novel views with hands blacked out
        novel_cameras.npz       ← camera matrices (cam0/cam1/novel w2c, novel K)
        tool_poses.npz          ← 6DOF poses in novel-view frames
        tool_poses_cam0.npz     ← raw tracked poses in cam0 frame
        tool_poses_cam1.npz     ← raw tracked poses in cam1 frame (if used)
        viz_poses/              ← visualize_poses.py output
```

## Notes

- **Camera calibration**: run `calibrate_cameras.py` once per camera placement.
  Files saved to `../Tool_as_Interface/ti/real_world/cam_extrinsics/`.
- **Tool mesh**: FoundationPose requires a 3D mesh (.obj or .ply). Meshes in cm units
  are auto-rescaled to metres. Scan with an iPhone/depth camera or download from
  ShapeNet/Objaverse.
- **Action representation**: 9D = 6D rotation (first two columns of R, flattened) + 3D translation (metres).
- **Temporal vs fallback**: re-recording with the current `01_record.py` (saves real depth)
  and re-running `02_augment.py` (saves `novel_cameras.npz`) is strongly preferred —
  temporal tracking is significantly more accurate and consistent than per-view registration.
