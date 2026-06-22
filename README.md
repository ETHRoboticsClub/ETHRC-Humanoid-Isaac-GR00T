# ETHRC Humanoid — Isaac GR00T N1.7 Fine-tuning

Fine-tuning and deployment of [GR00T N1.7](https://huggingface.co/nvidia/GR00T-N1.7-3B), NVIDIA's open vision-language-action model for generalist humanoid robot skills, on custom robot datasets.

---

## Table of contents

1. [Environments](#1-environments)
2. [Installation — training env](#2-installation--training-env)
3. [Installation — data-processing env](#3-installation--data-processing-env)
4. [G1 locomanipulation pipeline](#4-g1-locomanipulation-pipeline)
   - 4.1 [Download dataset from HuggingFace](#41-download-dataset-from-huggingface)
   - 4.2 [Convert dataset (quat → rot6d)](#42-convert-dataset-quat--rot6d)
   - 4.3 [Download base model](#43-download-base-model)
   - 4.4 [Generate statistics](#44-generate-statistics)
   - 4.5 [Fine-tune](#45-fine-tune)
5. [Adding a new robot](#5-adding-a-new-robot)
6. [Directory layout](#6-directory-layout)

---

## 1. Environments

Two separate environments are used:

| Environment | Purpose | File |
|---|---|---|
| `gr00t` | Model training, evaluation, inference | `pyproject.toml` (managed by `uv`) |
| `gr00t-data` | Dataset conversion & stats generation | `environments/data-processing.yml` (conda) |

---

## 2. Installation — training env

Requires Python 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repo
git clone <repo-url>
git checkout rguntz/dev
cd ETHRC-Humanoid-Isaac-GR00T

# Create the virtual environment and install all dependencies
uv sync --all-extras

# Activate
source .venv/bin/activate
```

> The training env requires a CUDA-capable GPU. See `scripts/deployment/dgpu/install_deps.sh` for system-level CUDA dependencies (CUDA 12.8 on dGPU).

---

## 3. Installation — data-processing env

This lightweight conda environment is used for dataset conversion and stats generation. No GPU required.

```bash
conda env create -f environments/data-processing.yml
conda activate gr00t-data
```

To update an existing env after changes to the yml file:

```bash
conda env update -f environments/data-processing.yml --prune
```

---

## 4. G1 locomanipulation pipeline

End-to-end workflow for fine-tuning GR00T N1.7 on the Unitree G1 locomanipulation dataset.

**Task:** Navigate to an object, pick it up, navigate to a target location, and place it.
**Robot:** Unitree G1 — ego-view camera, dual-arm EEF control via IK, base navigation via WBC.
**Dataset:** LeRobot v2.1 format — 150 episodes, ~83k frames at 20 fps.

### Action space

| Dims | Key | Representation | Description |
|---|---|---|---|
| 0–8 | `left_eef_rot6d` | **RELATIVE** | Left wrist pose: xyz(3) + rot6d(6) — relative to current pose via SE(3) |
| 9–17 | `right_eef_rot6d` | **RELATIVE** | Right wrist pose: xyz(3) + rot6d(6) — relative to current pose via SE(3) |
| 18–20 | `navigate_command` | ABSOLUTE | Base velocity (lin\_x, lin\_y, ang\_z) |
| 21 | `base_height_command` | ABSOLUTE | Base height target |

### State space (model inputs)

| Key | Dims | Description |
|---|---|---|
| `left_eef_rot6d` | 9 | Left wrist current pose: xyz(3) + rot6d(6) |
| `right_eef_rot6d` | 9 | Right wrist current pose: xyz(3) + rot6d(6) |
| `ego_view` | 360×640×3 | Front RGB camera |
| Language annotation | string | Task description |

### 4.1 Download dataset from HuggingFace

The dataset is hosted on HuggingFace in LeRobot v2.1 format. `huggingface-cli` is already
available after `uv sync` (it ships with `huggingface_hub`).

```bash
conda activate gr00t-data
# Log in if the dataset repo is private
hf auth login

cd /home/rguntz/Desktop/ETHRC-Humanoid-Isaac-GR00T

hf download ETHRC-humanoid/g1-sim-locomanipulation \
    --repo-type dataset \
    --local-dir ./datasets/g1-sim-locomanipulation
```

### 4.2 Convert dataset (quat → rot6d)

The raw dataset stores EEF and base-pose orientations as quaternions. GR00T's SE(3) relative
action computation requires rot6d format (xyz + first two rows of the rotation matrix). This
script accepts **any path** to a LeRobot v2.1 dataset and creates a full copy with three new
columns added — the original is never modified.

**Required columns in the source dataset** (names are configurable via CLI args):

| Source column | Arg to override | Dims | Layout |
|---|---|---|---|
| `action.eef` | `--action-eef-key` | 14 | left pos(3) + quat(4), right pos(3) + quat(4) |
| `observation.eef_state` | `--state-eef-key` | 14 | same layout |
| `observation.robot_base_pose` | `--base-pose-key` | 7 | x, y, z, qx, qy, qz, qw |

```bash
conda activate gr00t-data

python scripts/convert_eef_quat_to_rot6d.py \
    --dataset-path    /path/to/your/raw/dataset \
    --output-path     /path/to/your/converted/dataset \
    --eef-quat-order  wxyz
```

> `--dataset-path` and `--output-path` accept any absolute or relative path.
> `--eef-quat-order` is `wxyz` for this dataset — check your data collection code if unsure.

New columns added to every episode parquet:

| New column | Dims | Description |
|---|---|---|
| `action.eef_rot6d` | 18 | Dual-arm EEF action: left xyz+rot6d, right xyz+rot6d |
| `observation.eef_state_rot6d` | 18 | Dual-arm EEF state: same layout |
| `observation.robot_base_pose_rot6d` | 9 | Base pose: xyz(3) + rot6d(6) |

Named slices are also registered in `meta/modality.json` and `meta/info.json` automatically.

### 4.3 Download base model

```bash
conda activate gr00t-data

hf download nvidia/GR00T-N1.7-3B \
    --local-dir ./models/GR00T-N1.7-3B
```

This downloads ~6 GB. The model is public — no login required.

> **Cosmos-Reason2-2B access required.** GR00T N1.7 uses `nvidia/Cosmos-Reason2-2B` as its VLM
> backbone. The weights are bundled in the GR00T download above, but the backbone architecture
> config is fetched from HuggingFace at runtime. Request access at
> https://huggingface.co/nvidia/Cosmos-Reason2-2B (approval is typically fast), then log in:
>
> ```bash
> hf auth login
> ```

### 4.4 Generate statistics

The data loader requires `meta/stats.json` (state + absolute action keys) and
`meta/relative_stats.json` (RELATIVE EEF action keys, computed on the actual SE(3) relative
transforms). Both files are generated in one run.

```bash
source .venv/bin/activate

python gr00t/data/stats.py \
    --dataset-path        ./data/g1-locomanip-rot6d \
    --embodiment-tag      NEW_EMBODIMENT \
    --modality-config-path examples/G1-LocoManip/g1_locomanip_config.py
```

> If you change `delta_indices` in the modality config (e.g. the action horizon), re-run this step.

### 4.5 Fine-tune

Log in to W&B before launching (once per machine):

```bash
wandb login
```

```bash
source .venv/bin/activate

bash examples/G1-LocoManip/finetune_g1_locomanip.sh \
    --base-model-path ./models/GR00T-N1.7-3B \
    --dataset-path    ./data/g1-locomanip-rot6d \
    --output-dir      ./outputs/g1_locomanip \
    --wandb-project   my-cluster-project
```

Key env-var overrides:

```bash
NUM_GPUS=4              # multi-GPU training via torchrun
MAX_STEPS=20000         # default: 10000
GLOBAL_BATCH_SIZE=64    # default: 32
USE_WANDB=0             # disable W&B logging
```

