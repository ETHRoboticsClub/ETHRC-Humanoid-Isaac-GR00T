#!/usr/bin/env bash
# Fine-tune GR00T N1.7 on a real G1 + SONIC dataset with 3 cameras.
#
# Robot  : Unitree G1 + SONIC whole-body controller
# Cameras: ego_view, left_wrist, right_wrist
# Action : SONIC motion-token latents + hand joints (ABSOLUTE, 40-step horizon)
#
# Usage:
#   bash examples/G1-SONIC-3cam/finetune_g1_sonic_3cam.sh \
#     --base-model-path   <path|hf-repo> \
#     --dataset-path      <local-path-to-dataset> \
#     --output-dir        <output-dir> \
#     [--max-steps        <int>] \
#     [--save-steps       <int>] \
#     [--save-total-limit <int>] \
#     [-- <extra launch_finetune.py args>]

set -x -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-0}"
MAX_STEPS="${MAX_STEPS:-10000}"
USE_WANDB="${USE_WANDB:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
SHARD_SIZE="${SHARD_SIZE:-1024}"
NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-100000}"
EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-0.1}"

BASE_MODEL_PATH=""
DATASET_PATH=""
OUTPUT_DIR=""
EXPERIMENT_NAME=""
WANDB_PROJECT=""
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage: bash examples/G1-SONIC-3cam/finetune_g1_sonic_3cam.sh \
  --base-model-path   <path>   HuggingFace repo or local checkpoint
  --dataset-path      <path>   Local path to the LeRobot v2.1 dataset
  --output-dir        <path>   Where to write checkpoints
  [--experiment-name  <name>]
  [--wandb-project    <name>]
  [--max-steps        <int>]   Default: 10000
  [--save-steps       <int>]   Default: 5000
  [--save-total-limit <int>]   Default: 0 (keep all)
  [-- <extra launch_finetune.py args>...]
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --base-model-path)   BASE_MODEL_PATH="$2";  shift 2 ;;
        --dataset-path)      DATASET_PATH="$2";     shift 2 ;;
        --output-dir)        OUTPUT_DIR="$2";        shift 2 ;;
        --experiment-name)   EXPERIMENT_NAME="$2";  shift 2 ;;
        --wandb-project)     WANDB_PROJECT="$2";    shift 2 ;;
        --max-steps)         MAX_STEPS="$2";        shift 2 ;;
        --save-steps)        SAVE_STEPS="$2";       shift 2 ;;
        --save-total-limit)  SAVE_TOTAL_LIMIT="$2"; shift 2 ;;
        --help|-h)           usage; exit 0 ;;
        --)                  shift; EXTRA_ARGS=("$@"); break ;;
        *)                   echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

for var in BASE_MODEL_PATH DATASET_PATH OUTPUT_DIR; do
    if [ -z "${!var}" ]; then
        echo "Missing required argument: ${var}" >&2
        usage >&2
        exit 1
    fi
done

WANDB_FLAG=()
[ "$USE_WANDB" = "1" ] && WANDB_FLAG+=(--use_wandb)

LAUNCH_CMD=(
    gr00t/experiment/launch_finetune.py
    --base_model_path      "$BASE_MODEL_PATH"
    --dataset_path         "$DATASET_PATH"
    --embodiment_tag       UNITREE_G1_SONIC
    --modality_config_path "$SCRIPT_DIR/g1_sonic_3cam_config.py"
    --num_gpus             "$NUM_GPUS"
    --output_dir           "$OUTPUT_DIR"
    --save_steps           "$SAVE_STEPS"
    --save_total_limit     "$SAVE_TOTAL_LIMIT"
    --max_steps            "$MAX_STEPS"
    --warmup_ratio         0.05
    --weight_decay         1e-5
    --learning_rate        1e-4
    "${WANDB_FLAG[@]}"
    --global_batch_size            "$GLOBAL_BATCH_SIZE"
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
    --dataloader_num_workers       "$DATALOADER_NUM_WORKERS"
    --shard_size                   "$SHARD_SIZE"
    --num_shards_per_epoch         "$NUM_SHARDS_PER_EPOCH"
    --episode_sampling_rate        "$EPISODE_SAMPLING_RATE"
)

[ -n "$EXPERIMENT_NAME" ] && LAUNCH_CMD+=(--experiment_name "$EXPERIMENT_NAME")
[ -n "$WANDB_PROJECT"   ] && LAUNCH_CMD+=(--wandb_project   "$WANDB_PROJECT")
[ "${#EXTRA_ARGS[@]}" -gt 0 ] && LAUNCH_CMD+=("${EXTRA_ARGS[@]}")

if [ "$NUM_GPUS" = "1" ]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    exec python "${LAUNCH_CMD[@]}"
fi

exec torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" "${LAUNCH_CMD[@]}"
