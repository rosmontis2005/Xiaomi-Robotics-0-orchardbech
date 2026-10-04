#!/usr/bin/env bash
# Frozen experiment entrypoint; all training logic remains in train_orchardbench.sh.
set -euo pipefail

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
if (( $# != 0 )); then
  fail 'This fixed experiment accepts no positional arguments or Hydra overrides.'
fi

XR0_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ORCHARD_REPO=/home/rosmontis/Projects/orchardbench
export DATA_CONFIG=orchardbench
export DATASET_ROOT=/home/rosmontis/Projects/orchardbench/data/orchard_v1_2650/filtered
export ORCHARD_STATS="$DATASET_ROOT/action_stats.json"
export PRETRAINED_CKPT=/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/checkpoints/Xiaomi-Robotics-0-Calvin-ABCD_D
export PROCESSOR_PATH="$PRETRAINED_CKPT"
export VLM_CONFIG_PATH="$PRETRAINED_CKPT/config.json"
export OUTPUT_ROOT="$XR0_ROOT/outputs/orchard_v1_2650"
export EXP_NAME=orchard_v1_2650_frozen_vlm_seed42
export BATCH_SIZE=1
export LR=1e-5
export MAX_STEPS=10000
export SEED=42
export FREEZE_VLM=true

# Runtime resource selection is user-configurable; experiment settings above are fixed.
export GPU_IDS="${GPU_IDS:-0}"
export RESOURCE_GPU="${RESOURCE_GPU:-1}"
export PYTHON_BIN="${PYTHON_BIN:-$XR0_ROOT/.venv-orchard/bin/python}"
export DRY_RUN="${DRY_RUN:-0}"
case "$DRY_RUN" in
  0|1) ;;
  *) fail 'DRY_RUN must be 0 (train) or 1 (resolve config and exit).' ;;
esac
[[ "$RESOURCE_GPU" =~ ^[1-9][0-9]*$ ]] || fail 'RESOURCE_GPU must be a positive integer.'
command -v "$PYTHON_BIN" >/dev/null || fail "Python executable not found: $PYTHON_BIN"

RUN_DIR="$OUTPUT_ROOT/project_orchardbench/$EXP_NAME"
export WANDB_MODE=offline

# Fail before handing off if the frozen inputs are missing. The existing Dataset
# additionally validates the contract and the exact train-annotation fingerprint.
for path in "$DATASET_ROOT/json/train" "$DATASET_ROOT/json/val" "$PRETRAINED_CKPT"; do
  [[ -d "$path" ]] || fail "Required directory not found: $path"
done
for path in \
  "$DATASET_ROOT/manifest.jsonl" "$ORCHARD_STATS" \
  "$ORCHARD_REPO/treesim/orchard_action.py" \
  "$VLM_CONFIG_PATH" "$PRETRAINED_CKPT/model.safetensors.index.json" \
  "$PRETRAINED_CKPT/model-00001-of-00002.safetensors" \
  "$PRETRAINED_CKPT/model-00002-of-00002.safetensors" \
  "$PRETRAINED_CKPT/processor_config.json" "$PRETRAINED_CKPT/preprocessor_config.json" \
  "$PRETRAINED_CKPT/tokenizer_config.json" "$PRETRAINED_CKPT/tokenizer.json"; do
  [[ -r "$path" && -s "$path" ]] || fail "Required file missing, empty or unreadable: $path"
done
if [[ "$DRY_RUN" == 0 && ( -e "$RUN_DIR" || -L "$RUN_DIR" ) ]]; then
  fail "Output already exists; this fresh experiment will not overwrite or resume it: $RUN_DIR"
fi

printf '%s\n' \
  'Orchard V1 / 2650 accepted cohort / frozen-VLM post-training' \
  "DATASET_ROOT=$DATASET_ROOT" \
  "ORCHARD_STATS=$ORCHARD_STATS" \
  "PRETRAINED_CKPT=$PRETRAINED_CKPT (weights only; fresh optimizer/scheduler; step 0)" \
  "OUTPUT_ROOT=$OUTPUT_ROOT" \
  "EXP_NAME=$EXP_NAME" \
  "RUN_DIR=$RUN_DIR" \
  "BATCH_SIZE=$BATCH_SIZE LR=$LR MAX_STEPS=$MAX_STEPS SEED=$SEED FREEZE_VLM=$FREEZE_VLM" \
  'precision=bf16-mixed optimizer=AdamW betas=[0.9,0.95] weight_decay=0.1 gradient_clip=1.0' \
  'training_repeat=1 enable_freq=false async_train=false accumulate_grad_batches=1' \
  'contract=orchard_cartesian_local_rotvec_width_v1 action_shape=[30,32] state_shape=[1,32] active_dims=0:7' \
  'save_interval=1000 save_top_k=-1 save_last=true (retain steps 1000/3000/6000/10000)' \
  "GPU_IDS=$GPU_IDS RESOURCE_GPU=$RESOURCE_GPU PYTHON_BIN=$PYTHON_BIN" \
  "DRY_RUN=$DRY_RUN WANDB_MODE=$WANDB_MODE"

# The generic launcher resolves and prints the full Hydra config first. With
# DRY_RUN=1 it exits before scripts/train.sh / torchrun / Trainer.fit.
exec bash "$XR0_ROOT/scripts/train_orchardbench.sh" \
  trainer.precision=bf16-mixed \
  trainer.optimizer.type=torch.optim.AdamW \
  'trainer.optimizer.params.betas=[0.9,0.95]' \
  trainer.optimizer.params.weight_decay=0.1 \
  trainer.gradient_clip_val=1.0 \
  trainer.accumulate_grad_batches=1 \
  trainer.save_interval=1000 \
  model.params.model.training_repeat=1 \
  model.params.model.enable_freq=false \
  "hydra.run.dir=$RUN_DIR/hydra" \
  hydra.job.chdir=false
