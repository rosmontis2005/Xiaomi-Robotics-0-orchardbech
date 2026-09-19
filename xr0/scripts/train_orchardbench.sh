#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
XR0_ROOT="$PWD"
export ORCHARD_REPO="${ORCHARD_REPO:-$(realpath "$XR0_ROOT/../../../orchardbench")}"
export PYTHONPATH="$XR0_ROOT:$ORCHARD_REPO:${PYTHONPATH:-}"
export DATASET_ROOT="${DATASET_ROOT:-$(realpath "$ORCHARD_REPO/../data/orchard_autopicker_v1")}"
export ORCHARD_STATS="${ORCHARD_STATS:-$XR0_ROOT/artifacts/orchardbench/action_stats.json}"
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-$XR0_ROOT/../checkpoints/Xiaomi-Robotics-0-Calvin-ABCD_D}"
export PROCESSOR_PATH="${PROCESSOR_PATH:-$PRETRAINED_CKPT}"
export VLM_CONFIG_PATH="${VLM_CONFIG_PATH:-$PROCESSOR_PATH/config.json}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
export RESOURCE_GPU="${RESOURCE_GPU:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ -n "${PYTHON_BIN:-}" ]]; then
  export PATH="$(dirname "$PYTHON_BIN"):$PATH"
elif [[ -x "$XR0_ROOT/.venv-orchard/bin/python" ]]; then
  export PATH="$XR0_ROOT/.venv-orchard/bin:$PATH"
fi
export PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
args=("data=${DATA_CONFIG:-orchardbench}" model=orchardbench trainer=orchardbench
  "trainer.default_root_dir=${OUTPUT_ROOT:-$XR0_ROOT/outputs}" "trainer.exp_name=${EXP_NAME:-cartesian_v1}"
  "data.params.train_datasets.batch_size=${BATCH_SIZE:-1}" "trainer.optimizer.params.lr=${LR:-1e-5}"
  "trainer.max_steps=${MAX_STEPS:-3000}" "trainer.seed=${SEED:-42}"
  "model.params.freeze_vlm=${FREEZE_VLM:-true}" model.params.model.async_train=false trainer.ckpt_path=null)
if (( RESOURCE_GPU > 1 )); then args+=(trainer.strategy.type=ddp); fi
printf 'Orchard dataset=%s\nstats=%s\nweights-only checkpoint=%s\nGPU IDs=%s resources=%s\n' "$DATASET_ROOT" "$ORCHARD_STATS" "$PRETRAINED_CKPT" "$CUDA_VISIBLE_DEVICES" "$RESOURCE_GPU"
# Hydra resolves and prints every effective override before launching.
"$PYTHON_BIN" tools/train.py "${args[@]}" "$@" --cfg job --resolve
if [[ "${DRY_RUN:-0}" == 1 ]]; then exit 0; fi
exec bash scripts/train.sh "${args[@]}" "$@"
