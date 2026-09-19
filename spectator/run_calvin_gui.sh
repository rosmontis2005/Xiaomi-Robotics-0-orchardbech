#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate xr0-calvin
cd "$REPO_ROOT"

python eval_calvin/main.py \
  --split abcd \
  --dataset-path /home/rosmontis/Projects/dualsys/calvin/dataset/task_D_D \
  --world-size 1 \
  --rank 0 \
  --num-workers 1 \
  --num-sequences 1 \
  --debug \
  --show-gui \
  --gui-step-sleep 0.033 \
  --CACHE-ROOT "$REPO_ROOT/logs/calvin_gui" \
  2>&1 | tee -a "$REPO_ROOT/logs/calvin_gui/evaluator.log"
