#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate xr0-mibot
cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES=0

python deploy/server.py \
  --model "$REPO_ROOT/checkpoints/Xiaomi-Robotics-0-Calvin-ABCD_D" \
  --host localhost \
  --port 10086 \
  2>&1 | tee -a "$REPO_ROOT/logs/calvin_gui/server.log"
