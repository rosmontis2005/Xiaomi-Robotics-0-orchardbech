#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0
DATASET_PATH=/home/rosmontis/Projects/dualsys/calvin/dataset/task_D_D
SEQUENCE_FILE="$REPO_ROOT/eval_calvin/config/eval_sequences.json"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate xr0-calvin
cd "$REPO_ROOT"

if ! ss -ltn | grep -q '127\.0\.0\.1:10086'; then
    printf '%s\n' \
        'XR-0 inference server is not listening on localhost:10086.' \
        '' \
        'Start it in another terminal with:' \
        '' \
        'spectator/run_server.sh'
    exit 1
fi

CALVIN_SEQUENCE_COUNT="$(python -c 'import json, sys; print(len(json.load(open(sys.argv[1]))))' "$SEQUENCE_FILE")"
NUM_SEQUENCES="${XR0_NUM_SEQUENCES:-$CALVIN_SEQUENCE_COUNT}"

if [[ ! "$NUM_SEQUENCES" =~ ^[1-9][0-9]*$ ]] || (( NUM_SEQUENCES > CALVIN_SEQUENCE_COUNT )); then
    printf 'XR0_NUM_SEQUENCES must be an integer from 1 to %s (got: %s).\n' \
        "$CALVIN_SEQUENCE_COUNT" "$NUM_SEQUENCES" >&2
    exit 2
fi

SESSION_ID="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="$REPO_ROOT/logs/calvin_gui_continuous/$SESSION_ID"
mkdir -p "$LOG_ROOT"

printf 'CALVIN evaluation sequence count: %s\n' "$CALVIN_SEQUENCE_COUNT"
printf 'Sequences requested: %s\n' "$NUM_SEQUENCES"
printf 'Continuous log root: %s\n' "$LOG_ROOT"
printf '%s\n' 'Press Ctrl+C to stop the spectator.'

python eval_calvin/main.py \
    --split abcd \
    --dataset-path "$DATASET_PATH" \
    --world-size 1 \
    --rank 0 \
    --num-workers 1 \
    --num-sequences "$NUM_SEQUENCES" \
    --debug \
    --show-gui \
    --gui-step-sleep 0.033 \
    --CACHE-ROOT "$LOG_ROOT" \
    2>&1 | tee -a "$LOG_ROOT/evaluator.log"
