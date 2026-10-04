#!/usr/bin/env bash
set -euo pipefail
XR0_BENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XR0_BENCH_PYTHON="${PYTHON_BIN:-$XR0_BENCH_ROOT/.venv-orchard/bin/python}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$XR0_BENCH_PYTHON" -u "$XR0_BENCH_ROOT/tools/eval_orchard_learning_curve.py" "$@"
