#!/usr/bin/env bash
set -euo pipefail
cd /home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/experiments/orchard_v2_1010
SIM=/home/rosmontis/Projects/orchardbench/.pixi/envs/default/bin/python
TRAIN=/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/.venv-orchard/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$SIM" -u visibility.py >> visibility.log 2>&1
"$TRAIN" -u freeze.py > freeze.log 2>&1
if [[ -f checkpoints/latest_resume.pt ]]; then
  "$TRAIN" -u train.py --resume >> training.log 2>&1
else
  "$TRAIN" -u train.py >> training.log 2>&1
fi
[[ -f training_complete.json ]]
"$TRAIN" -u analyze.py > analyze.log 2>&1
"$SIM" -u evaluate.py --expert > expert.log 2>&1
for step in 0 10000 20000 40000 60000; do
  "$TRAIN" -u evaluate.py --step "$step" > "evaluation_${step}.log" 2>&1
  "$TRAIN" -u analyze.py >> analyze.log 2>&1
done
