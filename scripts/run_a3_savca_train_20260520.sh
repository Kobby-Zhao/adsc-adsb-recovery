#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_a3_savca_v1"
CONFIG="${RUN_ROOT}/configs/a3_savca_routed.yaml"
RUN_DIR="${RUN_ROOT}/a3_savca_routed"

mkdir -p "${RUN_DIR}"

echo "===== A3-SAVCA training ====="
echo "time: $(date -Is)"
echo "config: ${CONFIG}"
echo "run_dir: ${RUN_DIR}"

python scripts/train.py --config "${CONFIG}" 2>&1 | tee -a "${RUN_DIR}/train_console.log"
