#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_savca_only_v1"
CONFIG="${RUN_ROOT}/configs/savca_only.yaml"
RUN_DIR="${RUN_ROOT}/savca_only"

mkdir -p "${RUN_DIR}"

echo "===== SAVCA-only training ====="
echo "time: $(date -Is)"
echo "config: ${CONFIG}"
echo "run_dir: ${RUN_DIR}"

python scripts/train.py --config "${CONFIG}" 2>&1 | tee -a "${RUN_DIR}/train_console.log"
