#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_TAG="${RUN_TAG:-obscons_gaponly_savca_supervised_v1}"
CONFIG_ROOT="outputs/experiments/obs_conditioned_gaponly/${RUN_TAG}"
CONFIG_PATH="${CONFIG_ROOT}/configs/savca_supervised.yaml"
RUN_DIR="${CONFIG_ROOT}/savca_supervised"

python scripts/make_savca_supervised_config_20260520.py
mkdir -p "$RUN_DIR"

echo "===== SAVCA supervised training ====="
echo "time: $(date -Is)"
echo "run_tag: ${RUN_TAG}"
echo "config: ${CONFIG_PATH}"
echo "run_dir: ${RUN_DIR}"

python scripts/train.py --config "$CONFIG_PATH" 2>&1 | tee -a "${RUN_DIR}/train_console.log"
