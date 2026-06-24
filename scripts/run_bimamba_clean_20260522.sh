#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate adsc_01
export PYTHONNOUSERSITE=1

python scripts/make_bimamba_clean_config_20260522.py

CFG="outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_bimamba_recurrent_anchorfix_v4/configs/bimamba_recurrent_clean_absolute.yaml"
RUN_DIR="outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_bimamba_recurrent_anchorfix_v4/bimamba_clean_absolute"
mkdir -p "$RUN_DIR"

echo "===== train bimamba_recurrent_clean_absolute $(date -Is) ====="
python scripts/train.py --config "$CFG" 2>&1 | tee -a "$RUN_DIR/train_console.log"
