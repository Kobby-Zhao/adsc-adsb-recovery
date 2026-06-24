#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate adsc_01
set -u
export PYTHONNOUSERSITE=1

python scripts/make_bimamba_hiddenfusion_config_20260523.py

CFG="outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_bimamba_hiddenfusion_anchorrelative_v3/configs/bimamba_hiddenfusion_clean_absolute.yaml"
RUN_DIR="outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_bimamba_hiddenfusion_anchorrelative_v3/bimamba_hiddenfusion_anchorrelative"
mkdir -p "$RUN_DIR"

echo "===== train bimamba_hiddenfusion_clean_absolute $(date -Is) ====="
python scripts/train.py --config "$CFG" 2>&1 | tee -a "$RUN_DIR/train_console.log"
