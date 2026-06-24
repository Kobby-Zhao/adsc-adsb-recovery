#!/usr/bin/env bash
set -euo pipefail

cd /home/jj/workspace/data-0313

python scripts/prepare_height_ablation_24ep_configs.py

CONFIG_DIR="outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/ablation_height_24ep"
RUN_ROOT="outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/ablation_height_24ep"

python scripts/evaluate.py \
  --config "${CONFIG_DIR}/A0_shared3d_24ep.yaml" \
  --checkpoint "${RUN_ROOT}/A0_shared3d_24ep/best.pt" \
  --split test \
  --plot-count 0

python scripts/evaluate.py \
  --config "${CONFIG_DIR}/A1_xyaux_zlinear_24ep.yaml" \
  --checkpoint "${RUN_ROOT}/A1_xyaux_zlinear_24ep/best.pt" \
  --split test \
  --plot-count 0

python scripts/evaluate.py \
  --config "${CONFIG_DIR}/A2_xyaux_zlinear_zadapter_24ep.yaml" \
  --checkpoint "${RUN_ROOT}/A2_xyaux_zlinear_zadapter_24ep/best.pt" \
  --split test \
  --plot-count 0

python scripts/evaluate.py \
  --config "${CONFIG_DIR}/A3_gapaware_small_24ep.yaml" \
  --checkpoint "${RUN_ROOT}/A3_gapaware_small_24ep/best.pt" \
  --split test \
  --plot-count 0
