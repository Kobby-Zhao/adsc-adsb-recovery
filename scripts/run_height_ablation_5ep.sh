#!/usr/bin/env bash
set -euo pipefail

cd /home/jj/workspace/data-0313

python scripts/prepare_height_ablation_5ep_configs.py

CONFIG_DIR="outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/ablation_height"

python scripts/train.py --config "${CONFIG_DIR}/A0_shared3d_5ep.yaml"
python scripts/train.py --config "${CONFIG_DIR}/A1_xyaux_zlinear_5ep.yaml"
python scripts/train.py --config "${CONFIG_DIR}/A2_xyaux_zlinear_zadapter_5ep.yaml"
python scripts/train.py --config "${CONFIG_DIR}/A3_gapaware_small_5ep.yaml"
