#!/bin/bash
set -e
cd /home/jj/workspace/data-0313
export PYTHONPATH=/home/jj/workspace/data-0313

echo "===== A0: our_method baseline ====="
python3 scripts/train.py --config configs/alt_focus/ablation_submodules/ablation_a0_baseline.yaml 2>&1
echo "DONE A0"

echo "===== A1: our_method + anchor_relative ====="
python3 scripts/train.py --config configs/alt_focus/ablation_submodules/ablation_a1_step1.yaml 2>&1
echo "DONE A1"
