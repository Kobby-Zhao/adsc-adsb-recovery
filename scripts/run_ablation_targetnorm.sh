#!/bin/bash
# Full ablation A0-A3 with target_norm enabled
set -e
cd /home/jj/workspace/data-0313
export PYTHONPATH=/home/jj/workspace/data-0313

for name in ablation_a0_baseline ablation_a1_step1 ablation_a2_step2 ablation_a3_step3; do
    echo ""
    echo "===== $(date): Starting $name ====="
    python3 scripts/train.py --config "configs/alt_focus/ablation_submodules/${name}.yaml" 2>&1
    echo "===== $(date): DONE $name ====="
done

echo ""
echo "===== ALL DONE ====="
