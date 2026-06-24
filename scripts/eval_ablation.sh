#!/bin/bash
# Evaluate A0-A3 on stage3 test set
set -e
cd /home/jj/workspace/data-0313
export PYTHONPATH=/home/jj/workspace/data-0313

CONFIGS=(
    "configs/alt_focus/ablation_submodules/ablation_a0_baseline.yaml"
    "configs/alt_focus/ablation_submodules/ablation_a1_step1.yaml"
    "configs/alt_focus/ablation_submodules/ablation_a2_step2.yaml"
    "configs/alt_focus/ablation_submodules/ablation_a3_step3.yaml"
)

for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .yaml)
    echo "===== Evaluating: $name ====="
    python3 scripts/evaluate.py --config "$cfg" --split test --plot-count 0 2>&1
    echo "DONE: $name"
    echo ""
done

echo "=== All evaluations complete ==="

# Print summary
echo ""
echo "===== SUMMARY ====="
for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .yaml)
    run_dir=$(grep "run_dir:" "$cfg" | awk '{print $2}')
    table="$run_dir/main_task_metrics_test_main_table.json"
    if [ -f "$table" ]; then
        echo "--- $name ---"
        python3 -c "
import json
d = json.load(open('$table'))
for row in d:
    print(f\"  {row['metric']:30s} {row['value']}\")
"
    fi
done
