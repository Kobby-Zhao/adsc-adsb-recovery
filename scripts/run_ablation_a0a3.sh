#!/bin/bash
# Ablation experiment: A0 → A1 → A2 → A3
# Height branch three-step ablation
set -e
cd /home/jj/workspace/data-0313
export PYTHONPATH=/home/jj/workspace/data-0313

CONFIGS=(
    "configs/alt_focus/ablation_submodules/ablation_a0_baseline.yaml"
    "configs/alt_focus/ablation_submodules/ablation_a1_step1.yaml"
    "configs/alt_focus/ablation_submodules/ablation_a2_step2.yaml"
    "configs/alt_focus/ablation_submodules/ablation_a3_step3.yaml"
)

LOGDIR="outputs/experiments/ablation_submodules"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/ablation_run_$(date +%Y%m%d_%H%M%S).log"

echo "=== Ablation experiment started at $(date) ===" | tee -a "$LOGFILE"
echo "Configs: ${CONFIGS[@]}" | tee -a "$LOGFILE"

for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .yaml)
    echo "" | tee -a "$LOGFILE"
    echo "===== Running: $name =====" | tee -a "$LOGFILE"
    echo "Start: $(date)" | tee -a "$LOGFILE"
    python3 scripts/train.py --config "$cfg" 2>&1 | tee -a "$LOGFILE"
    rc=${PIPESTATUS[0]}
    if [ $rc -ne 0 ]; then
        echo "FAILED with exit code $rc" | tee -a "$LOGFILE"
    else
        echo "DONE: $name at $(date)" | tee -a "$LOGFILE"
    fi
done

echo "" | tee -a "$LOGFILE"
echo "=== Ablation experiment finished at $(date) ===" | tee -a "$LOGFILE"
