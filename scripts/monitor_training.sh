#!/usr/bin/env bash
# Live training monitor — refreshes every 10s.
# Usage: bash scripts/monitor_training.sh

LOG_DIR="outputs/experiments/curriculum_20260415_exp4cmp_s2v2"

MODELS=("UniLSTM" "Kalman" "CNN_LSTM" "LSTM_Attention" "Transformer" "BiLSTM" "OurMethod")

while true; do
  clear
  echo "════════════════════════════════════════════════════════════════"
  echo "  BASELINE RETRAINING MONITOR  ($(date +%H:%M:%S))"
  echo "════════════════════════════════════════════════════════════════"
  printf "%-18s %6s %14s %14s %14s %10s\n" "MODEL" "EPOCH" "TRAIN_LOSS" "VAL_GAP_ALT" "VAL_ALT" "EPOCH_SEC"
  echo "──────────────────────────────────────────────────────────────────"

  for name in "${MODELS[@]}"; do
    log="${LOG_DIR}/retrain_${name}.log"
    if [ -f "$log" ]; then
      last=$(grep "epoch_end" "$log" | tail -1)
      if [ -n "$last" ]; then
        ep=$(echo "$last" | grep -oP 'epoch_end \K\d+')
        tl=$(echo "$last" | grep -oP 'train_loss=\K[0-9.]+')
        va=$(echo "$last" | grep -oP 'val_gap_alt_rmse=\K[0-9.]+')
        vh=$(echo "$last" | grep -oP 'val_alt_rmse=\K[0-9.]+')
        es=$(echo "$last" | grep -oP 'epoch_sec=\K[0-9.]+')
        printf "%-18s %3s/24 %14s %14s %14s %9ss\n" \
          "$name" "${ep:-?}" "${tl:-?}" "${va:-?}" "${vh:-?}" "${es:-?}"
      else
        printf "%-18s %6s\n" "$name" "init..."
      fi
    else
      printf "%-18s %6s\n" "$name" "(no log)"
    fi
  done

  echo "──────────────────────────────────────────────────────────────────"
  echo "  tail -f ${LOG_DIR}/retrain_<MODEL>.log   (detailed log)"
  echo "════════════════════════════════════════════════════════════════"
  sleep 10
done
