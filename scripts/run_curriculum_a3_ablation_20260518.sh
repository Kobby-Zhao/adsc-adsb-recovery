#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # Keep the training environment consistent with previous ADS-C experiments.
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-adsc_01}"
fi
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

RUN_TAG="${RUN_TAG:-obscons_gaponly_curriculum_a3_v1}"
DEVICE="${DEVICE:-cuda}"
ONLY="${ONLY:-g0_progressive_curriculum,g1_uniform_sampling,g2_stage3_only,g3_reverse_curriculum}"
GENERATE_ONLY="${GENERATE_ONLY:-0}"

EXP_ROOT="outputs/experiments/obs_conditioned_gaponly/${RUN_TAG}"
CONFIG_DIR="${EXP_ROOT}/configs"
SRC_CFG="outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_physical_time_ablation_v1/configs/a3_risk_routed.yaml"
LOG_FILE="${EXP_ROOT}/train_live.log"

mkdir -p "${CONFIG_DIR}"

python - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

import yaml

run_tag = os.environ.get("RUN_TAG", "obscons_gaponly_curriculum_a3_v1")
device = os.environ.get("DEVICE", "cuda")
root = Path("outputs/experiments/obs_conditioned_gaponly") / run_tag
config_dir = root / "configs"
src_cfg = Path("outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_physical_time_ablation_v1/configs/a3_risk_routed.yaml")

with src_cfg.open("r", encoding="utf-8") as f:
    base = yaml.safe_load(f)

schedules = {
    "g0_progressive_curriculum": [
        {"end_epoch": 6, "weights": {"stage1": 0.70, "stage2": 0.25, "stage3": 0.05}},
        {"end_epoch": 12, "weights": {"stage1": 0.20, "stage2": 0.60, "stage3": 0.20}},
        {"end_epoch": 18, "weights": {"stage1": 0.15, "stage2": 0.35, "stage3": 0.60}},
        {"end_epoch": 24, "weights": {"stage1": 0.05, "stage2": 0.20, "stage3": 0.75}},
    ],
    "g1_uniform_sampling": [
        {"end_epoch": 24, "weights": {"stage1": 1 / 3, "stage2": 1 / 3, "stage3": 1 / 3}},
    ],
    "g2_stage3_only": [
        {"end_epoch": 24, "weights": {"stage1": 0.0, "stage2": 0.0, "stage3": 1.0}},
    ],
    "g3_reverse_curriculum": [
        {"end_epoch": 6, "weights": {"stage1": 0.05, "stage2": 0.25, "stage3": 0.70}},
        {"end_epoch": 12, "weights": {"stage1": 0.15, "stage2": 0.40, "stage3": 0.45}},
        {"end_epoch": 18, "weights": {"stage1": 0.35, "stage2": 0.40, "stage3": 0.25}},
        {"end_epoch": 24, "weights": {"stage1": 0.65, "stage2": 0.25, "stage3": 0.10}},
    ],
}

notes = {
    "g0_progressive_curriculum": "G0: proposed easy-to-hard progressive curriculum using latest A3.",
    "g1_uniform_sampling": "G1: uniform random sampling from S1/S2/S3 using latest A3.",
    "g2_stage3_only": "G2: only stage3 hard sparse samples using latest A3.",
    "g3_reverse_curriculum": "G3: hard-to-easy reverse curriculum using latest A3.",
}

for name, schedule in schedules.items():
    cfg = dict(base)
    cfg["training"] = dict(base["training"])
    cfg["training"]["device"] = device
    cfg["training"]["curriculum"] = dict(base["training"]["curriculum"])
    cfg["training"]["curriculum"]["enabled"] = True
    cfg["training"]["curriculum"]["schedule"] = schedule
    cfg["outputs"] = dict(base["outputs"])
    cfg["outputs"]["run_dir"] = str(root / name)
    cfg["outputs"]["checkpoint_name"] = "best.pt"
    cfg["experiment_note"] = notes[name]

    out = config_dir / f"{name}.yaml"
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(out)
PY

IFS=',' read -r -a SELECTED <<< "${ONLY}"

{
  echo "===== curriculum A3 ablation ====="
  echo "time: $(date -Is)"
  echo "run_tag: ${RUN_TAG}"
  echo "device: ${DEVICE}"
  echo "selected: ${ONLY}"
  echo "source_config: ${SRC_CFG}"
  echo
} | tee -a "${LOG_FILE}"

if [[ "${GENERATE_ONLY}" == "1" ]]; then
  echo "===== generate only; training not started =====" | tee -a "${LOG_FILE}"
  exit 0
fi

for name in "${SELECTED[@]}"; do
  cfg="${CONFIG_DIR}/${name}.yaml"
  run_dir="${EXP_ROOT}/${name}"
  ckpt="${run_dir}/best.pt"
  if [[ ! -f "${cfg}" ]]; then
    echo "[error] missing config: ${cfg}" | tee -a "${LOG_FILE}"
    exit 1
  fi

  echo "===== train ${name} $(date -Is) =====" | tee -a "${LOG_FILE}"
  python scripts/train.py --config "${cfg}" 2>&1 | tee -a "${LOG_FILE}"

  echo "===== evaluate ${name} $(date -Is) =====" | tee -a "${LOG_FILE}"
  python scripts/evaluate.py --config "${cfg}" --checkpoint "${ckpt}" --split test --plot-count 0 2>&1 | tee -a "${LOG_FILE}"
done

echo "===== done $(date -Is) =====" | tee -a "${LOG_FILE}"
