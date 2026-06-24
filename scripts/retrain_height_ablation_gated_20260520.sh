#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/jj/workspace/data-0313"
RUN_TAG="${RUN_TAG:-obscons_gaponly_height_ablation_gated_v1}"
DEVICE="${DEVICE:-cuda}"
ONLY="${ONLY:-ours_backbone_absolute,a1_linear_alt_baseline,a2_gated_offset,a3_gated_routed}"
GENERATE_ONLY="${GENERATE_ONLY:-0}"

EXP_ROOT="${ROOT_DIR}/outputs/experiments/obs_conditioned_gaponly/${RUN_TAG}"
CONFIG_DIR="${EXP_ROOT}/configs"
LOG_FILE="${EXP_ROOT}/train_live.log"
SUMMARY_FILE="${EXP_ROOT}/run_summary.txt"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate adsc_01
export PYTHONNOUSERSITE=1
cd "${ROOT_DIR}"

mkdir -p "${CONFIG_DIR}"

python - <<'PY'
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

run_tag = os.environ.get("RUN_TAG", "obscons_gaponly_height_ablation_gated_v1")
device = os.environ.get("DEVICE", "cuda")
config_dir = Path("outputs/experiments/obs_conditioned_gaponly") / run_tag / "configs"
template_path = Path("configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_proposed_24e.yaml")
base_config = Path("configs/alt_focus/train_v1_control_e5_20260321.yaml").resolve()
template = yaml.safe_load(template_path.read_text(encoding="utf-8"))


def deep_update(dst: dict, src: dict) -> dict:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


common_loss = {
    "anchor_weight": 0.0,
    "gap_weight": 1.0,
    "lambda_unc": 0.0,
    "fusion_reg_lambda": 0.0,
}

common_main = {
    "main_rmax_m": 152.4,
    "main_rmax_min_m": 91.44,
    "main_rmax_slope_m_per_min": 4.572,
    "main_rmax_max_m": 365.76,
}

residual_gate = {
    "alt_residual_anchor_delta_gate_enabled": True,
    "alt_residual_anchor_delta_gate_low_m": 60.0,
    "alt_residual_anchor_delta_gate_high_m": 180.0,
    "alt_residual_anchor_delta_gate_min_scale": 0.0,
    "alt_residual_edge_taper_enabled": True,
    "alt_residual_edge_taper_steps": 3.0,
}

zero_alt_losses = {
    "lambda_vertical_smooth": 0.0,
    "lambda_alt_gate_supervision": 0.0,
    "lambda_alt_gate_risk_shrink": 0.0,
    "lambda_alt_edge_first_diff": 0.0,
    "lambda_alt_edge_second_diff": 0.0,
    "lambda_alt_segment_bound": 0.0,
    "lambda_alt_vertical_rate_penalty": 0.0,
    "lambda_alt_boundary_anchor": 0.0,
}

experiments = [
    {
        "name": "ours_backbone_absolute",
        "note": "A0: bidirectional backbone only after fixed anchor-fill logic.",
        "override": {
            "model": {
                "backbone_type": "bilstm",
                "model_variant": "structured_fusion",
                "alt_gate_enabled": False,
                "alt_main_mode": "absolute",
                "main_rmax_m": 0.0,
            },
            "training": {"risk_aware": {"use_segment_teacher": False, "use_alt_baseline_residual": False}},
            "loss": zero_alt_losses,
        },
    },
    {
        "name": "a1_linear_alt_baseline",
        "note": "A1: corrected nearest-anchor linear altitude baseline.",
        "override": {
            "model": {
                "backbone_type": "bilstm",
                "model_variant": "structured_fusion",
                "alt_gate_enabled": False,
                "alt_main_mode": "anchor_relative",
                "alt_anchor_reference_mode": "local_linear",
                "main_rmax_m": 0.0,
            },
            "training": {"risk_aware": {"use_segment_teacher": False, "use_alt_baseline_residual": False}},
            "loss": zero_alt_losses,
        },
    },
    {
        "name": "a2_gated_offset",
        "note": "A2-gated: corrected A1 plus anchor-delta-aware bounded main offset.",
        "override": {
            "model": {
                "backbone_type": "bilstm",
                "model_variant": "structured_fusion",
                "alt_gate_enabled": False,
                "alt_main_mode": "anchor_relative",
                "alt_anchor_reference_mode": "local_linear",
                **common_main,
                **residual_gate,
            },
            "training": {"risk_aware": {"use_segment_teacher": False, "use_alt_baseline_residual": False}},
            "loss": {
                **zero_alt_losses,
                "lambda_vertical_smooth": 0.02,
            },
        },
    },
    {
        "name": "a3_gated_routed",
        "note": "A3-gated: A2-gated plus DMS residual routed by gap threshold.",
        "override": {
            "model": {
                "backbone_type": "bilstm",
                "model_variant": "bilstm_alt_dms_refiner_v2_1",
                "alt_gate_enabled": False,
                "alt_main_mode": "anchor_relative",
                "alt_anchor_reference_mode": "local_linear",
                **common_main,
                **residual_gate,
                "alt_dms_route_mode": "gap_threshold",
                "alt_dms_route_gap_threshold_min": 9.0,
                "alt_dms_route_low_risk_scale": 0.0,
                "alt_dms_route_high_risk_scale": 1.0,
            },
            "training": {"risk_aware": {"use_segment_teacher": False, "use_alt_baseline_residual": True}},
            "loss": {
                "lambda_vertical_smooth": 0.02,
                "lambda_alt_gate_supervision": 0.0,
                "lambda_alt_gate_risk_shrink": 0.0,
                "lambda_alt_edge_first_diff": 0.05,
                "lambda_alt_edge_second_diff": 0.03,
                "lambda_alt_segment_bound": 0.04,
                "lambda_alt_vertical_rate_penalty": 0.02,
                "lambda_alt_boundary_anchor": 0.05,
            },
        },
    },
]

for exp in experiments:
    cfg = copy.deepcopy(template)
    cfg["base_config"] = str(base_config)
    deep_update(cfg, {"loss": common_loss})
    deep_update(cfg, exp["override"])
    cfg.setdefault("training", {})
    cfg["training"]["device"] = device
    cfg["training"]["checkpoint_monitor_metric"] = "gap_alt_rmse"
    cfg["training"]["save_every_epoch"] = True
    cfg["training"]["save_epoch_interval"] = 1
    cfg.setdefault("outputs", {})
    cfg["outputs"]["run_dir"] = f"outputs/experiments/obs_conditioned_gaponly/{run_tag}/{exp['name']}"
    cfg["outputs"]["checkpoint_name"] = "best.pt"
    cfg["experiment_note"] = exp["note"]
    cfg["model"].pop("main_rmax_ft", None)
    out = config_dir / f"{exp['name']}.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(out)
PY

IFS=',' read -r -a SELECTED <<< "${ONLY}"

{
  echo "===== height ablation gated retrain ====="
  echo "time: $(date -Is)"
  echo "run_tag: ${RUN_TAG}"
  echo "device: ${DEVICE}"
  echo "selected: ${ONLY}"
  echo "config_dir: ${CONFIG_DIR}"
  echo
} | tee -a "${LOG_FILE}" > "${SUMMARY_FILE}"

if [[ "${GENERATE_ONLY}" == "1" ]]; then
  echo "===== generate only; training not started =====" | tee -a "${LOG_FILE}" | tee -a "${SUMMARY_FILE}"
  exit 0
fi

for name in "${SELECTED[@]}"; do
  cfg="${CONFIG_DIR}/${name}.yaml"
  run_dir="${EXP_ROOT}/${name}"
  ckpt="${run_dir}/best.pt"
  if [[ ! -f "${cfg}" ]]; then
    echo "[error] missing config: ${cfg}" | tee -a "${LOG_FILE}" | tee -a "${SUMMARY_FILE}"
    exit 1
  fi
  echo "===== train ${name} $(date -Is) =====" | tee -a "${LOG_FILE}" | tee -a "${SUMMARY_FILE}"
  python scripts/train.py --config "${cfg}" 2>&1 | tee -a "${LOG_FILE}"
  echo "===== evaluate ${name} $(date -Is) =====" | tee -a "${LOG_FILE}" | tee -a "${SUMMARY_FILE}"
  python scripts/evaluate.py --config "${cfg}" --checkpoint "${ckpt}" --split test --plot-count 0 2>&1 | tee -a "${LOG_FILE}"
  echo "${name}: ${cfg} -> ${run_dir}" | tee -a "${SUMMARY_FILE}"
done

echo "===== done $(date -Is) =====" | tee -a "${LOG_FILE}" | tee -a "${SUMMARY_FILE}"
