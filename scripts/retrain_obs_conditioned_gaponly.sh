#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/jj/workspace/data-0313"
RUN_TAG="${RUN_TAG:-obscons_gaponly_$(date +%Y%m%d_%H%M%S)}"
CONFIG_DIR="${ROOT_DIR}/outputs/experiments/obs_conditioned_gaponly/${RUN_TAG}/configs"
SUMMARY_FILE="${ROOT_DIR}/outputs/experiments/obs_conditioned_gaponly/${RUN_TAG}/run_summary.txt"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate adsc_01
export PYTHONNOUSERSITE=1
cd "${ROOT_DIR}"

mkdir -p "${CONFIG_DIR}"

python - <<'PY' "${CONFIG_DIR}" "${RUN_TAG}"
from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml

config_dir = Path(sys.argv[1])
run_tag = sys.argv[2]
device = str(__import__("os").environ.get("DEVICE", "")).strip()

template_path = Path("configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_proposed_24e.yaml")
template = yaml.safe_load(template_path.read_text(encoding="utf-8"))


def deep_update(dst: dict, src: dict) -> dict:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def strip_keys(d: dict, keys: list[str]) -> None:
    for key in keys:
        d.pop(key, None)


common_loss = {
    "anchor_weight": 0.0,
    "gap_weight": 1.0,
    "lambda_unc": 0.0,
    "fusion_reg_lambda": 0.0,
}

common_model_m_units = {
    "main_rmax_min_m": 91.44,
    "main_rmax_slope_m_per_min": 4.572,
    "main_rmax_max_m": 365.76,
}

experiments = [
    {
        "name": "bilstm_clean_absolute",
        "desc": "Clean BiLSTM baseline: shared anchor-conditioned input + hard anchor consistency, no altitude baseline branch.",
        "override": {
            "model": {
                "backbone_type": "bilstm",
                "model_variant": "default",
                "alt_gate_enabled": False,
                "alt_main_mode": "absolute",
                "main_rmax_m": 0.0,
            },
            "training": {"risk_aware": {"use_segment_teacher": False, "use_alt_baseline_residual": False}},
            "loss": {
                "lambda_vertical_smooth": 0.0,
                "lambda_alt_gate_supervision": 0.0,
                "lambda_alt_gate_risk_shrink": 0.0,
                "lambda_alt_edge_first_diff": 0.0,
                "lambda_alt_edge_second_diff": 0.0,
                "lambda_alt_segment_bound": 0.0,
                "lambda_alt_vertical_rate_penalty": 0.0,
                "lambda_alt_boundary_anchor": 0.0,
            },
        },
    },
    {
        "name": "ours_backbone_absolute",
        "desc": "本文双向预测主干 only: position-aware fusion, no altitude baseline/residual branch.",
        "override": {
            "model": {
                "backbone_type": "bilstm",
                "model_variant": "structured_fusion",
                "alt_gate_enabled": False,
                "alt_main_mode": "absolute",
                "main_rmax_m": 0.0,
            },
            "training": {"risk_aware": {"use_segment_teacher": False, "use_alt_baseline_residual": False}},
            "loss": {
                "lambda_vertical_smooth": 0.0,
                "lambda_alt_gate_supervision": 0.0,
                "lambda_alt_gate_risk_shrink": 0.0,
                "lambda_alt_edge_first_diff": 0.0,
                "lambda_alt_edge_second_diff": 0.0,
                "lambda_alt_segment_bound": 0.0,
                "lambda_alt_vertical_rate_penalty": 0.0,
                "lambda_alt_boundary_anchor": 0.0,
            },
        },
    },
    {
        "name": "a1_linear_alt_baseline",
        "desc": "A1: shared position-aware backbone + anchor-to-anchor linear altitude baseline.",
        "override": {
            "model": {
                "backbone_type": "bilstm",
                "model_variant": "structured_fusion",
                "alt_gate_enabled": False,
                "alt_main_mode": "anchor_relative",
                "main_rmax_m": 0.0,
            },
            "training": {"risk_aware": {"use_segment_teacher": False, "use_alt_baseline_residual": False}},
            "loss": {
                "lambda_vertical_smooth": 0.0,
                "lambda_alt_gate_supervision": 0.0,
                "lambda_alt_gate_risk_shrink": 0.0,
                "lambda_alt_edge_first_diff": 0.0,
                "lambda_alt_edge_second_diff": 0.0,
                "lambda_alt_segment_bound": 0.0,
                "lambda_alt_vertical_rate_penalty": 0.0,
                "lambda_alt_boundary_anchor": 0.0,
            },
        },
    },
    {
        "name": "a2_linear_plus_offset",
        "desc": "A2: A1 + gap-aware bounded altitude offset in meters.",
        "override": {
            "model": {
                "backbone_type": "bilstm",
                "model_variant": "structured_fusion",
                "alt_gate_enabled": False,
                "alt_main_mode": "anchor_relative",
                "main_rmax_m": 152.4,
                **common_model_m_units,
            },
            "training": {"risk_aware": {"use_segment_teacher": False, "use_alt_baseline_residual": False}},
            "loss": {
                "lambda_vertical_smooth": 0.02,
                "lambda_alt_gate_supervision": 0.0,
                "lambda_alt_gate_risk_shrink": 0.0,
                "lambda_alt_edge_first_diff": 0.0,
                "lambda_alt_edge_second_diff": 0.0,
                "lambda_alt_segment_bound": 0.0,
                "lambda_alt_vertical_rate_penalty": 0.0,
                "lambda_alt_boundary_anchor": 0.0,
            },
        },
    },
    {
        "name": "a3_risk_routed",
        "desc": "A3-routed: A2 + DMS residual enabled only on observable long-gap positions.",
        "override": {
            "model": {
                "backbone_type": "bilstm",
                "model_variant": "bilstm_alt_dms_refiner_v2_1",
                "alt_gate_enabled": False,
                "alt_main_mode": "anchor_relative",
                "main_rmax_m": 152.4,
                **common_model_m_units,
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
    {
        "name": "proposed_learned_gate",
        "desc": "Proposed: A2 + DMS residual + learned gate, under the corrected observation-conditioned objective.",
        "override": {
            "model": {
                "backbone_type": "bilstm",
                "model_variant": "bilstm_alt_dms_refiner_v2_1",
                "alt_gate_enabled": True,
                "alt_gate_mode": "learned",
                "alt_main_mode": "anchor_relative",
                "main_rmax_m": 152.4,
                **common_model_m_units,
                "alt_dms_route_mode": "none",
            },
            "training": {"risk_aware": {"use_segment_teacher": True, "use_alt_baseline_residual": True}},
            "loss": {
                "lambda_vertical_smooth": 0.02,
                "lambda_alt_gate_supervision": 0.1,
                "lambda_alt_gate_risk_shrink": 0.08,
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
    cfg["base_config"] = str(Path("configs/alt_focus/train_v1_control_e5_20260321.yaml").resolve())
    deep_update(cfg, {"loss": common_loss})
    deep_update(cfg, exp["override"])
    cfg.setdefault("outputs", {})
    cfg["outputs"]["run_dir"] = f"outputs/experiments/obs_conditioned_gaponly/{run_tag}/{exp['name']}"
    cfg["outputs"]["checkpoint_name"] = "best.pt"
    cfg.setdefault("training", {})
    if device:
        cfg["training"]["device"] = device
    cfg["training"]["checkpoint_monitor_metric"] = "gap_alt_rmse"
    cfg["training"]["save_every_epoch"] = True
    cfg["training"]["save_epoch_interval"] = 1
    cfg["experiment_note"] = exp["desc"]
    strip_keys(cfg["model"], ["main_rmax_ft"])

    out_path = config_dir / f"{exp['name']}.yaml"
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(out_path)
PY

mapfile -t CONFIGS < <(find "${CONFIG_DIR}" -maxdepth 1 -type f -name '*.yaml' | sort)
if [[ -n "${ONLY:-}" ]]; then
  IFS=',' read -r -a ONLY_NAMES <<< "${ONLY}"
  FILTERED_CONFIGS=()
  for cfg in "${CONFIGS[@]}"; do
    base="$(basename "${cfg}" .yaml)"
    for name in "${ONLY_NAMES[@]}"; do
      if [[ "${base}" == "${name}" ]]; then
        FILTERED_CONFIGS+=("${cfg}")
      fi
    done
  done
  CONFIGS=("${FILTERED_CONFIGS[@]}")
fi

if [[ "${#CONFIGS[@]}" -eq 0 ]]; then
  echo "No configs selected. Check ONLY=${ONLY:-<unset>}" >&2
  exit 2
fi

{
  echo "RUN_TAG=${RUN_TAG}"
  echo "CONFIG_DIR=${CONFIG_DIR}"
  echo "ONLY=${ONLY:-<all>}"
  echo "START=$(date -Is)"
  printf '%s\n' "${CONFIGS[@]}"
} > "${SUMMARY_FILE}"

for cfg in "${CONFIGS[@]}"; do
  echo "[train] ${cfg}"
  python scripts/train.py --config "${cfg}"

  run_dir="$(python - <<'PY' "${cfg}"
import sys
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(cfg["outputs"]["run_dir"])
PY
)"

  ckpt="${run_dir}/best.pt"
  echo "[eval] ${cfg} ${ckpt}"
  python scripts/evaluate.py --config "${cfg}" --checkpoint "${ckpt}" --split test --plot-count 0
  echo "${cfg} -> ${run_dir}" >> "${SUMMARY_FILE}"
done

echo "END=$(date -Is)" >> "${SUMMARY_FILE}"
echo "[done] ${SUMMARY_FILE}"
