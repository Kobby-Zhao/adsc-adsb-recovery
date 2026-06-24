#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/jj/workspace/data-0313"
RUN_TAG="${RUN_TAG:-clean_baselines_gaponly_minimal_v1}"
BASE_DIR="${ROOT_DIR}/outputs/experiments/obs_conditioned_gaponly/${RUN_TAG}"
CONFIG_DIR="${BASE_DIR}/configs"
SUMMARY_FILE="${BASE_DIR}/run_summary.txt"
PLOT_COUNT="${PLOT_COUNT:-0}"

set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate adsc_01
set -u
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

template_path = Path("configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_proposed_24e.yaml")
template = yaml.safe_load(template_path.read_text(encoding="utf-8"))


def deep_update(dst: dict, src: dict) -> dict:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def clean_common(name: str, backbone_type: str, desc: str) -> dict:
    cfg = copy.deepcopy(template)
    cfg["base_config"] = str(Path("configs/alt_focus/train_v1_control_e5_20260321.yaml").resolve())
    deep_update(
        cfg,
        {
            "model": {
                "backbone_type": backbone_type,
                "model_variant": "default",
                "minimal_task_adapt_baseline": True,
                "alt_gate_enabled": False,
                "alt_gate_mode": "learned",
                "alt_main_mode": "absolute",
                "vertical_projector_enabled": False,
                "alt_bias_enabled": False,
            },
            "training": {
                "epochs": 5,
                "checkpoint_monitor_metric": "gap_alt_rmse",
                "save_every_epoch": True,
                "save_epoch_interval": 1,
                "risk_aware": {
                    "use_segment_teacher": False,
                    "use_alt_baseline_residual": False,
                },
                "curriculum": {
                    "enabled": True,
                    "val_stage": "stage3",
                    "train_samples_per_epoch": 800,
                    "stage_paths": {
                        "stage1": "outputs/mvp_merged_250_20260514_clean/stage1_clean/samples.parquet",
                        "stage2": "outputs/mvp_merged_250_20260514_clean/stage2_clean/samples.parquet",
                        "stage3": "outputs/mvp_merged_250_20260514_clean/stage3_clean/samples.parquet",
                    },
                    "schedule": [
                        {"end_epoch": 2, "weights": {"stage1": 0.7, "stage2": 0.25, "stage3": 0.05}},
                        {"end_epoch": 3, "weights": {"stage1": 0.2, "stage2": 0.6, "stage3": 0.2}},
                        {"end_epoch": 4, "weights": {"stage1": 0.15, "stage2": 0.35, "stage3": 0.6}},
                        {"end_epoch": 5, "weights": {"stage1": 0.05, "stage2": 0.2, "stage3": 0.75}},
                    ],
                },
            },
            "loss": {
                "anchor_weight": 0.0,
                "gap_weight": 1.0,
                "lambda_unc": 0.0,
                "fusion_reg_lambda": 0.0,
                "lambda_vertical_smooth": 0.0,
                "lambda_multi_scale": 0.0,
                "lambda_cruise_phys": 0.0,
                "lambda_alt_gate_supervision": 0.0,
                "lambda_alt_gate_risk_shrink": 0.0,
                "lambda_alt_edge_first_diff": 0.0,
                "lambda_alt_edge_second_diff": 0.0,
                "lambda_alt_segment_bound": 0.0,
                "lambda_alt_vertical_rate_penalty": 0.0,
                "lambda_alt_boundary_anchor": 0.0,
                "alpha_vertical": 100.0,
                "lambda_smooth": 0.0,
                "gap_alt_weight": 2.0,
            },
            "outputs": {
                "run_dir": f"outputs/experiments/obs_conditioned_gaponly/{run_tag}/{name}",
                "checkpoint_name": "best.pt",
            },
            "experiment_note": desc,
        },
    )
    cfg["model"].pop("main_rmax_ft", None)
    return cfg


experiments = {
    "bilstm_minimal": clean_common(
        "bilstm_minimal",
        "bilstm",
        "Minimal-task-adaptation BiLSTM baseline under hard-anchor gap recovery protocol.",
    ),
    "unilstm_minimal": clean_common(
        "unilstm_minimal",
        "unilstm",
        "Minimal-task-adaptation uni-LSTM baseline under hard-anchor gap recovery protocol.",
    ),
    "cnnlstm_minimal": clean_common(
        "cnnlstm_minimal",
        "cnnlstm",
        "Minimal-task-adaptation CNN+LSTM baseline under hard-anchor gap recovery protocol.",
    ),
}

for name, cfg in experiments.items():
    path = config_dir / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(path)
PY

python - <<'PY' "${CONFIG_DIR}"
from __future__ import annotations

import sys
from pathlib import Path

import yaml

config_dir = Path(sys.argv[1])
required = {
    "loss.anchor_weight": 0.0,
    "loss.gap_weight": 1.0,
    "training.risk_aware.use_segment_teacher": False,
    "training.risk_aware.use_alt_baseline_residual": False,
    "model.alt_gate_enabled": False,
    "model.alt_main_mode": "absolute",
    "model.minimal_task_adapt_baseline": True,
    "training.epochs": 5,
}


def get_nested(d: dict, dotted: str):
    cur = d
    for part in dotted.split("."):
        cur = cur[part]
    return cur


bad = []
for path in sorted(config_dir.glob("*.yaml")):
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key, expected in required.items():
        actual = get_nested(cfg, key)
        if actual != expected:
            bad.append(f"{path.name}: {key}={actual!r}, expected {expected!r}")
    note = str(cfg.get("experiment_note", ""))
    if "hard-anchor" not in note and "hard-anchor" not in str(cfg.get("base_config", "")):
        bad.append(f"{path.name}: experiment_note does not document hard-anchor protocol")

if bad:
    print("[audit] FAILED")
    for item in bad:
        print(item)
    raise SystemExit(2)

print("[audit] clean baseline configs passed")
for path in sorted(config_dir.glob("*.yaml")):
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    print(
        f"{path.name}: backbone={cfg['model']['backbone_type']} "
        f"run_dir={cfg['outputs']['run_dir']}"
    )
PY

if [[ -n "${ONLY:-}" ]]; then
  IFS=',' read -r -a MODELS <<< "${ONLY}"
else
  MODELS=(
    bilstm_minimal
    unilstm_minimal
    cnnlstm_minimal
  )
fi

{
  echo "RUN_TAG=${RUN_TAG}"
  echo "CONFIG_DIR=${CONFIG_DIR}"
  echo "ONLY=${ONLY:-<default neural clean baselines>}"
  echo "START=$(date -Is)"
} > "${SUMMARY_FILE}"

for model in "${MODELS[@]}"; do
  cfg="${CONFIG_DIR}/${model}.yaml"
  if [[ ! -f "${cfg}" ]]; then
    echo "[error] missing config: ${cfg}" >&2
    exit 2
  fi
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
  python scripts/evaluate.py --config "${cfg}" --checkpoint "${ckpt}" --split test --plot-count "${PLOT_COUNT}"
  echo "${cfg} -> ${run_dir}" >> "${SUMMARY_FILE}"
done

echo "END=$(date -Is)" >> "${SUMMARY_FILE}"
echo "[done] ${SUMMARY_FILE}"
