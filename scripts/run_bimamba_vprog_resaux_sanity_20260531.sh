#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/jj/workspace/data-0313"
RUN_TAG="${RUN_TAG:-bimamba_vprog_resaux_5ep_v1}"
BASE_DIR="${ROOT_DIR}/outputs/experiments/obs_conditioned_gaponly/${RUN_TAG}"
CONFIG_DIR="${BASE_DIR}/configs"
SUMMARY_FILE="${BASE_DIR}/run_summary.txt"
PLOT_COUNT="${PLOT_COUNT:-0}"
EPOCHS="${EPOCHS:-5}"

set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate adsc_01
set -u
export PYTHONNOUSERSITE=1
cd "${ROOT_DIR}"

mkdir -p "${CONFIG_DIR}"

python - <<'PY' "${CONFIG_DIR}" "${RUN_TAG}" "${EPOCHS}"
from __future__ import annotations

import copy
import sys
from pathlib import Path
import yaml

config_dir = Path(sys.argv[1])
run_tag = sys.argv[2]
epochs = int(sys.argv[3])

template_path = Path("outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_bimamba.yaml")
template = yaml.safe_load(template_path.read_text(encoding="utf-8"))


def deep_update(dst: dict, src: dict) -> dict:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def build_cfg(name: str, backbone_type: str, note: str, lambda_vprog_res: float) -> dict:
    cfg = copy.deepcopy(template)
    deep_update(
        cfg,
        {
            "model": {
                "backbone_type": backbone_type,
                "use_z_adapter": True,
                "z_adapter_ratio": 0.25,
                "z_adapter_gamma_init": 0.0,
            },
            "loss": {
                "lambda_vprog_res": lambda_vprog_res,
                "vprog_res_enable_abs_dz_min": 300.0,
            },
            "training": {
                "epochs": epochs,
            },
            "outputs": {
                "run_dir": f"outputs/experiments/obs_conditioned_gaponly/{run_tag}/{name}",
                "checkpoint_name": "best.pt",
            },
            "experiment_note": note,
        },
    )
    return cfg


cfgs = {
    "B_bimamba_context_xyaux_zlinear_zadapter_gapaware_small": build_cfg(
        "B_bimamba_context_xyaux_zlinear_zadapter_gapaware_small",
        "bimamba_context_xyaux_zlinear_zadapter_gapaware_small",
        "BiMamba xyaux + zlinear + gap-aware small z adapter, 5-epoch sanity.",
        0.0,
    ),
    "F_bimamba_context_xyaux_zlinear_zadapter_gapaware_small_vprog_resaux": build_cfg(
        "F_bimamba_context_xyaux_zlinear_zadapter_gapaware_small_vprog_resaux",
        "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_vprog_resaux",
        "BiMamba xyaux + zlinear + gap-aware small z adapter + vertical progress residual auxiliary supervision, 5-epoch sanity.",
        0.01,
    ),
}

for name, cfg in cfgs.items():
    path = config_dir / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(path)
PY

MODELS=(B_bimamba_context_xyaux_zlinear_zadapter_gapaware_small F_bimamba_context_xyaux_zlinear_zadapter_gapaware_small_vprog_resaux)
if [[ -n "${ONLY:-}" ]]; then
  IFS=',' read -r -a MODELS <<< "${ONLY}"
fi

{
  echo "RUN_TAG=${RUN_TAG}"
  echo "EPOCHS=${EPOCHS}"
  echo "CONFIG_DIR=${CONFIG_DIR}"
  echo "ONLY=${ONLY:-<default B_bimamba_context_xyaux_zlinear_zadapter_gapaware_small,F_bimamba_context_xyaux_zlinear_zadapter_gapaware_small_vprog_resaux>}"
  echo "START=$(date -Is)"
} > "${SUMMARY_FILE}"

for model in "${MODELS[@]}"; do
  cfg="${CONFIG_DIR}/${model}.yaml"
  echo "[train] ${model}" | tee -a "${SUMMARY_FILE}"
  python scripts/train.py --config "${cfg}" 2>&1 | tee "${BASE_DIR}/${model}.train.log"
  ckpt="$(python - <<'PY' "${cfg}"
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], 'r', encoding='utf-8'))
print(f"{cfg['outputs']['run_dir']}/{cfg['outputs']['checkpoint_name']}")
PY
)"
  echo "[eval-test] ${model}" | tee -a "${SUMMARY_FILE}"
  python scripts/evaluate.py --config "${cfg}" --checkpoint "${ckpt}" --split test --plot-count "${PLOT_COUNT}"
  echo "[eval-val] ${model}" | tee -a "${SUMMARY_FILE}"
  python scripts/evaluate.py --config "${cfg}" --checkpoint "${ckpt}" --split val --plot-count 0
done

echo "END=$(date -Is)" >> "${SUMMARY_FILE}"
echo "[done] ${SUMMARY_FILE}"
