#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/jj/workspace/data-0313"
RUN_TAG="${RUN_TAG:-obscons_gaponly_bilstm_vs_backbone_v1}"
BASE_DIR="${ROOT_DIR}/outputs/experiments/obs_conditioned_gaponly/${RUN_TAG}"
CONFIG_DIR="${BASE_DIR}/configs"
STAGEWISE_DIR="${BASE_DIR}/stagewise_eval"
STAGEWISE_CONFIG_DIR="${STAGEWISE_DIR}/configs"
SPLIT="${SPLIT:-test}"
PLOT_COUNT="${PLOT_COUNT:-0}"
DEVICE="${DEVICE:-}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate adsc_01
export PYTHONNOUSERSITE=1
cd "${ROOT_DIR}"

mkdir -p "${STAGEWISE_CONFIG_DIR}"

python - <<'PY' "${CONFIG_DIR}" "${STAGEWISE_CONFIG_DIR}" "${STAGEWISE_DIR}" "${DEVICE}"
from __future__ import annotations

import sys
from pathlib import Path

import yaml

config_dir = Path(sys.argv[1])
out_config_dir = Path(sys.argv[2])
stagewise_dir = Path(sys.argv[3])
device_override = str(sys.argv[4]).strip()

stage_paths = {
    "stage1": "outputs/mvp_merged_250_20260514_clean/stage1_clean/samples.parquet",
    "stage2": "outputs/mvp_merged_250_20260514_clean/stage2_clean/samples.parquet",
    "stage3": "outputs/mvp_merged_250_20260514_clean/stage3_clean/samples.parquet",
}

for cfg_path in sorted(config_dir.glob("*.yaml")):
    model_name = cfg_path.stem
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    for stage, samples_path in stage_paths.items():
        stage_cfg = dict(cfg)
        stage_cfg["data"] = dict(cfg["data"])
        stage_cfg["outputs"] = dict(cfg["outputs"])
        if device_override:
            stage_cfg["training"] = dict(cfg.get("training", {}))
            stage_cfg["training"]["device"] = device_override
        stage_cfg["data"]["samples_path"] = samples_path
        stage_cfg["outputs"]["run_dir"] = str(stagewise_dir / f"{model_name}_{stage}")
        out_path = out_config_dir / f"{model_name}_{stage}.yaml"
        out_path.write_text(yaml.safe_dump(stage_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(out_path)
PY

if [[ -n "${ONLY:-}" ]]; then
  IFS=',' read -r -a MODELS <<< "${ONLY}"
else
  mapfile -t MODELS < <(find "${CONFIG_DIR}" -maxdepth 1 -type f -name '*.yaml' -printf '%f\n' | sed 's/\.yaml$//' | sort)
fi

for model in "${MODELS[@]}"; do
  ckpt="${BASE_DIR}/${model}/best.pt"
  if [[ ! -f "${ckpt}" ]]; then
    echo "[skip] missing checkpoint: ${ckpt}" >&2
    continue
  fi
  for stage in stage1 stage2 stage3; do
    cfg="${STAGEWISE_CONFIG_DIR}/${model}_${stage}.yaml"
    eval_dir="${STAGEWISE_DIR}/${model}_${stage}"
    mkdir -p "${eval_dir}"
    if [[ -f "${BASE_DIR}/${model}/feature_standardizer.json" ]]; then
      cp "${BASE_DIR}/${model}/feature_standardizer.json" "${eval_dir}/feature_standardizer.json"
    fi
    echo "[eval] model=${model} stage=${stage}"
    python scripts/evaluate.py --config "${cfg}" --checkpoint "${ckpt}" --split "${SPLIT}" --plot-count "${PLOT_COUNT}"
  done
done

python - <<'PY' "${STAGEWISE_DIR}"
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

stagewise_dir = Path(sys.argv[1])
rows = []
for metrics_path in sorted(stagewise_dir.glob("*/main_task_metrics_test.json")):
    name = metrics_path.parent.name
    if "_" not in name:
        continue
    model, stage = name.rsplit("_", 1)
    if stage not in {"stage1", "stage2", "stage3"}:
        continue
    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    long_gap = m.get("subsets", {}).get("long_gap", {})
    few_anchor = m.get("subsets", {}).get("few_anchor", {})
    rows.append(
        {
            "model": model,
            "stage": stage,
            "count": m.get("subsets", {}).get("overall", {}).get("count"),
            "gap_lat_rmse": m.get("gap_lat_rmse"),
            "gap_lon_rmse": m.get("gap_lon_rmse"),
            "gap_alt_rmse_m": m.get("gap_alt_rmse"),
            "gap_alt_mae_m": m.get("gap_alt_mae"),
            "gap_haversine_m": m.get("gap_haversine_m"),
            "lat_rmse": m.get("lat_rmse"),
            "lon_rmse": m.get("lon_rmse"),
            "alt_rmse_m": m.get("alt_rmse"),
            "long_gap_count": long_gap.get("count"),
            "long_gap_alt_rmse_m": long_gap.get("gap_alt_rmse"),
            "few_anchor_count": few_anchor.get("count"),
            "few_anchor_alt_rmse_m": few_anchor.get("gap_alt_rmse"),
        }
    )

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("[summary] no stagewise metrics found")
df = df.sort_values(["stage", "model"])
out = stagewise_dir / "stagewise_all_models_main_metrics.csv"
df.to_csv(out, index=False)

baseline = "bilstm_clean_absolute"
delta_rows = []
for stage, g in df.groupby("stage"):
    b = g[g["model"] == baseline]
    if b.empty:
        continue
    b = b.iloc[0]
    for _, r in g.iterrows():
        row = {"stage": stage, "model": r["model"], "baseline": baseline}
        for metric in ["gap_lat_rmse", "gap_lon_rmse", "gap_alt_rmse_m", "gap_haversine_m"]:
            bv = float(b[metric])
            rv = float(r[metric])
            row[f"{metric}_baseline"] = bv
            row[f"{metric}_model"] = rv
            row[f"{metric}_delta"] = rv - bv
            row[f"{metric}_rel_improve_pct"] = (bv - rv) / bv * 100.0 if bv and not math.isnan(bv) else float("nan")
        delta_rows.append(row)
delta = pd.DataFrame(delta_rows)
delta_out = stagewise_dir / "stagewise_all_models_vs_bilstm_delta.csv"
delta.to_csv(delta_out, index=False)
print(f"[summary] {out}")
print(df.to_string(index=False))
print(f"[summary] {delta_out}")
print(delta.to_string(index=False))
PY
