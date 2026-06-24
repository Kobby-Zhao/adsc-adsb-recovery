#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BASE_CFG="outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_gapaware_small.yaml"
ABLATION_CFG_DIR="outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/ablation_curriculum"

MODE="${1:-all}"

generate_configs() {
  python - <<'PY'
from pathlib import Path
import copy
import yaml

base_path = Path("outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_gapaware_small.yaml")
out_dir = Path("outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/ablation_curriculum")
out_dir.mkdir(parents=True, exist_ok=True)

with base_path.open("r", encoding="utf-8") as f:
    base = yaml.safe_load(f)

def dump_cfg(name, run_dir_name, schedule):
    cfg = copy.deepcopy(base)
    cfg["training"]["epochs"] = 24
    cfg["training"]["curriculum"]["enabled"] = True
    cfg["training"]["curriculum"]["schedule"] = schedule
    cfg["outputs"]["run_dir"] = f"outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/{run_dir_name}"
    cfg["experiment_note"] = name
    out_path = out_dir / f"{run_dir_name}.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    print(out_path)

dump_cfg(
    name="G0 progressive curriculum",
    run_dir_name="ablation_G0_progressive_gapaware_small_24ep",
    schedule=[
        {"end_epoch": 4,  "weights": {"stage1": 0.60, "stage2": 0.30, "stage3": 0.10}},
        {"end_epoch": 10, "weights": {"stage1": 0.30, "stage2": 0.45, "stage3": 0.25}},
        {"end_epoch": 18, "weights": {"stage1": 0.15, "stage2": 0.35, "stage3": 0.50}},
        {"end_epoch": 24, "weights": {"stage1": 0.05, "stage2": 0.25, "stage3": 0.70}},
    ],
)

dump_cfg(
    name="G1 uniform mixed-stage sampling",
    run_dir_name="ablation_G1_uniformmix_gapaware_small_24ep",
    schedule=[
        {"end_epoch": 24, "weights": {"stage1": 1/3, "stage2": 1/3, "stage3": 1/3}},
    ],
)

dump_cfg(
    name="G2 S3-only training",
    run_dir_name="ablation_G2_s3only_gapaware_small_24ep",
    schedule=[
        {"end_epoch": 24, "weights": {"stage1": 0.0, "stage2": 0.0, "stage3": 1.0}},
    ],
)

dump_cfg(
    name="G3 reverse curriculum",
    run_dir_name="ablation_G3_reverse_gapaware_small_24ep",
    schedule=[
        {"end_epoch": 4,  "weights": {"stage1": 0.05, "stage2": 0.25, "stage3": 0.70}},
        {"end_epoch": 10, "weights": {"stage1": 0.15, "stage2": 0.35, "stage3": 0.50}},
        {"end_epoch": 18, "weights": {"stage1": 0.30, "stage2": 0.45, "stage3": 0.25}},
        {"end_epoch": 24, "weights": {"stage1": 0.60, "stage2": 0.30, "stage3": 0.10}},
    ],
)
PY
}

train_one() {
  local name="$1"
  local cfg="$ABLATION_CFG_DIR/${name}.yaml"
  python scripts/train.py --config "$cfg"
}

eval_one() {
  local name="$1"
  local cfg="$ABLATION_CFG_DIR/${name}.yaml"
  local ckpt="outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/${name}/best.pt"
  python scripts/evaluate.py --config "$cfg" --checkpoint "$ckpt" --split test --plot-count 0
}

train_all() {
  train_one "ablation_G0_progressive_gapaware_small_24ep"
  train_one "ablation_G1_uniformmix_gapaware_small_24ep"
  train_one "ablation_G2_s3only_gapaware_small_24ep"
  train_one "ablation_G3_reverse_gapaware_small_24ep"
}

eval_all() {
  eval_one "ablation_G0_progressive_gapaware_small_24ep"
  eval_one "ablation_G1_uniformmix_gapaware_small_24ep"
  eval_one "ablation_G2_s3only_gapaware_small_24ep"
  eval_one "ablation_G3_reverse_gapaware_small_24ep"
}

case "$MODE" in
  generate)
    generate_configs
    ;;
  train)
    train_all
    ;;
  eval)
    eval_all
    ;;
  all)
    generate_configs
    train_all
    eval_all
    ;;
  *)
    echo "Usage: $0 [generate|train|eval|all]"
    exit 1
    ;;
esac
