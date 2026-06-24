from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.training.utils import load_config


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs/analysis/stagewise_eval_proto_compare_24e"

STAGE_PATHS = {
    "S1": "outputs/mvp_merged_250_20260514_clean/stage1_clean/samples.parquet",
    "S2": "outputs/mvp_merged_250_20260514_clean/stage2_medium_clean/samples.parquet",
    "S3": "outputs/mvp_merged_250_20260514_clean/stage3_clean/samples.parquet",
}


MODEL_SPECS = {
    "kalman_filter": {
        "label": "Kalman Filter",
        "config": "outputs/experiments/obs_conditioned_gaponly/clean_baselines_gaponly_v1/configs/kalman_filter_clean_absolute.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/clean_baselines_gaponly_v1/kalman_filter_clean_absolute/best.pt",
    },
    "lstm_proto": {
        "label": "LSTM (proto)",
        "config": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_unilstm_proto.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_unilstm_proto/best.pt",
    },
    "bilstm_proto": {
        "label": "BiLSTM (proto)",
        "config": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_bilstm_proto.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_bilstm_proto/best.pt",
    },
    "cnnlstm_proto": {
        "label": "CNN+LSTM (proto)",
        "config": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_cnnlstm_proto.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_cnnlstm_proto/best.pt",
    },
    "transformer_proto": {
        "label": "Transformer (proto)",
        "config": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_transformer_proto.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_transformer_proto/best.pt",
    },
    "bimamba": {
        "label": "BiMamba",
        "config": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_bimamba.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_bimamba/best.pt",
    },
    "mamba_proto": {
        "label": "Mamba (proto)",
        "config": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_mamba_proto.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_mamba_proto/best.pt",
    },
}

KALMAN_STAGEWISE_EXISTING = {
    "S1": "outputs/experiments/obs_conditioned_gaponly/clean_baselines_gaponly_v1/stagewise_eval/kalman_filter_clean_absolute_stage1",
    "S2": "outputs/experiments/obs_conditioned_gaponly/clean_baselines_gaponly_v1/stagewise_eval/kalman_filter_clean_absolute_stage2",
    "S3": "outputs/experiments/obs_conditioned_gaponly/clean_baselines_gaponly_v1/stagewise_eval/kalman_filter_clean_absolute_stage3",
}


def _abs(p: str) -> Path:
    return (ROOT / p).resolve()


def _build_eval_config(model_key: str, stage_key: str, spec: dict, out_dir: Path, device_override: str | None) -> Path:
    cfg = load_config(str(_abs(spec["config"])))
    cfg = copy.deepcopy(cfg)
    original_run_dir = cfg["outputs"]["run_dir"]
    cfg["data"]["samples_path"] = STAGE_PATHS[stage_key]
    cfg.setdefault("training", {})
    if device_override:
        cfg["training"]["device"] = device_override
    cfg["outputs"]["run_dir"] = str(out_dir)
    cfg["outputs"]["checkpoint_name"] = "best.pt"
    cfg["experiment_note"] = f"Stagewise eval only: {model_key} on {stage_key}"
    cfg_dir = out_dir.parent / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"{model_key}_{stage_key}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return cfg_path, _abs(original_run_dir)


def _copy_eval_artifacts(src_run_dir: Path, dst_run_dir: Path) -> None:
    dst_run_dir.mkdir(parents=True, exist_ok=True)
    for fname in [
        "feature_standardizer.json",
        "target_model_scaler.json",
        "alt_base_residual_bounds.json",
        "segment_risk_rules.yaml",
    ]:
        src = src_run_dir / fname
        if src.exists():
            shutil.copy2(src, dst_run_dir / fname)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--models", nargs="*", default=list(MODEL_SPECS.keys()))
    ap.add_argument("--stages", nargs="*", default=["S1", "S2", "S3"])
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--device", default=None, help="Optional eval device override, e.g. cuda or cpu")
    ap.add_argument("--proposed-config", default=None, help="Optional 24-epoch proposed-model config")
    ap.add_argument("--proposed-checkpoint", default=None, help="Optional 24-epoch proposed-model checkpoint")
    args = ap.parse_args()

    specs = copy.deepcopy(MODEL_SPECS)
    if args.proposed_config and args.proposed_checkpoint:
        specs = {
            "proposed": {
                "label": "Proposed",
                "config": args.proposed_config,
                "checkpoint": args.proposed_checkpoint,
            },
            **specs,
        }

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    commands: list[list[str]] = []
    for model_key in args.models:
        if model_key not in specs:
            print(f"[skip] unknown model key: {model_key}")
            continue
        spec = specs[model_key]
        ckpt = _abs(spec["checkpoint"])
        cfg_src = _abs(spec["config"])
        if model_key != "kalman_filter" and not ckpt.exists():
            print(f"[skip] checkpoint missing for {model_key}: {ckpt}")
            continue
        if not cfg_src.exists():
            print(f"[skip] config missing for {model_key}: {cfg_src}")
            continue

        for stage_key in args.stages:
            if stage_key not in STAGE_PATHS:
                print(f"[skip] unknown stage: {stage_key}")
                continue
            eval_dir = out_root / f"{model_key}_{stage_key}"

            if model_key == "kalman_filter":
                src_dir = _abs(KALMAN_STAGEWISE_EXISTING[stage_key])
                if not src_dir.exists():
                    print(f"[skip] existing kalman stagewise result missing: {src_dir}")
                    continue
                eval_dir.mkdir(parents=True, exist_ok=True)
                for fname in [
                    "main_task_metrics_test_summary_dim.csv",
                    "main_task_metrics_test_summary_latlon.csv",
                    "main_task_metrics_test.json",
                    "main_task_metrics_test_per_sample.csv",
                ]:
                    src = src_dir / fname
                    if src.exists():
                        shutil.copy2(src, eval_dir / fname)
                print(f"[copy] kalman existing stagewise result -> {eval_dir}")
                continue

            cfg_path, original_run_dir = _build_eval_config(
                model_key,
                stage_key,
                spec,
                eval_dir,
                args.device,
            )
            _copy_eval_artifacts(original_run_dir, eval_dir)
            cmd = [
                args.python,
                "scripts/evaluate.py",
                "--config",
                str(cfg_path),
                "--checkpoint",
                str(ckpt),
                "--split",
                "test",
                "--plot-count",
                "0",
            ]
            commands.append(cmd)
            print(" ".join(cmd))

    if not args.run:
        print("\n[info] Commands printed only. Re-run with --run to execute.")
        return 0

    for cmd in commands:
        print(f"\n[run] {' '.join(cmd)}")
        subprocess.run(cmd, cwd=ROOT, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
