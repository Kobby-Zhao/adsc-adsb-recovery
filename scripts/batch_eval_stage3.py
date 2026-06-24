#!/usr/bin/env python3
"""
Batch re-evaluate all trained models on Stage3 test data.

Handles the code-version mismatch (old fusion MLP output dim=2) by
monkey-patching SimpleFusionHead before evaluate.py imports it.
"""

import json, os, shutil, sys, tempfile
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CONDA_PYTHON = "/home/jj/miniconda3/envs/adsc_01/bin/python"

STAGE3_DATA = "outputs/mvp_merged_nostage_20260415/stage_datasets_20260415_s2v2/stage3/samples.parquet"

MODELS = [
    ("OurMethod",       "configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_proposed_24e.yaml",
                        "outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_proposed_24e/best.pt"),
    ("UniLSTM",         "configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_unilstm_baseline_24e.yaml",
                        "outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_unilstm_baseline_24e/best.pt"),
    ("Kalman_Filter",   "configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_kalman_filter_baseline_24e.yaml",
                        "outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_kalman_filter_baseline_24e/best.pt"),
    ("BiLSTM",          "configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_bilstm_baseline_24e.yaml",
                        "outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_bilstm_baseline_24e/best.pt"),
    ("CNN_LSTM",        "configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_cnnlstm_baseline_24e.yaml",
                        "outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_cnnlstm_baseline_24e/best.pt"),
    ("Transformer",     "configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_transformer_baseline_24e.yaml",
                        "outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_transformer_baseline_24e/best.pt"),
]


def patch_config(config_path: Path, stage3_data: str, stage3_run_dir: str, out_path: Path) -> None:
    with open(config_path, "r") as f:
        text = f.read()
    import re
    m = re.search(r"base_config:\s*(\S+)", text)
    if m and not m.group(1).startswith("/"):
        base_abs = str((config_path.parent / m.group(1)).resolve())
        text = text.replace(m.group(0), f"base_config: {base_abs}")
    text = re.sub(r"(samples_path:\s*).*", rf"\1{stage3_data}", text)
    text = re.sub(r"(run_dir:\s*).*", rf"\1{stage3_run_dir}", text)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(text)


def main():
    out_dir = ROOT / "outputs/experiments/batch_eval_stage3"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="eval_stage3_"))

    # Use the old-fusion wrapper instead of raw evaluate.py
    eval_wrapper = ROOT / "scripts/eval_wrapper_oldfusion.py"

    results = {}
    for name, config_rel, ckpt_rel in MODELS:
        config_path = ROOT / config_rel
        ckpt_path = ROOT / ckpt_rel
        if not ckpt_path.exists():
            print(f"\n[SKIP] {name}: checkpoint not found")
            continue

        tmp_config = tmp_dir / f"{name}_stage3.yaml"
        stage3_run_dir = str(out_dir / name)
        patch_config(config_path, STAGE3_DATA, stage3_run_dir, tmp_config)

        print(f"\n{'='*60}")
        print(f"[EVAL] {name}")
        cmd = [
            CONDA_PYTHON, str(eval_wrapper),
            "--config", str(tmp_config),
            "--checkpoint", str(ckpt_path),
            "--split", "test",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=1200)

        if result.returncode != 0:
            print(f"  [FAIL] rc={result.returncode}")
            # Show last lines of stderr for diagnosis
            err_lines = result.stderr.strip().split('\n')
            print(f"  stderr: {' | '.join(err_lines[-3:])}")
            continue

        # Read metrics from JSON output
        json_path = out_dir / name / "main_task_metrics_test.json"
        if not json_path.exists():
            print(f"  [FAIL] no metrics JSON at {json_path}")
            print(f"  stdout tail: {result.stdout[-200:]}")
            continue

        with open(json_path) as f:
            stats = json.load(f)

        m = {
            "lat_rmse": stats.get("lat_rmse"),
            "lon_rmse": stats.get("lon_rmse"),
            "alt_rmse": stats.get("alt_rmse"),
            "gap_lat_rmse": stats.get("gap_lat_rmse"),
            "gap_lon_rmse": stats.get("gap_lon_rmse"),
            "gap_alt_rmse": stats.get("gap_alt_rmse"),
        }
        results[name] = m
        print(f"  gap_lat={m['gap_lat_rmse']:.4f}  gap_lon={m['gap_lon_rmse']:.4f}  gap_alt={m['gap_alt_rmse']:.4f}")

    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Summary table
    print(f"\n{'='*80}")
    print("STAGE3 TEST SET — ALL MODELS")
    print(f"{'='*80}")
    hdr = f"{'Model':<18} {'gap_lat_rmse':>13} {'gap_lon_rmse':>13} {'gap_alt_rmse':>13}"
    print(hdr)
    print("-" * 80)
    for name, m in results.items():
        print(f"{name:<18} {m['gap_lat_rmse']:>13.4f} {m['gap_lon_rmse']:>13.4f} {m['gap_alt_rmse']:>13.4f}")
    print("=" * 80)

    with open(out_dir / "batch_eval_stage3_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n[ok] → {out_dir}")


if __name__ == "__main__":
    main()
