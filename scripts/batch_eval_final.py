#!/usr/bin/env python3
"""Final Stage3 evaluation for all 7 models after retraining."""
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONDA_PYTHON = "/home/jj/miniconda3/envs/adsc_01/bin/python"

MODELS = [
    # Non-bilstm baselines: use regular evaluate.py
    ("UniLSTM",       "configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_unilstm_baseline_24e.yaml",
                      "outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_unilstm_baseline_24e/best.pt", False),
    ("Kalman",        "configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_kalman_filter_baseline_24e.yaml",
                      "outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_kalman_filter_baseline_24e/best.pt", False),
    ("CNN+LSTM",      "configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_cnnlstm_baseline_24e.yaml",
                      "outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_cnnlstm_baseline_24e/best.pt", False),
    ("Transformer",   "configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_transformer_baseline_24e.yaml",
                      "outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_transformer_baseline_24e/best.pt", False),
    # Bilstm-based: newly trained, use regular evaluate.py
    ("BiLSTM",        "configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_bilstm_baseline_24e.yaml",
                      "outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_bilstm_baseline_24e/best.pt", False),
    ("OurMethod",     "configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_proposed_24e.yaml",
                      "outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_proposed_24e/best.pt", False),
]

results = {}
for name, config_rel, ckpt_rel, __use_wrapper in MODELS:
    ckpt_path = ROOT / ckpt_rel
    if not ckpt_path.exists():
        print(f"[SKIP] {name}: ckpt not found")
        continue

    eval_script = ROOT / "scripts/evaluate.py"
    cmd = [CONDA_PYTHON, str(eval_script), "--config", str(ROOT / config_rel),
           "--checkpoint", str(ckpt_path), "--split", "test"]

    print(f"\n[EVAL] {name} ...")
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=600)
    if r.returncode != 0:
        print(f"  [FAIL] {r.stderr[-200:]}")
        continue

    # Read metrics from the run_dir JSON
    run_dir = ROOT / Path(config_rel).parent.parent / Path(config_rel).stem
    # Actually the evaluate.py writes to cfg["outputs"]["run_dir"] which is in the config
    import yaml
    with open(str(ROOT / config_rel)) as f:
        cfg = yaml.safe_load(f.read())
    run_dir = ROOT / cfg["outputs"]["run_dir"]
    json_path = run_dir / "main_task_metrics_test.json"
    if not json_path.exists():
        print(f"  [FAIL] no JSON at {json_path}")
        continue

    with open(json_path) as f:
        stats = json.load(f)

    m = {
        "lat_rmse": stats.get("lat_rmse"), "lon_rmse": stats.get("lon_rmse"), "alt_rmse": stats.get("alt_rmse"),
        "gap_lat_rmse": stats.get("gap_lat_rmse"), "gap_lon_rmse": stats.get("gap_lon_rmse"), "gap_alt_rmse": stats.get("gap_alt_rmse"),
    }
    results[name] = m
    print(f"  gap_lat={m['gap_lat_rmse']:.4f}  gap_lon={m['gap_lon_rmse']:.4f}  gap_alt={m['gap_alt_rmse']:.4f}")

# Summary
print(f"\n{'='*80}")
print("STAGE3 FINAL RESULTS")
print(f"{'='*80}")
hdr = f"{'Model':<18} {'gap_lat_rmse':>13} {'gap_lon_rmse':>13} {'gap_alt_rmse':>13}"
print(hdr)
print("-"*80)
for name, m in results.items():
    print(f"{name:<18} {m['gap_lat_rmse']:>13.4f} {m['gap_lon_rmse']:>13.4f} {m['gap_alt_rmse']:>13.4f}")
print("="*80)

out_dir = ROOT / "outputs/experiments/batch_eval_final"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "final_results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"\n[ok] → {out_dir}")
