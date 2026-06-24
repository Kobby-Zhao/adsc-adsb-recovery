#!/usr/bin/env python3
"""
Risk behavior analysis from per-sample evaluation metrics.

Analyses the test-set metrics CSV (already containing segment_bucket,
anchor_pattern, gap_alt_rmse, etc.) to answer:

  Q1  Do risk-level assignments align with actual recovery difficulty?
  Q2  How does altitude error distribute across segment buckets & anchor patterns?
  Q3  Is the risk rule logic consistent with observed error patterns?
"""

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(df: pd.DataFrame) -> dict:
    seg_names = {0: "short", 1: "medium", 2: "long"}
    anchor_names = ["two_anchor", "asymmetric", "sparse_context"]

    summary = {}

    # ---- 1. By segment bucket ----
    for bucket, label in seg_names.items():
        sub = df[df["segment_bucket"] == bucket]
        if len(sub) == 0:
            continue
        summary[label] = {
            "n_samples": int(len(sub)),
            "mean_gap_alt_rmse": float(sub["gap_alt_rmse"].mean()),
            "median_gap_alt_rmse": float(sub["gap_alt_rmse"].median()),
            "std_gap_alt_rmse": float(sub["gap_alt_rmse"].std()),
            "mean_gap_lat_rmse": float(sub["gap_lat_rmse"].mean()),
            "mean_gap_lon_rmse": float(sub["gap_lon_rmse"].mean()),
            "mean_max_gap_min": float(sub["max_gap_minutes"].mean()),
            "mean_anchor_count": float(sub["anchor_count"].mean()),
        }

    # ---- 2. By bucket × anchor pattern (risk cells) ----
    cells = []
    for bucket, blabel in seg_names.items():
        for ap in anchor_names:
            sub = df[(df["segment_bucket"] == bucket) & (df["anchor_pattern_name"] == ap)]
            if len(sub) == 0:
                continue
            cells.append({
                "segment_bucket": blabel,
                "anchor_pattern": ap,
                "n": int(len(sub)),
                "mean_gap_alt_rmse": float(sub["gap_alt_rmse"].mean()),
                "median_gap_alt_rmse": float(sub["gap_alt_rmse"].median()),
                "mean_gap_lat_rmse": float(sub["gap_lat_rmse"].mean()),
                "mean_gap_lon_rmse": float(sub["gap_lon_rmse"].mean()),
                "mean_max_gap_min": float(sub["max_gap_minutes"].mean()),
            })
    summary["risk_cells"] = cells

    # ---- 3. Error vs gap length correlation ----
    if "max_gap_minutes" in df.columns:
        valid = df[df["max_gap_minutes"] > 0]
        if len(valid) > 1:
            summary["corr_gap_len_vs_alt_rmse"] = float(
                np.corrcoef(valid["max_gap_minutes"], valid["gap_alt_rmse"])[0, 1]
            )

    # ---- 4. Distribution check: are risk rules consistent with error? ----
    # From risk rules (segment_risk_rules.yaml):
    # - short+* = high risk (teacher 0.15-0.20)
    # - medium+asymmetric = high risk, medium+two_anchor = medium risk
    # - long+asymmetric = low risk (teacher 0.85), long+two_anchor = medium risk
    # - long+sparse_context = high risk
    risk_map = {
        ("short", "two_anchor"): "high",
        ("short", "asymmetric"): "high",
        ("short", "sparse_context"): "high",
        ("medium", "two_anchor"): "medium",
        ("medium", "asymmetric"): "high",
        ("medium", "sparse_context"): "high",
        ("long", "two_anchor"): "medium",
        ("long", "asymmetric"): "low",
        ("long", "sparse_context"): "high",
    }

    risk_groups = {"high": [], "medium": [], "low": []}
    for cell in cells:
        key = (cell["segment_bucket"], cell["anchor_pattern"])
        rl = risk_map.get(key, "unknown")
        risk_groups[rl].append(cell["mean_gap_alt_rmse"])

    for rl, vals in risk_groups.items():
        if vals:
            summary[f"risk_{rl}_mean_alt_rmse"] = float(np.mean(vals))
            summary[f"risk_{rl}_n_cells"] = len(vals)
            summary[f"risk_{rl}_alt_rmse_range"] = [float(np.min(vals)), float(np.max(vals))]

    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_proposed_24e/main_task_metrics_test_per_sample.csv",
    )
    parser.add_argument("--out-dir", default="outputs/experiments/risk_behavior_analysis")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    csv_path = root / args.csv
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")
    print(f"Columns: {list(df.columns)}")

    # Filter to only gap-containing samples
    df_gap = df[df["gap_count"] > 0].copy()
    print(f"  {len(df_gap)} samples have gaps (gap_count > 0)")

    s = summarize(df_gap)

    # Print summary
    print(f"\n{'='*70}")
    print("BY SEGMENT BUCKET")
    print(f"{'='*70}")
    for label in ["short", "medium", "long"]:
        d = s.get(label, {})
        if d:
            print(f"\n  [{label}] n={d['n_samples']}  "
                  f"alt_rmse μ={d['mean_gap_alt_rmse']:.3f}  "
                  f"median={d['median_gap_alt_rmse']:.3f}  "
                  f"σ={d['std_gap_alt_rmse']:.3f}")
            print(f"         lat_rmse μ={d['mean_gap_lat_rmse']:.4f}  "
                  f"lon_rmse μ={d['mean_gap_lon_rmse']:.4f}  "
                  f"max_gap={d['mean_max_gap_min']:.1f}min  "
                  f"anchors={d['mean_anchor_count']:.1f}")

    print(f"\n{'='*70}")
    print("BY RISK CELL (bucket × anchor pattern)")
    print(f"{'='*70}")
    hdr = f"{'Bucket':<8} {'Anchor':<16} {'n':>6} {'alt_rmse':>10} {'lat_rmse':>10} {'lon_rmse':>10} {'gap_min':>8}"
    print(hdr)
    print("-" * len(hdr))
    for cell in sorted(s["risk_cells"], key=lambda c: c["mean_gap_alt_rmse"], reverse=True):
        print(f"{cell['segment_bucket']:<8} {cell['anchor_pattern']:<16} "
              f"{cell['n']:>6} {cell['mean_gap_alt_rmse']:>10.3f} "
              f"{cell['mean_gap_lat_rmse']:>10.4f} {cell['mean_gap_lon_rmse']:>10.4f} "
              f"{cell['mean_max_gap_min']:>8.1f}")

    print(f"\n{'='*70}")
    print("RISK RULE VALIDATION")
    print(f"{'='*70}")
    for rl in ["high", "medium", "low"]:
        mu = s.get(f"risk_{rl}_mean_alt_rmse")
        n = s.get(f"risk_{rl}_n_cells", 0)
        rng = s.get(f"risk_{rl}_alt_rmse_range", [0, 0])
        if mu is not None:
            print(f"  {rl:>6} risk: μ_alt_rmse={mu:.3f}  range=[{rng[0]:.3f}, {rng[1]:.3f}]  n_cells={n}")

    if "corr_gap_len_vs_alt_rmse" in s:
        print(f"\ncorr(max_gap_min, gap_alt_rmse) = {s['corr_gap_len_vs_alt_rmse']:.4f}")

    json_path = out_dir / "risk_behavior_summary.json"
    with json_path.open("w") as f:
        json.dump(s, f, indent=2, default=float)
    print(f"\n[ok] → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
