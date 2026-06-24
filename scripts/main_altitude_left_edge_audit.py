#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXPERIMENTS = [
    "exp0_baseline_recheck",
    "exp1_bounded_residual_uniform",
    "exp2_risk_aware_teacher_fixed",
    "exp3_left_edge_directional_constraint",
]


def signed_overflow(x: pd.Series, lo: pd.Series, hi: pd.Series) -> pd.Series:
    # positive => above band, negative => below band, zero => inside band
    return np.where(x > hi, x - hi, np.where(x < lo, x - lo, 0.0))


def overflow_amount(x: pd.Series, lo: pd.Series, hi: pd.Series) -> pd.Series:
    return np.abs(signed_overflow(x, lo, hi))


def classify_violation(row: pd.Series) -> str:
    main_diff = float(row.get("main_vs_left_boundary_diff", np.nan))
    jump = float(row.get("main_jump_ratio", np.nan))
    main_ov = float(row.get("main_vs_envelope_overflow", 0.0))
    base_ov = float(row.get("baseline_vs_envelope_overflow", 0.0))
    final_ov = float(row.get("final_vs_envelope_overflow", 0.0))
    step2 = float(row.get("pred_alt_main_step2", np.nan))
    step1 = float(row.get("pred_alt_main_step1", np.nan))

    if np.isnan(main_diff):
        return "unknown"

    if main_ov > 0 and abs(main_diff) >= 1500 and (abs(jump) >= 2.0 if not np.isnan(jump) else True):
        return "scale_mismatch"

    if main_ov > 0 and not np.isnan(step1) and not np.isnan(step2):
        if abs(step1 - step2) <= 80 and abs(main_diff) >= 400:
            return "offset_mismatch"

    if base_ov <= 1.0 and main_ov > 200 and final_ov > 200:
        return "anchor_inconsistent"

    if base_ov <= 1.0 and main_ov > 50:
        return "free_unconstrained_output"

    return "unknown"


def load_one(exp_dir: Path, experiment: str) -> pd.DataFrame:
    p = exp_dir / experiment / "trajectory_alignment_audit.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)

    # Build unique row key to avoid segment_id duplication across task_type
    if "task_type" in df.columns:
        df["segment_uid"] = df["segment_id"].astype(str) + "|" + df["task_type"].astype(str)
    else:
        df["segment_uid"] = df["segment_id"].astype(str)

    # proxy boundaries: step-1 baseline at left/right are the closest available explicit boundary-like values.
    df["left_boundary_alt"] = df["alt_baseline_left_1"]
    df["right_boundary_alt"] = df["alt_baseline_right_1"]
    df["anchor_envelope_min"] = df["overshoot_ref_global_min"]
    df["anchor_envelope_max"] = df["overshoot_ref_global_max"]

    # step1/step2 already in trajectory alignment audit
    for c in [
        "pred_alt_main_left_1", "pred_alt_main_left_2",
        "alt_baseline_left_1", "alt_baseline_left_2",
        "pred_alt_final_left_1", "pred_alt_final_left_2",
    ]:
        if c not in df.columns:
            df[c] = np.nan

    df["pred_alt_main_step1"] = df["pred_alt_main_left_1"]
    df["pred_alt_main_step2"] = df["pred_alt_main_left_2"]
    df["alt_baseline_step1"] = df["alt_baseline_left_1"]
    df["alt_baseline_step2"] = df["alt_baseline_left_2"]
    df["pred_alt_final_step1"] = df["pred_alt_final_left_1"]
    df["pred_alt_final_step2"] = df["pred_alt_final_left_2"]
    if "policy_post_left_1" in df.columns:
        df["policy_post_step1"] = df["policy_post_left_1"]
        df["policy_post_step2"] = df["policy_post_left_2"]
    else:
        df["policy_post_step1"] = df["pred_alt_final_step1"]
        df["policy_post_step2"] = df["pred_alt_final_step2"]

    df["main_vs_left_boundary_diff"] = df["pred_alt_main_step1"] - df["left_boundary_alt"]
    df["main_vs_envelope_signed"] = signed_overflow(df["pred_alt_main_step1"], df["anchor_envelope_min"], df["anchor_envelope_max"])
    df["baseline_vs_envelope_signed"] = signed_overflow(df["alt_baseline_step1"], df["anchor_envelope_min"], df["anchor_envelope_max"])
    df["final_vs_envelope_signed"] = signed_overflow(df["pred_alt_final_step1"], df["anchor_envelope_min"], df["anchor_envelope_max"])

    df["main_vs_envelope_overflow"] = np.abs(df["main_vs_envelope_signed"])
    df["baseline_vs_envelope_overflow"] = np.abs(df["baseline_vs_envelope_signed"])
    df["final_vs_envelope_overflow"] = np.abs(df["final_vs_envelope_signed"])

    denom = (df["right_boundary_alt"] - df["left_boundary_alt"]).replace(0.0, np.nan)
    df["main_jump_ratio"] = df["main_vs_left_boundary_diff"] / (denom + 1e-6)

    df["main_boundary_violation_type"] = df.apply(classify_violation, axis=1)
    df["experiment"] = experiment

    keep_cols = [
        "experiment", "segment_uid", "segment_id", "task_type", "flight_id", "segment_bucket", "anchor_pattern", "risk_level", "matched_risk_rule",
        "left_boundary_alt", "right_boundary_alt", "anchor_envelope_min", "anchor_envelope_max",
        "pred_alt_main_step1", "pred_alt_main_step2",
        "alt_baseline_step1", "alt_baseline_step2",
        "pred_alt_final_step1", "pred_alt_final_step2",
        "policy_post_step1", "policy_post_step2",
        "main_vs_left_boundary_diff", "main_vs_envelope_overflow",
        "baseline_vs_envelope_overflow", "final_vs_envelope_overflow", "main_jump_ratio",
        "overshoot_flag", "overshoot_trigger_step", "overshoot_trigger_stage", "overshoot_trigger_series",
        "main_boundary_violation_type",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].copy()


def save_plot(row: pd.Series, out_path: Path) -> None:
    x = np.array([1, 2])
    baseline = np.array([row["alt_baseline_step1"], row["alt_baseline_step2"]], dtype=float)
    main = np.array([row["pred_alt_main_step1"], row["pred_alt_main_step2"]], dtype=float)
    final = np.array([row["pred_alt_final_step1"], row["pred_alt_final_step2"]], dtype=float)
    lo = float(row["anchor_envelope_min"])
    hi = float(row["anchor_envelope_max"])

    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.fill_between([0.7, 2.3], [lo, lo], [hi, hi], color="#ffe082", alpha=0.35, label="anchor_envelope_band")
    ax.plot(x, baseline, "-o", label="alt_baseline", linewidth=1.8)
    ax.plot(x, main, "-o", label="pred_alt_main", linewidth=2.2)
    ax.plot(x, final, "-o", label="pred_alt_final", linewidth=1.8)
    ax.axhline(float(row["left_boundary_alt"]), color="#444", linestyle="--", linewidth=1.0, label="left_boundary_alt")
    ax.axhline(float(row["right_boundary_alt"]), color="#888", linestyle="--", linewidth=1.0, label="right_boundary_alt")
    ax.scatter([1], [main[0]], s=70, marker="x", color="red", zorder=5)
    ax.text(1.03, main[0], f"overflow={row['main_vs_envelope_overflow']:.1f}", fontsize=8, color="red")

    title = (
        f"{row['experiment']} | {row['segment_id']} | bucket={row.get('segment_bucket','?')}"
        f"\nmain-left-diff={row['main_vs_left_boundary_diff']:.1f}, jump_ratio={row.get('main_jump_ratio', np.nan):.2f},"
        f" type={row['main_boundary_violation_type']}"
    )
    ax.set_title(title, fontsize=9)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["step1", "step2"])
    ax.set_ylabel("Altitude")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", type=Path, default=Path("data-0313/outputs/reference_consistency_audit"))
    ap.add_argument("--out-dir", type=Path, default=Path("data-0313/outputs/main_altitude_audit"))
    ap.add_argument("--plot-count", type=int, default=20)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "left_edge_step1_plots_20").mkdir(parents=True, exist_ok=True)

    dfs = []
    for e in EXPERIMENTS:
        dfs.append(load_one(args.ref_dir, e))
    all_df = pd.concat(dfs, ignore_index=True)

    # Save core alignment table
    all_df.to_csv(args.out_dir / "left_edge_step1_alignment.csv", index=False)

    # Step1 overflow stats
    rows = []
    for exp, g in all_df.groupby("experiment"):
        for series in ["main", "baseline", "final"]:
            col = f"{series}_vs_envelope_overflow"
            vals = g[col].fillna(0.0)
            rows.append({
                "experiment": exp,
                "series_name": series,
                "overflow_rate": float((vals > 1e-6).mean()),
                "mean_overflow": float(vals.mean()),
                "max_overflow": float(vals.max()),
            })
    step1_stats = pd.DataFrame(rows)
    step1_stats.to_csv(args.out_dir / "step1_overflow_stats.csv", index=False)

    # Jump ratio distribution
    bins = [(-np.inf, 0.1, "<0.1"), (0.1, 0.5, "0.1~0.5"), (0.5, 1.0, "0.5~1.0"), (1.0, np.inf, ">1.0")]
    jrows = []
    for exp, g in all_df.groupby("experiment"):
        s = g["main_jump_ratio"].replace([np.inf, -np.inf], np.nan)
        total = int(s.notna().sum())
        for lo, hi, label in bins:
            if np.isinf(lo):
                m = s < hi
            elif np.isinf(hi):
                m = s > lo
            else:
                m = (s >= lo) & (s < hi)
            cnt = int(m.sum())
            jrows.append({"experiment": exp, "bucket": label, "ratio": float(cnt / total) if total else np.nan, "count": cnt, "total_non_nan": total})
    pd.DataFrame(jrows).to_csv(args.out_dir / "main_jump_ratio_distribution.csv", index=False)

    # Violation type distribution
    vrows = []
    for exp, g in all_df.groupby("experiment"):
        n = len(g)
        for typ, k in g["main_boundary_violation_type"].value_counts(dropna=False).items():
            vrows.append({"experiment": exp, "type": str(typ), "total": int(k), "ratio": float(k / n) if n else np.nan})
    pd.DataFrame(vrows).to_csv(args.out_dir / "violation_type_distribution.csv", index=False)

    # left-edge series overflow table
    lrows = []
    for exp, g in all_df.groupby("experiment"):
        for series, step_col in [
            ("baseline", "alt_baseline_step1"),
            ("main", "pred_alt_main_step1"),
            ("final", "pred_alt_final_step1"),
            ("policy_post", "policy_post_step1"),
        ]:
            ov1 = overflow_amount(g[step_col], g["anchor_envelope_min"], g["anchor_envelope_max"])
            s1 = signed_overflow(g[step_col], g["anchor_envelope_min"], g["anchor_envelope_max"])
            lrows.append({
                "experiment": exp,
                "series_name": series,
                "left_step": 1,
                "overflow_rate": float((ov1 > 1e-6).mean()),
                "mean_signed_overflow": float(np.nanmean(s1)),
                "max_signed_overflow": float(np.nanmax(s1)),
            })
        for series, step_col in [
            ("baseline", "alt_baseline_step2"),
            ("main", "pred_alt_main_step2"),
            ("final", "pred_alt_final_step2"),
            ("policy_post", "policy_post_step2"),
        ]:
            ov2 = overflow_amount(g[step_col], g["anchor_envelope_min"], g["anchor_envelope_max"])
            s2 = signed_overflow(g[step_col], g["anchor_envelope_min"], g["anchor_envelope_max"])
            lrows.append({
                "experiment": exp,
                "series_name": series,
                "left_step": 2,
                "overflow_rate": float((ov2 > 1e-6).mean()),
                "mean_signed_overflow": float(np.nanmean(s2)),
                "max_signed_overflow": float(np.nanmax(s2)),
            })
    pd.DataFrame(lrows).to_csv(args.out_dir / "left_edge_series_overflow.csv", index=False)

    # Generation pipeline doc (static, code-anchored)
    pipe_md = """# Main Altitude Generation Pipeline (Code-Anchored Audit)\n\n## Files inspected\n- `src/models/full_model.py`\n- `scripts/real_adsc_replay_eval.py`\n\n## Pipeline\n1. Forward/backward gap-aware LSTM produce branch outputs (`mu_f`, `mu_b`).\n2. Fusion module generates `pred` (3D trajectory).\n3. `pred_main = pred` is captured immediately after fusion (`full_model.py`, around `pred_main = pred`).\n4. Optional altitude modules then modify altitude channel in sequence:\n   - alt baseline+residual head (`alt_base_residual_enabled`)\n   - DMS refiner (`alt_dms_refiner_enabled`)\n   - vertical projector\n   - alt bias\n5. Replay reads:\n   - `pred_alt_main` from `out['pred_pos_main']`\n   - `pred_alt_final` from final `out[pred_key]` (typically `pred_pos`)\n\n## Altitude definition\n- `pred_alt_main` is treated as **absolute altitude after inverse normalization/transform** in replay.\n- It is not the DMS residual output.\n\n## Boundary conditioning check\n- `pred_main` is emitted before explicit anchor-aware residual/policy modules.\n- No explicit hard boundary conditioning is applied at `pred_main` emission point.\n- Boundary information can only be implicitly encoded via sequence inputs.\n\n## Normalization chain (replay)\n- Input normalization: `normalize_coords(...)`\n- Output restore: `denormalize_coords(...) -> invert_alt_target_transform(...) -> restore_to_latlon(...)`\n- Same restore path is used for main and final series.\n"""
    (args.out_dir / "main_alt_generation_pipeline.md").write_text(pipe_md, encoding="utf-8")

    # Build concise summary with 8 Q answers scaffold
    summary_lines = []
    summary_lines.append("# Main Altitude Left-Boundary Consistency Summary")
    summary_lines.append("")
    for exp in EXPERIMENTS:
        g = all_df[all_df["experiment"] == exp]
        m = g["main_vs_envelope_overflow"].fillna(0)
        b = g["baseline_vs_envelope_overflow"].fillna(0)
        f = g["final_vs_envelope_overflow"].fillna(0)
        diff = g["main_vs_left_boundary_diff"].dropna()
        summary_lines.append(f"## {exp}")
        summary_lines.append(f"- segments: {len(g)}")
        summary_lines.append(f"- main step1 overflow rate: {(m>1e-6).mean():.4f}, mean: {m.mean():.2f}, max: {m.max():.2f}")
        summary_lines.append(f"- baseline step1 overflow rate: {(b>1e-6).mean():.4f}, mean: {b.mean():.2f}")
        summary_lines.append(f"- final step1 overflow rate: {(f>1e-6).mean():.4f}, mean: {f.mean():.2f}")
        summary_lines.append(f"- main vs left-boundary diff mean/max: {diff.mean():.2f} / {diff.max():.2f}")
        vc = g["main_boundary_violation_type"].value_counts(normalize=True).round(4).to_dict()
        summary_lines.append(f"- violation type mix: {vc}")
        summary_lines.append("")

    summary_lines.append("## Notes")
    summary_lines.append("- `left_boundary_alt` / `right_boundary_alt` are proxied by baseline edge values (`alt_baseline_left_1` / `alt_baseline_right_1`) due missing explicit raw boundary columns in trajectory_alignment_audit.")
    summary_lines.append("- Envelope uses replay overshoot reference (`overshoot_ref_global_min/max`) to stay consistent with current overshoot flag reference frame.")
    (args.out_dir / "main_altitude_audit_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    # 20 plots: largest main step1 overflow across all experiments
    plot_df = all_df.sort_values("main_vs_envelope_overflow", ascending=False).head(args.plot_count)
    for i, (_, row) in enumerate(plot_df.iterrows(), start=1):
        fname = f"{i:02d}_{row['experiment']}_{row['segment_id'].replace('/', '_')}_leftedge_step1.png"
        save_plot(row, args.out_dir / "left_edge_step1_plots_20" / fname)

    print(f"[done] wrote audit to {args.out_dir}")
    print(f"[done] rows={len(all_df)}, plots={len(plot_df)}")


if __name__ == "__main__":
    main()
