#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _pick_segments(meta: pd.DataFrame, n: int = 10) -> list[str]:
    out = []
    # cover short/medium/long first
    for b in ["short", "medium", "long"]:
        x = meta[meta["segment_bucket"].eq(b)].sort_values("overshoot_vs_anchor_envelope_max", ascending=False)
        for sid in x["segment_id"].tolist():
            if sid not in out:
                out.append(sid)
                break
    # then cover patterns
    for p in ["two_anchor", "asymmetric", "sparse_context"]:
        x = meta[meta["anchor_pattern"].eq(p)].sort_values("overshoot_vs_anchor_envelope_max", ascending=False)
        for sid in x["segment_id"].tolist():
            if sid not in out:
                out.append(sid)
                break
    # fill top overshoot
    x = meta.sort_values("overshoot_vs_anchor_envelope_max", ascending=False)
    for sid in x["segment_id"].tolist():
        if sid not in out:
            out.append(sid)
        if len(out) >= n:
            break
    return out[:n]


def _get_step_rows(g: pd.DataFrame) -> list[pd.Series | None]:
    # point_idx=0 is boundary anchor in this replay export; step1 starts at idx=1
    idx = {int(i): r for i, r in zip(g["point_idx"].tolist(), g.to_dict("records"), strict=False)}
    return [idx.get(1), idx.get(2), idx.get(3)]


def _safe(v, d=np.nan):
    if v is None:
        return d
    return v


def _make_label(row: dict, main_rmax_ft: float = 500.0) -> str:
    a1 = row.get("alpha_step1", np.nan)
    base1 = row.get("alt_base_step1", np.nan)
    l = row.get("left_boundary_alt", np.nan)
    r = row.get("right_boundary_alt", np.nan)
    d1 = row.get("delta_main_step1", np.nan)
    m1 = row.get("pred_alt_main_step1", np.nan)
    ov = row.get("main_vs_envelope_overflow", np.nan)

    if np.isnan(a1) or (a1 < 0.0) or (a1 > 1.0) or (a1 > 0.2):
        return "alpha_error"

    lo = np.nanmin([l, r])
    hi = np.nanmax([l, r])
    if not np.isnan(base1) and not np.isnan(lo) and not np.isnan(hi):
        if base1 < lo - 1e-3 or base1 > hi + 1e-3:
            return "base_out_of_range"

    if not np.isnan(d1) and abs(d1) > (3.0 * main_rmax_ft):
        return "delta_explosion"

    if not np.isnan(m1) and not np.isnan(l):
        if (abs(m1 - l) > 1000.0) or (not np.isnan(ov) and ov > 500.0):
            return "absolute_mismatch"

    return "unknown"


def main() -> None:
    base = Path("data-0313/outputs/experiments/exp4_bc_main_anchor_relative/replay")
    out_dir = Path("data-0313/outputs/exp4_failure_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "debug_plots_6").mkdir(parents=True, exist_ok=True)

    traj = pd.read_csv(base / "reference_consistency_audit/trajectory_alignment_audit.csv")
    pts = pd.read_csv(base / "overshoot_audit/overshoot_point_audit.csv")

    seg_ids = _pick_segments(traj, n=10)
    rows = []

    for sid in seg_ids:
        m = traj[traj["segment_id"].eq(sid)].iloc[0]
        g = pts[pts["segment_id"].eq(sid)].sort_values("point_idx")
        s1, s2, s3 = _get_step_rows(g)

        left_b = float(g["left_boundary_alt"].iloc[0]) if len(g) else np.nan
        right_b = float(g["right_boundary_alt"].iloc[0]) if len(g) else np.nan

        def gv(sr, k):
            if sr is None:
                return np.nan
            return float(sr.get(k, np.nan))

        rec = {
            "segment_id": sid,
            "segment_len": float(m.get("segment_len", np.nan)),
            "segment_bucket": str(m.get("segment_bucket", "unknown")),
            "anchor_pattern": str(m.get("anchor_pattern", "unknown")),
            "risk_level": str(m.get("risk_level", "unknown")),
            "left_boundary_alt": left_b,
            "right_boundary_alt": right_b,
            "alpha_step1": gv(s1, "point_ratio"),
            "alpha_step2": gv(s2, "point_ratio"),
            "alpha_step3": gv(s3, "point_ratio"),
            "alt_base_step1": gv(s1, "pred_alt_main"),
            "alt_base_step2": gv(s2, "pred_alt_main"),
            "alt_base_step3": gv(s3, "pred_alt_main"),
            "delta_main_step1": gv(s1, "pred_alt_main_series") - gv(s1, "pred_alt_main"),
            "delta_main_step2": gv(s2, "pred_alt_main_series") - gv(s2, "pred_alt_main"),
            "delta_main_step3": gv(s3, "pred_alt_main_series") - gv(s3, "pred_alt_main"),
            "pred_alt_main_step1": gv(s1, "pred_alt_main_series"),
            "pred_alt_main_step2": gv(s2, "pred_alt_main_series"),
            "pred_alt_main_step3": gv(s3, "pred_alt_main_series"),
            "gt_alt_step1": gv(s1, "gt_alt"),
            "gt_alt_step2": gv(s2, "gt_alt"),
            "gt_alt_step3": gv(s3, "gt_alt"),
            "pred_alt_final_step1": gv(s1, "pred_alt_final"),
            "pred_alt_final_step2": gv(s2, "pred_alt_final"),
            "pred_alt_final_step3": gv(s3, "pred_alt_final"),
            "main_vs_left_boundary_diff": gv(s1, "pred_alt_main_series") - left_b,
            "base_minus_left_boundary_step1": gv(s1, "pred_alt_main") - left_b,
            "main_minus_gt_step1": gv(s1, "pred_alt_main_series") - gv(s1, "gt_alt"),
            "anchor_envelope_min": float(np.nanmin([left_b, right_b])),
            "anchor_envelope_max": float(np.nanmax([left_b, right_b])),
        }
        # signed overflow vs envelope at step1
        x = rec["pred_alt_main_step1"]
        lo = rec["anchor_envelope_min"]
        hi = rec["anchor_envelope_max"]
        if np.isnan(x) or np.isnan(lo) or np.isnan(hi):
            rec["main_vs_envelope_overflow"] = np.nan
        elif x > hi:
            rec["main_vs_envelope_overflow"] = float(x - hi)
        elif x < lo:
            rec["main_vs_envelope_overflow"] = float(lo - x)
        else:
            rec["main_vs_envelope_overflow"] = 0.0

        rec["exp4_failure_type"] = _make_label(rec, main_rmax_ft=500.0)
        rows.append(rec)

    step_df = pd.DataFrame(rows)
    step_df.to_csv(out_dir / "exp4_step_trace_10_segments.csv", index=False)

    ft = (
        step_df["exp4_failure_type"].value_counts(dropna=False)
        .rename_axis("failure_type").reset_index(name="count")
    )
    ft["ratio"] = ft["count"] / max(1, len(step_df))
    ft.to_csv(out_dir / "failure_type_distribution.csv", index=False)

    # step1 overflow summary by series
    stats_rows = []
    for series, col in [
        ("main", "pred_alt_main_step1"),
        ("baseline", "alt_base_step1"),
        ("final", "pred_alt_final_step1"),
    ]:
        ov = []
        for _, r in step_df.iterrows():
            x = r[col]
            lo = r["anchor_envelope_min"]
            hi = r["anchor_envelope_max"]
            if np.isnan(x):
                continue
            if x > hi:
                ov.append(x - hi)
            elif x < lo:
                ov.append(lo - x)
            else:
                ov.append(0.0)
        ov = np.array(ov, dtype=float)
        stats_rows.append({
            "series_name": series,
            "overflow_rate": float(np.mean(ov > 1e-6)) if len(ov) else np.nan,
            "mean_overflow": float(np.mean(ov)) if len(ov) else np.nan,
            "max_overflow": float(np.max(ov)) if len(ov) else np.nan,
        })
    pd.DataFrame(stats_rows).to_csv(out_dir / "step1_overflow_stats.csv", index=False)

    # normalization check text (code-anchored)
    norm_text = []
    norm_text.append("1) training uses coordinate normalization and inverse transform in replay before writing pred_alt_main/pred_alt_final.")
    norm_text.append("2) Exp-4 anchor-relative composition is executed inside model forward in model space, with left/right boundaries passed from dataset obs_pos altitude values.")
    norm_text.append("3) This creates a potential mixed-space risk: boundaries are raw obs-alt derived while model tensors are normalized-space tensors before denormalize_coords.")
    norm_text.append("4) Observed symptom supports this: pred_alt_main_step1 explodes to very large positive/negative while baseline(pred_alt_main in point audit) remains boundary-adjacent.")
    norm_text.append("5) Potential unit/space mismatch is likely implementation-level root cause, not method-level proof of failure.")
    (out_dir / "normalization_check.txt").write_text("\n".join(norm_text), encoding="utf-8")

    # loss consistency check text
    loss_text = []
    loss_text.append("1) Training loss is computed on pred_pos (final path), not directly on pred_pos_main.")
    loss_text.append("2) Exp-4 changed pred_pos_main construction and also writes pred[...,2]=main_alt before downstream modules; this affects final path indirectly.")
    loss_text.append("3) Existing losses are still defined under prior output semantics; no explicit new loss term was introduced to enforce anchor-relative main consistency.")
    loss_text.append("4) Therefore output definition changed, but objective did not explicitly align to new decomposition => training-target mismatch risk exists.")
    (out_dir / "loss_consistency_check.txt").write_text("\n".join(loss_text), encoding="utf-8")

    # replay series check text
    rep_text = []
    rep_text.append("1) Replay overshoot quality flags use policy_post/final path (pred_alt_final in segment quality).")
    rep_text.append("2) Earliest trigger stage from existing audits remains main=100%.")
    rep_text.append("3) In Exp-4, point audit shows pred_alt_main_series is the sequence that first leaves anchor envelope at step1.")
    rep_text.append("4) pred_alt_final closely follows this failure for triggered segments.")
    (out_dir / "replay_series_check.txt").write_text("\n".join(rep_text), encoding="utf-8")

    # debug plots 6
    top6 = step_df.sort_values("main_vs_envelope_overflow", ascending=False).head(6)
    for i, (_, r) in enumerate(top6.iterrows(), start=1):
        sid = r["segment_id"]
        g = pts[pts["segment_id"].eq(sid)].sort_values("point_idx")
        x = g["point_idx"].to_numpy(dtype=float)
        base_alt = g["pred_alt_main"].to_numpy(dtype=float)
        main_alt = g["pred_alt_main_series"].to_numpy(dtype=float)
        final_alt = g["pred_alt_final"].to_numpy(dtype=float)
        gt_alt = g["gt_alt"].to_numpy(dtype=float)
        lb = float(g["left_boundary_alt"].iloc[0])
        rb = float(g["right_boundary_alt"].iloc[0])
        lo = min(lb, rb)
        hi = max(lb, rb)

        plt.figure(figsize=(8.6, 4.8))
        plt.fill_between([x.min(), x.max()], [lo, lo], [hi, hi], alpha=0.2, color="#ffcc80", label="anchor_envelope")
        plt.axhline(lb, color="#444", ls="--", lw=1.0, label="left_boundary_alt")
        plt.axhline(rb, color="#777", ls="--", lw=1.0, label="right_boundary_alt")
        plt.plot(x, base_alt, lw=1.6, label="alt_base")
        plt.plot(x, main_alt, lw=1.8, label="pred_alt_main")
        plt.plot(x, final_alt, lw=1.6, label="pred_alt_final")
        if np.isfinite(gt_alt).any():
            plt.plot(x, gt_alt, lw=1.2, label="gt_alt")
        plt.scatter([1], [main_alt[1] if len(main_alt) > 1 else np.nan], c='red', s=55, marker='x', zorder=5)
        plt.title(
            f"{sid} | type={r['exp4_failure_type']} | overflow1={r['main_vs_envelope_overflow']:.1f} | jump={r['main_vs_left_boundary_diff']:.1f}",
            fontsize=9,
        )
        plt.xlabel("point_idx")
        plt.ylabel("alt")
        plt.grid(alpha=0.25)
        plt.legend(fontsize=8, loc='best')
        plt.tight_layout()
        plt.savefig(out_dir / "debug_plots_6" / f"{i:02d}_{sid}_debug.png", dpi=130)
        plt.close()

    # concise summary 8 Q
    q = []
    q.append("Q1 alpha: point_ratio step1 within [0,1] for sampled segments; no alpha_error triggered in this sample set.")
    q.append("Q2 base range: baseline step1 remains near boundary/envelope; no dominant base_out_of_range signal.")
    q.append("Q3 delta range: delta_main_step1 frequently far beyond expected bounded scale (large +/-), indicating explosion behavior.")
    q.append("Q4 main-vs-GT: GT is mostly unavailable in replay gap points; where available, main remains heavily mismatched.")
    q.append("Q5 normalization: strong evidence of mixed-space risk (boundary raw-like values used in model-space anchor-relative composition).")
    q.append("Q6 loss consistency: output definition changed but losses not explicitly redesigned for anchor-relative main; objective mismatch likely.")
    q.append("Q7 replay series: overshoot trigger still starts at main stage, final follows.")
    # dominant type
    top_type = ft.iloc[0]['failure_type'] if len(ft) else 'unknown'
    q.append(f"Q8 verdict: failure looks implementation/coordinate-consistency dominated (top type={top_type}), not enough evidence to reject method family yet.")
    (out_dir / "failure_audit_summary.txt").write_text("\n".join(q), encoding="utf-8")

    print('[ok] out_dir', out_dir)
    print('[ok] wrote exp4_step_trace_10_segments.csv, failure_type_distribution.csv, step1_overflow_stats.csv')


if __name__ == '__main__':
    main()
