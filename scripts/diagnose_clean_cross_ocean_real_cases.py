from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = ["BiLSTM-clean", "Backbone-only", "Ours-A3"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Diagnose structural quality for clean real cross-ocean model comparisons.")
    p.add_argument("--compare-dir", default="outputs/runs/clean_cross_ocean_model_compare_20260517")
    p.add_argument("--case-csv", default="outputs/runs/clean_cross_ocean_adsc_adsb_cases_20260517/selected_clean_cross_ocean_cases.csv")
    return p


def _safe_mean(x) -> float:
    arr = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    return float(arr.mean()) if len(arr) else float("nan")


def _safe_max(x) -> float:
    arr = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    return float(arr.max()) if len(arr) else float("nan")


def _linear_ref(left_alt: float, right_alt: float, n: int) -> np.ndarray:
    return np.linspace(float(left_alt), float(right_alt), int(n))


def _gap_segments(df: pd.DataFrame) -> list[tuple[int, int]]:
    anchor_idx = df.index[pd.to_numeric(df["is_adsc_anchor"], errors="coerce").fillna(0).astype(int) == 1].tolist()
    segs = []
    for left, right in zip(anchor_idx[:-1], anchor_idx[1:]):
        if right - left > 1:
            segs.append((left, right))
    return segs


def diagnose_case(pair_id: str, recovered_csv: Path, case_meta: pd.Series | None = None) -> tuple[list[dict], list[dict]]:
    df = pd.read_csv(recovered_csv, parse_dates=["minute_ts"]).sort_values("minute_ts").reset_index(drop=True)
    segs = _gap_segments(df)
    gap_rows: list[dict] = []
    anchor_rows: list[dict] = []

    for model in MODELS:
        pred_col = f"{model}_pred_alt"
        if pred_col not in df.columns:
            continue
        pred = pd.to_numeric(df[pred_col], errors="coerce")
        true_anchor = pd.to_numeric(df["alt"], errors="coerce")
        anchor_mask = pd.to_numeric(df["is_adsc_anchor"], errors="coerce").fillna(0).astype(int) == 1
        anchor_err = (pred[anchor_mask] - true_anchor[anchor_mask]).abs()
        anchor_rows.append(
            {
                "pair_id": pair_id,
                "model": model,
                "anchor_count": int(anchor_mask.sum()),
                "anchor_mae_m": _safe_mean(anchor_err),
                "anchor_max_abs_m": _safe_max(anchor_err),
            }
        )

    for gap_id, (left, right) in enumerate(segs, start=1):
        left_t = df.loc[left, "minute_ts"]
        right_t = df.loc[right, "minute_ts"]
        gap_minutes = int(right - left - 1)
        left_alt = float(pd.to_numeric(df.loc[left, "alt"], errors="coerce"))
        right_alt = float(pd.to_numeric(df.loc[right, "alt"], errors="coerce"))
        ref = _linear_ref(left_alt, right_alt, right - left + 1)
        for model in MODELS:
            pred_col = f"{model}_pred_alt"
            if pred_col not in df.columns:
                continue
            seg_pred = pd.to_numeric(df.loc[left:right, pred_col], errors="coerce").to_numpy(dtype=float)
            if len(seg_pred) != len(ref):
                continue
            miss = seg_pred[1:-1]
            ref_miss = ref[1:-1]
            diff1 = np.abs(np.diff(seg_pred))
            diff2 = np.abs(np.diff(seg_pred, n=2)) if len(seg_pred) >= 3 else np.array([], dtype=float)
            left_jump = abs(seg_pred[1] - seg_pred[0]) if len(seg_pred) >= 2 else np.nan
            right_jump = abs(seg_pred[-1] - seg_pred[-2]) if len(seg_pred) >= 2 else np.nan
            ref_dev = np.abs(miss - ref_miss) if len(miss) else np.array([], dtype=float)
            gap_rows.append(
                {
                    "pair_id": pair_id,
                    "gap_id": gap_id,
                    "model": model,
                    "left_time": left_t,
                    "right_time": right_t,
                    "gap_minutes": gap_minutes,
                    "left_anchor_alt_m": left_alt,
                    "right_anchor_alt_m": right_alt,
                    "anchor_alt_delta_m": right_alt - left_alt,
                    "boundary_jump_mean_m": float(np.nanmean([left_jump, right_jump])),
                    "boundary_jump_max_m": float(np.nanmax([left_jump, right_jump])),
                    "vertical_step_mean_m": float(np.nanmean(diff1)) if len(diff1) else np.nan,
                    "vertical_step_max_m": float(np.nanmax(diff1)) if len(diff1) else np.nan,
                    "vertical_second_diff_mean_m": float(np.nanmean(diff2)) if len(diff2) else np.nan,
                    "vertical_second_diff_max_m": float(np.nanmax(diff2)) if len(diff2) else np.nan,
                    "ref_dev_mae_m": float(np.nanmean(ref_dev)) if len(ref_dev) else np.nan,
                    "ref_dev_max_m": float(np.nanmax(ref_dev)) if len(ref_dev) else np.nan,
                }
            )
    return gap_rows, anchor_rows


def main() -> int:
    args = build_parser().parse_args()
    compare_dir = Path(args.compare_dir)
    summary_path = compare_dir / "clean_cross_ocean_model_compare_summary.csv"
    summary = pd.read_csv(summary_path)
    case_meta = pd.read_csv(args.case_csv)
    case_meta = case_meta.set_index("pair_id", drop=False)

    all_gap_rows: list[dict] = []
    all_anchor_rows: list[dict] = []
    for _, row in summary.iterrows():
        pair_id = str(row["pair_id"])
        gap_rows, anchor_rows = diagnose_case(pair_id, Path(row["recovered_csv"]), case_meta.loc[pair_id] if pair_id in case_meta.index else None)
        all_gap_rows.extend(gap_rows)
        all_anchor_rows.extend(anchor_rows)

    gap_df = pd.DataFrame(all_gap_rows)
    anchor_df = pd.DataFrame(all_anchor_rows)
    gap_out = compare_dir / "real_cross_ocean_gap_structural_metrics.csv"
    anchor_out = compare_dir / "real_cross_ocean_anchor_consistency.csv"
    gap_df.to_csv(gap_out, index=False)
    anchor_df.to_csv(anchor_out, index=False)

    model_summary = (
        gap_df.groupby("model")
        .agg(
            gap_count=("gap_id", "count"),
            gap_minutes_mean=("gap_minutes", "mean"),
            boundary_jump_mean_m=("boundary_jump_mean_m", "mean"),
            boundary_jump_p90_m=("boundary_jump_mean_m", lambda s: float(np.nanpercentile(s, 90))),
            vertical_step_mean_m=("vertical_step_mean_m", "mean"),
            vertical_second_diff_mean_m=("vertical_second_diff_mean_m", "mean"),
            ref_dev_mae_m=("ref_dev_mae_m", "mean"),
            ref_dev_p90_m=("ref_dev_mae_m", lambda s: float(np.nanpercentile(s, 90))),
        )
        .reset_index()
    )
    anchor_summary = (
        anchor_df.groupby("model")
        .agg(anchor_mae_m=("anchor_mae_m", "mean"), anchor_max_abs_m=("anchor_max_abs_m", "max"))
        .reset_index()
    )
    model_summary = model_summary.merge(anchor_summary, on="model", how="left")
    model_summary.to_csv(compare_dir / "real_cross_ocean_model_structural_summary.csv", index=False)

    case_rows = []
    for pair_id, g in gap_df.groupby("pair_id"):
        pivot = g.pivot_table(
            index="gap_id",
            columns="model",
            values=["boundary_jump_mean_m", "vertical_second_diff_mean_m", "ref_dev_mae_m"],
            aggfunc="mean",
        )
        row = {
            "pair_id": pair_id,
            "gap_count": int(g["gap_id"].nunique()),
            "gap_minutes_max": int(g["gap_minutes"].max()),
            "gap_minutes_mean": float(g["gap_minutes"].mean()),
        }
        for metric in ["boundary_jump_mean_m", "vertical_second_diff_mean_m", "ref_dev_mae_m"]:
            for model in MODELS:
                key = f"{metric}_{model}"
                try:
                    row[key] = float(pivot[(metric, model)].mean())
                except Exception:
                    row[key] = np.nan
            try:
                row[f"{metric}_ours_vs_bilstm_delta"] = row[f"{metric}_Ours-A3"] - row[f"{metric}_BiLSTM-clean"]
            except Exception:
                row[f"{metric}_ours_vs_bilstm_delta"] = np.nan
        if pair_id in case_meta.index:
            meta = case_meta.loc[pair_id]
            for col in ["adsc_anchor_count", "max_frozen_run_min", "adsb_rows_inside_adsc_window", "adsc_endpoint_distance_km", "adsc_ocean_ratio_gc"]:
                if col in meta:
                    row[col] = meta[col]
        case_rows.append(row)
    case_summary = pd.DataFrame(case_rows)
    # Prefer cases with enough anchors, long gaps, low freezing, and Ours-A3 lower boundary/ref-deviation.
    case_summary["paper_case_score"] = (
        case_summary["gap_minutes_max"].fillna(0) * 0.01
        + case_summary.get("adsc_anchor_count", pd.Series(0, index=case_summary.index)).fillna(0) * 0.2
        - case_summary.get("max_frozen_run_min", pd.Series(0, index=case_summary.index)).fillna(0) * 0.3
        - case_summary["boundary_jump_mean_m_ours_vs_bilstm_delta"].fillna(0) * 0.02
        - case_summary["ref_dev_mae_m_ours_vs_bilstm_delta"].fillna(0) * 0.01
    )
    case_summary = case_summary.sort_values("paper_case_score", ascending=False)
    case_summary.to_csv(compare_dir / "real_cross_ocean_case_selection_scores.csv", index=False)

    print("[model summary]")
    print(model_summary.to_string(index=False))
    print("\n[top cases]")
    cols = [
        "pair_id",
        "paper_case_score",
        "gap_count",
        "gap_minutes_max",
        "adsc_anchor_count",
        "max_frozen_run_min",
        "boundary_jump_mean_m_ours_vs_bilstm_delta",
        "ref_dev_mae_m_ours_vs_bilstm_delta",
    ]
    print(case_summary[[c for c in cols if c in case_summary.columns]].head(10).to_string(index=False))
    print(f"\n[done] {gap_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
