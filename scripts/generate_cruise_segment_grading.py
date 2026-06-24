from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/home/jj/workspace/data-0313")
SAMPLES_PATH = ROOT / "outputs" / "mvp_adsb3864_20260322" / "samples.parquet"
FILL_DETAIL_PATH = ROOT / "outputs" / "runs" / "fill_segment_strategy_compare_20260331_v5" / "fill_segment_strategy_detail.csv"
TEST_SAMPLE_PATH = ROOT / "outputs" / "runs" / "fair_train_adsb3864_ourmethod_bilstm_e20_20260402" / "main_task_metrics_test_per_sample.csv"
OUT_ROOT = ROOT / "outputs" / "runs" / "cruise_segment_grading_20260406"


def ensure_dirs() -> None:
    (OUT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "figures").mkdir(parents=True, exist_ok=True)


def robust_rank(x: pd.Series) -> pd.Series:
    return x.rank(pct=True, method="average")


def build_sample_level_features(samples: pd.DataFrame) -> pd.DataFrame:
    gap = samples[samples["obs_mask"].eq(0)].copy()
    if gap.empty:
        raise RuntimeError("No gap rows found in samples.parquet")
    all_sample = (
        samples.groupby("sample_id")
        .agg(
            total_anchor_count=("is_anchor", "sum"),
            window_minutes=("minute_ts", "size"),
        )
        .reset_index()
    )

    agg = (
        gap.groupby("sample_id")
        .agg(
            flight_id=("flight_id", "first"),
            mean_abs_heading_rate=("heading_rate", lambda s: float(pd.to_numeric(s, errors="coerce").abs().mean())),
            p90_abs_heading_rate=("heading_rate", lambda s: float(pd.to_numeric(s, errors="coerce").abs().quantile(0.9))),
            mean_abs_speed_delta=("speed_delta", lambda s: float(pd.to_numeric(s, errors="coerce").abs().mean())),
            p90_abs_speed_delta=("speed_delta", lambda s: float(pd.to_numeric(s, errors="coerce").abs().quantile(0.9))),
            mean_altitude_variance_proxy=("local_alt_std", lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
            p90_altitude_variance_proxy=("local_alt_std", lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.9))),
            max_gap_len=("gap_len", "max"),
            mean_gap_pos_ratio=("gap_pos_ratio", "mean"),
            gap_rows=("obs_mask", "size"),
        )
        .reset_index()
    )
    agg = agg.merge(all_sample, on="sample_id", how="left")

    agg["heading_rank"] = robust_rank(agg["mean_abs_heading_rate"])
    agg["speed_rank"] = robust_rank(agg["mean_abs_speed_delta"])
    agg["altvar_rank"] = robust_rank(agg["mean_altitude_variance_proxy"])
    agg["disturbance_score"] = (agg["heading_rank"] + agg["speed_rank"] + agg["altvar_rank"]) / 3.0

    q40 = agg["disturbance_score"].quantile(0.4)
    q80 = agg["disturbance_score"].quantile(0.8)
    h90 = agg["mean_abs_heading_rate"].quantile(0.9)
    s90 = agg["mean_abs_speed_delta"].quantile(0.9)
    a90 = agg["mean_altitude_variance_proxy"].quantile(0.9)

    def classify(row: pd.Series) -> str:
        if (
            row["disturbance_score"] >= q80
            or row["mean_abs_heading_rate"] >= h90
            or row["mean_abs_speed_delta"] >= s90
            or row["mean_altitude_variance_proxy"] >= a90
        ):
            return "disturbed"
        if row["disturbance_score"] <= q40:
            return "stable"
        return "normal"

    agg["segment_grade"] = agg.apply(classify, axis=1)
    agg["boundary_sensitive"] = (
        (agg["total_anchor_count"] <= 2)
        | (agg["max_gap_len"] >= agg["max_gap_len"].quantile(0.9))
    )

    threshold_meta = {
        "disturbance_score_q40": float(q40),
        "disturbance_score_q80": float(q80),
        "heading_p90": float(h90),
        "speed_delta_p90": float(s90),
        "altitude_variance_proxy_p90": float(a90),
    }
    (OUT_ROOT / "tables" / "segment_grade_thresholds.json").write_text(json.dumps(threshold_meta, indent=2), encoding="utf-8")
    agg.to_csv(OUT_ROOT / "tables" / "cruise_segment_grade_by_sample.csv", index=False)
    return agg


def build_grade_summary(sample_grade: pd.DataFrame) -> pd.DataFrame:
    summary = (
        sample_grade.groupby("segment_grade")
        .agg(
            sample_count=("sample_id", "size"),
            ratio=("sample_id", lambda s: float(len(s) / len(sample_grade))),
            mean_abs_heading_rate=("mean_abs_heading_rate", "mean"),
            mean_abs_speed_delta=("mean_abs_speed_delta", "mean"),
            mean_altitude_variance_proxy=("mean_altitude_variance_proxy", "mean"),
            median_max_gap_len=("max_gap_len", "median"),
        )
        .reset_index()
    )
    summary.to_csv(OUT_ROOT / "tables" / "cruise_segment_grade_summary.csv", index=False)
    return summary


def build_gap_anchor_stats(samples: pd.DataFrame) -> pd.DataFrame:
    subsets = {
        "anchor_region": samples[samples["is_anchor"].eq(1)],
        "gap_region": samples[samples["obs_mask"].eq(0)],
        "boundary_gap_region": samples[samples["obs_mask"].eq(0) & ((samples["gap_pos_ratio"] <= 0.2) | (samples["gap_pos_ratio"] >= 0.8))],
        "interior_gap_region": samples[samples["obs_mask"].eq(0) & (samples["gap_pos_ratio"] > 0.2) & (samples["gap_pos_ratio"] < 0.8)],
    }
    rows = []
    for region_name, df in subsets.items():
        for feat in ["vertical_speed", "heading_rate", "speed_delta"]:
            vals = pd.to_numeric(df[feat], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            rows.append(
                {
                    "region": region_name,
                    "feature": feat,
                    "count": int(vals.shape[0]),
                    "mean_abs": float(vals.abs().mean()) if not vals.empty else np.nan,
                    "p50_abs": float(vals.abs().quantile(0.5)) if not vals.empty else np.nan,
                    "p90_abs": float(vals.abs().quantile(0.9)) if not vals.empty else np.nan,
                    "std": float(vals.std(ddof=0)) if not vals.empty else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_ROOT / "tables" / "gap_anchor_region_stats.csv", index=False)
    return out


def build_error_by_grade(sample_grade: pd.DataFrame) -> pd.DataFrame:
    test = pd.read_csv(TEST_SAMPLE_PATH)
    merged = sample_grade.merge(test[["sample_id", "alt_rmse", "gap_alt_rmse", "lat_rmse", "lon_rmse", "anchor_count", "max_gap_minutes"]], on="sample_id", how="left")
    summary = (
        merged.groupby("segment_grade")
        .agg(
            sample_count=("sample_id", "size"),
            matched_test_samples=("alt_rmse", lambda s: int(s.notna().sum())),
            mean_alt_rmse=("alt_rmse", "mean"),
            mean_gap_alt_rmse=("gap_alt_rmse", "mean"),
            mean_lat_rmse=("lat_rmse", "mean"),
            mean_lon_rmse=("lon_rmse", "mean"),
            mean_max_gap_minutes=("max_gap_minutes", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(OUT_ROOT / "tables" / "cruise_segment_error_by_grade.csv", index=False)
    merged.to_csv(OUT_ROOT / "tables" / "cruise_segment_grade_with_test_metrics.csv", index=False)
    return summary


def build_fill_segment_summary(sample_grade: pd.DataFrame) -> pd.DataFrame:
    detail = pd.read_csv(FILL_DETAIL_PATH)
    current = detail[detail["strategy_name"].eq("production_chain_v1")].copy()
    if current.empty:
        # Fallback to the most deployment-like strategy if production rows are absent.
        current = detail[detail["strategy_name"].eq("default_soft_taper_plus_len_bucket_cap")].copy()
    merged = current.merge(sample_grade[["sample_id", "segment_grade", "boundary_sensitive"]], on="sample_id", how="left")
    merged["segment_grade"] = merged["segment_grade"].fillna("unknown")

    summary = (
        merged.groupby(["segment_grade", "fill_type", "length_bucket"], dropna=False)
        .agg(
            segment_count=("fill_id", "size"),
            abnormal_ratio=("shape_abnormal_flag", "mean"),
            overshoot_ratio=("overshoot_flag", "mean"),
            edge_spike_ratio=("edge_spike_flag", "mean"),
            keep_ratio=("quality_class", lambda s: float((s == "keep").mean())),
            warn_ratio=("quality_class", lambda s: float((s == "warn").mean())),
            abnormal_quality_ratio=("quality_class", lambda s: float((s == "abnormal").mean())),
            mean_overshoot_up=("overshoot_up", "mean"),
            mean_max_vertical_rate_inside=("max_vertical_rate_inside", "mean"),
        )
        .reset_index()
        .sort_values(["segment_grade", "segment_count"], ascending=[True, False])
    )
    summary.to_csv(OUT_ROOT / "tables" / "fill_segment_stats_by_grade.csv", index=False)

    overall = (
        merged.groupby("segment_grade")
        .agg(
            segment_count=("fill_id", "size"),
            ratio=("fill_id", lambda s: float(len(s) / len(merged))),
            abnormal_ratio=("shape_abnormal_flag", "mean"),
            overshoot_ratio=("overshoot_flag", "mean"),
            edge_spike_ratio=("edge_spike_flag", "mean"),
            mean_fill_minutes=("fill_minutes", "mean"),
        )
        .reset_index()
    )
    overall.to_csv(OUT_ROOT / "tables" / "fill_segment_grade_overall_summary.csv", index=False)
    return merged, summary, overall


def build_strategy_recommendation(sample_grade: pd.DataFrame, fill_overall: pd.DataFrame) -> pd.DataFrame:
    recommendations = [
        {
            "segment_class": "stable",
            "definition": "Low heading variation, low speed fluctuation, and low local altitude variance inside cruise windows.",
            "recommended_policy": "Use current model and default recovery pipeline.",
            "expected_action": "No residual intervention beyond existing default constraints.",
        },
        {
            "segment_class": "normal",
            "definition": "Intermediate cruise dynamics without strong disturbance indicators.",
            "recommended_policy": "Use current model with monitoring; keep existing residual policy.",
            "expected_action": "Preserve current recovery path and inspect only if boundary anomalies appear.",
        },
        {
            "segment_class": "disturbed",
            "definition": "High heading variation, high speed fluctuation, or high altitude variance within the cruise window.",
            "recommended_policy": "Prefer conservative residual correction or stronger post-check policy.",
            "expected_action": "Enable residual correction / fallback rather than using unrestricted output.",
        },
        {
            "segment_class": "boundary",
            "definition": "Samples with few anchors or very long gaps near the window boundaries.",
            "recommended_policy": "Mark for lower confidence or down-weight in later fill-segment policy design.",
            "expected_action": "Treat as boundary-sensitive segments in the inference policy table.",
        },
    ]
    out = pd.DataFrame(recommendations)
    out.to_csv(OUT_ROOT / "tables" / "fill_segment_policy_recommendation.csv", index=False)
    return out


def plot_grade_distribution(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(summary["segment_grade"], summary["sample_count"], color=["#4c78a8", "#f58518", "#e45756"])
    ax.set_title("Cruise-window grading distribution")
    ax.set_xlabel("Segment grade")
    ax.set_ylabel("Sample count")
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "figures" / "fig_cruise_segment_grade_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_gap_anchor_stats(stats: pd.DataFrame) -> None:
    pivot = stats[stats["region"].isin(["anchor_region", "gap_region", "boundary_gap_region"])].pivot(index="feature", columns="region", values="mean_abs")
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = np.arange(len(pivot.index))
    width = 0.24
    colors = ["#4c78a8", "#f58518", "#e45756"]
    for i, col in enumerate(pivot.columns):
        ax.bar(x + (i - 1) * width, pivot[col].values, width=width, label=col.replace("_", " "), color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel("Mean absolute value")
    ax.set_title("Anchor vs. gap-region dynamics within cruise samples")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "figures" / "fig_gap_anchor_stats.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_error_by_grade(err: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(err["segment_grade"], err["mean_gap_alt_rmse"], color=["#4c78a8", "#f58518", "#e45756"])
    ax.set_title("Gap altitude error by cruise-window grade")
    ax.set_xlabel("Segment grade")
    ax.set_ylabel("Mean gap altitude RMSE")
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "figures" / "fig_error_by_segment_grade.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_fill_abnormal_by_grade(fill_overall: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(fill_overall["segment_grade"], fill_overall["abnormal_ratio"], color=["#4c78a8", "#f58518", "#e45756"])
    ax.set_title("Recovered fill abnormal ratio by cruise-window grade")
    ax.set_xlabel("Segment grade")
    ax.set_ylabel("Abnormal ratio")
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "figures" / "fig_fill_abnormal_by_grade.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary(grade_summary: pd.DataFrame, error_summary: pd.DataFrame, fill_summary: pd.DataFrame) -> None:
    md = [
        "# Cruise-internal grading and fill-segment policy preparation",
        "",
        "This folder contains cruise-window grading results based on heading-rate, speed-delta, and local altitude variance, together with gap-anchor contrast statistics and fill-segment strategy hints.",
        "",
        "## Key outputs",
        "",
        "- cruise_segment_grade_summary.csv",
        "- gap_anchor_region_stats.csv",
        "- cruise_segment_error_by_grade.csv",
        "- fill_segment_stats_by_grade.csv",
        "- fill_segment_policy_recommendation.csv",
        "",
        "## Notes",
        "",
        "- `stable`: low internal disturbance, suitable for the current model path.",
        "- `disturbed`: higher internal dynamics, suggested to use residual correction / conservative fill policy.",
        "- `boundary`: not a mutually exclusive grade; it marks samples with boundary-sensitive geometry and should be down-weighted or flagged in later fill-segment policy design.",
        "",
    ]
    (OUT_ROOT / "README.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    samples = pd.read_parquet(SAMPLES_PATH)
    samples["minute_ts"] = pd.to_datetime(samples["minute_ts"], utc=True)

    sample_grade = build_sample_level_features(samples)
    grade_summary = build_grade_summary(sample_grade)
    gap_anchor_stats = build_gap_anchor_stats(samples)
    error_summary = build_error_by_grade(sample_grade)
    _, _, fill_overall = build_fill_segment_summary(sample_grade)
    build_strategy_recommendation(sample_grade, fill_overall)

    plot_grade_distribution(grade_summary)
    plot_gap_anchor_stats(gap_anchor_stats)
    plot_error_by_grade(error_summary)
    plot_fill_abnormal_by_grade(fill_overall)
    write_summary(grade_summary, error_summary, fill_overall)
    print(f"[ok] outputs written to {OUT_ROOT}")


if __name__ == "__main__":
    main()
