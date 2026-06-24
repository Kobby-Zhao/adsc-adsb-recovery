from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/home/jj/workspace/data-0313")
DATA_ROOT = ROOT / "outputs" / "mvp_adsb3864_20260322"
OUT_ROOT = ROOT / "outputs" / "runs" / "paper_dataset_figures_20260406"


@dataclass
class CruiseThresholds:
    min_cruise_minutes: int = 30
    max_abs_vertical_rate: float = 300.0
    max_speed_delta: float = 30.0
    max_heading_rate: float = 5.0


def ensure_dirs() -> None:
    (OUT_ROOT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "tables").mkdir(parents=True, exist_ok=True)


def circular_delta_deg(series: pd.Series) -> pd.Series:
    a = pd.to_numeric(series, errors="coerce")
    return (a.diff() + 180.0) % 360.0 - 180.0


def add_full_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["flight_id", "minute_ts"]).copy()
    dt_min = out.groupby("flight_id")["minute_ts"].diff().dt.total_seconds().div(60.0)
    dt_min = dt_min.replace(0.0, np.nan)
    out["dt_min"] = dt_min
    out["vertical_speed_calc"] = out.groupby("flight_id")["alt"].diff().div(dt_min)
    out["speed_delta_calc"] = out.groupby("flight_id")["speed"].diff().abs().div(dt_min)
    out["heading_delta_signed"] = out.groupby("flight_id")["heading"].transform(circular_delta_deg)
    out["heading_rate_calc"] = out["heading_delta_signed"].abs().div(dt_min)
    return out


def contiguous_segment_lengths(df: pd.DataFrame, mask_col: str) -> pd.Series:
    rows: list[int] = []
    for _, g in df.sort_values(["flight_id", "minute_ts"]).groupby("flight_id"):
        if g.empty:
            continue
        gaps = g["minute_ts"].diff().dt.total_seconds().div(60.0).fillna(1.0)
        block = ((g[mask_col] != g[mask_col].shift()) | (gaps > 1.0)).cumsum()
        tmp = g.assign(_block=block)
        kept = tmp[tmp[mask_col].eq(1)]
        if kept.empty:
            continue
        rows.extend(kept.groupby("_block").size().tolist())
    return pd.Series(rows, dtype=float)


def summarize_distribution(x: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return {k: np.nan for k in ["mean", "std", "p50", "p90"]}
    return {
        "mean": float(x.mean()),
        "std": float(x.std(ddof=0)),
        "p50": float(x.quantile(0.5)),
        "p90": float(x.quantile(0.9)),
    }


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    adsb_minute = pd.read_parquet(DATA_ROOT / "adsb_minute.parquet")
    adsb_cruise = pd.read_parquet(DATA_ROOT / "adsb_cruise.parquet")
    samples = pd.read_parquet(DATA_ROOT / "samples.parquet")
    obs_sim = json.loads((DATA_ROOT / "obs_sim_audit.json").read_text())
    for df, time_col in [
        (adsb_minute, "minute_ts"),
        (adsb_cruise, "minute_ts"),
        (samples, "minute_ts"),
    ]:
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
    return adsb_minute, adsb_cruise, samples, obs_sim


def build_cruise_rule_table(th: CruiseThresholds) -> pd.DataFrame:
    rows = [
        {
            "rule_name": "Vertical-rate stability",
            "feature": "Absolute vertical speed",
            "criterion": f"<= {th.max_abs_vertical_rate:.0f} m/min",
            "physical_meaning": "Exclude climb/descent segments and retain height-holding flight states.",
            "task_relevance": "Matches the weak vertical motion characteristic of oceanic cruise.",
        },
        {
            "rule_name": "Speed-change stability",
            "feature": "Per-minute speed variation",
            "criterion": f"<= {th.max_speed_delta:.0f} m/s/min",
            "physical_meaning": "Exclude acceleration/deceleration dominated phases near takeoff and arrival.",
            "task_relevance": "Keeps segments with relatively stable cruise power settings.",
        },
        {
            "rule_name": "Heading-change stability",
            "feature": "Per-minute heading rate",
            "criterion": f"<= {th.max_heading_rate:.1f} deg/min",
            "physical_meaning": "Exclude turning-intensive procedures and retain route-following motion.",
            "task_relevance": "Makes planar dynamics consistent with long-haul cruise trajectories.",
        },
        {
            "rule_name": "Minimum duration",
            "feature": "Continuous stable run length",
            "criterion": f">= {th.min_cruise_minutes:d} min",
            "physical_meaning": "Avoids treating isolated stable points as cruise phase.",
            "task_relevance": "Ensures the retained segment is long enough to construct recovery windows.",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_ROOT / "tables" / "table_cruise_filter_rules.csv", index=False)
    return out


def build_cruise_before_after_table(adsb_minute: pd.DataFrame, adsb_cruise: pd.DataFrame) -> pd.DataFrame:
    full = add_full_dynamics(adsb_minute)
    cruise = adsb_cruise.copy()
    if "vertical_speed" not in cruise.columns:
        cruise = add_full_dynamics(cruise).rename(
            columns={
                "vertical_speed_calc": "vertical_speed",
                "speed_delta_calc": "speed_delta",
                "heading_rate_calc": "heading_rate",
            }
        )

    full_seg = contiguous_segment_lengths(full.assign(is_all=1), "is_all")
    cruise_seg = contiguous_segment_lengths(cruise, "is_cruise")

    rows = []
    for name, df, seg in [
        ("Before cruise filtering", full, full_seg),
        ("After cruise filtering", cruise, cruise_seg),
    ]:
        rows.append(
            {
                "stage": name,
                "rows": int(len(df)),
                "unique_flights": int(df["flight_id"].nunique()),
                "mean_altitude_m": float(df["alt"].mean()),
                "std_altitude_m": float(df["alt"].std(ddof=0)),
                "mean_abs_vertical_speed_mpm": float(df.filter(regex="vertical_speed").iloc[:, 0].abs().mean()),
                "p90_abs_vertical_speed_mpm": float(df.filter(regex="vertical_speed").iloc[:, 0].abs().quantile(0.9)),
                "mean_speed_delta": float(df.filter(regex="speed_delta").iloc[:, 0].mean()),
                "p90_speed_delta": float(df.filter(regex="speed_delta").iloc[:, 0].quantile(0.9)),
                "mean_heading_rate_degpm": float(df.filter(regex="heading_rate").iloc[:, 0].mean()),
                "p90_heading_rate_degpm": float(df.filter(regex="heading_rate").iloc[:, 0].quantile(0.9)),
                "median_contiguous_duration_min": float(seg.quantile(0.5) if not seg.empty else np.nan),
                "p90_contiguous_duration_min": float(seg.quantile(0.9) if not seg.empty else np.nan),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_ROOT / "tables" / "table_cruise_before_after_stats.csv", index=False)
    return out


def choose_representative_flight(adsb_minute: pd.DataFrame, adsb_cruise: pd.DataFrame) -> str:
    cruise_counts = adsb_cruise.groupby("flight_id").size().rename("cruise_rows")
    full_counts = adsb_minute.groupby("flight_id").size().rename("full_rows")
    both = pd.concat([full_counts, cruise_counts], axis=1).fillna(0.0)
    both["ratio"] = both["cruise_rows"] / both["full_rows"].clip(lower=1)
    both = both[(both["cruise_rows"] >= 30) & (both["full_rows"] >= 80)]
    if both.empty:
        return adsb_cruise["flight_id"].iloc[0]
    return both.sort_values(["cruise_rows", "ratio"], ascending=False).index[0]


def plot_cruise_example(adsb_minute: pd.DataFrame, adsb_cruise: pd.DataFrame) -> None:
    flight_id = choose_representative_flight(adsb_minute, adsb_cruise)
    full = adsb_minute[adsb_minute["flight_id"].eq(flight_id)].sort_values("minute_ts").copy()
    cruise = adsb_cruise[adsb_cruise["flight_id"].eq(flight_id)].sort_values("minute_ts").copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].plot(full["minute_ts"], full["alt"], color="#b0b0b0", lw=1.3, label="Full flight profile")
    axes[0].plot(cruise["minute_ts"], cruise["alt"], color="#d55e00", lw=2.0, label="Retained cruise segment")
    axes[0].set_title("Altitude profile before and after cruise filtering")
    axes[0].set_xlabel("Time (UTC)")
    axes[0].set_ylabel("Altitude (m)")
    axes[0].legend(frameon=False)

    axes[1].plot(full["lon"], full["lat"], color="#b0b0b0", lw=1.2, label="Full trajectory")
    axes[1].plot(cruise["lon"], cruise["lat"], color="#0072b2", lw=2.0, label="Cruise-only trajectory")
    axes[1].set_title("Planar trajectory after cruise extraction")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    axes[1].legend(frameon=False)

    fig.suptitle(f"Representative flight: {flight_id}", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "figures" / "fig_cruise_filter_example.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cruise_distribution_compare(adsb_minute: pd.DataFrame, adsb_cruise: pd.DataFrame) -> None:
    full = add_full_dynamics(adsb_minute)
    cruise = adsb_cruise.copy()

    metrics = [
        ("vertical_speed_calc", "Absolute vertical speed (m/min)"),
        ("speed_delta_calc", "Speed change per minute"),
        ("heading_rate_calc", "Heading rate (deg/min)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    for ax, (col, title) in zip(axes, metrics):
        full_vals = full[col].abs().replace([np.inf, -np.inf], np.nan).dropna()
        cruise_col = col.replace("_calc", "")
        cruise_vals = cruise[cruise_col].abs().replace([np.inf, -np.inf], np.nan).dropna()
        hi = np.nanquantile(pd.concat([full_vals, cruise_vals]), 0.98)
        bins = np.linspace(0, hi, 60)
        ax.hist(full_vals.clip(upper=hi), bins=bins, alpha=0.45, color="#999999", label="Before filtering", density=True)
        ax.hist(cruise_vals.clip(upper=hi), bins=bins, alpha=0.60, color="#d55e00", label="After filtering", density=True)
        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "figures" / "fig_cruise_distribution_compare.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_sparse_stats_table(samples: pd.DataFrame, obs_sim: dict) -> pd.DataFrame:
    per_sample = (
        samples.groupby("sample_id")
        .agg(
            window_minutes=("minute_ts", "size"),
            observed_points=("obs_mask", "sum"),
            observed_ratio=("obs_mask", "mean"),
            anchor_points=("is_anchor", "sum"),
            max_gap_len=("gap_len", "max"),
            mean_dt_prev=("dt_prev", "mean"),
            mean_dt_next=("dt_next", "mean"),
        )
        .reset_index()
    )
    rows = [
        {
            "data_form": "Continuous minute-level reference",
            "rows_or_windows": int(len(samples)),
            "mean_window_minutes": float(per_sample["window_minutes"].mean()),
            "mean_observed_points": float(per_sample["window_minutes"].mean()),
            "mean_observed_ratio": 1.0,
            "mean_anchor_points": float(per_sample["window_minutes"].mean()),
            "mean_max_gap_len": 0.0,
            "mean_observation_interval_min": 1.0,
            "p90_gap_len": 0.0,
        },
        {
            "data_form": "Sparse recovery samples",
            "rows_or_windows": int(per_sample.shape[0]),
            "mean_window_minutes": float(per_sample["window_minutes"].mean()),
            "mean_observed_points": float(per_sample["observed_points"].mean()),
            "mean_observed_ratio": float(per_sample["observed_ratio"].mean()),
            "mean_anchor_points": float(per_sample["anchor_points"].mean()),
            "mean_max_gap_len": float(per_sample["max_gap_len"].mean()),
            "mean_observation_interval_min": float((per_sample["mean_dt_prev"] + per_sample["mean_dt_next"]).mean() / 2.0),
            "p90_gap_len": float(per_sample["max_gap_len"].quantile(0.9)),
        },
    ]
    out = pd.DataFrame(rows)
    out["audit_missing_ratio_mean"] = [np.nan, obs_sim.get("sample_missing_ratio_mean")]
    out["audit_gap_length_mean"] = [np.nan, obs_sim.get("gap_length_mean")]
    out.to_csv(OUT_ROOT / "tables" / "table_sparse_recovery_stats.csv", index=False)
    return per_sample, out


def choose_representative_sample(per_sample: pd.DataFrame) -> str:
    target_gap = float(per_sample["max_gap_len"].quantile(0.9))
    target_ratio = float(per_sample["observed_ratio"].mean())
    cand = per_sample.copy()
    cand["score"] = (cand["max_gap_len"] - target_gap).abs() + 20.0 * (cand["observed_ratio"] - target_ratio).abs()
    cand = cand[(cand["window_minutes"] >= 60) & (cand["anchor_points"] >= 2)]
    return cand.sort_values("score").iloc[0]["sample_id"]


def plot_sparse_example(samples: pd.DataFrame, per_sample: pd.DataFrame) -> None:
    sample_id = choose_representative_sample(per_sample)
    s = samples[samples["sample_id"].eq(sample_id)].sort_values("minute_ts").copy()
    obs = s[s["obs_mask"].eq(1)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(s["lon"], s["lat"], color="#b0b0b0", lw=1.5, label="Minute-level reference")
    axes[0].scatter(obs["lon"], obs["lat"], s=20, color="#0072b2", label="Retained sparse observations")
    anchors = s[s["is_anchor"].eq(1)]
    if not anchors.empty:
        axes[0].scatter(anchors["lon"], anchors["lat"], s=34, color="#d55e00", marker="x", label="Boundary anchors")
    axes[0].set_title("Planar trajectory before and after sparse sampling")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].legend(frameon=False, loc="best")

    axes[1].plot(s["minute_ts"], s["alt"], color="#b0b0b0", lw=1.5, label="Minute-level reference")
    axes[1].scatter(obs["minute_ts"], obs["alt"], s=18, color="#0072b2", label="Retained sparse observations")
    if not anchors.empty:
        axes[1].scatter(anchors["minute_ts"], anchors["alt"], s=34, color="#d55e00", marker="x", label="Boundary anchors")
    axes[1].set_title("Altitude profile under sparse observation")
    axes[1].set_xlabel("Time (UTC)")
    axes[1].set_ylabel("Altitude (m)")
    axes[1].legend(frameon=False, loc="best")

    fig.suptitle(f"Representative sparse recovery sample: {sample_id}", y=1.03, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "figures" / "fig_sparse_recovery_example.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_gap_and_interval_distributions(samples: pd.DataFrame) -> None:
    gap = samples.loc[samples["obs_mask"].eq(0), "gap_len"].replace(0, np.nan).dropna()
    interval = pd.concat(
        [
            samples.loc[samples["obs_mask"].eq(0), "dt_prev"],
            samples.loc[samples["obs_mask"].eq(0), "dt_next"],
        ],
        ignore_index=True,
    ).replace(0, np.nan).dropna()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].hist(gap.clip(upper=gap.quantile(0.98)), bins=40, color="#d55e00", alpha=0.8)
    axes[0].set_title("Distribution of missing-segment lengths")
    axes[0].set_xlabel("Gap length (minutes)")
    axes[0].set_ylabel("Count")

    axes[1].hist(interval.clip(upper=interval.quantile(0.98)), bins=40, color="#0072b2", alpha=0.8)
    axes[1].set_title("Distribution of observation intervals")
    axes[1].set_xlabel("Interval to nearest observation (minutes)")
    axes[1].set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(OUT_ROOT / "figures" / "fig_gap_and_interval_distributions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_dataset_section_summary(cruise_stats: pd.DataFrame, sparse_stats: pd.DataFrame, obs_sim: dict) -> None:
    md = [
        "# Dataset Construction Figures",
        "",
        "This directory contains the paper-oriented tables and figures for the cruise-stage filtering and sparse recovery sample construction section.",
        "",
        "## Key numerical highlights",
        "",
        f"- Mean simulated missing ratio: {obs_sim.get('sample_missing_ratio_mean', float('nan')):.4f}",
        f"- Mean missing-segment length: {obs_sim.get('gap_length_mean', float('nan')):.2f} min",
        f"- 90th percentile missing-segment length: {obs_sim.get('gap_length_q90', float('nan')):.2f} min",
        "",
        "## Generated tables",
        "",
        "- table_cruise_filter_rules.csv",
        "- table_cruise_before_after_stats.csv",
        "- table_sparse_recovery_stats.csv",
        "",
        "## Generated figures",
        "",
        "- fig_cruise_filter_example.png",
        "- fig_cruise_distribution_compare.png",
        "- fig_sparse_recovery_example.png",
        "- fig_gap_and_interval_distributions.png",
        "",
    ]
    (OUT_ROOT / "README.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    thresholds = CruiseThresholds()
    adsb_minute, adsb_cruise, samples, obs_sim = load_data()
    build_cruise_rule_table(thresholds)
    cruise_stats = build_cruise_before_after_table(adsb_minute, adsb_cruise)
    plot_cruise_example(adsb_minute, adsb_cruise)
    plot_cruise_distribution_compare(adsb_minute, adsb_cruise)
    per_sample, sparse_stats = build_sparse_stats_table(samples, obs_sim)
    plot_sparse_example(samples, per_sample)
    plot_gap_and_interval_distributions(samples)
    build_dataset_section_summary(cruise_stats, sparse_stats, obs_sim)
    print(f"[ok] outputs written to {OUT_ROOT}")


if __name__ == "__main__":
    main()
