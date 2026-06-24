"""
Curriculum Distribution Analysis
S1 / S2_new (stratified) / S3 / test  — statistics, histograms, boxplots, CSV export.

Usage:
  PYTHONNOUSERSITE=1 python scripts/analyze_curriculum_distribution.py
  PYTHONNOUSERSITE=1 python scripts/analyze_curriculum_distribution.py --out-dir outputs/analysis/curriculum_dist
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_curriculum_stage_distributions_20260528 import (
    _sample_stats_from_stage_frame,
    _summarize,
)

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

S1_PATH = Path("outputs/mvp_merged_250_20260514_clean/stage1_clean/samples.parquet")
S2_PATH = Path("outputs/mvp_merged_250_20260514_clean/stage2_clean/samples.parquet")
S3_PATH = Path("outputs/mvp_merged_250_20260514_clean/stage3_clean/samples.parquet")

LG, FA = 20, 6


def _frozen_s2_new(s2_s: pd.DataFrame, s3_s: pd.DataFrame) -> pd.DataFrame:
    """Frozen stratified S2_new definition."""
    s2_hard = s2_s[(s2_s["gap_len_max"] >= 10) & (s2_s["anchor_count"] >= 8)].copy()
    s3_easy = s3_s[(s3_s["gap_len_max"].between(10, 35, inclusive="both")) & (s3_s["anchor_count"] >= 4)].copy()
    return pd.concat([s2_hard.assign(pool="S2"), s3_easy.assign(pool="S3")], ignore_index=True)


def _build_datasets() -> dict[str, pd.DataFrame]:
    """Load parquet, compute per-sample stats, build S2_new, return dict."""
    s1_df = pd.read_parquet(S1_PATH)
    s2_df = pd.read_parquet(S2_PATH)
    s3_df = pd.read_parquet(S3_PATH)

    s1_s = _sample_stats_from_stage_frame(s1_df)
    s2_s = _sample_stats_from_stage_frame(s2_df)
    s3_s = _sample_stats_from_stage_frame(s3_df)

    # S2_new: frozen stratified
    s2_new_s = _frozen_s2_new(s2_s, s3_s)

    # test: S3-based split from base config
    from src.training import load_config, split_by_flight_id
    base_cfg = load_config(
        "outputs/experiments/obs_conditioned_gaponly/bimamba_xyaux_zlinear_24e_v1/"
        "configs/C_bimamba_context_xyaux_zlinear.yaml"
    )
    splits = split_by_flight_id(
        df=s3_df, flight_id_col=base_cfg["data"]["flight_id_col"],
        train_ratio=float(base_cfg["data"]["split"]["train_ratio"]),
        val_ratio=float(base_cfg["data"]["split"]["val_ratio"]),
        seed=int(base_cfg.get("seed", 42)),
    )
    test_ids = set(splits["test"][base_cfg["data"]["flight_id_col"]].astype(str).unique())
    test_s = s3_s[s3_s["flight_id"].isin(test_ids)].copy()

    return {"S1": s1_s, "S2_new": s2_new_s, "S3": s3_s, "test": test_s}


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

FIGSIZE = (14, 10)
COLORS = {"S1": "#2ecc71", "S2_new": "#3498db", "S3": "#e74c3c", "test": "#9b59b6"}
DISPLAY_NAMES = {"S1": "S1", "S2_new": "S2", "S3": "S3", "test": "test"}


def _plot_boxplots(datasets: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Save separate English boxplots for key metrics."""
    metrics = [
        ("missing_ratio", "Missing Ratio", "box_missing_ratio_zh.png"),
        ("gap_len_max", "Max Gap Length (min)", "box_gap_len_max_zh.png"),
        ("gap_len_mean", "Mean Gap Length (min)", "box_gap_len_mean_zh.png"),
        ("anchor_count", "Anchor Count", "box_anchor_count_zh.png"),
    ]
    labels = [DISPLAY_NAMES.get(name, name) for name in datasets.keys()]
    for col, ylabel, filename in metrics:
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        data = [datasets[name][col].dropna().values for name in datasets]
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
        for patch, name in zip(bp["boxes"], datasets.keys()):
            patch.set_facecolor(COLORS[name])
            patch.set_alpha(0.7)
        ax.set_xlabel("Dataset Stage")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180)
        plt.close(fig)
        print(f"  [ok] {out_dir / filename}")


def _plot_histograms(datasets: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Overlaid histograms for key metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics = [
        ("anchor_count", "Anchor Count", (0, 80), 40),
        ("missing_ratio", "Missing Ratio", (0, 1), 40),
        ("gap_len_mean", "Mean Gap Length (min)", (0, 40), 40),
        ("gap_len_max", "Max Gap Length (min)", (0, 80), 40),
        ("delta_z_abs_m", "|Delta Z| (m)", (0, 1200), 50),
        ("gap_len_max", "Max Gap Length — log scale", (1, 200), 50),
    ]
    for ax, (col, title, xlim, bins) in zip(axes.flat, metrics):
        for name in datasets:
            vals = datasets[name][col].dropna().values
            if "log" in title.lower():
                vals = vals[vals > 0]
            ax.hist(vals, bins=bins, alpha=0.4, label=name, color=COLORS[name],
                    density=True, range=xlim if "log" not in title.lower() else None)
        ax.set_xlabel(title.split(" — ")[0])
        ax.set_ylabel("Density")
        ax.set_title(title)
        if "log" in title.lower():
            ax.set_xscale("log")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes.flat[-1].set_visible(False)  # hide extra subplot
    fig.suptitle("Curriculum Stage Distributions — Histograms", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "histograms.png", dpi=150)
    plt.close(fig)
    print(f"  [ok] {out_dir / 'histograms.png'}")


def _plot_hard_bucket_bars(datasets: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Grouped bar chart of hard-bucket ratios."""
    ratios = {}
    for name, df in datasets.items():
        x_anc = df["anchor_count"].values
        x_gap = df["gap_len_max"].values
        x_dz = df["delta_z_abs_m"].values
        ratios[name] = {
            "long_gap": (x_gap >= LG).mean(),
            "few_anchor": (x_anc <= FA).mean(),
            "dz>300": (x_dz > 300).mean(),
            "long_gap & dz>300": ((x_gap >= LG) & (x_dz > 300)).mean(),
            "few_anchor & dz>300": ((x_anc <= FA) & (x_dz > 300)).mean(),
        }

    buckets = list(next(iter(ratios.values())).keys())
    x = np.arange(len(buckets))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (name, r) in enumerate(ratios.items()):
        vals = [r[b] for b in buckets]
        ax.bar(x + i * width, vals, width, label=name, color=COLORS[name], alpha=0.8)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(buckets, rotation=15)
    ax.set_ylabel("Ratio")
    ax.set_title("Hard-Bucket Coverage by Stage")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "hard_bucket_bars.png", dpi=150)
    plt.close(fig)
    print(f"  [ok] {out_dir / 'hard_bucket_bars.png'}")


def _plot_median_iqr_point_intervals(datasets: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Single figure: median point with interquartile interval."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [
        ("anchor_count", "Anchor Count"),
        ("missing_ratio", "Missing Ratio"),
        ("gap_len_mean", "Mean Gap Length (min)"),
        ("gap_len_max", "Max Gap Length (min)"),
    ]
    stage_names = list(datasets.keys())
    display_names = [DISPLAY_NAMES.get(name, name) for name in stage_names]
    x = np.arange(len(stage_names))

    for ax, (col, title) in zip(axes.flat, metrics):
        medians, q25s, q75s = [], [], []
        for name in stage_names:
            s = datasets[name][col].dropna()
            q25s.append(float(s.quantile(0.25)))
            medians.append(float(s.quantile(0.50)))
            q75s.append(float(s.quantile(0.75)))
        medians = np.array(medians, dtype=float)
        q25s = np.array(q25s, dtype=float)
        q75s = np.array(q75s, dtype=float)
        lower = medians - q25s
        upper = q75s - medians

        for i, name in enumerate(stage_names):
            ax.errorbar(
                x[i],
                medians[i],
                yerr=np.array([[lower[i]], [upper[i]]]),
                fmt="o",
                color=COLORS[name],
                ecolor=COLORS[name],
                elinewidth=2.0,
                capsize=5,
                markersize=7,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(display_names)
        ax.set_title(title)
        ax.set_ylabel("Value")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Median with Interquartile Range by Curriculum Stage", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "median_iqr_point_interval_en.png", dpi=180)
    plt.close(fig)
    print(f"  [ok] {out_dir / 'median_iqr_point_interval_en.png'}")


def _plot_gap_vs_dz_scatter(datasets: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Scatter: gap_len_max vs delta_z_abs_m, colored by stage."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, (name, df) in zip(axes, datasets.items()):
        ax.scatter(df["gap_len_max"], df["delta_z_abs_m"], alpha=0.3, s=4,
                   color=COLORS[name], edgecolors="none")
        ax.axvline(LG, color="gray", linestyle="--", alpha=0.5, label=f"long_gap≥{LG}")
        ax.axhline(300, color="red", linestyle="--", alpha=0.5, label="|dz|=300")
        ax.set_xlabel("Max Gap Length (min)")
        ax.set_ylabel("|Delta Z| (m)")
        ax.set_title(name)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("Gap Length vs Altitude Change", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "gap_vs_dz_scatter.png", dpi=150)
    plt.close(fig)
    print(f"  [ok] {out_dir / 'gap_vs_dz_scatter.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# Tables
# ═══════════════════════════════════════════════════════════════════════════

def _print_summary_table(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Print and return multi-metric summary table."""
    rows = []
    for name, df in datasets.items():
        r = _summarize(df, name, LG, FA)
        rows.append(r)
    df = pd.DataFrame(rows)

    # Select and order columns for display
    cols = [
        "dataset", "n_samples",
        "anchor_count_mean", "anchor_count_p50",
        "missing_ratio_mean", "missing_ratio_p50",
        "gap_len_max_mean", "gap_len_max_p75", "gap_len_max_p90",
        "long_gap_ratio", "few_anchor_ratio",
        "delta_z_abs_mean", "delta_z_abs_p75", "delta_z_abs_p90",
        "delta_z_gt100_ratio", "delta_z_gt300_ratio",
        "long_gap_large_dz300_ratio",
    ]
    disp = df[cols].copy()

    print("\n" + "=" * 120)
    print("SUMMARY TABLE")
    print("=" * 120)
    fmt_map = {
        "dataset": "{:<12s}",
        "n_samples": "{:>7d}",
        "anchor_count_mean": "{:>9.2f}", "anchor_count_p50": "{:>9.1f}",
        "missing_ratio_mean": "{:>9.3f}", "missing_ratio_p50": "{:>9.3f}",
        "gap_len_max_mean": "{:>9.1f}", "gap_len_max_p75": "{:>9.1f}", "gap_len_max_p90": "{:>9.1f}",
        "long_gap_ratio": "{:>9.3f}", "few_anchor_ratio": "{:>9.3f}",
        "delta_z_abs_mean": "{:>9.1f}", "delta_z_abs_p75": "{:>9.1f}", "delta_z_abs_p90": "{:>9.1f}",
        "delta_z_gt100_ratio": "{:>9.3f}", "delta_z_gt300_ratio": "{:>9.3f}",
        "long_gap_large_dz300_ratio": "{:>9.3f}",
    }
    # Header
    header_parts = []
    for c in cols:
        label = c.replace("_", "\n").replace("delta z", "dz").replace("ratio", "r").replace("mean", "m").replace("count", "cnt")
        header_parts.append(f"{label:>9s}" if c != "dataset" else f"{'dataset':<12s}")
    print(" ".join(header_parts))
    print("-" * 120)
    for _, row in disp.iterrows():
        parts = []
        for c in cols:
            v = row[c]
            if c == "dataset":
                parts.append(f"{v:<12s}")
            elif isinstance(v, float):
                parts.append(f"{v:>9.3f}" if "ratio" in c else f"{v:>9.1f}" if "mean" in c or "p" in c else f"{v:>9.1f}")
            else:
                parts.append(f"{v:>9d}")
        print(" ".join(parts))
    return df


def _print_hard_bucket_table(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Print hard-bucket coverage table."""
    rows = []
    for name, df in datasets.items():
        x_anc = df["anchor_count"].values
        x_gap = df["gap_len_max"].values
        x_dz = df["delta_z_abs_m"].values
        rows.append({
            "dataset": name,
            "n_samples": len(df),
            "long_gap_n": int((x_gap >= LG).sum()),
            "long_gap_ratio": float((x_gap >= LG).mean()),
            "few_anchor_n": int((x_anc <= FA).sum()),
            "few_anchor_ratio": float((x_anc <= FA).mean()),
            "dz_gt300_n": int((x_dz > 300).sum()),
            "dz_gt300_ratio": float((x_dz > 300).mean()),
            "lg_dz300_n": int(((x_gap >= LG) & (x_dz > 300)).sum()),
            "lg_dz300_ratio": float(((x_gap >= LG) & (x_dz > 300)).mean()),
        })
    df = pd.DataFrame(rows)
    print("\n" + "=" * 120)
    print("HARD-BUCKET COVERAGE TABLE")
    print("=" * 120)
    print(df.to_string(index=False))
    return df


def _print_percentile_table(datasets: dict[str, pd.DataFrame]) -> None:
    """Print P25/P50/P75/P90 for key metrics."""
    print("\n" + "=" * 120)
    print("PERCENTILE BREAKDOWN")
    print("=" * 120)
    for name, df in datasets.items():
        print(f"\n--- {name} (n={len(df)}) ---")
        for col, label in [
            ("anchor_count", "Anchor Count"),
            ("missing_ratio", "Missing Ratio"),
            ("gap_len_max", "Max Gap (min)"),
            ("delta_z_abs_m", "|Delta Z| (m)"),
        ]:
            vals = df[col].dropna()
            print(f"  {label:<20s}: mean={vals.mean():>8.2f}  "
                  f"p25={vals.quantile(0.25):>8.2f}  p50={vals.quantile(0.50):>8.2f}  "
                  f"p75={vals.quantile(0.75):>8.2f}  p90={vals.quantile(0.90):>8.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/analysis/curriculum_distribution")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Build datasets ─────────────────────────────────────────────────
    print("[1/5] Building datasets...")
    datasets = _build_datasets()
    for name, df in datasets.items():
        print(f"  {name}: {len(df)} samples")

    # ── 2. Print tables ───────────────────────────────────────────────────
    print("\n[2/5] Computing statistics...")
    summary_df = _print_summary_table(datasets)
    hard_df = _print_hard_bucket_table(datasets)
    _print_percentile_table(datasets)

    # ── 3. Generate plots ─────────────────────────────────────────────────
    print(f"\n[3/5] Generating plots -> {out_dir}/")
    _plot_boxplots(datasets, out_dir)
    _plot_median_iqr_point_intervals(datasets, out_dir)
    _plot_histograms(datasets, out_dir)
    _plot_hard_bucket_bars(datasets, out_dir)
    _plot_gap_vs_dz_scatter(datasets, out_dir)

    # ── 4. Save CSV ───────────────────────────────────────────────────────
    print("\n[4/5] Saving CSV files...")
    summary_df.to_csv(out_dir / "summary_table.csv", index=False)
    print(f"  [ok] {out_dir / 'summary_table.csv'}")
    hard_df.to_csv(out_dir / "hard_bucket_table.csv", index=False)
    print(f"  [ok] {out_dir / 'hard_bucket_table.csv'}")

    # Save per-sample stats for each stage
    for name, df in datasets.items():
        path = out_dir / f"{name}_per_sample_stats.csv"
        df.to_csv(path, index=False)
        print(f"  [ok] {path}")

    # ── 5. Print final summary ────────────────────────────────────────────
    print("\n[5/5]" + "=" * 120)
    print("FINAL SUMMARY")
    print("=" * 120)
    for _, row in summary_df.iterrows():
        name = row["dataset"]
        print(f"\n  {name} (n={int(row['n_samples'])}):")
        print(f"    anchor_count:  mean={row['anchor_count_mean']:.1f}  p50={row['anchor_count_p50']:.1f}")
        print(f"    missing_ratio: mean={row['missing_ratio_mean']:.3f}  p50={row['missing_ratio_p50']:.3f}")
        print(f"    gap_len_max:   mean={row['gap_len_max_mean']:.1f}  p75={row['gap_len_max_p75']:.1f}  p90={row['gap_len_max_p90']:.1f}")
        print(f"    long_gap: {row['long_gap_ratio']:.3f}  few_anchor: {row['few_anchor_ratio']:.3f}")
        print(f"    |dz|>300m: {row['delta_z_gt300_ratio']:.3f}  long_gap+|dz|>300: {row['long_gap_large_dz300_ratio']:.3f}")

    print(f"\n  All outputs saved to: {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
