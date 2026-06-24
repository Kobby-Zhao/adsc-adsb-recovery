#!/usr/bin/env python3
"""
Naive interpolation baselines for trajectory recovery.

Evaluates two standard interpolation methods that use only the sparse
anchor-point information (no learning, no trainable parameters):

  - Piecewise linear interpolation (per-gap boundary anchors)
  - Cubic spline interpolation (all anchors within a sample)

Both methods are evaluated on the same test split used by all deep models,
with metrics reported separately for all-time and gap-only subsets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _load_test_split(
    samples_path: str,
    flight_id_col: str = "flight_id",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load the test split, replicating the same flight_id-based grouping
    used by all model training (seed=42, 0.8/0.1/0.1).
    """
    df = pd.read_parquet(samples_path)
    flights = sorted(df[flight_id_col].astype(str).unique().tolist())
    rng = np.random.default_rng(seed)
    rng.shuffle(flights)

    n = len(flights)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    test_flights = set(flights[n_train + n_val:])

    df_test = df[df[flight_id_col].astype(str).isin(test_flights)].copy()
    return df_test


def _interp_linear_gapwise(
    sample_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-gap piecewise linear interpolation.

    For each contiguous gap (obs_mask==0), interpolate between the
    immediately preceding and following anchor values.
    """
    n = len(sample_df)
    mask = sample_df["obs_mask"].values.astype(bool)
    lat_true = sample_df["lat"].values.astype(np.float64)
    lon_true = sample_df["lon"].values.astype(np.float64)
    alt_true = sample_df["alt"].values.astype(np.float64)

    lat_pred = lat_true.copy()
    lon_pred = lon_true.copy()
    alt_pred = alt_true.copy()

    # Find gap boundaries
    anchor_idx = np.where(mask)[0]
    if len(anchor_idx) < 2:
        return lat_pred, lon_pred, alt_pred

    for i in range(len(anchor_idx) - 1):
        left = anchor_idx[i]
        right = anchor_idx[i + 1]
        gap_len = right - left - 1
        if gap_len <= 0:
            continue

        gap_range = np.arange(left + 1, right)
        frac = (gap_range - left) / (right - left)

        lat_pred[gap_range] = lat_true[left] + frac * (lat_true[right] - lat_true[left])
        lon_pred[gap_range] = lon_true[left] + frac * (lon_true[right] - lon_true[left])
        alt_pred[gap_range] = alt_true[left] + frac * (alt_true[right] - alt_true[left])

    return lat_pred, lon_pred, alt_pred


def _interp_cubic_spline(
    sample_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cubic spline interpolation through all anchor points within a sample.
    Falls back to linear interpolation when there are fewer than 4 anchors.
    """
    n = len(sample_df)
    mask = sample_df["obs_mask"].values.astype(bool)
    anchor_idx = np.where(mask)[0]

    lat_true = sample_df["lat"].values.astype(np.float64)
    lon_true = sample_df["lon"].values.astype(np.float64)
    alt_true = sample_df["alt"].values.astype(np.float64)

    t_all = np.arange(n, dtype=np.float64)
    try:
        from scipy.interpolate import CubicSpline
    except Exception as e:
        raise RuntimeError("Cubic spline interpolation requires scipy to be installed.") from e

    if len(anchor_idx) < 4:
        # Fallback: linear interpolation
        return _interp_linear_gapwise(sample_df)

    t_anchor = t_all[anchor_idx]

    def _spline_or_linear(x_anchor: np.ndarray, raw: np.ndarray) -> np.ndarray:
        try:
            cs = CubicSpline(t_anchor, x_anchor, extrapolate=True)
            pred = cs(t_all)
            # Clamp extrapolated values to observed anchor range
            lo, hi = x_anchor.min(), x_anchor.max()
            return np.clip(pred, lo, hi)
        except Exception:
            # Degenerate spline → fallback to linear
            lat_p, lon_p, alt_p = _interp_linear_gapwise(sample_df)
            if x_anchor is lat_true[anchor_idx]:
                return lat_p
            elif x_anchor is lon_true[anchor_idx]:
                return lon_p
            else:
                return alt_p

    lat_pred = _spline_or_linear(lat_true[anchor_idx].astype(np.float64), lat_true)
    lon_pred = _spline_or_linear(lon_true[anchor_idx].astype(np.float64), lon_true)
    alt_pred = _spline_or_linear(alt_true[anchor_idx].astype(np.float64), alt_true)

    return lat_pred, lon_pred, alt_pred


def _compute_metrics(
    true: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    """Compute RMSE and MAE for a single dimension."""
    err = np.abs(true - pred)
    sq_err = err ** 2

    # All-time metrics
    mae_all = float(np.mean(err))
    rmse_all = float(np.sqrt(np.mean(sq_err)))

    # Gap-only metrics
    gap = ~mask.astype(bool)
    mae_gap = float(np.mean(err[gap])) if gap.any() else float("nan")
    rmse_gap = float(np.sqrt(np.mean(sq_err[gap]))) if gap.any() else float("nan")

    return {
        "mae": mae_all,
        "rmse": rmse_all,
        "gap_mae": mae_gap,
        "gap_rmse": rmse_gap,
    }


def evaluate(
    df: pd.DataFrame,
    method: str = "linear",
) -> Dict[str, float]:
    """
    Evaluate interpolation method on the full test set.

    Returns aggregate metrics for lat, lon, alt.
    """
    interp_fn = _interp_linear_gapwise if method == "linear" else _interp_cubic_spline

    sample_ids = sorted(df["sample_id"].unique())
    all_lat_true, all_lon_true, all_alt_true = [], [], []
    all_lat_pred, all_lon_pred, all_alt_pred = [], [], []
    all_mask = []

    for sid in sample_ids:
        sdf = df[df["sample_id"] == sid].sort_values("minute_ts")
        if len(sdf) < 3:
            continue

        lat_pred, lon_pred, alt_pred = interp_fn(sdf)
        mask = sdf["obs_mask"].values.astype(bool)

        all_lat_true.append(sdf["lat"].values.astype(np.float64))
        all_lon_true.append(sdf["lon"].values.astype(np.float64))
        all_alt_true.append(sdf["alt"].values.astype(np.float64))
        all_lat_pred.append(lat_pred)
        all_lon_pred.append(lon_pred)
        all_alt_pred.append(alt_pred)
        all_mask.append(mask)

    lat_true = np.concatenate(all_lat_true)
    lon_true = np.concatenate(all_lon_true)
    alt_true = np.concatenate(all_alt_true)
    lat_pred = np.concatenate(all_lat_pred)
    lon_pred = np.concatenate(all_lon_pred)
    alt_pred = np.concatenate(all_alt_pred)
    mask = np.concatenate(all_mask)

    lat_m = _compute_metrics(lat_true, lat_pred, mask)
    lon_m = _compute_metrics(lon_true, lon_pred, mask)
    alt_m = _compute_metrics(alt_true, alt_pred, mask)

    return {
        # Lat (note: paper may use lat=dim0, lon=dim1)
        "lat_rmse": lat_m["rmse"],
        "lat_mae": lat_m["mae"],
        "gap_lat_rmse": lat_m["gap_rmse"],
        "gap_lat_mae": lat_m["gap_mae"],
        # Lon
        "lon_rmse": lon_m["rmse"],
        "lon_mae": lon_m["mae"],
        "gap_lon_rmse": lon_m["gap_rmse"],
        "gap_lon_mae": lon_m["gap_mae"],
        # Alt
        "alt_rmse": alt_m["rmse"],
        "alt_mae": alt_m["mae"],
        "gap_alt_rmse": alt_m["gap_rmse"],
        "gap_alt_mae": alt_m["gap_mae"],
    }


def _load_model_metrics(model_csv: Path) -> Dict[str, float]:
    """Aggregate per-sample model metrics into dataset-level RMSE/MAE."""
    df = pd.read_csv(model_csv)
    metrics = {}
    # All-time metrics
    for dim, name in [("lat", "lat"), ("lon", "lon"), ("alt", "alt")]:
        col_rmse = f"{name}_rmse"
        col_mae = f"{name}_mae"
        if col_rmse in df.columns:
            # RMSE of RMSEs isn't valid; recompute from per-sample values
            # For approximate comparison, use mean if per-sample has >=1 gap point
            metrics[f"{dim}_rmse"] = float(np.sqrt((df[col_rmse] ** 2).mean()))
            metrics[f"{dim}_mae"] = float(df[col_mae].mean())
        gap_rmse = f"gap_{name}_rmse"
        gap_mae = f"gap_{name}_mae"
        if gap_rmse in df.columns:
            metrics[f"gap_{dim}_rmse"] = float(
                np.sqrt((df[gap_rmse] ** 2).mean())
            )
            metrics[f"gap_{dim}_mae"] = float(df[gap_mae].mean())
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate interpolation baselines on ADS-B test split"
    )
    parser.add_argument(
        "--samples",
        default="outputs/mvp_merged_nostage_20260415/stage_datasets_20260415_s2v2/stage3/samples.parquet",
        help="Path to the parquet file containing all samples (with splits).",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8,
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1,
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--model-csv",
        default="",
        help="Optional: OurMethod per-sample metrics CSV for side-by-side comparison.",
    )
    parser.add_argument(
        "--out-json",
        default="outputs/experiments/interpolation_baselines/metrics.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--method",
        choices=["linear", "cubic", "both"],
        default="both",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    samples_path = root / args.samples
    out_json = root / args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading test split from {samples_path} ...")
    df_test = _load_test_split(
        str(samples_path),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    n_samples = df_test["sample_id"].nunique()
    n_rows = len(df_test)
    n_anchors = int(df_test["obs_mask"].sum())
    n_gap = int(n_rows - n_anchors)
    print(f"      {n_samples} samples, {n_rows} rows, "
          f"{n_anchors} anchors, {n_gap} gap points")

    results: Dict[str, dict] = {}

    if args.method in ("linear", "both"):
        print("[2/4] Evaluating piecewise linear interpolation ...")
        metrics_lin = evaluate(df_test, method="linear")
        results["PiecewiseLinear"] = metrics_lin
        print(f"      gap_lat_rmse={metrics_lin['gap_lat_rmse']:.4f}  "
              f"gap_lon_rmse={metrics_lin['gap_lon_rmse']:.4f}  "
              f"gap_alt_rmse={metrics_lin['gap_alt_rmse']:.4f}")

    if args.method in ("cubic", "both"):
        print("[3/4] Evaluating cubic spline interpolation ...")
        metrics_cubic = evaluate(df_test, method="cubic")
        results["CubicSpline"] = metrics_cubic
        print(f"      gap_lat_rmse={metrics_cubic['gap_lat_rmse']:.4f}  "
              f"gap_lon_rmse={metrics_cubic['gap_lon_rmse']:.4f}  "
              f"gap_alt_rmse={metrics_cubic['gap_alt_rmse']:.4f}")

    # Load model metrics for comparison (optional)
    model_csv = root / args.model_csv if args.model_csv else None
    if model_csv and model_csv.exists():
        print(f"[4/4] Loading model metrics for comparison ...")
        model_metrics = _load_model_metrics(model_csv)
        results["OurMethod_ref"] = model_metrics
        print(f"      gap_lat_rmse={model_metrics['gap_lat_rmse']:.4f}  "
              f"gap_lon_rmse={model_metrics['gap_lon_rmse']:.4f}  "
              f"gap_alt_rmse={model_metrics['gap_alt_rmse']:.4f}")
    else:
        print(f"[4/4] Skipped model comparison (no --model-csv provided).")

    # Save
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n[ok] results saved to {out_json}")

    # Print comparison table
    if len(results) >= 2:
        print("\n" + "=" * 75)
        print("INTERPOLATION BASELINES vs PROPOSED METHOD")
        print("=" * 75)
        header = f"{'Method':<22} {'gap_lat_rmse':>14} {'gap_lon_rmse':>14} {'gap_alt_rmse':>14}"
        print(header)
        print("-" * 75)
        for name, m in results.items():
            print(
                f"{name:<22} "
                f"{m.get('gap_lat_rmse', float('nan')):>14.4f} "
                f"{m.get('gap_lon_rmse', float('nan')):>14.4f} "
                f"{m.get('gap_alt_rmse', float('nan')):>14.4f}"
            )
        print("=" * 75)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
