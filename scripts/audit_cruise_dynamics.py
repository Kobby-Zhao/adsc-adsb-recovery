from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit cruise-stage dynamics statistics from minute-level ADS-B truth.")
    p.add_argument("--samples-path", required=True, help="Parquet path with minute-level samples.")
    p.add_argument("--output-dir", default="outputs/cruise_audit", help="Directory to save audit csv files.")
    p.add_argument("--flight-id-col", default="flight_id")
    p.add_argument("--time-col", default="minute_ts")
    p.add_argument("--max-abs-vertical-rate", type=float, default=300.0)
    p.add_argument("--max-speed-delta", type=float, default=30.0)
    p.add_argument("--max-heading-rate", type=float, default=5.0)
    return p


def _haversine_m(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6371000.0
    lat1r = np.deg2rad(lat1)
    lon1r = np.deg2rad(lon1)
    lat2r = np.deg2rad(lat2)
    lon2r = np.deg2rad(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return r * c


def _heading_deg(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lat1r = np.deg2rad(lat1)
    lat2r = np.deg2rad(lat2)
    dlon = np.deg2rad(lon2 - lon1)
    y = np.sin(dlon) * np.cos(lat2r)
    x = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    brg = np.rad2deg(np.arctan2(y, x))
    return (brg + 360.0) % 360.0


def _circular_delta_deg(a: np.ndarray) -> np.ndarray:
    return (a + 180.0) % 360.0 - 180.0


def _summary(series: pd.Series, name: str, subset: str) -> dict[str, float | str]:
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return {"subset": subset, "metric": name, "count": 0, "mean": np.nan, "std": np.nan, "p50": np.nan, "p90": np.nan, "p95": np.nan, "p99": np.nan}
    return {
        "subset": subset,
        "metric": name,
        "count": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "p50": float(x.quantile(0.50)),
        "p90": float(x.quantile(0.90)),
        "p95": float(x.quantile(0.95)),
        "p99": float(x.quantile(0.99)),
    }


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.samples_path)
    keep = [args.flight_id_col, args.time_col, "lat", "lon", "alt", "speed", "speed_delta", "vertical_speed", "turn_rate"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    df[args.time_col] = pd.to_datetime(df[args.time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[args.flight_id_col, args.time_col, "lat", "lon", "alt"])
    df = df.drop_duplicates(subset=[args.flight_id_col, args.time_col]).sort_values([args.flight_id_col, args.time_col]).reset_index(drop=True)

    g = df.groupby(args.flight_id_col, sort=False)
    dt_min = g[args.time_col].diff().dt.total_seconds().div(60.0)
    df["dt_min"] = dt_min
    df["vertical_speed_calc"] = g["alt"].diff() / df["dt_min"].replace(0.0, np.nan)
    if "vertical_speed" not in df.columns:
        df["vertical_speed"] = df["vertical_speed_calc"]
    if "speed" not in df.columns:
        dist = _haversine_m(
            g["lat"].shift(1).to_numpy(),
            g["lon"].shift(1).to_numpy(),
            df["lat"].to_numpy(),
            df["lon"].to_numpy(),
        )
        df["speed"] = dist / df["dt_min"].replace(0.0, np.nan)
    if "speed_delta" not in df.columns:
        df["speed_delta"] = g["speed"].diff() / df["dt_min"].replace(0.0, np.nan)

    hdg = _heading_deg(
        g["lat"].shift(1).to_numpy(),
        g["lon"].shift(1).to_numpy(),
        df["lat"].to_numpy(),
        df["lon"].to_numpy(),
    )
    df["heading"] = hdg
    d_h = _circular_delta_deg(g["heading"].diff().to_numpy())
    df["heading_rate_calc"] = d_h / df["dt_min"].replace(0.0, np.nan).to_numpy()
    if "turn_rate" not in df.columns:
        df["turn_rate"] = df["heading_rate_calc"]

    df["planar_accel"] = g["speed"].diff() / df["dt_min"].replace(0.0, np.nan)
    df["curvature_proxy"] = np.deg2rad(df["turn_rate"].abs()) / (df["speed"].abs() + 1e-6)

    cruise_mask = (
        df["dt_min"].notna()
        & (df["dt_min"] > 0)
        & (df["vertical_speed"].abs() <= float(args.max_abs_vertical_rate))
        & (df["speed_delta"].abs() <= float(args.max_speed_delta))
        & (df["turn_rate"].abs() <= float(args.max_heading_rate))
    )
    df["is_cruise_by_rule"] = cruise_mask.astype(int)

    metrics = ["speed", "speed_delta", "turn_rate", "vertical_speed", "planar_accel", "curvature_proxy"]
    rows = []
    for m in metrics:
        if m in df.columns:
            rows.append(_summary(df[m], m, "all"))
            rows.append(_summary(df.loc[cruise_mask, m], m, "cruise"))
    summary = pd.DataFrame(rows)
    summary_path = out_dir / "cruise_dynamics_summary.csv"
    summary.to_csv(summary_path, index=False)

    by_flight = (
        df.groupby(args.flight_id_col)["is_cruise_by_rule"]
        .agg(["count", "sum"])
        .rename(columns={"count": "points", "sum": "cruise_points"})
        .reset_index()
    )
    by_flight["cruise_ratio"] = by_flight["cruise_points"] / by_flight["points"].clip(lower=1)
    by_flight_path = out_dir / "cruise_ratio_by_flight.csv"
    by_flight.to_csv(by_flight_path, index=False)

    total = int(len(df))
    cruise_total = int(cruise_mask.sum())
    print(
        f"[cruise_audit] rows={total} cruise_rows={cruise_total} cruise_ratio={cruise_total / max(1, total):.4f} "
        f"summary={summary_path} by_flight={by_flight_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

