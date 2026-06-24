from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _stats(s: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return {k: np.nan for k in ["count", "mean", "std", "min", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max"]}
    return {
        "count": float(len(x)),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "min": float(x.min()),
        "p10": float(x.quantile(0.10)),
        "p25": float(x.quantile(0.25)),
        "p50": float(x.quantile(0.50)),
        "p75": float(x.quantile(0.75)),
        "p90": float(x.quantile(0.90)),
        "p95": float(x.quantile(0.95)),
        "p99": float(x.quantile(0.99)),
        "max": float(x.max()),
    }


def _long_stats(frame: pd.DataFrame, dataset: str, metrics: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for col, label in metrics:
        if col not in frame.columns:
            continue
        rows.append({"dataset": dataset, "metric": label, **_stats(frame[col])})
    return pd.DataFrame(rows)


def _build_gap_stats(points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = points.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["flight_id", "timestamp"]).sort_values(["flight_id", "timestamp"])
    rows = []
    per_flight_rows = []
    for fid, g in frame.groupby("flight_id", sort=False):
        times = g["timestamp"].to_numpy()
        if len(times) < 2:
            continue
        gaps = np.diff(times).astype("timedelta64[s]").astype(float) / 60.0
        gaps = gaps[np.isfinite(gaps) & (gaps >= 0)]
        if len(gaps) == 0:
            continue
        for i, gap in enumerate(gaps):
            rows.append({"flight_id": fid, "gap_id": i, "gap_min": float(gap)})
        per_flight_rows.append(
            {
                "flight_id": fid,
                "mean_adjacent_gap_min": float(np.mean(gaps)),
                "median_adjacent_gap_min": float(np.median(gaps)),
                "max_adjacent_gap_min": float(np.max(gaps)),
                "p90_adjacent_gap_min": float(np.quantile(gaps, 0.90)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(per_flight_rows)


def _augment_flight_stats(summary: pd.DataFrame, gap_per_flight: pd.DataFrame) -> pd.DataFrame:
    out = summary.merge(gap_per_flight, on="flight_id", how="left")
    out["duration_grid_min"] = pd.to_numeric(out["duration_min"], errors="coerce") + 1.0
    out["anchor_density_per_hour"] = pd.to_numeric(out["anchor_count"], errors="coerce") / (
        pd.to_numeric(out["duration_min"], errors="coerce") / 60.0
    )
    out["observed_minute_ratio"] = pd.to_numeric(out["anchor_count"], errors="coerce") / out["duration_grid_min"]
    out["minute_missing_ratio_between_first_last"] = 1.0 - out["observed_minute_ratio"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--points-csv",
        default="outputs/runs/adsc_flight_segmentation_4h_20260421_fix_epoch_v2/adsc_points_with_flight_id_4h_min2_latest1200_minute_agg.csv",
    )
    parser.add_argument(
        "--all-routes-csv",
        default="outputs/runs/cross_ocean_adsc_anchor_routes_20260517/adsc_anchor_route_summary_all.csv",
    )
    parser.add_argument(
        "--selected-routes-csv",
        default="outputs/runs/cross_ocean_adsc_anchor_routes_20260517/cross_ocean_adsc_anchor_routes_selected.csv",
    )
    parser.add_argument("--out-dir", default="outputs/runs/cross_ocean_adsc_anchor_routes_20260517/adsc_quartile_stats")
    args = parser.parse_args()

    points = pd.read_csv(_resolve(args.points_csv))
    all_routes = pd.read_csv(_resolve(args.all_routes_csv))
    selected_routes = pd.read_csv(_resolve(args.selected_routes_csv))
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gap_rows, gap_per_flight = _build_gap_stats(points)
    all_flights = _augment_flight_stats(all_routes, gap_per_flight)
    selected_ids = set(selected_routes["flight_id"].astype(str))
    cross_points = points[points["flight_id"].astype(str).isin(selected_ids)].copy()
    cross_gap_rows = gap_rows[gap_rows["flight_id"].astype(str).isin(selected_ids)].copy()
    cross_flights = _augment_flight_stats(selected_routes, gap_per_flight)

    flight_metrics = [
        ("anchor_count", "锚点数/航班"),
        ("duration_min", "首末锚点持续时间/min"),
        ("anchor_density_per_hour", "锚点密度/(个/h)"),
        ("observed_minute_ratio", "首末锚点区间分钟观测率"),
        ("minute_missing_ratio_between_first_last", "首末锚点区间分钟缺失率"),
        ("mean_adjacent_gap_min", "航班内平均相邻锚点间隔/min"),
        ("median_adjacent_gap_min", "航班内中位相邻锚点间隔/min"),
        ("max_adjacent_gap_min", "航班内最大相邻锚点间隔/min"),
        ("p90_adjacent_gap_min", "航班内P90相邻锚点间隔/min"),
        ("endpoint_distance_km", "首末锚点大圆距离/km"),
        ("avg_endpoint_speed_kmh", "首末锚点平均速度/(km/h)"),
        ("ocean_ratio_gc", "大圆路径海洋比例"),
    ]
    gap_metrics = [("gap_min", "相邻锚点间隔/min")]

    flight_stats = pd.concat(
        [
            _long_stats(all_flights, "All ADS-C segmented flights", flight_metrics),
            _long_stats(cross_flights, "Selected cross-ocean ADS-C flights", flight_metrics),
        ],
        ignore_index=True,
    )
    gap_stats = pd.concat(
        [
            _long_stats(gap_rows, "All ADS-C adjacent gaps", gap_metrics),
            _long_stats(cross_gap_rows, "Selected cross-ocean ADS-C adjacent gaps", gap_metrics),
        ],
        ignore_index=True,
    )

    all_flights.to_csv(out_dir / "all_adsc_flight_level_features.csv", index=False, encoding="utf-8-sig")
    cross_flights.to_csv(out_dir / "cross_ocean_adsc_flight_level_features.csv", index=False, encoding="utf-8-sig")
    gap_rows.to_csv(out_dir / "all_adsc_adjacent_gap_rows.csv", index=False, encoding="utf-8-sig")
    cross_gap_rows.to_csv(out_dir / "cross_ocean_adsc_adjacent_gap_rows.csv", index=False, encoding="utf-8-sig")
    flight_stats.to_csv(out_dir / "adsc_flight_level_quartile_stats.csv", index=False, encoding="utf-8-sig")
    gap_stats.to_csv(out_dir / "adsc_adjacent_gap_quartile_stats.csv", index=False, encoding="utf-8-sig")

    print(f"[done] out_dir={out_dir}")
    print("\n[flight-level quartiles]")
    print(flight_stats.round(3).to_string(index=False))
    print("\n[gap-level quartiles]")
    print(gap_stats.round(3).to_string(index=False))
    print(
        "\n[counts] "
        f"all_flights={len(all_flights)} selected_cross_ocean_flights={len(cross_flights)} "
        f"all_anchor_points={len(points)} selected_anchor_points={len(cross_points)} "
        f"all_adjacent_gaps={len(gap_rows)} selected_adjacent_gaps={len(cross_gap_rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
