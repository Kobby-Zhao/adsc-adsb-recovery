from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cross_ocean_gap_recovery_compare import (
    _build_full_frame,
    _plot_3d,
    _plot_3d_cruise_only,
    _plot_altitude,
    _plot_altitude_cruise_only,
    _run_models,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Batch recover selected cross-ocean ADS-C flights with all models.")
    p.add_argument(
        "--overlay-summary-csv",
        default="outputs/runs/adsc_cross_ocean_top10_highest_anchor_20260429/overlay_summary.csv",
    )
    p.add_argument(
        "--adsb-minute-csv",
        default="outputs/runs/adsc_cross_ocean_top10_highest_anchor_20260429/top10_cross_ocean_highest_anchor_adsb_minute_full_flights.csv",
    )
    p.add_argument("--models", default="all")
    p.add_argument("--cruise-alt-threshold-m", type=float, default=10000.0)
    p.add_argument("--out-dir", required=True)
    return p


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.overlay_summary_csv)
    rows = []

    for _, row in summary.iterrows():
        pair_id = str(row["pair_id"])
        overlay_csv = Path(str(row["overlay_csv"]))
        pair_dir = out_dir / pair_id
        pair_dir.mkdir(parents=True, exist_ok=True)

        frame, adsb, adsc = _build_full_frame(
            pair_id=pair_id,
            overlay_csv=overlay_csv,
            adsb_minute_csv=Path(args.adsb_minute_csv),
        )
        frame.to_csv(pair_dir / "input_recovery_frame.csv", index=False)
        adsb.to_csv(pair_dir / "known_adsb_minute.csv", index=False)
        adsc.to_csv(pair_dir / "adsc_anchor_points.csv", index=False)

        stitched = _run_models(frame, args.models)

        merged = frame[
            ["sample_id", "flight_id", "minute_ts", "obs_mask", "known_adsb", "is_adsc_anchor", "obs_source", "lat", "lon", "alt"]
        ].copy()
        for model_name, s in stitched.items():
            cols = s[["minute_ts", "pred_lat", "pred_lon", "pred_alt"]].copy()
            cols = cols.rename(
                columns={
                    "pred_lat": f"{model_name}_pred_lat",
                    "pred_lon": f"{model_name}_pred_lon",
                    "pred_alt": f"{model_name}_pred_alt",
                }
            )
            merged = merged.merge(cols, on="minute_ts", how="left")
        merged.to_csv(pair_dir / "recovered_minute_compare.csv", index=False)

        _plot_altitude(frame, stitched, pair_dir / "altitude_2d_compare.png")
        _plot_3d(frame, stitched, pair_dir / "trajectory_3d_compare.png")
        _plot_altitude_cruise_only(
            frame, stitched, pair_dir / "altitude_2d_compare_cruise_only.png", args.cruise_alt_threshold_m
        )
        _plot_3d_cruise_only(
            frame, stitched, pair_dir / "trajectory_3d_compare_cruise_only.png", args.cruise_alt_threshold_m
        )

        rows.append(
            {
                "pair_id": pair_id,
                "flight_id": str(frame["flight_id"].iloc[0]),
                "known_adsb_minutes": int(frame["known_adsb"].astype(int).sum()),
                "adsc_anchor_minutes": int(frame["is_adsc_anchor"].astype(int).sum()),
                "missing_minutes": int((frame["obs_mask"].astype(int) == 0).sum()),
                "pair_dir": str(pair_dir),
                "recovered_csv": str(pair_dir / "recovered_minute_compare.csv"),
                "plot_2d": str(pair_dir / "altitude_2d_compare.png"),
                "plot_3d": str(pair_dir / "trajectory_3d_compare.png"),
                "plot_2d_cruise_only": str(pair_dir / "altitude_2d_compare_cruise_only.png"),
                "plot_3d_cruise_only": str(pair_dir / "trajectory_3d_compare_cruise_only.png"),
            }
        )

    pd.DataFrame(rows).to_csv(out_dir / "batch_recovery_summary.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
