from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_PREFIXES = [
    ("ourmethod", "ourmethod"),
    ("unilstm_baseline", "unilstm"),
    ("bilstm_baseline", "bilstm"),
    ("cnnlstm_baseline", "cnnlstm"),
    ("transformer_baseline", "transformer"),
    ("kalman_filter_baseline", "kalman_filter"),
]


def _build_one(flight_dir: Path) -> Path:
    recovered_path = flight_dir / "recovered_minute_compare.csv"
    if not recovered_path.exists():
        raise FileNotFoundError(recovered_path)
    df = pd.read_csv(recovered_path).copy()
    df["minute_ts"] = df["minute_ts"].astype(str)

    out = pd.DataFrame({"time_minute": df["minute_ts"]})

    known_adsb = pd.to_numeric(df["known_adsb"], errors="coerce").fillna(0).astype(int) == 1
    is_anchor = pd.to_numeric(df["is_adsc_anchor"], errors="coerce").fillna(0).astype(int) == 1

    out["adsb_lat_deg"] = np.where(known_adsb, df["lat"], np.nan)
    out["adsb_lon_deg"] = np.where(known_adsb, df["lon"], np.nan)
    out["adsb_alt_m"] = np.where(known_adsb, df["alt"], np.nan)

    out["adsc_anchor_lat_deg"] = np.where(is_anchor, df["lat"], np.nan)
    out["adsc_anchor_lon_deg"] = np.where(is_anchor, df["lon"], np.nan)
    out["adsc_anchor_alt_m"] = np.where(is_anchor, df["alt"], np.nan)

    for src_prefix, out_prefix in MODEL_PREFIXES:
        out[f"{out_prefix}_lat_deg"] = df[f"{src_prefix}_pred_lat"]
        out[f"{out_prefix}_lon_deg"] = df[f"{src_prefix}_pred_lon"]
        out[f"{out_prefix}_alt_m"] = df[f"{src_prefix}_pred_alt"]

    out_path = flight_dir / "plot_ready_3d.csv"
    out.to_csv(out_path, index=False)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser("Build plot-ready 3D CSVs for selected cross-ocean recovery flights.")
    ap.add_argument("--root-dir", default="outputs/runs/cross_ocean_gap_recovery_batch_20260429_allmodels")
    ap.add_argument(
        "--flight-ids",
        nargs="+",
        required=True,
        help="Flight directory names under root-dir, e.g. 86dce6_2024-05-03",
    )
    args = ap.parse_args()

    root = Path(args.root_dir)
    rows = []
    for flight_id in args.flight_ids:
        flight_dir = root / flight_id
        out_path = _build_one(flight_dir)
        rows.append({"flight_id": flight_id, "plot_ready_3d_csv": str(out_path)})

    pd.DataFrame(rows).to_csv(root / "plot_ready_3d_index.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
