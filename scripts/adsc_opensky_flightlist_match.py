from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Match ADS-C windows to OpenSky flightlist by ICAO24 and time.")
    parser.add_argument("--adsc-decoded-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-icao", type=int, default=0, help="Limit number of ICAO24 to process (0=all).")
    parser.add_argument("--max-windows", type=int, default=0, help="Limit number of ADS-C windows to process (0=all).")
    parser.add_argument("--rate-limit-seconds", type=float, default=10.0)
    return parser.parse_args()


def to_dt(value: pd.Timestamp) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.to_pydatetime()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load settings for OpenSky credentials
    from src.io_utils import load_settings

    settings = load_settings("config/settings.yaml")
    username = settings["opensky"].get("username")
    password = settings["opensky"].get("password")
    if username:
        os.environ["OPENSKY_USERNAME"] = username.lower()
        os.environ["OPENSKY_TRINO_USERNAME"] = username.lower()
    if password:
        os.environ["OPENSKY_PASSWORD"] = password
        os.environ["OPENSKY_TRINO_PASSWORD"] = password

    from pyopensky.trino import Trino

    trino = Trino()

    # Aggregate ADS-C windows by icao24/day
    adsc_cols = ["icao24", "day", "timestamp", "latitude", "longitude", "altitude_m"]
    adsc_df = pd.read_csv(args.adsc_decoded_csv, usecols=adsc_cols)
    adsc_df["timestamp"] = pd.to_datetime(adsc_df["timestamp"], utc=True, errors="coerce")
    adsc_df = adsc_df.dropna(subset=["icao24", "day", "timestamp"])

    windows = (
        adsc_df.groupby(["icao24", "day"])
        .agg(adsc_start=("timestamp", "min"), adsc_end=("timestamp", "max"), adsc_points=("timestamp", "count"))
        .reset_index()
    )
    if args.max_icao and args.max_icao > 0:
        windows = windows[windows["icao24"].isin(windows["icao24"].drop_duplicates().head(args.max_icao))]

    if args.max_windows and args.max_windows > 0:
        windows = windows.head(args.max_windows)

    flightlist_rows = []
    match_rows = []

    for _, w in windows.iterrows():
        icao = str(w["icao24"]).lower()
        day_start = pd.to_datetime(w["day"], utc=True, errors="coerce")
        if pd.isna(day_start):
            continue
        day_end = day_start + pd.Timedelta(days=1)
        w_start = w["adsc_start"]
        w_end = w["adsc_end"]

        try:
            flights = trino.flightlist(start=to_dt(day_start), stop=to_dt(day_end), icao24=icao, cached=True)
        except Exception:
            flights = None

        if flights is None or len(flights) == 0:
            flightlist_rows.append(
                {
                    "icao24": icao,
                    "day": w["day"],
                    "start": day_start,
                    "stop": day_end,
                    "rows": 0,
                }
            )
            time.sleep(args.rate_limit_seconds)
            continue

        flights = flights.copy()
        cols_lower = {c.lower(): c for c in flights.columns}
        first_col = cols_lower.get("firstseen")
        last_col = cols_lower.get("lastseen")
        if first_col is None or last_col is None:
            flightlist_rows.append(
                {
                    "icao24": icao,
                    "day": w["day"],
                    "start": day_start,
                    "stop": day_end,
                    "rows": len(flights),
                    "error": "missing_firstseen_lastseen_cols",
                }
            )
            time.sleep(args.rate_limit_seconds)
            continue
        flights["firstSeen"] = pd.to_datetime(flights[first_col], unit="s", utc=True, errors="coerce")
        flights["lastSeen"] = pd.to_datetime(flights[last_col], unit="s", utc=True, errors="coerce")
        flights = flights.dropna(subset=["firstSeen", "lastSeen"])
        flightlist_rows.append(
            {
                "icao24": icao,
                "day": w["day"],
                "start": day_start,
                "stop": day_end,
                "rows": len(flights),
            }
        )

        # candidates where flight time contains ADS-C window
        cand = flights[(flights["firstSeen"] <= w_start) & (flights["lastSeen"] >= w_end)].copy()
        if cand.empty:
            time.sleep(args.rate_limit_seconds)
            continue

        # pick the smallest containing flight (tightest interval)
        cand["contain_span_sec"] = (cand["lastSeen"] - cand["firstSeen"]).dt.total_seconds()
        cand = cand.sort_values("contain_span_sec", ascending=True)
        best = cand.iloc[0]
        match_rows.append(
            {
                "icao24": icao,
                "day": w["day"],
                "adsc_start": w_start,
                "adsc_end": w_end,
                "adsc_points": int(w["adsc_points"]),
                "flight_id": best.get("callsign", "").strip(),
                "firstSeen": best["firstSeen"],
                "lastSeen": best["lastSeen"],
                "estDepartureAirport": best.get("estDepartureAirport"),
                "estArrivalAirport": best.get("estArrivalAirport"),
            }
        )

        time.sleep(args.rate_limit_seconds)

    pd.DataFrame(flightlist_rows).to_csv(out_dir / "opensky_flightlist_query_log.csv", index=False)
    pd.DataFrame(match_rows).to_csv(out_dir / "adsc_adsb_flight_matches.csv", index=False)
    print(f"[done] windows_processed={len(windows)} matches={len(match_rows)} out={out_dir}")


if __name__ == "__main__":
    main()
