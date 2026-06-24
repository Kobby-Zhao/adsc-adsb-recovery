from __future__ import annotations

import argparse
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mine non-Atlantic ADS-C-corresponding flight candidates from ADS-C icao/day keys.")
    p.add_argument("--adsc-csv", default="ads-c_data/adsc_decoded_2024-05-01_to_2024-12.csv")
    p.add_argument("--airports-csv", default="outputs/flights/raw/airports.csv")
    p.add_argument("--out-dir", default="outputs/runs/adsc_multi_ocean_candidates_20260421/from_adsc_keys_mining")
    p.add_argument("--target", type=int, default=80)
    p.add_argument("--max-keys", type=int, default=6000)
    p.add_argument("--rate-limit-seconds", type=float, default=0.2)
    p.add_argument("--min-distance-km", type=float, default=3000.0)
    return p.parse_args()


def haversine_km(lon1, lat1, lon2, lat2):
    r = 6371.0
    p1 = np.radians(float(lat1))
    p2 = np.radians(float(lat2))
    dphi = np.radians(float(lat2) - float(lat1))
    dl = np.radians(float(lon2) - float(lon1))
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return float(2.0 * r * np.arcsin(min(1.0, np.sqrt(a))))


def region(lat, lon):
    lon = ((float(lon) + 180) % 360) - 180
    lat = float(lat)
    if -20 < lon < 55 and lat > 35:
        return "Europe"
    if -20 < lon < 55 and -35 < lat <= 35:
        return "AfricaMid"
    if 55 < lon < 105 and -10 < lat < 40:
        return "MiddleEast_India"
    if 100 < lon < 150 and 5 < lat < 55:
        return "EastAsia"
    if 95 < lon < 160 and lat < -10:
        return "SEAsia_Oceania"
    if 140 <= lon <= 180 and lat < 0:
        return "Oceania"
    if lon < -120 and lat > 20:
        return "NorthAmericaWest"
    if -120 < lon < -60 and lat > 20:
        return "NorthAmericaEast"
    if lon < -70 and lat < -5:
        return "SouthAmericaWest"
    if -70 < lon < -30 and lat < -5:
        return "SouthAmericaEast"
    if -20 < lon < 20 and lat < -5:
        return "SouthAfrica_Atlantic"
    return "Other"


def classify_ocean(lat1, lon1, lat2, lon2):
    a = region(lat1, lon1)
    b = region(lat2, lon2)
    s = {a, b}
    latm = (float(lat1) + float(lat2)) / 2.0
    if {"EastAsia", "NorthAmericaWest"} <= s and latm > 20:
        return "North_Pacific"
    if (
        ("MiddleEast_India" in s and ("SEAsia_Oceania" in s or "Oceania" in s or "AfricaMid" in s))
        or ("SEAsia_Oceania" in s and "AfricaMid" in s)
    ) and -30 < latm < 25:
        return "Indian_Ocean"
    if {"Oceania", "SouthAmericaWest"} <= s:
        return "South_Pacific"
    if "SouthAmericaEast" in s and ("SouthAfrica_Atlantic" in s or "AfricaMid" in s):
        return "South_Atlantic"
    if ("NorthAmericaEast" in s or "NorthAmericaWest" in s) and ("Europe" in s or "AfricaMid" in s) and latm > 5:
        return "North_Atlantic"
    return "Other_Mixed"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from src.io_utils import load_settings

    settings = load_settings("config/settings.yaml")
    username = settings["opensky"].get("username") or os.getenv("OPENSKY_USERNAME") or os.getenv("OPENSKY_TRINO_USERNAME")
    password = settings["opensky"].get("password") or os.getenv("OPENSKY_PASSWORD") or os.getenv("OPENSKY_TRINO_PASSWORD")
    if username:
        os.environ["OPENSKY_USERNAME"] = str(username).lower()
        os.environ["OPENSKY_TRINO_USERNAME"] = str(username).lower()
    if password:
        os.environ["OPENSKY_PASSWORD"] = str(password)
        os.environ["OPENSKY_TRINO_PASSWORD"] = str(password)

    # IMPORTANT: pyopensky reads trino credentials at import-time.
    # If imported before setting env vars, it falls back to OAuth browser auth.
    if not os.getenv("OPENSKY_TRINO_USERNAME") or not os.getenv("OPENSKY_TRINO_PASSWORD"):
        raise RuntimeError(
            "Missing OPENSKY_TRINO_USERNAME/OPENSKY_TRINO_PASSWORD. "
            "Refusing to run to avoid OAuth popup fallback."
        )

    from pyopensky.trino import Trino

    ap = pd.read_csv(args.airports_csv, usecols=["icao", "lat", "lon"]).dropna().copy()
    ap["icao"] = ap["icao"].astype(str).str.upper().str.strip()
    ap = ap[ap["icao"].str.match(r"^[A-Z0-9]{4}$", na=False)].copy()
    ap_map = ap.set_index("icao")[["lat", "lon"]]

    adsc = pd.read_csv(args.adsc_csv, usecols=["icao24", "timestamp"])
    adsc["timestamp"] = pd.to_datetime(adsc["timestamp"], utc=True, errors="coerce")
    adsc = adsc.dropna(subset=["icao24", "timestamp"]).copy()
    adsc["icao24"] = adsc["icao24"].astype(str).str.lower()
    adsc["day"] = adsc["timestamp"].dt.strftime("%Y-%m-%d")
    key_df = adsc.groupby(["icao24", "day"], as_index=False).size().rename(columns={"size": "adsc_points_same_day"})
    key_df = key_df.sort_values("adsc_points_same_day", ascending=False).reset_index(drop=True)
    if int(args.max_keys) > 0:
        key_df = key_df.head(int(args.max_keys)).copy()

    trino = Trino()
    selected = []
    qlog = []

    for i, r in key_df.iterrows():
        icao = str(r["icao24"]).lower()
        day = str(r["day"])
        day_start = pd.Timestamp(day, tz="UTC")
        day_end = day_start + timedelta(days=1)

        try:
            flights = trino.flightlist(start=day_start.to_pydatetime(), stop=day_end.to_pydatetime(), icao24=icao, cached=True)
        except Exception as e:
            qlog.append({"icao24": icao, "day": day, "ok": 0, "rows": 0, "err": str(type(e).__name__)})
            if args.rate_limit_seconds > 0:
                time.sleep(float(args.rate_limit_seconds))
            continue

        if flights is None or len(flights) == 0:
            qlog.append({"icao24": icao, "day": day, "ok": 1, "rows": 0})
            if args.rate_limit_seconds > 0:
                time.sleep(float(args.rate_limit_seconds))
            continue

        flights = flights.copy()
        cols = {c.lower(): c for c in flights.columns}
        c_first = cols.get("firstseen")
        c_last = cols.get("lastseen")
        if c_first is None or c_last is None:
            qlog.append({"icao24": icao, "day": day, "ok": 1, "rows": len(flights), "err": "missing_firstseen_lastseen"})
            if args.rate_limit_seconds > 0:
                time.sleep(float(args.rate_limit_seconds))
            continue

        flights["firstSeen"] = pd.to_datetime(flights[c_first], unit="s", utc=True, errors="coerce")
        flights["lastSeen"] = pd.to_datetime(flights[c_last], unit="s", utc=True, errors="coerce")
        flights = flights.dropna(subset=["firstSeen", "lastSeen"])
        qlog.append({"icao24": icao, "day": day, "ok": 1, "rows": len(flights)})

        for _, f in flights.iterrows():
            dep = str(f.get("estDepartureAirport") or "").strip().upper()
            arr = str(f.get("estArrivalAirport") or "").strip().upper()
            if dep not in ap_map.index or arr not in ap_map.index:
                continue
            lat1, lon1 = ap_map.loc[dep, ["lat", "lon"]]
            lat2, lon2 = ap_map.loc[arr, ["lat", "lon"]]
            dist = haversine_km(lon1, lat1, lon2, lat2)
            if dist < float(args.min_distance_km):
                continue
            bucket = classify_ocean(lat1, lon1, lat2, lon2)
            if bucket in {"North_Atlantic", "Other_Mixed"}:
                continue
            selected.append(
                {
                    "icao24": icao,
                    "day": day,
                    "adsc_points_same_day": int(r["adsc_points_same_day"]),
                    "callsign": str(f.get("callsign") or "").strip(),
                    "firstseen": int(pd.Timestamp(f["firstSeen"]).timestamp()),
                    "lastseen": int(pd.Timestamp(f["lastSeen"]).timestamp()),
                    "estdepartureairport": dep,
                    "estarrivalairport": arr,
                    "start_lat": float(lat1),
                    "start_lon": float(lon1),
                    "end_lat": float(lat2),
                    "end_lon": float(lon2),
                    "distance_km": float(dist),
                    "ocean_bucket": bucket,
                }
            )
            if len(selected) >= int(args.target):
                break
        if len(selected) >= int(args.target):
            break

        if args.rate_limit_seconds > 0:
            time.sleep(float(args.rate_limit_seconds))
        if (i + 1) % 200 == 0:
            print(f"[progress] keys={i+1}/{len(key_df)} selected={len(selected)}")

    sel = pd.DataFrame(selected).drop_duplicates(subset=["icao24", "day", "firstseen", "lastseen"])
    sel.to_csv(out_dir / "mined_other_ocean_candidates.csv", index=False)
    if not sel.empty:
        sel[["icao24", "callsign", "firstseen", "lastseen", "day"]].to_csv(
            out_dir / "mined_other_ocean_candidates_for_fetch.csv", index=False
        )
    pd.DataFrame(qlog).to_csv(out_dir / "query_log.csv", index=False)
    summary = sel["ocean_bucket"].value_counts() if not sel.empty else pd.Series(dtype=int)
    with (out_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"keys_scanned={min(len(key_df), len(qlog))}\n")
        f.write(f"selected={len(sel)}\n")
        for k, v in summary.items():
            f.write(f"{k}={int(v)}\n")
    print(f"[done] keys_scanned={min(len(key_df), len(qlog))} selected={len(sel)}")
    if not sel.empty:
        print(sel["ocean_bucket"].value_counts().to_string())


if __name__ == "__main__":
    main()
