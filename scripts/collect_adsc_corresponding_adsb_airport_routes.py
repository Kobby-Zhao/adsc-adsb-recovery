from __future__ import annotations

import argparse
import math
import os
import struct
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-thread collect ADS-C corresponding OpenSky flights, keep cross-ocean airport routes."
    )
    p.add_argument("--adsc-csv", default="ads-c_data/adsc_decoded_2024-05-01_to_2024-12.csv")
    p.add_argument("--airports-csv", default="outputs/flights/raw/airports.csv")
    p.add_argument("--out-dir", default="outputs/runs/adsc_corresponding_adsb_airport_routes_20260420_500")
    p.add_argument("--target", type=int, default=500)
    p.add_argument("--min-distance-km", type=float, default=3000.0)
    p.add_argument("--segment-gap-min", type=float, default=180.0)
    p.add_argument("--max-keys", type=int, default=0, help="0 means all keys")
    p.add_argument("--rate-limit-seconds", type=float, default=0.2)
    p.add_argument("--globe-shp", default="ne_110m_land.shp")
    p.add_argument("--center-lon", type=float, default=20.0)
    p.add_argument("--center-lat", type=float, default=10.0)
    return p.parse_args()


def haversine_km(lon1, lat1, lon2, lat2) -> float:
    r = 6371.0
    p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def read_shp_polygons(shp_path: Path):
    polys = []
    with shp_path.open("rb") as f:
        header = f.read(100)
        if len(header) < 100:
            return polys
        while True:
            rec_header = f.read(8)
            if not rec_header or len(rec_header) < 8:
                break
            _, rec_len_words = struct.unpack(">2i", rec_header)
            rec_len = rec_len_words * 2
            rec = f.read(rec_len)
            if len(rec) < 44:
                continue
            shape_type = struct.unpack("<i", rec[:4])[0]
            if shape_type not in (3, 5):
                continue
            num_parts, num_points = struct.unpack("<2i", rec[36:44])
            parts = struct.unpack("<" + "i" * num_parts, rec[44 : 44 + 4 * num_parts]) if num_parts > 0 else []
            pts_off = 44 + 4 * num_parts
            pts = []
            for i in range(num_points):
                off = pts_off + 16 * i
                if off + 16 > len(rec):
                    break
                x, y = struct.unpack("<2d", rec[off : off + 16])
                pts.append((x, y))
            if not pts:
                continue
            if not parts:
                polys.append(pts)
            else:
                idx = list(parts) + [len(pts)]
                for i in range(len(parts)):
                    seg = pts[idx[i] : idx[i + 1]]
                    if len(seg) >= 2:
                        polys.append(seg)
    return polys


def ortho_project(lon_deg, lat_deg, lon0_deg=20.0, lat0_deg=10.0):
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    lon0 = math.radians(lon0_deg)
    lat0 = math.radians(lat0_deg)
    cosc = np.sin(lat0) * np.sin(lat) + np.cos(lat0) * np.cos(lat) * np.cos(lon - lon0)
    visible = cosc >= 0
    x = np.cos(lat) * np.sin(lon - lon0)
    y = np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(lon - lon0)
    return x, y, visible


def great_circle_points(lon1, lat1, lon2, lat2, n=60):
    lon1r, lat1r = map(math.radians, (float(lon1), float(lat1)))
    lon2r, lat2r = map(math.radians, (float(lon2), float(lat2)))
    p1 = np.array([math.cos(lat1r) * math.cos(lon1r), math.cos(lat1r) * math.sin(lon1r), math.sin(lat1r)])
    p2 = np.array([math.cos(lat2r) * math.cos(lon2r), math.cos(lat2r) * math.sin(lon2r), math.sin(lat2r)])
    dot = float(np.clip(np.dot(p1, p2), -1.0, 1.0))
    omega = math.acos(dot)
    if abs(omega) < 1e-9:
        return np.full(n, lon1), np.full(n, lat1)
    so = math.sin(omega)
    ts = np.linspace(0.0, 1.0, n)
    pts = []
    for t in ts:
        v = (math.sin((1 - t) * omega) / so) * p1 + (math.sin(t * omega) / so) * p2
        v = v / np.linalg.norm(v)
        lon = math.degrees(math.atan2(v[1], v[0]))
        lat = math.degrees(math.asin(v[2]))
        pts.append((lon, lat))
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def to_dt(ts: pd.Timestamp) -> datetime:
    if ts.tzinfo is None:
        return ts.to_pydatetime().replace(tzinfo=timezone.utc)
    return ts.to_pydatetime()


def build_adsc_segments(adsc_csv: str, gap_min: float) -> pd.DataFrame:
    adsc = pd.read_csv(adsc_csv, usecols=["icao24", "timestamp"])
    adsc["timestamp"] = pd.to_datetime(adsc["timestamp"], utc=True, errors="coerce")
    adsc = adsc.dropna(subset=["icao24", "timestamp"]).copy()
    adsc["icao24"] = adsc["icao24"].astype(str).str.lower()
    adsc["day"] = adsc["timestamp"].dt.strftime("%Y-%m-%d")
    adsc = adsc.sort_values(["icao24", "day", "timestamp"])
    adsc["diff_min"] = adsc.groupby(["icao24", "day"])["timestamp"].diff().dt.total_seconds().div(60.0)
    adsc["seg_break"] = adsc["diff_min"].isna() | (adsc["diff_min"] > float(gap_min))
    adsc["seg_idx"] = adsc.groupby(["icao24", "day"])["seg_break"].cumsum().astype(int)
    seg = (
        adsc.groupby(["icao24", "day", "seg_idx"], as_index=False)
        .agg(seg_start=("timestamp", "min"), seg_end=("timestamp", "max"), seg_points=("timestamp", "count"))
        .reset_index(drop=True)
    )
    return seg


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from src.io_utils import load_settings

    settings = load_settings("config/settings.yaml")
    username = settings["opensky"].get("username")
    password = settings["opensky"].get("password")
    if username:
        os.environ["OPENSKY_USERNAME"] = str(username).lower()
        os.environ["OPENSKY_TRINO_USERNAME"] = str(username).lower()
    if password:
        os.environ["OPENSKY_PASSWORD"] = str(password)
        os.environ["OPENSKY_TRINO_PASSWORD"] = str(password)

    from pyopensky.trino import Trino

    airports = pd.read_csv(args.airports_csv, usecols=["icao", "lat", "lon"]).dropna().copy()
    airports["icao"] = airports["icao"].astype(str).str.upper()
    airport_map = airports.set_index("icao")[["lat", "lon"]]

    seg = build_adsc_segments(args.adsc_csv, gap_min=args.segment_gap_min)
    keys = seg[["icao24", "day"]].drop_duplicates().reset_index(drop=True)
    if int(args.max_keys) > 0:
        keys = keys.head(int(args.max_keys)).copy()

    seg_by_key = {(r["icao24"], r["day"]): g for (r, g) in []}
    seg_group = {}
    for (icao, day), g in seg.groupby(["icao24", "day"]):
        seg_group[(icao, day)] = g.sort_values("seg_start")

    trino = Trino()
    rows = []
    query_count = 0
    matched_segments = 0

    for i, key in keys.iterrows():
        icao = str(key["icao24"]).lower()
        day = str(key["day"])
        day_start = pd.Timestamp(day, tz="UTC")
        day_end = day_start + timedelta(days=1)
        s = seg_group.get((icao, day))
        if s is None or s.empty:
            continue

        query_count += 1
        try:
            flights = trino.flightlist(start=to_dt(day_start), stop=to_dt(day_end), icao24=icao, cached=True)
        except Exception:
            flights = None
        if flights is None or len(flights) == 0:
            if args.rate_limit_seconds > 0:
                time.sleep(float(args.rate_limit_seconds))
            continue

        flights = flights.copy()
        cols_lower = {c.lower(): c for c in flights.columns}
        c_first = cols_lower.get("firstseen")
        c_last = cols_lower.get("lastseen")
        if c_first is None or c_last is None:
            if args.rate_limit_seconds > 0:
                time.sleep(float(args.rate_limit_seconds))
            continue
        flights["firstSeen"] = pd.to_datetime(flights[c_first], unit="s", utc=True, errors="coerce")
        flights["lastSeen"] = pd.to_datetime(flights[c_last], unit="s", utc=True, errors="coerce")
        flights = flights.dropna(subset=["firstSeen", "lastSeen"])
        if flights.empty:
            if args.rate_limit_seconds > 0:
                time.sleep(float(args.rate_limit_seconds))
            continue

        for _, seg_r in s.iterrows():
            seg_start = seg_r["seg_start"]
            seg_end = seg_r["seg_end"]
            cand = flights[(flights["firstSeen"] <= seg_start) & (flights["lastSeen"] >= seg_end)].copy()
            if cand.empty:
                continue
            cand["contain_span_sec"] = (cand["lastSeen"] - cand["firstSeen"]).dt.total_seconds()
            cand = cand.sort_values("contain_span_sec", ascending=True)
            best = cand.iloc[0]

            dep = str(best.get("estDepartureAirport") or "").strip().upper()
            arr = str(best.get("estArrivalAirport") or "").strip().upper()
            if dep not in airport_map.index or arr not in airport_map.index:
                continue
            dep_lat, dep_lon = airport_map.loc[dep, ["lat", "lon"]]
            arr_lat, arr_lon = airport_map.loc[arr, ["lat", "lon"]]
            d_km = haversine_km(dep_lon, dep_lat, arr_lon, arr_lat)
            if d_km < float(args.min_distance_km):
                continue

            matched_segments += 1
            rows.append(
                {
                    "icao24": icao,
                    "day": day,
                    "seg_idx": int(seg_r["seg_idx"]),
                    "seg_start": seg_start,
                    "seg_end": seg_end,
                    "seg_points": int(seg_r["seg_points"]),
                    "callsign": str(best.get("callsign") or "").strip(),
                    "firstSeen": best["firstSeen"],
                    "lastSeen": best["lastSeen"],
                    "dep_airport": dep,
                    "arr_airport": arr,
                    "dep_lat": float(dep_lat),
                    "dep_lon": float(dep_lon),
                    "arr_lat": float(arr_lat),
                    "arr_lon": float(arr_lon),
                    "airport_distance_km": float(d_km),
                }
            )
            if len(rows) >= int(args.target):
                break
        if len(rows) >= int(args.target):
            break
        if args.rate_limit_seconds > 0:
            time.sleep(float(args.rate_limit_seconds))
        if (i + 1) % 100 == 0:
            print(f"[progress] keys={i+1}/{len(keys)} queries={query_count} selected={len(rows)}")

    out_csv = out_dir / "adsc_corresponding_cross_ocean_airport_routes_top500.csv"
    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df = out_df.drop_duplicates(subset=["icao24", "day", "seg_idx", "dep_airport", "arr_airport"]).reset_index(
            drop=True
        )
    out_df.to_csv(out_csv, index=False)

    # Plot (light blue trajectories)
    polys = read_shp_polygons(Path(args.globe_shp))
    fig = plt.figure(figsize=(10, 10), facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")
    th = np.linspace(0, 2 * math.pi, 720)
    ax.plot(np.cos(th), np.sin(th), color="#c9c9c9", lw=1.2, zorder=5)
    for seg_poly in polys:
        lon = np.array([p[0] for p in seg_poly], dtype=float)
        lat = np.array([p[1] for p in seg_poly], dtype=float)
        x, y, vis = ortho_project(lon, lat, lon0_deg=args.center_lon, lat0_deg=args.center_lat)
        ax.plot(np.where(vis, x, np.nan), np.where(vis, y, np.nan), color="#d2d2d2", lw=0.6, alpha=0.8, zorder=6)

    for _, r in out_df.iterrows():
        lons, lats = great_circle_points(r["dep_lon"], r["dep_lat"], r["arr_lon"], r["arr_lat"], n=60)
        x, y, vis = ortho_project(lons, lats, lon0_deg=args.center_lon, lat0_deg=args.center_lat)
        ax.plot(np.where(vis, x, np.nan), np.where(vis, y, np.nan), color="#9ED8FF", lw=1.25, alpha=0.62, zorder=8)

    if not out_df.empty:
        xs, ys, vs = ortho_project(
            out_df["dep_lon"].to_numpy(), out_df["dep_lat"].to_numpy(), lon0_deg=args.center_lon, lat0_deg=args.center_lat
        )
        ax.scatter(xs[vs], ys[vs], s=9, color="#9ED8FF", alpha=0.75, zorder=9)
        xe, ye, ve = ortho_project(
            out_df["arr_lon"].to_numpy(), out_df["arr_lat"].to_numpy(), lon0_deg=args.center_lon, lat0_deg=args.center_lat
        )
        ax.scatter(xe[ve], ye[ve], s=9, color="#9ED8FF", alpha=0.75, zorder=9)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axis("off")
    ax.set_title(f"ADS-C Corresponding Cross-Ocean Routes (OpenSky, n={len(out_df)})", fontsize=12, pad=12)
    out_png = out_dir / "adsc_corresponding_cross_ocean_routes_globe.png"
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    plt.close(fig)

    summary = out_dir / "summary.txt"
    with summary.open("w", encoding="utf-8") as f:
        f.write(f"keys_total={len(keys)}\n")
        f.write(f"queries_executed={query_count}\n")
        f.write(f"matched_segments_cross_ocean={matched_segments}\n")
        f.write(f"selected_unique_routes={len(out_df)}\n")
        f.write(f"target={args.target}\n")
        f.write(f"min_distance_km={args.min_distance_km}\n")
        f.write(f"segment_gap_min={args.segment_gap_min}\n")
        f.write(f"output_csv={out_csv}\n")
        f.write(f"output_png={out_png}\n")

    print(f"[done] queries={query_count} selected={len(out_df)} target={args.target}")
    print(f"[done] csv={out_csv}")
    print(f"[done] png={out_png}")


if __name__ == "__main__":
    main()
