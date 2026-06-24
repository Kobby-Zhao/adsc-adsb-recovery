from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Match ADS-C windows to local ADS-B flights and export cross-ocean trajectories."
    )
    p.add_argument(
        "--adsc-csv",
        default="ads-c_data/adsc_decoded_2024-05-01_to_2024-12.csv",
    )
    p.add_argument(
        "--adsb-parquet",
        default="outputs/mvp_merged_nostage_20260415/adsb_minute_merged.parquet",
    )
    p.add_argument("--airports-csv", default="outputs/flights/raw/airports.csv")
    p.add_argument("--out-dir", default="outputs/runs/adsc_corresponding_adsb_cross_ocean_20260420_500")
    p.add_argument("--target", type=int, default=500)
    p.add_argument("--min-distance-km", type=float, default=3000.0)
    p.add_argument("--max-candidates-per-key", type=int, default=20)
    p.add_argument("--globe-shp", default="ne_110m_land.shp")
    p.add_argument("--center-lon", type=float, default=20.0)
    p.add_argument("--center-lat", type=float, default=10.0)
    return p.parse_args()


def haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.asarray(lon1, dtype=float)
    lat1 = np.asarray(lat1, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    r = 6371.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.minimum(1.0, np.sqrt(a)))


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
            if shape_type not in (3, 5):  # polyline / polygon
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


def great_circle_points(lon1, lat1, lon2, lat2, n=50):
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
    lons = np.array([p[0] for p in pts], dtype=float)
    lats = np.array([p[1] for p in pts], dtype=float)
    return lons, lats


def nearest_airports(endpoints: pd.DataFrame, airports: pd.DataFrame, side: str) -> pd.DataFrame:
    lat_col = f"{side}_lat"
    lon_col = f"{side}_lon"
    a_lat = airports["lat"].to_numpy(dtype=float)
    a_lon = airports["lon"].to_numpy(dtype=float)
    a_icao = airports["icao"].astype(str).to_numpy()
    batch = 256
    nearest_idx = []
    nearest_dist = []
    for i in range(0, len(endpoints), batch):
        sub = endpoints.iloc[i : i + batch]
        slat = sub[lat_col].to_numpy(dtype=float)[:, None]
        slon = sub[lon_col].to_numpy(dtype=float)[:, None]
        d = haversine_km(slon, slat, a_lon[None, :], a_lat[None, :])
        idx = np.argmin(d, axis=1)
        nearest_idx.append(idx)
        nearest_dist.append(d[np.arange(d.shape[0]), idx])
    idx = np.concatenate(nearest_idx)
    dist = np.concatenate(nearest_dist)
    out = pd.DataFrame(
        {
            f"{side}_airport_icao": a_icao[idx],
            f"{side}_airport_lat": a_lat[idx],
            f"{side}_airport_lon": a_lon[idx],
            f"{side}_airport_dist_km": dist,
        }
    )
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) ADS-C windows by icao/day
    adsc = pd.read_csv(args.adsc_csv, usecols=["icao24", "timestamp"])
    adsc["timestamp"] = pd.to_datetime(adsc["timestamp"], utc=True, errors="coerce")
    adsc = adsc.dropna(subset=["icao24", "timestamp"]).copy()
    adsc["icao24"] = adsc["icao24"].astype(str).str.lower()
    adsc["day"] = adsc["timestamp"].dt.strftime("%Y-%m-%d")
    windows = (
        adsc.groupby(["icao24", "day"], as_index=False)
        .agg(adsc_start=("timestamp", "min"), adsc_end=("timestamp", "max"), adsc_points=("timestamp", "count"))
        .reset_index(drop=True)
    )

    # 2) ADS-B flight summary by icao/day/flight_id
    adsb = pd.read_parquet(args.adsb_parquet, columns=["flight_id", "minute_ts", "lat", "lon", "adsb_icao"])
    adsb["minute_ts"] = pd.to_datetime(adsb["minute_ts"], utc=True, errors="coerce")
    adsb = adsb.dropna(subset=["flight_id", "minute_ts", "lat", "lon", "adsb_icao"]).copy()
    adsb["adsb_icao"] = adsb["adsb_icao"].astype(str).str.lower()
    adsb["day"] = adsb["minute_ts"].dt.strftime("%Y-%m-%d")
    adsb = adsb.sort_values(["flight_id", "minute_ts"])
    g = adsb.groupby(["adsb_icao", "day", "flight_id"], as_index=False)
    fsum = g.agg(
        flight_start=("minute_ts", "min"),
        flight_end=("minute_ts", "max"),
        start_lat=("lat", "first"),
        start_lon=("lon", "first"),
        end_lat=("lat", "last"),
        end_lon=("lon", "last"),
        points=("minute_ts", "count"),
    )
    fsum["duration_min"] = (fsum["flight_end"] - fsum["flight_start"]).dt.total_seconds() / 60.0

    # 3) Match windows to containing ADS-B flights (same icao/day)
    merged = windows.merge(fsum, left_on=["icao24", "day"], right_on=["adsb_icao", "day"], how="inner")
    merged = merged[(merged["flight_start"] <= merged["adsc_start"]) & (merged["flight_end"] >= merged["adsc_end"])].copy()
    if merged.empty:
        raise RuntimeError("No ADS-C windows matched to containing ADS-B flights in current data.")
    merged["contain_span_sec"] = (merged["flight_end"] - merged["flight_start"]).dt.total_seconds()
    merged["window_span_sec"] = (merged["adsc_end"] - merged["adsc_start"]).dt.total_seconds()
    merged = merged.sort_values(["icao24", "day", "contain_span_sec"], ascending=[True, True, True])
    best = merged.groupby(["icao24", "day"], as_index=False).head(1).copy()

    # 4) Cross-ocean filter by endpoint distance
    best["distance_km"] = haversine_km(best["start_lon"], best["start_lat"], best["end_lon"], best["end_lat"])
    cross = best[best["distance_km"] >= float(args.min_distance_km)].copy()
    cross = cross.sort_values(["distance_km", "duration_min"], ascending=[False, False]).drop_duplicates(
        subset=["flight_id", "flight_start", "flight_end"]
    )
    if len(cross) > int(args.target):
        cross = cross.head(int(args.target)).copy()

    # 5) Attach nearest airport coords (approximate dep/arr airport coordinates)
    airports = pd.read_csv(args.airports_csv, usecols=["icao", "lat", "lon"]).dropna().copy()
    dep = nearest_airports(cross.rename(columns={"start_lat": "dep_lat", "start_lon": "dep_lon"}), airports, "dep")
    arr = nearest_airports(cross.rename(columns={"end_lat": "arr_lat", "end_lon": "arr_lon"}), airports, "arr")
    cross = pd.concat([cross.reset_index(drop=True), dep, arr], axis=1)

    # 6) Save tabular outputs
    cross_path = out_dir / "adsc_corresponding_adsb_cross_ocean_top500.csv"
    cross.to_csv(cross_path, index=False)

    summary_path = out_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"adsc_windows={len(windows)}\n")
        f.write(f"matched_windows={len(best)}\n")
        f.write(f"cross_ocean_candidates={len(best[best['distance_km'] >= float(args.min_distance_km)])}\n")
        f.write(f"selected_count={len(cross)}\n")
        f.write(f"min_distance_km={args.min_distance_km}\n")
        f.write(f"target={args.target}\n")
        f.write(f"output_csv={cross_path}\n")

    # 7) Plot globe with new cross-ocean trajectories in light blue
    polys = read_shp_polygons(Path(args.globe_shp))
    fig = plt.figure(figsize=(10, 10), facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")
    th = np.linspace(0, 2 * math.pi, 720)
    ax.plot(np.cos(th), np.sin(th), color="#c9c9c9", lw=1.2, zorder=5)
    for seg in polys:
        lon = np.array([p[0] for p in seg], dtype=float)
        lat = np.array([p[1] for p in seg], dtype=float)
        x, y, vis = ortho_project(lon, lat, lon0_deg=args.center_lon, lat0_deg=args.center_lat)
        ax.plot(np.where(vis, x, np.nan), np.where(vis, y, np.nan), color="#d2d2d2", lw=0.6, alpha=0.8, zorder=6)

    for _, r in cross.iterrows():
        lons, lats = great_circle_points(r["start_lon"], r["start_lat"], r["end_lon"], r["end_lat"], n=60)
        x, y, vis = ortho_project(lons, lats, lon0_deg=args.center_lon, lat0_deg=args.center_lat)
        ax.plot(np.where(vis, x, np.nan), np.where(vis, y, np.nan), color="#9ED8FF", lw=1.25, alpha=0.62, zorder=8)

    xs, ys, vs = ortho_project(cross["start_lon"].to_numpy(), cross["start_lat"].to_numpy(), lon0_deg=args.center_lon, lat0_deg=args.center_lat)
    ax.scatter(xs[vs], ys[vs], s=8, color="#9ED8FF", alpha=0.75, zorder=9)
    xe, ye, ve = ortho_project(cross["end_lon"].to_numpy(), cross["end_lat"].to_numpy(), lon0_deg=args.center_lon, lat0_deg=args.center_lat)
    ax.scatter(xe[ve], ye[ve], s=8, color="#9ED8FF", alpha=0.75, zorder=9)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axis("off")
    ax.set_title(f"ADS-C Corresponding Cross-Ocean ADS-B Trajectories (n={len(cross)})", fontsize=12, pad=12)
    out_png = out_dir / "adsc_corresponding_adsb_cross_ocean_globe.png"
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    plt.close(fig)

    print(f"[ok] selected={len(cross)} csv={cross_path}")
    print(f"[ok] globe={out_png}")
    print(f"[ok] summary={summary_path}")


if __name__ == "__main__":
    main()

