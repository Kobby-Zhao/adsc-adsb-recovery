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
    p = argparse.ArgumentParser(description="Build 500 ADS-C-corresponding ADS-B cross-ocean routes from local merged ADS-B.")
    p.add_argument("--adsc-csv", default="ads-c_data/adsc_decoded_2024-05-01_to_2024-12.csv")
    p.add_argument("--adsb-parquet", default="outputs/mvp_merged_nostage_20260415/adsb_minute_merged.parquet")
    p.add_argument("--airports-csv", default="outputs/flights/raw/airports.csv")
    p.add_argument("--target", type=int, default=500)
    p.add_argument("--min-distance-km", type=float, default=2900.0)
    p.add_argument("--out-dir", default="outputs/runs/adsc_corresponding_adsb500_20260420_local")
    p.add_argument("--globe-shp", default="ne_110m_land.shp")
    p.add_argument("--center-lon", type=float, default=20.0)
    p.add_argument("--center-lat", type=float, default=10.0)
    p.add_argument(
        "--red-routes-csv",
        default="outputs/runs/adsb_global_distribution_20260420/adsb_global_distribution_endpoints.csv",
        help="Reference routes drawn in red on same globe (optional).",
    )
    p.add_argument("--red-max-routes", type=int, default=1200)
    p.add_argument("--red-distance-min-km", type=float, default=0.0)
    p.add_argument("--red-distance-max-km", type=float, default=2000.0)
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
            if shape_type != 5:  # polygon only (land)
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


def nearest_airports(df: pd.DataFrame, airports: pd.DataFrame, side: str) -> pd.DataFrame:
    lat_col = f"{side}_lat"
    lon_col = f"{side}_lon"
    a_lat = airports["lat"].to_numpy(dtype=float)
    a_lon = airports["lon"].to_numpy(dtype=float)
    a_icao = airports["icao"].astype(str).to_numpy()
    out_rows = []
    for _, r in df.iterrows():
        d = haversine_km(a_lon, a_lat, float(r[lon_col]), float(r[lat_col]))
        idx = int(np.argmin(d))
        out_rows.append(
            {
                f"{side}_airport_icao": a_icao[idx],
                f"{side}_airport_lat": float(a_lat[idx]),
                f"{side}_airport_lon": float(a_lon[idx]),
                f"{side}_airport_dist_km": float(d[idx]),
            }
        )
    return pd.DataFrame(out_rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adsc_set = set(pd.read_csv(args.adsc_csv, usecols=["icao24"])["icao24"].astype(str).str.lower().unique())
    adsb = pd.read_parquet(args.adsb_parquet, columns=["flight_id", "minute_ts", "lat", "lon", "adsb_icao"])
    adsb["adsb_icao"] = adsb["adsb_icao"].astype(str).str.lower()
    adsb = adsb[adsb["adsb_icao"].isin(adsc_set)].copy()
    adsb["minute_ts"] = pd.to_datetime(adsb["minute_ts"], utc=True, errors="coerce")
    adsb = adsb.dropna(subset=["flight_id", "minute_ts", "lat", "lon"]).sort_values(["flight_id", "minute_ts"])

    g = adsb.groupby("flight_id", as_index=False).agg(
        icao24=("adsb_icao", "first"),
        start_ts=("minute_ts", "min"),
        end_ts=("minute_ts", "max"),
        start_lat=("lat", "first"),
        start_lon=("lon", "first"),
        end_lat=("lat", "last"),
        end_lon=("lon", "last"),
        points=("minute_ts", "count"),
    )
    g["duration_min"] = (g["end_ts"] - g["start_ts"]).dt.total_seconds().div(60.0)
    g["distance_km"] = haversine_km(g["start_lon"], g["start_lat"], g["end_lon"], g["end_lat"])
    cross = g[g["distance_km"] >= float(args.min_distance_km)].sort_values(["distance_km", "duration_min"], ascending=[False, False]).copy()
    if len(cross) > int(args.target):
        cross = cross.head(int(args.target)).copy()

    airports = pd.read_csv(args.airports_csv, usecols=["icao", "lat", "lon"]).dropna().copy()
    airports["icao"] = airports["icao"].astype(str).str.upper().str.strip()
    airports = airports[airports["icao"].str.match(r"^[A-Z0-9]{4}$", na=False)].copy()
    dep = nearest_airports(cross.rename(columns={"start_lat": "dep_lat", "start_lon": "dep_lon"}), airports, "dep")
    arr = nearest_airports(cross.rename(columns={"end_lat": "arr_lat", "end_lon": "arr_lon"}), airports, "arr")
    cross = pd.concat([cross.reset_index(drop=True), dep, arr], axis=1)

    out_csv = out_dir / "adsc_corresponding_adsb_top500_routes.csv"
    cross.to_csv(out_csv, index=False)

    polys = read_shp_polygons(Path(args.globe_shp))
    fig = plt.figure(figsize=(10, 10), facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#e9f4ff")  # ocean tone
    th = np.linspace(0, 2 * math.pi, 720)
    ax.plot(np.cos(th), np.sin(th), color="#c9c9c9", lw=1.2, zorder=5)
    for seg in polys:
        lon = np.array([p[0] for p in seg], dtype=float)
        lat = np.array([p[1] for p in seg], dtype=float)
        x, y, vis = ortho_project(lon, lat, lon0_deg=args.center_lon, lat0_deg=args.center_lat)
        # fill visible land in grey
        if vis.sum() >= 3 and (vis.mean() > 0.85):
            ax.fill(x[vis], y[vis], facecolor="#d0d0d0", edgecolor="none", alpha=1.0, zorder=6)
        ax.plot(np.where(vis, x, np.nan), np.where(vis, y, np.nan), color="#b3b3b3", lw=0.5, alpha=0.9, zorder=7)

    # optional red reference routes
    red_csv = Path(args.red_routes_csv)
    red_count = 0
    if red_csv.exists():
        red = pd.read_csv(red_csv)
        need = {"lon_start", "lat_start", "lon_end", "lat_end"}
        if need.issubset(set(red.columns)):
            if "dist_km" in red.columns:
                red = red[
                    (red["dist_km"] >= float(args.red_distance_min_km))
                    & (red["dist_km"] <= float(args.red_distance_max_km))
                ].copy()
            else:
                # compute distance if missing
                red["dist_km"] = haversine_km(
                    red["lon_start"], red["lat_start"], red["lon_end"], red["lat_end"]
                )
                red = red[
                    (red["dist_km"] >= float(args.red_distance_min_km))
                    & (red["dist_km"] <= float(args.red_distance_max_km))
                ].copy()
            if len(red) > int(args.red_max_routes):
                red = red.sample(int(args.red_max_routes), random_state=42).reset_index(drop=True)
            for _, rr in red.iterrows():
                lons, lats = great_circle_points(rr["lon_start"], rr["lat_start"], rr["lon_end"], rr["lat_end"], n=56)
                x, y, vis = ortho_project(lons, lats, lon0_deg=args.center_lon, lat0_deg=args.center_lat)
                ax.plot(np.where(vis, x, np.nan), np.where(vis, y, np.nan), color="#d95f5f", lw=0.9, alpha=0.30, zorder=8)
            red_count = len(red)

    # blue: newly collected ADS-C corresponding cross-ocean routes
    for _, r in cross.iterrows():
        lons, lats = great_circle_points(r["start_lon"], r["start_lat"], r["end_lon"], r["end_lat"], n=60)
        x, y, vis = ortho_project(lons, lats, lon0_deg=args.center_lon, lat0_deg=args.center_lat)
        ax.plot(np.where(vis, x, np.nan), np.where(vis, y, np.nan), color="#8fd3ff", lw=1.35, alpha=0.70, zorder=9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axis("off")
    ax.set_title(
        f"ADS-C-Corresponding Routes (Blue, n={len(cross)}) + Reference Routes (Red, n={red_count})",
        fontsize=11,
        pad=12,
    )
    out_png = out_dir / "adsc_corresponding_adsb_globe_red_blue_landfill.png"
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    plt.close(fig)

    with (out_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"adsc_unique_icao={len(adsc_set)}\n")
        f.write(f"matched_adsb_flights={g.shape[0]}\n")
        f.write(f"cross_candidates_before_cap={(g['distance_km'] >= float(args.min_distance_km)).sum()}\n")
        f.write(f"selected_count={len(cross)}\n")
        f.write(f"target={args.target}\n")
        f.write(f"min_distance_km={args.min_distance_km}\n")
        f.write(f"output_csv={out_csv}\n")
        f.write(f"output_png={out_png}\n")

    print(f"[ok] selected={len(cross)} (target={args.target})")
    print(f"[ok] csv={out_csv}")
    print(f"[ok] png={out_png}")


if __name__ == "__main__":
    main()
