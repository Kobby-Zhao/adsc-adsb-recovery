from __future__ import annotations

import argparse
import struct
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strict world-map plot using minute-level pure ADS-B trajectories.")
    p.add_argument("--adsb-parquet", default="outputs/mvp_merged_nostage_20260415/adsb_minute_merged.parquet")
    p.add_argument("--adsc-csv", default="ads-c_data/adsc_decoded_2024-05-01_to_2024-12.csv")
    p.add_argument("--land-shp", default="ne_110m_land.shp")
    p.add_argument("--out-dir", default="outputs/runs/adsc_adsb_strict_worldmap_20260420")
    p.add_argument("--blue-target", type=int, default=500)
    p.add_argument("--red-target", type=int, default=600)
    p.add_argument("--blue-min-adsc-points", type=int, default=2)
    p.add_argument(
        "--blue-adsc-link-mode",
        choices=["strict_window", "icao_day"],
        default="icao_day",
        help="How to link ADS-C points to ADS-B flights for blue set.",
    )
    p.add_argument("--blue-min-dist-km", type=float, default=2500.0)
    p.add_argument("--blue-max-land-ratio", type=float, default=0.85)
    p.add_argument("--red-max-dist-km", type=float, default=1800.0)
    p.add_argument("--red-min-land-ratio", type=float, default=0.98)
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


def unwrap_lons(lons: np.ndarray) -> np.ndarray:
    out = lons.copy().astype(float)
    for i in range(1, len(out)):
        d = out[i] - out[i - 1]
        if d > 180:
            out[i:] -= 360
        elif d < -180:
            out[i:] += 360
    return out


def read_land_polygons(shp_path: Path):
    polygons = []
    with shp_path.open("rb") as f:
        _ = f.read(100)
        while True:
            rec_header = f.read(8)
            if not rec_header or len(rec_header) < 8:
                break
            _, rec_len_words = struct.unpack(">2i", rec_header)
            rec = f.read(rec_len_words * 2)
            if len(rec) < 44:
                continue
            shape_type = struct.unpack("<i", rec[:4])[0]
            if shape_type != 5:  # polygon
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
                polygons.append(np.asarray(pts, dtype=float))
            else:
                idx = list(parts) + [len(pts)]
                for i in range(len(parts)):
                    seg = pts[idx[i] : idx[i + 1]]
                    if len(seg) >= 3:
                        polygons.append(np.asarray(seg, dtype=float))
    return polygons


def build_land_index(polygons):
    paths = []
    bboxes = []
    for poly in polygons:
        lons = unwrap_lons(poly[:, 0])
        lats = poly[:, 1]
        arr = np.column_stack([lons, lats])
        paths.append(MplPath(arr))
        bboxes.append((lons.min(), lons.max(), lats.min(), lats.max()))
    return paths, bboxes


def points_on_land(lons: np.ndarray, lats: np.ndarray, paths, bboxes) -> np.ndarray:
    pts = np.column_stack([lons, lats])
    on_land = np.zeros(len(pts), dtype=bool)
    for path, (xmin, xmax, ymin, ymax) in zip(paths, bboxes):
        mask = (~on_land) & (lons >= xmin) & (lons <= xmax) & (lats >= ymin) & (lats <= ymax)
        if not mask.any():
            continue
        inside = path.contains_points(pts[mask])
        if inside.any():
            idx = np.where(mask)[0]
            on_land[idx[inside]] = True
    return on_land


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[info] loading data ...")
    adsb = pd.read_parquet(args.adsb_parquet, columns=["flight_id", "minute_ts", "lat", "lon", "adsb_icao"])
    adsb["minute_ts"] = pd.to_datetime(adsb["minute_ts"], utc=True, errors="coerce")
    adsb = adsb.dropna(subset=["flight_id", "minute_ts", "lat", "lon", "adsb_icao"]).copy()
    adsb["adsb_icao"] = adsb["adsb_icao"].astype(str).str.lower()
    adsb = adsb.sort_values(["flight_id", "minute_ts"])

    adsc = pd.read_csv(args.adsc_csv, usecols=["icao24", "timestamp"])
    adsc["timestamp"] = pd.to_datetime(adsc["timestamp"], utc=True, errors="coerce")
    adsc = adsc.dropna(subset=["icao24", "timestamp"]).copy()
    adsc["icao24"] = adsc["icao24"].astype(str).str.lower()
    adsc["day"] = adsc["timestamp"].dt.strftime("%Y-%m-%d")

    print("[info] building flight summary ...")
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
    g["dist_km"] = haversine_km(g["start_lon"], g["start_lat"], g["end_lon"], g["end_lat"])

    print(f"[info] counting ADS-C points per flight (mode={args.blue_adsc_link_mode}) ...")
    if args.blue_adsc_link_mode == "strict_window":
        adsc_group = {k: v["timestamp"].astype("int64").to_numpy() for k, v in adsc.groupby("icao24")}
        overlap = []
        for _, r in g.iterrows():
            ts = adsc_group.get(r["icao24"])
            if ts is None or len(ts) == 0:
                overlap.append(0)
                continue
            start_ns = int(pd.Timestamp(r["start_ts"]).value)
            end_ns = int(pd.Timestamp(r["end_ts"]).value)
            n = int(((ts >= start_ns) & (ts <= end_ns)).sum())
            overlap.append(n)
        g["adsc_points_in_flight"] = overlap
    else:
        g["day"] = pd.to_datetime(g["start_ts"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
        cnt = adsc.groupby(["icao24", "day"]).size().rename("adsc_points_in_flight").reset_index()
        g = g.merge(cnt, on=["icao24", "day"], how="left")
        g["adsc_points_in_flight"] = g["adsc_points_in_flight"].fillna(0).astype(int)

    print("[info] loading land polygons and computing land ratio ...")
    polygons = read_land_polygons(Path(args.land_shp))
    paths, bboxes = build_land_index(polygons)

    land_ratio = {}
    for fid, sub in adsb.groupby("flight_id"):
        lons = unwrap_lons(sub["lon"].to_numpy(dtype=float))
        lats = sub["lat"].to_numpy(dtype=float)
        mask = points_on_land(lons, lats, paths, bboxes)
        land_ratio[fid] = float(mask.mean()) if len(mask) > 0 else 1.0
    g["land_ratio"] = g["flight_id"].map(land_ratio).fillna(1.0)
    g["ocean_ratio"] = 1.0 - g["land_ratio"]

    # Blue: ADS-C-corresponding, >=2 ADS-C points, cross-ocean leaning
    blue = g[
        (g["adsc_points_in_flight"] >= int(args.blue_min_adsc_points))
        & (g["dist_km"] >= float(args.blue_min_dist_km))
        & (g["land_ratio"] <= float(args.blue_max_land_ratio))
    ].copy()
    blue = blue.sort_values(["adsc_points_in_flight", "dist_km", "ocean_ratio"], ascending=[False, False, False])
    if len(blue) > int(args.blue_target):
        blue = blue.head(int(args.blue_target)).copy()

    # Red: pure ADS-B land routes (no ADS-C overlap), short/medium distance
    red = g[
        (g["adsc_points_in_flight"] == 0)
        & (g["dist_km"] <= float(args.red_max_dist_km))
        & (g["land_ratio"] >= float(args.red_min_land_ratio))
    ].copy()
    red = red.sort_values(["land_ratio", "duration_min"], ascending=[False, False])
    if len(red) > int(args.red_target):
        red = red.head(int(args.red_target)).copy()

    blue.to_csv(out_dir / "blue_manifest_strict.csv", index=False)
    red.to_csv(out_dir / "red_manifest_strict.csv", index=False)

    blue_ids = set(blue["flight_id"].tolist())
    red_ids = set(red["flight_id"].tolist())
    plot_df = adsb[adsb["flight_id"].isin(blue_ids | red_ids)].copy()

    print(f"[info] selected blue={len(blue)} red={len(red)} plotting trajectories ...")
    fig, ax = plt.subplots(figsize=(14, 7), facecolor="white")
    ax.set_facecolor("#e9f4ff")

    # land fill
    for poly in polygons:
        lon = unwrap_lons(poly[:, 0])
        lat = poly[:, 1]
        if lon.min() < -220 or lon.max() > 220:
            continue
        ax.fill(lon, lat, facecolor="#d0d0d0", edgecolor="#b0b0b0", linewidth=0.25, zorder=1)

    # red pure-land minute trajectories
    for fid, sub in plot_df[plot_df["flight_id"].isin(red_ids)].groupby("flight_id"):
        lons = unwrap_lons(sub["lon"].to_numpy(dtype=float))
        lats = sub["lat"].to_numpy(dtype=float)
        ax.plot(lons, lats, color="#d95f5f", lw=0.7, alpha=0.35, zorder=2)

    # blue ADS-C-corresponding minute trajectories
    for fid, sub in plot_df[plot_df["flight_id"].isin(blue_ids)].groupby("flight_id"):
        lons = unwrap_lons(sub["lon"].to_numpy(dtype=float))
        lats = sub["lat"].to_numpy(dtype=float)
        ax.plot(lons, lats, color="#8fd3ff", lw=1.0, alpha=0.75, zorder=3)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-75, 85)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.14, linewidth=0.5)
    ax.set_title(
        f"Strict Minute ADS-B Trajectories | Blue=ADS-C overlap >= {args.blue_min_adsc_points} (n={len(blue)}) "
        f"| Red=Pure land ADS-B (n={len(red)})"
    )
    out_png = out_dir / "strict_red_blue_minute_world_map.png"
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    plt.close(fig)

    with (out_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"total_adsb_flights={len(g)}\n")
        f.write(f"blue_selected={len(blue)}\n")
        f.write(f"red_selected={len(red)}\n")
        f.write(f"blue_min_adsc_points={args.blue_min_adsc_points}\n")
        f.write(f"blue_min_dist_km={args.blue_min_dist_km}\n")
        f.write(f"blue_max_land_ratio={args.blue_max_land_ratio}\n")
        f.write(f"red_max_dist_km={args.red_max_dist_km}\n")
        f.write(f"red_min_land_ratio={args.red_min_land_ratio}\n")
        f.write(f"output_png={out_png}\n")

    print(f"[ok] plot={out_png}")


if __name__ == "__main__":
    main()
