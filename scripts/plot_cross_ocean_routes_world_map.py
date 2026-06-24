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
    p = argparse.ArgumentParser(description="Plot blue cross-ocean + red land routes on a common world-map view.")
    p.add_argument(
        "--blue-csv",
        default="outputs/runs/adsc_corresponding_adsb500_20260420_local/adsc_corresponding_adsb_top500_routes.csv",
    )
    p.add_argument(
        "--red-csv",
        default="outputs/runs/adsb_global_distribution_20260420/adsb_global_distribution_endpoints.csv",
    )
    p.add_argument("--red-distance-max-km", type=float, default=1800.0)
    p.add_argument("--red-max-routes", type=int, default=1200)
    p.add_argument("--shp", default="ne_110m_land.shp")
    p.add_argument("--out-dir", default="outputs/runs/adsc_corresponding_adsb500_20260420_world_view")
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


def read_land_polygons(shp_path: Path):
    polys = []
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
            if shape_type != 5:
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
                    if len(seg) >= 3:
                        polys.append(seg)
    return polys


def unwrap_lons(lons: np.ndarray):
    out = lons.copy().astype(float)
    for i in range(1, len(out)):
        d = out[i] - out[i - 1]
        if d > 180:
            out[i:] -= 360
        elif d < -180:
            out[i:] += 360
    return out


def great_circle_points(lon1, lat1, lon2, lat2, n=70):
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
    return unwrap_lons(lons), lats


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    blue = pd.read_csv(args.blue_csv)
    need_blue = {"start_lon", "start_lat", "end_lon", "end_lat"}
    if not need_blue.issubset(set(blue.columns)):
        raise RuntimeError(f"Blue CSV missing columns: {need_blue - set(blue.columns)}")

    red = pd.read_csv(args.red_csv)
    need_red = {"lon_start", "lat_start", "lon_end", "lat_end"}
    if not need_red.issubset(set(red.columns)):
        raise RuntimeError(f"Red CSV missing columns: {need_red - set(red.columns)}")
    if "dist_km" not in red.columns:
        red["dist_km"] = haversine_km(red["lon_start"], red["lat_start"], red["lon_end"], red["lat_end"])
    red = red[red["dist_km"] <= float(args.red_distance_max_km)].copy()
    if len(red) > int(args.red_max_routes):
        red = red.sample(int(args.red_max_routes), random_state=42)

    land = read_land_polygons(Path(args.shp))

    fig, ax = plt.subplots(figsize=(14, 7), facecolor="white")
    ax.set_facecolor("#e9f4ff")

    for seg in land:
        lon = np.array([p[0] for p in seg], dtype=float)
        lat = np.array([p[1] for p in seg], dtype=float)
        lon = unwrap_lons(lon)
        if lon.min() < -200 or lon.max() > 200:
            # skip pathological wrap polygons in this simple view
            continue
        ax.fill(lon, lat, facecolor="#d0d0d0", edgecolor="#b0b0b0", linewidth=0.3, zorder=1)

    for _, r in red.iterrows():
        lons, lats = great_circle_points(r["lon_start"], r["lat_start"], r["lon_end"], r["lat_end"], n=56)
        ax.plot(lons, lats, color="#d95f5f", lw=0.8, alpha=0.35, zorder=2)

    for _, r in blue.iterrows():
        lons, lats = great_circle_points(r["start_lon"], r["start_lat"], r["end_lon"], r["end_lat"], n=64)
        ax.plot(lons, lats, color="#8fd3ff", lw=1.15, alpha=0.72, zorder=3)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-75, 85)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.15, linewidth=0.5)
    ax.set_title(f"Common World View: Blue=ADS-C Corresponding Cross-Ocean (n={len(blue)}), Red=Land Routes")

    out_png = out_dir / "adsc_corresponding_routes_world_common_view.png"
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    plt.close(fig)

    with (out_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"blue_count={len(blue)}\n")
        f.write(f"red_count={len(red)}\n")
        f.write(f"red_distance_max_km={args.red_distance_max_km}\n")
        f.write(f"output_png={out_png}\n")

    print(f"[ok] png={out_png}")


if __name__ == "__main__":
    main()

