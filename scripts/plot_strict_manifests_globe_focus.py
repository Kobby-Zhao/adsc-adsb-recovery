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
    p = argparse.ArgumentParser(description="Globe plot focused on dense blue routes from strict manifests.")
    p.add_argument("--blue-manifest", default="outputs/runs/adsc_blue_aligned50_red_land_20260420/blue_manifest_strict.csv")
    p.add_argument("--red-manifest", default="outputs/runs/adsc_blue_aligned50_red_land_20260420/red_manifest_strict.csv")
    p.add_argument(
        "--blue-flight-dir",
        default="outputs/runs/adsc_adsb_alignment_from_decoded_20260329_all50/aligned_series_flight_csv",
    )
    p.add_argument("--adsb-parquet", default="outputs/mvp_merged_nostage_20260415/adsb_minute_merged.parquet")
    p.add_argument("--land-shp", default="ne_110m_land.shp")
    p.add_argument("--out-dir", default="outputs/runs/adsc_blue_aligned50_red_land_20260420")
    return p.parse_args()


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
                polys.append(np.asarray(pts, dtype=float))
            else:
                idx = list(parts) + [len(pts)]
                for i in range(len(parts)):
                    seg = pts[idx[i] : idx[i + 1]]
                    if len(seg) >= 3:
                        polys.append(np.asarray(seg, dtype=float))
    return polys


def ortho_project(lon_deg, lat_deg, lon0_deg, lat0_deg):
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    lon0 = math.radians(float(lon0_deg))
    lat0 = math.radians(float(lat0_deg))
    cosc = np.sin(lat0) * np.sin(lat) + np.cos(lat0) * np.cos(lat) * np.cos(lon - lon0)
    visible = cosc >= 0
    x = np.cos(lat) * np.sin(lon - lon0)
    y = np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(lon - lon0)
    return x, y, visible


def circular_mean_deg(lons: np.ndarray) -> float:
    r = np.deg2rad(lons)
    s = np.sin(r).mean()
    c = np.cos(r).mean()
    return float(np.rad2deg(np.arctan2(s, c)))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    blue_m = pd.read_csv(args.blue_manifest)
    red_m = pd.read_csv(args.red_manifest)
    blue_ids = set(blue_m["flight_id"].astype(str))
    red_ids = set(red_m["flight_id"].astype(str))

    # blue trajectories from aligned flight csv (source=adsb)
    blue_traj = {}
    blue_lons_all, blue_lats_all = [], []
    for fid in blue_ids:
        fp = Path(args.blue_flight_dir) / f"{fid}.csv"
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        if "source" in df.columns:
            df = df[df["source"].astype(str).str.lower() == "adsb"]
        df = df.dropna(subset=["lon", "lat"])
        if len(df) < 2:
            continue
        lons = df["lon"].to_numpy(dtype=float)
        lats = df["lat"].to_numpy(dtype=float)
        blue_traj[fid] = (lons, lats)
        blue_lons_all.append(lons)
        blue_lats_all.append(lats)

    # red trajectories from adsb parquet by flight_id
    adsb = pd.read_parquet(args.adsb_parquet, columns=["flight_id", "minute_ts", "lon", "lat"])
    adsb["flight_id"] = adsb["flight_id"].astype(str)
    adsb = adsb[adsb["flight_id"].isin(red_ids)].dropna(subset=["lon", "lat"]).copy()
    adsb["minute_ts"] = pd.to_datetime(adsb["minute_ts"], utc=True, errors="coerce")
    adsb = adsb.dropna(subset=["minute_ts"]).sort_values(["flight_id", "minute_ts"])
    red_traj = {}
    for fid, g in adsb.groupby("flight_id"):
        if len(g) < 2:
            continue
        red_traj[fid] = (g["lon"].to_numpy(dtype=float), g["lat"].to_numpy(dtype=float))

    if blue_lons_all:
        lons = np.concatenate(blue_lons_all)
        lats = np.concatenate(blue_lats_all)
        center_lon = circular_mean_deg(lons)
        center_lat = float(np.clip(np.median(lats), -75, 75))
    else:
        center_lon, center_lat = 165.0, 38.0

    polys = read_land_polygons(Path(args.land_shp))
    fig = plt.figure(figsize=(10, 10), facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#e9f4ff")
    th = np.linspace(0, 2 * math.pi, 720)
    ax.plot(np.cos(th), np.sin(th), color="#c9c9c9", lw=1.2, zorder=5)

    for poly in polys:
        x, y, vis = ortho_project(poly[:, 0], poly[:, 1], center_lon, center_lat)
        if vis.sum() >= 3 and (vis.mean() > 0.85):
            ax.fill(x[vis], y[vis], facecolor="#d0d0d0", edgecolor="none", alpha=1.0, zorder=6)
        ax.plot(np.where(vis, x, np.nan), np.where(vis, y, np.nan), color="#b3b3b3", lw=0.45, alpha=0.8, zorder=7)

    for lons, lats in red_traj.values():
        x, y, vis = ortho_project(lons, lats, center_lon, center_lat)
        ax.plot(np.where(vis, x, np.nan), np.where(vis, y, np.nan), color="#d95f5f", lw=0.7, alpha=0.30, zorder=8)

    for lons, lats in blue_traj.values():
        x, y, vis = ortho_project(lons, lats, center_lon, center_lat)
        ax.plot(np.where(vis, x, np.nan), np.where(vis, y, np.nan), color="#8fd3ff", lw=1.2, alpha=0.80, zorder=9)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axis("off")
    ax.set_title(
        f"Focused Globe (Blue-dense view) | Blue={len(blue_traj)} strict ADS-C-corresponding, Red={len(red_traj)} land",
        fontsize=11,
        pad=12,
    )
    out_png = out_dir / "strict_blue_dense_focus_globe.png"
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    plt.close(fig)

    with (out_dir / "strict_blue_dense_focus_globe_summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"center_lon={center_lon}\n")
        f.write(f"center_lat={center_lat}\n")
        f.write(f"blue_traj_count={len(blue_traj)}\n")
        f.write(f"red_traj_count={len(red_traj)}\n")
        f.write(f"output_png={out_png}\n")

    print(f"[ok] center=({center_lon:.2f},{center_lat:.2f}) blue={len(blue_traj)} red={len(red_traj)}")
    print(f"[ok] png={out_png}")


if __name__ == "__main__":
    main()

