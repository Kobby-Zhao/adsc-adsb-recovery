from __future__ import annotations

import argparse
import struct
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--blue-manifest", default="outputs/runs/adsc_blue_aligned50_red_land_20260420/blue_manifest_strict.csv")
    p.add_argument("--red-manifest", default="outputs/runs/adsc_blue_aligned50_red_land_20260420/red_manifest_strict.csv")
    p.add_argument("--blue-flight-dir", default="outputs/runs/adsc_adsb_alignment_from_decoded_20260329_all50/aligned_series_flight_csv")
    p.add_argument("--adsb-parquet", default="outputs/mvp_merged_nostage_20260415/adsb_minute_merged.parquet")
    p.add_argument("--land-shp", default="ne_110m_land.shp")
    p.add_argument("--out-png", default="outputs/runs/adsc_adsb_real50_pack_20260419/adsc_adsb_real50_mollweide_blue_red.png")
    return p.parse_args()


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
                polygons.append(np.asarray(pts, dtype=float))
            else:
                idx = list(parts) + [len(pts)]
                for i in range(len(parts)):
                    seg = pts[idx[i] : idx[i + 1]]
                    if len(seg) >= 3:
                        polygons.append(np.asarray(seg, dtype=float))
    return polygons


def normalize_lon(lon: np.ndarray) -> np.ndarray:
    return ((lon + 180.0) % 360.0) - 180.0


def split_segments(lon_deg: np.ndarray, lat_deg: np.ndarray, jump_thresh: float = 100.0):
    if len(lon_deg) < 2:
        return [(lon_deg, lat_deg)]
    cuts = [0]
    d = np.abs(np.diff(lon_deg))
    for j in np.where(d > jump_thresh)[0]:
        cuts.append(j + 1)
    cuts.append(len(lon_deg))
    out = []
    for i in range(len(cuts) - 1):
        s, e = cuts[i], cuts[i + 1]
        if e - s >= 2:
            out.append((lon_deg[s:e], lat_deg[s:e]))
    return out


def load_blue(blue_manifest: Path, blue_flight_dir: Path):
    m = pd.read_csv(blue_manifest)
    trajs = []
    for fid in m["flight_id"].astype(str).tolist():
        fp = blue_flight_dir / f"{fid}.csv"
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        if "source" in df.columns:
            df = df[df["source"].astype(str).str.lower() == "adsb"].copy()
        if not {"lon", "lat"}.issubset(df.columns):
            continue
        dff = df[["lon", "lat"]].dropna()
        if len(dff) < 2:
            continue
        lon = normalize_lon(dff["lon"].to_numpy(float))
        lat = np.clip(dff["lat"].to_numpy(float), -89.9, 89.9)
        trajs.append((lon, lat))
    return trajs


def load_red(red_manifest: Path, adsb_parquet: Path):
    m = pd.read_csv(red_manifest)
    fids = set(m["flight_id"].astype(str).tolist())
    df = pd.read_parquet(adsb_parquet, columns=["flight_id", "lat", "lon"]) 
    df["flight_id"] = df["flight_id"].astype(str)
    df = df[df["flight_id"].isin(fids)].copy()
    trajs = []
    for _, g in df.groupby("flight_id"):
        g = g[["lon", "lat"]].dropna()
        if len(g) < 2:
            continue
        lon = normalize_lon(g["lon"].to_numpy(float))
        lat = np.clip(g["lat"].to_numpy(float), -89.9, 89.9)
        trajs.append((lon, lat))
    return trajs


def main():
    args = parse_args()
    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    blue_trajs = load_blue(Path(args.blue_manifest), Path(args.blue_flight_dir))
    red_trajs = load_red(Path(args.red_manifest), Path(args.adsb_parquet))
    land_polys = read_land_polygons(Path(args.land_shp))

    fig = plt.figure(figsize=(14, 7), facecolor="white")
    ax = fig.add_subplot(111, projection="mollweide")
    ax.set_facecolor("#f7f7f7")

    for poly in land_polys:
        lon_deg = normalize_lon(poly[:, 0])
        lat_deg = np.clip(poly[:, 1], -89.9, 89.9)
        for seg_lon, seg_lat in split_segments(lon_deg, lat_deg):
            x = np.radians(seg_lon)
            y = np.radians(seg_lat)
            ax.fill(x, y, facecolor="#efefef", edgecolor="#222222", linewidth=0.55, zorder=1)

    for lon_deg, lat_deg in red_trajs:
        for seg_lon, seg_lat in split_segments(lon_deg, lat_deg):
            ax.plot(np.radians(seg_lon), np.radians(seg_lat), color="#d62728", lw=0.8, alpha=0.28, zorder=2)

    for lon_deg, lat_deg in blue_trajs:
        for seg_lon, seg_lat in split_segments(lon_deg, lat_deg):
            ax.plot(np.radians(seg_lon), np.radians(seg_lat), color="#5aa9ff", lw=1.15, alpha=0.85, zorder=3)

    ax.grid(True, color="#cfcfcf", alpha=0.5, linewidth=0.6)
    ax.set_title(f"ADS-C-Corresponding (Blue) vs Land ADS-B (Red) | blue={len(blue_trajs)} red={len(red_trajs)}", fontsize=19, pad=12)

    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] blue={len(blue_trajs)} red={len(red_trajs)}")
    print(f"[ok] out={out_png}")


if __name__ == "__main__":
    main()
