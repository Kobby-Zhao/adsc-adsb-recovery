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
    p = argparse.ArgumentParser(description="Plot ADS-B trajectories on a Mollweide world map.")
    p.add_argument(
        "--manifest-csv",
        default="outputs/runs/adsc_adsb_real50_pack_20260419/adsc_adsb_real50_manifest.csv",
    )
    p.add_argument(
        "--land-shp",
        default="ne_110m_land.shp",
    )
    p.add_argument(
        "--out-png",
        default="outputs/runs/adsc_adsb_real50_pack_20260419/adsc_adsb_real50_mollweide_red.png",
    )
    p.add_argument("--line-color", default="#d62728")
    p.add_argument("--line-width", type=float, default=1.0)
    p.add_argument("--line-alpha", type=float, default=0.75)
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
    jump_idx = np.where(d > jump_thresh)[0]
    for j in jump_idx:
        cuts.append(j + 1)
    cuts.append(len(lon_deg))
    out = []
    for i in range(len(cuts) - 1):
        s, e = cuts[i], cuts[i + 1]
        if e - s >= 2:
            out.append((lon_deg[s:e], lat_deg[s:e]))
    return out


def load_trajectories(manifest_csv: Path):
    man = pd.read_csv(manifest_csv)
    trajs = []
    for _, r in man.iterrows():
        p = Path(str(r.get("flight_series_csv", "")))
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "source" in df.columns:
            df = df[df["source"].astype(str).str.lower() == "adsb"].copy()
        cols = {c.lower(): c for c in df.columns}
        lon_col = cols.get("lon")
        lat_col = cols.get("lat")
        if lon_col is None or lat_col is None:
            continue
        dff = df[[lon_col, lat_col]].dropna()
        if len(dff) < 2:
            continue
        lon = normalize_lon(dff[lon_col].to_numpy(dtype=float))
        lat = np.clip(dff[lat_col].to_numpy(dtype=float), -89.9, 89.9)
        trajs.append((lon, lat))
    return trajs


def main() -> None:
    args = parse_args()
    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    trajs = load_trajectories(Path(args.manifest_csv))
    land_polys = read_land_polygons(Path(args.land_shp))

    fig = plt.figure(figsize=(14, 7), facecolor="white")
    ax = fig.add_subplot(111, projection="mollweide")
    ax.set_facecolor("#f7f7f7")

    # land fill + coast edge
    for poly in land_polys:
        lon_deg = normalize_lon(poly[:, 0])
        lat_deg = np.clip(poly[:, 1], -89.9, 89.9)
        for seg_lon, seg_lat in split_segments(lon_deg, lat_deg):
            x = np.radians(seg_lon)
            y = np.radians(seg_lat)
            ax.fill(x, y, facecolor="#efefef", edgecolor="#202020", linewidth=0.6, zorder=1)

    # trajectories
    for lon_deg, lat_deg in trajs:
        for seg_lon, seg_lat in split_segments(lon_deg, lat_deg):
            x = np.radians(seg_lon)
            y = np.radians(seg_lat)
            ax.plot(x, y, color=args.line_color, lw=args.line_width, alpha=args.line_alpha, zorder=3)

    ax.grid(True, color="#cfcfcf", alpha=0.5, linewidth=0.6)
    ax.set_title("ADS-B Flight Trajectory", fontsize=26, pad=16)

    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] traj_count={len(trajs)}")
    print(f"[ok] out_png={out_png}")


if __name__ == "__main__":
    main()
