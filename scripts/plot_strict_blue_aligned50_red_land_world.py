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
    p = argparse.ArgumentParser(description="Strict world map: blue from aligned50 minute ADS-B, red from pure-land ADS-B.")
    p.add_argument(
        "--blue-flight-dir",
        default="outputs/runs/adsc_adsb_alignment_from_decoded_20260329_all50/aligned_series_flight_csv",
    )
    p.add_argument("--adsc-csv", default="ads-c_data/adsc_decoded_2024-05-01_to_2024-12.csv")
    p.add_argument("--adsb-parquet", default="outputs/mvp_merged_nostage_20260415/adsb_minute_merged.parquet")
    p.add_argument("--land-shp", default="ne_110m_land.shp")
    p.add_argument("--out-dir", default="outputs/runs/adsc_blue_aligned50_red_land_20260420")
    p.add_argument("--blue-min-adsc-points", type=int, default=2)
    p.add_argument("--blue-max-land-ratio", type=float, default=0.95)
    p.add_argument("--red-target", type=int, default=450)
    p.add_argument("--red-max-dist-km", type=float, default=1800.0)
    p.add_argument("--red-min-land-ratio", type=float, default=0.99)
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


def build_land_index(polygons):
    paths, bboxes = [], []
    for poly in polygons:
        lons = unwrap_lons(poly[:, 0])
        lats = poly[:, 1]
        arr = np.column_stack([lons, lats])
        paths.append(MplPath(arr))
        bboxes.append((lons.min(), lons.max(), lats.min(), lats.max()))
    return paths, bboxes


def points_on_land(lons: np.ndarray, lats: np.ndarray, paths, bboxes):
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

    # ADS-C index for overlap counting
    adsc = pd.read_csv(args.adsc_csv, usecols=["icao24", "timestamp"])
    adsc["timestamp"] = pd.to_datetime(adsc["timestamp"], utc=True, errors="coerce")
    adsc = adsc.dropna(subset=["icao24", "timestamp"]).copy()
    adsc["icao24"] = adsc["icao24"].astype(str).str.lower()
    adsc_idx = {k: v["timestamp"].astype("int64").to_numpy() for k, v in adsc.groupby("icao24")}

    polygons = read_land_polygons(Path(args.land_shp))
    paths, bboxes = build_land_index(polygons)

    # Blue set from aligned50 (minute ADS-B rows only)
    blue_rows = []
    blue_trajs = {}
    for fp in sorted(Path(args.blue_flight_dir).glob("*.csv")):
        df = pd.read_csv(fp)
        if "source" in df.columns:
            df = df[df["source"].astype(str).str.lower() == "adsb"].copy()
        if df.empty:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp", "lat", "lon", "icao24"]).sort_values("timestamp")
        if len(df) < 3:
            continue
        icao = str(df["icao24"].iloc[0]).lower()
        start_ts = df["timestamp"].iloc[0]
        end_ts = df["timestamp"].iloc[-1]
        ts = adsc_idx.get(icao, np.array([], dtype=np.int64))
        start_ns = int(pd.Timestamp(start_ts).value)
        end_ns = int(pd.Timestamp(end_ts).value)
        adsc_n = int(((ts >= start_ns) & (ts <= end_ns)).sum()) if len(ts) > 0 else 0

        lons = unwrap_lons(df["lon"].to_numpy(dtype=float))
        lats = df["lat"].to_numpy(dtype=float)
        on_land = points_on_land(lons, lats, paths, bboxes)
        land_ratio = float(on_land.mean()) if len(on_land) else 1.0
        dist = float(haversine_km(lons[0], lats[0], lons[-1], lats[-1]))

        if adsc_n < int(args.blue_min_adsc_points):
            continue
        if land_ratio > float(args.blue_max_land_ratio):
            continue

        fid = fp.stem
        blue_trajs[fid] = (lons, lats)
        blue_rows.append(
            {
                "flight_id": fid,
                "icao24": icao,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "points": int(len(df)),
                "adsc_points_in_flight": adsc_n,
                "land_ratio": land_ratio,
                "ocean_ratio": 1.0 - land_ratio,
                "dist_km": dist,
            }
        )

    blue_manifest = pd.DataFrame(blue_rows).sort_values(["adsc_points_in_flight", "ocean_ratio"], ascending=[False, False])
    blue_manifest.to_csv(out_dir / "blue_manifest_strict.csv", index=False)

    # Red set from pure ADS-B merged (land flights, no ADS-C overlap)
    adsb = pd.read_parquet(args.adsb_parquet, columns=["flight_id", "minute_ts", "lat", "lon", "adsb_icao"])
    adsb["minute_ts"] = pd.to_datetime(adsb["minute_ts"], utc=True, errors="coerce")
    adsb = adsb.dropna(subset=["flight_id", "minute_ts", "lat", "lon", "adsb_icao"]).copy()
    adsb["adsb_icao"] = adsb["adsb_icao"].astype(str).str.lower()
    adsb = adsb.sort_values(["flight_id", "minute_ts"])
    g = adsb.groupby("flight_id", as_index=False).agg(
        icao24=("adsb_icao", "first"),
        start_ts=("minute_ts", "min"),
        end_ts=("minute_ts", "max"),
        start_lat=("lat", "first"),
        start_lon=("lon", "first"),
        end_lat=("lat", "last"),
        end_lon=("lon", "last"),
    )
    g["dist_km"] = haversine_km(g["start_lon"], g["start_lat"], g["end_lon"], g["end_lat"])

    overlap = []
    for _, r in g.iterrows():
        ts = adsc_idx.get(r["icao24"], np.array([], dtype=np.int64))
        if len(ts) == 0:
            overlap.append(0)
            continue
        start_ns = int(pd.Timestamp(r["start_ts"]).value)
        end_ns = int(pd.Timestamp(r["end_ts"]).value)
        overlap.append(int(((ts >= start_ns) & (ts <= end_ns)).sum()))
    g["adsc_points_in_flight"] = overlap

    red_rows = []
    red_trajs = {}
    cand = g[(g["adsc_points_in_flight"] == 0) & (g["dist_km"] <= float(args.red_max_dist_km))].copy()
    for _, r in cand.iterrows():
        sub = adsb[adsb["flight_id"] == r["flight_id"]]
        lons = unwrap_lons(sub["lon"].to_numpy(dtype=float))
        lats = sub["lat"].to_numpy(dtype=float)
        on_land = points_on_land(lons, lats, paths, bboxes)
        land_ratio = float(on_land.mean()) if len(on_land) else 1.0
        if land_ratio < float(args.red_min_land_ratio):
            continue
        fid = str(r["flight_id"])
        red_trajs[fid] = (lons, lats)
        red_rows.append(
            {
                "flight_id": fid,
                "icao24": r["icao24"],
                "start_ts": r["start_ts"],
                "end_ts": r["end_ts"],
                "dist_km": float(r["dist_km"]),
                "adsc_points_in_flight": int(r["adsc_points_in_flight"]),
                "land_ratio": land_ratio,
                "ocean_ratio": 1.0 - land_ratio,
                "points": int(len(sub)),
            }
        )
        if len(red_rows) >= int(args.red_target):
            break
    red_manifest = pd.DataFrame(red_rows)
    red_manifest.to_csv(out_dir / "red_manifest_strict.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 7), facecolor="white")
    ax.set_facecolor("#e9f4ff")
    for poly in polygons:
        lon = unwrap_lons(poly[:, 0])
        lat = poly[:, 1]
        if lon.min() < -220 or lon.max() > 220:
            continue
        ax.fill(lon, lat, facecolor="#d0d0d0", edgecolor="#b0b0b0", linewidth=0.25, zorder=1)

    for lons, lats in red_trajs.values():
        ax.plot(lons, lats, color="#d95f5f", lw=0.7, alpha=0.35, zorder=2)
    for lons, lats in blue_trajs.values():
        ax.plot(lons, lats, color="#8fd3ff", lw=1.1, alpha=0.78, zorder=3)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-75, 85)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.14, linewidth=0.5)
    ax.set_title(
        f"Strict Minute ADS-B | Blue: aligned ADS-C-corresponding (n={len(blue_manifest)}) | "
        f"Red: pure-land ADS-B (n={len(red_manifest)})"
    )
    out_png = out_dir / "strict_blue_aligned50_red_land_world_map.png"
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    plt.close(fig)

    with (out_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"blue_selected={len(blue_manifest)}\n")
        f.write(f"red_selected={len(red_manifest)}\n")
        f.write(f"blue_min_adsc_points={args.blue_min_adsc_points}\n")
        f.write(f"blue_max_land_ratio={args.blue_max_land_ratio}\n")
        f.write(f"red_max_dist_km={args.red_max_dist_km}\n")
        f.write(f"red_min_land_ratio={args.red_min_land_ratio}\n")
        f.write(f"output_png={out_png}\n")

    print(f"[ok] blue={len(blue_manifest)} red={len(red_manifest)}")
    print(f"[ok] plot={out_png}")


if __name__ == "__main__":
    main()

