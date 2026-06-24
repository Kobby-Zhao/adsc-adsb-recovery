from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Plot cross-ocean ADS-C anchor routes in green.")
    p.add_argument(
        "--adsc-csv",
        default="outputs/runs/adsc_flight_segmentation_4h_20260421_fix_epoch_v2/"
        "adsc_points_with_flight_id_4h_min2_latest1200_minute_agg.csv",
    )
    p.add_argument("--land-shp", default="ne_110m_land.shp")
    p.add_argument("--out-dir", default="outputs/runs/cross_ocean_adsc_anchor_routes_20260517")
    p.add_argument("--min-anchors", type=int, default=2)
    p.add_argument("--min-distance-km", type=float, default=2500.0)
    p.add_argument("--min-duration-min", type=float, default=120.0)
    p.add_argument("--max-avg-speed-kmh", type=float, default=1200.0)
    p.add_argument("--min-ocean-ratio", type=float, default=0.50)
    p.add_argument("--sample-points", type=int, default=80)
    p.add_argument("--max-routes", type=int, default=0, help="0 means plot all selected routes.")
    p.add_argument("--projection", choices=["mollweide", "platecarree"], default="mollweide")
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


def wrap_lon(lon: np.ndarray) -> np.ndarray:
    return ((lon + 180.0) % 360.0) - 180.0


def interpolate_great_circle(lon1: float, lat1: float, lon2: float, lat2: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Spherical linear interpolation between two lon/lat points."""
    n = max(2, int(n))
    lon1r, lat1r, lon2r, lat2r = map(math.radians, [lon1, lat1, lon2, lat2])
    p0 = np.array([math.cos(lat1r) * math.cos(lon1r), math.cos(lat1r) * math.sin(lon1r), math.sin(lat1r)])
    p1 = np.array([math.cos(lat2r) * math.cos(lon2r), math.cos(lat2r) * math.sin(lon2r), math.sin(lat2r)])
    dot = float(np.clip(np.dot(p0, p1), -1.0, 1.0))
    omega = math.acos(dot)
    if omega < 1e-8:
        lons = np.linspace(lon1, lon2, n)
        lats = np.linspace(lat1, lat2, n)
        return wrap_lon(lons), lats
    so = math.sin(omega)
    ts = np.linspace(0.0, 1.0, n)
    pts = (np.sin((1.0 - ts)[:, None] * omega) / so) * p0 + (np.sin(ts[:, None] * omega) / so) * p1
    lats = np.degrees(np.arctan2(pts[:, 2], np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)))
    lons = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    return wrap_lon(lons), lats


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
    paths = []
    bboxes = []
    for poly in polygons:
        lons = unwrap_lons(poly[:, 0])
        lats = poly[:, 1]
        paths.append(MplPath(np.column_stack([lons, lats])))
        bboxes.append((lons.min(), lons.max(), lats.min(), lats.max()))
    return paths, bboxes


def points_on_land(lons: np.ndarray, lats: np.ndarray, paths, bboxes) -> np.ndarray:
    lons = unwrap_lons(np.asarray(lons, dtype=float))
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


def split_dateline(lons: np.ndarray, lats: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    chunks = []
    start = 0
    for i in range(1, len(lons)):
        if abs(lons[i] - lons[i - 1]) > 180:
            chunks.append((lons[start:i], lats[start:i]))
            start = i
    chunks.append((lons[start:], lats[start:]))
    return [(x, y) for x, y in chunks if len(x) >= 2]


def plot_lonlat(ax, lon: np.ndarray, lat: np.ndarray, *args, projection: str, **kwargs):
    if projection == "mollweide":
        return ax.plot(np.radians(wrap_lon(np.asarray(lon, dtype=float))), np.radians(lat), *args, **kwargs)
    return ax.plot(lon, lat, *args, **kwargs)


def fill_lonlat(ax, lon: np.ndarray, lat: np.ndarray, *, projection: str, **kwargs):
    if projection == "mollweide":
        return ax.fill(np.radians(wrap_lon(np.asarray(lon, dtype=float))), np.radians(lat), **kwargs)
    return ax.fill(lon, lat, **kwargs)


def scatter_lonlat(ax, lon: np.ndarray, lat: np.ndarray, *, projection: str, **kwargs):
    if projection == "mollweide":
        return ax.scatter(np.radians(wrap_lon(np.asarray(lon, dtype=float))), np.radians(lat), **kwargs)
    return ax.scatter(lon, lat, **kwargs)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adsc = pd.read_csv(args.adsc_csv)
    adsc["timestamp"] = pd.to_datetime(adsc["timestamp"], utc=True, errors="coerce")
    adsc = adsc.dropna(subset=["flight_id", "timestamp", "latitude", "longitude"]).copy()
    adsc = adsc.sort_values(["flight_id", "timestamp"])

    polygons = read_land_polygons(Path(args.land_shp))
    paths, bboxes = build_land_index(polygons)

    rows = []
    route_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for flight_id, g in adsc.groupby("flight_id", sort=False):
        if len(g) < int(args.min_anchors):
            continue
        first = g.iloc[0]
        last = g.iloc[-1]
        dist_km = float(haversine_km(first["longitude"], first["latitude"], last["longitude"], last["latitude"]))
        lons, lats = interpolate_great_circle(
            float(first["longitude"]),
            float(first["latitude"]),
            float(last["longitude"]),
            float(last["latitude"]),
            int(args.sample_points),
        )
        on_land = points_on_land(lons, lats, paths, bboxes)
        ocean_ratio = float((~on_land).mean())
        route_cache[str(flight_id)] = (lons, lats)
        rows.append(
            {
                "flight_id": str(flight_id),
                "icao24": str(g["icao24"].iloc[0]),
                "anchor_count": int(len(g)),
                "start_ts": first["timestamp"].isoformat(),
                "end_ts": last["timestamp"].isoformat(),
                "duration_min": float((last["timestamp"] - first["timestamp"]).total_seconds() / 60.0),
                "start_lat": float(first["latitude"]),
                "start_lon": float(first["longitude"]),
                "end_lat": float(last["latitude"]),
                "end_lon": float(last["longitude"]),
                "endpoint_distance_km": dist_km,
                "avg_endpoint_speed_kmh": dist_km / max(1e-6, float((last["timestamp"] - first["timestamp"]).total_seconds() / 3600.0)),
                "ocean_ratio_gc": ocean_ratio,
                "land_ratio_gc": 1.0 - ocean_ratio,
            }
        )

    summary = pd.DataFrame(rows)
    selected = summary[
        (summary["anchor_count"] >= int(args.min_anchors))
        & (summary["endpoint_distance_km"] >= float(args.min_distance_km))
        & (summary["duration_min"] >= float(args.min_duration_min))
        & (summary["avg_endpoint_speed_kmh"] <= float(args.max_avg_speed_kmh))
        & (summary["ocean_ratio_gc"] >= float(args.min_ocean_ratio))
    ].copy()
    selected = selected.sort_values(["ocean_ratio_gc", "endpoint_distance_km", "anchor_count"], ascending=[False, False, False])
    if int(args.max_routes) > 0 and len(selected) > int(args.max_routes):
        selected = selected.head(int(args.max_routes)).copy()

    summary.to_csv(out_dir / "adsc_anchor_route_summary_all.csv", index=False)
    selected.to_csv(out_dir / "cross_ocean_adsc_anchor_routes_selected.csv", index=False)

    subplot_kw = {"projection": "mollweide"} if args.projection == "mollweide" else {}
    fig, ax = plt.subplots(figsize=(14, 7), facecolor="white", subplot_kw=subplot_kw)
    ax.set_facecolor("#f8faf7")
    for poly in polygons:
        lon = poly[:, 0] if args.projection == "mollweide" else unwrap_lons(poly[:, 0])
        lat = poly[:, 1]
        if args.projection != "mollweide" and (lon.min() < -220 or lon.max() > 220):
            continue
        poly_chunks = split_dateline(wrap_lon(lon), lat) if args.projection == "mollweide" else [(lon, lat)]
        for x, y in poly_chunks:
            fill_lonlat(
                ax,
                x,
                y,
                projection=args.projection,
                facecolor="#eeeeee",
                edgecolor="#8a8a8a",
                linewidth=0.35,
                zorder=1,
            )

    for _, row in selected.iterrows():
        lons, lats = route_cache[row["flight_id"]]
        for x, y in split_dateline(lons, lats):
            plot_lonlat(ax, x, y, color="#168a35", lw=0.9, alpha=0.55, zorder=3, projection=args.projection)

    # Plot anchors over the selected flights.
    sel_ids = set(selected["flight_id"])
    anchor_df = adsc[adsc["flight_id"].astype(str).isin(sel_ids)].copy()
    scatter_lonlat(
        ax,
        anchor_df["longitude"].to_numpy(),
        anchor_df["latitude"].to_numpy(),
        projection=args.projection,
        s=7,
        color="#0b5d24",
        alpha=0.85,
        zorder=4,
    )

    if args.projection == "mollweide":
        ax.grid(True, color="#d8d8d8", linewidth=0.45, alpha=0.7)
        ax.set_xticklabels(["150°W", "120°W", "90°W", "60°W", "30°W", "0°", "30°E", "60°E", "90°E", "120°E", "150°E"])
    else:
        ax.set_xlim(-180, 180)
        ax.set_ylim(-75, 80)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, color="#d8d8d8", linewidth=0.45, alpha=0.7)
    ax.set_title(
        "Cross-Ocean ADS-C Anchor Routes "
        f"(green, n={len(selected)}, min_dist={args.min_distance_km:.0f} km, "
        f"speed<={args.max_avg_speed_kmh:.0f} km/h)"
    )
    fig.tight_layout()
    out_png = out_dir / "cross_ocean_adsc_anchor_routes_green.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    report = [
        f"input_csv={args.adsc_csv}",
        f"all_flights={len(summary)}",
        f"selected_cross_ocean={len(selected)}",
        f"min_anchors={args.min_anchors}",
        f"min_distance_km={args.min_distance_km}",
        f"min_duration_min={args.min_duration_min}",
        f"max_avg_speed_kmh={args.max_avg_speed_kmh}",
        f"min_ocean_ratio={args.min_ocean_ratio}",
        f"plot={out_png}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
