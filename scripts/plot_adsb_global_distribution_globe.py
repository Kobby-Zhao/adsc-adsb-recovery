from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_shp_polygons(shp_path: Path):
    polys = []
    with shp_path.open('rb') as f:
        header = f.read(100)
        if len(header) < 100:
            return polys
        while True:
            rec_header = f.read(8)
            if not rec_header or len(rec_header) < 8:
                break
            rec_num, rec_len_words = struct.unpack('>2i', rec_header)
            rec_len = rec_len_words * 2
            rec = f.read(rec_len)
            if len(rec) < 4:
                continue
            shape_type = struct.unpack('<i', rec[:4])[0]
            if shape_type not in (5, 3):  # polygon/polyline
                continue
            if len(rec) < 44:
                continue
            num_parts, num_points = struct.unpack('<2i', rec[36:44])
            parts = struct.unpack('<' + 'i' * num_parts, rec[44:44 + 4 * num_parts]) if num_parts > 0 else []
            pts_off = 44 + 4 * num_parts
            pts = []
            for i in range(num_points):
                off = pts_off + 16 * i
                if off + 16 > len(rec):
                    break
                x, y = struct.unpack('<2d', rec[off:off + 16])
                pts.append((x, y))
            if not pts:
                continue
            if not parts:
                polys.append(pts)
                continue
            idx = list(parts) + [len(pts)]
            for i in range(len(parts)):
                seg = pts[idx[i]:idx[i + 1]]
                if len(seg) >= 2:
                    polys.append(seg)
    return polys


def ortho_project(lon_deg, lat_deg, lon0_deg=20.0, lat0_deg=10.0, radius=1.0):
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    lon0 = math.radians(lon0_deg)
    lat0 = math.radians(lat0_deg)

    cosc = np.sin(lat0) * np.sin(lat) + np.cos(lat0) * np.cos(lat) * np.cos(lon - lon0)
    visible = cosc >= 0
    x = radius * np.cos(lat) * np.sin(lon - lon0)
    y = radius * (np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(lon - lon0))
    return x, y, visible


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


def great_circle_points(lon1, lat1, lon2, lat2, n=40):
    lon1r, lat1r = map(math.radians, (lon1, lat1))
    lon2r, lat2r = map(math.radians, (lon2, lat2))
    p1 = np.array([math.cos(lat1r) * math.cos(lon1r), math.cos(lat1r) * math.sin(lon1r), math.sin(lat1r)])
    p2 = np.array([math.cos(lat2r) * math.cos(lon2r), math.cos(lat2r) * math.sin(lon2r), math.sin(lat2r)])
    dot = float(np.clip(np.dot(p1, p2), -1.0, 1.0))
    omega = math.acos(dot)
    if abs(omega) < 1e-9:
        lons = np.full(n, lon1)
        lats = np.full(n, lat1)
        return lons, lats
    ts = np.linspace(0.0, 1.0, n)
    pts = []
    so = math.sin(omega)
    for t in ts:
        v = (math.sin((1 - t) * omega) / so) * p1 + (math.sin(t * omega) / so) * p2
        v = v / np.linalg.norm(v)
        lon = math.degrees(math.atan2(v[1], v[0]))
        lat = math.degrees(math.asin(v[2]))
        pts.append((lon, lat))
    lons = np.array([p[0] for p in pts])
    lats = np.array([p[1] for p in pts])
    return lons, lats


def build_flight_endpoints(df: pd.DataFrame):
    x = df[['flight_id', 'minute_ts', 'lon', 'lat']].dropna().copy()
    x['minute_ts'] = pd.to_datetime(x['minute_ts'], utc=True, errors='coerce')
    x = x.dropna(subset=['minute_ts'])
    x = x.sort_values(['flight_id', 'minute_ts'])
    g = x.groupby('flight_id', as_index=False)
    first = g.first()
    last = g.last()
    m = first[['flight_id', 'lon', 'lat']].merge(
        last[['flight_id', 'lon', 'lat']], on='flight_id', suffixes=('_start', '_end')
    )
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adsb-parquet', default='outputs/mvp_merged_nostage_20260415/adsb_minute_merged.parquet')
    ap.add_argument('--shp', default='ne_110m_land.shp')
    ap.add_argument('--out-dir', default='outputs/runs/adsb_global_distribution_20260420')
    ap.add_argument('--max-flights', type=int, default=2000)
    ap.add_argument('--center-lon', type=float, default=20.0)
    ap.add_argument('--center-lat', type=float, default=10.0)
    ap.add_argument('--highlight-km', type=float, default=3000.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.adsb_parquet)
    endpoints = build_flight_endpoints(df)
    if len(endpoints) == 0:
        raise RuntimeError('No flight endpoints found')

    # sample for readability
    if len(endpoints) > args.max_flights:
        endpoints = endpoints.sample(args.max_flights, random_state=42).reset_index(drop=True)

    endpoints['dist_km'] = endpoints.apply(
        lambda r: haversine_km(r['lon_start'], r['lat_start'], r['lon_end'], r['lat_end']), axis=1
    )

    polys = read_shp_polygons(Path(args.shp))

    fig = plt.figure(figsize=(10, 10), facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_facecolor('white')

    # globe boundary
    th = np.linspace(0, 2 * math.pi, 720)
    ax.plot(np.cos(th), np.sin(th), color='#c9c9c9', lw=1.2, zorder=5)

    # land outlines
    for seg in polys:
        lon = np.array([p[0] for p in seg], dtype=float)
        lat = np.array([p[1] for p in seg], dtype=float)
        x, y, vis = ortho_project(lon, lat, lon0_deg=args.center_lon, lat0_deg=args.center_lat)
        # hide back side
        xx = np.where(vis, x, np.nan)
        yy = np.where(vis, y, np.nan)
        ax.plot(xx, yy, color='#d2d2d2', lw=0.6, alpha=0.8, zorder=6)

    # flight arcs
    for _, r in endpoints.iterrows():
        lons, lats = great_circle_points(r['lon_start'], r['lat_start'], r['lon_end'], r['lat_end'], n=50)
        x, y, vis = ortho_project(lons, lats, lon0_deg=args.center_lon, lat0_deg=args.center_lat)
        xx = np.where(vis, x, np.nan)
        yy = np.where(vis, y, np.nan)
        if r['dist_km'] >= args.highlight_km:
            ax.plot(xx, yy, color='#c84b5a', lw=1.15, alpha=0.45, zorder=8)
        else:
            ax.plot(xx, yy, color='#8ec5eb', lw=0.9, alpha=0.20, zorder=7)

    # highlighted endpoints
    hi = endpoints[endpoints['dist_km'] >= args.highlight_km]
    if len(hi):
        xs, ys, vs = ortho_project(hi['lon_start'].to_numpy(), hi['lat_start'].to_numpy(), lon0_deg=args.center_lon, lat0_deg=args.center_lat)
        ax.scatter(xs[vs], ys[vs], s=8, color='#c84b5a', alpha=0.65, zorder=9)
        xe, ye, ve = ortho_project(hi['lon_end'].to_numpy(), hi['lat_end'].to_numpy(), lon0_deg=args.center_lon, lat0_deg=args.center_lat)
        ax.scatter(xe[ve], ye[ve], s=8, color='#c84b5a', alpha=0.65, zorder=9)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axis('off')
    title = f"ADS-B Global Distribution (n={len(endpoints)} flights, center=({args.center_lon:.0f},{args.center_lat:.0f}))"
    ax.set_title(title, fontsize=12, pad=12)

    out_png = out_dir / 'adsb_global_distribution_globe.png'
    fig.savefig(out_png, dpi=250, bbox_inches='tight')
    plt.close(fig)

    out_csv = out_dir / 'adsb_global_distribution_endpoints.csv'
    endpoints.to_csv(out_csv, index=False)
    with (out_dir / 'summary.txt').open('w', encoding='utf-8') as f:
        f.write(f'total_flights_used={len(endpoints)}\n')
        f.write(f'highlight_km={args.highlight_km}\n')
        f.write(f'highlight_count={(endpoints.dist_km>=args.highlight_km).sum()}\n')
        f.write(f'output_png={out_png}\n')

    print(f'[ok] png={out_png}')
    print(f'[ok] endpoints={out_csv}')


if __name__ == '__main__':
    main()
