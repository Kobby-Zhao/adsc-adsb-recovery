from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.real_adsc_replay_eval import _predict_on_frame
from scripts.real_adsc_truequal_gapwise_eval import _build_gapwise_segments, _stitch_flight_prediction
from scripts.real_adsc_truequal_gapwise_multi_model_compare import MODEL_STYLES, _resolve_specs
from scripts.train import load_config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Recover cross-ocean missing minute trajectory with baseline + ourmethod.")
    p.add_argument("--pair-id", required=True)
    p.add_argument(
        "--overlay-csv",
        default="outputs/runs/adsc_cross_ocean_top10_highest_anchor_20260429/per_pair_overlay_csv/86dce6_2024-05-03_overlay.csv",
    )
    p.add_argument(
        "--adsb-minute-csv",
        default="outputs/runs/adsc_cross_ocean_top10_highest_anchor_20260429/top10_cross_ocean_highest_anchor_adsb_minute_full_flights.csv",
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--models",
        default="all",
        help="Comma-separated model names, or 'all'. Defaults to all available models.",
    )
    p.add_argument(
        "--cruise-alt-threshold-m",
        type=float,
        default=10000.0,
        help="Altitude threshold for cruise-only plots.",
    )
    return p


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371000.0 * 2 * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def _build_full_frame(pair_id: str, overlay_csv: Path, adsb_minute_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overlay = pd.read_csv(overlay_csv, parse_dates=["plot_ts", "adsc_start_ts", "adsc_end_ts", "adsb_flight_start_ts", "adsb_flight_end_ts"])
    overlay = overlay[overlay["pair_id"].astype(str) == pair_id].copy()
    if overlay.empty:
        raise RuntimeError(f"No rows found for pair_id={pair_id} in {overlay_csv}")

    adsb_full = pd.read_csv(adsb_minute_csv, parse_dates=["minute_ts"])
    flight_id = str(overlay["adsb_flight_id"].iloc[0])
    adsb = adsb_full[adsb_full["flight_id"].astype(str) == flight_id].copy().sort_values("minute_ts").reset_index(drop=True)
    if adsb.empty:
        raise RuntimeError(f"No ADS-B minute rows found for flight_id={flight_id}")

    adsc = overlay[overlay["source"].astype(str) == "adsc_anchor"].copy().sort_values("plot_ts").reset_index(drop=True)
    if adsc.empty:
        raise RuntimeError(f"No ADS-C anchors found for pair_id={pair_id}")

    start_ts = pd.to_datetime(adsb["minute_ts"].min(), utc=True)
    end_ts = pd.to_datetime(adsb["minute_ts"].max(), utc=True)
    full_ts = pd.date_range(start=start_ts, end=end_ts, freq="1min", tz="UTC")
    full = pd.DataFrame({"minute_ts": full_ts})
    full["flight_id"] = pair_id
    full["sample_id"] = f"{pair_id}_cross_ocean_recover"

    known_adsb = adsb.rename(columns={"adsb_icao": "icao24"}).copy()
    known_adsb["minute_ts"] = pd.to_datetime(known_adsb["minute_ts"], utc=True)
    known_adsb = known_adsb[["minute_ts", "lat", "lon", "alt", "speed", "heading", "num_points_in_minute"]].copy()
    known_adsb["known_adsb"] = 1
    full = full.merge(known_adsb, on="minute_ts", how="left")
    full["known_adsb"] = full["known_adsb"].fillna(0).astype(int)

    # Map ADS-C anchors to nearest minute on the full time axis.
    adsc_obs = adsc[["plot_ts", "lat", "lon", "alt"]].copy()
    adsc_obs["anchor_minute_ts"] = pd.to_datetime(adsc_obs["plot_ts"], utc=True).dt.round("1min")
    adsc_obs = adsc_obs.rename(columns={"lat": "anchor_lat", "lon": "anchor_lon", "alt": "anchor_alt"})
    adsc_obs = adsc_obs.groupby("anchor_minute_ts", as_index=False).last()
    full = full.merge(adsc_obs, left_on="minute_ts", right_on="anchor_minute_ts", how="left")
    full["is_adsc_anchor"] = full["anchor_minute_ts"].notna().astype(int)

    full["obs_mask"] = ((full["known_adsb"] == 1) | (full["is_adsc_anchor"] == 1)).astype(int)
    full["obs_source"] = np.where(full["known_adsb"] == 1, "adsb_minute", np.where(full["is_adsc_anchor"] == 1, "adsc_anchor", "missing"))

    # Build initialization track:
    # - keep real ADS-B where available
    # - inject ADS-C anchor coordinates on anchor minutes
    # - interpolate the rest over a complete minute grid
    full["lat_init"] = full["lat"]
    full["lon_init"] = full["lon"]
    full["alt_init"] = full["alt"]
    anchor_mask = full["is_adsc_anchor"] == 1
    full.loc[anchor_mask, "lat_init"] = pd.to_numeric(full.loc[anchor_mask, "anchor_lat"], errors="coerce")
    full.loc[anchor_mask, "lon_init"] = pd.to_numeric(full.loc[anchor_mask, "anchor_lon"], errors="coerce")
    full.loc[anchor_mask, "alt_init"] = pd.to_numeric(full.loc[anchor_mask, "anchor_alt"], errors="coerce")

    for col in ["lat_init", "lon_init", "alt_init"]:
        full[col] = pd.to_numeric(full[col], errors="coerce").interpolate(limit_direction="both")

    # Recompute speed / heading on the completed minute grid.
    lat = full["lat_init"].to_numpy(dtype=float)
    lon = full["lon_init"].to_numpy(dtype=float)
    alt = full["alt_init"].to_numpy(dtype=float)
    speed = np.zeros(len(full), dtype=float)
    heading = np.zeros(len(full), dtype=float)
    for i in range(1, len(full)):
        speed[i] = _haversine_m(lat[i - 1], lon[i - 1], lat[i], lon[i]) / 60.0
        heading[i] = _bearing_deg(lat[i - 1], lon[i - 1], lat[i], lon[i])
    if len(full) >= 2:
        heading[0] = heading[1]
        speed[0] = speed[1]

    frame = full[["sample_id", "flight_id", "minute_ts", "obs_mask", "known_adsb", "is_adsc_anchor", "obs_source"]].copy()
    frame["lat"] = lat
    frame["lon"] = lon
    frame["alt"] = alt
    frame["speed"] = speed
    frame["heading"] = heading
    # Match existing truequal sample convention: obs_* carries the full initialized track, while obs_mask says what is truly observed.
    frame["obs_lat"] = frame["lat"]
    frame["obs_lon"] = frame["lon"]
    frame["obs_alt"] = frame["alt"]

    return frame, adsb, adsc


def _run_models(frame: pd.DataFrame, models_arg: str) -> dict[str, pd.DataFrame]:
    frames, _ = _build_gapwise_segments(frame)
    if not frames:
        raise RuntimeError("No gapwise segments constructed from the input frame.")
    frame_all = pd.concat(frames, ignore_index=True)

    wanted = None
    if str(models_arg).strip().lower() != "all":
        wanted = {x.strip() for x in str(models_arg).split(",") if x.strip()}

    stitched: dict[str, pd.DataFrame] = {}
    for spec in _resolve_specs():
        if wanted is not None and spec.name not in wanted:
            continue
        cfg = load_config(spec.config)
        pred = _predict_on_frame(cfg=cfg, checkpoint=spec.checkpoint, frame=frame_all, pred_key="pred_pos")
        stitched[spec.name] = _stitch_flight_prediction(frame, pred)
    return stitched


def _plot_altitude(frame: pd.DataFrame, stitched: dict[str, pd.DataFrame], out_png: Path) -> None:
    x = frame.sort_values("minute_ts").reset_index(drop=True).copy()
    start = pd.to_datetime(x["minute_ts"], utc=True).min()
    x_min = (pd.to_datetime(x["minute_ts"], utc=True) - start).dt.total_seconds().div(60.0)
    known_adsb = x["known_adsb"].astype(int) == 1
    anchors = x["is_adsc_anchor"].astype(int) == 1

    fig, ax = plt.subplots(figsize=(12.2, 5.6), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(x_min[known_adsb], pd.to_numeric(x.loc[known_adsb, "obs_alt"], errors="coerce"), color="#111111", lw=2.0, alpha=0.9, label="Known ADS-B minute")
    ax.scatter(x_min[anchors], pd.to_numeric(x.loc[anchors, "obs_alt"], errors="coerce"), color="#d95f02", s=34, zorder=6, label="ADS-C anchors")
    for model_name, s in stitched.items():
        g = s.sort_values("minute_ts").reset_index(drop=True)
        gx = (pd.to_datetime(g["minute_ts"], utc=True) - start).dt.total_seconds().div(60.0)
        style = MODEL_STYLES.get(model_name, {}).copy()
        ax.plot(gx, pd.to_numeric(g["pred_alt"], errors="coerce"), label=model_name, **style)
    ax.set_title(f"{x['flight_id'].iloc[0]} | Cross-ocean 3-stage minute recovery")
    ax.set_xlabel("Minutes from flight start")
    ax.set_ylabel("Altitude (m)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_altitude_cruise_only(frame: pd.DataFrame, stitched: dict[str, pd.DataFrame], out_png: Path, threshold_m: float) -> None:
    x = frame.sort_values("minute_ts").reset_index(drop=True).copy()
    cruise_mask = pd.to_numeric(x["alt"], errors="coerce") >= float(threshold_m)
    if cruise_mask.sum() < 5:
        cruise_mask = pd.to_numeric(x["alt"], errors="coerce") >= float(max(pd.to_numeric(x["alt"], errors="coerce").quantile(0.7), threshold_m * 0.9))
    x = x.loc[cruise_mask].copy().reset_index(drop=True)
    if x.empty:
        return

    start = pd.to_datetime(x["minute_ts"], utc=True).min()
    x_min = (pd.to_datetime(x["minute_ts"], utc=True) - start).dt.total_seconds().div(60.0)
    known_adsb = x["known_adsb"].astype(int) == 1
    anchors = x["is_adsc_anchor"].astype(int) == 1

    fig, ax = plt.subplots(figsize=(12.2, 5.4), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(x_min[known_adsb], pd.to_numeric(x.loc[known_adsb, "obs_alt"], errors="coerce"), color="#111111", lw=2.0, alpha=0.9, label="Known ADS-B minute")
    ax.scatter(x_min[anchors], pd.to_numeric(x.loc[anchors, "obs_alt"], errors="coerce"), color="#d95f02", s=34, zorder=6, label="ADS-C anchors")
    for model_name, s in stitched.items():
        g = s.sort_values("minute_ts").copy()
        g = g[g["minute_ts"].isin(set(x["minute_ts"]))].reset_index(drop=True)
        gx = (pd.to_datetime(g["minute_ts"], utc=True) - start).dt.total_seconds().div(60.0)
        style = MODEL_STYLES.get(model_name, {}).copy()
        ax.plot(gx, pd.to_numeric(g["pred_alt"], errors="coerce"), label=model_name, **style)
    ax.set_title(f"{x['flight_id'].iloc[0]} | Cruise-only altitude compare")
    ax.set_xlabel("Minutes from cruise segment start")
    ax.set_ylabel("Altitude (m)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_3d(frame: pd.DataFrame, stitched: dict[str, pd.DataFrame], out_png: Path) -> None:
    x = frame.sort_values("minute_ts").reset_index(drop=True).copy()
    known_adsb = x["known_adsb"].astype(int) == 1
    anchors = x["is_adsc_anchor"].astype(int) == 1

    lon = pd.to_numeric(x["lon"], errors="coerce")
    lat = pd.to_numeric(x["lat"], errors="coerce")
    alt = pd.to_numeric(x["alt"], errors="coerce")
    lon_span = float(max(lon.max() - lon.min(), 1e-6))
    lat_span = float(max(lat.max() - lat.min(), 1e-6))
    alt_span = float(max(alt.max() - alt.min(), 1e-6))
    mean_lat = float(lat.mean()) if len(lat) else 0.0
    lon_m = lon_span * 111000.0 * max(0.2, abs(math.cos(math.radians(mean_lat))))
    lat_m = lat_span * 111000.0
    z_display = max(alt_span * 0.22, min(lon_m, lat_m) * 0.45, 1.0)

    fig = plt.figure(figsize=(11.2, 7.4), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    ax.plot(
        pd.to_numeric(x.loc[known_adsb, "obs_lon"], errors="coerce"),
        pd.to_numeric(x.loc[known_adsb, "obs_lat"], errors="coerce"),
        pd.to_numeric(x.loc[known_adsb, "obs_alt"], errors="coerce"),
        color="#111111",
        lw=2.0,
        alpha=0.9,
        label="Known ADS-B minute",
    )
    ax.scatter(
        pd.to_numeric(x.loc[anchors, "obs_lon"], errors="coerce"),
        pd.to_numeric(x.loc[anchors, "obs_lat"], errors="coerce"),
        pd.to_numeric(x.loc[anchors, "obs_alt"], errors="coerce"),
        color="#d95f02",
        s=28,
        alpha=0.95,
        label="ADS-C anchors",
    )

    for model_name, s in stitched.items():
        g = s.sort_values("minute_ts").reset_index(drop=True)
        style = MODEL_STYLES.get(model_name, {}).copy()
        ax.plot(
            pd.to_numeric(g["pred_lon"], errors="coerce"),
            pd.to_numeric(g["pred_lat"], errors="coerce"),
            pd.to_numeric(g["pred_alt"], errors="coerce"),
            label=model_name,
            **style,
        )

    ax.set_title(f"{x['flight_id'].iloc[0]} | Cross-ocean 3-stage 3D recovery")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Altitude (m)")
    # Restore the simpler overlay-like camera instead of the more aggressive compare view.
    ax.view_init(elev=18, azim=-58)
    ax.set_proj_type("persp")
    ax.set_box_aspect((max(lon_m, 1.0), max(lat_m, 1.0), max(alt_span * 0.28, z_display * 0.75, 1.0)))
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_3d_cruise_only(frame: pd.DataFrame, stitched: dict[str, pd.DataFrame], out_png: Path, threshold_m: float) -> None:
    x = frame.sort_values("minute_ts").reset_index(drop=True).copy()
    cruise_mask = pd.to_numeric(x["alt"], errors="coerce") >= float(threshold_m)
    if cruise_mask.sum() < 5:
        cruise_mask = pd.to_numeric(x["alt"], errors="coerce") >= float(max(pd.to_numeric(x["alt"], errors="coerce").quantile(0.7), threshold_m * 0.9))
    x = x.loc[cruise_mask].copy().reset_index(drop=True)
    if x.empty:
        return

    known_adsb = x["known_adsb"].astype(int) == 1
    anchors = x["is_adsc_anchor"].astype(int) == 1

    lon = pd.to_numeric(x["lon"], errors="coerce")
    lat = pd.to_numeric(x["lat"], errors="coerce")
    alt = pd.to_numeric(x["alt"], errors="coerce")
    lon_span = float(max(lon.max() - lon.min(), 1e-6))
    lat_span = float(max(lat.max() - lat.min(), 1e-6))
    alt_span = float(max(alt.max() - alt.min(), 1e-6))
    mean_lat = float(lat.mean()) if len(lat) else 0.0
    lon_m = lon_span * 111000.0 * max(0.2, abs(math.cos(math.radians(mean_lat))))
    lat_m = lat_span * 111000.0

    fig = plt.figure(figsize=(11.2, 7.4), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    ax.plot(
        pd.to_numeric(x.loc[known_adsb, "obs_lon"], errors="coerce"),
        pd.to_numeric(x.loc[known_adsb, "obs_lat"], errors="coerce"),
        pd.to_numeric(x.loc[known_adsb, "obs_alt"], errors="coerce"),
        color="#111111",
        lw=2.0,
        alpha=0.9,
        label="Known ADS-B minute",
    )
    ax.scatter(
        pd.to_numeric(x.loc[anchors, "obs_lon"], errors="coerce"),
        pd.to_numeric(x.loc[anchors, "obs_lat"], errors="coerce"),
        pd.to_numeric(x.loc[anchors, "obs_alt"], errors="coerce"),
        color="#d95f02",
        s=28,
        alpha=0.95,
        label="ADS-C anchors",
    )

    keep_ts = set(x["minute_ts"])
    for model_name, s in stitched.items():
        g = s.sort_values("minute_ts").copy()
        g = g[g["minute_ts"].isin(keep_ts)].reset_index(drop=True)
        style = MODEL_STYLES.get(model_name, {}).copy()
        ax.plot(
            pd.to_numeric(g["pred_lon"], errors="coerce"),
            pd.to_numeric(g["pred_lat"], errors="coerce"),
            pd.to_numeric(g["pred_alt"], errors="coerce"),
            label=model_name,
            **style,
        )

    ax.set_title(f"{x['flight_id'].iloc[0]} | Cruise-only 3D recovery")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Altitude (m)")
    ax.view_init(elev=18, azim=-58)
    ax.set_proj_type("persp")
    ax.set_box_aspect((max(lon_m, 1.0), max(lat_m, 1.0), max(alt_span * 0.28, 1.0)))
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame, adsb, adsc = _build_full_frame(
        pair_id=str(args.pair_id),
        overlay_csv=Path(args.overlay_csv),
        adsb_minute_csv=Path(args.adsb_minute_csv),
    )
    frame.to_csv(out_dir / "input_recovery_frame.csv", index=False)
    adsb.to_csv(out_dir / "known_adsb_minute.csv", index=False)
    adsc.to_csv(out_dir / "adsc_anchor_points.csv", index=False)

    stitched = _run_models(frame, args.models)
    merged = frame[["sample_id", "flight_id", "minute_ts", "obs_mask", "known_adsb", "is_adsc_anchor", "obs_source", "lat", "lon", "alt"]].copy()
    for model_name, s in stitched.items():
        cols = s[["minute_ts", "pred_lat", "pred_lon", "pred_alt"]].copy()
        cols = cols.rename(
            columns={
                "pred_lat": f"{model_name}_pred_lat",
                "pred_lon": f"{model_name}_pred_lon",
                "pred_alt": f"{model_name}_pred_alt",
            }
        )
        merged = merged.merge(cols, on="minute_ts", how="left")
    merged.to_csv(out_dir / "recovered_minute_compare.csv", index=False)

    _plot_altitude(frame, stitched, out_dir / "altitude_2d_compare.png")
    _plot_3d(frame, stitched, out_dir / "trajectory_3d_compare.png")
    _plot_altitude_cruise_only(frame, stitched, out_dir / "altitude_2d_compare_cruise_only.png", args.cruise_alt_threshold_m)
    _plot_3d_cruise_only(frame, stitched, out_dir / "trajectory_3d_compare_cruise_only.png", args.cruise_alt_threshold_m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
