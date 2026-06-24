from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_current_cross_ocean_altitude_compare import SPECS
from scripts.real_adsc_replay_eval import _predict_on_frame
from scripts.real_adsc_truequal_gapwise_eval import _build_gapwise_segments, _stitch_flight_prediction
from scripts.train import load_config


MODEL_STYLES = {
    "Kalman Filter": {"color": "#8c8c8c", "linestyle": (0, (1, 2)), "linewidth": 1.25},
    "LSTM-clean": {"color": "#9467bd", "linestyle": "--", "linewidth": 1.35},
    "BiLSTM-clean": {"color": "#6b6b6b", "linestyle": "--", "linewidth": 1.7},
    "CNN+LSTM-clean": {"color": "#8c564b", "linestyle": "--", "linewidth": 1.35},
    "Transformer-clean": {"color": "#ff7f0e", "linestyle": "--", "linewidth": 1.45},
    "Backbone-only": {"color": "#005f73", "linestyle": "-.", "linewidth": 1.8},
    "A1-linear-alt": {"color": "#2ca02c", "linestyle": "-", "linewidth": 1.85},
    "Ours-A3": {"color": "#d00000", "linestyle": "-", "linewidth": 2.25},
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Plot model recovery on clean cross-ocean cases with ADS-B quality annotations.")
    p.add_argument("--case-csv", default="outputs/runs/clean_cross_ocean_adsc_adsb_cases_20260517/selected_clean_cross_ocean_cases.csv")
    p.add_argument("--out-dir", default="outputs/runs/clean_cross_ocean_model_compare_20260517")
    p.add_argument("--models", default="Kalman Filter,LSTM-clean,BiLSTM-clean,CNN+LSTM-clean,Transformer-clean,Backbone-only,A1-linear-alt,Ours-A3")
    p.add_argument("--max-cases", type=int, default=10)
    p.add_argument("--freeze-min-len", type=int, default=2)
    p.add_argument("--gap-threshold-min", type=float, default=1.5)
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


def _same_latlon_runs(adsb: pd.DataFrame, min_len: int) -> list[tuple[int, int]]:
    if adsb.empty:
        return []
    same = adsb[["lat", "lon"]].round(6).eq(adsb[["lat", "lon"]].round(6).shift()).all(axis=1).to_numpy()
    runs: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(adsb)):
        if not same[i]:
            if i - start >= min_len:
                runs.append((start, i - 1))
            start = i
    if len(adsb) - start >= min_len:
        runs.append((start, len(adsb) - 1))
    return runs


def _gap_edges(adsb: pd.DataFrame, threshold_min: float) -> list[tuple[int, int, float]]:
    if len(adsb) < 2:
        return []
    t = pd.to_datetime(adsb["minute_ts"], utc=True, errors="coerce")
    dt = t.diff().dt.total_seconds().div(60.0)
    gaps: list[tuple[int, int, float]] = []
    for i, minutes in enumerate(dt):
        if i == 0 or not np.isfinite(minutes):
            continue
        if float(minutes) > float(threshold_min):
            gaps.append((i - 1, i, float(minutes)))
    return gaps


def _build_frame(pair_id: str, adsb_csv: Path, adsc_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    adsb = pd.read_csv(adsb_csv, parse_dates=["minute_ts"]).sort_values("minute_ts").reset_index(drop=True)
    adsc = pd.read_csv(adsc_csv, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if adsb.empty:
        raise RuntimeError(f"Empty ADS-B minute CSV: {adsb_csv}")
    if adsc.empty:
        raise RuntimeError(f"Empty ADS-C anchor CSV: {adsc_csv}")
    adsb["minute_ts"] = pd.to_datetime(adsb["minute_ts"], utc=True, errors="coerce")
    adsc["timestamp"] = pd.to_datetime(adsc["timestamp"], utc=True, errors="coerce")

    start_ts = min(adsb["minute_ts"].min(), adsc["timestamp"].min().floor("min"))
    end_ts = max(adsb["minute_ts"].max(), adsc["timestamp"].max().ceil("min"))
    full = pd.DataFrame({"minute_ts": pd.date_range(start=start_ts, end=end_ts, freq="1min", tz="UTC")})
    full["flight_id"] = pair_id
    full["sample_id"] = f"{pair_id}_clean_cross_ocean_recover"

    known = adsb[["minute_ts", "lat", "lon", "alt", "speed", "heading", "num_points_in_minute"]].copy()
    known["known_adsb"] = 1
    full = full.merge(known, on="minute_ts", how="left")
    full["known_adsb"] = full["known_adsb"].fillna(0).astype(int)

    anchors = adsc[["timestamp", "latitude", "longitude", "altitude_m"]].copy()
    anchors["anchor_minute_ts"] = anchors["timestamp"].dt.round("1min")
    anchors = anchors.rename(columns={"latitude": "anchor_lat", "longitude": "anchor_lon", "altitude_m": "anchor_alt"})
    anchors = anchors.groupby("anchor_minute_ts", as_index=False).last()
    full = full.merge(anchors, left_on="minute_ts", right_on="anchor_minute_ts", how="left")
    full["is_adsc_anchor"] = full["anchor_minute_ts"].notna().astype(int)

    full["obs_mask"] = ((full["known_adsb"] == 1) | (full["is_adsc_anchor"] == 1)).astype(int)
    full["obs_source"] = np.where(full["known_adsb"] == 1, "adsb_minute", np.where(full["is_adsc_anchor"] == 1, "adsc_anchor", "missing"))

    full["lat_init"] = full["lat"]
    full["lon_init"] = full["lon"]
    full["alt_init"] = full["alt"]
    anchor_mask = full["is_adsc_anchor"] == 1
    full.loc[anchor_mask, "lat_init"] = pd.to_numeric(full.loc[anchor_mask, "anchor_lat"], errors="coerce")
    full.loc[anchor_mask, "lon_init"] = pd.to_numeric(full.loc[anchor_mask, "anchor_lon"], errors="coerce")
    full.loc[anchor_mask, "alt_init"] = pd.to_numeric(full.loc[anchor_mask, "anchor_alt"], errors="coerce")
    for col in ["lat_init", "lon_init", "alt_init"]:
        full[col] = pd.to_numeric(full[col], errors="coerce").interpolate(limit_direction="both")

    lat = full["lat_init"].to_numpy(dtype=float)
    lon = full["lon_init"].to_numpy(dtype=float)
    alt = full["alt_init"].to_numpy(dtype=float)
    speed = np.zeros(len(full), dtype=float)
    heading = np.zeros(len(full), dtype=float)
    for i in range(1, len(full)):
        speed[i] = _haversine_m(lat[i - 1], lon[i - 1], lat[i], lon[i]) / 60.0
        heading[i] = _bearing_deg(lat[i - 1], lon[i - 1], lat[i], lon[i])
    if len(full) >= 2:
        speed[0] = speed[1]
        heading[0] = heading[1]

    frame = full[["sample_id", "flight_id", "minute_ts", "obs_mask", "known_adsb", "is_adsc_anchor", "obs_source"]].copy()
    frame["lat"] = lat
    frame["lon"] = lon
    frame["alt"] = alt
    frame["speed"] = speed
    frame["heading"] = heading
    frame["obs_lat"] = frame["lat"]
    frame["obs_lon"] = frame["lon"]
    frame["obs_alt"] = frame["alt"]
    return frame, adsb, adsc


def _run_models(frame: pd.DataFrame, model_names: list[str]) -> dict[str, pd.DataFrame]:
    frames, _ = _build_gapwise_segments(frame)
    if not frames:
        raise RuntimeError("No gapwise segments constructed.")
    frame_all = pd.concat(frames, ignore_index=True)
    stitched: dict[str, pd.DataFrame] = {}
    for name in model_names:
        cfg_rel, ckpt_rel = SPECS[name]
        cfg = load_config(str(ROOT / cfg_rel))
        pred = _predict_on_frame(cfg=cfg, checkpoint=ROOT / ckpt_rel, frame=frame_all, pred_key="pred_pos")
        stitched[name] = _stitch_flight_prediction(frame, pred)
    return stitched


def _plot_case(
    pair_id: str,
    frame: pd.DataFrame,
    adsb: pd.DataFrame,
    adsc: pd.DataFrame,
    stitched: dict[str, pd.DataFrame],
    out_png: Path,
    freeze_min_len: int,
    gap_threshold_min: float,
) -> dict[str, object]:
    adsb = adsb.sort_values("minute_ts").reset_index(drop=True).copy()
    adsc = adsc.sort_values("timestamp").reset_index(drop=True).copy()
    freeze_runs = _same_latlon_runs(adsb, freeze_min_len)
    gaps = _gap_edges(adsb, gap_threshold_min)
    t0 = pd.to_datetime(frame["minute_ts"], utc=True).min()

    x_adsb = (pd.to_datetime(adsb["minute_ts"], utc=True) - t0).dt.total_seconds().div(60.0).to_numpy(dtype=float)
    x_adsc = (pd.to_datetime(adsc["timestamp"], utc=True) - t0).dt.total_seconds().div(60.0).to_numpy(dtype=float)
    adsc_lon = pd.to_numeric(adsc["longitude"], errors="coerce")
    adsc_lat = pd.to_numeric(adsc["latitude"], errors="coerce")
    adsc_alt = pd.to_numeric(adsc["altitude_m"], errors="coerce")

    fig, (ax_map, ax_alt) = plt.subplots(2, 1, figsize=(13.8, 8.4), facecolor="white")
    ax_map.set_facecolor("white")
    ax_alt.set_facecolor("white")

    gap_after_idx = {a for a, _, _ in gaps}
    start = 0
    normal_labeled = False
    lon = pd.to_numeric(adsb["lon"], errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(adsb["lat"], errors="coerce").to_numpy(dtype=float)
    alt = pd.to_numeric(adsb["alt"], errors="coerce").to_numpy(dtype=float)
    for i in range(len(adsb) - 1):
        if i in gap_after_idx:
            ax_map.plot(lon[start : i + 1], lat[start : i + 1], color="#1f77b4", lw=1.35, label="ADS-B normal" if not normal_labeled else None)
            ax_alt.plot(x_adsb[start : i + 1], alt[start : i + 1], color="#1f77b4", lw=1.35, label="ADS-B normal" if not normal_labeled else None)
            normal_labeled = True
            start = i + 1
    ax_map.plot(lon[start:], lat[start:], color="#1f77b4", lw=1.35, label="ADS-B normal" if not normal_labeled else None)
    ax_alt.plot(x_adsb[start:], alt[start:], color="#1f77b4", lw=1.35, label="ADS-B normal" if not normal_labeled else None)

    freeze_labeled = False
    for a, b in freeze_runs:
        ax_map.plot(lon[a : b + 1], lat[a : b + 1], color="#d62728", lw=3.0, label="ADS-B frozen lat/lon" if not freeze_labeled else None, zorder=6)
        ax_alt.plot(x_adsb[a : b + 1], alt[a : b + 1], color="#d62728", lw=3.0, label="ADS-B frozen lat/lon" if not freeze_labeled else None, zorder=6)
        freeze_labeled = True

    gap_labeled = False
    for left, right, minutes in gaps:
        ax_map.plot([lon[left], lon[right]], [lat[left], lat[right]], color="#ff9f1c", lw=2.2, linestyle=(0, (4, 3)), label="ADS-B missing interval" if not gap_labeled else None)
        ax_alt.axvspan(x_adsb[left], x_adsb[right], color="#ff9f1c", alpha=0.18, linewidth=0)
        ax_alt.plot([x_adsb[left], x_adsb[right]], [alt[left], alt[right]], color="#ff9f1c", lw=1.8, linestyle=(0, (4, 3)), label="ADS-B missing interval" if not gap_labeled else None)
        if minutes >= 30:
            y_top = float(np.nanmax(alt)) if np.isfinite(np.nanmax(alt)) else 0.0
            ax_alt.text((x_adsb[left] + x_adsb[right]) / 2.0, y_top, f"{minutes:.0f} min gap", fontsize=7, color="#9a5a00", ha="center", va="bottom")
        gap_labeled = True

    ax_map.scatter(adsc_lon, adsc_lat, color="#2ca02c", edgecolor="#111111", linewidth=0.5, s=44, label="ADS-C anchors", zorder=10)
    ax_alt.scatter(x_adsc, adsc_alt, color="#2ca02c", edgecolor="#111111", linewidth=0.5, s=44, label="ADS-C anchors", zorder=10)

    for name, pred in stitched.items():
        p = pred.sort_values("minute_ts").reset_index(drop=True)
        px = (pd.to_datetime(p["minute_ts"], utc=True) - t0).dt.total_seconds().div(60.0)
        style = MODEL_STYLES.get(name, {"linewidth": 1.8}).copy()
        ax_map.plot(pd.to_numeric(p["pred_lon"], errors="coerce"), pd.to_numeric(p["pred_lat"], errors="coerce"), label=name, alpha=0.92, **style)
        ax_alt.plot(px, pd.to_numeric(p["pred_alt"], errors="coerce"), label=name, alpha=0.95, **style)

    ax_map.set_title(f"{pair_id} | Cross-ocean recovery with ADS-B quality annotations")
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.grid(alpha=0.25)
    ax_map.legend(fontsize=8, ncol=3, loc="best")

    ax_alt.set_xlabel("Minutes from recovery frame start")
    ax_alt.set_ylabel("Altitude (m)")
    ax_alt.grid(alpha=0.25)
    ax_alt.legend(fontsize=8, ncol=3, loc="best")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "plot": str(out_png.relative_to(ROOT)),
        "freeze_run_count": len(freeze_runs),
        "freeze_minutes": int(sum(b - a + 1 for a, b in freeze_runs)),
        "missing_gap_count": len(gaps),
        "max_missing_gap_min": float(max([x[2] for x in gaps], default=0.0)),
    }


def main() -> int:
    args = build_parser().parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(ROOT / args.case_csv).head(int(args.max_cases))
    model_names = [x.strip() for x in str(args.models).split(",") if x.strip()]
    unknown = [m for m in model_names if m not in SPECS]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}; available={sorted(SPECS)}")

    rows = []
    for _, row in cases.iterrows():
        pair_id = str(row["pair_id"])
        case_dir = out_dir / pair_id
        case_dir.mkdir(parents=True, exist_ok=True)
        print(f"[case] {pair_id}", flush=True)
        frame, adsb, adsc = _build_frame(pair_id, ROOT / str(row["adsb_minute_csv"]), ROOT / str(row["adsc_anchor_csv"]))
        frame.to_csv(case_dir / "input_recovery_frame.csv", index=False)
        stitched = _run_models(frame, model_names)
        merged = frame[["sample_id", "flight_id", "minute_ts", "obs_mask", "known_adsb", "is_adsc_anchor", "obs_source", "lat", "lon", "alt"]].copy()
        for name, pred in stitched.items():
            cols = pred[["minute_ts", "pred_lat", "pred_lon", "pred_alt"]].rename(
                columns={"pred_lat": f"{name}_pred_lat", "pred_lon": f"{name}_pred_lon", "pred_alt": f"{name}_pred_alt"}
            )
            merged = merged.merge(cols, on="minute_ts", how="left")
        recovered_csv = case_dir / "recovered_minute_compare.csv"
        merged.to_csv(recovered_csv, index=False)
        plot_png = case_dir / f"{pair_id}_model_compare_annotated.png"
        metrics = _plot_case(pair_id, frame, adsb, adsc, stitched, plot_png, int(args.freeze_min_len), float(args.gap_threshold_min))
        rows.append(
            {
                "pair_id": pair_id,
                "recovered_csv": str(recovered_csv.relative_to(ROOT)),
                "input_frame_csv": str((case_dir / "input_recovery_frame.csv").relative_to(ROOT)),
                "known_adsb_minutes": int(frame["known_adsb"].sum()),
                "adsc_anchor_minutes": int(frame["is_adsc_anchor"].sum()),
                "missing_minutes": int((frame["obs_mask"].astype(int) == 0).sum()),
                **metrics,
            }
        )
        print(f"[ok] {pair_id} -> {plot_png}", flush=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "clean_cross_ocean_model_compare_summary.csv", index=False)
    print(f"[done] summary={out_dir / 'clean_cross_ocean_model_compare_summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
