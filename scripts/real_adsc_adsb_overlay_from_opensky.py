from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.adsc_adsb_raw_alignment_check import minute_agg_adsb
from scripts.real_adsc_replay_eval import _predict_on_frame
from scripts.real_adsc_truequal_gapwise_eval import _build_gapwise_segments, _stitch_flight_prediction
from scripts.real_adsc_truequal_gapwise_multi_model_compare import MODEL_STYLES, _resolve_specs
from scripts.train import load_config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Align OpenSky ADS-B with real ADS-C multi-model recovery outputs.")
    p.add_argument("--samples-parquet", required=True)
    p.add_argument("--matches-csv", required=True)
    p.add_argument("--points-dir", required=True)
    p.add_argument("--points-dir-extra", default=None)
    p.add_argument("--out-dir", required=True)
    return p


def _find_points_file(fid_row: pd.Series, search_dirs: list[Path]) -> Path | None:
    icao = str(fid_row["icao24"]).lower()
    callsign = str(fid_row["adsb_callsign"]).strip().upper()
    for d in search_dirs:
        if d is None or not d.exists():
            continue
        pats = [
            f"*{callsign}*-{icao}.csv",
            f"*{icao}.csv",
        ]
        for pat in pats:
            hits = sorted(d.rglob(pat))
            if hits:
                return hits[0]
    return None


def _load_adsb_minute(points_csv: Path, adsc_start: pd.Timestamp, adsc_end: pd.Timestamp) -> pd.DataFrame:
    raw = pd.read_csv(points_csv)
    raw["time"] = pd.to_datetime(raw["time"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["time"]).copy()
    raw = raw.rename(columns={"time": "timestamp"})
    raw["source"] = "adsb"
    raw = raw.dropna(subset=["lat", "lon", "baroaltitude"]).copy()
    minute = minute_agg_adsb(raw)
    minute = minute[(minute["minute_ts"] >= adsc_start.floor("min")) & (minute["minute_ts"] <= adsc_end.ceil("min"))].copy()
    return minute.sort_values("minute_ts").reset_index(drop=True)


def _prepare_predictions(samples: pd.DataFrame, flight_ids: list[str]) -> dict[str, dict[str, pd.DataFrame]]:
    original_by_flight: dict[str, pd.DataFrame] = {}
    all_frames: list[pd.DataFrame] = []
    for fid in flight_ids:
        g = samples[samples["flight_id"].astype(str) == fid].copy()
        if g.empty:
            continue
        original_by_flight[fid] = g
        frames, _ = _build_gapwise_segments(g)
        all_frames.extend(frames)
    if not all_frames:
        raise RuntimeError("No gapwise segments found in provided samples.")
    frame_all = pd.concat(all_frames, ignore_index=True)
    stitched_by_model_by_flight: dict[str, dict[str, pd.DataFrame]] = {fid: {} for fid in original_by_flight}
    for spec in _resolve_specs():
        cfg = load_config(spec.config)
        pred = _predict_on_frame(cfg=cfg, checkpoint=spec.checkpoint, frame=frame_all, pred_key="pred_pos")
        for fid, base in original_by_flight.items():
            fpred = pred[pred["flight_id"].astype(str) == fid].copy()
            if fpred.empty:
                continue
            stitched_by_model_by_flight[fid][spec.name] = _stitch_flight_prediction(base, fpred)
    return stitched_by_model_by_flight


def _plot_3d(out_png: Path, fid: str, adsb_min: pd.DataFrame, base: pd.DataFrame, stitched_by_model: dict[str, pd.DataFrame]) -> None:
    anchor_mask = pd.to_numeric(base["obs_mask"], errors="coerce").fillna(0.0) > 0.5
    lon = pd.to_numeric(adsb_min["lon"], errors="coerce")
    lat = pd.to_numeric(adsb_min["lat"], errors="coerce")
    alt = pd.to_numeric(adsb_min["alt"], errors="coerce")
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
    ax.plot(lon, lat, alt, color="#111111", lw=1.8, alpha=0.9, label="ADS-B minute GT")
    ax.scatter(
        pd.to_numeric(base.loc[anchor_mask, "obs_lon"], errors="coerce"),
        pd.to_numeric(base.loc[anchor_mask, "obs_lat"], errors="coerce"),
        pd.to_numeric(base.loc[anchor_mask, "obs_alt"], errors="coerce"),
        s=22,
        color="#000000",
        alpha=0.9,
        label="ADS-C anchors",
        zorder=6,
    )
    for model_name, stitched in stitched_by_model.items():
        s = stitched.sort_values("minute_ts").reset_index(drop=True)
        style = MODEL_STYLES.get(model_name, {}).copy()
        ax.plot(
            pd.to_numeric(s["pred_lon"], errors="coerce"),
            pd.to_numeric(s["pred_lat"], errors="coerce"),
            pd.to_numeric(s["pred_alt"], errors="coerce"),
            label=model_name,
            **style,
        )
    ax.set_title(f"{fid} | ADS-B aligned 3D recovery")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Altitude")
    ax.view_init(elev=20, azim=-122)
    ax.set_proj_type("persp")
    ax.set_box_aspect((max(lon_m, 1.0), max(lat_m, 1.0), z_display))
    ax.grid(True, alpha=0.28)
    ax.xaxis.pane.set_alpha(0.04)
    ax.yaxis.pane.set_alpha(0.04)
    ax.zaxis.pane.set_alpha(0.04)
    ax.legend(fontsize=8, ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _plot_alt_2d(out_png: Path, fid: str, adsb_min: pd.DataFrame, base: pd.DataFrame, stitched_by_model: dict[str, pd.DataFrame]) -> None:
    anchor_mask = pd.to_numeric(base["obs_mask"], errors="coerce").fillna(0.0) > 0.5
    start_ts = adsb_min["minute_ts"].min()
    x_adsb = (pd.to_datetime(adsb_min["minute_ts"], utc=True) - start_ts).dt.total_seconds().div(60.0)
    fig, ax = plt.subplots(figsize=(11.6, 5.6), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(x_adsb, pd.to_numeric(adsb_min["alt"], errors="coerce"), color="#111111", lw=2.0, alpha=0.9, label="ADS-B minute GT")
    x_anchor = (pd.to_datetime(base.loc[anchor_mask, "minute_ts"], utc=True) - start_ts).dt.total_seconds().div(60.0)
    ax.scatter(
        x_anchor,
        pd.to_numeric(base.loc[anchor_mask, "obs_alt"], errors="coerce"),
        s=24,
        color="#000000",
        alpha=0.9,
        label="ADS-C anchors",
        zorder=6,
    )
    for model_name, stitched in stitched_by_model.items():
        s = stitched.sort_values("minute_ts").reset_index(drop=True)
        x = (pd.to_datetime(s["minute_ts"], utc=True) - start_ts).dt.total_seconds().div(60.0)
        style = MODEL_STYLES.get(model_name, {}).copy()
        ax.plot(x, pd.to_numeric(s["pred_alt"], errors="coerce"), label=model_name, **style)
    ax.set_title(f"{fid} | Altitude-only minute aggregation")
    ax.set_xlabel("Minutes from ADS-B window start")
    ax.set_ylabel("Altitude")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adsb_min_dir = out_dir / "adsb_minute"
    merged_dir = out_dir / "merged_minute"
    plot3d_dir = out_dir / "plots_3d_adsb_aligned"
    plotalt_dir = out_dir / "plots_alt_2d_adsb_aligned"
    for d in [adsb_min_dir, merged_dir, plot3d_dir, plotalt_dir]:
        d.mkdir(parents=True, exist_ok=True)

    samples = pd.read_parquet(args.samples_parquet).copy()
    samples["minute_ts"] = pd.to_datetime(samples["minute_ts"], utc=True)
    matches = pd.read_csv(args.matches_csv)
    matches = matches[(pd.to_numeric(matches["overlap_ratio"], errors="coerce") > 0.99) & (pd.to_numeric(matches["contain_flag"], errors="coerce") >= 1)].copy()
    search_dirs = [Path(args.points_dir)]
    if args.points_dir_extra:
        search_dirs.append(Path(args.points_dir_extra))
    flight_ids = matches["adsc_flight_id"].astype(str).tolist()
    stitched_by_model_by_flight = _prepare_predictions(samples=samples, flight_ids=flight_ids)

    summary_rows: list[dict] = []
    for _, row in matches.iterrows():
        fid = str(row["adsc_flight_id"])
        base = samples[samples["flight_id"].astype(str) == fid].sort_values("minute_ts").reset_index(drop=True).copy()
        if base.empty:
            continue
        points_csv = _find_points_file(row, search_dirs)
        if points_csv is None:
            continue
        adsb_min = _load_adsb_minute(points_csv, base["minute_ts"].min(), base["minute_ts"].max())
        if adsb_min.empty:
            continue
        adsb_min.to_csv(adsb_min_dir / f"{fid}_adsb_minute.csv", index=False)

        merged = adsb_min.rename(columns={"alt": "adsb_alt", "lat": "adsb_lat", "lon": "adsb_lon"}).copy()
        for model_name, stitched in stitched_by_model_by_flight.get(fid, {}).items():
            s = stitched[["minute_ts", "pred_lat", "pred_lon", "pred_alt"]].copy()
            s = s.rename(
                columns={
                    "pred_lat": f"{model_name}_pred_lat",
                    "pred_lon": f"{model_name}_pred_lon",
                    "pred_alt": f"{model_name}_pred_alt",
                }
            )
            merged = merged.merge(s, on="minute_ts", how="left")
        merged.to_csv(merged_dir / f"{fid}_merged_minute.csv", index=False)

        _plot_3d(plot3d_dir / f"{fid}_adsb_aligned_3d.png", fid=fid, adsb_min=adsb_min, base=base, stitched_by_model=stitched_by_model_by_flight.get(fid, {}))
        _plot_alt_2d(plotalt_dir / f"{fid}_alt_2d.png", fid=fid, adsb_min=adsb_min, base=base, stitched_by_model=stitched_by_model_by_flight.get(fid, {}))
        summary_rows.append(
            {
                "adsc_flight_id": fid,
                "adsb_callsign": row["adsb_callsign"],
                "points_csv": str(points_csv),
                "adsb_minutes": int(len(adsb_min)),
                "adsc_minutes": int(len(base)),
            }
        )

    pd.DataFrame(summary_rows).to_csv(out_dir / "summary_adsb_aligned.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
