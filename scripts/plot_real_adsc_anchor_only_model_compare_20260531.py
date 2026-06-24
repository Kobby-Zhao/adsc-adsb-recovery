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
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.real_adsc_replay_eval import _predict_on_frame  # noqa: E402
from scripts.train import load_config  # noqa: E402


STYLE = {
    "ADS-C anchors": {"color": "black", "marker": "*", "markersize": 10, "linestyle": "None"},
    "BiMamba": {"color": "red", "linestyle": "-", "linewidth": 2.2},
    "UniLSTM-proto": {"color": "tab:blue", "linestyle": "--", "linewidth": 1.6},
    "BiLSTM-proto": {"color": "tab:green", "linestyle": "-.", "linewidth": 1.6},
    "CNN-LSTM-proto": {"color": "tab:purple", "linestyle": "--", "linewidth": 1.6},
    "Transformer-proto": {"color": "tab:orange", "linestyle": ":", "linewidth": 1.8},
}

MODEL_SPECS = [
    {
        "name": "BiMamba",
        "config": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_bimamba.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_bimamba/best.pt",
    },
    {
        "name": "UniLSTM-proto",
        "config": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_unilstm_proto.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_unilstm_proto/best.pt",
    },
    {
        "name": "BiLSTM-proto",
        "config": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_bilstm_proto.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_bilstm_proto/best.pt",
    },
    {
        "name": "CNN-LSTM-proto",
        "config": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_cnnlstm_proto.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_cnnlstm_proto/best.pt",
    },
    {
        "name": "Transformer-proto",
        "config": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_transformer_proto.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_transformer_proto/best.pt",
    },
]


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "SimSun", "DejaVu Serif"],
            "axes.unicode_minus": False,
            "savefig.dpi": 300,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
        }
    )


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


def _build_anchor_only_frame(adsc_anchor_csv: Path, pair_id: str) -> pd.DataFrame:
    adsc = pd.read_csv(adsc_anchor_csv).copy()
    adsc["timestamp"] = pd.to_datetime(adsc["timestamp"], utc=True)
    adsc["anchor_minute_ts"] = adsc["timestamp"].dt.round("1min")
    adsc = (
        adsc.rename(columns={"latitude": "anchor_lat", "longitude": "anchor_lon", "altitude_m": "anchor_alt"})
        .groupby("anchor_minute_ts", as_index=False)
        .last()
        .sort_values("anchor_minute_ts")
        .reset_index(drop=True)
    )
    if len(adsc) < 2:
        raise RuntimeError(f"Not enough ADS-C anchors in {adsc_anchor_csv}")

    full_ts = pd.date_range(
        start=pd.to_datetime(adsc["anchor_minute_ts"].min(), utc=True),
        end=pd.to_datetime(adsc["anchor_minute_ts"].max(), utc=True),
        freq="1min",
        tz="UTC",
    )
    full = pd.DataFrame({"minute_ts": full_ts})
    full["sample_id"] = f"{pair_id}_adsc_anchor_only"
    full["flight_id"] = pair_id
    full = full.merge(adsc, left_on="minute_ts", right_on="anchor_minute_ts", how="left")
    # Some source CSVs already contain a flight_id column; keep the replay id stable.
    full["flight_id"] = pair_id
    full["is_adsc_anchor"] = full["anchor_minute_ts"].notna().astype(int)
    full["known_adsb"] = 0
    full["obs_mask"] = full["is_adsc_anchor"].astype(int)
    full["obs_source"] = np.where(full["is_adsc_anchor"] == 1, "adsc_anchor", "missing")

    for c in ["anchor_lat", "anchor_lon", "anchor_alt"]:
        full[c] = pd.to_numeric(full[c], errors="coerce")

    full["lat"] = full["anchor_lat"].interpolate(limit_direction="both")
    full["lon"] = full["anchor_lon"].interpolate(limit_direction="both")
    full["alt"] = full["anchor_alt"].interpolate(limit_direction="both")

    lat = full["lat"].to_numpy(dtype=float)
    lon = full["lon"].to_numpy(dtype=float)
    alt = full["alt"].to_numpy(dtype=float)
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
    frame["obs_lat"] = lat
    frame["obs_lon"] = lon
    frame["obs_alt"] = alt
    return frame


def _run_models(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    stitched: dict[str, pd.DataFrame] = {}
    for spec in MODEL_SPECS:
        cfg = load_config(_resolve(spec["config"]))
        pred = _predict_on_frame(cfg=cfg, checkpoint=_resolve(spec["checkpoint"]), frame=frame, pred_key="pred_pos")
        stitched[spec["name"]] = pred.sort_values("minute_ts").reset_index(drop=True)
    return stitched


def _plot_altitude(frame: pd.DataFrame, stitched: dict[str, pd.DataFrame], out_base: Path) -> None:
    x = frame.sort_values("minute_ts").reset_index(drop=True).copy()
    x_min = (pd.to_datetime(x["minute_ts"], utc=True) - pd.to_datetime(x["minute_ts"], utc=True).min()).dt.total_seconds() / 60.0
    anchors = x["is_adsc_anchor"].astype(int) == 1

    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ax.plot(x_min[anchors], pd.to_numeric(x.loc[anchors, "obs_alt"], errors="coerce"), label="ADS-C anchors", zorder=8, **STYLE["ADS-C anchors"])
    for model_name, s in stitched.items():
        gx = (pd.to_datetime(s["minute_ts"], utc=True) - pd.to_datetime(s["minute_ts"], utc=True).min()).dt.total_seconds() / 60.0
        ax.plot(gx, pd.to_numeric(s["pred_alt"], errors="coerce"), label=model_name, zorder=6 if model_name == "BiMamba" else 4, **STYLE[model_name])

    ax.set_title(f"{x['flight_id'].iloc[0]} | real ADS-C anchor-only recovery")
    ax.set_xlabel("Minutes from first ADS-C anchor")
    ax.set_ylabel("Altitude (m)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_latlon(frame: pd.DataFrame, stitched: dict[str, pd.DataFrame], out_base: Path) -> None:
    x = frame.sort_values("minute_ts").reset_index(drop=True).copy()
    anchors = x["is_adsc_anchor"].astype(int) == 1

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.plot(x.loc[anchors, "obs_lon"], x.loc[anchors, "obs_lat"], label="ADS-C anchors", zorder=8, **STYLE["ADS-C anchors"])
    for model_name, s in stitched.items():
        ax.plot(pd.to_numeric(s["pred_lon"], errors="coerce"), pd.to_numeric(s["pred_lat"], errors="coerce"), label=model_name, zorder=6 if model_name == "BiMamba" else 4, **STYLE[model_name])
    ax.set_title(f"{x['flight_id'].iloc[0]} | real ADS-C 2D recovery")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--selected-cases-csv",
        default="outputs/runs/clean_cross_ocean_adsc_adsb_cases_extra_20260517/selected_clean_cross_ocean_cases.csv",
    )
    ap.add_argument("--pair-ids", default="", help="Comma-separated pair_ids. Empty means all selected rows.")
    ap.add_argument(
        "--out-dir",
        default="outputs/runs/real_adsc_anchor_only_current_models_20260531",
    )
    args = ap.parse_args()

    _set_plot_style()
    selected = pd.read_csv(_resolve(args.selected_cases_csv))
    if args.pair_ids.strip():
        wanted = {x.strip() for x in args.pair_ids.split(",") if x.strip()}
        selected = selected[selected["pair_id"].astype(str).isin(wanted)].copy()
    if selected.empty:
        raise RuntimeError("No selected real ADS-C cases found after filtering.")

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for spec in MODEL_SPECS:
        for k in ("config", "checkpoint"):
            if not _resolve(spec[k]).exists():
                raise FileNotFoundError(f"Missing {k}: {_resolve(spec[k])}")

    rows = []
    for _, row in selected.iterrows():
        pair_id = str(row["pair_id"])
        frame = _build_anchor_only_frame(_resolve(row["adsc_anchor_csv"]), pair_id)
        stitched = _run_models(frame)
        case_dir = out_dir / pair_id
        case_dir.mkdir(parents=True, exist_ok=True)
        _plot_altitude(frame, stitched, case_dir / f"real_adsc_alt_compare_{pair_id}")
        _plot_latlon(frame, stitched, case_dir / f"real_adsc_latlon_compare_{pair_id}")

        merged = frame[["sample_id", "flight_id", "minute_ts", "obs_mask", "is_adsc_anchor", "obs_source", "obs_lat", "obs_lon", "obs_alt"]].copy()
        for model_name, s in stitched.items():
            merged[f"{model_name}_pred_lat"] = pd.to_numeric(s["pred_lat"], errors="coerce")
            merged[f"{model_name}_pred_lon"] = pd.to_numeric(s["pred_lon"], errors="coerce")
            merged[f"{model_name}_pred_alt"] = pd.to_numeric(s["pred_alt"], errors="coerce")
            rows.append(
                {
                    "pair_id": pair_id,
                    "model": model_name,
                    "anchor_count": int(frame["is_adsc_anchor"].sum()),
                    "minutes": int(len(frame)),
                }
            )
        merged.to_csv(case_dir / "recovered_minute_compare_current_models.csv", index=False)

    pd.DataFrame(rows).to_csv(out_dir / "real_adsc_current_model_compare_index.csv", index=False)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
