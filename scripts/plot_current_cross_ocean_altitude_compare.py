from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cross_ocean_gap_recovery_compare import _build_full_frame
from scripts.real_adsc_replay_eval import _predict_on_frame
from scripts.real_adsc_truequal_gapwise_eval import _build_gapwise_segments, _stitch_flight_prediction
from scripts.train import load_config


SPECS = {
    "Kalman Filter": (
        "outputs/experiments/obs_conditioned_gaponly/clean_baselines_gaponly_v1/configs/kalman_filter_clean_absolute.yaml",
        "outputs/experiments/obs_conditioned_gaponly/clean_baselines_gaponly_v1/kalman_filter_clean_absolute/best.pt",
    ),
    "LSTM-clean": (
        "outputs/experiments/obs_conditioned_gaponly/clean_baselines_gaponly_v1/configs/lstm_clean_absolute.yaml",
        "outputs/experiments/obs_conditioned_gaponly/clean_baselines_gaponly_v1/lstm_clean_absolute/best.pt",
    ),
    "BiLSTM-clean": (
        "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_bilstm_vs_backbone_v1/configs/bilstm_clean_absolute.yaml",
        "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_bilstm_vs_backbone_v1/bilstm_clean_absolute/best.pt",
    ),
    "CNN+LSTM-clean": (
        "outputs/experiments/obs_conditioned_gaponly/clean_baselines_gaponly_v1/configs/cnn_lstm_clean_absolute.yaml",
        "outputs/experiments/obs_conditioned_gaponly/clean_baselines_gaponly_v1/cnn_lstm_clean_absolute/best.pt",
    ),
    "Transformer-clean": (
        "outputs/experiments/obs_conditioned_gaponly/clean_baselines_gaponly_v1/configs/transformer_clean_absolute.yaml",
        "outputs/experiments/obs_conditioned_gaponly/clean_baselines_gaponly_v1/transformer_clean_absolute/best.pt",
    ),
    "Backbone-only": (
        "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_bilstm_vs_backbone_v1/configs/ours_backbone_absolute.yaml",
        "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_bilstm_vs_backbone_v1/ours_backbone_absolute/best.pt",
    ),
    "A1-linear-alt": (
        "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_bilstm_vs_backbone_v1/configs/a1_linear_alt_baseline.yaml",
        "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_bilstm_vs_backbone_v1/a1_linear_alt_baseline/best.pt",
    ),
    "Ours-A3": (
        "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_bilstm_vs_backbone_v1/configs/a3_risk_routed.yaml",
        "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_bilstm_vs_backbone_v1/a3_risk_routed/best.pt",
    ),
}

STYLES = {
    "Kalman Filter": {"color": "#8c8c8c", "linestyle": (0, (1, 2)), "linewidth": 1.3},
    "LSTM-clean": {"color": "#9467bd", "linestyle": "--", "linewidth": 1.45},
    "BiLSTM-clean": {"color": "#7f7f7f", "linestyle": "--", "linewidth": 1.7},
    "CNN+LSTM-clean": {"color": "#8c564b", "linestyle": "--", "linewidth": 1.45},
    "Transformer-clean": {"color": "#ff7f0e", "linestyle": "--", "linewidth": 1.55},
    "Backbone-only": {"color": "#1f77b4", "linestyle": "-.", "linewidth": 1.8},
    "A1-linear-alt": {"color": "#2ca02c", "linestyle": "-", "linewidth": 1.9},
    "Ours-A3": {"color": "#d62728", "linestyle": "-", "linewidth": 2.3},
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Plot current gap-only trained model altitude recovery on cross-ocean cases.")
    p.add_argument(
        "--overlay-summary-csv",
        default="outputs/runs/adsc_cross_ocean_top10_highest_anchor_20260429/overlay_summary.csv",
    )
    p.add_argument(
        "--adsb-minute-csv",
        default="outputs/runs/adsc_cross_ocean_top10_highest_anchor_20260429/top10_cross_ocean_highest_anchor_adsb_minute_full_flights.csv",
    )
    p.add_argument("--out-dir", default="outputs/runs/current_cross_ocean_altitude_compare_20260517")
    p.add_argument("--max-cases", type=int, default=10)
    p.add_argument("--models", default="BiLSTM-clean,Backbone-only,A1-linear-alt,Ours-A3")
    return p


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


def _shade_gaps(ax: plt.Axes, minutes: pd.Series, obs_mask: pd.Series) -> None:
    missing = pd.to_numeric(obs_mask, errors="coerce").fillna(0).to_numpy() <= 0.5
    x = minutes.to_numpy(dtype=float)
    if len(x) == 0:
        return
    start = None
    for i, is_missing in enumerate(missing):
        if is_missing and start is None:
            start = i
        if start is not None and (not is_missing or i == len(missing) - 1):
            end = i if is_missing and i == len(missing) - 1 else i - 1
            ax.axvspan(x[start], x[end], color="#ff9999", alpha=0.16, linewidth=0)
            start = None


def _plot_case(pair_id: str, frame: pd.DataFrame, stitched: dict[str, pd.DataFrame], out_png: Path) -> dict[str, float]:
    x = frame.sort_values("minute_ts").reset_index(drop=True).copy()
    start = pd.to_datetime(x["minute_ts"], utc=True).min()
    x_min = (pd.to_datetime(x["minute_ts"], utc=True) - start).dt.total_seconds().div(60.0)
    known = x["known_adsb"].astype(int) == 1
    anchors = x["is_adsc_anchor"].astype(int) == 1
    missing = (pd.to_numeric(x["obs_mask"], errors="coerce").fillna(0.0).to_numpy() <= 0.5)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.2, 7.4), sharex=True, facecolor="white")
    ax1.set_facecolor("white")
    ax2.set_facecolor("white")

    _shade_gaps(ax1, x_min, x["obs_mask"])
    _shade_gaps(ax2, x_min, x["obs_mask"])

    ax1.plot(x_min[known], pd.to_numeric(x.loc[known, "alt"], errors="coerce"), color="#111111", lw=2.0, label="ADS-B minute GT")
    ax1.scatter(x_min[anchors], pd.to_numeric(x.loc[anchors, "alt"], errors="coerce"), s=34, color="#ff7f0e", edgecolor="#111111", zorder=6, label="ADS-C anchor")

    summary: dict[str, float] = {}
    true_alt = pd.to_numeric(x["alt"], errors="coerce").to_numpy(dtype=float)
    for name, s in stitched.items():
        g = s.sort_values("minute_ts").reset_index(drop=True)
        gx = (pd.to_datetime(g["minute_ts"], utc=True) - start).dt.total_seconds().div(60.0)
        pred_alt = pd.to_numeric(g["pred_alt"], errors="coerce").to_numpy(dtype=float)
        style = STYLES.get(name, {})
        ax1.plot(gx, pred_alt, label=name, **style)
        n = min(len(pred_alt), len(true_alt), len(missing))
        # Missing intervals in this replay have no dense ADS-B ground truth. The frame altitude is
        # the anchor/ADS-B initialized reference used to condition the recovery, not a hidden label.
        dev = np.abs(pred_alt[:n] - true_alt[:n])
        valid = missing[:n] & np.isfinite(dev)
        mae = float(np.mean(dev[valid])) if valid.any() else float("nan")
        summary[f"{name}_missing_ref_dev_mae_m"] = mae
        ax2.plot(gx[:n], dev[:n], label=f"{name} ref-dev={mae:.1f}m", **style)

    ax1.set_title(f"{pair_id} | Current Trained Models: Cross-Ocean Altitude Recovery")
    ax1.set_ylabel("Altitude (m)")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8, ncol=3)

    ax2.set_title("Missing-Interval Deviation from Anchor-Interpolated Altitude Reference")
    ax2.set_xlabel("Minutes from flight start")
    ax2.set_ylabel("|Prediction - Reference| (m)")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return summary


def main() -> int:
    args = build_parser().parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model_names = [x.strip() for x in args.models.split(",") if x.strip()]
    unknown = [m for m in model_names if m not in SPECS]
    if unknown:
        raise ValueError(f"Unknown model names: {unknown}; available={sorted(SPECS)}")

    summary = pd.read_csv(ROOT / args.overlay_summary_csv)
    rows = []
    for _, row in summary.head(args.max_cases).iterrows():
        pair_id = str(row["pair_id"])
        frame, adsb, adsc = _build_full_frame(
            pair_id=pair_id,
            overlay_csv=ROOT / str(row["overlay_csv"]),
            adsb_minute_csv=ROOT / args.adsb_minute_csv,
        )
        case_dir = out_dir / pair_id
        case_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(case_dir / "input_recovery_frame.csv", index=False)
        adsb.to_csv(case_dir / "known_adsb_minute.csv", index=False)
        adsc.to_csv(case_dir / "adsc_anchor_points.csv", index=False)

        stitched = _run_models(frame, model_names)
        merged = frame[["sample_id", "flight_id", "minute_ts", "obs_mask", "known_adsb", "is_adsc_anchor", "obs_source", "lat", "lon", "alt"]].copy()
        for name, s in stitched.items():
            cols = s[["minute_ts", "pred_lat", "pred_lon", "pred_alt"]].rename(
                columns={
                    "pred_lat": f"{name}_pred_lat",
                    "pred_lon": f"{name}_pred_lon",
                    "pred_alt": f"{name}_pred_alt",
                }
            )
            merged = merged.merge(cols, on="minute_ts", how="left")
        merged.to_csv(case_dir / "recovered_minute_compare.csv", index=False)

        plot_path = case_dir / f"{pair_id}_current_altitude_compare.png"
        metrics = _plot_case(pair_id, frame, stitched, plot_path)
        rows.append(
            {
                "pair_id": pair_id,
                "plot": str(plot_path.relative_to(ROOT)),
                "recovered_csv": str((case_dir / "recovered_minute_compare.csv").relative_to(ROOT)),
                "known_adsb_minutes": int(frame["known_adsb"].astype(int).sum()),
                "adsc_anchor_minutes": int(frame["is_adsc_anchor"].astype(int).sum()),
                "missing_minutes": int((frame["obs_mask"].astype(int) == 0).sum()),
                **metrics,
            }
        )
        print(f"[ok] {pair_id} -> {plot_path}")

    pd.DataFrame(rows).to_csv(out_dir / "current_model_altitude_compare_summary.csv", index=False)
    print(f"[done] summary={out_dir / 'current_model_altitude_compare_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
