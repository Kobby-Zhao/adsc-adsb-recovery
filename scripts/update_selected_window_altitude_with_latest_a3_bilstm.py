from __future__ import annotations

import argparse
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

from scripts.real_adsc_replay_eval import _predict_on_frame
from scripts.real_adsc_truequal_gapwise_eval import _build_gapwise_segments, _stitch_flight_prediction
from scripts.train import load_config


LATEST_SPECS = {
    "BiLSTM-clean": (
        "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_physical_time_ablation_v1/configs/bilstm_clean_absolute.yaml",
        "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_physical_time_ablation_v1/bilstm_clean_absolute/best.pt",
    ),
    "Ours-A3": (
        "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_physical_time_ablation_v1/configs/a3_risk_routed.yaml",
        "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_physical_time_ablation_v1/a3_risk_routed/best.pt",
    ),
}

WINDOWS = {
    "39d2a8_0013": (150, 560),
    "407fcd_0019": (150, 530),
    "4076e8_0021": (150, 540),
    "a9c5c2_0001": (70, 350),
    "407943_0020": (50, 500),
}

TABLE_COLUMNS = {
    "相对分钟": "rel_min",
    "时间_UTC_精确到分钟": "minute_ts",
    "原始ADS-B高度_m": "adsb_alt_m",
    "原始ADS-C锚点高度_m": "adsc_anchor_alt_m",
    "本文方案A3恢复高度_m": "ours_a3_alt_m",
    "LSTM恢复高度_m": "lstm_alt_m",
    "BiLSTM恢复高度_m": "bilstm_alt_m",
    "CNN+LSTM恢复高度_m": "cnnlstm_alt_m",
    "Transformer恢复高度_m": "transformer_alt_m",
    "Kalman Filter恢复高度_m": "kalman_alt_m",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Update selected cross-ocean altitude windows with latest A3 and BiLSTM.")
    p.add_argument(
        "--base-dir",
        default="outputs/runs/paper_showcase_cross_ocean_all_model_compare_20260518",
    )
    p.add_argument("--device", default="cpu", help="Use cpu by default for portable inference.")
    return p


def _run_latest_models(frame: pd.DataFrame, device: str) -> dict[str, pd.DataFrame]:
    frames, _ = _build_gapwise_segments(frame)
    if not frames:
        raise RuntimeError("No gapwise segments constructed.")
    frame_all = pd.concat(frames, ignore_index=True)
    out: dict[str, pd.DataFrame] = {}
    for name, (cfg_rel, ckpt_rel) in LATEST_SPECS.items():
        cfg = load_config(str(ROOT / cfg_rel))
        cfg["training"]["device"] = device
        pred = _predict_on_frame(cfg=cfg, checkpoint=ROOT / ckpt_rel, frame=frame_all, pred_key="pred_pos")
        out[name] = _stitch_flight_prediction(frame, pred)
    return out


def _replace_model_columns(recovered: pd.DataFrame, stitched: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = recovered.copy()
    for name, pred in stitched.items():
        drop_cols = [c for c in out.columns if c.startswith(f"{name}_pred_")]
        if drop_cols:
            out = out.drop(columns=drop_cols)
        cols = pred[["minute_ts", "pred_lat", "pred_lon", "pred_alt"]].copy()
        cols["minute_ts"] = pd.to_datetime(cols["minute_ts"], utc=True)
        cols = cols.rename(
            columns={
                "pred_lat": f"{name}_pred_lat",
                "pred_lon": f"{name}_pred_lon",
                "pred_alt": f"{name}_pred_alt",
            }
        )
        out["minute_ts"] = pd.to_datetime(out["minute_ts"], utc=True)
        out = out.merge(cols, on="minute_ts", how="left")
    return out


def _build_window_table(recovered: pd.DataFrame, start_min: int, end_min: int) -> pd.DataFrame:
    x = recovered.sort_values("minute_ts").reset_index(drop=True).copy()
    t0 = pd.to_datetime(x["minute_ts"], utc=True).min()
    x["rel_min"] = (pd.to_datetime(x["minute_ts"], utc=True) - t0).dt.total_seconds().div(60).round().astype(int)
    w = x[(x["rel_min"] >= start_min) & (x["rel_min"] <= end_min)].copy()

    table = pd.DataFrame(
        {
            "相对分钟": w["rel_min"],
            "时间_UTC_精确到分钟": pd.to_datetime(w["minute_ts"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S%z"),
            "原始ADS-B高度_m": np.where(w["known_adsb"].astype(int).eq(1), pd.to_numeric(w["alt"], errors="coerce"), np.nan),
            "原始ADS-C锚点高度_m": np.where(w["is_adsc_anchor"].astype(int).eq(1), pd.to_numeric(w["alt"], errors="coerce"), np.nan),
            "本文方案A3恢复高度_m": pd.to_numeric(w["Ours-A3_pred_alt"], errors="coerce"),
            "LSTM恢复高度_m": pd.to_numeric(w["LSTM-clean_pred_alt"], errors="coerce"),
            "BiLSTM恢复高度_m": pd.to_numeric(w["BiLSTM-clean_pred_alt"], errors="coerce"),
            "CNN+LSTM恢复高度_m": pd.to_numeric(w["CNN+LSTM-clean_pred_alt"], errors="coerce"),
            "Transformer恢复高度_m": pd.to_numeric(w["Transformer-clean_pred_alt"], errors="coerce"),
            "Kalman Filter恢复高度_m": pd.to_numeric(w["Kalman Filter_pred_alt"], errors="coerce"),
        }
    )
    return table


def _plot_table(pair_id: str, table: pd.DataFrame, out_png: Path) -> None:
    x = pd.to_numeric(table["相对分钟"], errors="coerce")
    fig, ax = plt.subplots(figsize=(12.6, 5.8), facecolor="white")
    ax.set_facecolor("white")

    ax.plot(x, table["原始ADS-B高度_m"], color="#111111", lw=2.0, label="ADS-B")
    ax.scatter(x, table["原始ADS-C锚点高度_m"], color="#2ca02c", edgecolor="#111111", s=42, zorder=7, label="ADS-C anchors")

    styles = [
        ("本文方案A3恢复高度_m", "Ours-A3", "#d00000", "-", 2.25),
        ("BiLSTM恢复高度_m", "BiLSTM", "#6b6b6b", "--", 1.75),
        ("LSTM恢复高度_m", "LSTM", "#9467bd", "--", 1.35),
        ("CNN+LSTM恢复高度_m", "CNN+LSTM", "#8c564b", "--", 1.35),
        ("Transformer恢复高度_m", "Transformer", "#ff7f0e", "--", 1.45),
        ("Kalman Filter恢复高度_m", "Kalman Filter", "#8c8c8c", (0, (1, 2)), 1.25),
    ]
    for col, label, color, ls, lw in styles:
        ax.plot(x, table[col], label=label, color=color, linestyle=ls, linewidth=lw, alpha=0.95)

    ax.set_title(f"{pair_id} | Selected-window altitude recovery")
    ax.set_xlabel("Minutes from recovery frame start")
    ax.set_ylabel("Altitude (m)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    base_dir = ROOT / args.base_dir
    window_dir = base_dir / "selected_window_altitude_tables_plots"
    rows = []

    for pair_id, (start_min, end_min) in WINDOWS.items():
        case_dir = base_dir / pair_id
        frame_csv = case_dir / "input_recovery_frame.csv"
        recovered_csv = case_dir / "recovered_minute_compare.csv"
        if not frame_csv.exists() or not recovered_csv.exists():
            raise RuntimeError(f"Missing case files for {pair_id}: {frame_csv}, {recovered_csv}")

        print(f"[case] {pair_id}: updating latest BiLSTM + A3", flush=True)
        frame = pd.read_csv(frame_csv, parse_dates=["minute_ts"])
        recovered = pd.read_csv(recovered_csv, parse_dates=["minute_ts"])
        stitched = _run_latest_models(frame, device=str(args.device))
        updated = _replace_model_columns(recovered, stitched)
        updated.to_csv(recovered_csv, index=False)

        table = _build_window_table(updated, start_min=start_min, end_min=end_min)
        out_case = window_dir / pair_id
        table_csv = out_case / f"{pair_id}_{start_min}_{end_min}_altitude_compare_table.csv"
        plot_png = out_case / f"{pair_id}_{start_min}_{end_min}_altitude_compare_plot.png"
        out_case.mkdir(parents=True, exist_ok=True)
        table.to_csv(table_csv, index=False, encoding="utf-8-sig")
        _plot_table(pair_id, table, plot_png)
        rows.append(
            {
                "pair_id": pair_id,
                "start_min": start_min,
                "end_min": end_min,
                "rows": len(table),
                "table_csv": str(table_csv.relative_to(ROOT)),
                "plot_png": str(plot_png.relative_to(ROOT)),
                "adsb_points_in_window": int(table["原始ADS-B高度_m"].notna().sum()),
                "adsc_anchor_points_in_window": int(table["原始ADS-C锚点高度_m"].notna().sum()),
                "updated_models": "BiLSTM-clean,Ours-A3",
            }
        )
        print(f"[ok] {pair_id} -> {plot_png}", flush=True)

    summary = pd.DataFrame(rows)
    summary.to_csv(window_dir / "selected_window_altitude_compare_summary.csv", index=False, encoding="utf-8-sig")
    print(f"[done] summary={window_dir / 'selected_window_altitude_compare_summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
