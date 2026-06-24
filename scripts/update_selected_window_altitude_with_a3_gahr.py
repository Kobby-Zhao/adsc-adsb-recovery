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


SPECS = {
    "Ours-A3-old": (
        "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_physical_time_ablation_v1/configs/a3_risk_routed.yaml",
        "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_physical_time_ablation_v1/a3_risk_routed/best.pt",
    ),
    "Ours-A3-GAHR": (
        "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_a3_gahr_v1/configs/a3_gahr_routed.yaml",
        "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_a3_gahr_v1/a3_gahr_routed/best.pt",
    ),
    "Ours-A3-GAHR-corrected": (
        "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_a3_gahr_corrected_v1/configs/a3_gahr_corrected_routed.yaml",
        "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_a3_gahr_corrected_v1/a3_gahr_corrected_routed/best.pt",
    ),
    "BiLSTM-clean": (
        "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_physical_time_ablation_v1/configs/bilstm_clean_absolute.yaml",
        "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_physical_time_ablation_v1/bilstm_clean_absolute/best.pt",
    ),
}

WINDOWS = {
    "39d2a8_0013": (150, 560),
    "407fcd_0019": (150, 530),
    "4076e8_0021": (150, 540),
    "a9c5c2_0001": (70, 350),
    "407943_0020": (50, 500),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", default="outputs/runs/paper_showcase_cross_ocean_all_model_compare_20260518")
    p.add_argument(
        "--out-dir",
        default="outputs/runs/paper_showcase_cross_ocean_a3_gahr_compare_20260520/selected_window_altitude_tables_plots",
    )
    p.add_argument("--device", default="cpu")
    return p


def _run_models(frame: pd.DataFrame, device: str) -> dict[str, pd.DataFrame]:
    frames, _ = _build_gapwise_segments(frame)
    if not frames:
        raise RuntimeError("No gapwise segments constructed.")
    frame_all = pd.concat(frames, ignore_index=True)
    out: dict[str, pd.DataFrame] = {}
    for name, (cfg_rel, ckpt_rel) in SPECS.items():
        cfg = load_config(str(ROOT / cfg_rel))
        cfg["training"]["device"] = str(device)
        pred = _predict_on_frame(cfg=cfg, checkpoint=ROOT / ckpt_rel, frame=frame_all, pred_key="pred_pos")
        out[name] = _stitch_flight_prediction(frame, pred)
    return out


def _build_table(frame: pd.DataFrame, stitched: dict[str, pd.DataFrame], start_min: int, end_min: int) -> pd.DataFrame:
    x = frame.sort_values("minute_ts").reset_index(drop=True).copy()
    t0 = pd.to_datetime(x["minute_ts"], utc=True).min()
    x["rel_min"] = (pd.to_datetime(x["minute_ts"], utc=True) - t0).dt.total_seconds().div(60).round().astype(int)
    w = x[(x["rel_min"] >= start_min) & (x["rel_min"] <= end_min)].copy()
    table = pd.DataFrame(
        {
            "相对分钟": w["rel_min"],
            "时间_UTC_精确到分钟": pd.to_datetime(w["minute_ts"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S%z"),
            "原始ADS-B高度_m": np.where(w["known_adsb"].astype(int).eq(1), pd.to_numeric(w["alt"], errors="coerce"), np.nan),
            "原始ADS-C锚点高度_m": np.where(w["is_adsc_anchor"].astype(int).eq(1), pd.to_numeric(w["alt"], errors="coerce"), np.nan),
        }
    )
    for name, pred in stitched.items():
        p = pred.copy()
        p["minute_ts"] = pd.to_datetime(p["minute_ts"], utc=True)
        alt_map = p.set_index("minute_ts")["pred_alt"]
        table[f"{name}_恢复高度_m"] = pd.to_datetime(w["minute_ts"], utc=True).map(alt_map).to_numpy()
    return table


def _plot(pair_id: str, table: pd.DataFrame, out_png: Path) -> None:
    x = pd.to_numeric(table["相对分钟"], errors="coerce")
    fig, ax = plt.subplots(figsize=(12.6, 5.8), facecolor="white")
    ax.plot(x, table["原始ADS-B高度_m"], color="#111111", lw=2.0, label="ADS-B")
    ax.scatter(x, table["原始ADS-C锚点高度_m"], color="#2ca02c", edgecolor="#111111", s=46, zorder=7, label="ADS-C anchors")
    styles = [
        ("Ours-A3-old_恢复高度_m", "Ours-A3 old", "#999999", "--", 1.8),
        ("Ours-A3-GAHR_恢复高度_m", "Ours-A3-GAHR", "#d00000", "-", 1.7),
        ("Ours-A3-GAHR-corrected_恢复高度_m", "Ours-A3-GAHR-corrected", "#0057b8", "-", 2.5),
        ("BiLSTM-clean_恢复高度_m", "BiLSTM", "#f77f00", "-.", 1.6),
    ]
    for col, label, color, ls, lw in styles:
        if col in table:
            ax.plot(x, table[col], label=label, color=color, linestyle=ls, linewidth=lw, alpha=0.95)
    ax.set_title(f"{pair_id} | A3-GAHR selected-window altitude recovery")
    ax.set_xlabel("Minutes from recovery frame start")
    ax.set_ylabel("Altitude (m)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, ncol=3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    base_dir = ROOT / args.base_dir
    out_dir = ROOT / args.out_dir
    rows = []
    for pair_id, (start_min, end_min) in WINDOWS.items():
        frame_csv = base_dir / pair_id / "input_recovery_frame.csv"
        if not frame_csv.exists():
            raise RuntimeError(f"Missing {frame_csv}")
        print(f"[case] {pair_id}", flush=True)
        frame = pd.read_csv(frame_csv, parse_dates=["minute_ts"])
        stitched = _run_models(frame, device=str(args.device))
        table = _build_table(frame, stitched, start_min, end_min)
        case_dir = out_dir / pair_id
        case_dir.mkdir(parents=True, exist_ok=True)
        table_csv = case_dir / f"{pair_id}_{start_min}_{end_min}_a3_gahr_compare_table.csv"
        plot_png = case_dir / f"{pair_id}_{start_min}_{end_min}_a3_gahr_compare_plot.png"
        table.to_csv(table_csv, index=False, encoding="utf-8-sig")
        _plot(pair_id, table, plot_png)
        rows.append(
            {
                "pair_id": pair_id,
                "start_min": start_min,
                "end_min": end_min,
                "rows": len(table),
                "adsb_points": int(table["原始ADS-B高度_m"].notna().sum()),
                "adsc_anchor_points": int(table["原始ADS-C锚点高度_m"].notna().sum()),
                "table_csv": str(table_csv.relative_to(ROOT)),
                "plot_png": str(plot_png.relative_to(ROOT)),
            }
        )
        print(f"[ok] {plot_png}", flush=True)
    pd.DataFrame(rows).to_csv(out_dir / "a3_gahr_selected_window_summary.csv", index=False, encoding="utf-8-sig")
    print(f"[done] {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
