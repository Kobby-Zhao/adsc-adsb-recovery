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


MODEL_SPECS = {
    "A0-backbone": (
        "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_height_ablation_gated_cpu_v1/configs/ours_backbone_absolute.yaml",
        "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_height_ablation_gated_cpu_v1/ours_backbone_absolute/best.pt",
    ),
    "A1-anchor-main": (
        "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_height_ablation_gated_cpu_v1/configs/a1_linear_alt_baseline.yaml",
        "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_height_ablation_gated_cpu_v1/a1_linear_alt_baseline/best.pt",
    ),
    "A2-gated-offset": (
        "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_height_ablation_gated_cpu_v1/configs/a2_gated_offset.yaml",
        "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_height_ablation_gated_cpu_v1/a2_gated_offset/best.pt",
    ),
    "A3-gated-routed": (
        "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_height_ablation_gated_cpu_v1/configs/a3_gated_routed.yaml",
        "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_height_ablation_gated_cpu_v1/a3_gated_routed/best.pt",
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
    p = argparse.ArgumentParser("Run new gated height ablation models on selected real ADS-C windows.")
    p.add_argument(
        "--base-dir",
        default="outputs/runs/paper_showcase_cross_ocean_all_model_compare_20260518",
    )
    p.add_argument(
        "--out-dir",
        default="outputs/runs/paper_showcase_cross_ocean_gated_height_ablation_20260520",
    )
    p.add_argument("--device", default="cpu")
    return p


def _predict_models(frame: pd.DataFrame, device: str) -> dict[str, pd.DataFrame]:
    frames, _ = _build_gapwise_segments(frame)
    if not frames:
        raise RuntimeError("No gapwise segments constructed.")
    frame_all = pd.concat(frames, ignore_index=True)
    out: dict[str, pd.DataFrame] = {}
    for name, (cfg_rel, ckpt_rel) in MODEL_SPECS.items():
        cfg = load_config(str(ROOT / cfg_rel))
        cfg["training"]["device"] = device
        pred = _predict_on_frame(
            cfg=cfg,
            checkpoint=ROOT / ckpt_rel,
            frame=frame_all,
            pred_key="pred_pos",
        )
        out[name] = _stitch_flight_prediction(frame, pred)
    return out


def _base_columns(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.sort_values("minute_ts").reset_index(drop=True).copy()
    x["minute_ts"] = pd.to_datetime(x["minute_ts"], utc=True)
    t0 = x["minute_ts"].min()
    x["rel_min"] = (x["minute_ts"] - t0).dt.total_seconds().div(60.0).round().astype(int)
    known_adsb = pd.to_numeric(x.get("known_adsb", 1), errors="coerce").fillna(1).astype(int).eq(1)
    is_anchor = pd.to_numeric(x.get("is_adsc_anchor", x.get("obs_mask", 0)), errors="coerce").fillna(0).astype(int).eq(1)
    alt_col = "alt" if "alt" in x.columns else "obs_alt"
    return pd.DataFrame(
        {
            "minute_ts": x["minute_ts"],
            "rel_min": x["rel_min"],
            "adsb_alt_m": np.where(known_adsb, pd.to_numeric(x[alt_col], errors="coerce"), np.nan),
            "adsc_anchor_alt_m": np.where(is_anchor, pd.to_numeric(x[alt_col], errors="coerce"), np.nan),
            "known_adsb": known_adsb.astype(int),
            "is_adsc_anchor": is_anchor.astype(int),
        }
    )


def _merge_predictions(base: pd.DataFrame, preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = base.copy()
    for name, pred in preds.items():
        p = pred[["minute_ts", "pred_lat", "pred_lon", "pred_alt"]].copy()
        p["minute_ts"] = pd.to_datetime(p["minute_ts"], utc=True)
        p = p.rename(
            columns={
                "pred_lat": f"{name}_pred_lat",
                "pred_lon": f"{name}_pred_lon",
                "pred_alt": f"{name}_pred_alt_m",
            }
        )
        out = out.merge(p, on="minute_ts", how="left")
    return out


def _plot_window(pair_id: str, table: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 5.8), facecolor="white")
    ax.set_facecolor("white")
    x = pd.to_numeric(table["rel_min"], errors="coerce")
    ax.plot(x, table["adsb_alt_m"], color="#111111", lw=2.0, label="ADS-B")
    ax.scatter(
        x,
        table["adsc_anchor_alt_m"],
        color="#000000",
        marker="*",
        s=96,
        zorder=8,
        label="ADS-C anchors",
    )
    styles = [
        ("A0-backbone_pred_alt_m", "A0 backbone", "#7f7f7f", "--", 1.45),
        ("A1-anchor-main_pred_alt_m", "A1 anchor-main", "#1f77b4", "-", 1.85),
        ("A2-gated-offset_pred_alt_m", "A2 gated-offset", "#ff7f0e", "-", 1.85),
        ("A3-gated-routed_pred_alt_m", "A3 gated-routed", "#d62728", "-", 2.25),
    ]
    for col, label, color, ls, lw in styles:
        ax.plot(x, table[col], label=label, color=color, linestyle=ls, linewidth=lw, alpha=0.96)
    ax.set_title(f"{pair_id} | gated height ablation on real ADS-C window")
    ax.set_xlabel("Minutes from recovery frame start")
    ax.set_ylabel("Altitude (m)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _visible_error_summary(pair_id: str, table: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    y = pd.to_numeric(table["adsb_alt_m"], errors="coerce")
    mask = y.notna()
    for name in MODEL_SPECS:
        pred = pd.to_numeric(table[f"{name}_pred_alt_m"], errors="coerce")
        m = mask & pred.notna()
        if not m.any():
            continue
        err = pred[m] - y[m]
        rows.append(
            {
                "pair_id": pair_id,
                "model": name,
                "visible_adsb_points": int(m.sum()),
                "alt_mae_m": float(err.abs().mean()),
                "alt_rmse_m": float(np.sqrt((err**2).mean())),
                "alt_bias_m": float(err.mean()),
                "alt_max_abs_m": float(err.abs().max()),
            }
        )
    return rows


def main() -> int:
    args = build_parser().parse_args()
    base_dir = ROOT / args.base_dir
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

    for pair_id, (start_min, end_min) in WINDOWS.items():
        frame_csv = base_dir / pair_id / "input_recovery_frame.csv"
        if not frame_csv.exists():
            raise FileNotFoundError(frame_csv)
        print(f"[case] {pair_id}", flush=True)
        frame = pd.read_csv(frame_csv, parse_dates=["minute_ts"])
        base = _base_columns(frame)
        preds = _predict_models(frame, device=str(args.device))
        merged = _merge_predictions(base, preds)

        case_dir = out_dir / pair_id
        case_dir.mkdir(parents=True, exist_ok=True)
        merged.to_csv(case_dir / "recovered_minute_compare.csv", index=False, encoding="utf-8-sig")
        window = merged[(merged["rel_min"] >= start_min) & (merged["rel_min"] <= end_min)].copy()
        table_csv = case_dir / f"{pair_id}_{start_min}_{end_min}_gated_height_ablation_table.csv"
        plot_png = case_dir / f"{pair_id}_{start_min}_{end_min}_gated_height_ablation_plot.png"
        window.to_csv(table_csv, index=False, encoding="utf-8-sig")
        _plot_window(pair_id, window, plot_png)
        metric_rows.extend(_visible_error_summary(pair_id, window))
        summary_rows.append(
            {
                "pair_id": pair_id,
                "start_min": start_min,
                "end_min": end_min,
                "rows": int(len(window)),
                "visible_adsb_points": int(window["adsb_alt_m"].notna().sum()),
                "adsc_anchor_points": int(window["adsc_anchor_alt_m"].notna().sum()),
                "table_csv": str(table_csv.relative_to(ROOT)),
                "plot_png": str(plot_png.relative_to(ROOT)),
            }
        )
        print(f"[ok] {plot_png}", flush=True)

    pd.DataFrame(summary_rows).to_csv(out_dir / "gated_height_ablation_real_adsc_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metric_rows).to_csv(out_dir / "gated_height_ablation_visible_adsb_metrics.csv", index=False, encoding="utf-8-sig")
    print(f"[done] out_dir={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
