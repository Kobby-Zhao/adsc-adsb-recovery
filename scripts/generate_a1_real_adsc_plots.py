from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

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


A1_CONFIG = ROOT / "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_physical_time_ablation_v1/configs/a1_linear_alt_baseline.yaml"
A1_CHECKPOINT = ROOT / "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_physical_time_ablation_v1/a1_linear_alt_baseline/best.pt"

WINDOWS = {
    "39d2a8_0013": (150, 560),
    "407fcd_0019": (150, 530),
    "4076e8_0021": (150, 540),
    "a9c5c2_0001": (70, 350),
    "407943_0020": (50, 500),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Generate A1-only real ADS-C recovery plots for selected showcase windows.")
    p.add_argument(
        "--base-dir",
        default="outputs/runs/paper_showcase_cross_ocean_all_model_compare_20260518",
    )
    p.add_argument(
        "--out-dir",
        default="outputs/runs/0524/a1_real_adsc_windows",
    )
    p.add_argument("--device", default="cpu")
    return p


def _predict_a1(frame: pd.DataFrame, device: str) -> pd.DataFrame:
    frames, _ = _build_gapwise_segments(frame)
    if not frames:
        raise RuntimeError("No gapwise segments constructed.")
    frame_all = pd.concat(frames, ignore_index=True)
    cfg = load_config(str(A1_CONFIG))
    cfg["training"]["device"] = str(device)
    pred = _predict_on_frame(cfg=cfg, checkpoint=A1_CHECKPOINT, frame=frame_all, pred_key="pred_pos")
    return _stitch_flight_prediction(frame, pred)


def _real_base(frame: pd.DataFrame) -> pd.DataFrame:
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


def _merge(base: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    p = pred[["minute_ts", "pred_lat", "pred_lon", "pred_alt"]].copy()
    p["minute_ts"] = pd.to_datetime(p["minute_ts"], utc=True)
    p = p.rename(
        columns={
            "pred_lat": "A1_pred_lat",
            "pred_lon": "A1_pred_lon",
            "pred_alt": "A1_pred_alt_m",
        }
    )
    return base.merge(p, on="minute_ts", how="left")


def _plot(pair_id: str, table: pd.DataFrame, out_png: Path) -> None:
    x = pd.to_numeric(table["rel_min"], errors="coerce")
    fig, ax = plt.subplots(figsize=(12.8, 5.8), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(x, table["adsb_alt_m"], color="#111111", lw=2.0, label="ADS-B")
    ax.scatter(
        x,
        table["adsc_anchor_alt_m"],
        color="#000000",
        marker="*",
        s=95,
        zorder=7,
        label="ADS-C anchors",
    )
    ax.plot(x, table["A1_pred_alt_m"], color="#1f77b4", linestyle="--", linewidth=2.0, alpha=0.96, label="A1 linear")
    ax.set_title(f"{pair_id} | A1 recovery on real ADS-C window")
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
    out_root = ROOT / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for pair_id, (start_min, end_min) in WINDOWS.items():
        frame_csv = base_dir / pair_id / "input_recovery_frame.csv"
        if not frame_csv.exists():
            raise RuntimeError(f"Missing input frame for {pair_id}: {frame_csv}")
        print(f"[case] {pair_id}", flush=True)
        frame = pd.read_csv(frame_csv, parse_dates=["minute_ts"])
        pred = _predict_a1(frame, args.device)
        merged = _merge(_real_base(frame), pred)
        case_dir = out_root / pair_id
        case_dir.mkdir(parents=True, exist_ok=True)
        merged.to_csv(case_dir / "recovered_minute_compare_a1.csv", index=False, encoding="utf-8-sig")
        window = merged[(merged["rel_min"] >= start_min) & (merged["rel_min"] <= end_min)].copy()
        table_csv = case_dir / f"{pair_id}_{start_min}_{end_min}_a1_table.csv"
        plot_png = case_dir / f"{pair_id}_{start_min}_{end_min}_a1_plot.png"
        window.to_csv(table_csv, index=False, encoding="utf-8-sig")
        _plot(pair_id, window, plot_png)
        rows.append(
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

    pd.DataFrame(rows).to_csv(out_root / "summary.csv", index=False, encoding="utf-8-sig")
    print(f"[done] summary={out_root / 'summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
