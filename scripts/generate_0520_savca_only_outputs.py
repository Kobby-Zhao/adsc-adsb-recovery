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
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_bidirectional_prediction_mechanism import _run_model_for_samples  # noqa: E402
from scripts.evaluate_complete_adsb_sparse_cruise import (  # noqa: E402
    _build_sparse_cruise_dataset,
    _collect_predictions,
)
from scripts.real_adsc_replay_eval import _predict_on_frame  # noqa: E402
from scripts.real_adsc_truequal_gapwise_eval import _build_gapwise_segments, _stitch_flight_prediction  # noqa: E402
from scripts.train import load_config  # noqa: E402


MODEL_SPECS = {
    "A1-linear": {
        "config": "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_height_ablation_gated_cpu_v1/configs/a1_linear_alt_baseline.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_height_ablation_gated_cpu_v1/a1_linear_alt_baseline/best.pt",
    },
    "A3-gated": {
        "config": "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_height_ablation_gated_cpu_v1/configs/a3_gated_routed.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_height_ablation_gated_cpu_v1/a3_gated_routed/best.pt",
    },
    "SAVCA-only": {
        "config": "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_savca_only_v1/configs/savca_only.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_savca_only_v1/savca_only/best.pt",
    },
    "SAVCA-supervised": {
        "config": "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_savca_supervised_fixlabel_v1/configs/savca_supervised.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_savca_supervised_fixlabel_v1/savca_supervised/best.pt",
    },
}

WINDOWS = {
    "39d2a8_0013": (150, 560),
    "407fcd_0019": (150, 530),
    "4076e8_0021": (150, 540),
    "a9c5c2_0001": (70, 350),
    "407943_0020": (50, 500),
}


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _write_eval_configs(samples_path: Path, out_dir: Path, device: str) -> dict[str, dict[str, str]]:
    cfg_dir = out_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    specs: dict[str, dict[str, str]] = {}
    for model_name, spec in MODEL_SPECS.items():
        cfg = load_config(str(_resolve(spec["config"])))
        cfg["data"]["samples_path"] = str(samples_path)
        cfg["data"]["split"] = {"train_ratio": 0.0, "val_ratio": 0.0, "test_ratio": 1.0}
        cfg["training"]["batch_size"] = 16
        cfg["training"]["device"] = str(device)
        out_cfg = cfg_dir / f"{model_name.replace('+', 'plus').replace(' ', '_')}.yaml"
        with out_cfg.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        specs[model_name] = {"config": str(out_cfg), "checkpoint": spec["checkpoint"]}
    return specs


def _plot_adsb_cases(pred_df: pd.DataFrame, out_dir: Path) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    styles = [
        ("A1-linear_alt_m", "A1 linear", "#1f77b4", "--", 1.7),
        ("A3-gated_alt_m", "A3 gated", "#d62728", "-", 2.0),
        ("SAVCA-only_alt_m", "SAVCA-only", "#2ca02c", "-", 2.4),
        ("SAVCA-supervised_alt_m", "SAVCA supervised", "#9467bd", "-", 2.4),
    ]
    for (source_case, anchor_count), g in pred_df.groupby(["source_case", "anchor_count"], sort=False):
        if int(anchor_count) not in {3, 8}:
            continue
        g = g.sort_values("minute_index")
        x = g["minute_index"].to_numpy()
        obs = g["obs_mask"].to_numpy(dtype=bool)
        fig, ax = plt.subplots(figsize=(12.6, 5.4), facecolor="white")
        ax.plot(x, g["adsb_alt_m"], color="#111111", lw=2.2, label="ADS-B truth")
        ax.scatter(x[obs], g.loc[obs, "adsb_alt_m"], color="#111111", marker="*", s=125, zorder=7, label="ADS-C-like anchors")
        for col, label, color, ls, lw in styles:
            if col in g.columns:
                ax.plot(x, g[col], lw=lw, color=color, linestyle=ls, alpha=0.95, label=label)
        ax.set_title(f"{source_case} | anchor_count={anchor_count}")
        ax.set_xlabel("Cruise window minute")
        ax.set_ylabel("Altitude (m)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9, ncol=3)
        fig.tight_layout()
        safe = str(source_case).replace("/", "_")
        fig.savefig(plot_dir / f"{safe}_anchor{anchor_count}_savca_only_compare.png", dpi=180)
        plt.close(fig)


def run_adsb_sparse(args: argparse.Namespace, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    anchor_counts = [int(x) for x in args.anchor_counts.split(",") if x.strip()]
    frame = _build_sparse_cruise_dataset(
        _resolve(args.adsb_input_dir),
        out_dir,
        anchor_counts,
        int(args.max_window),
        float(args.min_alt),
    )
    samples_path = out_dir / "sparse_cruise_samples.parquet"
    specs = _write_eval_configs(samples_path, out_dir, args.device)
    selected_ids = set(frame["sample_id"].astype(str).unique())
    model_results: dict[str, dict] = {}
    for model_name, spec in specs.items():
        print(f"[adsb model] {model_name}", flush=True)
        model_results[model_name] = _run_model_for_samples(
            model_name,
            model_specs={model_name: spec},
            selected_ids=selected_ids,
            split_name="test",
            device=torch.device(args.device),
        )
    metrics, predictions = _collect_predictions(frame, model_results)
    metrics.to_csv(out_dir / "savca_only_adsb_metrics_by_sample.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(out_dir / "savca_only_adsb_predictions.csv", index=False, encoding="utf-8-sig")
    by_model = (
        metrics.groupby("model", as_index=False)
        .agg(
            sample_count=("sample_id", "nunique"),
            alt_RMSE_m=("alt_RMSE_m", "mean"),
            alt_MAE_m=("alt_MAE_m", "mean"),
            lat_RMSE=("lat_RMSE", "mean"),
            lon_RMSE=("lon_RMSE", "mean"),
        )
        .sort_values(["alt_RMSE_m", "model"])
    )
    by_anchor = (
        metrics.groupby(["model", "anchor_count"], as_index=False)
        .agg(sample_count=("sample_id", "nunique"), alt_RMSE_m=("alt_RMSE_m", "mean"), alt_MAE_m=("alt_MAE_m", "mean"))
        .sort_values(["anchor_count", "alt_RMSE_m", "model"])
    )
    by_model.to_csv(out_dir / "savca_only_adsb_metrics_by_model.csv", index=False, encoding="utf-8-sig")
    by_anchor.to_csv(out_dir / "savca_only_adsb_metrics_by_model_anchor_count.csv", index=False, encoding="utf-8-sig")
    _plot_adsb_cases(predictions, out_dir)
    print("[adsb summary]")
    print(by_model.round(3).to_string(index=False), flush=True)


def _predict_real_models(frame: pd.DataFrame, device: str) -> dict[str, pd.DataFrame]:
    frames, _ = _build_gapwise_segments(frame)
    if not frames:
        raise RuntimeError("No gapwise segments constructed.")
    frame_all = pd.concat(frames, ignore_index=True)
    out: dict[str, pd.DataFrame] = {}
    for name, spec in MODEL_SPECS.items():
        print(f"[adsc model] {name}", flush=True)
        cfg = load_config(str(_resolve(spec["config"])))
        cfg["training"]["device"] = str(device)
        pred = _predict_on_frame(cfg=cfg, checkpoint=_resolve(spec["checkpoint"]), frame=frame_all, pred_key="pred_pos")
        out[name] = _stitch_flight_prediction(frame, pred)
    return out


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


def _real_table(base: pd.DataFrame, preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = base.copy()
    for name, pred in preds.items():
        p = pred[["minute_ts", "pred_lat", "pred_lon", "pred_alt"]].copy()
        p["minute_ts"] = pd.to_datetime(p["minute_ts"], utc=True)
        p = p.rename(columns={"pred_lat": f"{name}_pred_lat", "pred_lon": f"{name}_pred_lon", "pred_alt": f"{name}_pred_alt_m"})
        out = out.merge(p, on="minute_ts", how="left")
    return out


def _plot_real(pair_id: str, table: pd.DataFrame, out_png: Path) -> None:
    x = pd.to_numeric(table["rel_min"], errors="coerce")
    fig, ax = plt.subplots(figsize=(12.8, 5.8), facecolor="white")
    ax.plot(x, table["adsb_alt_m"], color="#111111", lw=2.0, label="ADS-B")
    ax.scatter(x, table["adsc_anchor_alt_m"], color="#000000", marker="*", s=95, zorder=7, label="ADS-C anchors")
    styles = [
        ("A1-linear_pred_alt_m", "A1 linear", "#1f77b4", "--", 1.7),
        ("A3-gated_pred_alt_m", "A3 gated", "#d62728", "-", 2.0),
        ("SAVCA-only_pred_alt_m", "SAVCA-only", "#2ca02c", "-", 2.4),
        ("SAVCA-supervised_pred_alt_m", "SAVCA supervised", "#9467bd", "-", 2.4),
    ]
    for col, label, color, ls, lw in styles:
        if col in table:
            ax.plot(x, table[col], color=color, linestyle=ls, linewidth=lw, alpha=0.95, label=label)
    ax.set_title(f"{pair_id} | real ADS-C selected window")
    ax.set_xlabel("Minutes from recovery frame start")
    ax.set_ylabel("Altitude (m)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, ncol=3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_real_adsc(args: argparse.Namespace, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_dir = _resolve(args.real_base_dir)
    rows = []
    for pair_id, (start_min, end_min) in WINDOWS.items():
        print(f"[real case] {pair_id}", flush=True)
        frame_csv = base_dir / pair_id / "input_recovery_frame.csv"
        frame = pd.read_csv(frame_csv, parse_dates=["minute_ts"])
        preds = _predict_real_models(frame, args.device)
        merged = _real_table(_real_base(frame), preds)
        case_dir = out_dir / pair_id
        case_dir.mkdir(parents=True, exist_ok=True)
        merged.to_csv(case_dir / "recovered_minute_compare_savca_only.csv", index=False, encoding="utf-8-sig")
        window = merged[(merged["rel_min"] >= start_min) & (merged["rel_min"] <= end_min)].copy()
        table_csv = case_dir / f"{pair_id}_{start_min}_{end_min}_savca_only_table.csv"
        plot_png = case_dir / f"{pair_id}_{start_min}_{end_min}_savca_only_plot.png"
        window.to_csv(table_csv, index=False, encoding="utf-8-sig")
        _plot_real(pair_id, window, plot_png)
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
    pd.DataFrame(rows).to_csv(out_dir / "savca_only_real_adsc_summary.csv", index=False, encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="outputs/runs/0520/savca_only")
    parser.add_argument("--adsb-input-dir", default="outputs/runs/complete_adsb_height_pattern_references_20260519_final")
    parser.add_argument("--real-base-dir", default="outputs/runs/paper_showcase_cross_ocean_all_model_compare_20260518")
    parser.add_argument("--anchor-counts", default="3,4,5,6,7,8")
    parser.add_argument("--max-window", type=int, default=180)
    parser.add_argument("--min-alt", type=float, default=8000.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    out_root = _resolve(args.out_root)
    run_adsb_sparse(args, out_root / "adsb_sparse_cruise")
    run_real_adsc(args, out_root / "real_adsc_windows")
    print(f"[done] {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
