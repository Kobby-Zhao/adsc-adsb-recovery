from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_bidirectional_prediction_mechanism import _run_model_for_samples  # noqa: E402
from scripts.evaluate_complete_adsb_sparse_cruise import (  # noqa: E402
    _collect_predictions,
    _linear_predictions,
    _metric_rows_for_prediction,
    _resolve,
)
from src.training.utils import load_config  # noqa: E402


MODEL_SPECS = {
    "A0-backbone": {
        "config": "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_height_ablation_gated_cpu_v1/configs/ours_backbone_absolute.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_height_ablation_gated_cpu_v1/ours_backbone_absolute/best.pt",
    },
    "A1-anchor-main": {
        "config": "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_height_ablation_gated_cpu_v1/configs/a1_linear_alt_baseline.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_height_ablation_gated_cpu_v1/a1_linear_alt_baseline/best.pt",
    },
    "A2-gated-offset": {
        "config": "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_height_ablation_gated_cpu_v1/configs/a2_gated_offset.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_height_ablation_gated_cpu_v1/a2_gated_offset/best.pt",
    },
    "A3-gated-routed": {
        "config": "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_height_ablation_gated_cpu_v1/configs/a3_gated_routed.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_height_ablation_gated_cpu_v1/a3_gated_routed/best.pt",
    },
}


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(value))[:180]


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
        out_cfg = cfg_dir / f"{_safe_name(model_name)}.yaml"
        with out_cfg.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        specs[model_name] = {"config": str(out_cfg), "checkpoint": spec["checkpoint"]}
    return specs


def _collect_gated_predictions(frame: pd.DataFrame, model_results: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict] = []
    pred_rows: list[dict] = []
    for sid, sample_df in frame.groupby("sample_id", sort=False):
        sample_df = sample_df.sort_values("minute_ts").reset_index(drop=True)
        truth = sample_df[["lat", "lon", "alt"]].to_numpy(dtype=float)
        obs = sample_df["obs_mask"].to_numpy(dtype=float) > 0.5
        preds = {"分段线性插值": _linear_predictions(sample_df)}
        for model_name, res in model_results.items():
            if sid in res:
                preds[model_name] = res[sid]["final"]
        for model_name, pred in preds.items():
            metric_rows.append(_metric_rows_for_prediction(sample_df, pred, model_name))
        for i in range(len(sample_df)):
            row = {
                "sample_id": sid,
                "source_case": sample_df["source_case"].iloc[i],
                "source_flight_id": sample_df["source_flight_id"].iloc[i],
                "anchor_count": int(sample_df["obs_mask"].sum()),
                "minute_index": int(i),
                "minute_ts": sample_df["minute_ts"].iloc[i],
                "obs_mask": int(obs[i]),
                "adsb_lat": float(truth[i, 0]),
                "adsb_lon": float(truth[i, 1]),
                "adsb_alt_m": float(truth[i, 2]),
                "adsc_anchor_alt_m": float(truth[i, 2]) if obs[i] else float("nan"),
            }
            for model_name, pred in preds.items():
                safe = _safe_name(model_name)
                row[f"{safe}_alt_m"] = float(pred[i, 2])
                row[f"{safe}_alt_abs_err_m"] = 0.0 if obs[i] else abs(float(pred[i, 2]) - float(truth[i, 2]))
            pred_rows.append(row)
    return pd.DataFrame(metric_rows), pd.DataFrame(pred_rows)


def _plot_cases(pred_df: pd.DataFrame, out_dir: Path, anchor_counts_to_plot: set[int]) -> None:
    plot_dir = out_dir / "plots"
    err_dir = out_dir / "error_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    err_dir.mkdir(parents=True, exist_ok=True)
    model_cols = [
        ("A0-backbone_alt_m", "A0 backbone", "#7f7f7f", "--", 1.5),
        ("A1-anchor-main_alt_m", "A1 anchor-main", "#1f77b4", "-", 1.9),
        ("A2-gated-offset_alt_m", "A2 gated-offset", "#ff7f0e", "-", 1.9),
        ("A3-gated-routed_alt_m", "A3 gated-routed", "#d62728", "-", 2.3),
        ("分段线性插值_alt_m", "Linear", "#2a9d8f", ":", 1.7),
    ]
    err_cols = [(c.replace("_alt_m", "_alt_abs_err_m"), label, color, ls, lw) for c, label, color, ls, lw in model_cols]
    for (source_case, anchor_count), g0 in pred_df.groupby(["source_case", "anchor_count"], sort=False):
        if int(anchor_count) not in anchor_counts_to_plot:
            continue
        g = g0.sort_values("minute_index")
        x = g["minute_index"].to_numpy(dtype=float)
        obs = g["obs_mask"].to_numpy(dtype=bool)
        fig, ax = plt.subplots(figsize=(12.8, 5.3), facecolor="white")
        ax.plot(x, g["adsb_alt_m"], color="black", lw=2.4, label="ADS-B truth")
        ax.scatter(x[obs], g.loc[obs, "adsb_alt_m"], color="black", marker="*", s=135, zorder=7, label="sparse anchors")
        for col, label, color, ls, lw in model_cols:
            if col in g.columns:
                ax.plot(x, g[col], color=color, linestyle=ls, lw=lw, alpha=0.93, label=label)
        ax.set_title(f"{source_case} | anchor_count={anchor_count}")
        ax.set_xlabel("Cruise window minute")
        ax.set_ylabel("Altitude (m)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=3)
        fig.tight_layout()
        name = f"{_safe_name(source_case)}_anchor{anchor_count}_gated_ablation_compare.png"
        fig.savefig(plot_dir / name, dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12.8, 4.6), facecolor="white")
        for col, label, color, ls, lw in err_cols:
            if col in g.columns:
                ax.plot(x, g[col], color=color, linestyle=ls, lw=lw, alpha=0.9, label=label)
        ax.scatter(x[obs], [0.0] * int(obs.sum()), color="black", marker="*", s=80, zorder=7, label="anchors")
        ax.set_title(f"{source_case} | anchor_count={anchor_count} | altitude absolute error")
        ax.set_xlabel("Cruise window minute")
        ax.set_ylabel("Absolute error (m)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=3)
        fig.tight_layout()
        fig.savefig(err_dir / name.replace("_compare.png", "_abs_error.png"), dpi=180)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-path", default="outputs/runs/complete_adsb_sparse_cruise_eval_20260519/sparse_cruise_samples.parquet")
    parser.add_argument("--out-dir", default="outputs/runs/complete_adsb_sparse_cruise_gated_height_ablation_20260520")
    parser.add_argument("--plot-anchor-counts", default="3,8")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    samples_path = _resolve(args.samples_path)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(samples_path)
    specs = _write_eval_configs(samples_path, out_dir, device=args.device)
    selected_ids = set(frame["sample_id"].astype(str).unique())
    model_results: dict[str, dict] = {}
    for model_name, spec in specs.items():
        model_results[model_name] = _run_model_for_samples(
            model_name,
            model_specs={model_name: spec},
            selected_ids=selected_ids,
            split_name="test",
            device=torch.device(args.device),
        )
    metrics, predictions = _collect_gated_predictions(frame, model_results)
    metrics.to_csv(out_dir / "sparse_cruise_gated_ablation_metrics_by_sample.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(out_dir / "sparse_cruise_gated_ablation_predictions.csv", index=False, encoding="utf-8-sig")
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
        .agg(
            sample_count=("sample_id", "nunique"),
            alt_RMSE_m=("alt_RMSE_m", "mean"),
            alt_MAE_m=("alt_MAE_m", "mean"),
        )
        .sort_values(["anchor_count", "alt_RMSE_m", "model"])
    )
    by_model.to_csv(out_dir / "sparse_cruise_gated_ablation_metrics_by_model.csv", index=False, encoding="utf-8-sig")
    by_anchor.to_csv(out_dir / "sparse_cruise_gated_ablation_metrics_by_anchor_count.csv", index=False, encoding="utf-8-sig")
    anchor_counts = {int(x) for x in args.plot_anchor_counts.split(",") if x.strip()}
    _plot_cases(predictions, out_dir, anchor_counts_to_plot=anchor_counts)
    print(f"[done] out_dir={out_dir}")
    print(by_model.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
