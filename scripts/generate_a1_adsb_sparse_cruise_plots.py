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

from scripts.analyze_bidirectional_prediction_mechanism import _run_model_for_samples
from scripts.evaluate_complete_adsb_sparse_cruise import _build_sparse_cruise_dataset, _collect_predictions
from src.training.utils import load_config


A1_SPEC = {
    "A1-linear": {
        "config": "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_physical_time_ablation_v1/configs/a1_linear_alt_baseline.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_physical_time_ablation_v1/a1_linear_alt_baseline/best.pt",
    }
}


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Generate A1-only ADS-B sparse cruise recovery plots.")
    p.add_argument("--input-dir", default="outputs/runs/complete_adsb_height_pattern_references_20260519_final")
    p.add_argument("--out-dir", default="outputs/runs/0524/a1_adsb_sparse_cruise")
    p.add_argument("--anchor-counts", default="3,4,5,6,7,8")
    p.add_argument("--max-window", type=int, default=180)
    p.add_argument("--min-alt", type=float, default=8000.0)
    p.add_argument("--device", default="cpu")
    return p


def _write_eval_config(samples_path: Path, out_dir: Path, device: str) -> dict[str, str]:
    cfg_dir = out_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(_resolve(A1_SPEC["A1-linear"]["config"])))
    cfg["data"]["samples_path"] = str(samples_path)
    cfg["data"]["split"] = {"train_ratio": 0.0, "val_ratio": 0.0, "test_ratio": 1.0}
    cfg["training"]["batch_size"] = 16
    cfg["training"]["device"] = str(device)
    out_cfg = cfg_dir / "A1_linear.yaml"
    with out_cfg.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return {"config": str(out_cfg), "checkpoint": A1_SPEC["A1-linear"]["checkpoint"]}


def _plot_cases(pred_df: pd.DataFrame, out_dir: Path, anchor_counts_to_plot: set[int]) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for (source_case, anchor_count), g in pred_df.groupby(["source_case", "anchor_count"], sort=False):
        if int(anchor_count) not in anchor_counts_to_plot:
            continue
        g = g.sort_values("minute_index")
        x = g["minute_index"].to_numpy()
        obs = g["obs_mask"].to_numpy(dtype=bool)
        fig, ax = plt.subplots(figsize=(12.5, 5.2), facecolor="white")
        ax.set_facecolor("white")
        ax.plot(x, g["adsb_alt_m"], color="black", lw=2.3, label="ADS-B truth")
        ax.scatter(x[obs], g.loc[obs, "adsb_alt_m"], color="black", marker="*", s=130, zorder=6, label="ADS-C-like anchors")
        ax.plot(x, g["A1-linear_alt_m"], lw=2.0, color="#1f77b4", linestyle="--", alpha=0.95, label="A1 linear")
        ax.set_title(f"{source_case} | anchor_count={anchor_count} | A1")
        ax.set_xlabel("Cruise window minute")
        ax.set_ylabel("Altitude (m)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9, ncol=3)
        fig.tight_layout()
        safe = str(source_case).replace("/", "_")
        fig.savefig(plot_dir / f"{safe}_anchor{anchor_count}_a1_compare.png", dpi=180)
        plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    out_dir = _resolve(args.out_dir)
    anchor_counts = [int(x) for x in args.anchor_counts.split(",") if x.strip()]
    frame = _build_sparse_cruise_dataset(_resolve(args.input_dir), out_dir, anchor_counts, args.max_window, args.min_alt)
    samples_path = out_dir / "sparse_cruise_samples.parquet"
    spec = _write_eval_config(samples_path, out_dir, args.device)
    selected_ids = set(frame["sample_id"].astype(str).unique())

    model_results = _run_model_for_samples(
        "A1-linear",
        model_specs={"A1-linear": spec},
        selected_ids=selected_ids,
        split_name="test",
        device=torch.device(args.device),
    )

    metrics, predictions = _collect_predictions(frame, {"A1-linear": model_results})
    metrics.to_csv(out_dir / "a1_sparse_cruise_metrics_by_sample.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(out_dir / "a1_sparse_cruise_predictions.csv", index=False, encoding="utf-8-sig")

    by_anchor = (
        metrics.groupby(["anchor_count"], as_index=False)
        .agg(
            sample_count=("sample_id", "nunique"),
            alt_RMSE_m=("alt_RMSE_m", "mean"),
            alt_MAE_m=("alt_MAE_m", "mean"),
            lat_RMSE=("lat_RMSE", "mean"),
            lon_RMSE=("lon_RMSE", "mean"),
        )
        .sort_values(["anchor_count"])
    )
    by_anchor.to_csv(out_dir / "a1_sparse_cruise_metrics_by_anchor_count.csv", index=False, encoding="utf-8-sig")
    _plot_cases(predictions, out_dir, anchor_counts_to_plot={3, 8})

    print(f"[done] out_dir={out_dir}", flush=True)
    print(by_anchor.round(3).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
