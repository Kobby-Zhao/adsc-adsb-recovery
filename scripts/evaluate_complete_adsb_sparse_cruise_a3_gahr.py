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
    _collect_predictions,
    _linear_predictions,
    _metric_rows_for_prediction,
    _resolve,
)
from src.training.utils import load_config  # noqa: E402


MODEL_SPECS = {
    "旧A3": {
        "config": "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_physical_time_ablation_v1/configs/a3_risk_routed.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_physical_time_ablation_v1/a3_risk_routed/best.pt",
    },
    "A3-GAHR": {
        "config": "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_a3_gahr_v1/configs/a3_gahr_routed.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_a3_gahr_v1/a3_gahr_routed/best.pt",
    },
    "A3-GAHR-gated": {
        "config": "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_a3_gahr_gated_v1/configs/a3_gahr_gated_routed.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_a3_gahr_v1/a3_gahr_routed/best.pt",
    },
    "A3-GAHR-corrected": {
        "config": "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_a3_gahr_corrected_v1/configs/a3_gahr_corrected_routed.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_a3_gahr_corrected_v1/a3_gahr_corrected_routed/best.pt",
    },
}


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
        if model_name == "A3-GAHR-gated":
            # This evaluation reuses the trained A3-GAHR checkpoint and scaler,
            # while enabling deterministic residual gates from the trial config.
            cfg["outputs"]["run_dir"] = "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_a3_gahr_v1/a3_gahr_routed"
        out_cfg = cfg_dir / f"{model_name}.yaml"
        with out_cfg.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        specs[model_name] = {"config": str(out_cfg), "checkpoint": spec["checkpoint"]}
    return specs


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(value))[:180]


def _collect_a3_predictions(frame: pd.DataFrame, model_results: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict] = []
    pred_rows: list[dict] = []
    for sid, sample_df in frame.groupby("sample_id", sort=False):
        sample_df = sample_df.sort_values("minute_ts").reset_index(drop=True)
        truth = sample_df[["lat", "lon", "alt"]].to_numpy(dtype=float)
        obs = sample_df["obs_mask"].to_numpy(dtype=float) > 0.5
        minute = np.arange(len(sample_df))
        preds: dict[str, np.ndarray] = {"分段线性插值": _linear_predictions(sample_df)}
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
                "minute_index": int(minute[i]),
                "minute_ts": sample_df["minute_ts"].iloc[i],
                "obs_mask": int(obs[i]),
                "adsb_lat": float(truth[i, 0]),
                "adsb_lon": float(truth[i, 1]),
                "adsb_alt_m": float(truth[i, 2]),
                "adsc_anchor_alt_m": float(truth[i, 2]) if obs[i] else np.nan,
            }
            for model_name, pred in preds.items():
                safe = model_name.replace("+", "plus").replace(" ", "_")
                row[f"{safe}_alt_m"] = float(pred[i, 2])
                row[f"{safe}_alt_abs_err_m"] = 0.0 if obs[i] else abs(float(pred[i, 2]) - float(truth[i, 2]))
            pred_rows.append(row)
    return pd.DataFrame(metric_rows), pd.DataFrame(pred_rows)


def _gap_runs(obs_mask: np.ndarray) -> list[tuple[int, int]]:
    obs = np.asarray(obs_mask, dtype=int) == 1
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(obs):
        if obs[i]:
            i += 1
            continue
        s = i
        while i < len(obs) and not obs[i]:
            i += 1
        runs.append((s, i))
    return runs


def _diagnose_errors(pred_df: pd.DataFrame, out_dir: Path) -> None:
    rows: list[dict] = []
    for sid, g0 in pred_df.groupby("sample_id", sort=False):
        g = g0.sort_values("minute_index").reset_index(drop=True)
        truth = g["adsb_alt_m"].to_numpy(dtype=float)
        obs = g["obs_mask"].to_numpy(dtype=int)
        for s, e in _gap_runs(obs):
            left = s - 1
            right = e
            if left < 0 or right >= len(g):
                continue
            gap_len = e - s
            dz_anchor = float(truth[right] - truth[left])
            truth_gap = truth[s:e]
            linear_gap = g["分段线性插值_alt_m"].to_numpy(dtype=float)[s:e]
            old_gap = g["旧A3_alt_m"].to_numpy(dtype=float)[s:e] if "旧A3_alt_m" in g else np.full(gap_len, np.nan)
            gahr_gap = g["A3-GAHR_alt_m"].to_numpy(dtype=float)[s:e] if "A3-GAHR_alt_m" in g else np.full(gap_len, np.nan)
            gated_gap = (
                g["A3-GAHR-gated_alt_m"].to_numpy(dtype=float)[s:e]
                if "A3-GAHR-gated_alt_m" in g
                else np.full(gap_len, np.nan)
            )
            for model_name, pred_gap in [
                ("分段线性插值", linear_gap),
                ("旧A3", old_gap),
                ("A3-GAHR", gahr_gap),
                ("A3-GAHR-gated", gated_gap),
            ]:
                err = pred_gap - truth_gap
                ok = np.isfinite(err)
                if not ok.any():
                    continue
                rows.append(
                    {
                        "sample_id": sid,
                        "source_case": g["source_case"].iloc[0],
                        "anchor_count": int(g["anchor_count"].iloc[0]),
                        "gap_start_minute": int(g["minute_index"].iloc[s]),
                        "gap_end_minute": int(g["minute_index"].iloc[e - 1]),
                        "gap_len": int(gap_len),
                        "anchor_delta_alt_m": dz_anchor,
                        "truth_gap_alt_range_m": float(np.nanmax(truth_gap) - np.nanmin(truth_gap)),
                        "model": model_name,
                        "gap_RMSE_m": float(np.sqrt(np.mean(np.square(err[ok])))),
                        "gap_MAE_m": float(np.mean(np.abs(err[ok]))),
                        "gap_MaxAE_m": float(np.max(np.abs(err[ok]))),
                        "mean_signed_error_m": float(np.mean(err[ok])),
                    }
                )
    diag = pd.DataFrame(rows)
    diag.to_csv(out_dir / "a3_gahr_gap_error_diagnostics.csv", index=False, encoding="utf-8-sig")
    if diag.empty:
        return
    by_model = diag.groupby("model", as_index=False).agg(
        gap_count=("sample_id", "count"),
        gap_RMSE_mean_m=("gap_RMSE_m", "mean"),
        gap_MAE_mean_m=("gap_MAE_m", "mean"),
        gap_MaxAE_mean_m=("gap_MaxAE_m", "mean"),
    )
    by_model.to_csv(out_dir / "a3_gahr_gap_error_by_model.csv", index=False, encoding="utf-8-sig")
    diag["gap_len_bucket"] = pd.cut(
        diag["gap_len"],
        bins=[-1, 20, 40, 80, 10**9],
        labels=["<=20", "21-40", "41-80", ">80"],
    )
    diag["anchor_delta_bucket"] = pd.cut(
        diag["anchor_delta_alt_m"].abs(),
        bins=[-1, 60, 120, 300, 600, 10**9],
        labels=["<=60", "61-120", "121-300", "301-600", ">600"],
    )
    by_bucket = diag.groupby(["model", "gap_len_bucket", "anchor_delta_bucket"], observed=True, as_index=False).agg(
        gap_count=("sample_id", "count"),
        gap_RMSE_mean_m=("gap_RMSE_m", "mean"),
        gap_MAE_mean_m=("gap_MAE_m", "mean"),
        gap_MaxAE_mean_m=("gap_MaxAE_m", "mean"),
    )
    by_bucket.to_csv(out_dir / "a3_gahr_gap_error_by_len_and_delta_bucket.csv", index=False, encoding="utf-8-sig")
    wide = diag.pivot_table(
        index=["sample_id", "source_case", "anchor_count", "gap_start_minute", "gap_end_minute", "gap_len", "anchor_delta_alt_m"],
        columns="model",
        values="gap_RMSE_m",
        aggfunc="first",
    ).reset_index()
    if {"A3-GAHR", "旧A3"}.issubset(wide.columns):
        wide["gahr_minus_old_gap_RMSE_m"] = wide["A3-GAHR"] - wide["旧A3"]
        if "A3-GAHR-gated" in wide.columns:
            wide["gahr_gated_minus_old_gap_RMSE_m"] = wide["A3-GAHR-gated"] - wide["旧A3"]
            wide["gahr_gated_minus_gahr_gap_RMSE_m"] = wide["A3-GAHR-gated"] - wide["A3-GAHR"]
        wide.sort_values("gahr_minus_old_gap_RMSE_m", ascending=False).to_csv(
            out_dir / "a3_gahr_worst_gap_vs_old_a3.csv", index=False, encoding="utf-8-sig"
        )


def _plot_cases(pred_df: pd.DataFrame, out_dir: Path, anchor_counts_to_plot: set[int]) -> None:
    plot_dir = out_dir / "plots"
    err_dir = out_dir / "error_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    err_dir.mkdir(parents=True, exist_ok=True)
    for (source_case, anchor_count), g0 in pred_df.groupby(["source_case", "anchor_count"], sort=False):
        if int(anchor_count) not in anchor_counts_to_plot:
            continue
        g = g0.sort_values("minute_index")
        x = g["minute_index"].to_numpy(dtype=float)
        obs = g["obs_mask"].to_numpy(dtype=bool)
        fig, ax = plt.subplots(figsize=(12.8, 5.3), facecolor="white")
        ax.plot(x, g["adsb_alt_m"], color="black", lw=2.4, label="ADS-B truth")
        ax.scatter(x[obs], g.loc[obs, "adsb_alt_m"], color="black", marker="*", s=140, zorder=7, label="sparse anchors")
        for col, label, color, lw in [
            ("旧A3_alt_m", "Old A3", "#6b7280", 1.8),
            ("A3-GAHR_alt_m", "A3-GAHR", "#d00000", 2.2),
            ("A3-GAHR-gated_alt_m", "A3-GAHR-gated", "#1d4ed8", 2.0),
            ("A3-GAHR-corrected_alt_m", "A3-GAHR-corrected", "#0057b8", 2.3),
            ("分段线性插值_alt_m", "Linear", "#2a9d8f", 1.5),
        ]:
            if col in g:
                ax.plot(x, g[col], color=color, lw=lw, alpha=0.92, label=label)
        ax.set_title(f"{source_case} | anchor_count={anchor_count}")
        ax.set_xlabel("Cruise window minute")
        ax.set_ylabel("Altitude (m)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=4)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{_safe_name(source_case)}_anchor{anchor_count}_a3_gahr_compare.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12.8, 4.6), facecolor="white")
        for col, label, color in [
            ("旧A3_alt_abs_err_m", "Old A3 abs error", "#6b7280"),
            ("A3-GAHR_alt_abs_err_m", "A3-GAHR abs error", "#d00000"),
            ("A3-GAHR-gated_alt_abs_err_m", "A3-GAHR-gated abs error", "#1d4ed8"),
            ("A3-GAHR-corrected_alt_abs_err_m", "A3-GAHR-corrected abs error", "#0057b8"),
            ("分段线性插值_alt_abs_err_m", "Linear abs error", "#2a9d8f"),
        ]:
            if col in g:
                ax.plot(x, g[col], color=color, lw=1.8, alpha=0.9, label=label)
        ax.scatter(x[obs], np.zeros(obs.sum()), color="black", marker="*", s=80, zorder=7, label="anchors")
        ax.set_title(f"{source_case} | anchor_count={anchor_count} | altitude absolute error")
        ax.set_xlabel("Cruise window minute")
        ax.set_ylabel("Absolute error (m)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=4)
        fig.tight_layout()
        fig.savefig(err_dir / f"{_safe_name(source_case)}_anchor{anchor_count}_a3_gahr_error.png", dpi=180)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs/runs/complete_adsb_sparse_cruise_eval_20260519")
    parser.add_argument("--out-dir", default="outputs/runs/complete_adsb_sparse_cruise_a3_gahr_eval_20260520")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--plot-anchor-counts", default="3,8")
    args = parser.parse_args()

    input_dir = _resolve(args.input_dir)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = input_dir / "sparse_cruise_samples.parquet"
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
    metrics, predictions = _collect_a3_predictions(frame, model_results)
    metrics.to_csv(out_dir / "a3_gahr_sparse_cruise_metrics_by_sample.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(out_dir / "a3_gahr_sparse_cruise_predictions.csv", index=False, encoding="utf-8-sig")
    by_model = metrics.groupby("model", as_index=False).agg(
        sample_count=("sample_id", "nunique"),
        alt_RMSE_m=("alt_RMSE_m", "mean"),
        alt_MAE_m=("alt_MAE_m", "mean"),
        lat_RMSE=("lat_RMSE", "mean"),
        lon_RMSE=("lon_RMSE", "mean"),
    )
    by_anchor = metrics.groupby(["model", "anchor_count"], as_index=False).agg(
        sample_count=("sample_id", "nunique"),
        alt_RMSE_m=("alt_RMSE_m", "mean"),
        alt_MAE_m=("alt_MAE_m", "mean"),
    )
    by_model.to_csv(out_dir / "a3_gahr_sparse_cruise_metrics_by_model.csv", index=False, encoding="utf-8-sig")
    by_anchor.to_csv(out_dir / "a3_gahr_sparse_cruise_metrics_by_anchor_count.csv", index=False, encoding="utf-8-sig")
    _diagnose_errors(predictions, out_dir)
    plot_counts = {int(x) for x in args.plot_anchor_counts.split(",") if x.strip()}
    _plot_cases(predictions, out_dir, anchor_counts_to_plot=plot_counts)
    print(f"[done] {out_dir}")
    print("\n[by model]")
    print(by_model.round(3).to_string(index=False))
    print("\n[by anchor_count]")
    print(by_anchor.round(3).to_string(index=False))
    err_by_model = out_dir / "a3_gahr_gap_error_by_model.csv"
    if err_by_model.exists():
        print("\n[gap diagnostics by model]")
        print(pd.read_csv(err_by_model).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
