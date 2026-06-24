from __future__ import annotations

import argparse
import json
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
from src.training.utils import load_config  # noqa: E402


MODEL_SPECS = {
    "本文方案": {
        "config": "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_physical_time_ablation_v1/configs/a3_risk_routed.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_physical_time_ablation_v1/a3_risk_routed/best.pt",
    },
    "LSTM": {
        "config": "outputs/experiments/obs_conditioned_gaponly/clean_baselines_gaponly_v1/configs/lstm_clean_absolute.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/clean_baselines_gaponly_v1/lstm_clean_absolute/best.pt",
    },
    "BiLSTM": {
        "config": "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_physical_time_ablation_v1/configs/bilstm_clean_absolute.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_physical_time_ablation_v1/bilstm_clean_absolute/best.pt",
    },
    "CNN+LSTM": {
        "config": "outputs/experiments/obs_conditioned_gaponly/clean_baselines_gaponly_v1/configs/cnn_lstm_clean_absolute.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/clean_baselines_gaponly_v1/cnn_lstm_clean_absolute/best.pt",
    },
    "Transformer": {
        "config": "outputs/experiments/obs_conditioned_gaponly/clean_baselines_gaponly_v1/configs/transformer_clean_absolute.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/clean_baselines_gaponly_v1/transformer_clean_absolute/best.pt",
    },
    "Kalman Filter": {
        "config": "outputs/experiments/obs_conditioned_gaponly/clean_baselines_gaponly_v1/configs/kalman_filter_clean_absolute.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/clean_baselines_gaponly_v1/kalman_filter_clean_absolute/best.pt",
    },
}


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _longest_true_run(mask: np.ndarray) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    i = 0
    while i < len(mask):
        if not bool(mask[i]):
            i += 1
            continue
        s = i
        while i < len(mask) and bool(mask[i]):
            i += 1
        e = i
        if best is None or (e - s) > (best[1] - best[0]):
            best = (s, e)
    return best


def _angle_diff_deg(a: pd.Series) -> pd.Series:
    d = a.diff().fillna(0.0)
    return ((d + 180.0) % 360.0) - 180.0


def _anchor_indices(n: int, k: int) -> np.ndarray:
    idx = np.unique(np.rint(np.linspace(0, n - 1, k)).astype(int))
    # Very short windows can collapse duplicate indices. Fill deterministically if needed.
    if len(idx) < k:
        extra = [i for i in range(n) if i not in set(idx)]
        idx = np.array(sorted(list(idx) + extra[: k - len(idx)]), dtype=int)
    return idx


def _apply_anchor_features(frame: pd.DataFrame, anchor_idx: np.ndarray) -> pd.DataFrame:
    out = frame.copy().reset_index(drop=True)
    n = len(out)
    anchor = np.zeros(n, dtype=bool)
    anchor[anchor_idx] = True
    out["obs_mask"] = anchor.astype(float)
    out["is_anchor"] = anchor.astype(int)
    out["obs_lat"] = np.where(anchor, out["lat"], 0.0)
    out["obs_lon"] = np.where(anchor, out["lon"], 0.0)
    out["obs_alt"] = np.where(anchor, out["alt"], 0.0)

    dt_prev = np.zeros(n, dtype=float)
    dt_next = np.zeros(n, dtype=float)
    gap_len = np.zeros(n, dtype=float)
    gap_pos = np.zeros(n, dtype=float)
    anchors = np.where(anchor)[0]
    for left, right in zip(anchors[:-1], anchors[1:]):
        if right - left <= 1:
            continue
        length = right - left - 1
        for t in range(left + 1, right):
            dt_prev[t] = t - left
            dt_next[t] = right - t
            gap_len[t] = length
            gap_pos[t] = (t - left) / (right - left)
    out["dt_prev"] = dt_prev
    out["dt_next"] = dt_next
    out["gap_len"] = gap_len
    out["gap_pos_ratio"] = gap_pos
    return out


def _prepare_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["minute_ts"] = pd.to_datetime(out["minute_ts"], utc=True, errors="coerce")
    out = out.dropna(subset=["minute_ts", "lat", "lon", "alt"]).sort_values("minute_ts").reset_index(drop=True)
    out["vertical_speed"] = pd.to_numeric(out["alt"], errors="coerce").diff().fillna(0.0)
    out["speed_delta"] = pd.to_numeric(out.get("speed", 0.0), errors="coerce").fillna(0.0).diff().fillna(0.0)
    out["heading_delta"] = _angle_diff_deg(pd.to_numeric(out.get("heading", 0.0), errors="coerce").fillna(0.0))
    out["heading_rate"] = out["heading_delta"]
    out["turn_rate"] = out["heading_rate"].abs()
    out["is_cruise_candidate"] = 1
    out["is_cruise"] = 1
    out["local_speed_std"] = pd.to_numeric(out.get("speed", 0.0), errors="coerce").fillna(0.0).rolling(5, center=True, min_periods=1).std().fillna(0.0)
    out["local_heading_std"] = pd.to_numeric(out.get("heading", 0.0), errors="coerce").fillna(0.0).rolling(5, center=True, min_periods=1).std().fillna(0.0)
    out["local_alt_std"] = pd.to_numeric(out["alt"], errors="coerce").rolling(5, center=True, min_periods=1).std().fillna(0.0)
    out["jump_flag"] = (out["vertical_speed"].abs() > 300.0).astype(int)
    out["smooth_residual_proxy"] = out["vertical_speed"].diff().abs().fillna(0.0)
    return out


def _build_sparse_cruise_dataset(input_dir: Path, out_dir: Path, anchor_counts: list[int], max_window: int, min_alt: float) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    for csv_path in sorted(input_dir.glob("*/adsb_complete_minute.csv")):
        raw = pd.read_csv(csv_path)
        base = _prepare_base_features(raw)
        run = _longest_true_run((base["alt"].to_numpy(dtype=float) >= float(min_alt)) & np.isfinite(base["lat"]) & np.isfinite(base["lon"]))
        if run is None:
            continue
        s, e = run
        seg = base.iloc[s:e].copy().reset_index(drop=True)
        if len(seg) > max_window:
            # Use central cruise window to avoid climb/descent transition edges.
            start = max(0, (len(seg) - max_window) // 2)
            seg = seg.iloc[start : start + max_window].copy().reset_index(drop=True)
        if len(seg) < max(anchor_counts):
            continue
        source_flight = str(seg["flight_id"].iloc[0])
        source_case = csv_path.parent.name
        for k in anchor_counts:
            anchor_idx = _anchor_indices(len(seg), k)
            sample_id = f"{source_flight}__cruise_anchor{k}"
            flight_id = f"{source_flight}__anchor{k}"
            x = _apply_anchor_features(seg, anchor_idx)
            x["source_case"] = source_case
            x["source_flight_id"] = source_flight
            x["sample_id"] = sample_id
            x["flight_id"] = flight_id
            x["minute_ts"] = x["minute_ts"].astype(str)
            rows.append(x)
            gaps = np.diff(anchor_idx) - 1
            summary_rows.append(
                {
                    "sample_id": sample_id,
                    "flight_id": flight_id,
                    "source_case": source_case,
                    "source_flight_id": source_flight,
                    "anchor_count": k,
                    "window_len": int(len(seg)),
                    "gap_point_count": int(len(seg) - k),
                    "missing_rate": float((len(seg) - k) / len(seg)),
                    "mean_gap_len": float(np.mean(gaps)) if len(gaps) else 0.0,
                    "max_gap_len": int(np.max(gaps)) if len(gaps) else 0,
                    "alt_min_m": float(seg["alt"].min()),
                    "alt_max_m": float(seg["alt"].max()),
                    "alt_range_m": float(seg["alt"].max() - seg["alt"].min()),
                    "start_ts": str(seg["minute_ts"].iloc[0]),
                    "end_ts": str(seg["minute_ts"].iloc[-1]),
                }
            )
    if not rows:
        raise RuntimeError("No valid cruise windows found.")
    frame = pd.concat(rows, ignore_index=True)
    for col in frame.select_dtypes(include=["object"]).columns:
        frame[col] = frame[col].astype(str)
    summary = pd.DataFrame(summary_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_dir / "sparse_cruise_samples.parquet", index=False)
    frame.to_csv(out_dir / "sparse_cruise_samples.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "sparse_cruise_sample_summary.csv", index=False, encoding="utf-8-sig")
    return frame


def _write_eval_configs(samples_path: Path, out_dir: Path) -> dict[str, dict[str, str]]:
    cfg_dir = out_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    specs: dict[str, dict[str, str]] = {}
    for model_name, spec in MODEL_SPECS.items():
        cfg = load_config(str(_resolve(spec["config"])))
        cfg["data"]["samples_path"] = str(samples_path)
        cfg["data"]["split"] = {"train_ratio": 0.0, "val_ratio": 0.0, "test_ratio": 1.0}
        cfg["training"]["batch_size"] = 16
        cfg["training"]["device"] = "cpu"
        # Keep outputs.run_dir unchanged, because it stores the fitted feature scaler.
        out_cfg = cfg_dir / f"{model_name.replace('+', 'plus').replace(' ', '_')}.yaml"
        with out_cfg.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        specs[model_name] = {"config": str(out_cfg), "checkpoint": spec["checkpoint"]}
    return specs


def _linear_predictions(sample_df: pd.DataFrame) -> np.ndarray:
    n = len(sample_df)
    pred = sample_df[["lat", "lon", "alt"]].to_numpy(dtype=float).copy()
    obs = sample_df["obs_mask"].to_numpy(dtype=float) > 0.5
    anchors = np.where(obs)[0]
    for left, right in zip(anchors[:-1], anchors[1:]):
        for dim, col in enumerate(["lat", "lon", "alt"]):
            pred[left : right + 1, dim] = np.linspace(float(sample_df[col].iloc[left]), float(sample_df[col].iloc[right]), right - left + 1)
    return pred


def _metric_rows_for_prediction(sample_df: pd.DataFrame, pred: np.ndarray, model_name: str) -> dict:
    truth = sample_df[["lat", "lon", "alt"]].to_numpy(dtype=float)
    gap = sample_df["obs_mask"].to_numpy(dtype=float) <= 0.5
    err = pred - truth
    ge = err[gap]
    abs_ge = np.abs(ge)
    rmse = np.sqrt(np.mean(np.square(ge), axis=0))
    mae = np.mean(abs_ge, axis=0)
    return {
        "model": model_name,
        "sample_id": str(sample_df["sample_id"].iloc[0]),
        "flight_id": str(sample_df["flight_id"].iloc[0]),
        "source_case": str(sample_df["source_case"].iloc[0]),
        "source_flight_id": str(sample_df["source_flight_id"].iloc[0]),
        "anchor_count": int(sample_df["obs_mask"].sum()),
        "point_count": int(len(sample_df)),
        "gap_point_count": int(gap.sum()),
        "lat_RMSE": float(rmse[0]),
        "lon_RMSE": float(rmse[1]),
        "alt_RMSE_m": float(rmse[2]),
        "lat_MAE": float(mae[0]),
        "lon_MAE": float(mae[1]),
        "alt_MAE_m": float(mae[2]),
    }


def _collect_predictions(frame: pd.DataFrame, model_results: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
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
                row[f"{safe}_alt_abs_err_m"] = abs(float(pred[i, 2]) - float(truth[i, 2])) if not obs[i] else 0.0
            pred_rows.append(row)
    return pd.DataFrame(metric_rows), pd.DataFrame(pred_rows)


def _plot_cases(pred_df: pd.DataFrame, out_dir: Path, anchor_counts_to_plot: set[int]) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    model_cols = [
        ("本文方案_alt_m", "Ours-A3", "#023047", 2.3),
        ("分段线性插值_alt_m", "Linear", "#6c757d", 1.7),
        ("LSTM_alt_m", "LSTM", "#2a9d8f", 1.5),
        ("BiLSTM_alt_m", "BiLSTM", "#e76f51", 1.5),
        ("CNNplusLSTM_alt_m", "CNN+LSTM", "#9b5de5", 1.5),
        ("Transformer_alt_m", "Transformer", "#f4a261", 1.5),
        ("Kalman_Filter_alt_m", "Kalman Filter", "#457b9d", 1.5),
    ]
    for (source_case, anchor_count), g in pred_df.groupby(["source_case", "anchor_count"], sort=False):
        if int(anchor_count) not in anchor_counts_to_plot:
            continue
        g = g.sort_values("minute_index")
        x = g["minute_index"].to_numpy()
        obs = g["obs_mask"].to_numpy(dtype=bool)
        fig, ax = plt.subplots(figsize=(12.5, 5.2), facecolor="white")
        ax.plot(x, g["adsb_alt_m"], color="black", lw=2.3, label="ADS-B truth")
        ax.scatter(x[obs], g.loc[obs, "adsb_alt_m"], color="black", marker="*", s=130, zorder=6, label="ADS-C-like anchors")
        for col, label, color, lw in model_cols:
            if col in g.columns:
                ax.plot(x, g[col], lw=lw, color=color, label=label, alpha=0.9)
        ax.set_title(f"{source_case} | anchor_count={anchor_count}")
        ax.set_xlabel("Cruise window minute")
        ax.set_ylabel("Altitude (m)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=4)
        fig.tight_layout()
        safe = str(source_case).replace("/", "_")
        fig.savefig(plot_dir / f"{safe}_anchor{anchor_count}_altitude_compare.png", dpi=180)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs/runs/complete_adsb_height_pattern_references_20260519_final")
    parser.add_argument("--out-dir", default="outputs/runs/complete_adsb_sparse_cruise_eval_20260519")
    parser.add_argument("--anchor-counts", default="3,4,5,6,7,8")
    parser.add_argument("--max-window", type=int, default=180)
    parser.add_argument("--min-alt", type=float, default=8000.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out_dir = _resolve(args.out_dir)
    anchor_counts = [int(x) for x in args.anchor_counts.split(",") if x.strip()]
    frame = _build_sparse_cruise_dataset(_resolve(args.input_dir), out_dir, anchor_counts, args.max_window, args.min_alt)
    samples_path = out_dir / "sparse_cruise_samples.parquet"
    specs = _write_eval_configs(samples_path, out_dir)
    selected_ids = set(frame["sample_id"].astype(str).unique())

    model_results: dict[str, dict] = {}
    for model_name, spec in specs.items():
        try:
            model_results[model_name] = _run_model_for_samples(
                model_name,
                model_specs={model_name: spec},
                selected_ids=selected_ids,
                split_name="test",
                device=torch.device(args.device),
            )
        except Exception as exc:
            print(f"[warn] model failed: {model_name}: {exc}")

    metrics, predictions = _collect_predictions(frame, model_results)
    metrics.to_csv(out_dir / "sparse_cruise_model_metrics_by_sample.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(out_dir / "sparse_cruise_model_predictions.csv", index=False, encoding="utf-8-sig")
    by_model_anchor = (
        metrics.groupby(["model", "anchor_count"], as_index=False)
        .agg(
            sample_count=("sample_id", "nunique"),
            alt_RMSE_m=("alt_RMSE_m", "mean"),
            alt_MAE_m=("alt_MAE_m", "mean"),
            lat_RMSE=("lat_RMSE", "mean"),
            lon_RMSE=("lon_RMSE", "mean"),
        )
        .sort_values(["anchor_count", "alt_RMSE_m", "model"])
    )
    by_model = (
        metrics.groupby(["model"], as_index=False)
        .agg(
            sample_count=("sample_id", "nunique"),
            alt_RMSE_m=("alt_RMSE_m", "mean"),
            alt_MAE_m=("alt_MAE_m", "mean"),
            lat_RMSE=("lat_RMSE", "mean"),
            lon_RMSE=("lon_RMSE", "mean"),
        )
        .sort_values(["alt_RMSE_m", "model"])
    )
    by_model_anchor.to_csv(out_dir / "sparse_cruise_metrics_by_model_anchor_count.csv", index=False, encoding="utf-8-sig")
    by_model.to_csv(out_dir / "sparse_cruise_metrics_by_model.csv", index=False, encoding="utf-8-sig")
    _plot_cases(predictions, out_dir, anchor_counts_to_plot={3, 8})

    print(f"[done] out_dir={out_dir}")
    print("\n[overall by model]")
    print(by_model.round(3).to_string(index=False))
    print("\n[by model/anchor_count]")
    print(by_model_anchor.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
