from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from scripts.analyze_bidirectional_prediction_mechanism import (  # noqa: E402
    _prepare_dataset,
    _run_model_for_samples,
)
from src.training.utils import load_config, set_seed  # noqa: E402


STYLE = {
    "ADS-B true": {"color": "black", "linestyle": "-", "linewidth": 2.5},
    "ADS-C anchors": {"color": "black", "marker": "*", "markersize": 10, "linestyle": "None"},
    "BiMamba": {"color": "red", "linestyle": "-", "linewidth": 2.2},
    "UniLSTM-proto": {"color": "tab:blue", "linestyle": "--", "linewidth": 1.6},
    "BiLSTM-proto": {"color": "tab:green", "linestyle": "-.", "linewidth": 1.6},
    "CNN-LSTM-proto": {"color": "tab:purple", "linestyle": "--", "linewidth": 1.6},
    "Transformer-proto": {"color": "tab:orange", "linestyle": ":", "linewidth": 1.8},
}

MODEL_SPECS = {
    "BiMamba": {
        "config": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_bimamba.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_bimamba/best.pt",
        "per_sample": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_bimamba/main_task_metrics_test_per_sample.csv",
    },
    "UniLSTM-proto": {
        "config": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_unilstm_proto.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_unilstm_proto/best.pt",
        "per_sample": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_unilstm_proto/main_task_metrics_test_per_sample.csv",
    },
    "BiLSTM-proto": {
        "config": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_bilstm_proto.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_bilstm_proto/best.pt",
        "per_sample": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_bilstm_proto/main_task_metrics_test_per_sample.csv",
    },
    "CNN-LSTM-proto": {
        "config": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_cnnlstm_proto.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_cnnlstm_proto/best.pt",
        "per_sample": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_cnnlstm_proto/main_task_metrics_test_per_sample.csv",
    },
    "Transformer-proto": {
        "config": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_transformer_proto.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_transformer_proto/best.pt",
        "per_sample": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_transformer_proto/main_task_metrics_test_per_sample.csv",
    },
}


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "SimSun", "DejaVu Serif"],
            "axes.unicode_minus": False,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
        }
    )


def _load_selected(selected_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(selected_csv)
    need = {"scenario", "sample_id", "flight_id"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"selected sample csv missing columns: {sorted(miss)}")
    return df


def _true_series_for_selected(ds, selected_ids: set[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for sample in ds.samples:
        sid = str(sample["sample_id"])
        if sid not in selected_ids:
            continue
        obs = sample["obs_mask"].numpy()
        target = sample["target_pos"].numpy()
        idx = np.arange(len(obs))
        out[sid] = {
            "obs_mask": obs,
            "minute_index": idx,
            "target": target,
            "flight_id": str(sample["flight_id"]),
            "sample_id": sid,
        }
    return out


def _main_gap_window(obs_mask: np.ndarray) -> tuple[int, int]:
    best_s = 0
    best_e = 0
    best_n = -1
    t = 0
    while t < len(obs_mask):
        if obs_mask[t] > 0.5:
            t += 1
            continue
        s = t
        while t < len(obs_mask) and obs_mask[t] <= 0.5:
            t += 1
        e = t
        if e - s > best_n:
            best_s, best_e, best_n = s, e, e - s
    return best_s, best_e


def _build_metric_lookup() -> dict[str, dict[str, dict[str, float]]]:
    lookup: dict[str, dict[str, dict[str, float]]] = {}
    for label, spec in MODEL_SPECS.items():
        df = pd.read_csv(_resolve(spec["per_sample"]))
        model_map: dict[str, dict[str, float]] = {}
        for _, row in df.iterrows():
            sid = str(row["sample_id"])
            model_map[sid] = {
                "gap_alt_rmse": float(row["gap_alt_rmse"]),
                "gap_alt_mae": float(row["gap_alt_mae"]),
                "gap_lat_rmse": float(row["gap_lat_rmse"]),
                "gap_lon_rmse": float(row["gap_lon_rmse"]),
            }
        lookup[label] = model_map
    return lookup


def _collect_predictions(selected_ids: set[str], split: str, device: torch.device) -> dict[str, dict[str, dict]]:
    outputs: dict[str, dict[str, dict]] = {}
    for label in MODEL_SPECS:
        outputs[label] = _run_model_for_samples(label, MODEL_SPECS, selected_ids, split, device)
    return outputs


def _annotate_jump(ax, x: np.ndarray, series: np.ndarray, obs_mask: np.ndarray, label: str, color: str) -> None:
    gap_idx = np.where(obs_mask <= 0.5)[0]
    if len(gap_idx) == 0:
        return
    last_gap = int(gap_idx[-1])
    if last_gap + 1 >= len(series):
        return
    jump = float(series[last_gap + 1] - series[last_gap])
    if abs(jump) <= 100.0:
        return
    ax.annotate(
        f"{label} jump {jump:.0f}m",
        xy=(x[last_gap + 1], series[last_gap + 1]),
        xytext=(x[last_gap + 1] - 0.18 * len(x), series[last_gap + 1] + np.sign(jump) * 0.08 * (series.max() - series.min() + 1.0)),
        arrowprops={"arrowstyle": "->", "color": color, "lw": 1.0},
        fontsize=8,
        color=color,
    )


def _plot_altitude_compare(out_dir: Path, sample_id: str, scenario: str, info: dict, preds: dict[str, dict], metrics: dict[str, dict[str, float]]) -> None:
    x = info["minute_index"]
    truth = info["target"][:, 2]
    obs_mask = info["obs_mask"]
    anchor = obs_mask > 0.5
    gap = ~anchor
    gap_s, gap_e = _main_gap_window(obs_mask)

    fig, ax = plt.subplots(figsize=(12.8, 4.8))
    ax.axvspan(gap_s, gap_e - 1, color="#d9d9d9", alpha=0.45, zorder=0, label="Gap")
    ax.plot(x, truth, label="ADS-B true", zorder=6, **STYLE["ADS-B true"])
    ax.plot(x[anchor], truth[anchor], label="ADS-C anchors", zorder=7, **STYLE["ADS-C anchors"])

    for label in ["BiMamba", "UniLSTM-proto", "BiLSTM-proto", "CNN-LSTM-proto", "Transformer-proto"]:
        series = preds[label]["final"][:, 2]
        m = metrics[label]
        legend_label = f"{label} (RMSE={m['gap_alt_rmse']:.2f}, MAE={m['gap_alt_mae']:.2f})"
        line = dict(STYLE[label])
        if label != "BiMamba":
            line["alpha"] = 0.9
        ax.plot(x, series, label=legend_label, zorder=5 if label == "BiMamba" else 3, **line)
        _annotate_jump(ax, x, series, obs_mask, label, STYLE[label]["color"])

    ax.set_title(f"{scenario}: altitude recovery comparison")
    ax.set_xlabel("Minute index")
    ax.set_ylabel("Altitude (m)")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / f"altitude_compare_{sample_id}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"altitude_compare_{sample_id}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_altitude_zoom(out_dir: Path, sample_id: str, scenario: str, info: dict, preds: dict[str, dict], metrics: dict[str, dict[str, float]]) -> None:
    x = info["minute_index"]
    truth = info["target"][:, 2]
    obs_mask = info["obs_mask"]
    anchor = obs_mask > 0.5
    gap_s, gap_e = _main_gap_window(obs_mask)
    pad = max(3, int(0.08 * len(x)))
    zs = max(0, gap_s - pad)
    ze = min(len(x), gap_e + pad)
    zx = x[zs:ze]
    ztruth = truth[zs:ze]
    zanchor = anchor[zs:ze]

    fig, ax = plt.subplots(figsize=(12.8, 4.8))
    ax.axvspan(gap_s, gap_e - 1, color="#d9d9d9", alpha=0.45, zorder=0)
    ax.plot(zx, ztruth, label="ADS-B true", zorder=6, **STYLE["ADS-B true"])
    ax.plot(zx[zanchor], ztruth[zanchor], label="ADS-C anchors", zorder=7, **STYLE["ADS-C anchors"])

    best_label = min(
        ["BiMamba", "UniLSTM-proto", "BiLSTM-proto", "CNN-LSTM-proto", "Transformer-proto"],
        key=lambda name: metrics[name]["gap_alt_rmse"],
    )
    for label in ["BiMamba", "UniLSTM-proto", "BiLSTM-proto", "CNN-LSTM-proto", "Transformer-proto"]:
        series = preds[label]["final"][:, 2][zs:ze]
        line = dict(STYLE[label])
        if label != "BiMamba":
            line["alpha"] = 0.75
        ax.plot(zx, series, label=label, zorder=5 if label == "BiMamba" else 3, **line)

    if best_label == "BiMamba":
        ax.text(
            0.02,
            0.94,
            "BiMamba follows the ADS-B altitude trend more closely",
            transform=ax.transAxes,
            fontsize=9,
            color=STYLE["BiMamba"]["color"],
            ha="left",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

    ax.set_title(f"{scenario}: zoomed altitude recovery")
    ax.set_xlabel("Minute index")
    ax.set_ylabel("Altitude (m)")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out_dir / f"altitude_zoom_{sample_id}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"altitude_zoom_{sample_id}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_latlon_compare(out_dir: Path, sample_id: str, scenario: str, info: dict, preds: dict[str, dict], metrics: dict[str, dict[str, float]]) -> None:
    truth = info["target"]
    obs_mask = info["obs_mask"] > 0.5
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.plot(truth[:, 1], truth[:, 0], label="ADS-B true", zorder=6, **STYLE["ADS-B true"])
    ax.plot(truth[obs_mask, 1], truth[obs_mask, 0], label="ADS-C anchors", zorder=7, **STYLE["ADS-C anchors"])

    for label in ["BiMamba", "UniLSTM-proto", "BiLSTM-proto", "CNN-LSTM-proto", "Transformer-proto"]:
        m = metrics[label]
        legend_label = f"{label} (lat={m['gap_lat_rmse']:.3f}, lon={m['gap_lon_rmse']:.3f})"
        line = dict(STYLE[label])
        if label != "BiMamba":
            line["alpha"] = 0.8
        ax.plot(preds[label]["final"][:, 1], preds[label]["final"][:, 0], label=legend_label, zorder=5 if label == "BiMamba" else 3, **line)

    ax.set_title(f"{scenario}: lat/lon recovery comparison")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out_dir / f"latlon_compare_{sample_id}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"latlon_compare_{sample_id}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument(
        "--selected-csv",
        default="outputs/experiments/obs_conditioned_gaponly/typical_recovery_compare_20260531/selected_typical_samples.csv",
    )
    ap.add_argument(
        "--out-dir",
        default="outputs/experiments/obs_conditioned_gaponly/selected_model_recovery_compare_20260531",
    )
    args = ap.parse_args()

    _set_plot_style()
    set_seed(42)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for plotting BiMamba and torch.cuda.is_available() is false.")
    device = torch.device(args.device)

    selected = _load_selected(_resolve(args.selected_csv))
    selected_ids = set(selected["sample_id"].astype(str))
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for spec in MODEL_SPECS.values():
        for field in ("config", "checkpoint", "per_sample"):
            if not _resolve(spec[field]).exists():
                raise FileNotFoundError(f"Missing {field}: {_resolve(spec[field])}")

    base_cfg = load_config(str(_resolve(MODEL_SPECS["BiMamba"]["config"])))
    ds = _prepare_dataset(base_cfg, split_name=args.split)
    truth_map = _true_series_for_selected(ds, selected_ids)
    metric_lookup = _build_metric_lookup()
    model_outputs = _collect_predictions(selected_ids, args.split, device)

    summary_rows: list[dict] = []
    for _, row in selected.iterrows():
        sid = str(row["sample_id"])
        sample_out_dir = out_dir / sid
        sample_out_dir.mkdir(parents=True, exist_ok=True)
        preds = {label: model_outputs[label][sid] for label in MODEL_SPECS}
        metrics = {label: metric_lookup[label][sid] for label in MODEL_SPECS}
        _plot_altitude_compare(sample_out_dir, sid, str(row["scenario"]), truth_map[sid], preds, metrics)
        _plot_altitude_zoom(sample_out_dir, sid, str(row["scenario"]), truth_map[sid], preds, metrics)
        _plot_latlon_compare(sample_out_dir, sid, str(row["scenario"]), truth_map[sid], preds, metrics)
        for label, m in metrics.items():
            summary_rows.append({"scenario": row["scenario"], "sample_id": sid, "model": label, **m})

    pd.DataFrame(summary_rows).to_csv(out_dir / "selected_model_compare_summary.csv", index=False)
    selected.to_csv(out_dir / "selected_typical_samples.csv", index=False)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
