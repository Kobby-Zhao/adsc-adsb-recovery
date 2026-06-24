from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_bidirectional_prediction_mechanism import (  # noqa: E402
    _prepare_dataset,
    _run_model_for_samples,
)
from scripts.eval_interpolation_baselines import _interp_linear_gapwise  # noqa: E402
from scripts.eval_rts_kalman_baseline import _smooth_sample  # noqa: E402
from src.training.utils import load_config, set_seed  # noqa: E402


STYLE = {
    "ADS-B true": {"color": "black", "linestyle": "-", "linewidth": 2.5},
    "ADS-C anchors": {"color": "black", "marker": "o", "markersize": 5, "linestyle": "None"},
    "ACT-BiMamba": {"color": "red", "linestyle": "-", "linewidth": 2.4},
    "PiecewiseLinear": {"color": "#7f7f7f", "linestyle": "-", "linewidth": 1.8},
    "Kalman Filter": {"color": "#17becf", "linestyle": "-", "linewidth": 1.8},
    "Mamba": {"color": "#8c564b", "linestyle": "-", "linewidth": 1.8},
    "LSTM": {"color": "#1f77b4", "linestyle": "-", "linewidth": 1.8},
    "BiLSTM": {"color": "#2ca02c", "linestyle": "-", "linewidth": 1.8},
    "CNN+LSTM": {"color": "#9467bd", "linestyle": "-", "linewidth": 1.8},
    "Transformer": {"color": "#ff7f0e", "linestyle": "-", "linewidth": 1.8},
}

NEURAL_MODEL_SPECS = {
    "ACT-BiMamba": {
        "config": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_gapaware_small.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_gapaware_small/best.pt",
    },
    "Mamba": {
        "config": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_mamba_proto.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_mamba_proto/best.pt",
    },
    "LSTM": {
        "config": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_unilstm_proto.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_unilstm_proto/best.pt",
    },
    "BiLSTM": {
        "config": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_bilstm_proto.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_bilstm_proto/best.pt",
    },
    "CNN+LSTM": {
        "config": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_cnnlstm_proto.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_cnnlstm_proto/best.pt",
    },
    "Transformer": {
        "config": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_transformer_proto.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_transformer_proto/best.pt",
    },
}

MODEL_ORDER = [
    "ACT-BiMamba",
    "PiecewiseLinear",
    "Kalman Filter",
    "Mamba",
    "LSTM",
    "BiLSTM",
    "CNN+LSTM",
    "Transformer",
]


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.unicode_minus": False,
            "savefig.dpi": 300,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
        }
    )


def _load_selected(selected_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(selected_csv)
    if "sample_id" not in df.columns or "flight_id" not in df.columns:
        raise ValueError("selected sample csv must contain sample_id and flight_id")
    if "scenario" not in df.columns:
        if "bucket" in df.columns:
            df["scenario"] = df["bucket"].astype(str)
        else:
            df["scenario"] = "selected_adsb_case"
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


def _compute_gap_metrics(true_xyz: np.ndarray, pred_xyz: np.ndarray, obs_mask: np.ndarray) -> dict[str, float]:
    gap = obs_mask <= 0.5
    diff = pred_xyz[gap] - true_xyz[gap]
    if diff.size == 0:
        return {
            "gap_lat_rmse": float("nan"),
            "gap_lon_rmse": float("nan"),
            "gap_alt_rmse": float("nan"),
            "gap_lat_mae": float("nan"),
            "gap_lon_mae": float("nan"),
            "gap_alt_mae": float("nan"),
        }
    mae = np.mean(np.abs(diff), axis=0)
    rmse = np.sqrt(np.mean(diff**2, axis=0))
    return {
        "gap_lat_rmse": float(rmse[0]),
        "gap_lon_rmse": float(rmse[1]),
        "gap_alt_rmse": float(rmse[2]),
        "gap_lat_mae": float(mae[0]),
        "gap_lon_mae": float(mae[1]),
        "gap_alt_mae": float(mae[2]),
    }


def _collect_neural_predictions(selected_ids: set[str], split: str, device: torch.device) -> dict[str, dict[str, dict]]:
    outputs: dict[str, dict[str, dict]] = {}
    for label in NEURAL_MODEL_SPECS:
        outputs[label] = _run_model_for_samples(label, NEURAL_MODEL_SPECS, selected_ids, split, device)
    return outputs


def _sample_df_from_info(info: dict) -> pd.DataFrame:
    n = len(info["obs_mask"])
    minute_ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "minute_ts": minute_ts,
            "obs_mask": info["obs_mask"].astype(int),
            "lat": info["target"][:, 0],
            "lon": info["target"][:, 1],
            "alt": info["target"][:, 2],
        }
    )


def _collect_baseline_predictions(truth_map: dict[str, dict]) -> dict[str, dict[str, dict]]:
    outputs = {"PiecewiseLinear": {}, "Kalman Filter": {}}
    for sid, info in truth_map.items():
        sdf = _sample_df_from_info(info)
        lat_p, lon_p, alt_p = _interp_linear_gapwise(sdf)
        outputs["PiecewiseLinear"][sid] = {"final": np.stack([lat_p, lon_p, alt_p], axis=-1)}
        lat_k, lon_k, alt_k = _smooth_sample(sdf)
        outputs["Kalman Filter"][sid] = {"final": np.stack([lat_k, lon_k, alt_k], axis=-1)}
    return outputs


def _plot_altitude_compare(out_dir: Path, sample_id: str, scenario: str, info: dict, preds: dict[str, np.ndarray], metrics: dict[str, dict[str, float]]) -> None:
    x = info["minute_index"]
    truth = info["target"][:, 2]
    obs_mask = info["obs_mask"]
    anchor = obs_mask > 0.5
    gap_s, gap_e = _main_gap_window(obs_mask)

    fig, ax = plt.subplots(figsize=(13.2, 4.8))
    ax.axvspan(gap_s, gap_e - 1, color="#d9d9d9", alpha=0.45, zorder=0, label="Gap")
    ax.plot(x, truth, label="ADS-B true", zorder=9, **STYLE["ADS-B true"])
    ax.plot(x[anchor], truth[anchor], label="ADS-C anchors", zorder=10, **STYLE["ADS-C anchors"])

    for label in MODEL_ORDER:
        series = preds[label][:, 2]
        m = metrics[label]
        legend_label = f"{label} (RMSE={m['gap_alt_rmse']:.2f}, MAE={m['gap_alt_mae']:.2f})"
        ax.plot(x, series, label=legend_label, zorder=8 if label == "ACT-BiMamba" else 5, **STYLE[label])

    ax.set_title(f"{scenario}: altitude recovery comparison")
    ax.set_xlabel("Minute index")
    ax.set_ylabel("Altitude (m)")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / f"altitude_compare_{sample_id}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"altitude_compare_{sample_id}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_latlon_compare(out_dir: Path, sample_id: str, scenario: str, info: dict, preds: dict[str, np.ndarray], metrics: dict[str, dict[str, float]]) -> None:
    truth = info["target"]
    obs_mask = info["obs_mask"] > 0.5
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.plot(truth[:, 1], truth[:, 0], label="ADS-B true", zorder=9, **STYLE["ADS-B true"])
    ax.plot(truth[obs_mask, 1], truth[obs_mask, 0], label="ADS-C anchors", zorder=10, **STYLE["ADS-C anchors"])

    for label in MODEL_ORDER:
        m = metrics[label]
        legend_label = f"{label} (lat={m['gap_lat_rmse']:.3f}, lon={m['gap_lon_rmse']:.3f})"
        ax.plot(preds[label][:, 1], preds[label][:, 0], label=legend_label, zorder=8 if label == "ACT-BiMamba" else 5, **STYLE[label])

    ax.set_title(f"{scenario}: lat/lon recovery comparison")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=7.2)
    fig.tight_layout()
    fig.savefig(out_dir / f"latlon_compare_{sample_id}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"latlon_compare_{sample_id}.pdf", bbox_inches="tight")
    plt.close(fig)


def _build_figure(df: pd.DataFrame, sample_id: str, scenario: str) -> go.Figure:
    fig = go.Figure()
    anchors = df[df["is_adsc_anchor"].astype(int) == 1].copy()
    fig.add_trace(
        go.Scatter3d(
            x=anchors["truth_lon"],
            y=anchors["truth_lat"],
            z=anchors["truth_alt"],
            mode="markers",
            name="ADS-C anchors",
            marker=dict(size=3, color="black", symbol="circle"),
            hovertemplate="Anchor<br>Lon=%{x:.3f}<br>Lat=%{y:.3f}<br>Alt=%{z:.1f} m<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=df["truth_lon"],
            y=df["truth_lat"],
            z=df["truth_alt"],
            mode="lines",
            name="ADS-B true",
            line=dict(color="black", width=6),
            hovertemplate="ADS-B true<br>Lon=%{x:.3f}<br>Lat=%{y:.3f}<br>Alt=%{z:.1f} m<extra></extra>",
        )
    )
    for model in MODEL_ORDER:
        fig.add_trace(
            go.Scatter3d(
                x=df[f"{model}_pred_lon"],
                y=df[f"{model}_pred_lat"],
                z=df[f"{model}_pred_alt"],
                mode="lines",
                name=model,
                line=dict(color=STYLE[model]["color"], dash="solid", width=6 if model == "ACT-BiMamba" else 4),
                hovertemplate=f"{model}<br>Lon=%{{x:.3f}}<br>Lat=%{{y:.3f}}<br>Alt=%{{z:.1f}} m<extra></extra>",
            )
        )
    fig.update_layout(
        title=f"Interactive 3D Recovery Comparison: {scenario} / {sample_id}",
        template="plotly_white",
        height=920,
        margin=dict(l=10, r=10, t=56, b=10),
        legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.9)"),
        scene=dict(
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            zaxis_title="Altitude (m)",
            aspectmode="cube",
            xaxis=dict(showbackground=True, backgroundcolor="white", gridcolor="#d9d9d9", zerolinecolor="#d9d9d9"),
            yaxis=dict(showbackground=True, backgroundcolor="white", gridcolor="#d9d9d9", zerolinecolor="#d9d9d9"),
            zaxis=dict(showbackground=True, backgroundcolor="white", gridcolor="#d9d9d9", zerolinecolor="#d9d9d9", tickformat=".0f", exponentformat="none"),
            camera=dict(eye=dict(x=1.55, y=-1.65, z=0.8)),
        ),
    )
    return fig


def _write_index(case_rows: list[dict], out_dir: Path) -> None:
    items = []
    for row in case_rows:
        sid = row["sample_id"]
        items.append(f'<li><a href="{sid}/interactive_3d_{sid}.html">{row["scenario"]} / {sid}</a></li>')
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8" />
<title>Selected ADS-B Interactive 3D Recovery Compare</title>
<style>
body {{ font-family: "Times New Roman", serif; margin: 24px; background: #fff; color: #111; }}
a {{ color: #0b57d0; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
</style></head>
<body>
<h1>Selected ADS-B Interactive 3D Recovery Compare</h1>
<p>Click a sample below to open an interactive 3D trajectory page.</p>
<ul>{''.join(items)}</ul>
</body></html>"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


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
        default="outputs/experiments/obs_conditioned_gaponly/selected_recovery_compare_all_models_20260601",
    )
    args = ap.parse_args()

    _set_plot_style()
    set_seed(42)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for neural-model plotting and torch.cuda.is_available() is false.")
    device = torch.device(args.device)

    selected = _load_selected(_resolve(args.selected_csv))
    selected_ids = set(selected["sample_id"].astype(str))
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for spec in NEURAL_MODEL_SPECS.values():
        for field in ("config", "checkpoint"):
            if not _resolve(spec[field]).exists():
                raise FileNotFoundError(f"Missing {field}: {_resolve(spec[field])}")

    base_cfg = load_config(str(_resolve(NEURAL_MODEL_SPECS["ACT-BiMamba"]["config"])))
    ds = _prepare_dataset(base_cfg, split_name=args.split)
    truth_map = _true_series_for_selected(ds, selected_ids)
    neural_outputs = _collect_neural_predictions(selected_ids, args.split, device)
    baseline_outputs = _collect_baseline_predictions(truth_map)

    summary_rows: list[dict] = []
    index_rows: list[dict] = []
    for _, row in selected.iterrows():
        sid = str(row["sample_id"])
        sample_out_dir = out_dir / sid
        sample_out_dir.mkdir(parents=True, exist_ok=True)
        info = truth_map[sid]
        preds: dict[str, np.ndarray] = {}
        for label in NEURAL_MODEL_SPECS:
            preds[label] = neural_outputs[label][sid]["final"]
        for label in baseline_outputs:
            preds[label] = baseline_outputs[label][sid]["final"]

        metrics = {label: _compute_gap_metrics(info["target"], preds[label], info["obs_mask"]) for label in MODEL_ORDER}
        _plot_altitude_compare(sample_out_dir, sid, str(row["scenario"]), info, preds, metrics)
        _plot_latlon_compare(sample_out_dir, sid, str(row["scenario"]), info, preds, metrics)

        merged = pd.DataFrame(
            {
                "sample_id": sid,
                "flight_id": info["flight_id"],
                "minute_index": info["minute_index"],
                "obs_mask": info["obs_mask"].astype(int),
                "is_adsc_anchor": (info["obs_mask"] > 0.5).astype(int),
                "truth_lat": info["target"][:, 0],
                "truth_lon": info["target"][:, 1],
                "truth_alt": info["target"][:, 2],
            }
        )
        for label in MODEL_ORDER:
            merged[f"{label}_pred_lat"] = preds[label][:, 0]
            merged[f"{label}_pred_lon"] = preds[label][:, 1]
            merged[f"{label}_pred_alt"] = preds[label][:, 2]
            summary_rows.append({"scenario": row["scenario"], "sample_id": sid, "model": label, **metrics[label]})
        merged.to_csv(sample_out_dir / "recovered_minute_compare_all_models.csv", index=False)

        fig = _build_figure(merged, sid, str(row["scenario"]))
        html = plot(fig, include_plotlyjs=True, output_type="div")
        page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8" />
<title>{sid} - Interactive 3D Recovery</title>
<style>
body {{ font-family: "Times New Roman", serif; margin: 0; background: #fff; }}
.wrap {{ padding: 8px 10px 10px; }}
.meta {{ color: #444; margin: 4px 0 12px; }}
a {{ color: #0b57d0; text-decoration: none; }}
.plot-wrap {{ width: 100%; min-height: 92vh; }}
</style></head>
<body><div class="wrap">
<div><a href="../index.html">Back to index</a></div>
<div class="meta">scenario={row['scenario']}, sample_id={sid}</div>
<div class="plot-wrap">{html}</div>
</div></body></html>"""
        (sample_out_dir / f"interactive_3d_{sid}.html").write_text(page, encoding="utf-8")
        index_rows.append({"scenario": row["scenario"], "sample_id": sid})

    pd.DataFrame(summary_rows).to_csv(out_dir / "selected_model_compare_all_summary.csv", index=False)
    selected.to_csv(out_dir / "selected_typical_samples.csv", index=False)
    _write_index(index_rows, out_dir)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
