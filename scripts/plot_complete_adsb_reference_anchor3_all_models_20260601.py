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
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_bidirectional_prediction_mechanism import _run_model_for_samples  # noqa: E402
from scripts.eval_interpolation_baselines import _interp_linear_gapwise  # noqa: E402
from scripts.eval_rts_kalman_baseline import _smooth_sample  # noqa: E402
from src.training.utils import load_config  # noqa: E402


STYLE = {
    "ADS-B true": {"color": "black", "linestyle": "-", "linewidth": 2.5},
    "ADS-C-like anchors": {"color": "black", "marker": "o", "markersize": 5, "linestyle": "None"},
    "Ours": {"color": "red", "linestyle": "-", "linewidth": 2.4},
    "PiecewiseLinear": {"color": "#7f7f7f", "linestyle": "-", "linewidth": 1.8},
    "Kalman Filter": {"color": "#17becf", "linestyle": "-", "linewidth": 1.8},
    "Mamba": {"color": "#8c564b", "linestyle": "-", "linewidth": 1.8},
    "Bi-Mamba": {"color": "#d62728", "linestyle": "-", "linewidth": 1.8},
    "LSTM": {"color": "#1f77b4", "linestyle": "-", "linewidth": 1.8},
    "BiLSTM": {"color": "#2ca02c", "linestyle": "-", "linewidth": 1.8},
    "CNN+LSTM": {"color": "#9467bd", "linestyle": "-", "linewidth": 1.8},
    "Transformer": {"color": "#ff7f0e", "linestyle": "-", "linewidth": 1.8},
}

BASE_MODEL_SPECS = {
    "Ours": {
        "config": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_gapaware_small.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_gapaware_small/best.pt",
    },
    "Mamba": {
        "config": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_mamba_proto.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_mamba_proto/best.pt",
    },
    "Bi-Mamba": {
        "config": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_bimamba.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_bimamba/best.pt",
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
    "Ours",
    "PiecewiseLinear",
    "Kalman Filter",
    "Mamba",
    "Bi-Mamba",
    "LSTM",
    "BiLSTM",
    "CNN+LSTM",
    "Transformer",
]

DISPLAY_NAME = {
    "Ours": "本文方案",
    "PiecewiseLinear": "分段线性插值",
    "Kalman Filter": "Kalman Filter",
    "Mamba": "Mamba",
    "Bi-Mamba": "Bi-Mamba",
    "LSTM": "LSTM",
    "BiLSTM": "BiLSTM",
    "CNN+LSTM": "CNN+LSTM",
    "Transformer": "Transformer",
}


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


def _build_model_specs(samples_path: Path, out_dir: Path) -> dict[str, dict[str, str]]:
    cfg_dir = out_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    specs: dict[str, dict[str, str]] = {}
    for model_name, spec in BASE_MODEL_SPECS.items():
        cfg = load_config(str(_resolve(spec["config"])))
        cfg["data"]["samples_path"] = str(samples_path)
        cfg["data"]["split"] = {"train_ratio": 0.0, "val_ratio": 0.0, "test_ratio": 1.0}
        cfg["training"]["batch_size"] = 16
        out_cfg = cfg_dir / f"{model_name.replace('+', 'plus').replace(' ', '_')}.yaml"
        with out_cfg.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        specs[model_name] = {"config": str(out_cfg), "checkpoint": str(_resolve(spec["checkpoint"]))}
    return specs


def _load_anchor3_frame(samples_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(samples_csv)
    df = df[df["sample_id"].astype(str).str.endswith("__cruise_anchor3")].copy()
    if df.empty:
        raise RuntimeError("No anchor3 ADS-B sparse cruise samples found.")
    return df


def _collect_neural_predictions(selected_ids: set[str], model_specs: dict[str, dict[str, str]], device: torch.device) -> dict[str, dict[str, dict]]:
    outputs: dict[str, dict[str, dict]] = {}
    for label in BASE_MODEL_SPECS:
        outputs[label] = _run_model_for_samples(label, model_specs, selected_ids, "test", device)
    return outputs


def _collect_baseline_predictions(frame: pd.DataFrame) -> dict[str, dict[str, dict]]:
    outputs = {"PiecewiseLinear": {}, "Kalman Filter": {}}
    for sid, g in frame.groupby("sample_id", sort=False):
        g = g.sort_values("minute_ts").reset_index(drop=True)
        sdf = g[["minute_ts", "obs_mask", "lat", "lon", "alt"]].copy()
        sdf["minute_ts"] = pd.to_datetime(sdf["minute_ts"], utc=True)
        lat_p, lon_p, alt_p = _interp_linear_gapwise(sdf)
        outputs["PiecewiseLinear"][sid] = {"final": np.stack([lat_p, lon_p, alt_p], axis=-1)}
        lat_k, lon_k, alt_k = _smooth_sample(sdf)
        outputs["Kalman Filter"][sid] = {"final": np.stack([lat_k, lon_k, alt_k], axis=-1)}
    return outputs


def _compute_gap_metrics(true_xyz: np.ndarray, pred_xyz: np.ndarray, obs_mask: np.ndarray) -> dict[str, float]:
    gap = obs_mask <= 0.5
    diff = pred_xyz[gap] - true_xyz[gap]
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


def _plot_altitude(out_dir: Path, case_id: str, info: dict, preds: dict[str, np.ndarray], metrics: dict[str, dict[str, float]]) -> None:
    x = info["minute_index"]
    truth = info["target"][:, 2]
    anchor = info["obs_mask"] > 0.5
    fig, ax = plt.subplots(figsize=(13.2, 4.8))
    ax.plot(x, truth, label="ADS-B true", zorder=9, **STYLE["ADS-B true"])
    ax.plot(x[anchor], truth[anchor], label="ADS-C-like anchors", zorder=10, **STYLE["ADS-C-like anchors"])
    for label in MODEL_ORDER:
        series = preds[label][:, 2]
        m = metrics[label]
        legend_label = f"{DISPLAY_NAME[label]} (RMSE={m['gap_alt_rmse']:.2f}, MAE={m['gap_alt_mae']:.2f})"
        ax.plot(x, series, label=legend_label, zorder=8 if label == "Ours" else 5, **STYLE[label])
    ax.set_title(f"{case_id}: ADS-B sparse recovery comparison")
    ax.set_xlabel("Minute index")
    ax.set_ylabel("Altitude (m)")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=7.3, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / f"adsb_alt_compare_{case_id}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"adsb_alt_compare_{case_id}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_latlon(out_dir: Path, case_id: str, info: dict, preds: dict[str, np.ndarray], metrics: dict[str, dict[str, float]]) -> None:
    truth = info["target"]
    anchor = info["obs_mask"] > 0.5
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.plot(truth[:, 1], truth[:, 0], label="ADS-B true", zorder=9, **STYLE["ADS-B true"])
    ax.plot(truth[anchor, 1], truth[anchor, 0], label="ADS-C-like anchors", zorder=10, **STYLE["ADS-C-like anchors"])
    for label in MODEL_ORDER:
        m = metrics[label]
        legend_label = f"{DISPLAY_NAME[label]} (lat={m['gap_lat_rmse']:.3f}, lon={m['gap_lon_rmse']:.3f})"
        ax.plot(preds[label][:, 1], preds[label][:, 0], label=legend_label, zorder=8 if label == "Ours" else 5, **STYLE[label])
    ax.set_title(f"{case_id}: ADS-B sparse 2D recovery")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=7.0)
    fig.tight_layout()
    fig.savefig(out_dir / f"adsb_latlon_compare_{case_id}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"adsb_latlon_compare_{case_id}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_static_3d(df: pd.DataFrame, case_id: str, out_dir: Path) -> None:
    anchors = df["obs_mask"].astype(int) == 1
    fig = plt.figure(figsize=(11.5, 8.2), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    ax.scatter(
        pd.to_numeric(df.loc[anchors, "true_lon"], errors="coerce"),
        pd.to_numeric(df.loc[anchors, "true_lat"], errors="coerce"),
        pd.to_numeric(df.loc[anchors, "true_alt"], errors="coerce"),
        label="ADS-C-like anchors",
        zorder=10,
        color="black",
        marker="o",
        s=26,
    )
    ax.plot(
        pd.to_numeric(df["true_lon"], errors="coerce"),
        pd.to_numeric(df["true_lat"], errors="coerce"),
        pd.to_numeric(df["true_alt"], errors="coerce"),
        label="ADS-B true",
        zorder=9,
        **STYLE["ADS-B true"],
    )
    for model_name in MODEL_ORDER:
        ax.plot(
            pd.to_numeric(df[f"{model_name}_pred_lon"], errors="coerce"),
            pd.to_numeric(df[f"{model_name}_pred_lat"], errors="coerce"),
            pd.to_numeric(df[f"{model_name}_pred_alt"], errors="coerce"),
            label=DISPLAY_NAME[model_name],
            zorder=8 if model_name == "Ours" else 5,
            **STYLE[model_name],
        )

    ax.set_title(f"3D Recovery Comparison: {case_id}", pad=20, fontsize=15, fontweight="semibold")
    ax.set_xlabel("Longitude", fontsize=13, labelpad=12, fontweight="semibold")
    ax.set_ylabel("Latitude", fontsize=13, labelpad=12, fontweight="semibold")
    ax.set_zlabel("Altitude (m)", fontsize=13, labelpad=14, fontweight="semibold")
    ax.view_init(elev=21, azim=-124)
    ax.set_box_aspect((1.45, 1.10, 0.95))
    ax.grid(True, alpha=0.34)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((0.985, 0.985, 0.985, 1.0))
        axis.pane.set_edgecolor((0.45, 0.45, 0.45, 1.0))
        axis._axinfo["grid"]["color"] = (0.70, 0.70, 0.70, 1.0)
        axis._axinfo["grid"]["linewidth"] = 0.95
        axis._axinfo["grid"]["linestyle"] = "-"
        axis._axinfo["axisline"]["color"] = (0.25, 0.25, 0.25, 1.0)
        axis._axinfo["axisline"]["linewidth"] = 1.2
    for axis_name in ["xaxis", "yaxis", "zaxis"]:
        axis_obj = getattr(ax, axis_name)
        if hasattr(axis_obj, "line"):
            axis_obj.line.set_color((0.2, 0.2, 0.2, 1.0))
            axis_obj.line.set_linewidth(1.3)
    ax.tick_params(axis="x", labelsize=14, width=1.2, length=5.5, pad=4)
    ax.tick_params(axis="y", labelsize=14, width=1.2, length=5.5, pad=4)
    ax.tick_params(axis="z", labelsize=14, width=1.2, length=5.5, pad=6)
    ax.legend(
        fontsize=10.5,
        loc="upper left",
        bbox_to_anchor=(0.70, 0.93),
        borderaxespad=0.2,
        framealpha=0.90,
        facecolor=(1.0, 1.0, 1.0, 0.90),
        edgecolor=(0.35, 0.35, 0.35, 1.0),
        borderpad=0.5,
        labelspacing=0.35,
        handlelength=2.0,
    )
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.05, top=0.90)
    fig.savefig(out_dir / f"adsb_3d_compare_{case_id}.png", dpi=600, facecolor="white", bbox_inches="tight")
    fig.savefig(out_dir / f"adsb_3d_compare_{case_id}.pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _build_figure(df: pd.DataFrame, case_id: str) -> go.Figure:
    fig = go.Figure()
    anchors = df[df["obs_mask"].astype(int) == 1].copy()
    fig.add_trace(
        go.Scatter3d(
            x=pd.to_numeric(anchors["true_lon"], errors="coerce"),
            y=pd.to_numeric(anchors["true_lat"], errors="coerce"),
            z=pd.to_numeric(anchors["true_alt"], errors="coerce"),
            mode="markers",
            name="ADS-C-like anchors",
            marker=dict(size=3, color="black", symbol="circle"),
            hovertemplate="Anchor<br>Lon=%{x:.3f}<br>Lat=%{y:.3f}<br>Alt=%{z:.1f} m<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=pd.to_numeric(df["true_lon"], errors="coerce"),
            y=pd.to_numeric(df["true_lat"], errors="coerce"),
            z=pd.to_numeric(df["true_alt"], errors="coerce"),
            mode="lines",
            name="ADS-B true",
            line=dict(color="black", width=6),
            hovertemplate="ADS-B true<br>Lon=%{x:.3f}<br>Lat=%{y:.3f}<br>Alt=%{z:.1f} m<extra></extra>",
        )
    )
    for model in MODEL_ORDER:
        fig.add_trace(
            go.Scatter3d(
                x=pd.to_numeric(df[f"{model}_pred_lon"], errors="coerce"),
                y=pd.to_numeric(df[f"{model}_pred_lat"], errors="coerce"),
                z=pd.to_numeric(df[f"{model}_pred_alt"], errors="coerce"),
                mode="lines",
                name=DISPLAY_NAME[model],
                line=dict(color=STYLE[model]["color"], dash="solid", width=6 if model == "Ours" else 4),
                hovertemplate=f"{DISPLAY_NAME[model]}<br>Lon=%{{x:.3f}}<br>Lat=%{{y:.3f}}<br>Alt=%{{z:.1f}} m<extra></extra>",
            )
        )
    fig.update_layout(
        title=f"Interactive 3D Recovery Comparison: {case_id}",
        template="plotly_white",
        height=920,
        margin=dict(l=10, r=10, t=56, b=10),
        legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.9)"),
        scene=dict(
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            zaxis_title="Altitude (m)",
            aspectmode="cube",
            xaxis=dict(showbackground=True, backgroundcolor="white", gridcolor="#d9d9d9", zerolinecolor="#d9d9d9", tickfont=dict(size=16)),
            yaxis=dict(showbackground=True, backgroundcolor="white", gridcolor="#d9d9d9", zerolinecolor="#d9d9d9", tickfont=dict(size=16)),
            zaxis=dict(showbackground=True, backgroundcolor="white", gridcolor="#d9d9d9", zerolinecolor="#d9d9d9", tickformat=".0f", exponentformat="none", tickfont=dict(size=16)),
            camera=dict(eye=dict(x=1.55, y=-1.65, z=0.8)),
        ),
    )
    return fig


def _write_index(rows: list[dict], out_dir: Path) -> None:
    cards = []
    for row in rows:
        case_id = row["case_id"]
        html_name = f"{case_id}/interactive_3d_{case_id}.html"
        cards.append(f'<li><a href="{html_name}">{case_id}</a><span> flight={row["flight_id"]}</span></li>')
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8" />
<title>ADS-B Reference Flights Interactive 3D Recovery Compare</title>
<style>
body {{ font-family: "Times New Roman", serif; margin: 24px; background: #fff; color: #111; }}
a {{ color: #0b57d0; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
</style></head>
<body>
<h1>ADS-B Reference Flights Interactive 3D Recovery Compare</h1>
<p>Each case uses the formal reference flight under the anchor_count=3 sparse recovery protocol.</p>
<ul>{''.join(cards)}</ul>
</body></html>"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-csv", default="outputs/runs/complete_adsb_sparse_cruise_eval_20260519/sparse_cruise_samples.csv")
    parser.add_argument("--samples-parquet", default="outputs/runs/complete_adsb_sparse_cruise_eval_20260519/sparse_cruise_samples.parquet")
    parser.add_argument("--out-dir", default="outputs/runs/adsb_reference_anchor3_all_models_20260601")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    _set_plot_style()
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = _load_anchor3_frame(_resolve(args.samples_csv))
    samples_path = _resolve(args.samples_parquet)
    selected_ids = set(frame["sample_id"].astype(str).unique())
    model_specs = _build_model_specs(samples_path, out_dir)
    neural = _collect_neural_predictions(selected_ids, model_specs, torch.device(args.device))
    baselines = _collect_baseline_predictions(frame)

    case_rows: list[dict] = []
    for sid, g in frame.groupby("sample_id", sort=False):
        g = g.sort_values("minute_ts").reset_index(drop=True)
        case_id = str(g["source_case"].iloc[0])
        case_dir = out_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        truth = g[["lat", "lon", "alt"]].to_numpy(dtype=float)
        obs_mask = g["obs_mask"].to_numpy(dtype=float)
        info = {
            "minute_index": np.arange(len(g)),
            "target": truth,
            "obs_mask": obs_mask,
        }
        preds: dict[str, np.ndarray] = {
            "PiecewiseLinear": baselines["PiecewiseLinear"][sid]["final"],
            "Kalman Filter": baselines["Kalman Filter"][sid]["final"],
        }
        for label in BASE_MODEL_SPECS:
            preds[label] = neural[label][sid]["final"]
        metrics = {label: _compute_gap_metrics(truth, preds[label], obs_mask) for label in MODEL_ORDER}
        _plot_altitude(case_dir, case_id, info, preds, metrics)
        _plot_latlon(case_dir, case_id, info, preds, metrics)

        merged = pd.DataFrame(
            {
                "minute_index": np.arange(len(g)),
                "minute_ts": g["minute_ts"].astype(str),
                "obs_mask": g["obs_mask"].astype(int),
                "true_lat": truth[:, 0],
                "true_lon": truth[:, 1],
                "true_alt": truth[:, 2],
            }
        )
        for label in MODEL_ORDER:
            merged[f"{label}_pred_lat"] = preds[label][:, 0]
            merged[f"{label}_pred_lon"] = preds[label][:, 1]
            merged[f"{label}_pred_alt"] = preds[label][:, 2]
        merged.to_csv(case_dir / "recovered_minute_compare_all_models.csv", index=False, encoding="utf-8-sig")
        _plot_static_3d(merged, case_id, case_dir)
        fig = _build_figure(merged, case_id)
        html = plot(fig, include_plotlyjs=True, output_type="div")
        page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8" />
<title>{case_id} - ADS-B Recovery Compare</title>
<style>html,body{{margin:0;padding:0;background:#fff;}} .wrap{{width:100vw;height:100vh;}}</style>
</head><body><div class="wrap">{html}</div></body></html>"""
        (case_dir / f"interactive_3d_{case_id}.html").write_text(page, encoding="utf-8")
        case_rows.append({"case_id": case_id, "flight_id": str(g["source_flight_id"].iloc[0]), "sample_id": sid})

    pd.DataFrame(case_rows).to_csv(out_dir / "adsb_reference_all_model_compare_index.csv", index=False, encoding="utf-8-sig")
    _write_index(case_rows, out_dir)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
