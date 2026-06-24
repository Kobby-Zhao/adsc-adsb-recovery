from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_interpolation_baselines import _interp_linear_gapwise  # noqa: E402
from scripts.eval_rts_kalman_baseline import _smooth_sample  # noqa: E402
from scripts.real_adsc_replay_eval import _predict_on_frame  # noqa: E402
from scripts.train import load_config  # noqa: E402


STYLE = {
    "ADS-C anchors": {"color": "black", "marker": "o", "markersize": 5, "linestyle": "None"},
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

MODEL_SPECS = [
    {
        "name": "Ours",
        "config": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_gapaware_small.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_gapaware_small/best.pt",
    },
    {
        "name": "Mamba",
        "config": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_mamba_proto.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_mamba_proto/best.pt",
    },
    {
        "name": "Bi-Mamba",
        "config": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_bimamba.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_bimamba/best.pt",
    },
    {
        "name": "LSTM",
        "config": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_unilstm_proto.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_unilstm_proto/best.pt",
    },
    {
        "name": "BiLSTM",
        "config": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_bilstm_proto.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_bilstm_proto/best.pt",
    },
    {
        "name": "CNN+LSTM",
        "config": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_cnnlstm_proto.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_cnnlstm_proto/best.pt",
    },
    {
        "name": "Transformer",
        "config": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_transformer_proto.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_transformer_proto/best.pt",
    },
]

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
            "font.serif": ["Times New Roman", "Noto Serif CJK JP", "SimSun", "DejaVu Serif"],
            "axes.unicode_minus": False,
            "savefig.dpi": 300,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
        }
    )


def _pick_legend_position(df: pd.DataFrame) -> tuple[str, tuple[float, float]]:
    xs = []
    ys = []
    for model_name in MODEL_ORDER:
        xs.extend(pd.to_numeric(df[f"{model_name}_pred_lon"], errors="coerce").tolist())
        ys.extend(pd.to_numeric(df[f"{model_name}_pred_lat"], errors="coerce").tolist())
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    valid = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[valid]
    ys = ys[valid]
    if len(xs) == 0:
        return "upper left", (0.70, 0.93)

    xmid = float(np.nanmedian(xs))
    ymid = float(np.nanmedian(ys))
    quadrant_counts = {
        "upper left": int(np.sum((xs <= xmid) & (ys >= ymid))),
        "upper right": int(np.sum((xs > xmid) & (ys >= ymid))),
        "lower left": int(np.sum((xs <= xmid) & (ys < ymid))),
        "lower right": int(np.sum((xs > xmid) & (ys < ymid))),
    }
    corner = min(quadrant_counts, key=quadrant_counts.get)
    anchors = {
        "upper left": (0.05, 0.94),
        "upper right": (0.70, 0.94),
        "lower left": (0.05, 0.22),
        "lower right": (0.70, 0.22),
    }
    return corner, anchors[corner]


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371000.0 * 2 * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def _build_anchor_only_frame(adsc_anchor_csv: Path, pair_id: str) -> pd.DataFrame:
    adsc = pd.read_csv(adsc_anchor_csv).copy()
    adsc["timestamp"] = pd.to_datetime(adsc["timestamp"], utc=True)
    adsc["anchor_minute_ts"] = adsc["timestamp"].dt.round("1min")
    adsc = (
        adsc.rename(columns={"latitude": "anchor_lat", "longitude": "anchor_lon", "altitude_m": "anchor_alt"})
        .groupby("anchor_minute_ts", as_index=False)
        .last()
        .sort_values("anchor_minute_ts")
        .reset_index(drop=True)
    )
    full_ts = pd.date_range(
        start=pd.to_datetime(adsc["anchor_minute_ts"].min(), utc=True),
        end=pd.to_datetime(adsc["anchor_minute_ts"].max(), utc=True),
        freq="1min",
        tz="UTC",
    )
    full = pd.DataFrame({"minute_ts": full_ts})
    full["sample_id"] = f"{pair_id}_adsc_anchor_only"
    full["flight_id"] = pair_id
    full = full.merge(adsc, left_on="minute_ts", right_on="anchor_minute_ts", how="left")
    full["flight_id"] = pair_id
    full["is_adsc_anchor"] = full["anchor_minute_ts"].notna().astype(int)
    full["known_adsb"] = 0
    full["obs_mask"] = full["is_adsc_anchor"].astype(int)
    full["obs_source"] = np.where(full["is_adsc_anchor"] == 1, "adsc_anchor", "missing")
    for c in ["anchor_lat", "anchor_lon", "anchor_alt"]:
        full[c] = pd.to_numeric(full[c], errors="coerce")
    full["lat"] = full["anchor_lat"].interpolate(limit_direction="both")
    full["lon"] = full["anchor_lon"].interpolate(limit_direction="both")
    full["alt"] = full["anchor_alt"].interpolate(limit_direction="both")
    lat = full["lat"].to_numpy(dtype=float)
    lon = full["lon"].to_numpy(dtype=float)
    alt = full["alt"].to_numpy(dtype=float)
    speed = np.zeros(len(full), dtype=float)
    heading = np.zeros(len(full), dtype=float)
    for i in range(1, len(full)):
        speed[i] = _haversine_m(lat[i - 1], lon[i - 1], lat[i], lon[i]) / 60.0
        heading[i] = _bearing_deg(lat[i - 1], lon[i - 1], lat[i], lon[i])
    if len(full) >= 2:
        speed[0] = speed[1]
        heading[0] = heading[1]
    frame = full[["sample_id", "flight_id", "minute_ts", "obs_mask", "known_adsb", "is_adsc_anchor", "obs_source"]].copy()
    frame["lat"] = lat
    frame["lon"] = lon
    frame["alt"] = alt
    frame["speed"] = speed
    frame["heading"] = heading
    frame["obs_lat"] = lat
    frame["obs_lon"] = lon
    frame["obs_alt"] = alt
    return frame


def _run_neural_models(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    stitched: dict[str, pd.DataFrame] = {}
    for spec in MODEL_SPECS:
        cfg = load_config(_resolve(spec["config"]))
        pred = _predict_on_frame(cfg=cfg, checkpoint=_resolve(spec["checkpoint"]), frame=frame, pred_key="pred_pos")
        stitched[spec["name"]] = pred.sort_values("minute_ts").reset_index(drop=True)
    return stitched


def _run_linear_baseline(frame: pd.DataFrame) -> pd.DataFrame:
    sdf = frame[["minute_ts", "obs_mask", "lat", "lon", "alt"]].copy()
    lat_p, lon_p, alt_p = _interp_linear_gapwise(sdf)
    out = frame[["minute_ts"]].copy()
    out["pred_lat"] = lat_p
    out["pred_lon"] = lon_p
    out["pred_alt"] = alt_p
    return out


def _run_kalman_baseline(frame: pd.DataFrame) -> pd.DataFrame:
    sdf = frame[["minute_ts", "obs_mask", "lat", "lon", "alt"]].copy()
    lat_p, lon_p, alt_p = _smooth_sample(sdf)
    out = frame[["minute_ts"]].copy()
    out["pred_lat"] = lat_p
    out["pred_lon"] = lon_p
    out["pred_alt"] = alt_p
    return out


def _plot_altitude(frame: pd.DataFrame, stitched: dict[str, pd.DataFrame], out_base: Path) -> None:
    x = frame.sort_values("minute_ts").reset_index(drop=True).copy()
    x_min = (pd.to_datetime(x["minute_ts"], utc=True) - pd.to_datetime(x["minute_ts"], utc=True).min()).dt.total_seconds() / 60.0
    anchors = x["is_adsc_anchor"].astype(int) == 1
    fig, ax = plt.subplots(figsize=(10.0, 8.0))
    ax.plot(x_min[anchors], pd.to_numeric(x.loc[anchors, "obs_alt"], errors="coerce"), label="ADS-C anchors", zorder=10, **STYLE["ADS-C anchors"])
    for model_name in MODEL_ORDER:
        s = stitched[model_name]
        gx = (pd.to_datetime(s["minute_ts"], utc=True) - pd.to_datetime(s["minute_ts"], utc=True).min()).dt.total_seconds() / 60.0
        ax.plot(
            gx,
            pd.to_numeric(s["pred_alt"], errors="coerce"),
            label=DISPLAY_NAME[model_name],
            zorder=8 if model_name == "Ours" else 5,
            **STYLE[model_name],
        )
    ax.set_title(f"{x['flight_id'].iloc[0]} | real ADS-C anchor-only recovery")
    ax.set_xlabel("Minutes from first ADS-C anchor")
    ax.set_ylabel("Altitude (m)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_latlon(frame: pd.DataFrame, stitched: dict[str, pd.DataFrame], out_base: Path) -> None:
    x = frame.sort_values("minute_ts").reset_index(drop=True).copy()
    anchors = x["is_adsc_anchor"].astype(int) == 1
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.plot(x.loc[anchors, "obs_lon"], x.loc[anchors, "obs_lat"], label="ADS-C anchors", zorder=10, **STYLE["ADS-C anchors"])
    for model_name in MODEL_ORDER:
        s = stitched[model_name]
        ax.plot(
            pd.to_numeric(s["pred_lon"], errors="coerce"),
            pd.to_numeric(s["pred_lat"], errors="coerce"),
            label=DISPLAY_NAME[model_name],
            zorder=8 if model_name == "Ours" else 5,
            **STYLE[model_name],
        )
    ax.set_title(f"{x['flight_id'].iloc[0]} | real ADS-C 2D recovery")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_static_3d(df: pd.DataFrame, pair_id: str, out_base: Path) -> None:
    anchors = df["is_adsc_anchor"].astype(int) == 1
    fig = plt.figure(figsize=(11.5, 8.2), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    ax.scatter(
        pd.to_numeric(df.loc[anchors, "obs_lon"], errors="coerce"),
        pd.to_numeric(df.loc[anchors, "obs_lat"], errors="coerce"),
        pd.to_numeric(df.loc[anchors, "obs_alt"], errors="coerce"),
        label="ADS-C anchors",
        zorder=10,
        color=STYLE["ADS-C anchors"]["color"],
        marker="o",
        s=26,
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

    ax.set_title(f"三维轨迹恢复对比: {pair_id}", pad=20, fontsize=15, fontweight="semibold")
    ax.set_xlabel("经度 (°)", fontsize=13, labelpad=12, fontweight="semibold")
    ax.set_ylabel("纬度 (°)", fontsize=13, labelpad=12, fontweight="semibold")
    ax.set_zlabel("高度 (m)", fontsize=13, labelpad=14, fontweight="semibold")
    ax.view_init(elev=18, azim=-63)
    ax.set_box_aspect((1.20, 1.00, 1.15))
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
    legend_loc, legend_anchor = _pick_legend_position(df)
    ax.legend(
        fontsize=10.5,
        loc=legend_loc,
        bbox_to_anchor=legend_anchor,
        borderaxespad=0.2,
        framealpha=0.90,
        facecolor=(1.0, 1.0, 1.0, 0.90),
        edgecolor=(0.35, 0.35, 0.35, 1.0),
        borderpad=0.5,
        labelspacing=0.35,
        handlelength=2.0,
    )
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.05, top=0.90)
    fig.savefig(out_base.with_suffix(".png"), dpi=600, facecolor="white", bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _build_figure(df: pd.DataFrame, pair_id: str) -> go.Figure:
    fig = go.Figure()
    anchors = df[df["is_adsc_anchor"].astype(int) == 1].copy()
    fig.add_trace(
        go.Scatter3d(
            x=pd.to_numeric(anchors["obs_lon"], errors="coerce"),
            y=pd.to_numeric(anchors["obs_lat"], errors="coerce"),
            z=pd.to_numeric(anchors["obs_alt"], errors="coerce"),
            mode="markers",
            name="ADS-C anchors",
            marker=dict(size=3, color="black", symbol="circle"),
            hovertemplate="Anchor<br>Lon=%{x:.3f}<br>Lat=%{y:.3f}<br>Alt=%{z:.1f} m<extra></extra>",
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
        title=f"Interactive 3D Recovery Comparison: {pair_id}",
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


def _write_index(case_rows: list[dict], out_dir: Path) -> None:
    cards = []
    for row in case_rows:
        pair_id = row["pair_id"]
        html_name = f"{pair_id}/interactive_3d_{pair_id}.html"
        cards.append(f'<li><a href="{html_name}">{pair_id}</a><span> anchors={row["anchor_count"]}, minutes={row["minutes"]}</span></li>')
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8" />
<title>Real ADS-C Interactive 3D Recovery Compare</title>
<style>
body {{ font-family: "Times New Roman", serif; margin: 24px; background: #fff; color: #111; }}
a {{ color: #0b57d0; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
</style></head>
<body>
<h1>Real ADS-C Interactive 3D Recovery Compare</h1>
<p>Click a case below to open an interactive 3D trajectory page. You can rotate, zoom, and pan directly in the browser.</p>
<ul>{''.join(cards)}</ul>
</body></html>"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--selected-cases-csv",
        default="outputs/runs/clean_cross_ocean_adsc_adsb_cases_extra_20260517/selected_clean_cross_ocean_cases.csv",
    )
    ap.add_argument("--pair-ids", default="", help="Comma-separated pair_ids. Empty means all selected rows.")
    ap.add_argument(
        "--out-dir",
        default="outputs/runs/real_adsc_anchor_only_all_models_20260601",
    )
    args = ap.parse_args()

    _set_plot_style()
    selected = pd.read_csv(_resolve(args.selected_cases_csv))
    if args.pair_ids.strip():
        wanted = {x.strip() for x in args.pair_ids.split(",") if x.strip()}
        selected = selected[selected["pair_id"].astype(str).isin(wanted)].copy()
    if selected.empty:
        raise RuntimeError("No selected real ADS-C cases found after filtering.")

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in MODEL_SPECS:
        for k in ("config", "checkpoint"):
            if not _resolve(spec[k]).exists():
                raise FileNotFoundError(f"Missing {k}: {_resolve(spec[k])}")

    rows = []
    case_rows = []
    for _, row in selected.iterrows():
        pair_id = str(row["pair_id"])
        frame = _build_anchor_only_frame(_resolve(row["adsc_anchor_csv"]), pair_id)
        stitched = _run_neural_models(frame)
        stitched["PiecewiseLinear"] = _run_linear_baseline(frame)
        stitched["Kalman Filter"] = _run_kalman_baseline(frame)

        case_dir = out_dir / pair_id
        case_dir.mkdir(parents=True, exist_ok=True)
        _plot_altitude(frame, stitched, case_dir / f"real_adsc_alt_compare_{pair_id}")
        _plot_latlon(frame, stitched, case_dir / f"real_adsc_latlon_compare_{pair_id}")

        merged = frame[["sample_id", "flight_id", "minute_ts", "obs_mask", "is_adsc_anchor", "obs_source", "obs_lat", "obs_lon", "obs_alt"]].copy()
        for model_name in MODEL_ORDER:
            s = stitched[model_name]
            merged[f"{model_name}_pred_lat"] = pd.to_numeric(s["pred_lat"], errors="coerce")
            merged[f"{model_name}_pred_lon"] = pd.to_numeric(s["pred_lon"], errors="coerce")
            merged[f"{model_name}_pred_alt"] = pd.to_numeric(s["pred_alt"], errors="coerce")
            rows.append({"pair_id": pair_id, "model": model_name, "anchor_count": int(frame["is_adsc_anchor"].sum()), "minutes": int(len(frame))})
        merged.to_csv(case_dir / "recovered_minute_compare_all_models.csv", index=False)
        _plot_static_3d(merged, pair_id, case_dir / f"real_adsc_3d_compare_{pair_id}")

        fig = _build_figure(merged, pair_id)
        html = plot(fig, include_plotlyjs=True, output_type="div")
        page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8" />
<title>{pair_id} - Interactive 3D Recovery</title>
<style>
body {{ font-family: "Times New Roman", serif; margin: 0; background: #fff; }}
.wrap {{ padding: 8px 10px 10px; }}
.meta {{ color: #444; margin: 4px 0 12px; }}
a {{ color: #0b57d0; text-decoration: none; }}
.plot-wrap {{ width: 100%; min-height: 92vh; }}
</style></head>
<body><div class="wrap">
<div><a href="../index.html">Back to index</a></div>
<div class="meta">pair_id={pair_id}, anchors={int(frame['is_adsc_anchor'].sum())}, minutes={len(frame)}</div>
<div class="plot-wrap">{html}</div>
</div></body></html>"""
        (case_dir / f"interactive_3d_{pair_id}.html").write_text(page, encoding="utf-8")
        case_rows.append({"pair_id": pair_id, "anchor_count": int(frame["is_adsc_anchor"].sum()), "minutes": int(len(frame))})

    pd.DataFrame(rows).to_csv(out_dir / "real_adsc_all_model_compare_index.csv", index=False)
    _write_index(case_rows, out_dir)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
