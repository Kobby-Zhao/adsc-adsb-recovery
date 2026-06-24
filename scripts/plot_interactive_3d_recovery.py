from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


MODEL_ORDER = [
    ("ourmethod", "OurMethod", "#d62728"),
    ("unilstm_baseline", "UniLSTM", "#1f77b4"),
    ("bilstm_baseline", "BiLSTM", "#2ca02c"),
    ("cnnlstm_baseline", "CNNLSTM", "#9467bd"),
    ("transformer_baseline", "Transformer", "#ff7f0e"),
    ("kalman_filter_baseline", "Kalman", "#17becf"),
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Create interactive 3D recovery plots as draggable HTML.")
    p.add_argument("--csv", required=True)
    p.add_argument("--out-html", required=True)
    p.add_argument("--title", default="Interactive 3D recovery")
    p.add_argument("--cruise-only", action="store_true")
    p.add_argument("--cruise-alt-threshold-m", type=float, default=10000.0)
    return p


def _filter_cruise(df: pd.DataFrame, threshold_m: float) -> pd.DataFrame:
    out = df.copy()
    mask = pd.to_numeric(out["alt"], errors="coerce") >= float(threshold_m)
    if int(mask.sum()) < 5:
        cutoff = float(max(pd.to_numeric(out["alt"], errors="coerce").quantile(0.7), threshold_m * 0.9))
        mask = pd.to_numeric(out["alt"], errors="coerce") >= cutoff
    return out.loc[mask].copy()


def main() -> int:
    args = build_parser().parse_args()
    df = pd.read_csv(args.csv, parse_dates=["minute_ts"])
    if args.cruise_only:
        df = _filter_cruise(df, args.cruise_alt_threshold_m)
    if df.empty:
        raise RuntimeError("No rows left after filtering.")

    fig = go.Figure()

    known_adsb = df[df["known_adsb"].astype(int) == 1].copy()
    fig.add_trace(
        go.Scatter3d(
            x=known_adsb["lon"],
            y=known_adsb["lat"],
            z=known_adsb["alt"],
            mode="lines+markers",
            name="Known ADS-B minute",
            line=dict(color="#111111", width=5),
            marker=dict(size=2, color="#111111"),
        )
    )

    anchors = df[df["is_adsc_anchor"].astype(int) == 1].copy()
    fig.add_trace(
        go.Scatter3d(
            x=anchors["lon"],
            y=anchors["lat"],
            z=anchors["alt"],
            mode="markers",
            name="ADS-C anchors",
            marker=dict(size=5, color="#d95f02", symbol="diamond"),
        )
    )

    for model_key, model_name, color in MODEL_ORDER:
        lon_col = f"{model_key}_pred_lon"
        lat_col = f"{model_key}_pred_lat"
        alt_col = f"{model_key}_pred_alt"
        if lon_col not in df.columns:
            continue
        g = df.dropna(subset=[lon_col, lat_col, alt_col]).copy()
        if g.empty:
            continue
        fig.add_trace(
            go.Scatter3d(
                x=g[lon_col],
                y=g[lat_col],
                z=g[alt_col],
                mode="lines",
                name=model_name,
                line=dict(color=color, width=4),
            )
        )

    fig.update_layout(
        title=args.title,
        scene=dict(
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            zaxis_title="Altitude (m)",
            camera=dict(
                eye=dict(x=1.65, y=-1.55, z=0.72),
            ),
        ),
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(255,255,255,0.78)",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        template="plotly_white",
    )

    out = Path(args.out_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
