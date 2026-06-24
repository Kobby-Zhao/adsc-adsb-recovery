from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot


DEFAULT_COLORS = {
    "ACT-BiMamba": "red",
    "BiMamba": "red",
    "PiecewiseLinear": "#7f7f7f",
    "Kalman Filter": "#17becf",
    "Mamba": "#8c564b",
    "LSTM": "rgb(31,119,180)",
    "UniLSTM-proto": "rgb(31,119,180)",
    "BiLSTM": "rgb(44,160,44)",
    "BiLSTM-proto": "rgb(44,160,44)",
    "CNN+LSTM": "rgb(148,103,189)",
    "CNN-LSTM-proto": "rgb(148,103,189)",
    "Transformer": "rgb(255,127,14)",
    "Transformer-proto": "rgb(255,127,14)",
}


def _extract_models(df: pd.DataFrame) -> list[str]:
    models = []
    for c in df.columns:
        if c.endswith("_pred_lat"):
            models.append(c[: -len("_pred_lat")])
    return sorted(models, key=lambda x: (0 if x in {"ACT-BiMamba", "BiMamba"} else 1, x))


def _series(df: pd.DataFrame, prefix: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    return (
        pd.to_numeric(df[f"{prefix}_pred_lon"], errors="coerce"),
        pd.to_numeric(df[f"{prefix}_pred_lat"], errors="coerce"),
        pd.to_numeric(df[f"{prefix}_pred_alt"], errors="coerce"),
    )


def _build_figure(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    models = _extract_models(df)

    if {"truth_lon", "truth_lat", "truth_alt"}.issubset(df.columns):
        fig.add_trace(
            go.Scatter3d(
                x=pd.to_numeric(df["truth_lon"], errors="coerce"),
                y=pd.to_numeric(df["truth_lat"], errors="coerce"),
                z=pd.to_numeric(df["truth_alt"], errors="coerce"),
                mode="lines",
                name="ADS-B true",
                line=dict(color="black", width=6),
                hovertemplate="ADS-B true<br>Lon=%{x:.3f}<br>Lat=%{y:.3f}<br>Alt=%{z:.1f} m<extra></extra>",
            )
        )

    anchor_lon_col = "obs_lon" if "obs_lon" in df.columns else "truth_lon"
    anchor_lat_col = "obs_lat" if "obs_lat" in df.columns else "truth_lat"
    anchor_alt_col = "obs_alt" if "obs_alt" in df.columns else "truth_alt"
    if {"is_adsc_anchor", anchor_lon_col, anchor_lat_col, anchor_alt_col}.issubset(df.columns):
        anchors = df[df["is_adsc_anchor"].astype(int) == 1].copy()
        fig.add_trace(
            go.Scatter3d(
                x=pd.to_numeric(anchors[anchor_lon_col], errors="coerce"),
                y=pd.to_numeric(anchors[anchor_lat_col], errors="coerce"),
                z=pd.to_numeric(anchors[anchor_alt_col], errors="coerce"),
                mode="markers",
                name="ADS-C anchors",
                marker=dict(size=3, color="black", symbol="circle"),
                hovertemplate="ADS-C anchor<br>Lon=%{x:.3f}<br>Lat=%{y:.3f}<br>Alt=%{z:.1f} m<extra></extra>",
            )
        )

    for model in models:
        x, y, z = _series(df, model)
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                name=model,
                line=dict(
                    color=DEFAULT_COLORS.get(model, "#444444"),
                    dash="solid",
                    width=6 if model in {"ACT-BiMamba", "BiMamba"} else 4,
                ),
                hovertemplate=f"{model}<br>Lon=%{{x:.3f}}<br>Lat=%{{y:.3f}}<br>Alt=%{{z:.1f}} m<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
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


def _write_index(rows: list[dict], out_dir: Path, title: str) -> None:
    items = []
    for row in rows:
        name = row["name"]
        items.append(f'<li><a href="{name}/interactive_3d_{name}.html">{name}</a></li>')
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8" />
<title>{title}</title>
<style>
body {{ font-family: "Times New Roman", serif; margin: 24px; background: #fff; color: #111; }}
a {{ color: #0b57d0; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
</style></head>
<body>
<h1>{title}</h1>
<p>Click a case below to open an interactive 3D trajectory page.</p>
<ul>{''.join(items)}</ul>
</body></html>"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--pattern", default="*/recovered_minute_compare*.csv")
    ap.add_argument("--title", default="Interactive 3D Recovery Compare")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    csvs = sorted(base_dir.glob(args.pattern))
    if not csvs:
        raise RuntimeError(f"No recovered csv found under {base_dir} with pattern {args.pattern}")

    rows = []
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        name = csv_path.parent.name
        fig = _build_figure(df, f"{args.title}: {name}")
        html_div = plot(fig, include_plotlyjs=True, output_type="div")
        page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8" />
<title>{name} - Interactive 3D Recovery</title>
<style>
body {{ font-family: "Times New Roman", serif; margin: 0; background: #fff; }}
.wrap {{ padding: 8px 10px 10px; }}
.meta {{ color: #444; margin: 4px 0 12px; }}
a {{ color: #0b57d0; text-decoration: none; }}
.plot-wrap {{ width: 100%; min-height: 92vh; }}
</style></head>
<body><div class="wrap">
<div><a href="../index.html">Back to index</a></div>
<div class="meta">case={name}</div>
<div class="plot-wrap">{html_div}</div>
</div></body></html>"""
        (csv_path.parent / f"interactive_3d_{name}.html").write_text(page, encoding="utf-8")
        rows.append({"name": name})

    _write_index(rows, base_dir, args.title)
    print(base_dir / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
