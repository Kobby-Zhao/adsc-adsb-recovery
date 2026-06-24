from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot


STYLE = {
    "ADS-C anchors": {"color": "black", "size": 2, "symbol": "circle"},
    "BiMamba": {"color": "red", "dash": "solid", "width": 6},
    "UniLSTM-proto": {"color": "rgb(31,119,180)", "dash": "solid", "width": 4},
    "BiLSTM-proto": {"color": "rgb(44,160,44)", "dash": "solid", "width": 4},
    "CNN-LSTM-proto": {"color": "rgb(148,103,189)", "dash": "solid", "width": 4},
    "Transformer-proto": {"color": "rgb(255,127,14)", "dash": "solid", "width": 4},
}

MODELS = [
    "BiMamba",
    "UniLSTM-proto",
    "BiLSTM-proto",
    "CNN-LSTM-proto",
    "Transformer-proto",
]


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
            marker=dict(
                size=STYLE["ADS-C anchors"]["size"],
                color=STYLE["ADS-C anchors"]["color"],
                symbol=STYLE["ADS-C anchors"]["symbol"],
            ),
            hovertemplate="Anchor<br>Lon=%{x:.3f}<br>Lat=%{y:.3f}<br>Alt=%{z:.1f} m<extra></extra>",
        )
    )

    for model in MODELS:
        fig.add_trace(
            go.Scatter3d(
                x=pd.to_numeric(df[f"{model}_pred_lon"], errors="coerce"),
                y=pd.to_numeric(df[f"{model}_pred_lat"], errors="coerce"),
                z=pd.to_numeric(df[f"{model}_pred_alt"], errors="coerce"),
                mode="lines",
                name=model,
                line=dict(
                    color=STYLE[model]["color"],
                    dash=STYLE[model]["dash"],
                    width=STYLE[model]["width"],
                ),
                hovertemplate=(
                    f"{model}<br>"
                    "Lon=%{x:.3f}<br>"
                    "Lat=%{y:.3f}<br>"
                    "Alt=%{z:.1f} m<extra></extra>"
                ),
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
            domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),
            xaxis=dict(showbackground=True, backgroundcolor="white", gridcolor="#d9d9d9", zerolinecolor="#d9d9d9"),
            yaxis=dict(showbackground=True, backgroundcolor="white", gridcolor="#d9d9d9", zerolinecolor="#d9d9d9"),
            zaxis=dict(
                showbackground=True,
                backgroundcolor="white",
                gridcolor="#d9d9d9",
                zerolinecolor="#d9d9d9",
                tickformat=".0f",
                exponentformat="none",
            ),
            camera=dict(eye=dict(x=1.55, y=-1.65, z=0.8)),
        ),
    )
    return fig


def _write_index(case_rows: list[dict], out_dir: Path) -> None:
    cards = []
    for row in case_rows:
        pair_id = row["pair_id"]
        html_name = f"{pair_id}/interactive_3d_{pair_id}.html"
        cards.append(
            f"""
            <li>
              <a href="{html_name}">{pair_id}</a>
              <span> anchors={row['anchor_count']}, minutes={row['minutes']}</span>
            </li>
            """
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Real ADS-C Interactive 3D Recovery Compare</title>
  <style>
    body {{ font-family: "Times New Roman", SimSun, serif; margin: 24px; background: #fff; color: #111; }}
    h1 {{ margin-bottom: 8px; }}
    p {{ margin-top: 0; color: #444; }}
    ul {{ line-height: 1.8; }}
    a {{ color: #0b57d0; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1>Real ADS-C Interactive 3D Recovery Comparison</h1>
  <p>Click a case below to open an interactive 3D trajectory page. You can rotate, zoom, and pan directly in the browser.</p>
  <ul>
    {''.join(cards)}
  </ul>
</body>
</html>"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-dir",
        default="outputs/runs/real_adsc_anchor_only_current_models_20260531",
    )
    ap.add_argument("--pair-ids", default="", help="Comma-separated pair_ids. Empty means all.")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    if args.pair_ids.strip():
        wanted = {x.strip() for x in args.pair_ids.split(",") if x.strip()}
        csvs = [base_dir / pid / "recovered_minute_compare_current_models.csv" for pid in sorted(wanted)]
    else:
        csvs = sorted(base_dir.glob("*/recovered_minute_compare_current_models.csv"))

    case_rows: list[dict] = []
    for csv_path in csvs:
        if not csv_path.exists():
            continue
        pair_id = csv_path.parent.name
        df = pd.read_csv(csv_path)
        fig = _build_figure(df, pair_id)
        html = plot(fig, include_plotlyjs=True, output_type="div")
        page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{pair_id} - Interactive 3D Recovery</title>
  <style>
    body {{ font-family: "Times New Roman", SimSun, serif; margin: 0; background: #fff; }}
    .wrap {{ padding: 8px 10px 10px; }}
    .meta {{ color: #444; margin: 4px 0 12px; }}
    a {{ color: #0b57d0; text-decoration: none; }}
    .plot-wrap {{ width: 100%; min-height: 92vh; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div><a href="../index.html">Back to index</a></div>
    <div class="meta">pair_id={pair_id}, anchors={int(df['is_adsc_anchor'].sum())}, minutes={len(df)}</div>
    <div class="plot-wrap">{html}</div>
  </div>
</body>
</html>"""
        out_html = csv_path.parent / f"interactive_3d_{pair_id}.html"
        out_html.write_text(page, encoding="utf-8")
        case_rows.append(
            {
                "pair_id": pair_id,
                "anchor_count": int(df["is_adsc_anchor"].sum()),
                "minutes": int(len(df)),
            }
        )

    _write_index(case_rows, base_dir)
    print(base_dir / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
