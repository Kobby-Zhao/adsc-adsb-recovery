from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train import load_config
from scripts.real_adsc_replay_eval import _predict_on_frame
from scripts.real_adsc_truequal_gapwise_eval import (
    _build_gapwise_segments,
    _compute_structural_metrics,
    _stitch_flight_prediction,
)


@dataclass
class ModelSpec:
    name: str
    config: Path
    checkpoint: Path


DEFAULT_MODELS = [
    (
        "ourmethod",
        "configs/alt_focus/ablation_submodules/ablation_a2_step2.yaml",
        "outputs/experiments/ablation_submodules/ablation_a2_step2_24e/best.pt",
    ),
    (
        "unilstm_baseline",
        "configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_unilstm_baseline_24e.yaml",
        "outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_unilstm_baseline_24e/best.pt",
    ),
    (
        "bilstm_baseline",
        "configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_bilstm_baseline_24e.yaml",
        "outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_bilstm_baseline_24e/best.pt",
    ),
    (
        "cnnlstm_baseline",
        "configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_cnnlstm_baseline_24e.yaml",
        "outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_cnnlstm_baseline_24e/best.pt",
    ),
    (
        "transformer_baseline",
        "configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_transformer_baseline_24e.yaml",
        "outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_transformer_baseline_24e/best.pt",
    ),
    (
        "kalman_filter_baseline",
        "configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_kalman_filter_baseline_24e.yaml",
        "outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_kalman_filter_baseline_24e/best.pt",
    ),
]


MODEL_STYLES = {
    "ourmethod": {"color": "#d62728", "linestyle": "--", "linewidth": 2.4},
    "unilstm_baseline": {"color": "#1f77b4", "linestyle": "-", "linewidth": 1.7},
    "bilstm_baseline": {"color": "#2ca02c", "linestyle": "-", "linewidth": 1.7},
    "cnnlstm_baseline": {"color": "#9467bd", "linestyle": "-", "linewidth": 1.7},
    "transformer_baseline": {"color": "#ff7f0e", "linestyle": "-", "linewidth": 1.9},
    "kalman_filter_baseline": {"color": "#17becf", "linestyle": "-.", "linewidth": 1.6},
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("Real ADS-C gapwise multi-model qualitative compare.")
    ap.add_argument("--samples-parquet", required=True)
    ap.add_argument("--selected-flights-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    return ap


def _resolve_specs() -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for name, cfg, ckpt in DEFAULT_MODELS:
        specs.append(
            ModelSpec(
                name=name,
                config=ROOT / cfg,
                checkpoint=ROOT / ckpt,
            )
        )
    return specs


def _plot_compare(
    base: pd.DataFrame,
    stitched_by_model: dict[str, pd.DataFrame],
    out_png: Path,
) -> None:
    x = base.sort_values("minute_ts").reset_index(drop=True).copy()
    anchor_mask = pd.to_numeric(x["obs_mask"], errors="coerce").fillna(0.0) > 0.5

    fig, ax = plt.subplots(figsize=(10.5, 5.5), facecolor="white")
    ax.set_facecolor("white")

    for model_name, stitched in stitched_by_model.items():
        s = stitched.sort_values("minute_ts").reset_index(drop=True)
        style = MODEL_STYLES.get(model_name, {})
        ax.plot(
            s.index,
            pd.to_numeric(s["pred_alt"], errors="coerce"),
            label=model_name,
            **style,
        )

    ax.scatter(
        x.index[anchor_mask],
        pd.to_numeric(x.loc[anchor_mask, "obs_alt"], errors="coerce"),
        s=32,
        color="#000000",
        label="ADS-C anchors",
        zorder=5,
    )
    fid = str(x["flight_id"].iloc[0])
    ax.set_title(f"{fid} | real ADS-C recovery (anchor-fixed, gapwise, multi-model)")
    ax.set_xlabel("Minute index")
    ax.set_ylabel("Altitude")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _plot_compare_3d(
    base: pd.DataFrame,
    stitched_by_model: dict[str, pd.DataFrame],
    out_png: Path,
) -> None:
    x = base.sort_values("minute_ts").reset_index(drop=True).copy()
    anchor_mask = pd.to_numeric(x["obs_mask"], errors="coerce").fillna(0.0) > 0.5

    fig = plt.figure(figsize=(10.8, 7.4), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    lon = pd.to_numeric(x["obs_lon"], errors="coerce")
    lat = pd.to_numeric(x["obs_lat"], errors="coerce")
    alt = pd.to_numeric(x["obs_alt"], errors="coerce")

    lon_span = float(max(lon.max() - lon.min(), 1e-6))
    lat_span = float(max(lat.max() - lat.min(), 1e-6))
    alt_span = float(max(alt.max() - alt.min(), 1e-6))
    mean_lat = float(lat.mean()) if len(lat) else 0.0
    lon_m = lon_span * 111000.0 * max(0.2, abs(__import__("math").cos(__import__("math").radians(mean_lat))))
    lat_m = lat_span * 111000.0
    z_display = max(alt_span * 0.22, min(lon_m, lat_m) * 0.45, 1.0)

    ax.plot(
        lon,
        lat,
        alt,
        color="#222222",
        lw=1.6,
        alpha=0.85,
        label="Ground Truth",
    )
    ax.scatter(
        pd.to_numeric(x.loc[anchor_mask, "obs_lon"], errors="coerce"),
        pd.to_numeric(x.loc[anchor_mask, "obs_lat"], errors="coerce"),
        pd.to_numeric(x.loc[anchor_mask, "obs_alt"], errors="coerce"),
        s=20,
        color="#000000",
        alpha=0.75,
        label="ADS-C anchors",
    )

    for model_name, stitched in stitched_by_model.items():
        s = stitched.sort_values("minute_ts").reset_index(drop=True)
        style = MODEL_STYLES.get(model_name, {})
        ax.plot(
            pd.to_numeric(s["pred_lon"], errors="coerce"),
            pd.to_numeric(s["pred_lat"], errors="coerce"),
            pd.to_numeric(s["pred_alt"], errors="coerce"),
            label=model_name,
            **style,
        )

    fid = str(x["flight_id"].iloc[0])
    ax.set_title(f"{fid} | real ADS-C recovery (3D gapwise multi-model)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Altitude")
    ax.view_init(elev=20, azim=-122)
    ax.set_proj_type("persp")
    ax.set_box_aspect((max(lon_m, 1.0), max(lat_m, 1.0), z_display))
    ax.grid(True, alpha=0.28)
    ax.xaxis.pane.set_alpha(0.04)
    ax.yaxis.pane.set_alpha(0.04)
    ax.zaxis.pane.set_alpha(0.04)
    ax.legend(fontsize=8, ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_3d_dir = out_dir / "plots_3d"
    plot_3d_dir.mkdir(parents=True, exist_ok=True)

    specs = _resolve_specs()
    samples = pd.read_parquet(args.samples_parquet).copy()
    samples["minute_ts"] = pd.to_datetime(samples["minute_ts"], utc=True)
    selected = pd.read_csv(args.selected_flights_csv)
    flight_ids = selected["flight_id"].astype(str).tolist()

    all_frames: list[pd.DataFrame] = []
    original_by_flight: dict[str, pd.DataFrame] = {}
    for fid in flight_ids:
        g = samples[samples["flight_id"].astype(str) == fid].copy()
        if g.empty:
            continue
        original_by_flight[fid] = g
        frames, _ = _build_gapwise_segments(g)
        all_frames.extend(frames)
    if not all_frames:
        raise RuntimeError("No gapwise real ADS-C segments were constructed.")
    frame_all = pd.concat(all_frames, ignore_index=True)

    stitched_by_model_by_flight: dict[str, dict[str, pd.DataFrame]] = {fid: {} for fid in original_by_flight}
    metrics_rows: list[dict] = []

    for spec in specs:
        cfg = load_config(spec.config)
        pred = _predict_on_frame(cfg=cfg, checkpoint=spec.checkpoint, frame=frame_all, pred_key="pred_pos")
        for fid, base in original_by_flight.items():
            fpred = pred[pred["flight_id"].astype(str) == fid].copy()
            if fpred.empty:
                continue
            stitched = _stitch_flight_prediction(base, fpred)
            stitched_by_model_by_flight[fid][spec.name] = stitched
            m = _compute_structural_metrics(stitched)
            m["flight_id"] = fid
            m["model"] = spec.name
            m["num_minutes"] = int(len(stitched))
            m["anchor_count"] = int((stitched["obs_mask_eval"] > 0.5).sum())
            metrics_rows.append(m)

    for fid, base in original_by_flight.items():
        if not stitched_by_model_by_flight[fid]:
            continue
        _plot_compare(
            base=base,
            stitched_by_model=stitched_by_model_by_flight[fid],
            out_png=plot_dir / f"multi_model_truequal_gapwise_alt_{fid}.png",
        )
        _plot_compare_3d(
            base=base,
            stitched_by_model=stitched_by_model_by_flight[fid],
            out_png=plot_3d_dir / f"multi_model_truequal_gapwise_3d_{fid}.png",
        )

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["flight_id", "model"]).reset_index(drop=True)
    metrics_df.to_csv(out_dir / "multi_model_truequal_gapwise_structural_metrics.csv", index=False)

    summary = {
        "num_flights": int(len(original_by_flight)),
        "num_models": int(len(specs)),
        "models": [s.name for s in specs],
        "samples_parquet": str(Path(args.samples_parquet)),
        "selected_flights_csv": str(Path(args.selected_flights_csv)),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
