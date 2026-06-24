from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
import re
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.adsb_test_multimodel_overlay import _infer_tracks_for_model


@dataclass
class ModelSpec:
    name: str
    config: str
    checkpoint: str


def _default_models() -> list[ModelSpec]:
    # Unified curriculum setting (same stage3 test protocol) for 6 baselines + ourmethod.
    return [
        ModelSpec(
            name="OurMethod",
            config="configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_proposed_24e.yaml",
            checkpoint="outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_proposed_24e/best.pt",
        ),
        ModelSpec(
            name="BiLSTM_Baseline",
            config="configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_bilstm_baseline_24e.yaml",
            checkpoint="outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_bilstm_baseline_24e/best.pt",
        ),
        ModelSpec(
            name="UniLSTM",
            config="configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_unilstm_baseline_24e.yaml",
            checkpoint="outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_unilstm_baseline_24e/best.pt",
        ),
        ModelSpec(
            name="Transformer",
            config="configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_transformer_baseline_24e.yaml",
            checkpoint="outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_transformer_baseline_24e/best.pt",
        ),
        ModelSpec(
            name="CNN+LSTM",
            config="configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_cnnlstm_baseline_24e.yaml",
            checkpoint="outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_cnnlstm_baseline_24e/best.pt",
        ),
        ModelSpec(
            name="Kalman-Filter",
            config="configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_kalman_filter_baseline_24e.yaml",
            checkpoint="outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_kalman_filter_baseline_24e/best.pt",
        ),
    ]


def _track_key_from_sample_id(sample_id: str) -> str:
    # Keep exactly the same grouping logic as evaluation/replay path.
    sid = str(sample_id)
    sid = sid.split("__seg")[0]
    if sid.endswith("_tail"):
        return sid[:-5]
    m = re.match(r"^(.*)_\d+$", sid)
    if m:
        return m.group(1)
    return sid


def _select_typical_track_keys(metrics_csv: Path, max_tracks: int) -> list[str]:
    df = pd.read_csv(metrics_csv)
    if "sample_id" not in df.columns:
        raise RuntimeError("main_task_metrics_test_per_sample.csv missing sample_id")
    if "gap_alt_rmse" not in df.columns:
        raise RuntimeError("main_task_metrics_test_per_sample.csv missing gap_alt_rmse")
    x = df.copy()
    x["track_key"] = x["sample_id"].astype(str).map(_track_key_from_sample_id)
    g = x.groupby("track_key", as_index=False)["gap_alt_rmse"].mean().sort_values("gap_alt_rmse")
    if len(g) <= max_tracks:
        return g["track_key"].astype(str).tolist()
    idx = np.linspace(0, len(g) - 1, max_tracks).round().astype(int)
    return g.iloc[idx]["track_key"].astype(str).tolist()


def _aligned_pred(merged: dict, ref_times: list[str]) -> np.ndarray:
    mt = [str(z) for z in merged["times"]]
    idx_map = {ts: i for i, ts in enumerate(mt)}
    hit = [idx_map.get(ts, None) for ts in ref_times]
    pred = np.asarray(
        [merged["pred"][h] if h is not None else [np.nan, np.nan, np.nan] for h in hit],
        dtype=np.float64,
    )
    return pred


def _plot_overlay_2d(
    png: Path,
    track_key: str,
    by_model: dict[str, dict],
    plot_mode: str = "altitude",
) -> bool:
    available = {k: v for k, v in by_model.items() if v and v.get("merged", {}).get("ok", False)}
    if not available:
        return False
    ref = next(iter(available.values()))["merged"]
    ref_times = [str(x) for x in ref["times"]]
    target = ref["target"]
    obs = ref["obs_mask"]
    anchor_idx = np.where(obs > 0.5)[0]
    t = np.arange(len(ref_times))

    colors = [
        "#d62728",  # our
        "#2ca02c",
        "#1f77b4",
        "#ff7f0e",
        "#9467bd",
        "#17becf",
        "#8c564b",
    ]
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(available.keys())}

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    axes[0].plot(target[:, 1], target[:, 0], color="#222222", lw=1.6, alpha=0.9, label="GT")
    for name, x in available.items():
        pred = _aligned_pred(x["merged"], ref_times)
        axes[0].plot(pred[:, 1], pred[:, 0], lw=1.8, alpha=0.95, color=color_map[name], label=name)
    if len(anchor_idx) > 0:
        axes[0].scatter(target[anchor_idx, 1], target[anchor_idx, 0], s=20, color="#000000", alpha=0.7, label="Anchor")
    axes[0].set_xlabel("Lon")
    axes[0].set_ylabel("Lat")
    axes[0].set_title("2D Lat/Lon Overlay")
    axes[0].legend(fontsize=8, ncol=2)

    if plot_mode == "abs_error":
        axes[1].axhline(0.0, color="#222222", lw=1.2, alpha=0.7, label="Zero Error")
        for name, x in available.items():
            pred = _aligned_pred(x["merged"], ref_times)
            abs_err = np.abs(pred[:, 2] - target[:, 2])
            axes[1].plot(t, abs_err, lw=1.8, alpha=0.95, color=color_map[name], label=name)
        if len(anchor_idx) > 0:
            axes[1].scatter(
                t[anchor_idx],
                np.zeros(len(anchor_idx), dtype=np.float64),
                s=20,
                color="#000000",
                alpha=0.7,
                label="Anchor",
            )
        axes[1].set_xlabel("Minute Index")
        axes[1].set_ylabel("Absolute Error")
        axes[1].set_title(r"Altitude Absolute Error ($|\hat{y}_t - y_t|$)")
    else:
        axes[1].plot(t, target[:, 2], color="#222222", lw=1.6, alpha=0.9, label="GT Alt")
        for name, x in available.items():
            pred = _aligned_pred(x["merged"], ref_times)
            axes[1].plot(t, pred[:, 2], lw=1.8, alpha=0.95, color=color_map[name], label=name)
        if len(anchor_idx) > 0:
            axes[1].scatter(t[anchor_idx], target[anchor_idx, 2], s=20, color="#000000", alpha=0.7, label="Anchor")
        axes[1].set_xlabel("Minute Index")
        axes[1].set_ylabel("Altitude")
        axes[1].set_title("2D Altitude Overlay")
    axes[1].legend(fontsize=8, ncol=2)

    suffix = "Absolute Error" if plot_mode == "abs_error" else "Recovery"
    fig.suptitle(f"Stage3 Test Typical {suffix} | {track_key}", fontsize=11)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(png, dpi=170)
    plt.close(fig)
    return True


def _plot_overlay_3d(png: Path, track_key: str, by_model: dict[str, dict]) -> bool:
    available = {k: v for k, v in by_model.items() if v and v.get("merged", {}).get("ok", False)}
    if not available:
        return False
    ref = next(iter(available.values()))["merged"]
    ref_times = [str(x) for x in ref["times"]]
    target = ref["target"]
    obs = ref["obs_mask"]
    anchor_idx = np.where(obs > 0.5)[0]

    colors = [
        "#d62728",
        "#2ca02c",
        "#1f77b4",
        "#ff7f0e",
        "#9467bd",
        "#17becf",
        "#8c564b",
    ]
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(available.keys())}

    fig = plt.figure(figsize=(9.6, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(target[:, 1], target[:, 0], target[:, 2], color="#222222", lw=1.7, alpha=0.9, label="GT")
    if len(anchor_idx) > 0:
        ax.scatter(
            target[anchor_idx, 1],
            target[anchor_idx, 0],
            target[anchor_idx, 2],
            s=18,
            c="#000000",
            alpha=0.7,
            label="Anchor",
        )
    for name, x in available.items():
        pred = _aligned_pred(x["merged"], ref_times)
        ax.plot(pred[:, 1], pred[:, 0], pred[:, 2], lw=1.7, alpha=0.95, color=color_map[name], label=name)
    ax.set_xlabel("Lon")
    ax.set_ylabel("Lat")
    ax.set_zlabel("Altitude")
    ax.set_title(f"3D Recovery Overlay | {track_key}")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(png, dpi=170)
    plt.close(fig)
    return True


def main() -> int:
    p = argparse.ArgumentParser("Stage3 test overlay plots for 6 baselines + OurMethod (2D/3D).")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument(
        "--metrics-csv",
        default="outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_proposed_24e/main_task_metrics_test_per_sample.csv",
    )
    p.add_argument("--max-tracks", type=int, default=10)
    p.add_argument("--selected-track-csv", default="", help="Optional CSV with column `track_key` to force plotting set.")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--out-dir", default="outputs/runs/stage3_typical_multimodel_overlay_20260419")
    p.add_argument("--plot-mode", default="altitude", choices=["altitude", "abs_error"])
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_2d = out_dir / "plots_2d_overlay"
    out_3d = out_dir / "plots_3d_overlay"
    out_2d.mkdir(parents=True, exist_ok=True)
    out_3d.mkdir(parents=True, exist_ok=True)

    if str(args.selected_track_csv).strip():
        sel = pd.read_csv(args.selected_track_csv)
        if "track_key" not in sel.columns:
            raise RuntimeError("--selected-track-csv must contain column `track_key`")
        track_keys = sel["track_key"].astype(str).dropna().tolist()
        if int(args.max_tracks) > 0:
            track_keys = track_keys[: int(args.max_tracks)]
    else:
        track_keys = _select_typical_track_keys(Path(args.metrics_csv), int(args.max_tracks))
    track_set = set(track_keys)

    specs = _default_models()
    per_model_track: dict[str, dict] = {}
    for spec in specs:
        print(f"[run] {spec.name}")
        per_model_track[spec.name] = _infer_tracks_for_model(
            spec=spec,
            split=args.split,
            track_keys=track_set,
            batch_size_override=int(args.batch_size),
        )

    rows = []
    for i, tk in enumerate(track_keys, start=1):
        by_model = {name: per_model_track.get(name, {}).get(tk, {}) for name in per_model_track}
        png2d = out_2d / f"{i:02d}_{tk}_overlay2d.png"
        png3d = out_3d / f"{i:02d}_{tk}_overlay3d.png"
        ok2d = _plot_overlay_2d(png2d, tk, by_model, plot_mode=str(args.plot_mode))
        ok3d = _plot_overlay_3d(png3d, tk, by_model)
        rows.append(
            {
                "track_key": tk,
                "plot_mode": str(args.plot_mode),
                "plotted_2d": bool(ok2d),
                "plotted_3d": bool(ok3d),
                "models_ok": int(
                    sum(1 for _name, x in by_model.items() if x and x.get("merged", {}).get("ok", False))
                ),
                "plot_2d": str(png2d),
                "plot_3d": str(png3d),
            }
        )

    pd.DataFrame(rows).to_csv(out_dir / "overlay_plot_index.csv", index=False)
    pd.DataFrame({"track_key": track_keys}).to_csv(out_dir / "selected_track_keys.csv", index=False)
    print(f"[done] out_dir={out_dir}")
    print(f"[done] plots_2d={out_2d}")
    print(f"[done] plots_3d={out_3d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
