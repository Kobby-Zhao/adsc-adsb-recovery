from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_bidirectional_prediction_mechanism import (  # noqa: E402
    MODEL_KEYS,
    _gap_position_from_obs_mask,
    _model_specs,
    _prepare_dataset,
    _run_model_for_samples,
)
from src.training.utils import load_config, set_seed  # noqa: E402


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _region(pos: float) -> str:
    if not np.isfinite(pos):
        return "anchor"
    if pos <= 1.0 / 3.0:
        return "left"
    if pos <= 2.0 / 3.0:
        return "middle"
    return "right"


def _build_gap_points(run_tag: str, split: str, max_samples: int | None, device: torch.device) -> pd.DataFrame:
    specs = _model_specs(run_tag)
    base_cfg = load_config(str(_resolve(specs["Backbone-only"]["config"])))
    ds = _prepare_dataset(base_cfg, split_name=split)
    samples = ds.samples
    if max_samples is not None and max_samples > 0:
        # Deterministic subset for fast smoke/debug runs.
        samples = samples[: int(max_samples)]
    selected_ids = {str(s["sample_id"]) for s in samples}

    all_results = {
        name: _run_model_for_samples(
            name,
            model_specs=specs,
            selected_ids=selected_ids,
            split_name=split,
            device=device,
        )
        for name in MODEL_KEYS
    }

    rows: list[dict] = []
    for sample in samples:
        sid = str(sample["sample_id"])
        obs = sample["obs_mask"].numpy()
        target = sample["target_pos"].numpy()
        gap_pos = _gap_position_from_obs_mask(obs)
        b = all_results["Backbone-only"].get(sid)
        a3 = all_results["Ours-A3"].get(sid)
        bi = all_results["BiLSTM-clean"].get(sid)
        if b is None or a3 is None or bi is None:
            continue
        for i in range(len(obs)):
            if obs[i] > 0.5 or not np.isfinite(gap_pos[i]):
                continue
            truth_alt = float(target[i, 2])
            row = {
                "sample_id": sid,
                "flight_id": sample["flight_id"],
                "minute_index": int(i),
                "gap_pos_ratio": float(gap_pos[i]),
                "gap_region": _region(float(gap_pos[i])),
                "target_alt_m": truth_alt,
                "Backbone_forward_abs_err_m": abs(float(b["mu_f"][i, 2]) - truth_alt),
                "Backbone_backward_abs_err_m": abs(float(b["mu_b"][i, 2]) - truth_alt),
                "Backbone_fusion_abs_err_m": abs(float(b["final"][i, 2]) - truth_alt),
                "Ours_A3_abs_err_m": abs(float(a3["final"][i, 2]) - truth_alt),
                "BiLSTM_abs_err_m": abs(float(bi["final"][i, 2]) - truth_alt),
                "Backbone_fusion_w_forward": float(b["weights"][i, 0]),
                "Backbone_fusion_w_backward": float(b["weights"][i, 1]),
                "Ours_A3_fusion_w_forward": float(a3["weights"][i, 0]),
                "Ours_A3_fusion_w_backward": float(a3["weights"][i, 1]),
                "BiLSTM_fusion_w_forward": float(bi["weights"][i, 0]),
                "BiLSTM_fusion_w_backward": float(bi["weights"][i, 1]),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def _summarize(points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    region_rows = []
    for region, g in points.groupby("gap_region", sort=False):
        region_rows.append(
            {
                "gap_region": region,
                "point_count": int(len(g)),
                "Backbone_forward_MAE_m": float(g["Backbone_forward_abs_err_m"].mean()),
                "Backbone_backward_MAE_m": float(g["Backbone_backward_abs_err_m"].mean()),
                "Backbone_fusion_MAE_m": float(g["Backbone_fusion_abs_err_m"].mean()),
                "Ours_A3_MAE_m": float(g["Ours_A3_abs_err_m"].mean()),
                "BiLSTM_MAE_m": float(g["BiLSTM_abs_err_m"].mean()),
                "Backbone_forward_weight_mean": float(g["Backbone_fusion_w_forward"].mean()),
                "Ours_A3_forward_weight_mean": float(g["Ours_A3_fusion_w_forward"].mean()),
                "BiLSTM_forward_weight_mean": float(g["BiLSTM_fusion_w_forward"].mean()),
            }
        )
    region_summary = pd.DataFrame(region_rows)
    order = {"left": 0, "middle": 1, "right": 2}
    region_summary = region_summary.sort_values("gap_region", key=lambda s: s.map(order)).reset_index(drop=True)

    bins = np.linspace(0.0, 1.0, 11)
    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)]
    binned = points.copy()
    binned["gap_pos_bin"] = pd.cut(binned["gap_pos_ratio"], bins=bins, labels=labels, include_lowest=True)
    bin_summary = (
        binned.groupby("gap_pos_bin", observed=False)
        .agg(
            point_count=("sample_id", "size"),
            Backbone_forward_MAE_m=("Backbone_forward_abs_err_m", "mean"),
            Backbone_backward_MAE_m=("Backbone_backward_abs_err_m", "mean"),
            Backbone_fusion_MAE_m=("Backbone_fusion_abs_err_m", "mean"),
            Ours_A3_MAE_m=("Ours_A3_abs_err_m", "mean"),
            BiLSTM_MAE_m=("BiLSTM_abs_err_m", "mean"),
            Backbone_forward_weight_mean=("Backbone_fusion_w_forward", "mean"),
            Backbone_backward_weight_mean=("Backbone_fusion_w_backward", "mean"),
            Ours_A3_forward_weight_mean=("Ours_A3_fusion_w_forward", "mean"),
            BiLSTM_forward_weight_mean=("BiLSTM_fusion_w_forward", "mean"),
        )
        .reset_index()
    )
    return region_summary, bin_summary


def _plot(bin_summary: pd.DataFrame, region_summary: pd.DataFrame, out_dir: Path) -> None:
    x = np.arange(len(bin_summary))
    labels = bin_summary["gap_pos_bin"].astype(str).tolist()

    fig, ax = plt.subplots(figsize=(10.5, 4.6), facecolor="white")
    ax.plot(x, bin_summary["Backbone_forward_weight_mean"], marker="o", lw=2, label="Backbone forward weight")
    ax.plot(x, bin_summary["Backbone_backward_weight_mean"], marker="o", lw=2, label="Backbone backward weight")
    ax.plot(x, bin_summary["Ours_A3_forward_weight_mean"], marker="s", lw=2, label="Ours-A3 forward weight")
    ax.plot(x, bin_summary["BiLSTM_forward_weight_mean"], marker="^", lw=1.8, ls="--", label="BiLSTM forward weight")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Normalized gap position")
    ax.set_ylabel("Mean fusion weight")
    ax.set_title("Fusion weight vs gap position")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fusion_weight_vs_gap_position_global.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 4.8), facecolor="white")
    regions = region_summary["gap_region"].tolist()
    width = 0.18
    pos = np.arange(len(regions))
    ax.bar(pos - 1.5 * width, region_summary["Backbone_forward_MAE_m"], width, label="Forward")
    ax.bar(pos - 0.5 * width, region_summary["Backbone_backward_MAE_m"], width, label="Backward")
    ax.bar(pos + 0.5 * width, region_summary["Backbone_fusion_MAE_m"], width, label="Fusion")
    ax.bar(pos + 1.5 * width, region_summary["Ours_A3_MAE_m"], width, label="Ours-A3")
    ax.set_xticks(pos)
    ax.set_xticklabels(regions)
    ax.set_xlabel("Gap region")
    ax.set_ylabel("Altitude MAE (m)")
    ax.set_title("Left/middle/right gap error")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "left_middle_right_gap_error_global.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default="obscons_gaponly_physical_time_ablation_v1")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", default="outputs/experiments/obs_conditioned_gaponly/bidirectional_global_bins_physical_time_20260519")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    set_seed(42)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    max_samples = None if args.max_samples <= 0 else args.max_samples

    points = _build_gap_points(args.run_tag, args.split, max_samples=max_samples, device=device)
    if points.empty:
        raise RuntimeError("No gap points generated.")
    region_summary, bin_summary = _summarize(points)
    points.to_csv(out_dir / "bidirectional_global_gap_points.csv", index=False, encoding="utf-8-sig")
    region_summary.to_csv(out_dir / "left_middle_right_gap_error.csv", index=False, encoding="utf-8-sig")
    bin_summary.to_csv(out_dir / "fusion_weight_vs_gap_position_bins.csv", index=False, encoding="utf-8-sig")
    _plot(bin_summary, region_summary, out_dir)
    print(f"[done] out_dir={out_dir}")
    print(region_summary.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
