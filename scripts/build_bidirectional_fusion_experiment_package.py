from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

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
    _model_specs,
    _prepare_dataset,
    _run_model_for_samples,
)
from scripts.plot_bidirectional_gap_forward_backward_fusion import (  # noqa: E402
    _abs_error_table,
    _find_gap_runs,
    _max_freeze_run,
    _resolve,
    _summary,
)
from src.training.utils import load_config, set_seed  # noqa: E402


def _region(pos: float) -> str:
    if pos <= 1.0 / 3.0:
        return "left"
    if pos <= 2.0 / 3.0:
        return "middle"
    return "right"


def _rmse(x: pd.Series) -> float:
    v = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.sqrt(np.mean(np.square(v)))) if len(v) else float("nan")


def _quality_rows(ds) -> pd.DataFrame:
    rows: list[dict] = []
    for sample in ds.samples:
        sid = str(sample["sample_id"])
        obs = sample["obs_mask"].numpy()
        target = sample["target_pos"].numpy()
        anchor_count = int(np.sum(obs > 0.5))
        for gap_id, (left, right) in enumerate(_find_gap_runs(obs)):
            internal = target[left + 1 : right]
            gap_len = int(right - left - 1)
            max_freeze = _max_freeze_run(internal[:, 0], internal[:, 1], internal[:, 2]) if len(internal) else 0
            alt_range = float(np.nanmax(internal[:, 2]) - np.nanmin(internal[:, 2])) if len(internal) else 0.0
            max_step = float(np.nanmax(np.abs(np.diff(internal[:, 2])))) if len(internal) > 1 else 0.0
            rows.append(
                {
                    "sample_id": sid,
                    "flight_id": sample["flight_id"],
                    "gap_id": gap_id,
                    "left_idx": int(left),
                    "right_idx": int(right),
                    "gap_len": gap_len,
                    "anchor_count": anchor_count,
                    "max_freeze_run": int(max_freeze),
                    "alt_range_m": alt_range,
                    "max_truth_step_m": max_step,
                    "left_anchor_alt_m": float(target[left, 2]),
                    "right_anchor_alt_m": float(target[right, 2]),
                    "anchor_delta_m": float(target[right, 2] - target[left, 2]),
                }
            )
    return pd.DataFrame(rows)


def _build_point_table(ds, results: dict[str, dict], gap_meta: pd.DataFrame) -> pd.DataFrame:
    meta_by_key = {
        (str(r.sample_id), int(r.left_idx), int(r.right_idx)): r
        for r in gap_meta.itertuples(index=False)
    }
    rows: list[dict] = []
    for sample in ds.samples:
        sid = str(sample["sample_id"])
        obs = sample["obs_mask"].numpy()
        target = sample["target_pos"].numpy()
        b = results["Backbone-only"].get(sid)
        a3 = results["Ours-A3"].get(sid)
        bi = results["BiLSTM-clean"].get(sid)
        if b is None or a3 is None or bi is None:
            continue
        for left, right in _find_gap_runs(obs):
            meta = meta_by_key[(sid, int(left), int(right))]
            denom = float(right - left)
            for i in range(left + 1, right):
                pos = (i - left) / denom
                truth_alt = float(target[i, 2])
                rows.append(
                    {
                        "sample_id": sid,
                        "flight_id": sample["flight_id"],
                        "gap_id": int(meta.gap_id),
                        "minute_index": int(i),
                        "left_idx": int(left),
                        "right_idx": int(right),
                        "gap_len": int(meta.gap_len),
                        "anchor_count": int(meta.anchor_count),
                        "max_freeze_run": int(meta.max_freeze_run),
                        "alt_range_m": float(meta.alt_range_m),
                        "max_truth_step_m": float(meta.max_truth_step_m),
                        "gap_pos_ratio": float(pos),
                        "gap_region": _region(float(pos)),
                        "target_alt_m": truth_alt,
                        "Backbone_forward_abs_err_m": abs(float(b["mu_f"][i, 2]) - truth_alt),
                        "Backbone_backward_abs_err_m": abs(float(b["mu_b"][i, 2]) - truth_alt),
                        "Backbone_fusion_abs_err_m": abs(float(b["final"][i, 2]) - truth_alt),
                        "Ours_A3_abs_err_m": abs(float(a3["final"][i, 2]) - truth_alt),
                        "BiLSTM_fusion_abs_err_m": abs(float(bi["final"][i, 2]) - truth_alt),
                        "Backbone_forward_weight": float(b["weights"][i, 0]),
                        "Backbone_backward_weight": float(b["weights"][i, 1]),
                        "Ours_A3_forward_weight": float(a3["weights"][i, 0]),
                        "BiLSTM_forward_weight": float(bi["weights"][i, 0]),
                    }
                )
    return pd.DataFrame(rows)


def _summaries(points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bins = np.linspace(0.0, 1.0, 11)
    labels = [f"{bins[i]:.1f}-{bins[i + 1]:.1f}" for i in range(len(bins) - 1)]
    binned = points.copy()
    binned["gap_pos_bin"] = pd.cut(binned["gap_pos_ratio"], bins=bins, labels=labels, include_lowest=True)
    weight_bins = (
        binned.groupby("gap_pos_bin", observed=False)
        .agg(
            point_count=("sample_id", "size"),
            Backbone_forward_weight=("Backbone_forward_weight", "mean"),
            Backbone_backward_weight=("Backbone_backward_weight", "mean"),
            Ours_A3_forward_weight=("Ours_A3_forward_weight", "mean"),
            BiLSTM_forward_weight=("BiLSTM_forward_weight", "mean"),
        )
        .reset_index()
    )

    region_rows = []
    for region, g in points.groupby("gap_region", sort=False):
        region_rows.append(
            {
                "gap_region": region,
                "point_count": int(len(g)),
                "Backbone_forward_MAE_m": float(g["Backbone_forward_abs_err_m"].mean()),
                "Backbone_backward_MAE_m": float(g["Backbone_backward_abs_err_m"].mean()),
                "Backbone_fusion_MAE_m": float(g["Backbone_fusion_abs_err_m"].mean()),
                "BiLSTM_fusion_MAE_m": float(g["BiLSTM_fusion_abs_err_m"].mean()),
                "Ours_A3_MAE_m": float(g["Ours_A3_abs_err_m"].mean()),
            }
        )
    region_summary = pd.DataFrame(region_rows)
    order = {"left": 0, "middle": 1, "right": 2}
    region_summary = region_summary.sort_values("gap_region", key=lambda s: s.map(order)).reset_index(drop=True)

    subsets = {
        "All": points,
        "Long-gap": points[points["gap_len"].ge(40)],
        "Few-anchor": points[points["anchor_count"].le(4)],
        "ADS-C-like": points[points["gap_len"].ge(40) & points["anchor_count"].le(4)],
        "Clean long-gap": points[
            points["gap_len"].ge(40) & points["max_freeze_run"].le(3) & points["max_truth_step_m"].le(150)
        ],
    }
    subset_rows = []
    for name, g in subsets.items():
        if g.empty:
            continue
        for model, col in [
            ("Backbone-forward", "Backbone_forward_abs_err_m"),
            ("Backbone-backward", "Backbone_backward_abs_err_m"),
            ("Backbone-fusion", "Backbone_fusion_abs_err_m"),
            ("BiLSTM-fusion", "BiLSTM_fusion_abs_err_m"),
            ("Ours-A3", "Ours_A3_abs_err_m"),
        ]:
            subset_rows.append(
                {
                    "subset": name,
                    "model": model,
                    "point_count": int(len(g)),
                    "sample_count": int(g["sample_id"].nunique()),
                    "gap_count": int(g[["sample_id", "gap_id"]].drop_duplicates().shape[0]),
                    "alt_MAE_m": float(g[col].mean()),
                    "alt_RMSE_m": _rmse(g[col]),
                }
            )
    subset_summary = pd.DataFrame(subset_rows)
    return weight_bins, region_summary, subset_summary


def _plot_weight(weight_bins: pd.DataFrame, out: Path) -> None:
    x = np.arange(len(weight_bins))
    labels = weight_bins["gap_pos_bin"].astype(str).tolist()
    fig, ax = plt.subplots(figsize=(10.5, 4.6), facecolor="white")
    ax.plot(x, weight_bins["Backbone_forward_weight"], marker="o", lw=2.4, label="Backbone forward weight")
    ax.plot(x, weight_bins["Backbone_backward_weight"], marker="o", lw=2.4, label="Backbone backward weight")
    ax.plot(x, weight_bins["Ours_A3_forward_weight"], marker="s", lw=2.0, label="Ours-A3 forward weight")
    ax.plot(x, weight_bins["BiLSTM_forward_weight"], marker="^", lw=1.8, ls="--", label="BiLSTM forward weight")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Normalized gap position")
    ax.set_ylabel("Mean fusion weight")
    ax.set_title("Fusion weight vs. position inside ADS-C gap")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _plot_region(region: pd.DataFrame, out: Path) -> None:
    regions = region["gap_region"].tolist()
    x = np.arange(len(regions))
    width = 0.17
    fig, ax = plt.subplots(figsize=(10.0, 4.8), facecolor="white")
    for offset, col, label in [
        (-2, "Backbone_forward_MAE_m", "Backbone forward"),
        (-1, "Backbone_backward_MAE_m", "Backbone backward"),
        (0, "Backbone_fusion_MAE_m", "Backbone fusion"),
        (1, "BiLSTM_fusion_MAE_m", "BiLSTM fusion"),
        (2, "Ours_A3_MAE_m", "Ours-A3"),
    ]:
        ax.bar(x + offset * width, region[col], width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    ax.set_xlabel("Gap region")
    ax.set_ylabel("Altitude MAE (m)")
    ax.set_title("Left/middle/right altitude error")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _plot_subset(subset: pd.DataFrame, out: Path) -> None:
    use = subset[subset["model"].isin(["Backbone-fusion", "BiLSTM-fusion", "Ours-A3"])].copy()
    subsets = ["All", "Long-gap", "Few-anchor", "ADS-C-like", "Clean long-gap"]
    models = ["Backbone-fusion", "BiLSTM-fusion", "Ours-A3"]
    x = np.arange(len(subsets))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11.0, 4.8), facecolor="white")
    for j, model in enumerate(models):
        vals = []
        for subset_name in subsets:
            r = use[(use["subset"].eq(subset_name)) & (use["model"].eq(model))]
            vals.append(float(r["alt_MAE_m"].iloc[0]) if not r.empty else np.nan)
        ax.bar(x + (j - 1) * width, vals, width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(subsets, rotation=20, ha="right")
    ax.set_ylabel("Altitude MAE (m)")
    ax.set_title("Altitude error on ADS-C-like difficult subsets")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _plot_case(table: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    x = table["gap_relative_min"].to_numpy(dtype=float)
    anchor = table["is_anchor"].astype(bool).to_numpy()

    fig, ax = plt.subplots(figsize=(12.2, 5.2), facecolor="white")
    ax.plot(x, table["adsb_alt_m"], color="black", lw=2.4, label="ADS-B truth")
    ax.scatter(x[anchor], table.loc[anchor, "adsb_alt_m"], color="black", marker="*", s=150, zorder=6, label="ADS-C anchors")
    ax.plot(x, table["Backbone_forward_alt_m"], color="#8ecae6", lw=1.5, ls="--", label="Backbone forward")
    ax.plot(x, table["Backbone_backward_alt_m"], color="#219ebc", lw=1.5, ls="--", label="Backbone backward")
    ax.plot(x, table["Backbone_fusion_alt_m"], color="#023047", lw=2.5, label="Backbone fusion")
    ax.plot(x, table["BiLSTM_fusion_alt_m"], color="#d62828", lw=2.1, label="BiLSTM fusion")
    ax.set_xlabel("Minutes from left ADS-C anchor")
    ax.set_ylabel("Altitude (m)")
    ax.set_title("Case visualization: bidirectional altitude recovery")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out_dir / "case_altitude_curve.png", dpi=200)
    plt.close(fig)

    internal = table[table["is_anchor"].eq(0)].copy()
    fig, ax = plt.subplots(figsize=(12.2, 5.0), facecolor="white")
    for col, label, color, ls in [
        ("Backbone_forward_abs_err_m", "Backbone forward", "#8ecae6", "--"),
        ("Backbone_backward_abs_err_m", "Backbone backward", "#219ebc", "--"),
        ("Backbone_fusion_abs_err_m", "Backbone fusion", "#023047", "-"),
        ("BiLSTM_fusion_abs_err_m", "BiLSTM fusion", "#d62828", "-"),
    ]:
        ax.plot(internal["gap_relative_min"], internal[col], lw=2.0, ls=ls, color=color, label=label)
    ax.set_xlabel("Minutes from left ADS-C anchor")
    ax.set_ylabel("Absolute altitude error (m)")
    ax.set_title("Case visualization: absolute error inside ADS-C gap")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "case_absolute_error_curve.png", dpi=200)
    plt.close(fig)


def _choose_case(ds, backbone: dict, bilstm: dict, out_dir: Path) -> None:
    sample_map = {str(s["sample_id"]): s for s in ds.samples}
    preferred = [
        ("2024-10-01-AMX039-17277871700-0d0ed6_a0_seg1_0", 30, 85),
        ("2024-11-01-N249QS-17304775050-a252dd_a1_seg0_0", 0, 46),
        ("2024-11-01-QTR756-17304227130-06a11e_a1_seg3_0", 0, 119),
    ]
    for sid, left, right in preferred:
        if sid in sample_map and sid in backbone and sid in bilstm:
            table = _abs_error_table(sid, sample_map[sid], (left, right), backbone[sid], bilstm[sid])
            summary = _summary(table)
            case_dir = out_dir / "case_visualization"
            case_dir.mkdir(parents=True, exist_ok=True)
            table.to_csv(case_dir / "case_forward_backward_fusion_points.csv", index=False, encoding="utf-8-sig")
            summary.to_csv(case_dir / "case_abs_error_summary.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(
                [
                    {
                        "sample_id": sid,
                        "flight_id": sample_map[sid]["flight_id"],
                        "left_idx": left,
                        "right_idx": right,
                        "gap_len": right - left - 1,
                    }
                ]
            ).to_csv(case_dir / "case_info.csv", index=False, encoding="utf-8-sig")
            _plot_case(table, case_dir)
            return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default="obscons_gaponly_physical_time_ablation_v1")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", default="outputs/experiments/obs_conditioned_gaponly/bidirectional_fusion_paper_experiments_20260519")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    set_seed(42)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = _model_specs(args.run_tag)
    cfg = load_config(str(_resolve(specs["Backbone-only"]["config"])))
    ds = _prepare_dataset(cfg, split_name=args.split)
    selected_ids = {str(s["sample_id"]) for s in ds.samples}
    device = torch.device(args.device)

    results = {
        name: _run_model_for_samples(name, specs, selected_ids, args.split, device)
        for name in ["Backbone-only", "Ours-A3", "BiLSTM-clean"]
    }
    gap_meta = _quality_rows(ds)
    points = _build_point_table(ds, results, gap_meta)
    weight_bins, region_summary, subset_summary = _summaries(points)

    gap_meta.to_csv(out_dir / "gap_quality_metadata.csv", index=False, encoding="utf-8-sig")
    points.to_csv(out_dir / "bidirectional_gap_point_errors.csv", index=False, encoding="utf-8-sig")
    weight_bins.to_csv(out_dir / "fusion_weight_vs_gap_position_bins.csv", index=False, encoding="utf-8-sig")
    region_summary.to_csv(out_dir / "left_middle_right_error_summary.csv", index=False, encoding="utf-8-sig")
    subset_summary.to_csv(out_dir / "difficulty_subset_error_summary.csv", index=False, encoding="utf-8-sig")

    _plot_weight(weight_bins, out_dir / "fig_fusion_weight_vs_gap_position.png")
    _plot_region(region_summary, out_dir / "fig_left_middle_right_error.png")
    _plot_subset(subset_summary, out_dir / "fig_difficulty_subset_error.png")
    _choose_case(ds, results["Backbone-only"], results["BiLSTM-clean"], out_dir)

    print(f"[done] out_dir={out_dir}")
    print("\n[left/middle/right error]")
    print(region_summary.round(3).to_string(index=False))
    print("\n[difficulty subsets]")
    print(subset_summary[subset_summary["model"].isin(["Backbone-fusion", "BiLSTM-fusion", "Ours-A3"])].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
