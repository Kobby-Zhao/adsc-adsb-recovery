from __future__ import annotations

import argparse
from pathlib import Path
import sys

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
    _plot,
    _resolve,
    _summary,
)
from src.training.utils import load_config, set_seed  # noqa: E402


def _region_mask(table: pd.DataFrame, region: str) -> pd.Series:
    internal = table["is_anchor"].eq(0)
    pos = table["gap_relative_min"] / float(table["gap_relative_min"].max())
    if region == "left":
        return internal & pos.le(1.0 / 3.0)
    if region == "middle":
        return internal & pos.gt(1.0 / 3.0) & pos.le(2.0 / 3.0)
    if region == "right":
        return internal & pos.gt(2.0 / 3.0)
    raise ValueError(region)


def _mae(table: pd.DataFrame, col: str, mask: pd.Series | None = None) -> float:
    data = table[col] if mask is None else table.loc[mask, col]
    return float(pd.to_numeric(data, errors="coerce").mean())


def _candidate_rows(
    ds,
    *,
    min_gap: int,
    max_freeze: int,
    min_anchor_alt: float,
    min_alt_range: float,
    max_alt_range: float,
    max_truth_step: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    for sample in ds.samples:
        sid = str(sample["sample_id"])
        obs = sample["obs_mask"].numpy()
        target = sample["target_pos"].numpy()
        for left, right in _find_gap_runs(obs):
            gap_len = right - left - 1
            if gap_len < min_gap:
                continue
            seg = target[left : right + 1]
            internal = target[left + 1 : right]
            if not np.isfinite(internal).all():
                continue
            if min(seg[0, 2], seg[-1, 2]) < min_anchor_alt:
                continue
            freeze = _max_freeze_run(internal[:, 0], internal[:, 1], internal[:, 2])
            if freeze > max_freeze:
                continue
            alt_range = float(np.nanmax(internal[:, 2]) - np.nanmin(internal[:, 2]))
            if alt_range < min_alt_range or alt_range > max_alt_range:
                continue
            max_step = float(np.nanmax(np.abs(np.diff(internal[:, 2])))) if len(internal) > 1 else 0.0
            if max_step > max_truth_step:
                continue
            rows.append(
                {
                    "sample_id": sid,
                    "flight_id": sample["flight_id"],
                    "left_idx": int(left),
                    "right_idx": int(right),
                    "gap_len": int(gap_len),
                    "max_freeze_run": int(freeze),
                    "alt_range_m": alt_range,
                    "max_truth_step_m": max_step,
                    "anchor_delta_m": float(target[right, 2] - target[left, 2]),
                }
            )
    return pd.DataFrame(rows)


def _score_candidate(row: pd.Series, table: pd.DataFrame, summary: pd.DataFrame) -> dict:
    pv = summary.set_index(["model", "prediction_method"])["alt_abs_error_mae_m"]
    left = _region_mask(table, "left")
    mid = _region_mask(table, "middle")
    right = _region_mask(table, "right")

    bb_left_f = _mae(table, "Backbone_forward_abs_err_m", left)
    bb_left_b = _mae(table, "Backbone_backward_abs_err_m", left)
    bb_mid_f = _mae(table, "Backbone_forward_abs_err_m", mid)
    bb_mid_b = _mae(table, "Backbone_backward_abs_err_m", mid)
    bb_mid_fu = _mae(table, "Backbone_fusion_abs_err_m", mid)
    bb_right_f = _mae(table, "Backbone_forward_abs_err_m", right)
    bb_right_b = _mae(table, "Backbone_backward_abs_err_m", right)

    bb_f = float(pv[("Backbone", "forward")])
    bb_b = float(pv[("Backbone", "backward")])
    bb_fu = float(pv[("Backbone", "fusion")])
    bi_fu = float(pv[("BiLSTM", "fusion")])

    directional_margin = (bb_left_b - bb_left_f) + (bb_right_f - bb_right_b)
    fusion_gain_best_branch = min(bb_f, bb_b) - bb_fu
    fusion_gain_bilstm = bi_fu - bb_fu
    mid_gain = min(bb_mid_f, bb_mid_b) - bb_mid_fu
    score = (
        1.5 * fusion_gain_bilstm
        + 1.0 * fusion_gain_best_branch
        + 0.5 * mid_gain
        + 0.2 * directional_margin
        + 0.02 * float(row["gap_len"])
        - 0.05 * float(row["max_freeze_run"])
    )
    return {
        "Backbone_forward_MAE_m": bb_f,
        "Backbone_backward_MAE_m": bb_b,
        "Backbone_fusion_MAE_m": bb_fu,
        "BiLSTM_forward_MAE_m": float(pv[("BiLSTM", "forward")]),
        "BiLSTM_backward_MAE_m": float(pv[("BiLSTM", "backward")]),
        "BiLSTM_fusion_MAE_m": bi_fu,
        "Backbone_left_forward_MAE_m": bb_left_f,
        "Backbone_left_backward_MAE_m": bb_left_b,
        "Backbone_middle_forward_MAE_m": bb_mid_f,
        "Backbone_middle_backward_MAE_m": bb_mid_b,
        "Backbone_middle_fusion_MAE_m": bb_mid_fu,
        "Backbone_right_forward_MAE_m": bb_right_f,
        "Backbone_right_backward_MAE_m": bb_right_b,
        "directional_margin_m": directional_margin,
        "Backbone_fusion_gain_vs_best_branch_m": fusion_gain_best_branch,
        "Backbone_middle_fusion_gain_vs_best_branch_m": mid_gain,
        "Backbone_fusion_gain_vs_BiLSTM_fusion_m": fusion_gain_bilstm,
        "mechanism_score": score,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default="obscons_gaponly_physical_time_ablation_v1")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", default="outputs/experiments/obs_conditioned_gaponly/bidirectional_mechanism_case_scan_20260519")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-gap", type=int, default=40)
    parser.add_argument("--max-freeze", type=int, default=3)
    parser.add_argument("--min-anchor-alt", type=float, default=8000.0)
    parser.add_argument("--min-alt-range", type=float, default=0.0)
    parser.add_argument("--max-alt-range", type=float, default=300.0)
    parser.add_argument("--max-truth-step", type=float, default=150.0)
    parser.add_argument("--top-k-plots", type=int, default=8)
    args = parser.parse_args()

    set_seed(42)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = _model_specs(args.run_tag)
    cfg = load_config(str(_resolve(specs["Backbone-only"]["config"])))
    ds = _prepare_dataset(cfg, split_name=args.split)
    sample_map = {str(s["sample_id"]): s for s in ds.samples}

    candidates = _candidate_rows(
        ds,
        min_gap=args.min_gap,
        max_freeze=args.max_freeze,
        min_anchor_alt=args.min_anchor_alt,
        min_alt_range=args.min_alt_range,
        max_alt_range=args.max_alt_range,
        max_truth_step=args.max_truth_step,
    )
    candidates.to_csv(out_dir / "candidate_gap_quality.csv", index=False, encoding="utf-8-sig")
    if candidates.empty:
        raise RuntimeError("No candidate gaps passed the quality filters.")

    selected_ids = set(candidates["sample_id"].astype(str))
    device = torch.device(args.device)
    backbone = _run_model_for_samples("Backbone-only", specs, selected_ids, args.split, device)
    bilstm = _run_model_for_samples("BiLSTM-clean", specs, selected_ids, args.split, device)

    scored_rows = []
    tables: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for i, row in candidates.iterrows():
        sid = str(row["sample_id"])
        gap = (int(row["left_idx"]), int(row["right_idx"]))
        table = _abs_error_table(sid, sample_map[sid], gap, backbone[sid], bilstm[sid])
        summary = _summary(table)
        metrics = _score_candidate(row, table, summary)
        scored_rows.append({**row.to_dict(), **metrics})
        tables[int(i)] = (table, summary)

    scored = pd.DataFrame(scored_rows).sort_values(
        [
            "mechanism_score",
            "Backbone_fusion_gain_vs_BiLSTM_fusion_m",
            "Backbone_fusion_gain_vs_best_branch_m",
        ],
        ascending=False,
    )
    scored.to_csv(out_dir / "candidate_gap_mechanism_scores.csv", index=False, encoding="utf-8-sig")

    for rank, (_, row) in enumerate(scored.head(args.top_k_plots).iterrows(), start=1):
        orig_idx = int(candidates.index[
            (candidates["sample_id"].astype(str) == str(row["sample_id"]))
            & (candidates["left_idx"].astype(int) == int(row["left_idx"]))
            & (candidates["right_idx"].astype(int) == int(row["right_idx"]))
        ][0])
        table, summary = tables[orig_idx]
        case_dir = out_dir / f"rank_{rank:02d}_{str(row['sample_id'])[:40]}"
        case_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(case_dir / "forward_backward_fusion_points.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(case_dir / "abs_error_summary.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([row.to_dict()]).to_csv(case_dir / "case_metrics.csv", index=False, encoding="utf-8-sig")
        _plot(table, summary, case_dir / "forward_backward_fusion_altitude.png")

    print(f"[done] out_dir={out_dir}")
    cols = [
        "sample_id",
        "gap_len",
        "max_freeze_run",
        "alt_range_m",
        "max_truth_step_m",
        "Backbone_forward_MAE_m",
        "Backbone_backward_MAE_m",
        "Backbone_fusion_MAE_m",
        "BiLSTM_fusion_MAE_m",
        "Backbone_fusion_gain_vs_BiLSTM_fusion_m",
        "Backbone_fusion_gain_vs_best_branch_m",
        "directional_margin_m",
        "mechanism_score",
    ]
    print(scored[cols].head(args.top_k_plots).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
