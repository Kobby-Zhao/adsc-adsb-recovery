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
    _model_specs,
    _prepare_dataset,
    _run_model_for_samples,
)
from src.training.utils import load_config, set_seed  # noqa: E402


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _find_gap_runs(obs_mask: np.ndarray) -> list[tuple[int, int]]:
    obs = np.asarray(obs_mask, dtype=float) > 0.5
    runs = []
    anchors = np.where(obs)[0]
    for left, right in zip(anchors[:-1], anchors[1:]):
        if right - left > 1:
            runs.append((int(left), int(right)))
    return runs


def _max_freeze_run(lat: np.ndarray, lon: np.ndarray, alt: np.ndarray, tol: float = 1e-8) -> int:
    if len(lat) < 2:
        return 0
    same = (
        np.isfinite(lat[1:])
        & np.isfinite(lat[:-1])
        & np.isfinite(lon[1:])
        & np.isfinite(lon[:-1])
        & np.isfinite(alt[1:])
        & np.isfinite(alt[:-1])
        & (np.abs(lat[1:] - lat[:-1]) <= tol)
        & (np.abs(lon[1:] - lon[:-1]) <= tol)
        & (np.abs(alt[1:] - alt[:-1]) <= 1e-6)
    )
    best = cur = 0
    for s in same:
        if bool(s):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best + 1) if best else 0


def _select_sample_and_gap(
    ds,
    min_gap: int = 40,
    max_freeze: int = 3,
    min_anchor_alt: float = 8000.0,
    min_alt_range: float = 20.0,
    max_alt_range: float = 150.0,
    max_truth_step: float = 30.0,
) -> tuple[str, tuple[int, int], pd.DataFrame]:
    rows = []
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
            alt_delta = float(target[right, 2] - target[left, 2])
            rows.append(
                {
                    "sample_id": sid,
                    "flight_id": sample["flight_id"],
                    "left_idx": left,
                    "right_idx": right,
                    "gap_len": gap_len,
                    "max_freeze_run": freeze,
                    "alt_range_m": alt_range,
                    "max_truth_step_m": max_step,
                    "anchor_delta_m": alt_delta,
                    "score": gap_len + 0.05 * alt_range - 10.0 * freeze,
                }
            )
    if not rows:
        raise RuntimeError(
            "No sample gap found with "
            f"gap_len >= {min_gap}, max_freeze <= {max_freeze}, "
            f"min_anchor_alt >= {min_anchor_alt}, and "
            f"{min_alt_range} <= alt_range <= {max_alt_range}, "
            f"max_truth_step <= {max_truth_step}."
        )
    cand = pd.DataFrame(rows).sort_values(["score", "gap_len", "alt_range_m"], ascending=False).reset_index(drop=True)
    pick = cand.iloc[0]
    return str(pick["sample_id"]), (int(pick["left_idx"]), int(pick["right_idx"])), cand


def _abs_error_table(sid: str, sample, gap: tuple[int, int], backbone: dict, bilstm: dict) -> pd.DataFrame:
    left, right = gap
    idx = np.arange(left, right + 1)
    target = sample["target_pos"].numpy()
    obs = sample["obs_mask"].numpy()
    rel_min = idx - left
    table = pd.DataFrame(
        {
            "sample_id": sid,
            "flight_id": sample["flight_id"],
            "minute_index": idx,
            "gap_relative_min": rel_min,
            "is_anchor": (obs[idx] > 0.5).astype(int),
            "adsb_alt_m": target[idx, 2],
            "Backbone_forward_alt_m": backbone["mu_f"][idx, 2],
            "Backbone_backward_alt_m": backbone["mu_b"][idx, 2],
            "Backbone_fusion_alt_m": backbone["final"][idx, 2],
            "BiLSTM_forward_alt_m": bilstm["mu_f"][idx, 2],
            "BiLSTM_backward_alt_m": bilstm["mu_b"][idx, 2],
            "BiLSTM_fusion_alt_m": bilstm["final"][idx, 2],
            "Backbone_forward_weight": backbone["weights"][idx, 0],
            "Backbone_backward_weight": backbone["weights"][idx, 1],
            "BiLSTM_forward_weight": bilstm["weights"][idx, 0],
            "BiLSTM_backward_weight": bilstm["weights"][idx, 1],
        }
    )
    for col in [
        "Backbone_forward",
        "Backbone_backward",
        "Backbone_fusion",
        "BiLSTM_forward",
        "BiLSTM_backward",
        "BiLSTM_fusion",
    ]:
        table[f"{col}_abs_err_m"] = (table[f"{col}_alt_m"] - table["adsb_alt_m"]).abs()
    return table


def _summary(table: pd.DataFrame) -> pd.DataFrame:
    internal = table[table["is_anchor"].eq(0)].copy()
    rows = []
    for model, prefix in [
        ("Backbone", "Backbone_forward"),
        ("Backbone", "Backbone_backward"),
        ("Backbone", "Backbone_fusion"),
        ("BiLSTM", "BiLSTM_forward"),
        ("BiLSTM", "BiLSTM_backward"),
        ("BiLSTM", "BiLSTM_fusion"),
    ]:
        method = prefix.split("_", 1)[1]
        err = pd.to_numeric(internal[f"{prefix}_abs_err_m"], errors="coerce")
        rows.append(
            {
                "model": model,
                "prediction_method": method,
                "point_count": int(err.notna().sum()),
                "alt_abs_error_mae_m": float(err.mean()),
                "alt_abs_error_median_m": float(err.median()),
                "alt_abs_error_max_m": float(err.max()),
            }
        )
    return pd.DataFrame(rows)


def _plot(table: pd.DataFrame, summary: pd.DataFrame, out_png: Path) -> None:
    x = table["gap_relative_min"].to_numpy(dtype=float)
    anchor = table["is_anchor"].astype(bool).to_numpy()
    fig, ax = plt.subplots(figsize=(13.5, 6.2), facecolor="white")
    ax.plot(x, table["adsb_alt_m"], color="black", lw=2.5, label="ADS-B truth")
    ax.scatter(x[anchor], table.loc[anchor, "adsb_alt_m"], color="black", marker="*", s=160, zorder=6, label="ADS-C anchors")

    styles = [
        ("Backbone_forward_alt_m", "Backbone forward", "#1f77b4", "-", 1.8),
        ("Backbone_backward_alt_m", "Backbone backward", "#1f77b4", "--", 1.8),
        ("Backbone_fusion_alt_m", "Backbone fusion", "#1f77b4", "-.", 2.3),
        ("BiLSTM_forward_alt_m", "BiLSTM forward", "#d62728", "-", 1.8),
        ("BiLSTM_backward_alt_m", "BiLSTM backward", "#d62728", "--", 1.8),
        ("BiLSTM_fusion_alt_m", "BiLSTM fusion", "#d62728", "-.", 2.3),
    ]
    for col, label, color, ls, lw in styles:
        ax.plot(x, table[col], color=color, linestyle=ls, lw=lw, label=label, alpha=0.92)

    ax.set_xlabel("Minutes from left ADS-C anchor")
    ax.set_ylabel("Altitude (m)")
    ax.set_title("Forward / backward / fusion altitude prediction in one ADS-C gap")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3, fontsize=8, frameon=True)
    txt = summary.pivot(index="model", columns="prediction_method", values="alt_abs_error_mae_m").round(2)
    ax.text(
        0.01,
        0.02,
        "Internal-gap mean absolute error (m)\n" + txt.to_string(),
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.78, edgecolor="#999999"),
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default="obscons_gaponly_physical_time_ablation_v1")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--min-gap", type=int, default=40)
    parser.add_argument("--max-freeze", type=int, default=3)
    parser.add_argument("--min-anchor-alt", type=float, default=8000.0)
    parser.add_argument("--min-alt-range", type=float, default=20.0)
    parser.add_argument("--max-alt-range", type=float, default=150.0)
    parser.add_argument("--max-truth-step", type=float, default=30.0)
    parser.add_argument("--out-dir", default="outputs/experiments/obs_conditioned_gaponly/single_gap_forward_backward_fusion_20260519")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    set_seed(42)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = _model_specs(args.run_tag)
    cfg = load_config(str(_resolve(specs["Backbone-only"]["config"])))
    ds = _prepare_dataset(cfg, split_name=args.split)
    sid, gap, candidates = _select_sample_and_gap(
        ds,
        min_gap=args.min_gap,
        max_freeze=args.max_freeze,
        min_anchor_alt=args.min_anchor_alt,
        min_alt_range=args.min_alt_range,
        max_alt_range=args.max_alt_range,
        max_truth_step=args.max_truth_step,
    )
    candidates.to_csv(out_dir / "candidate_gap_selection.csv", index=False, encoding="utf-8-sig")
    sample = {str(s["sample_id"]): s for s in ds.samples}[sid]
    selected_ids = {sid}
    device = torch.device(args.device)
    backbone = _run_model_for_samples(
        "Backbone-only",
        model_specs=specs,
        selected_ids=selected_ids,
        split_name=args.split,
        device=device,
    )[sid]
    bilstm = _run_model_for_samples(
        "BiLSTM-clean",
        model_specs=specs,
        selected_ids=selected_ids,
        split_name=args.split,
        device=device,
    )[sid]
    table = _abs_error_table(sid, sample, gap, backbone, bilstm)
    summary = _summary(table)
    table.to_csv(out_dir / "single_gap_forward_backward_fusion_points.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "single_gap_forward_backward_fusion_abs_error_summary.csv", index=False, encoding="utf-8-sig")
    _plot(table, summary, out_dir / "single_gap_forward_backward_fusion_altitude.png")
    selected = candidates[candidates["sample_id"].astype(str).eq(sid) & candidates["left_idx"].eq(gap[0]) & candidates["right_idx"].eq(gap[1])]
    selected.to_csv(out_dir / "selected_gap_info.csv", index=False, encoding="utf-8-sig")
    print(f"[done] out_dir={out_dir}")
    print(selected.to_string(index=False))
    print(summary.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
