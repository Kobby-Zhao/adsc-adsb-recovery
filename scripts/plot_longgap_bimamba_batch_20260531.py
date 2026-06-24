from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from scripts.analyze_bidirectional_prediction_mechanism import _prepare_dataset, _run_model_for_samples  # noqa: E402
from src.training.utils import load_config, set_seed  # noqa: E402


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _truth_for_selected(ds, selected_ids: set[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for sample in ds.samples:
        sid = str(sample["sample_id"])
        if sid not in selected_ids:
            continue
        obs = sample["obs_mask"].numpy()
        target = sample["target_pos"].numpy()
        idx = np.arange(len(obs))
        gap_idx = np.where(obs <= 0.5)[0]
        last_gap_idx = int(gap_idx[-1]) if len(gap_idx) else int(len(obs) - 1)
        out[sid] = {
            "obs_mask": obs,
            "minute_index": idx,
            "target": target,
            "last_gap_idx": last_gap_idx,
        }
    return out


def _choose_samples(per_sample_csv: Path, top_k: int) -> pd.DataFrame:
    df = pd.read_csv(per_sample_csv)
    df = df[df["segment_bucket_name"].eq("long")].copy()
    df = df.sort_values(["max_gap_minutes", "gap_alt_rmse"], ascending=[False, False])
    return df.head(top_k).reset_index(drop=True)


def _plot_case(case_dir: Path, row: pd.Series, truth: dict, pred: dict) -> dict:
    x = truth["minute_index"]
    obs_mask = truth["obs_mask"] > 0.5
    tgt = truth["target"][:, 2]
    est = pred["final"][:, 2]
    last_gap_idx = truth["last_gap_idx"]
    right_anchor_idx = int(np.where(obs_mask)[0][-1])

    pre_anchor_jump = float(abs(est[last_gap_idx] - tgt[right_anchor_idx]))
    final_step_jump = float(abs(est[right_anchor_idx] - est[last_gap_idx]))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(x, tgt, color="#111111", lw=2.4, label="ADS-B truth")
    ax.scatter(x[obs_mask], tgt[obs_mask], s=24, color="#1b9e77", zorder=5, label="Observed anchors")
    ax.plot(x, est, color="#d62728", lw=2.0, label="BiMamba-best")
    gap = ~obs_mask
    ax.fill_between(x, tgt.min(), tgt.max(), where=gap, color="#f1f1f1", alpha=0.45, label="Gap")
    ax.axvline(last_gap_idx, color="#ff7f0e", ls="--", lw=1.2, label="Last gap step")
    ax.set_title(
        f"{row['sample_id']} | anchors={int(row['anchor_count'])} max_gap={int(row['max_gap_minutes'])} "
        f"| gap_alt_rmse={row['gap_alt_rmse']:.1f} | jump={final_step_jump:.1f}m"
    )
    ax.set_xlabel("Minute index")
    ax.set_ylabel("Altitude (m)")
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(case_dir / "altitude_curve.png", dpi=180)
    plt.close(fig)

    return {
        "sample_id": row["sample_id"],
        "flight_id": row["flight_id"],
        "anchor_count": int(row["anchor_count"]),
        "max_gap_minutes": int(row["max_gap_minutes"]),
        "gap_alt_rmse": float(row["gap_alt_rmse"]),
        "pre_anchor_gap_to_right_anchor_m": pre_anchor_jump,
        "final_step_jump_m": final_step_jump,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument(
        "--out-dir",
        default="outputs/experiments/obs_conditioned_gaponly/bimamba_longgap_batch_20260531",
    )
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for BiMamba plotting, but torch.cuda.is_available() is false.")
    device = torch.device(args.device)

    set_seed(42)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = {
        "BiMamba-best": {
            "config": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/ab24_xyaux_zlinear_zadapter.yaml",
            "checkpoint": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/ab24_xyaux_zlinear_zadapter/best.pt",
        }
    }
    base_cfg = load_config(str(_resolve(spec["BiMamba-best"]["config"])))
    ds = _prepare_dataset(base_cfg, split_name=args.split)

    per_sample_csv = _resolve(
        "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/ab24_xyaux_zlinear_zadapter/main_task_metrics_test_per_sample.csv"
    )
    selected = _choose_samples(per_sample_csv, args.top_k)
    selected.to_csv(out_dir / "selected_longgap_samples.csv", index=False)

    selected_ids = set(selected["sample_id"].astype(str))
    truth = _truth_for_selected(ds, selected_ids)
    pred = _run_model_for_samples("BiMamba-best", spec, selected_ids, args.split, device)

    rows = []
    for _, row in selected.iterrows():
        sid = str(row["sample_id"])
        case_dir = out_dir / sid
        case_dir.mkdir(parents=True, exist_ok=True)
        rows.append(_plot_case(case_dir, row, truth[sid], pred[sid]))

    report = pd.DataFrame(rows)
    report.to_csv(out_dir / "longgap_jump_summary.csv", index=False)
    print(out_dir / "selected_longgap_samples.csv")
    print(out_dir / "longgap_jump_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
