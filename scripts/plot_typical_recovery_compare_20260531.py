from __future__ import annotations

import argparse
import json
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

from scripts.analyze_bidirectional_prediction_mechanism import (  # noqa: E402
    _prepare_dataset,
    _run_model_for_samples,
)
from src.training.utils import load_config, set_seed  # noqa: E402


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _sample_summary(ds) -> pd.DataFrame:
    rows: list[dict] = []
    for sample in ds.samples:
        obs = sample["obs_mask"].numpy()
        target = sample["target_pos"].numpy()
        rows.append(
            {
                "sample_id": str(sample["sample_id"]),
                "flight_id": str(sample["flight_id"]),
                "length": int(len(obs)),
                "anchor_count": int((obs > 0.5).sum()),
                "gap_count": int((obs <= 0.5).sum()),
                "max_gap": int(_max_gap_len(obs)),
                "alt_range_m": float(np.nanmax(target[:, 2]) - np.nanmin(target[:, 2])),
            }
        )
    return pd.DataFrame(rows)


def _max_gap_len(obs_mask: np.ndarray) -> int:
    best = 0
    cur = 0
    for v in obs_mask:
        if v < 0.5:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _choose_typical_samples(summary: pd.DataFrame) -> pd.DataFrame:
    adsc = summary[(summary["anchor_count"] <= 4) & (summary["max_gap"] >= 40)].copy()
    adsb = summary[(summary["anchor_count"].between(8, 12)) & (summary["max_gap"] >= 20)].copy()
    if adsc.empty or adsb.empty:
        raise RuntimeError("Cannot find both ADS-C-like and ADS-B-like typical samples in current split.")

    adsc_pick = adsc.sort_values(["max_gap", "alt_range_m", "gap_count"], ascending=False).iloc[0]
    same_flight = adsb[adsb["flight_id"].eq(adsc_pick["flight_id"])]
    if not same_flight.empty:
        adsb_pick = same_flight.sort_values(["max_gap", "alt_range_m", "gap_count"], ascending=False).iloc[0]
    else:
        adsb_pick = adsb.sort_values(["max_gap", "alt_range_m", "gap_count"], ascending=False).iloc[0]

    return pd.DataFrame(
        [
            {"scenario": "adsc_sparse_long", **adsc_pick.to_dict()},
            {"scenario": "adsb_dense_gap", **adsb_pick.to_dict()},
        ]
    )


def _true_series_for_selected(ds, selected_ids: set[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for sample in ds.samples:
        sid = str(sample["sample_id"])
        if sid not in selected_ids:
            continue
        obs = sample["obs_mask"].numpy()
        target = sample["target_pos"].numpy()
        idx = np.arange(len(obs))
        out[sid] = {
            "obs_mask": obs,
            "minute_index": idx,
            "target": target,
            "anchor_alt": target[:, 2].copy(),
            "anchor_lat": target[:, 0].copy(),
            "anchor_lon": target[:, 1].copy(),
        }
    return out


def _plot_altitude(case_dir: Path, scenario_name: str, info: dict, preds: dict[str, dict]) -> None:
    x = info["minute_index"]
    obs_mask = info["obs_mask"] > 0.5
    truth = info["target"][:, 2]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(x, truth, color="#111111", lw=2.4, label="ADS-B truth")
    ax.scatter(x[obs_mask], truth[obs_mask], s=24, color="#1b9e77", zorder=5, label="Observed anchors")

    colors = {
        "BiMamba-best": "#d62728",
        "BiLSTM": "#1f77b4",
        "Transformer": "#9467bd",
    }
    for label, restored in preds.items():
        ax.plot(x, restored["final"][:, 2], lw=1.9, color=colors[label], label=label)

    gap = ~obs_mask
    ax.fill_between(x, truth.min(), truth.max(), where=gap, color="#f1f1f1", alpha=0.45, label="Gap")
    ax.set_title(f"{scenario_name}: altitude recovery")
    ax.set_xlabel("Minute index")
    ax.set_ylabel("Altitude (m)")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(case_dir / "01_altitude_compare.png", dpi=180)
    plt.close(fig)


def _plot_xy(case_dir: Path, scenario_name: str, info: dict, preds: dict[str, dict]) -> None:
    truth = info["target"]
    obs_mask = info["obs_mask"] > 0.5

    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    ax.plot(truth[:, 1], truth[:, 0], color="#111111", lw=2.2, label="ADS-B truth")
    ax.scatter(truth[obs_mask, 1], truth[obs_mask, 0], s=24, color="#1b9e77", zorder=5, label="Observed anchors")

    colors = {
        "BiMamba-best": "#d62728",
        "BiLSTM": "#1f77b4",
        "Transformer": "#9467bd",
    }
    for label, restored in preds.items():
        ax.plot(restored["final"][:, 1], restored["final"][:, 0], lw=1.7, color=colors[label], label=label)

    ax.set_title(f"{scenario_name}: lat/lon trajectory")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(case_dir / "02_xy_compare.png", dpi=180)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument(
        "--out-dir",
        default="outputs/experiments/obs_conditioned_gaponly/typical_recovery_compare_20260531",
    )
    args = ap.parse_args()

    set_seed(42)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for BiMamba plotting, but torch.cuda.is_available() is false.")
    device = torch.device(args.device)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_specs = {
        "BiMamba-best": {
            "config": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/ab24_xyaux_zlinear_zadapter.yaml",
            "checkpoint": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/ab24_xyaux_zlinear_zadapter/best.pt",
        },
        "BiLSTM": {
            "config": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_bilstm.yaml",
            "checkpoint": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_bilstm/best.pt",
        },
        "Transformer": {
            "config": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_transformer_proto.yaml",
            "checkpoint": "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_transformer_proto/best.pt",
        },
    }
    for spec in model_specs.values():
        for key in ("config", "checkpoint"):
            if not _resolve(spec[key]).exists():
                raise FileNotFoundError(f"Missing {key}: {_resolve(spec[key])}")

    base_cfg = load_config(str(_resolve(model_specs["BiMamba-best"]["config"])))
    ds = _prepare_dataset(base_cfg, split_name=args.split)
    summary = _sample_summary(ds)
    selection = _choose_typical_samples(summary)
    selection.to_csv(out_dir / "selected_typical_samples.csv", index=False)

    selected_ids = set(selection["sample_id"].astype(str))
    truth = _true_series_for_selected(ds, selected_ids)

    model_outputs = {
        label: _run_model_for_samples(label, model_specs, selected_ids, args.split, device)
        for label in model_specs
    }

    rows = []
    for _, row in selection.iterrows():
        sid = str(row["sample_id"])
        case_dir = out_dir / str(row["scenario"])
        case_dir.mkdir(parents=True, exist_ok=True)

        preds = {label: model_outputs[label][sid] for label in model_specs}
        _plot_altitude(case_dir, str(row["scenario"]), truth[sid], preds)
        _plot_xy(case_dir, str(row["scenario"]), truth[sid], preds)

        for label, restored in preds.items():
            gap = truth[sid]["obs_mask"] <= 0.5
            gap_alt_err = restored["final"][gap, 2] - truth[sid]["target"][gap, 2]
            rows.append(
                {
                    "scenario": row["scenario"],
                    "sample_id": sid,
                    "flight_id": row["flight_id"],
                    "model": label,
                    "anchor_count": int(row["anchor_count"]),
                    "max_gap": int(row["max_gap"]),
                    "gap_alt_mae": float(np.mean(np.abs(gap_alt_err))),
                    "gap_alt_rmse": float(np.sqrt(np.mean(gap_alt_err**2))),
                }
            )

    pd.DataFrame(rows).to_csv(out_dir / "typical_compare_summary.csv", index=False)
    print(out_dir / "selected_typical_samples.csv")
    print(out_dir / "typical_compare_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
