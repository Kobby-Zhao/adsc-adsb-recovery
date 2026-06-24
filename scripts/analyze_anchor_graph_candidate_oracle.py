from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.anchor_graph_height_reference import AnchorGraphParams, _edge_profiles


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _sample_metrics(df: pd.DataFrame, col: str, model: str) -> pd.DataFrame:
    rows = []
    for sid, g in df.groupby("sample_id", sort=False):
        gap = g["obs_mask"].to_numpy(dtype=int) == 0
        err = g.loc[gap, col].to_numpy(dtype=float) - g.loc[gap, "adsb_alt_m"].to_numpy(dtype=float)
        rows.append(
            {
                "model": model,
                "sample_id": sid,
                "source_case": g["source_case"].iloc[0],
                "anchor_count": int(g["anchor_count"].iloc[0]),
                "alt_RMSE_m": float(np.sqrt(np.mean(np.square(err)))),
                "alt_MAE_m": float(np.mean(np.abs(err))),
                "alt_MaxAE_m": float(np.max(np.abs(err))),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/runs/complete_adsb_sparse_cruise_eval_20260519/sparse_cruise_model_predictions.csv")
    parser.add_argument("--out-dir", default="outputs/runs/anchor_graph_height_reference_soft_trial_20260520/oracle_analysis")
    args = parser.parse_args()
    df = pd.read_csv(_resolve(args.input))
    params = AnchorGraphParams()
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    oracle_ref = np.full(len(df), np.nan, dtype=float)
    oracle_rows = []
    for sid, g in df.groupby("sample_id", sort=False):
        g = g.sort_values("minute_index")
        idx = g.index.to_numpy()
        obs = g["obs_mask"].to_numpy(dtype=int) == 1
        anchors = np.where(obs)[0]
        sample_oracle = g["本文方案_alt_m"].to_numpy(dtype=float).copy()
        for left, right in zip(anchors[:-1], anchors[1:]):
            n = int(right - left + 1)
            z_left = float(g["adsb_alt_m"].iloc[left])
            z_right = float(g["adsb_alt_m"].iloc[right])
            raw = g["本文方案_alt_m"].iloc[left : right + 1].to_numpy(dtype=float)
            truth = g["adsb_alt_m"].iloc[left : right + 1].to_numpy(dtype=float)
            gap_local = np.ones(n, dtype=bool)
            gap_local[0] = False
            gap_local[-1] = False
            profiles = _edge_profiles(z_left, z_right, n, raw, params)
            scores = {}
            for name, prof in profiles.items():
                e = prof[gap_local] - truth[gap_local]
                scores[name] = float(np.mean(np.abs(e)))
            best = min(scores, key=scores.get)
            sample_oracle[left : right + 1] = profiles[best]
            oracle_rows.append(
                {
                    "sample_id": sid,
                    "source_case": g["source_case"].iloc[0],
                    "anchor_count": int(g["anchor_count"].iloc[0]),
                    "left_minute": int(g["minute_index"].iloc[left]),
                    "right_minute": int(g["minute_index"].iloc[right]),
                    "gap_len": int(right - left - 1),
                    "delta_z_m": float(z_right - z_left),
                    "best_candidate": best,
                    "hold_mae_m": scores["hold"],
                    "switch_mae_m": scores["switch"],
                    "trend_mae_m": scores["trend"],
                }
            )
        oracle_ref[idx] = sample_oracle
    df["AnchorGraph_oracle_candidate_alt_m"] = oracle_ref
    df.to_csv(out_dir / "oracle_candidate_predictions.csv", index=False, encoding="utf-8-sig")
    oracle = pd.DataFrame(oracle_rows)
    oracle.to_csv(out_dir / "oracle_candidate_gap_choices.csv", index=False, encoding="utf-8-sig")

    metrics = pd.concat(
        [
            _sample_metrics(df, "本文方案_alt_m", "Ours-A3 raw"),
            _sample_metrics(df, "分段线性插值_alt_m", "Linear"),
            _sample_metrics(df, "AnchorGraph_oracle_candidate_alt_m", "Oracle candidate upper-bound"),
        ],
        ignore_index=True,
    )
    metrics.to_csv(out_dir / "oracle_candidate_metrics_by_sample.csv", index=False, encoding="utf-8-sig")
    by_model = metrics.groupby("model", as_index=False).agg(
        sample_count=("sample_id", "nunique"),
        alt_RMSE_m=("alt_RMSE_m", "mean"),
        alt_MAE_m=("alt_MAE_m", "mean"),
        alt_MaxAE_m=("alt_MaxAE_m", "mean"),
    )
    by_anchor = metrics.groupby(["model", "anchor_count"], as_index=False).agg(
        sample_count=("sample_id", "nunique"),
        alt_RMSE_m=("alt_RMSE_m", "mean"),
        alt_MAE_m=("alt_MAE_m", "mean"),
        alt_MaxAE_m=("alt_MaxAE_m", "mean"),
    )
    by_model.to_csv(out_dir / "oracle_candidate_metrics_by_model.csv", index=False, encoding="utf-8-sig")
    by_anchor.to_csv(out_dir / "oracle_candidate_metrics_by_anchor_count.csv", index=False, encoding="utf-8-sig")
    print("[oracle choices]")
    print(oracle["best_candidate"].value_counts().to_string())
    print("\n[by model]")
    print(by_model.round(3).to_string(index=False))
    print("\n[by anchor]")
    print(by_anchor.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
