from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.anchor_graph_height_reference import (
    AnchorGraphParams,
    build_anchor_graph_reference,
    build_anchor_graph_score_reference,
    build_anchor_graph_soft_reference,
)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _metrics_by_sample(df: pd.DataFrame, pred_col: str, model: str) -> pd.DataFrame:
    rows = []
    for sid, g in df.groupby("sample_id", sort=False):
        gap = g["obs_mask"].to_numpy(dtype=int) == 0
        truth = g.loc[gap, "adsb_alt_m"].to_numpy(dtype=float)
        pred = g.loc[gap, pred_col].to_numpy(dtype=float)
        ok = np.isfinite(truth) & np.isfinite(pred)
        if not ok.any():
            continue
        err = pred[ok] - truth[ok]
        rows.append(
            {
                "model": model,
                "sample_id": sid,
                "source_case": g["source_case"].iloc[0],
                "anchor_count": int(g["anchor_count"].iloc[0]),
                "gap_point_count": int(ok.sum()),
                "alt_RMSE_m": float(np.sqrt(np.mean(np.square(err)))),
                "alt_MAE_m": float(np.mean(np.abs(err))),
                "alt_MaxAE_m": float(np.max(np.abs(err))),
            }
        )
    return pd.DataFrame(rows)


def _plot_complete_adsb(df: pd.DataFrame, out_dir: Path, anchor_counts: set[int]) -> None:
    plot_dir = out_dir / "complete_adsb_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for (source_case, anchor_count), g in df.groupby(["source_case", "anchor_count"], sort=False):
        if int(anchor_count) not in anchor_counts:
            continue
        g = g.sort_values("minute_index")
        x = g["minute_index"].to_numpy(dtype=float)
        obs = g["obs_mask"].to_numpy(dtype=int) == 1
        fig, ax = plt.subplots(figsize=(12.5, 5.2), facecolor="white")
        ax.plot(x, g["adsb_alt_m"], color="black", lw=2.2, label="ADS-B truth")
        ax.scatter(x[obs], g.loc[obs, "adsb_alt_m"], color="black", marker="*", s=130, zorder=7, label="anchors")
        ax.plot(x, g["本文方案_alt_m"], color="#999999", lw=1.4, alpha=0.85, label="Ours-A3 raw")
        ax.plot(x, g["AnchorGraph_ref_alt_m"], color="#d62828", lw=2.0, alpha=0.8, label="AnchorGraph hard")
        if "AnchorGraph_soft_ref_alt_m" in g:
            ax.plot(x, g["AnchorGraph_soft_ref_alt_m"], color="#023047", lw=1.7, alpha=0.85, label="AnchorGraph soft")
        if "AnchorGraph_score_ref_alt_m" in g:
            ax.plot(x, g["AnchorGraph_score_ref_alt_m"], color="#ff006e", lw=2.3, label="AnchorGraph score")
        if "分段线性插值_alt_m" in g:
            ax.plot(x, g["分段线性插值_alt_m"], color="#2a9d8f", lw=1.4, alpha=0.8, label="Linear")
        ax.set_title(f"{source_case} | anchors={anchor_count}")
        ax.set_xlabel("Cruise window minute")
        ax.set_ylabel("Altitude (m)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=4)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{source_case}_anchor{anchor_count}_anchor_graph_ref.png", dpi=180)
        plt.close(fig)


def _run_complete_adsb(input_dir: Path, out_dir: Path, params: AnchorGraphParams) -> None:
    pred_path = input_dir / "sparse_cruise_model_predictions.csv"
    df = pd.read_csv(pred_path)
    refs = []
    soft_refs = []
    score_refs = []
    decisions = []
    soft_decisions = []
    score_decisions = []
    for sid, g in df.groupby("sample_id", sort=False):
        g = g.sort_values("minute_index")
        ref, dec = build_anchor_graph_reference(
            time_index=g["minute_index"].to_numpy(dtype=float),
            anchor_mask=g["obs_mask"].to_numpy(dtype=int) == 1,
            anchor_alt_or_truth=np.where(g["obs_mask"].to_numpy(dtype=int) == 1, g["adsb_alt_m"].to_numpy(dtype=float), np.nan),
            raw_alt=g["本文方案_alt_m"].to_numpy(dtype=float),
            params=params,
        )
        soft_ref, soft_dec = build_anchor_graph_soft_reference(
            time_index=g["minute_index"].to_numpy(dtype=float),
            anchor_mask=g["obs_mask"].to_numpy(dtype=int) == 1,
            anchor_alt_or_truth=np.where(g["obs_mask"].to_numpy(dtype=int) == 1, g["adsb_alt_m"].to_numpy(dtype=float), np.nan),
            raw_alt=g["本文方案_alt_m"].to_numpy(dtype=float),
            params=params,
        )
        score_ref, score_dec = build_anchor_graph_score_reference(
            time_index=g["minute_index"].to_numpy(dtype=float),
            anchor_mask=g["obs_mask"].to_numpy(dtype=int) == 1,
            anchor_alt_or_truth=np.where(g["obs_mask"].to_numpy(dtype=int) == 1, g["adsb_alt_m"].to_numpy(dtype=float), np.nan),
            raw_alt=g["本文方案_alt_m"].to_numpy(dtype=float),
            params=params,
        )
        refs.append(pd.Series(ref, index=g.index))
        soft_refs.append(pd.Series(soft_ref, index=g.index))
        score_refs.append(pd.Series(score_ref, index=g.index))
        for d in dec:
            d.update(
                {
                    "sample_id": sid,
                    "source_case": g["source_case"].iloc[0],
                    "anchor_count": int(g["anchor_count"].iloc[0]),
                }
            )
            decisions.append(d)
        for d in soft_dec:
            d.update(
                {
                    "sample_id": sid,
                    "source_case": g["source_case"].iloc[0],
                    "anchor_count": int(g["anchor_count"].iloc[0]),
                }
            )
            soft_decisions.append(d)
        for d in score_dec:
            d.update(
                {
                    "sample_id": sid,
                    "source_case": g["source_case"].iloc[0],
                    "anchor_count": int(g["anchor_count"].iloc[0]),
                }
            )
            score_decisions.append(d)
    df["AnchorGraph_ref_alt_m"] = pd.concat(refs).sort_index()
    df["AnchorGraph_soft_ref_alt_m"] = pd.concat(soft_refs).sort_index()
    df["AnchorGraph_score_ref_alt_m"] = pd.concat(score_refs).sort_index()
    df["AnchorGraph_ref_alt_abs_err_m"] = np.where(
        df["obs_mask"].to_numpy(dtype=int) == 1,
        0.0,
        np.abs(df["AnchorGraph_ref_alt_m"].to_numpy(dtype=float) - df["adsb_alt_m"].to_numpy(dtype=float)),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "complete_adsb_anchor_graph_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(decisions).to_csv(out_dir / "complete_adsb_anchor_graph_decisions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(soft_decisions).to_csv(out_dir / "complete_adsb_anchor_graph_soft_decisions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(score_decisions).to_csv(out_dir / "complete_adsb_anchor_graph_score_decisions.csv", index=False, encoding="utf-8-sig")
    metric_parts = [
        _metrics_by_sample(df, "本文方案_alt_m", "Ours-A3 raw"),
        _metrics_by_sample(df, "AnchorGraph_ref_alt_m", "AnchorGraph reference"),
        _metrics_by_sample(df, "AnchorGraph_soft_ref_alt_m", "AnchorGraph soft reference"),
        _metrics_by_sample(df, "AnchorGraph_score_ref_alt_m", "AnchorGraph score reference"),
        _metrics_by_sample(df, "分段线性插值_alt_m", "Linear"),
    ]
    metrics = pd.concat(metric_parts, ignore_index=True)
    metrics.to_csv(out_dir / "complete_adsb_anchor_graph_metrics_by_sample.csv", index=False, encoding="utf-8-sig")
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
    by_model.to_csv(out_dir / "complete_adsb_anchor_graph_metrics_by_model.csv", index=False, encoding="utf-8-sig")
    by_anchor.to_csv(out_dir / "complete_adsb_anchor_graph_metrics_by_anchor_count.csv", index=False, encoding="utf-8-sig")
    _plot_complete_adsb(df, out_dir, anchor_counts={3, 8})


def _plot_cross_ocean(df: pd.DataFrame, case_id: str, out_path: Path) -> None:
    x = df["相对分钟"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(12.0, 5.6), facecolor="white")
    ax.plot(x, df["原始ADS-B高度_m"], color="black", lw=2.0, label="ADS-B visible")
    obs = df["原始ADS-C锚点高度_m"].notna().to_numpy()
    ax.scatter(x[obs], df.loc[obs, "原始ADS-C锚点高度_m"], color="black", marker="*", s=150, zorder=8, label="ADS-C anchors")
    ax.plot(x, df["本文方案A3恢复高度_m"], color="#999999", lw=1.5, alpha=0.85, label="Ours-A3 raw")
    ax.plot(x, df["AnchorGraph参考高度_m"], color="#d62828", lw=1.9, alpha=0.8, label="AnchorGraph hard")
    if "AnchorGraph软门控参考高度_m" in df:
        ax.plot(x, df["AnchorGraph软门控参考高度_m"], color="#023047", lw=1.7, alpha=0.85, label="AnchorGraph soft")
    if "AnchorGraph打分门控参考高度_m" in df:
        ax.plot(x, df["AnchorGraph打分门控参考高度_m"], color="#ff006e", lw=2.4, label="AnchorGraph score")
    if "BiLSTM恢复高度_m" in df:
        ax.plot(x, df["BiLSTM恢复高度_m"], color="#f77f00", lw=1.2, alpha=0.6, label="BiLSTM")
    ax.set_title(case_id)
    ax.set_xlabel("Time (minute)")
    ax.set_ylabel("Altitude (m)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _run_cross_ocean(input_dir: Path, out_dir: Path, params: AnchorGraphParams) -> None:
    rows = []
    for csv_path in sorted(input_dir.glob("*/*_altitude_compare_table.csv")):
        case_id = csv_path.parent.name
        df = pd.read_csv(csv_path)
        anchor_mask = df["原始ADS-C锚点高度_m"].notna().to_numpy()
        anchor_alt = np.where(anchor_mask, df["原始ADS-C锚点高度_m"].to_numpy(dtype=float), np.nan)
        ref, dec = build_anchor_graph_reference(
            time_index=df["相对分钟"].to_numpy(dtype=float),
            anchor_mask=anchor_mask,
            anchor_alt_or_truth=anchor_alt,
            raw_alt=df["本文方案A3恢复高度_m"].to_numpy(dtype=float),
            params=params,
        )
        soft_ref, soft_dec = build_anchor_graph_soft_reference(
            time_index=df["相对分钟"].to_numpy(dtype=float),
            anchor_mask=anchor_mask,
            anchor_alt_or_truth=anchor_alt,
            raw_alt=df["本文方案A3恢复高度_m"].to_numpy(dtype=float),
            params=params,
        )
        score_ref, score_dec = build_anchor_graph_score_reference(
            time_index=df["相对分钟"].to_numpy(dtype=float),
            anchor_mask=anchor_mask,
            anchor_alt_or_truth=anchor_alt,
            raw_alt=df["本文方案A3恢复高度_m"].to_numpy(dtype=float),
            params=params,
        )
        df["AnchorGraph参考高度_m"] = ref
        df["AnchorGraph软门控参考高度_m"] = soft_ref
        df["AnchorGraph打分门控参考高度_m"] = score_ref
        case_dir = out_dir / "cross_ocean_showcases" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(case_dir / csv_path.name.replace("_altitude_compare_table.csv", "_anchor_graph_reference_table.csv"), index=False, encoding="utf-8-sig")
        _plot_cross_ocean(df, case_id, case_dir / csv_path.name.replace("_altitude_compare_table.csv", "_anchor_graph_reference_plot.png"))
        for d in dec:
            d.update({"case_id": case_id})
            rows.append(d)
        for d in soft_dec:
            d.update({"case_id": case_id, "soft_gate": True})
            rows.append(d)
        for d in score_dec:
            d.update({"case_id": case_id, "score_gate": True})
            rows.append(d)
    pd.DataFrame(rows).to_csv(out_dir / "cross_ocean_anchor_graph_decisions.csv", index=False, encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--complete-adsb-dir", default="outputs/runs/complete_adsb_sparse_cruise_eval_20260519")
    parser.add_argument(
        "--cross-ocean-dir",
        default="outputs/runs/paper_showcase_cross_ocean_all_model_compare_20260518/selected_window_altitude_tables_plots",
    )
    parser.add_argument("--out-dir", default="outputs/runs/anchor_graph_height_reference_trial_20260520")
    args = parser.parse_args()

    params = AnchorGraphParams()
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _run_complete_adsb(_resolve(args.complete_adsb_dir), out_dir, params)
    _run_cross_ocean(_resolve(args.cross_ocean_dir), out_dir, params)

    print(f"[done] out_dir={out_dir}")
    by_model = pd.read_csv(out_dir / "complete_adsb_anchor_graph_metrics_by_model.csv")
    print("\n[complete ADS-B sparse cruise | by model]")
    print(by_model.round(3).to_string(index=False))
    by_anchor = pd.read_csv(out_dir / "complete_adsb_anchor_graph_metrics_by_anchor_count.csv")
    print("\n[complete ADS-B sparse cruise | by anchor count]")
    print(by_anchor.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
