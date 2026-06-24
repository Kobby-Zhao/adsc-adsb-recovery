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


ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _transition_profile(z_left: float, z_right: float, n: int, raw: np.ndarray) -> np.ndarray:
    """Anchor-only altitude-layer transition profile.

    The transition location is inferred from the raw prediction's midpoint crossing.
    ADS-B truth is not used by this function.
    """
    out = np.full(n, float(z_left), dtype=float)
    dz = float(z_right - z_left)
    abs_dz = abs(dz)
    if n <= 2:
        return np.linspace(z_left, z_right, n)
    if abs_dz < 90.0:
        return np.linspace(z_left, z_right, n)

    midpoint = (float(z_left) + float(z_right)) / 2.0
    center = int(np.argmin(np.abs(raw - midpoint)))
    duration = int(np.clip(round(abs_dz / 150.0), 2, 8))
    start = int(np.clip(center - duration // 2, 1, max(1, n - duration - 1)))
    end = int(np.clip(start + duration, start + 1, n - 1))

    out[:start] = float(z_left)
    t = np.linspace(0.0, 1.0, end - start + 1)
    out[start : end + 1] = float(z_left) + dz * _smoothstep(t)
    out[end + 1 :] = float(z_right)
    out[0] = float(z_left)
    out[-1] = float(z_right)
    return out


def _postprocess_sample(g: pd.DataFrame, raw_col: str) -> np.ndarray:
    g = g.sort_values("minute_index").reset_index(drop=True)
    raw = g[raw_col].to_numpy(dtype=float)
    out = raw.copy()
    obs = g["obs_mask"].to_numpy(dtype=int) == 1
    anchors = np.where(obs)[0]
    if len(anchors) < 2:
        return out
    anchor_alt = g["adsb_alt_m"].to_numpy(dtype=float)
    for left, right in zip(anchors[:-1], anchors[1:]):
        seg_raw = raw[left : right + 1]
        out[left : right + 1] = _transition_profile(
            float(anchor_alt[left]),
            float(anchor_alt[right]),
            int(right - left + 1),
            seg_raw,
        )
    return out


def _metrics(df: pd.DataFrame, col: str, model_name: str) -> pd.DataFrame:
    rows = []
    for sid, g in df.groupby("sample_id", sort=False):
        gap = g["obs_mask"].to_numpy(dtype=int) == 0
        err = g.loc[gap, col].to_numpy(dtype=float) - g.loc[gap, "adsb_alt_m"].to_numpy(dtype=float)
        rows.append(
            {
                "model": model_name,
                "sample_id": sid,
                "source_case": g["source_case"].iloc[0],
                "anchor_count": int(g["anchor_count"].iloc[0]),
                "gap_point_count": int(gap.sum()),
                "alt_RMSE_m": float(np.sqrt(np.mean(np.square(err)))),
                "alt_MAE_m": float(np.mean(np.abs(err))),
                "alt_MaxAE_m": float(np.max(np.abs(err))),
            }
        )
    return pd.DataFrame(rows)


def _plot(df: pd.DataFrame, out_dir: Path, anchor_counts: set[int]) -> None:
    plot_dir = out_dir / "plots_postprocessed"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for (source_case, anchor_count), g in df.groupby(["source_case", "anchor_count"], sort=False):
        if int(anchor_count) not in anchor_counts:
            continue
        g = g.sort_values("minute_index")
        x = g["minute_index"].to_numpy()
        obs = g["obs_mask"].to_numpy(dtype=int) == 1
        fig, ax = plt.subplots(figsize=(12.5, 5.2), facecolor="white")
        ax.plot(x, g["adsb_alt_m"], color="black", lw=2.3, label="ADS-B truth")
        ax.scatter(x[obs], g.loc[obs, "adsb_alt_m"], color="black", marker="*", s=130, zorder=6, label="anchors")
        ax.plot(x, g["本文方案_alt_m"], color="#999999", lw=1.6, alpha=0.9, label="Ours-A3 raw")
        ax.plot(x, g["OursA3_morph_alt_m"], color="#d62828", lw=2.4, alpha=0.95, label="Ours-A3 + morphology postprocess")
        ax.plot(x, g["分段线性插值_alt_m"], color="#2a9d8f", lw=1.5, alpha=0.85, label="Linear")
        ax.set_title(f"{source_case} | anchor_count={anchor_count}")
        ax.set_xlabel("Cruise window minute")
        ax.set_ylabel("Altitude (m)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9, ncol=4)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{source_case}_anchor{anchor_count}_morph_postprocess.png", dpi=180)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", default="outputs/runs/complete_adsb_sparse_cruise_eval_20260519")
    parser.add_argument("--anchor-counts-to-plot", default="3,8")
    args = parser.parse_args()

    in_dir = _resolve(args.in_dir)
    pred_path = in_dir / "sparse_cruise_model_predictions.csv"
    df = pd.read_csv(pred_path)
    post_values = []
    for _, g in df.groupby("sample_id", sort=False):
        post_values.append(pd.Series(_postprocess_sample(g, "本文方案_alt_m"), index=g.index))
    df["OursA3_morph_alt_m"] = pd.concat(post_values).sort_index()
    df["OursA3_morph_alt_abs_err_m"] = np.where(
        df["obs_mask"].to_numpy(dtype=int) == 1,
        0.0,
        np.abs(df["OursA3_morph_alt_m"].to_numpy(dtype=float) - df["adsb_alt_m"].to_numpy(dtype=float)),
    )

    out_pred = in_dir / "sparse_cruise_model_predictions_with_morph_postprocess.csv"
    df.to_csv(out_pred, index=False, encoding="utf-8-sig")

    metric_parts = [
        _metrics(df, "本文方案_alt_m", "Ours-A3 raw"),
        _metrics(df, "OursA3_morph_alt_m", "Ours-A3 + morphology postprocess"),
        _metrics(df, "分段线性插值_alt_m", "Linear"),
    ]
    metrics = pd.concat(metric_parts, ignore_index=True)
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
    metrics.to_csv(in_dir / "morph_postprocess_metrics_by_sample.csv", index=False, encoding="utf-8-sig")
    by_model.to_csv(in_dir / "morph_postprocess_metrics_by_model.csv", index=False, encoding="utf-8-sig")
    by_anchor.to_csv(in_dir / "morph_postprocess_metrics_by_anchor_count.csv", index=False, encoding="utf-8-sig")

    anchor_counts = {int(x) for x in args.anchor_counts_to_plot.split(",") if x.strip()}
    _plot(df, in_dir, anchor_counts)

    print(f"[done] {in_dir}")
    print("\n[by model]")
    print(by_model.round(3).to_string(index=False))
    print("\n[by anchor count]")
    print(by_anchor.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
