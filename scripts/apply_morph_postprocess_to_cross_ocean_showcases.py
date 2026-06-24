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

TIME_COL = "相对分钟"
TS_COL = "时间_UTC_精确到分钟"
ADSB_COL = "原始ADS-B高度_m"
ANCHOR_COL = "原始ADS-C锚点高度_m"
OURS_COL = "本文方案A3恢复高度_m"
POST_COL = "本文方案A3形态后处理高度_m"


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _transition_profile(z_left: float, z_right: float, n: int, raw: np.ndarray) -> np.ndarray:
    out = np.full(n, float(z_left), dtype=float)
    dz = float(z_right - z_left)
    abs_dz = abs(dz)
    if n <= 2:
        return np.linspace(z_left, z_right, n)
    if abs_dz < 90.0:
        return np.linspace(z_left, z_right, n)
    midpoint = (float(z_left) + float(z_right)) / 2.0
    finite_raw = np.where(np.isfinite(raw), raw, np.linspace(z_left, z_right, n))
    center = int(np.argmin(np.abs(finite_raw - midpoint)))
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


def _postprocess(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    post = out[OURS_COL].to_numpy(dtype=float).copy()
    anchors = np.where(out[ANCHOR_COL].notna().to_numpy())[0]
    for left, right in zip(anchors[:-1], anchors[1:]):
        raw_seg = out[OURS_COL].iloc[left : right + 1].to_numpy(dtype=float)
        if not np.isfinite(raw_seg).any():
            continue
        z_left = float(out[ANCHOR_COL].iloc[left])
        z_right = float(out[ANCHOR_COL].iloc[right])
        prof = _transition_profile(z_left, z_right, right - left + 1, raw_seg)
        valid = np.isfinite(raw_seg) | out[ANCHOR_COL].iloc[left : right + 1].notna().to_numpy()
        seg_post = post[left : right + 1]
        seg_post[valid] = prof[valid]
        post[left : right + 1] = seg_post
    anchor_mask = out[ANCHOR_COL].notna().to_numpy()
    post[anchor_mask] = out.loc[anchor_mask, ANCHOR_COL].to_numpy(dtype=float)
    out[POST_COL] = post
    return out


def _visible_adsb_metrics(df: pd.DataFrame, case_id: str) -> dict:
    rows = df[ADSB_COL].notna() & df[OURS_COL].notna() & df[POST_COL].notna()
    if not rows.any():
        return {
            "case_id": case_id,
            "visible_adsb_points": 0,
            "raw_MAE_m": np.nan,
            "post_MAE_m": np.nan,
            "raw_RMSE_m": np.nan,
            "post_RMSE_m": np.nan,
        }
    truth = df.loc[rows, ADSB_COL].to_numpy(dtype=float)
    raw_err = df.loc[rows, OURS_COL].to_numpy(dtype=float) - truth
    post_err = df.loc[rows, POST_COL].to_numpy(dtype=float) - truth
    return {
        "case_id": case_id,
        "visible_adsb_points": int(rows.sum()),
        "raw_MAE_m": float(np.mean(np.abs(raw_err))),
        "post_MAE_m": float(np.mean(np.abs(post_err))),
        "raw_RMSE_m": float(np.sqrt(np.mean(np.square(raw_err)))),
        "post_RMSE_m": float(np.sqrt(np.mean(np.square(post_err)))),
    }


def _plot(df: pd.DataFrame, case_id: str, out_path: Path) -> None:
    x = df[TIME_COL].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(12.0, 5.6), facecolor="white")
    if ADSB_COL in df:
        ax.plot(x, df[ADSB_COL], color="black", lw=2.2, label="ADS-B")
    anchor_mask = df[ANCHOR_COL].notna().to_numpy()
    ax.scatter(
        x[anchor_mask],
        df.loc[anchor_mask, ANCHOR_COL],
        color="black",
        marker="*",
        s=150,
        zorder=8,
        label="ADS-C anchors",
    )
    ax.plot(x, df[OURS_COL], color="#999999", lw=1.7, alpha=0.9, label="Ours-A3 raw")
    ax.plot(x, df[POST_COL], color="#d62828", lw=2.5, alpha=0.95, label="Ours-A3 + morphology postprocess")
    for col, label, color in [
        ("BiLSTM恢复高度_m", "BiLSTM", "#f77f00"),
        ("LSTM恢复高度_m", "LSTM", "#2a9d8f"),
        ("CNN+LSTM恢复高度_m", "CNN+LSTM", "#9b5de5"),
        ("Transformer恢复高度_m", "Transformer", "#457b9d"),
        ("Kalman Filter恢复高度_m", "Kalman Filter", "#6c757d"),
    ]:
        if col in df:
            ax.plot(x, df[col], lw=1.2, alpha=0.55, color=color, label=label)
    ax.set_title(case_id)
    ax.set_xlabel("Time (minute)")
    ax.set_ylabel("Altitude (m)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="outputs/runs/paper_showcase_cross_ocean_all_model_compare_20260518/selected_window_altitude_tables_plots",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/runs/paper_showcase_cross_ocean_all_model_compare_20260518/selected_window_altitude_tables_plots_morph_postprocess",
    )
    args = parser.parse_args()
    input_dir = _resolve(args.input_dir)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    for csv_path in sorted(input_dir.glob("*/*_altitude_compare_table.csv")):
        case_id = csv_path.parent.name
        case_dir = out_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(csv_path)
        post = _postprocess(df)
        out_csv = case_dir / csv_path.name.replace("_altitude_compare_table.csv", "_altitude_compare_table_morph_postprocess.csv")
        out_png = case_dir / csv_path.name.replace("_altitude_compare_table.csv", "_altitude_compare_plot_morph_postprocess.png")
        post.to_csv(out_csv, index=False, encoding="utf-8-sig")
        _plot(post, case_id, out_png)
        metrics.append(_visible_adsb_metrics(post, case_id))
    metric_df = pd.DataFrame(metrics)
    metric_df.to_csv(out_dir / "morph_postprocess_visible_adsb_metrics.csv", index=False, encoding="utf-8-sig")
    print(f"[done] out_dir={out_dir}")
    print(metric_df.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
