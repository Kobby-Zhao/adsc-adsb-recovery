from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_bidirectional_prediction_mechanism import _prepare_dataset  # noqa: E402
from src.training.utils import load_config  # noqa: E402


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _gap_runs(obs_mask: np.ndarray) -> list[int]:
    obs = np.asarray(obs_mask, dtype=float) > 0.5
    anchors = np.where(obs)[0]
    gaps: list[int] = []
    for left, right in zip(anchors[:-1], anchors[1:]):
        gap_len = int(right - left - 1)
        if gap_len > 0:
            gaps.append(gap_len)
    return gaps


def _sample_rows(stage: str, cfg_path: Path, split: str) -> tuple[list[dict], list[dict]]:
    cfg = load_config(str(cfg_path))
    ds = _prepare_dataset(cfg, split_name=split)
    sample_rows: list[dict] = []
    gap_rows: list[dict] = []
    for sample in ds.samples:
        sid = str(sample["sample_id"])
        obs = sample["obs_mask"].numpy()
        valid_len = int(len(obs))
        anchor_count = int(np.sum(obs > 0.5))
        gap_point_count = int(valid_len - anchor_count)
        gaps = _gap_runs(obs)
        missing_rate = gap_point_count / valid_len if valid_len > 0 else np.nan
        sample_rows.append(
            {
                "stage": stage,
                "split": split,
                "sample_id": sid,
                "flight_id": sample["flight_id"],
                "seq_len": valid_len,
                "anchor_count": anchor_count,
                "gap_point_count": gap_point_count,
                "missing_rate": missing_rate,
                "gap_segment_count": len(gaps),
                "mean_gap_len_min": float(np.mean(gaps)) if gaps else 0.0,
                "median_gap_len_min": float(np.median(gaps)) if gaps else 0.0,
                "max_gap_len_min": int(max(gaps)) if gaps else 0,
            }
        )
        for gid, gap_len in enumerate(gaps):
            gap_rows.append(
                {
                    "stage": stage,
                    "split": split,
                    "sample_id": sid,
                    "flight_id": sample["flight_id"],
                    "gap_id": gid,
                    "gap_len_min": gap_len,
                    "anchor_count": anchor_count,
                    "seq_len": valid_len,
                }
            )
    return sample_rows, gap_rows


def _metric_summary(s: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(s, errors="coerce").dropna()
    return {
        "mean": float(x.mean()) if len(x) else np.nan,
        "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "p25": float(x.quantile(0.25)) if len(x) else np.nan,
        "p50": float(x.quantile(0.50)) if len(x) else np.nan,
        "p75": float(x.quantile(0.75)) if len(x) else np.nan,
        "p90": float(x.quantile(0.90)) if len(x) else np.nan,
        "min": float(x.min()) if len(x) else np.nan,
        "max": float(x.max()) if len(x) else np.nan,
    }


def _build_summary(sample_df: pd.DataFrame, gap_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    metrics = [
        ("seq_len", "序列长度/min"),
        ("anchor_count", "锚点数"),
        ("gap_point_count", "缺失点数/min"),
        ("missing_rate", "缺失率"),
        ("gap_segment_count", "缺失片段数"),
        ("mean_gap_len_min", "样本内平均gap/min"),
        ("max_gap_len_min", "样本内最大gap/min"),
    ]
    for stage, g in sample_df.groupby("stage", sort=False):
        row = {
            "测试集": stage,
            "样本数": int(len(g)),
            "总有效点数": int(g["seq_len"].sum()),
            "总锚点数": int(g["anchor_count"].sum()),
            "总缺失点数": int(g["gap_point_count"].sum()),
            "总缺失片段数": int(g["gap_segment_count"].sum()),
        }
        for col, label in metrics:
            stats = _metric_summary(g[col])
            row[f"{label} mean"] = stats["mean"]
            row[f"{label} std"] = stats["std"]
            row[f"{label} P25"] = stats["p25"]
            row[f"{label} P50"] = stats["p50"]
            row[f"{label} P75"] = stats["p75"]
            row[f"{label} P90"] = stats["p90"]
            row[f"{label} min"] = stats["min"]
            row[f"{label} max"] = stats["max"]
        gg = gap_df[gap_df["stage"].eq(stage)]
        gap_stats = _metric_summary(gg["gap_len_min"]) if not gg.empty else {}
        for key, val in gap_stats.items():
            row[f"单片段gap/min {key}"] = val
        rows.append(row)
    return pd.DataFrame(rows)


def _compact_summary(summary: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "测试集",
        "样本数",
        "总有效点数",
        "总锚点数",
        "总缺失点数",
        "总缺失片段数",
        "锚点数 mean",
        "锚点数 P25",
        "锚点数 P50",
        "锚点数 P75",
        "缺失率 mean",
        "缺失率 P25",
        "缺失率 P50",
        "缺失率 P75",
        "样本内平均gap/min mean",
        "样本内平均gap/min P50",
        "样本内最大gap/min mean",
        "样本内最大gap/min P50",
        "样本内最大gap/min P75",
        "样本内最大gap/min P90",
        "单片段gap/min mean",
        "单片段gap/min p50",
        "单片段gap/min p75",
        "单片段gap/min p90",
        "单片段gap/min max",
    ]
    return summary[[c for c in keep if c in summary.columns]].copy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--config-dir",
        default="outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_physical_time_ablation_v1/stagewise_eval/configs",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/experiments/obs_conditioned_gaponly/stage_dataset_condition_stats_20260519",
    )
    args = parser.parse_args()

    cfg_dir = _resolve(args.config_dir)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = {
        "S1": cfg_dir / "a3_risk_routed_stage1.yaml",
        "S2": cfg_dir / "a3_risk_routed_stage2.yaml",
        "S3": cfg_dir / "a3_risk_routed_stage3.yaml",
    }
    sample_rows: list[dict] = []
    gap_rows: list[dict] = []
    for stage, cfg_path in configs.items():
        rows, gaps = _sample_rows(stage, cfg_path, args.split)
        sample_rows.extend(rows)
        gap_rows.extend(gaps)

    sample_df = pd.DataFrame(sample_rows)
    gap_df = pd.DataFrame(gap_rows)
    summary = _build_summary(sample_df, gap_df)
    compact = _compact_summary(summary)

    sample_df.to_csv(out_dir / f"stage_{args.split}_sample_condition_stats.csv", index=False, encoding="utf-8-sig")
    gap_df.to_csv(out_dir / f"stage_{args.split}_gap_segment_stats.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / f"stage_{args.split}_condition_summary_full.csv", index=False, encoding="utf-8-sig")
    compact.to_csv(out_dir / f"stage_{args.split}_condition_summary_compact.csv", index=False, encoding="utf-8-sig")

    print(f"[done] out_dir={out_dir}")
    print(compact.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
