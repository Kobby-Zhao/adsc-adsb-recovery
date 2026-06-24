from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


STRUCTURAL_TIME_COLS = frozenset(
    {
        "dt_prev",
        "dt_next",
        "gap_len",
        "gap_pos_ratio",
    }
)


def _is_binary_like(series: pd.Series) -> bool:
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return False
    uniq = np.unique(vals)
    if uniq.size > 2:
        return False
    return set(np.round(uniq, 6).tolist()).issubset({0.0, 1.0})


def select_continuous_feature_cols(frame: pd.DataFrame, candidate_cols: list[str], exclude_cols: set[str] | None = None) -> list[str]:
    exclude = exclude_cols or set()
    cols: list[str] = []
    for col in dict.fromkeys(candidate_cols):
        # These are structural recovery coordinates, not generic continuous
        # features. Fusion, altitude baseline, and gap routing all require their
        # physical units/range (minutes and [0, 1] position) to remain intact.
        if col in STRUCTURAL_TIME_COLS:
            continue
        if col in exclude or col not in frame.columns:
            continue
        if _is_binary_like(frame[col]):
            continue
        vals = pd.to_numeric(frame[col], errors="coerce")
        if vals.notna().sum() == 0:
            continue
        cols.append(col)
    return cols


def fit_standardizer(train_frame: pd.DataFrame, feature_cols: list[str], eps: float = 1e-6) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for col in feature_cols:
        raw = pd.to_numeric(train_frame[col], errors="coerce")
        vals = raw.to_numpy(dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        mean = float(vals.mean())
        raw_std = float(vals.std(ddof=0))
        used_std = raw_std
        small_std = (not np.isfinite(raw_std)) or (raw_std < eps)
        if small_std:
            used_std = 1.0
        stats[col] = {
            "mean": mean,
            "std": float(used_std),
            "raw_std": float(raw_std) if np.isfinite(raw_std) else float("nan"),
            "small_std": bool(small_std),
            "n_non_null": int(raw.notna().sum()),
            "n_missing": int(raw.isna().sum()),
        }
    return stats


def apply_standardizer(frame: pd.DataFrame, stats: dict[str, dict[str, float]], fillna_value: float = 0.0) -> pd.DataFrame:
    out = frame.copy()
    for col, ms in stats.items():
        # Backward-compatible guard: older scaler JSON files may contain these
        # columns. Do not standardize them during future evaluation/replay.
        if col in STRUCTURAL_TIME_COLS:
            continue
        if col not in out.columns:
            continue
        vals = pd.to_numeric(out[col], errors="coerce")
        z = (vals - float(ms["mean"])) / float(ms["std"])
        out[col] = z.fillna(fillna_value)
    return out


def build_standardization_report(before_train: pd.DataFrame, after_train: pd.DataFrame, stats: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows: list[dict] = []
    for col in stats.keys():
        b = pd.to_numeric(before_train[col], errors="coerce")
        a = pd.to_numeric(after_train[col], errors="coerce")
        rows.append(
            {
                "feature": col,
                "mean_before": float(b.mean(skipna=True)),
                "std_before": float(b.std(skipna=True, ddof=0)),
                "mean_after": float(a.mean(skipna=True)),
                "std_after": float(a.std(skipna=True, ddof=0)),
                "missing_before": int(b.isna().sum()),
                "missing_after": int(a.isna().sum()),
                "small_std_replaced": bool(stats[col].get("small_std", False)),
                "raw_std": float(stats[col].get("raw_std", float("nan"))),
                "std_used": float(stats[col]["std"]),
                "n_non_null_train": int(stats[col].get("n_non_null", 0)),
                "n_missing_train": int(stats[col].get("n_missing", 0)),
            }
        )
    return pd.DataFrame(rows)


def save_standardizer(stats: dict[str, dict[str, float]], out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def load_standardizer(path: str | Path) -> dict[str, dict[str, float]] | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))
