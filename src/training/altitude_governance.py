from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def add_anchor_alt_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    req = {"sample_id", "minute_ts", "obs_mask", "obs_alt"}
    if not req.issubset(frame.columns):
        return frame.copy()
    out = frame.sort_values(["sample_id", "minute_ts"]).copy()
    out["obs_mask"] = pd.to_numeric(out["obs_mask"], errors="coerce").fillna(0.0)
    out["obs_alt"] = pd.to_numeric(out["obs_alt"], errors="coerce").fillna(0.0)
    out["dt_prev"] = pd.to_numeric(out.get("dt_prev", 0.0), errors="coerce").fillna(0.0)
    out["dt_next"] = pd.to_numeric(out.get("dt_next", 0.0), errors="coerce").fillna(0.0)
    out["gap_len"] = pd.to_numeric(out.get("gap_len", out["dt_prev"] + out["dt_next"]), errors="coerce").fillna(
        out["dt_prev"] + out["dt_next"]
    )

    prev_col = []
    next_col = []
    for _, g in out.groupby("sample_id", sort=False):
        mask = g["obs_mask"].to_numpy() > 0.5
        obs_alt = g["obs_alt"].to_numpy(dtype=float)
        vals = np.where(mask, obs_alt, np.nan)
        prev = pd.Series(vals).ffill().bfill().to_numpy(dtype=float)
        nxt = pd.Series(vals).bfill().ffill().to_numpy(dtype=float)
        prev_col.extend(prev.tolist())
        next_col.extend(nxt.tolist())

    out["anchor_alt_prev"] = np.asarray(prev_col, dtype=float)
    out["anchor_alt_next"] = np.asarray(next_col, dtype=float)
    out["anchor_alt_delta"] = out["anchor_alt_next"] - out["anchor_alt_prev"]
    ratio = out["dt_prev"] / (out["gap_len"] + 1e-6)
    out["anchor_alt_interp"] = out["anchor_alt_prev"] + ratio * out["anchor_alt_delta"]
    out["alt_rel_prev_anchor"] = pd.to_numeric(out.get("alt", np.nan), errors="coerce") - out["anchor_alt_prev"]
    return out


def add_vertical_v2_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    dt_prev = pd.to_numeric(out.get("dt_prev", 0.0), errors="coerce").fillna(0.0)
    dt_next = pd.to_numeric(out.get("dt_next", 0.0), errors="coerce").fillna(0.0)
    gap_len = pd.to_numeric(out.get("gap_len", dt_prev + dt_next), errors="coerce").fillna(dt_prev + dt_next)
    r = dt_prev / (gap_len + 1e-6)
    one_minus_r = 1.0 - r

    left_alt = pd.to_numeric(out.get("anchor_alt_prev", 0.0), errors="coerce").fillna(0.0)
    right_alt = pd.to_numeric(out.get("anchor_alt_next", left_alt), errors="coerce").fillna(left_alt)
    delta_alt = right_alt - left_alt

    out["alt_linear_interp"] = left_alt + r * delta_alt
    out["v2_r"] = r
    out["v2_one_minus_r"] = one_minus_r
    out["v2_r_delta_alt"] = r * delta_alt
    out["v2_one_minus_r_delta_alt"] = one_minus_r * delta_alt
    out["v2_left_dist_norm"] = r
    out["v2_right_dist_norm"] = one_minus_r
    return out


def summarize_alt_distribution(frame: pd.DataFrame, split_name: str) -> list[dict]:
    out = []
    if frame.empty:
        return out
    for metric, col in [("alt", "alt"), ("alt_rel", "alt_rel_prev_anchor")]:
        if col not in frame.columns:
            continue
        x_all = pd.to_numeric(frame[col], errors="coerce").dropna().to_numpy(dtype=float)
        for scope, mask in [
            ("all", np.ones((len(frame),), dtype=bool)),
            ("gap", pd.to_numeric(frame["obs_mask"], errors="coerce").fillna(0.0).to_numpy() <= 0.5),
        ]:
            x = pd.to_numeric(frame.loc[mask, col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(x) == 0:
                out.append({"split": split_name, "metric": metric, "scope": scope, "count": 0})
                continue
            q = np.quantile(x, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
            out.append(
                {
                    "split": split_name,
                    "metric": metric,
                    "scope": scope,
                    "count": int(len(x)),
                    "mean": float(np.mean(x)),
                    "std": float(np.std(x)),
                    "min": float(np.min(x)),
                    "q01": float(q[0]),
                    "q05": float(q[1]),
                    "q25": float(q[2]),
                    "q50": float(q[3]),
                    "q75": float(q[4]),
                    "q95": float(q[5]),
                    "q99": float(q[6]),
                    "max": float(np.max(x)),
                }
            )
    return out


def compute_split_drift(dist_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ref = dist_df[(dist_df["split"] == "train") & (dist_df["metric"] == "alt_rel") & (dist_df["scope"] == "gap")]
    if ref.empty:
        return pd.DataFrame(rows)
    r = ref.iloc[0]
    for sp in ["val", "test"]:
        cur = dist_df[(dist_df["split"] == sp) & (dist_df["metric"] == "alt_rel") & (dist_df["scope"] == "gap")]
        if cur.empty:
            continue
        c = cur.iloc[0]
        rows.append(
            {
                "split": sp,
                "std_over_train": float(c.get("std", np.nan)) / (float(r.get("std", np.nan)) + 1e-9),
                "q95_over_train": float(c.get("q95", np.nan)) / (float(r.get("q95", np.nan)) + 1e-9),
                "q99_over_train": float(c.get("q99", np.nan)) / (float(r.get("q99", np.nan)) + 1e-9),
                "mean_diff_from_train": float(c.get("mean", np.nan)) - float(r.get("mean", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def apply_alt_label_governance(
    train_df: pd.DataFrame,
    cfg: dict,
    out_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    enabled = bool(cfg.get("enabled", False))
    report = {"enabled": enabled}
    if not enabled or train_df.empty:
        report["kept_samples_ratio"] = 1.0
        return train_df, report

    q = float(cfg.get("train_gap_abs_altrel_quantile", 0.995))
    mode = str(cfg.get("mode", "drop_sample_by_max_abs_altrel")).lower()
    out = train_df.copy()
    if mode == "drop_sample_by_max_abs_altrel":
        altrel = pd.to_numeric(train_df.get("alt_rel_prev_anchor", np.nan), errors="coerce")
        gap = pd.to_numeric(train_df.get("obs_mask", 0.0), errors="coerce").fillna(0.0) <= 0.5
        abs_gap = np.abs(altrel[gap].dropna().to_numpy(dtype=float))
        if len(abs_gap) == 0:
            report["note"] = "no_gap_points"
            report["kept_samples_ratio"] = 1.0
            return train_df, report
        thr = float(np.quantile(abs_gap, q))
        report["train_gap_abs_altrel_quantile"] = q
        report["train_gap_abs_altrel_threshold"] = thr
        agg = (
            out.assign(abs_altrel=np.abs(pd.to_numeric(out["alt_rel_prev_anchor"], errors="coerce")))
            .groupby("sample_id", as_index=False)["abs_altrel"]
            .max()
            .rename(columns={"abs_altrel": "sample_max_abs_altrel"})
        )
        keep_samples = agg.loc[agg["sample_max_abs_altrel"] <= thr, "sample_id"].astype(str).to_numpy()
        out = out[out["sample_id"].astype(str).isin(set(keep_samples))].copy()
    elif mode in {"drop_sample_by_outlier_csv", "drop_outlier_csv_only"}:
        report["mode"] = mode
    else:
        report["mode"] = mode

    outlier_csv = cfg.get("outlier_csv")
    if outlier_csv:
        p = Path(str(outlier_csv))
        if p.exists():
            odf = pd.read_csv(p)
            outlier_split = str(cfg.get("outlier_split", "train")).lower()
            if "split" in odf.columns and outlier_split in {"train", "val", "test"}:
                odf = odf[odf["split"].astype(str).str.lower().eq(outlier_split)].copy()
            outlier_scope = str(cfg.get("outlier_scope", "sample_and_flight")).lower()
            drop_samples = set(odf.loc[odf["kind"].astype(str).str.contains("sample"), "id"].astype(str).tolist())
            drop_flights = (
                set(odf.loc[odf["kind"].astype(str).str.contains("flight"), "id"].astype(str).tolist())
                if outlier_scope in {"sample_and_flight", "flight", "all"}
                else set()
            )
            if drop_samples:
                out = out[~out["sample_id"].astype(str).isin(drop_samples)].copy()
            if drop_flights and "flight_id" in out.columns:
                out = out[~out["flight_id"].astype(str).isin(drop_flights)].copy()
            report["outlier_csv_applied"] = str(p)
            report["outlier_split"] = outlier_split
            report["outlier_scope"] = outlier_scope

    n0 = int(train_df["sample_id"].astype(str).nunique())
    n1 = int(out["sample_id"].astype(str).nunique())
    report["samples_before"] = n0
    report["samples_after"] = n1
    report["kept_samples_ratio"] = float(n1) / max(1, float(n0))
    report["rows_before"] = int(len(train_df))
    report["rows_after"] = int(len(out))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "alt_label_governance_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out, report
