from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.training import load_config, set_seed, split_by_flight_id
from src.training.altitude_governance import (
    add_anchor_alt_features,
    add_vertical_v2_features,
    apply_alt_label_governance,
    compute_split_drift,
    summarize_alt_distribution,
)


DEFAULT_DELTA_BUCKETS = [0.0, 30.0, 100.0, 300.0, np.inf]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit SAVCA allocation/state labels under training-equivalent splits.")
    parser.add_argument("--config", required=True, help="Training config used for SAVCA.")
    parser.add_argument("--out-dir", default="", help="Output directory. Defaults to outputs/runs/0522/<config-stem>_savca_label_audit")
    parser.add_argument(
        "--delta-buckets",
        default="0,30,100,300,inf",
        help="Absolute anchor-delta buckets in meters, e.g. 0,30,100,300,inf",
    )
    parser.add_argument("--with-plots", action="store_true", help="Optional diagnostic plots. Default off.")
    return parser


def _parse_bucket_edges(spec: str) -> list[float]:
    out: list[float] = []
    for token in str(spec).split(","):
        t = token.strip().lower()
        if not t:
            continue
        if t in {"inf", "+inf", "infinity"}:
            out.append(float("inf"))
        else:
            out.append(float(t))
    if len(out) < 2:
        raise ValueError("delta-buckets must contain at least two edges")
    if out[0] != 0.0:
        out = [0.0] + out
    return out


def _bucket_label(lo: float, hi: float) -> str:
    if np.isinf(hi):
        return f"[{lo:.0f}, inf)"
    return f"[{lo:.0f}, {hi:.0f})"


def _median_smooth_1d(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or x.size <= 2:
        return x.astype(float, copy=True)
    half = int(window) // 2
    out = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        start = max(0, i - half)
        end = min(x.size, i + half + 1)
        out[i] = float(np.median(x[start:end]))
    return out


def _sample_has_anchor(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    obs = pd.to_numeric(frame["obs_mask"], errors="coerce").fillna(0.0)
    return frame.assign(_obs_anchor=(obs > 0.5)).groupby("sample_id")["_obs_anchor"].any().astype(bool)


def _anchor_gate_splits_local(splits: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict]:
    kept: dict[str, pd.DataFrame] = {}
    dropped: dict[str, pd.DataFrame] = {}
    audit: dict[str, dict] = {}
    for split_name, frame in splits.items():
        if frame.empty:
            kept[split_name] = frame.copy()
            dropped[split_name] = frame.copy()
            audit[split_name] = {
                "samples_total": 0,
                "samples_has_anchor_true": 0,
                "samples_has_anchor_false": 0,
                "ratio_has_anchor_false": 0.0,
                "rows_total": 0,
                "rows_has_anchor_true": 0,
                "rows_has_anchor_false": 0,
            }
            continue
        has_anchor = _sample_has_anchor(frame)
        keep_ids = set(has_anchor[has_anchor].index.astype(str).tolist())
        drop_ids = set(has_anchor[~has_anchor].index.astype(str).tolist())
        sid = frame["sample_id"].astype(str)
        keep_df = frame[sid.isin(keep_ids)].copy()
        drop_df = frame[sid.isin(drop_ids)].copy()
        kept[split_name] = keep_df
        dropped[split_name] = drop_df
        audit[split_name] = {
            "samples_total": int(len(has_anchor)),
            "samples_has_anchor_true": int(has_anchor.sum()),
            "samples_has_anchor_false": int((~has_anchor).sum()),
            "ratio_has_anchor_false": float((~has_anchor).mean()),
            "rows_total": int(len(frame)),
            "rows_has_anchor_true": int(len(keep_df)),
            "rows_has_anchor_false": int(len(drop_df)),
        }
    return kept, dropped, audit


def _build_training_like_splits(cfg: dict) -> tuple[dict[str, pd.DataFrame], dict]:
    curriculum_cfg = cfg["training"].get("curriculum", {})
    use_curriculum = bool(curriculum_cfg.get("enabled", False))
    split_cfg = cfg["data"]["split"]
    seed = int(cfg.get("seed", 42))
    flight_id_col = cfg["data"]["flight_id_col"]
    audit: dict[str, object] = {"use_curriculum": use_curriculum}

    if not use_curriculum:
        df = pd.read_parquet(cfg["data"]["samples_path"])
        splits = split_by_flight_id(
            df=df,
            flight_id_col=flight_id_col,
            train_ratio=float(split_cfg["train_ratio"]),
            val_ratio=float(split_cfg["val_ratio"]),
            seed=seed,
        )
        audit["source_mode"] = "single_dataset"
        audit["samples_path"] = str(cfg["data"]["samples_path"])
        return splits, audit

    stage_paths = curriculum_cfg.get("stage_paths", {})
    stage_frames = {k: pd.read_parquet(v) for k, v in stage_paths.items()}
    ref_df = stage_frames["stage1"]
    ref_splits = split_by_flight_id(
        df=ref_df,
        flight_id_col=flight_id_col,
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        seed=seed,
    )
    train_ids = set(ref_splits["train"][flight_id_col].astype(str).unique().tolist())
    val_ids = set(ref_splits["val"][flight_id_col].astype(str).unique().tolist())
    test_ids = set(ref_splits["test"][flight_id_col].astype(str).unique().tolist())
    val_stage = str(curriculum_cfg.get("val_stage", "stage3"))
    if val_stage not in stage_frames:
        raise RuntimeError(f"curriculum.val_stage={val_stage} not in stage_paths")

    train_frames = []
    for stage_name, stage_df in stage_frames.items():
        fid = stage_df[flight_id_col].astype(str)
        part = stage_df[fid.isin(train_ids)].copy()
        part["audit_stage"] = stage_name
        train_frames.append(part)

    val_df = stage_frames[val_stage]
    val_fid = val_df[flight_id_col].astype(str)
    splits = {
        "train": pd.concat(train_frames, ignore_index=True),
        "val": val_df[val_fid.isin(val_ids)].copy(),
        "test": val_df[val_fid.isin(test_ids)].copy(),
    }
    audit["source_mode"] = "curriculum"
    audit["val_stage"] = val_stage
    audit["stage_paths"] = {k: str(v) for k, v in stage_paths.items()}
    audit["train_flight_count"] = len(train_ids)
    audit["val_flight_count"] = len(val_ids)
    audit["test_flight_count"] = len(test_ids)
    return splits, audit


def _quantile_row(x: np.ndarray, prefix: str = "") -> dict[str, float]:
    if x.size == 0:
        return {
            f"{prefix}count": 0,
            f"{prefix}mean": float("nan"),
            f"{prefix}std": float("nan"),
            f"{prefix}min": float("nan"),
            f"{prefix}q25": float("nan"),
            f"{prefix}q50": float("nan"),
            f"{prefix}q75": float("nan"),
            f"{prefix}q90": float("nan"),
            f"{prefix}q95": float("nan"),
            f"{prefix}q99": float("nan"),
            f"{prefix}max": float("nan"),
        }
    q = np.quantile(x, [0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    return {
        f"{prefix}count": int(x.size),
        f"{prefix}mean": float(np.mean(x)),
        f"{prefix}std": float(np.std(x)),
        f"{prefix}min": float(np.min(x)),
        f"{prefix}q25": float(q[0]),
        f"{prefix}q50": float(q[1]),
        f"{prefix}q75": float(q[2]),
        f"{prefix}q90": float(q[3]),
        f"{prefix}q95": float(q[4]),
        f"{prefix}q99": float(q[5]),
        f"{prefix}max": float(np.max(x)),
    }


def _assign_delta_bucket(delta_abs: float, bucket_edges: list[float]) -> str:
    for lo, hi in zip(bucket_edges[:-1], bucket_edges[1:]):
        if delta_abs >= lo and delta_abs < hi:
            return _bucket_label(lo, hi)
    return _bucket_label(bucket_edges[-2], bucket_edges[-1])


def _compute_curriculum_sampling_summary(frame: pd.DataFrame, loss_cfg: dict, sampling_cfg: dict) -> pd.DataFrame:
    rows: list[dict] = []
    if frame.empty or "audit_stage" not in frame.columns:
        return pd.DataFrame(rows)
    state_thr = float(loss_cfg.get("savca_state_min_anchor_delta_m", 30.0))
    alloc_thr = float(loss_cfg.get("savca_alloc_min_anchor_delta_m", 100.0))
    long_gap_min = float(sampling_cfg.get("long_gap_min_minutes", 30.0))
    state_boost = float(max(1.0, sampling_cfg.get("state_boost", 1.5)))
    alloc_boost = float(max(1.0, sampling_cfg.get("alloc_boost", 3.0)))
    long_gap_boost = float(max(1.0, sampling_cfg.get("long_gap_boost", 1.5)))

    prof_rows: list[dict] = []
    for (stage_name, sample_id), g in frame.groupby(["audit_stage", "sample_id"], sort=False):
        gap = pd.to_numeric(g.get("obs_mask", 0.0), errors="coerce").fillna(0.0) <= 0.5
        gx = g.loc[gap].copy()
        if gx.empty:
            anchor_delta = 0.0
            gap_len = 0.0
        else:
            anchor_delta = float(
                pd.to_numeric(gx.get("anchor_alt_delta", 0.0), errors="coerce")
                .abs()
                .fillna(0.0)
                .max()
            )
            gap_len = float(pd.to_numeric(gx.get("gap_len", 0.0), errors="coerce").fillna(0.0).max())
        state_eligible = bool(anchor_delta >= state_thr)
        alloc_eligible = bool(anchor_delta >= alloc_thr)
        w = 1.0
        if state_eligible:
            w *= state_boost
        if alloc_eligible:
            w *= alloc_boost
        if alloc_eligible and gap_len >= long_gap_min:
            w *= long_gap_boost
        if anchor_delta < state_thr:
            bucket = f"[0,{int(state_thr)})"
        elif anchor_delta < alloc_thr:
            bucket = f"[{int(state_thr)},{int(alloc_thr)})"
        else:
            bucket = f"[{int(alloc_thr)},inf)"
        prof_rows.append(
            {
                "stage": str(stage_name),
                "sample_id": str(sample_id),
                "delta_bucket": bucket,
                "state_eligible": int(state_eligible),
                "alloc_eligible": int(alloc_eligible),
                "sample_weight": float(w),
            }
        )
    prof = pd.DataFrame(prof_rows)
    if prof.empty:
        return prof
    for (stage_name, bucket), part in prof.groupby(["stage", "delta_bucket"], sort=False):
        stage_mask = prof["stage"].astype(str).eq(str(stage_name))
        total_count = float(stage_mask.sum())
        total_weight = float(prof.loc[stage_mask, "sample_weight"].sum()) + 1e-9
        rows.append(
            {
                "stage": str(stage_name),
                "delta_bucket": str(bucket),
                "sample_count": int(len(part)),
                "sample_ratio": float(len(part) / total_count),
                "weight_mass": float(part["sample_weight"].sum()),
                "weight_ratio": float(part["sample_weight"].sum() / total_weight),
                "state_eligible_ratio": float(part["state_eligible"].mean()),
                "alloc_eligible_ratio": float(part["alloc_eligible"].mean()),
                "weight_mean": float(part["sample_weight"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _extract_savca_labels_for_split(
    frame: pd.DataFrame,
    split_name: str,
    *,
    state_min_anchor_delta_m: float,
    alloc_min_anchor_delta_m: float,
    active_min_anchor_delta_m: float,
    deadband_m: float,
    median_window: int,
    active_ratio_to_max: float,
    active_min_abs_change_m: float,
    active_expand_steps: int,
    bucket_edges: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows_step: list[dict] = []
    rows_seg: list[dict] = []
    work = frame.sort_values(["sample_id", "minute_ts"]).copy()

    for sample_id, g in work.groupby("sample_id", sort=False):
        g = g.reset_index(drop=True)
        obs_mask = pd.to_numeric(g["obs_mask"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        z = pd.to_numeric(g["alt"], errors="coerce").to_numpy(dtype=float)
        ts = g["minute_ts"].astype(str).to_numpy()
        gap_len_arr = pd.to_numeric(g.get("gap_len", np.nan), errors="coerce").to_numpy(dtype=float)
        gap_pos_arr = pd.to_numeric(g.get("gap_pos_ratio", np.nan), errors="coerce").to_numpy(dtype=float)
        anchor_prev_arr = pd.to_numeric(g.get("anchor_alt_prev", np.nan), errors="coerce").to_numpy(dtype=float)
        anchor_next_arr = pd.to_numeric(g.get("anchor_alt_next", np.nan), errors="coerce").to_numpy(dtype=float)
        audit_stage_arr = g["audit_stage"].astype(str).to_numpy() if "audit_stage" in g.columns else np.array([""] * len(g))
        flight_id = str(g["flight_id"].iloc[0]) if "flight_id" in g.columns else ""

        anchors = np.where(obs_mask > 0.5)[0]
        if anchors.size < 2:
            continue

        for seg_idx, (left, right) in enumerate(zip(anchors[:-1], anchors[1:]), start=1):
            if right <= left:
                continue
            interval = np.arange(left + 1, right + 1, dtype=int)
            if interval.size < 2:
                continue

            z_seg = _median_smooth_1d(z[left : right + 1], median_window)
            diffs = np.abs(np.diff(z_seg))
            if deadband_m > 0.0:
                diffs = np.where(diffs >= deadband_m, diffs, 0.0)

            anchor_delta = float(abs(z[right] - z[left]))
            diff_sum = float(diffs.sum())
            diff_max = float(diffs.max()) if diffs.size else 0.0
            delta_bucket = _assign_delta_bucket(anchor_delta, bucket_edges)

            active_mask = np.zeros_like(diffs, dtype=bool)
            if (
                anchor_delta >= active_min_anchor_delta_m
                and diff_sum > 1e-6
                and diff_max > 1e-6
            ):
                active_thr = max(
                    active_min_abs_change_m,
                    deadband_m,
                    active_ratio_to_max * diff_max,
                )
                active_mask = diffs >= active_thr
                if active_expand_steps > 0 and bool(active_mask.any()):
                    mask = active_mask.copy()
                    for _ in range(int(active_expand_steps)):
                        prev = np.concatenate([mask[:1], mask[:-1]])
                        nxt = np.concatenate([mask[1:], mask[-1:]])
                        mask = mask | prev | nxt
                    active_mask = mask

            active_diffs = np.where(active_mask, diffs, 0.0)
            active_sum = float(active_diffs.sum())
            state_supervised = bool(anchor_delta >= state_min_anchor_delta_m and active_mask.any())
            alloc_supervised = bool(anchor_delta >= alloc_min_anchor_delta_m and active_mask.sum() >= 2 and active_sum > 1e-6)
            any_supervised = bool(state_supervised or alloc_supervised)
            q = active_diffs / (active_sum + 1e-6) if active_sum > 1e-6 else np.zeros_like(active_diffs)
            y_state = active_mask.astype(float)

            rows_seg.append(
                {
                    "split": split_name,
                    "sample_id": str(sample_id),
                    "flight_id": flight_id,
                    "audit_stage": str(audit_stage_arr[left]),
                    "segment_index": seg_idx,
                    "left_index": int(left),
                    "right_index": int(right),
                    "left_minute_ts": str(ts[left]),
                    "right_minute_ts": str(ts[right]),
                    "segment_len": int(right - left),
                    "anchor_alt_left_m": float(z[left]),
                    "anchor_alt_right_m": float(z[right]),
                    "anchor_delta_abs_m": anchor_delta,
                    "anchor_delta_bucket": delta_bucket,
                    "diff_sum_m": diff_sum,
                    "diff_max_m": diff_max,
                    "active_step_count": int(active_mask.sum()),
                    "active_ratio": float(active_mask.mean()) if active_mask.size else 0.0,
                    "state_supervised": int(state_supervised),
                    "alloc_supervised": int(alloc_supervised),
                    "supervised": int(any_supervised),
                    "gap_len_left_row": float(gap_len_arr[left]) if np.isfinite(gap_len_arr[left]) else float("nan"),
                    "gap_pos_left_row": float(gap_pos_arr[left]) if np.isfinite(gap_pos_arr[left]) else float("nan"),
                }
            )

            for local_i, idx in enumerate(interval):
                rows_step.append(
                    {
                        "split": split_name,
                        "sample_id": str(sample_id),
                        "flight_id": flight_id,
                        "audit_stage": str(audit_stage_arr[idx]),
                        "segment_index": seg_idx,
                        "step_offset": int(local_i + 1),
                        "minute_ts": str(ts[idx]),
                        "segment_len": int(right - left),
                        "anchor_alt_left_m": float(z[left]),
                        "anchor_alt_right_m": float(z[right]),
                        "anchor_delta_abs_m": anchor_delta,
                        "anchor_delta_bucket": delta_bucket,
                        "supervised": int(any_supervised),
                        "state_supervised": int(state_supervised),
                        "alloc_supervised": int(alloc_supervised),
                        "active_mask": int(active_mask[local_i]) if local_i < active_mask.size else 0,
                        "alloc_q": float(q[local_i]) if local_i < q.size else 0.0,
                        "state_y": float(y_state[local_i]) if local_i < y_state.size else 0.0,
                        "diff_abs_m": float(diffs[local_i]) if local_i < diffs.size else 0.0,
                        "gap_len_row": float(gap_len_arr[idx]) if np.isfinite(gap_len_arr[idx]) else float("nan"),
                        "gap_pos_ratio_row": float(gap_pos_arr[idx]) if np.isfinite(gap_pos_arr[idx]) else float("nan"),
                        "anchor_alt_prev_row": float(anchor_prev_arr[idx]) if np.isfinite(anchor_prev_arr[idx]) else float("nan"),
                        "anchor_alt_next_row": float(anchor_next_arr[idx]) if np.isfinite(anchor_next_arr[idx]) else float("nan"),
                    }
                )

    return pd.DataFrame(rows_step), pd.DataFrame(rows_seg)


def _compute_label_distribution(step_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    if step_df.empty:
        return pd.DataFrame(rows)
    for split_name, sdf in step_df.groupby("split", sort=False):
        for metric in ["alloc_q", "state_y", "diff_abs_m", "anchor_delta_abs_m"]:
            scopes = [("all", np.ones((len(sdf),), dtype=bool))]
            if metric == "alloc_q":
                scopes.append(("supervised", sdf["alloc_supervised"].to_numpy(dtype=int) > 0))
            elif metric == "state_y":
                scopes.append(("supervised", sdf["state_supervised"].to_numpy(dtype=int) > 0))
            else:
                scopes.append(("supervised", sdf["supervised"].to_numpy(dtype=int) > 0))
            for scope_name, mask in scopes:
                x = pd.to_numeric(sdf.loc[mask, metric], errors="coerce").dropna().to_numpy(dtype=float)
                row = {"split": split_name, "metric": metric, "scope": scope_name}
                row.update(_quantile_row(x))
                rows.append(row)
    return pd.DataFrame(rows)


def _compute_label_drift(dist_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    if dist_df.empty:
        return pd.DataFrame(rows)
    for metric in sorted(dist_df["metric"].astype(str).unique().tolist()):
        ref = dist_df[
            (dist_df["split"] == "train")
            & (dist_df["metric"] == metric)
            & (dist_df["scope"] == "supervised")
        ]
        if ref.empty:
            continue
        r = ref.iloc[0]
        for split_name in ["val", "test"]:
            cur = dist_df[
                (dist_df["split"] == split_name)
                & (dist_df["metric"] == metric)
                & (dist_df["scope"] == "supervised")
            ]
            if cur.empty:
                continue
            c = cur.iloc[0]
            rows.append(
                {
                    "metric": metric,
                    "split": split_name,
                    "std_over_train": float(c["std"]) / (float(r["std"]) + 1e-9),
                    "q95_over_train": float(c["q95"]) / (float(r["q95"]) + 1e-9),
                    "q99_over_train": float(c["q99"]) / (float(r["q99"]) + 1e-9),
                    "mean_diff_from_train": float(c["mean"]) - float(r["mean"]),
                }
            )
    return pd.DataFrame(rows)


def _compute_bucket_summary(step_df: pd.DataFrame, seg_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    if seg_df.empty:
        return pd.DataFrame(rows)
    for (split_name, bucket), seg_part in seg_df.groupby(["split", "anchor_delta_bucket"], sort=False):
        step_part = step_df[
            (step_df["split"].astype(str) == str(split_name))
            & (step_df["anchor_delta_bucket"].astype(str) == str(bucket))
        ].copy()
        step_sup = step_part[step_part["supervised"] > 0].copy()
        row = {
            "split": split_name,
            "anchor_delta_bucket": bucket,
            "segment_count": int(len(seg_part)),
            "supervised_segment_count": int(seg_part["supervised"].sum()),
            "supervised_segment_ratio": float(seg_part["supervised"].mean()) if len(seg_part) else float("nan"),
            "state_supervised_segment_count": int(seg_part["state_supervised"].sum()),
            "state_supervised_segment_ratio": float(seg_part["state_supervised"].mean()) if len(seg_part) else float("nan"),
            "alloc_supervised_segment_count": int(seg_part["alloc_supervised"].sum()),
            "alloc_supervised_segment_ratio": float(seg_part["alloc_supervised"].mean()) if len(seg_part) else float("nan"),
            "step_count": int(len(step_part)),
            "supervised_step_count": int(len(step_sup)),
        }
        row.update(_quantile_row(pd.to_numeric(seg_part["segment_len"], errors="coerce").dropna().to_numpy(dtype=float), "segment_len_"))
        row.update(_quantile_row(pd.to_numeric(seg_part["anchor_delta_abs_m"], errors="coerce").dropna().to_numpy(dtype=float), "anchor_delta_abs_m_"))
        row.update(_quantile_row(pd.to_numeric(seg_part["diff_sum_m"], errors="coerce").dropna().to_numpy(dtype=float), "diff_sum_m_"))
        row.update(_quantile_row(pd.to_numeric(seg_part["diff_max_m"], errors="coerce").dropna().to_numpy(dtype=float), "diff_max_m_"))
        row.update(_quantile_row(pd.to_numeric(seg_part["active_step_count"], errors="coerce").dropna().to_numpy(dtype=float), "active_step_count_"))
        row.update(_quantile_row(pd.to_numeric(seg_part["active_ratio"], errors="coerce").dropna().to_numpy(dtype=float), "active_ratio_"))
        row.update(_quantile_row(pd.to_numeric(step_sup["alloc_q"], errors="coerce").dropna().to_numpy(dtype=float), "alloc_q_"))
        row.update(_quantile_row(pd.to_numeric(step_sup["state_y"], errors="coerce").dropna().to_numpy(dtype=float), "state_y_"))
        rows.append(row)
    return pd.DataFrame(rows)


def _build_key_summary(label_dist: pd.DataFrame, label_drift: pd.DataFrame, bucket_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    if not label_dist.empty:
        for metric in ["alloc_q", "state_y", "diff_abs_m", "anchor_delta_abs_m"]:
            part = label_dist[(label_dist["metric"] == metric) & (label_dist["scope"] == "supervised")].copy()
            if part.empty:
                continue
            row = {"block": "supervised_label_distribution", "metric": metric}
            for split_name in ["train", "val", "test"]:
                cur = part[part["split"] == split_name]
                if cur.empty:
                    continue
                c = cur.iloc[0]
                row[f"{split_name}_count"] = int(c["count"])
                row[f"{split_name}_mean"] = float(c["mean"])
                row[f"{split_name}_q95"] = float(c["q95"])
                row[f"{split_name}_q99"] = float(c["q99"])
            rows.append(row)
    if not label_drift.empty:
        for _, r in label_drift.iterrows():
            rows.append(
                {
                    "block": "drift_vs_train",
                    "metric": str(r["metric"]),
                    "split": str(r["split"]),
                    "std_over_train": float(r["std_over_train"]),
                    "q95_over_train": float(r["q95_over_train"]),
                    "q99_over_train": float(r["q99_over_train"]),
                    "mean_diff_from_train": float(r["mean_diff_from_train"]),
                }
            )
    if not bucket_summary.empty:
        for split_name in ["train", "val", "test"]:
            part = bucket_summary[bucket_summary["split"] == split_name].copy()
            if part.empty:
                continue
            row = {"block": "bucket_supervision_ratio", "split": split_name}
            for _, r in part.iterrows():
                row[f"ratio_{r['anchor_delta_bucket']}"] = float(r["supervised_segment_ratio"])
                row[f"state_ratio_{r['anchor_delta_bucket']}"] = float(r["state_supervised_segment_ratio"])
                row[f"alloc_ratio_{r['anchor_delta_bucket']}"] = float(r["alloc_supervised_segment_ratio"])
                row[f"count_{r['anchor_delta_bucket']}"] = int(r["segment_count"])
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    out_dir = Path(args.out_dir) if args.out_dir else Path(
        f"outputs/runs/0522/{Path(args.config).stem}_savca_label_audit"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    bucket_edges = _parse_bucket_edges(args.delta_buckets)

    splits_raw, split_meta = _build_training_like_splits(cfg)
    for split_name in list(splits_raw.keys()):
        splits_raw[split_name] = add_anchor_alt_features(splits_raw[split_name])
        splits_raw[split_name] = add_vertical_v2_features(splits_raw[split_name])

    splits_anchor_kept, splits_anchor_dropped, anchor_audit = _anchor_gate_splits_local(splits_raw)
    lg_cfg = cfg["training"].get("alt_label_governance", {})
    train_post, lg_report = apply_alt_label_governance(
        splits_anchor_kept.get("train", pd.DataFrame()),
        lg_cfg,
        out_dir=out_dir,
    )
    splits_ready = {
        "train": train_post,
        "val": splits_anchor_kept.get("val", pd.DataFrame()).copy(),
        "test": splits_anchor_kept.get("test", pd.DataFrame()).copy(),
    }

    alt_rows_raw: list[dict] = []
    alt_rows_ready: list[dict] = []
    for split_name in ["train", "val", "test"]:
        if split_name in splits_raw:
            alt_rows_raw.extend(summarize_alt_distribution(splits_raw[split_name], split_name=split_name))
        if split_name in splits_ready:
            alt_rows_ready.extend(summarize_alt_distribution(splits_ready[split_name], split_name=split_name))
    alt_df_raw = pd.DataFrame(alt_rows_raw)
    alt_df_ready = pd.DataFrame(alt_rows_ready)
    alt_df_raw.to_csv(out_dir / "alt_distribution_raw_by_split.csv", index=False)
    alt_df_ready.to_csv(out_dir / "alt_distribution_training_ready_by_split.csv", index=False)
    compute_split_drift(alt_df_ready).to_csv(out_dir / "alt_rel_training_ready_split_drift.csv", index=False)

    loss_cfg = cfg.get("loss", {})
    state_min_anchor_delta_m = float(loss_cfg.get("savca_state_min_anchor_delta_m", 30.0))
    alloc_min_anchor_delta_m = float(loss_cfg.get("savca_alloc_min_anchor_delta_m", 100.0))
    active_min_anchor_delta_m = float(loss_cfg.get("savca_active_min_anchor_delta_m", 30.0))
    deadband_m = float(loss_cfg.get("savca_change_deadband_m", 3.0))
    median_window = int(loss_cfg.get("savca_label_median_window", 5))
    active_ratio_to_max = float(loss_cfg.get("savca_active_ratio_to_max", 0.25))
    active_min_abs_change_m = float(loss_cfg.get("savca_active_min_abs_change_m", 10.0))
    active_expand_steps = int(loss_cfg.get("savca_active_expand_steps", 1))

    step_parts = []
    seg_parts = []
    for split_name in ["train", "val", "test"]:
        step_df, seg_df = _extract_savca_labels_for_split(
            splits_ready.get(split_name, pd.DataFrame()),
            split_name,
            state_min_anchor_delta_m=state_min_anchor_delta_m,
            alloc_min_anchor_delta_m=alloc_min_anchor_delta_m,
            active_min_anchor_delta_m=active_min_anchor_delta_m,
            deadband_m=deadband_m,
            median_window=median_window,
            active_ratio_to_max=active_ratio_to_max,
            active_min_abs_change_m=active_min_abs_change_m,
            active_expand_steps=active_expand_steps,
            bucket_edges=bucket_edges,
        )
        step_parts.append(step_df)
        seg_parts.append(seg_df)
    step_all = pd.concat(step_parts, ignore_index=True) if any(not x.empty for x in step_parts) else pd.DataFrame()
    seg_all = pd.concat(seg_parts, ignore_index=True) if any(not x.empty for x in seg_parts) else pd.DataFrame()

    if not step_all.empty:
        step_all.to_parquet(out_dir / "savca_step_labels_training_ready.parquet", index=False)
    if not seg_all.empty:
        seg_all.to_csv(out_dir / "savca_segment_labels_training_ready.csv", index=False)

    label_dist = _compute_label_distribution(step_all)
    label_drift = _compute_label_drift(label_dist)
    bucket_summary = _compute_bucket_summary(step_all, seg_all)
    label_dist.to_csv(out_dir / "savca_label_distribution_by_split.csv", index=False)
    label_drift.to_csv(out_dir / "savca_label_split_drift.csv", index=False)
    bucket_summary.to_csv(out_dir / "savca_label_bucket_summary.csv", index=False)
    _build_key_summary(label_dist, label_drift, bucket_summary).to_csv(
        out_dir / "savca_label_key_summary.csv",
        index=False,
    )

    curriculum_sampling_cfg = cfg["training"].get("curriculum", {}).get("savca_sampling", {})
    curriculum_sampling_summary = _compute_curriculum_sampling_summary(
        splits_ready.get("train", pd.DataFrame()),
        loss_cfg=loss_cfg,
        sampling_cfg=curriculum_sampling_cfg,
    )
    if not curriculum_sampling_summary.empty:
        curriculum_sampling_summary.to_csv(out_dir / "curriculum_stage_sampling_summary.csv", index=False)

    meta = {
        "config": str(args.config),
        "output_dir": str(out_dir),
        "split_meta": split_meta,
        "anchor_gate_audit": anchor_audit,
        "alt_label_governance_report": lg_report,
        "savca_label_params": {
            "savca_state_min_anchor_delta_m": state_min_anchor_delta_m,
            "savca_alloc_min_anchor_delta_m": alloc_min_anchor_delta_m,
            "savca_active_min_anchor_delta_m": active_min_anchor_delta_m,
            "savca_change_deadband_m": deadband_m,
            "savca_label_median_window": median_window,
            "savca_active_ratio_to_max": active_ratio_to_max,
            "savca_active_min_abs_change_m": active_min_abs_change_m,
            "savca_active_expand_steps": active_expand_steps,
            "delta_buckets": [_bucket_label(lo, hi) for lo, hi in zip(bucket_edges[:-1], bucket_edges[1:])],
        },
        "training_ready_row_counts": {k: int(len(v)) for k, v in splits_ready.items()},
        "training_ready_sample_counts": {k: int(v["sample_id"].astype(str).nunique()) if not v.empty else 0 for k, v in splits_ready.items()},
        "curriculum_savca_sampling_enabled": bool(curriculum_sampling_cfg.get("enabled", False)),
        "with_plots": bool(args.with_plots),
    }
    (out_dir / "audit_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[ok] SAVCA label audit written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
