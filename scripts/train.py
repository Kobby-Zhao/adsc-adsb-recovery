from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.datasets import DatasetConfig, TrajectoryDataset, trajectory_collate_fn
from src.losses import TrajectoryLoss
from src.models import TrajectoryRecoveryModel
from src.preprocessing.standardize import (
    apply_standardizer,
    build_standardization_report,
    fit_standardizer,
    save_standardizer,
    select_continuous_feature_cols,
)
from src.training import Trainer, load_config, set_seed, split_by_flight_id
from src.training.altitude_governance import (
    add_anchor_alt_features,
    add_vertical_v2_features,
    apply_alt_label_governance,
    compute_split_drift,
    summarize_alt_distribution,
)
from src.training.target_norm import compute_target_stats_from_loader, save_target_stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train trajectory recovery model.")
    parser.add_argument("--config", default="configs/train.yaml")
    return parser


def _normalize_stage_weights(weights: dict[str, float]) -> dict[str, float]:
    out = {str(k): max(0.0, float(v)) for k, v in (weights or {}).items()}
    s = float(sum(out.values()))
    if s <= 0:
        return {"stage1": 1.0, "stage2": 0.0, "stage3": 0.0}
    return {k: v / s for k, v in out.items()}


def _pick_schedule_weights(epoch: int, schedule: list[dict]) -> dict[str, float]:
    if not schedule:
        return {"stage1": 1.0, "stage2": 0.0, "stage3": 0.0}
    for item in schedule:
        if int(epoch) <= int(item.get("end_epoch", 0)):
            return _normalize_stage_weights(item.get("weights", {}))
    return _normalize_stage_weights(schedule[-1].get("weights", {}))


def _interp_scalar_schedule(epoch: int, schedule_epochs: list[int], schedule_values: list[float]) -> float:
    if not schedule_epochs or not schedule_values or len(schedule_epochs) != len(schedule_values):
        raise ValueError("savca_beta_schedule requires same-length epochs and values.")
    if epoch <= int(schedule_epochs[0]):
        return float(schedule_values[0])
    for i in range(1, len(schedule_epochs)):
        e0 = int(schedule_epochs[i - 1])
        e1 = int(schedule_epochs[i])
        v0 = float(schedule_values[i - 1])
        v1 = float(schedule_values[i])
        if epoch <= e1:
            if e1 <= e0:
                return v1
            r = float(epoch - e0) / float(e1 - e0)
            return v0 + r * (v1 - v0)
    return float(schedule_values[-1])


def _build_stage_split_audit(
    stage_frames: dict[str, pd.DataFrame],
    flight_id_col: str,
    train_ids: set[str],
    val_ids: set[str],
    test_ids: set[str],
) -> pd.DataFrame:
    rows: list[dict] = []
    split_to_ids = {
        "train": set(train_ids),
        "val": set(val_ids),
        "test": set(test_ids),
    }
    for stage, sdf in stage_frames.items():
        fid = sdf[flight_id_col].astype(str)
        all_flights = set(fid.unique().tolist())
        for split_name, keep_ids in split_to_ids.items():
            mask = fid.isin(keep_ids)
            sub = sdf[mask]
            rows.append(
                {
                    "stage": stage,
                    "split": split_name,
                    "rows": int(len(sub)),
                    "samples": int(sub["sample_id"].astype(str).nunique()) if "sample_id" in sub.columns else int(len(sub)),
                    "flights": int(sub[flight_id_col].astype(str).nunique()),
                    "coverage_vs_stage_flights": float(
                        (sub[flight_id_col].astype(str).nunique() / max(len(all_flights), 1))
                    ),
                }
            )
    return pd.DataFrame(rows)


def _gap_lengths(mask_series: pd.Series) -> list[int]:
    arr = pd.to_numeric(mask_series, errors="coerce").fillna(0.0).to_numpy()
    out: list[int] = []
    cur = 0
    for v in arr:
        if v < 0.5:
            cur += 1
        elif cur > 0:
            out.append(cur)
            cur = 0
    if cur > 0:
        out.append(cur)
    return out


def _sample_ids(
    ids: np.ndarray,
    n: int,
    rng: np.random.Generator,
    probs: np.ndarray | None = None,
) -> np.ndarray:
    if n <= 0 or len(ids) == 0:
        return np.array([], dtype=object)
    replace = n > len(ids)
    if probs is None:
        return rng.choice(ids, size=n, replace=replace)
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 1 or len(p) != len(ids):
        raise ValueError("sampling probs must be 1D and aligned with ids")
    p = np.where(np.isfinite(p) & (p > 0.0), p, 0.0)
    s = float(p.sum())
    if s <= 0.0:
        return rng.choice(ids, size=n, replace=replace)
    p = p / s
    return rng.choice(ids, size=n, replace=replace, p=p)


def _filter_frame_by_sample_ids(frame: pd.DataFrame, keep_sample_ids: set[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame[frame["sample_id"].astype(str).isin(keep_sample_ids)].copy()


def _build_curriculum_savca_sampling_profile(
    frame: pd.DataFrame,
    loss_cfg: dict,
    sampling_cfg: dict,
) -> pd.DataFrame:
    cols = [
        "sample_id",
        "max_anchor_delta_abs_m",
        "max_gap_len_min",
        "state_eligible",
        "alloc_eligible",
        "sample_weight",
        "delta_bucket",
    ]
    if frame.empty:
        return pd.DataFrame(columns=cols)

    state_thr = float(loss_cfg.get("savca_state_min_anchor_delta_m", 30.0))
    alloc_thr = float(loss_cfg.get("savca_alloc_min_anchor_delta_m", 100.0))
    long_gap_min = float(sampling_cfg.get("long_gap_min_minutes", 30.0))
    state_boost = float(max(1.0, sampling_cfg.get("state_boost", 1.5)))
    alloc_boost = float(max(1.0, sampling_cfg.get("alloc_boost", 3.0)))
    long_gap_boost = float(max(1.0, sampling_cfg.get("long_gap_boost", 1.5)))

    out_rows: list[dict] = []
    for sample_id, g in frame.groupby("sample_id", sort=False):
        gx = g.copy()
        obs = pd.to_numeric(gx.get("obs_mask", 0.0), errors="coerce").fillna(0.0) <= 0.5
        gap_part = gx.loc[obs].copy()
        if gap_part.empty:
            anchor_delta = 0.0
            gap_len = 0.0
        else:
            anchor_delta = float(
                pd.to_numeric(gap_part.get("anchor_alt_delta", 0.0), errors="coerce")
                .abs()
                .fillna(0.0)
                .max()
            )
            gap_len = float(pd.to_numeric(gap_part.get("gap_len", 0.0), errors="coerce").fillna(0.0).max())
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
        out_rows.append(
            {
                "sample_id": str(sample_id),
                "max_anchor_delta_abs_m": anchor_delta,
                "max_gap_len_min": gap_len,
                "state_eligible": int(state_eligible),
                "alloc_eligible": int(alloc_eligible),
                "sample_weight": float(w),
                "delta_bucket": bucket,
            }
        )
    return pd.DataFrame(out_rows, columns=cols)


def _summarize_curriculum_sampling_profile(
    profile: pd.DataFrame,
    stage_name: str,
) -> pd.DataFrame:
    if profile.empty:
        return pd.DataFrame(
            [
                {
                    "stage": stage_name,
                    "delta_bucket": "empty",
                    "sample_count": 0,
                    "sample_ratio": 0.0,
                    "weight_mass": 0.0,
                    "weight_ratio": 0.0,
                    "weight_mean": 0.0,
                }
            ]
        )
    total_count = float(len(profile))
    total_weight = float(profile["sample_weight"].sum()) + 1e-9
    rows = []
    for bucket, part in profile.groupby("delta_bucket", sort=False):
        rows.append(
            {
                "stage": stage_name,
                "delta_bucket": str(bucket),
                "sample_count": int(len(part)),
                "sample_ratio": float(len(part) / total_count),
                "weight_mass": float(part["sample_weight"].sum()),
                "weight_ratio": float(part["sample_weight"].sum() / total_weight),
                "weight_mean": float(part["sample_weight"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _compute_alt_base_residual_bounds(train_df: pd.DataFrame, alt_col: str) -> list[float]:
    if train_df.empty or alt_col not in train_df.columns:
        return [80.0, 120.0, 180.0, 300.0, 500.0]
    x = train_df.copy()
    x["alt"] = pd.to_numeric(x[alt_col], errors="coerce")
    x["obs_mask"] = pd.to_numeric(x.get("obs_mask", 0.0), errors="coerce").fillna(0.0)
    x["dt_prev"] = pd.to_numeric(x.get("dt_prev", 0.0), errors="coerce").fillna(0.0)
    x["dt_next"] = pd.to_numeric(x.get("dt_next", 0.0), errors="coerce").fillna(0.0)
    x["gap_len"] = x["dt_prev"] + x["dt_next"]
    a_prev = pd.to_numeric(x.get("anchor_alt_prev", np.nan), errors="coerce")
    a_next = pd.to_numeric(x.get("anchor_alt_next", np.nan), errors="coerce")
    ratio = x["dt_prev"] / (x["gap_len"] + 1e-6)
    alt_base = a_prev + ratio * (a_next - a_prev)
    # Use observed value itself on anchors.
    alt_base = np.where(x["obs_mask"].to_numpy() > 0.5, x["alt"].to_numpy(), alt_base.to_numpy())
    resid = np.abs(x["alt"].to_numpy() - alt_base)
    gap = x["gap_len"].to_numpy()
    buckets = [
        gap <= 15.0,
        (gap > 15.0) & (gap <= 30.0),
        (gap > 30.0) & (gap <= 60.0),
        (gap > 60.0) & (gap <= 180.0),
        gap > 180.0,
    ]
    out = []
    for m in buckets:
        vals = resid[m]
        vals = vals[np.isfinite(vals)]
        if vals.size < 50:
            out.append(200.0)
        else:
            out.append(float(max(np.quantile(vals, 0.95), 10.0)))
    return out


def _apply_v2_segment_augmentation(train_df: pd.DataFrame, cfg: dict, run_dir: Path) -> pd.DataFrame:
    """Optional V2-only sample augmentation.

    This is intentionally lightweight: duplicate high-risk short/boundary/disturbed
    samples at sample_id granularity. It does not alter backbone or global losses.
    """
    model_variant = str(cfg.get("model", {}).get("model_variant", "default")).lower()
    aug_cfg = cfg.get("training", {}).get("segment_augmentation", {})
    enabled = bool(aug_cfg.get("enabled", False)) and (
        model_variant in {"bilstm_alt_dms_refiner_v2", "bilstm_alt_dms_refiner_v3"}
    )
    if (not enabled) or train_df.empty:
        return train_df

    short_min = float(aug_cfg.get("short_max_minutes", 60.0))
    disturbed_std = float(aug_cfg.get("disturbed_alt_std_threshold", 120.0))
    boundary_ratio_thres = float(aug_cfg.get("boundary_incomplete_ratio_threshold", 0.2))
    mult_short = max(1, int(aug_cfg.get("multiplier_short", 2)))
    mult_disturbed = max(1, int(aug_cfg.get("multiplier_disturbed", 2)))
    mult_boundary = max(1, int(aug_cfg.get("multiplier_boundary", 2)))

    rows: list[pd.DataFrame] = [train_df]
    audit_rows = []
    sid_col = str(cfg["data"]["sample_id_col"])
    for sid, g in train_df.groupby(sid_col, sort=False):
        x = g.copy()
        obs = pd.to_numeric(x.get("obs_mask", 0.0), errors="coerce").fillna(0.0)
        gap_mask = obs < 0.5
        if not bool(gap_mask.any()):
            continue
        gx = x[gap_mask].copy()
        gap_len = pd.to_numeric(gx.get("gap_len", np.nan), errors="coerce").dropna()
        gap_m = float(gap_len.median()) if len(gap_len) else float("inf")
        alt = pd.to_numeric(gx.get(cfg["data"]["target_cols"][2], np.nan), errors="coerce").dropna()
        alt_std = float(alt.std()) if len(alt) >= 3 else 0.0
        a_prev = pd.to_numeric(gx.get("anchor_alt_prev", np.nan), errors="coerce")
        a_next = pd.to_numeric(gx.get("anchor_alt_next", np.nan), errors="coerce")
        incomplete_ratio = float(((a_prev.isna()) | (a_next.isna())).mean()) if len(gx) else 0.0

        is_short = bool(np.isfinite(gap_m) and (gap_m <= short_min))
        is_disturbed = bool(np.isfinite(alt_std) and (alt_std >= disturbed_std))
        is_boundary = bool(incomplete_ratio >= boundary_ratio_thres)

        if not (is_short or is_disturbed or is_boundary):
            continue
        rep = 1
        if is_short:
            rep = max(rep, mult_short)
        if is_disturbed:
            rep = max(rep, mult_disturbed)
        if is_boundary:
            rep = max(rep, mult_boundary)
        # rep includes original copy; add rep-1 duplicates.
        for i in range(rep - 1):
            c = x.copy()
            c[sid_col] = c[sid_col].astype(str) + f"__v2aug{i+1}"
            rows.append(c)
        audit_rows.append(
            {
                "sample_id": str(sid),
                "gap_minutes_median": gap_m,
                "alt_std_gap": alt_std,
                "boundary_incomplete_ratio": incomplete_ratio,
                "is_short": int(is_short),
                "is_disturbed": int(is_disturbed),
                "is_boundary": int(is_boundary),
                "replication_factor": int(rep),
            }
        )

    out = pd.concat(rows, ignore_index=True) if len(rows) > 1 else train_df
    pd.DataFrame(audit_rows).to_csv(run_dir / f"{model_variant}_segment_augmentation_audit.csv", index=False)
    print(
        f"[v2_segment_augmentation] enabled=1 rows_before={len(train_df)} rows_after={len(out)} "
        f"samples_before={train_df[sid_col].astype(str).nunique()} samples_after={out[sid_col].astype(str).nunique()}"
    )
    return out


def _build_failure_mode_sampler(ds: TrajectoryDataset):
    if len(ds) == 0:
        return None
    w = []
    for s in ds.samples:
        sw = s.get("sample_weight", None)
        if sw is None:
            w.append(1.0)
        else:
            v = float(sw.item()) if hasattr(sw, "item") else float(sw)
            w.append(max(1e-6, v))
    wt = torch.tensor(w, dtype=torch.double)
    return WeightedRandomSampler(weights=wt, num_samples=len(ds), replacement=True)


def _dump_reweight_coverage(ds: TrajectoryDataset, out_csv: Path, epoch: int | None = None) -> None:
    if len(ds) == 0:
        pd.DataFrame(
            [{"epoch": epoch if epoch is not None else 0, "num_samples": 0, "hard_count": 0, "hard_ratio": 0.0, "mean_sample_weight": 0.0, "geom_threshold": float(getattr(ds, "reweight_geom_threshold", 0.0))}]
        ).to_csv(out_csv, index=False)
        return
    hard = []
    sw = []
    for s in ds.samples:
        h = s.get("is_medium_two_anchor_high_last_two_step_geom", 0.0)
        hard.append(float(h.item()) if hasattr(h, "item") else float(h))
        ww = s.get("sample_weight", 1.0)
        sw.append(float(ww.item()) if hasattr(ww, "item") else float(ww))
    row = {
        "epoch": int(epoch) if epoch is not None else 0,
        "num_samples": int(len(ds)),
        "hard_count": int(np.sum(np.array(hard) > 0.5)),
        "hard_ratio": float(np.mean(np.array(hard) > 0.5)),
        "mean_sample_weight": float(np.mean(sw)),
        "geom_threshold": float(getattr(ds, "reweight_geom_threshold", 0.0)),
    }
    mode = "a" if out_csv.exists() else "w"
    header = not out_csv.exists()
    pd.DataFrame([row]).to_csv(out_csv, index=False, mode=mode, header=header)


def _materialize_mixed_epoch_frame(
    stage_train_frames: dict[str, pd.DataFrame],
    weights: dict[str, float],
    n_samples_total: int,
    seed: int,
    epoch: int,
    stage_sampling_profiles: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rng = np.random.default_rng(int(seed) + int(epoch) * 1009)
    stage_order = list(stage_train_frames.keys())
    if not stage_order:
        raise RuntimeError("Curriculum has no stage_train_frames.")
    w = np.array([float(weights.get(s, 0.0)) for s in stage_order], dtype=np.float64)
    if w.sum() <= 0:
        w = np.zeros((len(stage_order),), dtype=np.float64)
        w[0] = 1.0
    w = w / w.sum()
    counts = np.floor(w * int(n_samples_total)).astype(int)
    while counts.sum() < int(n_samples_total):
        counts[int(np.argmax(w - counts / max(1, n_samples_total)))] += 1

    pieces = []
    picked = {}
    for idx, stage in enumerate(stage_order):
        df = stage_train_frames.get(stage)
        if df is None or df.empty:
            picked[stage] = 0
            continue
        sample_ids = df["sample_id"].astype(str).drop_duplicates().to_numpy()
        probs = None
        if stage_sampling_profiles:
            prof = stage_sampling_profiles.get(stage)
            if prof is not None and (not prof.empty):
                prof_map = {
                    str(k): float(v)
                    for k, v in zip(
                        prof["sample_id"].astype(str).tolist(),
                        pd.to_numeric(prof["sample_weight"], errors="coerce").fillna(1.0).tolist(),
                    )
                }
                probs = np.asarray([prof_map.get(str(sid), 1.0) for sid in sample_ids], dtype=np.float64)
        chosen = _sample_ids(sample_ids, int(counts[idx]), rng, probs=probs)
        picked[stage] = int(len(chosen))
        for j, sid in enumerate(chosen.tolist()):
            part = df[df["sample_id"].astype(str).eq(str(sid))].copy()
            part["sample_id"] = part["sample_id"].astype(str) + f"__{stage}_ep{epoch}_{j}"
            part["sim_stage"] = stage
            pieces.append(part)

    if not pieces:
        raise RuntimeError("Curriculum produced empty epoch frame.")
    frame = pd.concat(pieces, ignore_index=True)

    mr = (
        frame.groupby("sample_id")["obs_mask"]
        .apply(lambda x: float((pd.to_numeric(x, errors="coerce").fillna(0.0) < 0.5).mean()))
        .mean()
    )
    gl = []
    for _, sg in frame.groupby("sample_id"):
        gl.extend(_gap_lengths(sg["obs_mask"]))
    gl_s = pd.Series(gl, dtype=np.float64) if gl else pd.Series([0.0], dtype=np.float64)
    audit = {
        "curr_missing_ratio_mean": float(mr),
        "curr_gap_len_mean": float(gl_s.mean()),
        "curr_gap_len_q90": float(gl_s.quantile(0.9)),
        "curr_num_samples": float(frame["sample_id"].nunique()),
        "curr_stage_weights": json.dumps(
            {stage: float(w[idx]) for idx, stage in enumerate(stage_order)},
            ensure_ascii=True,
        ),
    }
    for idx, stage in enumerate(stage_order):
        audit[f"curr_{stage}_weight"] = float(w[idx])
        audit[f"curr_{stage}_samples"] = float(picked.get(stage, 0))
    if stage_sampling_profiles:
        for stage in stage_order:
            prof = stage_sampling_profiles.get(stage)
            if prof is None or prof.empty:
                audit[f"{stage}_alloc_eligible_ratio"] = 0.0
                audit[f"{stage}_weight_mean"] = 0.0
                continue
            audit[f"{stage}_alloc_eligible_ratio"] = float(pd.to_numeric(prof["alloc_eligible"], errors="coerce").fillna(0.0).mean())
            audit[f"{stage}_weight_mean"] = float(pd.to_numeric(prof["sample_weight"], errors="coerce").fillna(1.0).mean())
    return frame, audit




def _sample_has_anchor(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    obs = pd.to_numeric(frame["obs_mask"], errors="coerce").fillna(0.0)
    by = frame.assign(_obs_anchor=(obs > 0.5)).groupby("sample_id")["_obs_anchor"].any()
    return by.astype(bool)


def _summarize_altrel(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {"count": 0, "mean": float("nan"), "std": float("nan"), "q95": float("nan"), "q99": float("nan")}
    x = pd.to_numeric(frame.get("alt_rel_prev_anchor", np.nan), errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return {"count": 0, "mean": float("nan"), "std": float("nan"), "q95": float("nan"), "q99": float("nan")}
    return {
        "count": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "q95": float(np.quantile(x, 0.95)),
        "q99": float(np.quantile(x, 0.99)),
    }


def _summarize_gap_len(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {"count": 0, "mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "q99": float("nan")}
    g = pd.to_numeric(frame.get("gap_len", np.nan), errors="coerce").dropna().to_numpy(dtype=float)
    if len(g) == 0:
        return {"count": 0, "mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "q99": float("nan")}
    return {
        "count": int(len(g)),
        "mean": float(np.mean(g)),
        "p50": float(np.quantile(g, 0.5)),
        "p90": float(np.quantile(g, 0.9)),
        "q99": float(np.quantile(g, 0.99)),
    }


def _anchor_gate_splits(splits: dict[str, pd.DataFrame], run_dir: Path, strict: bool = True) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict]:
    kept: dict[str, pd.DataFrame] = {}
    dropped: dict[str, pd.DataFrame] = {}
    rows = []
    audit = {"strict": bool(strict), "splits": {}}

    for sp, frame in splits.items():
        if frame.empty:
            kept[sp] = frame.copy()
            dropped[sp] = frame.copy()
            audit["splits"][sp] = {
                "samples_total": 0,
                "samples_has_anchor_true": 0,
                "samples_has_anchor_false": 0,
                "ratio_has_anchor_false": 0.0,
                "rows_total": 0,
                "rows_has_anchor_false": 0,
                "rows_has_anchor_true": 0,
            }
            continue

        has_anchor = _sample_has_anchor(frame)
        keep_ids = set(has_anchor[has_anchor].index.astype(str).tolist())
        drop_ids = set(has_anchor[~has_anchor].index.astype(str).tolist())

        f = frame.copy()
        sid = f["sample_id"].astype(str)
        keep_df = f[sid.isin(keep_ids)].copy()
        drop_df = f[sid.isin(drop_ids)].copy()

        kept[sp] = keep_df
        dropped[sp] = drop_df

        st = {
            "samples_total": int(len(has_anchor)),
            "samples_has_anchor_true": int(has_anchor.sum()),
            "samples_has_anchor_false": int((~has_anchor).sum()),
            "ratio_has_anchor_false": float((~has_anchor).mean()),
            "rows_total": int(len(f)),
            "rows_has_anchor_false": int(len(drop_df)),
            "rows_has_anchor_true": int(len(keep_df)),
        }
        st["no_anchor_altrel"] = _summarize_altrel(drop_df)
        st["no_anchor_gap_len"] = _summarize_gap_len(drop_df)
        audit["splits"][sp] = st

        rows.append({"split": sp, **st})

        if strict and st["samples_has_anchor_false"] > 0:
            print(
                f"[anchor_gate][{sp}] filtered_no_anchor_samples={st['samples_has_anchor_false']} "
                f"({st['ratio_has_anchor_false']:.3%})"
            )

    pd.DataFrame(rows).to_csv(run_dir / "anchor_gate_summary.csv", index=False)
    (run_dir / "anchor_gate_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    # fail-fast: only enforce when strict=true and the split originally exists.
    if strict:
        for sp in ["train", "val", "test"]:
            orig_df = splits.get(sp, pd.DataFrame())
            if orig_df is None or orig_df.empty:
                continue
            kdf = kept.get(sp, pd.DataFrame())
            rem = _sample_has_anchor(kdf)
            if len(rem) == 0:
                raise RuntimeError(f"[anchor_gate] {sp} is empty after filtering has_anchor=false samples.")
            bad = rem[~rem]
            if len(bad) > 0:
                sid0 = str(bad.index.astype(str)[0])
                ex = kdf[kdf["sample_id"].astype(str).eq(sid0)].head(1)
                fid0 = str(ex["flight_id"].iloc[0]) if (not ex.empty and "flight_id" in ex.columns) else "unknown"
                total = int(len(rem))
                count_bad = int(len(bad))
                raise RuntimeError(
                    "FATAL: alt_rel main task contains has_anchor=false samples "
                    f"in split={sp}. count={count_bad} / total={total}. "
                    f"example_sample_id={sid0}, example_flight_id={fid0}. "
                    "This violates current task definition."
                )

    return kept, dropped, audit

def main() -> int:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    curriculum_cfg = cfg["training"].get("curriculum", {})
    use_curriculum = bool(curriculum_cfg.get("enabled", False))
    run_dir = Path(cfg["outputs"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    if not use_curriculum:
        df = pd.read_parquet(cfg["data"]["samples_path"])
        splits = split_by_flight_id(
            df=df,
            flight_id_col=cfg["data"]["flight_id_col"],
            train_ratio=float(cfg["data"]["split"]["train_ratio"]),
            val_ratio=float(cfg["data"]["split"]["val_ratio"]),
            seed=int(cfg.get("seed", 42)),
        )
        stage_train_frames = None
        stage_sampling_profiles = None
        epoch_schedule = []
        curr_total_samples = 0
    else:
        stage_paths = curriculum_cfg.get("stage_paths", {})
        required = ["stage1", "stage2", "stage3"]
        missing = [k for k in required if k not in stage_paths]
        if missing:
            raise RuntimeError(f"curriculum.stage_paths missing required keys: {missing}")
        stage_frames = {k: pd.read_parquet(v) for k, v in stage_paths.items()}
        split_reference_stage = str(curriculum_cfg.get("split_reference_stage", "stage3"))
        if split_reference_stage not in stage_frames:
            raise RuntimeError(
                f"curriculum.split_reference_stage={split_reference_stage!r} not in stage_paths keys"
            )
        ref_df = stage_frames[split_reference_stage]
        ref_splits = split_by_flight_id(
            df=ref_df,
            flight_id_col=cfg["data"]["flight_id_col"],
            train_ratio=float(cfg["data"]["split"]["train_ratio"]),
            val_ratio=float(cfg["data"]["split"]["val_ratio"]),
            seed=int(cfg.get("seed", 42)),
        )
        train_ids = set(ref_splits["train"][cfg["data"]["flight_id_col"]].astype(str).unique().tolist())
        val_ids = set(ref_splits["val"][cfg["data"]["flight_id_col"]].astype(str).unique().tolist())
        test_ids = set(ref_splits["test"][cfg["data"]["flight_id_col"]].astype(str).unique().tolist())
        print(
            "[curriculum_split] "
            f"reference_stage={split_reference_stage} "
            f"train_flights={len(train_ids)} val_flights={len(val_ids)} test_flights={len(test_ids)}"
        )
        split_audit_df = _build_stage_split_audit(
            stage_frames=stage_frames,
            flight_id_col=cfg["data"]["flight_id_col"],
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
        )
        split_audit_path = run_dir / "curriculum_stage_split_audit.csv"
        split_audit_df.to_csv(split_audit_path, index=False)
        for row in split_audit_df.itertuples(index=False):
            print(
                "[curriculum_split_stage] "
                f"stage={row.stage} split={row.split} "
                f"flights={row.flights} samples={row.samples} rows={row.rows}"
            )
        print(f"[curriculum_split_audit] path={split_audit_path}")

        stage_train_frames = {}
        for stage, sdf in stage_frames.items():
            fid = sdf[cfg["data"]["flight_id_col"]].astype(str)
            stage_train_frames[stage] = sdf[fid.isin(train_ids)].copy()
        stage_sampling_profiles = {}

        val_stage = str(curriculum_cfg.get("val_stage", "stage3"))
        if val_stage not in stage_frames:
            raise RuntimeError(f"curriculum.val_stage={val_stage} not in stage_paths keys")
        vdf = stage_frames[val_stage]
        vid = vdf[cfg["data"]["flight_id_col"]].astype(str)
        splits = {
            "train": pd.concat(stage_train_frames.values(), ignore_index=True),
            "val": vdf[vid.isin(val_ids)].copy(),
            "test": pd.DataFrame(),
        }
        epoch_schedule = curriculum_cfg.get("schedule", [])
        curr_total_samples = int(curriculum_cfg.get("train_samples_per_epoch", 0))
        if curr_total_samples <= 0:
            curr_total_samples = int(stage_train_frames["stage1"]["sample_id"].nunique())

    # Altitude governance: add anchor-derived altitude features before standardization.
    for k in list(splits.keys()):
        splits[k] = add_anchor_alt_features(splits[k])
        splits[k] = add_vertical_v2_features(splits[k])
    if use_curriculum and stage_train_frames is not None:
        for k in list(stage_train_frames.keys()):
            stage_train_frames[k] = add_anchor_alt_features(stage_train_frames[k])
            stage_train_frames[k] = add_vertical_v2_features(stage_train_frames[k])

    # Anchor gate for alt-rel main task: has_anchor=false samples are excluded from main train/val/test.
    # They are audited separately and saved as appendix stats.
    strict_anchor_gate = bool(cfg["training"].get("anchor_gate", {}).get("strict", True))
    splits, dropped_splits, _anchor_audit = _anchor_gate_splits(splits, run_dir=run_dir, strict=strict_anchor_gate)
    if use_curriculum and stage_train_frames is not None:
        keep_ids_train = set(splits["train"]["sample_id"].astype(str).unique().tolist())
        for k in list(stage_train_frames.keys()):
            stage_train_frames[k] = _filter_frame_by_sample_ids(stage_train_frames[k], keep_ids_train)

    # Label governance on train split only (minimal, no target-definition change).
    lg_cfg = cfg["training"].get("alt_label_governance", {})
    splits["train"], lg_report = apply_alt_label_governance(splits["train"], lg_cfg, out_dir=run_dir)
    if use_curriculum and stage_train_frames is not None:
        keep_ids_train = set(splits["train"]["sample_id"].astype(str).unique().tolist())
        for k in list(stage_train_frames.keys()):
            stage_train_frames[k] = _filter_frame_by_sample_ids(stage_train_frames[k], keep_ids_train)
    print(
        f"[alt_label_governance] enabled={int(bool(lg_report.get('enabled', False)))} "
        f"kept_samples_ratio={lg_report.get('kept_samples_ratio', 1.0):.3f} "
        f"samples_before={lg_report.get('samples_before', 0)} "
        f"samples_after={lg_report.get('samples_after', 0)}"
    )
    if use_curriculum and stage_train_frames is not None:
        savca_sampling_cfg = curriculum_cfg.get("savca_sampling", {})
        savca_sampling_enabled = bool(savca_sampling_cfg.get("enabled", False))
        stage_profile_rows = []
        stage_summary_rows = []
        for stage, sdf in stage_train_frames.items():
            if savca_sampling_enabled:
                prof = _build_curriculum_savca_sampling_profile(
                    frame=sdf,
                    loss_cfg=cfg.get("loss", {}),
                    sampling_cfg=savca_sampling_cfg,
                )
            else:
                prof = pd.DataFrame(
                    {
                        "sample_id": sdf["sample_id"].astype(str).drop_duplicates().tolist(),
                        "max_anchor_delta_abs_m": 0.0,
                        "max_gap_len_min": 0.0,
                        "state_eligible": 0,
                        "alloc_eligible": 0,
                        "sample_weight": 1.0,
                        "delta_bucket": "uniform",
                    }
                )
            prof["stage"] = stage
            stage_sampling_profiles[stage] = prof
            stage_profile_rows.append(prof.copy())
            stage_summary_rows.append(_summarize_curriculum_sampling_profile(prof, stage_name=stage))
        if stage_profile_rows:
            pd.concat(stage_profile_rows, ignore_index=True).to_csv(
                run_dir / "curriculum_stage_sampling_profiles.csv",
                index=False,
            )
        if stage_summary_rows:
            pd.concat(stage_summary_rows, ignore_index=True).to_csv(
                run_dir / "curriculum_stage_sampling_summary.csv",
                index=False,
            )
    # Optional V2-only segment augmentation (default off).
    splits["train"] = _apply_v2_segment_augmentation(splits["train"], cfg=cfg, run_dir=run_dir)

    # Split drift audit for alt/alt_rel after governance.
    dist_rows = []
    for sp in ["train", "val", "test"]:
        if sp in splits:
            dist_rows.extend(summarize_alt_distribution(splits[sp], split_name=sp))
    dist_df = pd.DataFrame(dist_rows)
    if not dist_df.empty:
        dist_df.to_csv(run_dir / "alt_label_distribution_by_split.csv", index=False)
        drift_df = compute_split_drift(dist_df)
        drift_df.to_csv(run_dir / "alt_label_split_drift.csv", index=False)

    appendix_rows = []
    for sp, ddf in dropped_splits.items():
        if ddf.empty:
            appendix_rows.append({"split": sp, "count_samples": 0, "count_rows": 0})
            continue
        x = pd.to_numeric(ddf.get("alt_rel_prev_anchor", np.nan), errors="coerce").dropna().to_numpy(dtype=float)
        g = pd.to_numeric(ddf.get("gap_len", np.nan), errors="coerce").dropna().to_numpy(dtype=float)
        appendix_rows.append(
            {
                "split": sp,
                "count_samples": int(ddf["sample_id"].astype(str).nunique()),
                "count_rows": int(len(ddf)),
                "altrel_mean": float(np.mean(x)) if len(x) else float("nan"),
                "altrel_std": float(np.std(x)) if len(x) else float("nan"),
                "altrel_q95": float(np.quantile(x, 0.95)) if len(x) else float("nan"),
                "altrel_q99": float(np.quantile(x, 0.99)) if len(x) else float("nan"),
                "gap_len_mean": float(np.mean(g)) if len(g) else float("nan"),
                "gap_len_p50": float(np.quantile(g, 0.5)) if len(g) else float("nan"),
                "gap_len_p90": float(np.quantile(g, 0.9)) if len(g) else float("nan"),
            }
        )
    pd.DataFrame(appendix_rows).to_csv(run_dir / "no_anchor_appendix_summary.csv", index=False)

    if str(cfg["model"].get("model_variant", "default")).lower() == "bilstm_alt_base_residual_v1":
        alt_col = str(cfg["data"]["target_cols"][2])
        bounds = _compute_alt_base_residual_bounds(splits["train"], alt_col=alt_col)
        cfg.setdefault("model", {})["alt_base_residual_bounds"] = [float(x) for x in bounds]
        with open(run_dir / "alt_base_residual_bounds.json", "w", encoding="utf-8") as f:
            json.dump({"alt_base_residual_bounds": [float(x) for x in bounds]}, f, ensure_ascii=False, indent=2)
        print(f"[alt_base_residual] train_bounds={bounds}")

    candidate_cols = list(
        dict.fromkeys(
            ["dt_prev", "dt_next"]
            + cfg["data"]["exo_cols"]
            + cfg["data"].get("vertical_exo_cols", [])
            + cfg["data"]["quality_cols"]
        )
    )
    continuous_cols = select_continuous_feature_cols(
        splits["train"],
        candidate_cols=candidate_cols,
        exclude_cols={cfg["data"]["obs_mask_col"]},
    )
    scaler_stats = fit_standardizer(splits["train"], feature_cols=continuous_cols)
    train_before = splits["train"][list(scaler_stats.keys())].copy()
    splits["train"] = apply_standardizer(splits["train"], scaler_stats)
    splits["val"] = apply_standardizer(splits["val"], scaler_stats)
    if use_curriculum and stage_train_frames is not None:
        for k in list(stage_train_frames.keys()):
            stage_train_frames[k] = apply_standardizer(stage_train_frames[k], scaler_stats)
    report = build_standardization_report(
        before_train=train_before,
        after_train=splits["train"][list(scaler_stats.keys())],
        stats=scaler_stats,
    )

    scaler_path = run_dir / "feature_standardizer.json"
    report_path = run_dir / "feature_standardization_stats.csv"
    save_standardizer(scaler_stats, scaler_path)
    report.to_csv(report_path, index=False)
    small_std_count = sum(1 for v in scaler_stats.values() if bool(v.get("small_std", False)))
    missing_before_total = int(report["missing_before"].sum()) if len(report) else 0
    missing_after_total = int(report["missing_after"].sum()) if len(report) else 0
    print(
        f"[norm] features={len(scaler_stats)} "
        f"small_std_replaced={small_std_count} "
        f"missing_before={missing_before_total} "
        f"missing_after={missing_after_total} "
        f"scaler={scaler_path} "
        f"report={report_path}"
    )

    dcfg = DatasetConfig(
        sample_id_col=cfg["data"]["sample_id_col"],
        flight_id_col=cfg["data"]["flight_id_col"],
        time_col=cfg["data"]["time_col"],
        target_cols=cfg["data"]["target_cols"],
        obs_cols=cfg["data"]["obs_cols"],
        obs_mask_col=cfg["data"]["obs_mask_col"],
        exo_cols=cfg["data"]["exo_cols"],
        vertical_exo_cols=cfg["data"].get("vertical_exo_cols", []),
        quality_cols=cfg["data"]["quality_cols"],
        max_time_gap_minutes=float(cfg["data"].get("max_time_gap_minutes", 5.0)),
        split_on_time_gap=bool(cfg["data"].get("split_on_time_gap", True)),
        segment_risk_rules_path=cfg["data"].get("segment_risk_rules_path"),
        use_failure_mode_reweighting=bool(cfg["training"].get("failure_mode_reweighting", {}).get("use_failure_mode_reweighting", False)),
        reweight_target_bucket=str(cfg["training"].get("failure_mode_reweighting", {}).get("reweight_target_bucket", "medium")),
        reweight_target_anchor_pattern=str(cfg["training"].get("failure_mode_reweighting", {}).get("reweight_target_anchor_pattern", "two_anchor")),
        reweight_target_feature=str(cfg["training"].get("failure_mode_reweighting", {}).get("reweight_target_feature", "high_last_two_step_geom")),
        reweight_weight=float(cfg["training"].get("failure_mode_reweighting", {}).get("reweight_weight", 2.5)),
        reweight_last_two_step_geom_threshold=float(cfg["training"].get("failure_mode_reweighting", {}).get("reweight_last_two_step_geom_threshold", 0.0)),
    )

    train_ds = TrajectoryDataset(splits["train"], dcfg)
    val_ds = TrajectoryDataset(splits["val"], dcfg)
    _dump_reweight_coverage(
        train_ds,
        Path(cfg["outputs"]["run_dir"]) / "reweight_coverage_stats.csv",
        epoch=0,
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("train/val split is empty. Please generate more samples.")

    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"]["num_workers"]),
        collate_fn=trajectory_collate_fn,
    )

    target_norm_cfg = cfg["training"].get("target_norm", {})
    use_target_norm = bool(target_norm_cfg.get("enabled", False))
    target_norm_stats = None
    if use_target_norm:
        apply_dims_cfg = target_norm_cfg.get("apply_dims")
        apply_dims = [int(d) for d in apply_dims_cfg] if apply_dims_cfg is not None else None
        center_target = bool(target_norm_cfg.get("center", True))
        center_per_dim_cfg = target_norm_cfg.get("center_per_dim")
        center_per_dim = (
            [bool(x) for x in center_per_dim_cfg]
            if center_per_dim_cfg is not None
            else None
        )
        train_loader_for_stats = DataLoader(
            train_ds,
            batch_size=int(cfg["training"]["batch_size"]),
            shuffle=False,
            num_workers=int(cfg["training"]["num_workers"]),
            collate_fn=trajectory_collate_fn,
        )
        target_norm_stats = compute_target_stats_from_loader(
            loader=train_loader_for_stats,
            coord_mode=str(cfg["model"].get("coord_mode", "latlon")),
            apply_dims=apply_dims,
            center=center_target,
            center_per_dim=center_per_dim,
            u_relative_anchor=bool(cfg["model"].get("u_relative_anchor", False)),
            en_relative_anchor=bool(cfg["model"].get("en_relative_anchor", True)),
            en_incremental=bool(cfg["model"].get("en_incremental", False)),
        )
        target_norm_path = Path(cfg["outputs"]["run_dir"]) / "target_model_scaler.json"
        save_target_stats(target_norm_stats, target_norm_path)
        m = target_norm_stats["mean"]
        s = target_norm_stats["std"]
        print(
            "[target_norm] enabled=1 "
            f"count={target_norm_stats.get('count', 0)} "
            f"center={target_norm_stats.get('center', True)} "
            f"center_per_dim={target_norm_stats.get('center_per_dim', None)} "
            f"apply_dims={target_norm_stats.get('apply_dims', None)} "
            f"mean={m} std={s} "
            f"path={target_norm_path}"
        )

    device = torch.device(cfg["training"].get("device", "cpu"))
    backbone_type = str(cfg["model"].get("backbone_type", "bilstm")).lower()
    minimal_task_adapt_baseline = bool(cfg["model"].get("minimal_task_adapt_baseline", False))
    minimal_baseline_backbones = {
        "unilstm",
        "bilstm",
        "cnnlstm",
        "transformer",
        "mamba_proto",
        "bimamba_proto",
        "unilstm_proto",
        "bilstm_proto",
        "cnnlstm_proto",
        "transformer_proto",
    }
    minimal_baseline_mode = minimal_task_adapt_baseline and backbone_type in minimal_baseline_backbones
    if backbone_type in {"bimamba", "bimamba_recurrent", "mamba_proto", "bimamba_proto", "bimamba_direct", "bimamba_context", "bimamba_context_xyzh", "bimamba_context_xyzh_zlinear", "bimamba_context_xyzh_sharedz", "bimamba_context_xyaux_zlinear", "bimamba_context_xyaux_zlinear_zadapter", "bimamba_context_xyaux_zlinear_zadapter_gapgate", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_coarsetrend", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_vprogaux", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_vprog_resaux"}:
        if device.type != "cuda":
            raise RuntimeError(
                f"backbone_type={backbone_type} requires CUDA, but training.device={device}. "
                "Set training.device=cuda and run on a host where torch.cuda.is_available() is true."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"backbone_type={backbone_type} requires CUDA, but torch.cuda.is_available() is false. "
                "This environment currently exposes no usable GPU to PyTorch."
            )
    model = TrajectoryRecoveryModel(
        exo_dim=len(cfg["data"]["exo_cols"]),
        vertical_exo_dim=len(cfg["data"].get("vertical_exo_cols", [])),
        quality_dim=len(cfg["data"]["quality_cols"]),
        backbone_type=backbone_type,
        hidden_size=int(cfg["model"]["hidden_size"]),
        num_layers=int(cfg["model"].get("num_layers", 1)),
        dropout=float(cfg["model"].get("dropout", 0.0)),
        transformer_num_heads=int(cfg["model"].get("transformer_num_heads", 4)),
        transformer_ff_multiplier=int(cfg["model"].get("transformer_ff_multiplier", 4)),
        fusion_hidden_size=int(cfg["model"].get("fusion_hidden_size", 32)),
        fusion_use_exo_quality=bool(cfg["model"].get("fusion_use_exo_quality", False)),
        fusion_position_prior_enabled=bool(cfg["model"].get("fusion_position_prior_enabled", True)),
        fusion_position_prior_deviation=float(cfg["model"].get("fusion_position_prior_deviation", 0.30)),
        fusion_weight_mode=str(cfg["model"].get("fusion_weight_mode", "scalar")),
        minimal_task_adapt_baseline=minimal_task_adapt_baseline,
        alt_bias_enabled=bool(cfg["model"].get("alt_bias_enabled", False)),
        alt_bias_hidden_size=int(cfg["model"].get("alt_bias_hidden_size", 32)),
        alt_bias_use_exo_quality=bool(cfg["model"].get("alt_bias_use_exo_quality", True)),
        vertical_projector_enabled=bool(cfg["model"].get("vertical_projector_enabled", False)),
        vertical_projector_hidden_size=int(cfg["model"].get("vertical_projector_hidden_size", 32)),
        vertical_projector_use_vertical_exo=bool(cfg["model"].get("vertical_projector_use_vertical_exo", True)),
        vertical_tune_enabled=bool(cfg["model"].get("vertical_tune_enabled", False)),
        vertical_tune_hidden_size=int(cfg["model"].get("vertical_tune_hidden_size", 16)),
        vertical_tune_temperature=float(cfg["model"].get("vertical_tune_temperature", 1.0)),
        vertical_tune_mode=str(cfg["model"].get("vertical_tune_mode", "combined")),
        model_variant=str(cfg["model"].get("model_variant", "default")),
        dms_refiner_hidden_size=int(cfg["model"].get("dms_refiner_hidden_size", 64)),
        dms_refiner_latent_dim=int(cfg["model"].get("dms_refiner_latent_dim", 32)),
        dms_refiner_num_heads=int(cfg["model"].get("dms_refiner_num_heads", 2)),
        dms_refiner_ff_multiplier=int(cfg["model"].get("dms_refiner_ff_multiplier", 2)),
        dms_refiner_dropout=float(cfg["model"].get("dms_refiner_dropout", 0.0)),
        alt_base_builder_type=str(cfg["model"].get("alt_base_builder_type", "auto")),
        alt_base_residual_hidden_size=int(cfg["model"].get("alt_base_residual_hidden_size", 64)),
        alt_base_residual_dropout=float(cfg["model"].get("alt_base_residual_dropout", 0.0)),
        alt_base_residual_bounds=cfg["model"].get("alt_base_residual_bounds"),
        alt_base_residual_bound_enabled=bool(cfg["model"].get("alt_base_residual_bound_enabled", True)),
        alt_gate_enabled=bool(cfg["model"].get("alt_gate_enabled", False)),
        alt_gate_hidden_size=int(cfg["model"].get("alt_gate_hidden_size", 32)),
        alt_gate_mode=str(cfg["model"].get("alt_gate_mode", "learned")),
        alt_gate_fixed_value=float(cfg["model"].get("alt_gate_fixed_value", 1.0)),
        alt_anchor_hard_consistency=bool(cfg["model"].get("alt_anchor_hard_consistency", False)),
        use_left_edge_directional_constraint=bool(cfg["model"].get("use_left_edge_directional_constraint", False)),
        left_edge_direction_mode=str(cfg["model"].get("left_edge_direction_mode", "anchor_based")),
        left_edge_width=int(cfg["model"].get("left_edge_width", 2)),
        left_edge_direction_strength=float(cfg["model"].get("left_edge_direction_strength", 1.0)),
        left_edge_clip_mode=str(cfg["model"].get("left_edge_clip_mode", "hard")),
        boundary_corrector_enabled=bool(cfg["model"].get("boundary_corrector_enabled", False)),
        boundary_corrector_hidden_size=int(cfg["model"].get("boundary_corrector_hidden_size", 16)),
        alt_main_mode=str(cfg["model"].get("alt_main_mode", "absolute")),
        alt_anchor_reference_mode=str(cfg["model"].get("alt_anchor_reference_mode", "local_linear")),
        main_rmax_m=float(cfg["model"].get("main_rmax_m", float(cfg["model"].get("main_rmax_ft", 500.0)) * 0.3048)),
        main_rmax_min_m=float(cfg["model"].get("main_rmax_min_m", 91.44)),
        main_rmax_slope_m_per_min=float(cfg["model"].get("main_rmax_slope_m_per_min", 4.572)),
        main_rmax_max_m=float(cfg["model"].get("main_rmax_max_m", 365.76)),
        alt_residual_anchor_delta_gate_enabled=bool(cfg["model"].get("alt_residual_anchor_delta_gate_enabled", False)),
        alt_residual_anchor_delta_gate_low_m=float(cfg["model"].get("alt_residual_anchor_delta_gate_low_m", 60.0)),
        alt_residual_anchor_delta_gate_high_m=float(cfg["model"].get("alt_residual_anchor_delta_gate_high_m", 180.0)),
        alt_residual_anchor_delta_gate_min_scale=float(cfg["model"].get("alt_residual_anchor_delta_gate_min_scale", 0.0)),
        alt_residual_edge_taper_enabled=bool(cfg["model"].get("alt_residual_edge_taper_enabled", False)),
        alt_residual_edge_taper_steps=float(cfg["model"].get("alt_residual_edge_taper_steps", 3.0)),
        alt_anchor_graph_min_step_gap_min=float(cfg["model"].get("alt_anchor_graph_min_step_gap_min", 8.0)),
        alt_anchor_graph_step_center_ratio=float(cfg["model"].get("alt_anchor_graph_step_center_ratio", 0.5)),
        savca_hidden_size=int(cfg["model"].get("savca_hidden_size", 32)),
        savca_min_uniform=float(cfg["model"].get("savca_min_uniform", 0.05)),
        savca_state_eps=float(cfg["model"].get("savca_state_eps", 0.05)),
        savca_beta_enabled=bool(cfg["model"].get("savca_beta_enabled", False)),
        savca_beta_hidden_size=int(cfg["model"].get("savca_beta_hidden_size", 32)),
        savca_beta_init_bias=float(cfg["model"].get("savca_beta_init_bias", -2.0)),
        savca_beta_default_max=float(cfg["model"].get("savca_beta_default_max", 1.0)),
        savca_beta_gap_cap_enabled=bool(cfg["model"].get("savca_beta_gap_cap_enabled", False)),
        savca_beta_medium_gap_thr=float(cfg["model"].get("savca_beta_medium_gap_thr", 15.0)),
        savca_beta_long_gap_thr=float(cfg["model"].get("savca_beta_long_gap_thr", 45.0)),
        savca_beta_cap_short=float(cfg["model"].get("savca_beta_cap_short", 0.20)),
        savca_beta_cap_medium=float(cfg["model"].get("savca_beta_cap_medium", 0.12)),
        savca_beta_cap_long=float(cfg["model"].get("savca_beta_cap_long", 0.05)),
        savca_beta_conf_gate_enabled=bool(cfg["model"].get("savca_beta_conf_gate_enabled", False)),
        savca_beta_state_conf_threshold=float(cfg["model"].get("savca_beta_state_conf_threshold", 0.10)),
        savca_beta_shape_conf_threshold=float(cfg["model"].get("savca_beta_shape_conf_threshold", 0.18)),
        savca_beta_gate_scale_state=float(cfg["model"].get("savca_beta_gate_scale_state", 10.0)),
        savca_beta_gate_scale_shape=float(cfg["model"].get("savca_beta_gate_scale_shape", 10.0)),
        savca_beta_shape_conf_type=str(cfg["model"].get("savca_beta_shape_conf_type", "pmax")),
        savca_beta_floor_enabled=bool(cfg["model"].get("savca_beta_floor_enabled", False)),
        savca_beta_floor_value=float(cfg["model"].get("savca_beta_floor_value", 0.03)),
        savca_change_score_enabled=bool(cfg["model"].get("savca_change_score_enabled", False)),
        savca_change_score_hidden_size=int(cfg["model"].get("savca_change_score_hidden_size", 32)),
        savca_beta_floor_from_change_score=bool(cfg["model"].get("savca_beta_floor_from_change_score", False)),
        fltp_hidden_size=int(cfg["model"].get("fltp_hidden_size", 32)),
        fltp_c_min=float(cfg["model"].get("fltp_c_min", 0.05)),
        fltp_c_max=float(cfg["model"].get("fltp_c_max", 0.95)),
        fltp_w_min=float(cfg["model"].get("fltp_w_min", 0.05)),
        fltp_w_max=float(cfg["model"].get("fltp_w_max", 0.50)),
        fltp_beta_init_bias=float(cfg["model"].get("fltp_beta_init_bias", -2.0)),
        fltp_gap_cap_enabled=bool(cfg["model"].get("fltp_gap_cap_enabled", True)),
        fltp_medium_gap_thr=float(cfg["model"].get("fltp_medium_gap_thr", 15.0)),
        fltp_long_gap_thr=float(cfg["model"].get("fltp_long_gap_thr", 45.0)),
        fltp_beta_cap_short=float(cfg["model"].get("fltp_beta_cap_short", 0.20)),
        fltp_beta_cap_medium=float(cfg["model"].get("fltp_beta_cap_medium", 0.12)),
        fltp_beta_cap_long=float(cfg["model"].get("fltp_beta_cap_long", 0.05)),
        ssvr_hidden_size=int(cfg["model"].get("ssvr_hidden_size", 64)),
        ssvr_rho_max=float(cfg["model"].get("ssvr_rho_max", 0.30)),
        ssvr_dropout=float(cfg["model"].get("ssvr_dropout", 0.0)),
        alt_transition_hidden_size=int(cfg["model"].get("alt_transition_hidden_size", 32)),
        alt_transition_logit_rmax=float(cfg["model"].get("alt_transition_logit_rmax", 6.0)),
        alt_dms_route_mode=str(cfg["model"].get("alt_dms_route_mode", "none")),
        alt_dms_route_gap_threshold_min=float(cfg["model"].get("alt_dms_route_gap_threshold_min", 9.0)),
        alt_dms_route_low_risk_scale=float(cfg["model"].get("alt_dms_route_low_risk_scale", 0.0)),
        alt_dms_route_high_risk_scale=float(cfg["model"].get("alt_dms_route_high_risk_scale", 1.0)),
        v3_anchor_hard_consistency=bool(cfg["model"].get("v3_anchor_hard_consistency", True)),
        v3_edge_residual_damp_enabled=bool(cfg["model"].get("v3_edge_residual_damp_enabled", True)),
        v3_edge_residual_damp_strength=float(cfg["model"].get("v3_edge_residual_damp_strength", 0.7)),
        v3_edge_residual_damp_steps=int(cfg["model"].get("v3_edge_residual_damp_steps", 2)),
        alt_target_mode=str(
            cfg["model"].get(
                "alt_target_mode",
                "relative_to_left_anchor" if bool(cfg["model"].get("u_relative_anchor", False)) else "absolute",
            )
        ),
        proto_use_anchor_features=bool(cfg["model"].get("proto_use_anchor_features", True)),
        proto_include_exo_quality=bool(cfg["model"].get("proto_include_exo_quality", False)),
        bimamba_include_exo_quality=bool(cfg["model"].get("bimamba_include_exo_quality", False)),
        use_z_adapter=bool(cfg["model"].get("use_z_adapter", False)),
        z_adapter_ratio=float(cfg["model"].get("z_adapter_ratio", 0.25)),
        z_adapter_gamma_init=float(cfg["model"].get("z_adapter_gamma_init", 0.0)),
        proto_gap_len_ref_min=float(cfg["model"].get("proto_gap_len_ref_min", 180.0)),
        recurrent_anchor_init=str(cfg["model"].get("recurrent_anchor_init", "none")),
        obs_anchor_feedback_update=bool(cfg["model"].get("obs_anchor_feedback_update", False)),
    ).to(device)
    if minimal_baseline_mode:
        print(
            "[baseline_minimal] enforcing baseline-safe loss settings: "
            "structured altitude/gating/aux losses set to 0",
            flush=True,
        )

    def _loss_value(name: str, default: float = 0.0) -> float:
        if not minimal_baseline_mode:
            return float(cfg["loss"].get(name, default))
        blocked = {
            "fusion_reg_lambda",
            "lambda_alt_residual",
            "lambda_alt_absolute_aux",
            "lambda_alt_edge_delta",
            "lambda_anchor_consistency",
            "lambda_alt_edge_first_diff",
            "lambda_alt_edge_second_diff",
            "lambda_alt_segment_bound",
            "lambda_alt_vertical_rate_penalty",
            "lambda_alt_boundary_anchor",
            "lambda_alt_aux",
            "lambda_savca_alloc",
            "lambda_savca_state",
            "lambda_savca_smooth",
            "lambda_alt_gate_supervision",
            "lambda_alt_gate_risk_shrink",
            "first_step_anchor_lambda",
            "second_step_anchor_lambda",
            "local_spike_lambda_jump",
            "local_spike_lambda_curve",
            "target_jump_lambda",
            "target_curve_lambda",
            "target_value_lambda",
            "target_rightstep2_boundary_pull_lambda",
            "lambda_var_reg",
        }
        return 0.0 if name in blocked else float(cfg["loss"].get(name, default))

    def _loss_bool(name: str, default: bool = False) -> bool:
        if not minimal_baseline_mode:
            return bool(cfg["loss"].get(name, default))
        blocked = {
            "use_first_step_anchor_loss",
            "use_second_step_anchor_loss",
            "use_local_spike_loss",
            "use_targeted_rightstep2_loss",
            "use_target_jump_loss",
            "use_target_curve_loss",
            "use_target_value_rightstep2_loss",
            "use_target_rightstep2_boundary_pull",
        }
        return False if name in blocked else bool(cfg["loss"].get(name, default))
    cfg_variant = str(cfg["model"].get("model_variant", "default")).lower()
    runtime_variant = str(getattr(model, "model_variant", "")).lower()
    if cfg_variant != runtime_variant:
        raise RuntimeError(
            "FATAL: model_variant mismatch between config and runtime model. "
            f"config={cfg_variant}, runtime={runtime_variant}."
        )

    criterion = TrajectoryLoss(
        anchor_weight=float(cfg["loss"]["anchor_weight"]),
        gap_weight=float(cfg["loss"]["gap_weight"]),
        lambda_pos=float(cfg["loss"]["lambda_pos"]),
        lambda_smooth=float(cfg["loss"]["lambda_smooth"]),
        lambda_unc=float(cfg["loss"]["lambda_unc"]),
        dim_weights=cfg["loss"].get("dim_weights"),
        alpha_vertical=float(cfg["loss"].get("alpha_vertical", 1.0)),
        exo_feature_names=list(cfg["data"].get("exo_cols", [])),
        lambda_cruise_phys=float(cfg["loss"].get("lambda_cruise_phys", 0.0)),
        cruise_speed_smooth_weight=float(cfg["loss"].get("cruise_speed_smooth_weight", 1.0)),
        cruise_heading_rate_weight=float(cfg["loss"].get("cruise_heading_rate_weight", 1.0)),
        cruise_vertical_rate_weight=float(cfg["loss"].get("cruise_vertical_rate_weight", 1.0)),
        cruise_planar_accel_weight=float(cfg["loss"].get("cruise_planar_accel_weight", 1.0)),
        cruise_max_abs_vertical_rate=float(cfg["loss"].get("cruise_max_abs_vertical_rate", 300.0)),
        cruise_max_speed_delta=float(cfg["loss"].get("cruise_max_speed_delta", 30.0)),
        cruise_max_heading_rate=float(cfg["loss"].get("cruise_max_heading_rate", 5.0)),
        cruise_quality_weight_gain=float(cfg["loss"].get("cruise_quality_weight_gain", 1.0)),
        lambda_multi_scale=float(cfg["loss"].get("lambda_multi_scale", 0.0)),
        multi_scale_scales=cfg["loss"].get("multi_scale_scales", []),
        multi_scale_include_alt=bool(cfg["loss"].get("multi_scale_include_alt", False)),
        fusion_reg_lambda=_loss_value("fusion_reg_lambda", 0.0),
        fusion_reg_long_gap_weight=float(cfg["loss"].get("fusion_reg_long_gap_weight", 1.0)),
        gap_alt_weight=float(cfg["loss"].get("gap_alt_weight", 1.0)),
        lambda_vertical_smooth=float(cfg["loss"].get("lambda_vertical_smooth", 0.0)),
        lambda_alt_residual=_loss_value("lambda_alt_residual", 0.0),
        lambda_alt_absolute_aux=_loss_value("lambda_alt_absolute_aux", 0.0),
        alt_edge_steps=int(cfg["loss"].get("alt_edge_steps", 0)),
        alt_edge_weight=float(cfg["loss"].get("alt_edge_weight", 1.0)),
        lambda_alt_edge_delta=_loss_value("lambda_alt_edge_delta", 0.0),
        lambda_anchor_consistency=_loss_value("lambda_anchor_consistency", 0.0),
        anchor_boundary_weight=float(cfg["loss"].get("anchor_boundary_weight", 2.0)),
        lambda_alt_edge_first_diff=_loss_value("lambda_alt_edge_first_diff", 0.0),
        lambda_alt_edge_second_diff=_loss_value("lambda_alt_edge_second_diff", 0.0),
        lambda_alt_segment_bound=_loss_value("lambda_alt_segment_bound", 0.0),
        lambda_alt_vertical_rate_penalty=_loss_value("lambda_alt_vertical_rate_penalty", 0.0),
        lambda_alt_boundary_anchor=_loss_value("lambda_alt_boundary_anchor", 0.0),
        alt_vertical_rate_max=float(cfg["loss"].get("alt_vertical_rate_max", 300.0)),
        segment_boundary_short_len=int(cfg["loss"].get("segment_boundary_short_len", 15)),
        segment_disturbed_alt_std_threshold=float(cfg["loss"].get("segment_disturbed_alt_std_threshold", 120.0)),
        alt_residual_cap_stable=float(cfg["loss"].get("alt_residual_cap_stable", 300.0)),
        alt_residual_cap_disturbed=float(cfg["loss"].get("alt_residual_cap_disturbed", 180.0)),
        alt_residual_cap_boundary=float(cfg["loss"].get("alt_residual_cap_boundary", 120.0)),
        alt_residual_cap_short=float(cfg["loss"].get("alt_residual_cap_short", 100.0)),
        lambda_alt_aux=_loss_value("lambda_alt_aux", 0.0),
        lambda_aux=float(cfg["loss"].get("lambda_aux", 0.0)),
        lambda_vprog=float(cfg["loss"].get("lambda_vprog", 0.0)),
        vprog_enable_abs_dz_min=float(cfg["loss"].get("vprog_enable_abs_dz_min", 100.0)),
        lambda_vprog_res=float(cfg["loss"].get("lambda_vprog_res", 0.0)),
        vprog_res_enable_abs_dz_min=float(cfg["loss"].get("vprog_res_enable_abs_dz_min", 300.0)),
        lambda_savca_alloc=_loss_value("lambda_savca_alloc", 0.0),
        lambda_savca_state=_loss_value("lambda_savca_state", 0.0),
        lambda_savca_smooth=_loss_value("lambda_savca_smooth", 0.0),
        lambda_savca_center=_loss_value("lambda_savca_center", 0.0),
        lambda_savca_final_shape=_loss_value("lambda_savca_final_shape", 0.0),
        lambda_savca_nonlinear=_loss_value("lambda_savca_nonlinear", 0.0),
        lambda_savca_change_score=_loss_value("lambda_savca_change_score", 0.0),
        lambda_fltp_shape=_loss_value("lambda_fltp_shape", 0.0),
        lambda_fltp_center=_loss_value("lambda_fltp_center", 0.0),
        lambda_ssvr_state=_loss_value("lambda_ssvr_state", 0.0),
        lambda_ssvr_smooth=_loss_value("lambda_ssvr_smooth", 0.0),
        ssvr_state_plateau_threshold=float(cfg["loss"].get("ssvr_state_plateau_threshold", 0.15)),
        savca_alloc_min_anchor_delta_m=float(cfg["loss"].get("savca_alloc_min_anchor_delta_m", 30.0)),
        savca_state_min_anchor_delta_m=float(cfg["loss"].get("savca_state_min_anchor_delta_m", 30.0)),
        savca_active_min_anchor_delta_m=float(cfg["loss"].get("savca_active_min_anchor_delta_m", 30.0)),
        savca_change_deadband_m=float(cfg["loss"].get("savca_change_deadband_m", 3.0)),
        savca_label_median_window=int(cfg["loss"].get("savca_label_median_window", 5)),
        savca_active_ratio_to_max=float(cfg["loss"].get("savca_active_ratio_to_max", 0.25)),
        savca_active_min_abs_change_m=float(cfg["loss"].get("savca_active_min_abs_change_m", 10.0)),
        savca_active_expand_steps=int(cfg["loss"].get("savca_active_expand_steps", 1)),
        savca_center_min_anchor_delta_m=float(cfg["loss"].get("savca_center_min_anchor_delta_m", 100.0)),
        savca_center_min_active_len=int(cfg["loss"].get("savca_center_min_active_len", 1)),
        savca_center_min_gap_len=int(cfg["loss"].get("savca_center_min_gap_len", 5)),
        savca_beta_floor_min_anchor_delta_m=float(cfg["loss"].get("savca_beta_floor_min_anchor_delta_m", 100.0)),
        savca_beta_floor_min_active_len=int(cfg["loss"].get("savca_beta_floor_min_active_len", 1)),
        savca_beta_floor_min_qmax=float(cfg["loss"].get("savca_beta_floor_min_qmax", 0.20)),
        savca_beta_floor_min_gap_len=int(cfg["loss"].get("savca_beta_floor_min_gap_len", 5)),
        savca_shape_min_anchor_delta_m=float(cfg["loss"].get("savca_shape_min_anchor_delta_m", 100.0)),
        savca_shape_min_active_len=int(cfg["loss"].get("savca_shape_min_active_len", 1)),
        savca_shape_min_qmax=float(cfg["loss"].get("savca_shape_min_qmax", 0.20)),
        savca_shape_min_gap_len=int(cfg["loss"].get("savca_shape_min_gap_len", 5)),
        savca_change_score_min_anchor_delta_m=float(cfg["loss"].get("savca_change_score_min_anchor_delta_m", 100.0)),
        savca_change_score_min_active_len=int(cfg["loss"].get("savca_change_score_min_active_len", 1)),
        savca_change_score_min_qmax=float(cfg["loss"].get("savca_change_score_min_qmax", 0.20)),
        savca_change_score_min_gap_len=int(cfg["loss"].get("savca_change_score_min_gap_len", 5)),
        savca_nonlinear_margin=float(cfg["loss"].get("savca_nonlinear_margin", 0.05)),
        savca_diag_long_gap_len=int(cfg["loss"].get("savca_diag_long_gap_len", 45)),
        fltp_shape_min_anchor_delta_m=float(cfg["loss"].get("fltp_shape_min_anchor_delta_m", 100.0)),
        fltp_shape_min_active_len=int(cfg["loss"].get("fltp_shape_min_active_len", 1)),
        fltp_shape_min_qmax=float(cfg["loss"].get("fltp_shape_min_qmax", 0.20)),
        fltp_shape_min_gap_len=int(cfg["loss"].get("fltp_shape_min_gap_len", 5)),
        lambda_alt_gate_supervision=_loss_value("lambda_alt_gate_supervision", 0.0),
        lambda_alt_gate_risk_shrink=_loss_value("lambda_alt_gate_risk_shrink", 0.0),
        alt_gate_risk_target=float(cfg["loss"].get("alt_gate_risk_target", 0.35)),
        use_first_step_anchor_loss=_loss_bool("use_first_step_anchor_loss", False),
        first_step_anchor_lambda=_loss_value("first_step_anchor_lambda", 0.0),
        use_second_step_anchor_loss=_loss_bool("use_second_step_anchor_loss", False),
        second_step_anchor_lambda=_loss_value("second_step_anchor_lambda", 0.0),
        use_local_spike_loss=_loss_bool("use_local_spike_loss", False),
        local_spike_target_bucket=str(cfg["loss"].get("local_spike_target_bucket", "medium")),
        local_spike_target_pattern=str(cfg["loss"].get("local_spike_target_pattern", "two_anchor")),
        local_spike_use_rightstep2=bool(cfg["loss"].get("local_spike_use_rightstep2", True)),
        local_spike_use_second_diff=bool(cfg["loss"].get("local_spike_use_second_diff", True)),
        local_spike_lambda_jump=_loss_value("local_spike_lambda_jump", 0.0),
        local_spike_lambda_curve=_loss_value("local_spike_lambda_curve", 0.0),
        use_targeted_rightstep2_loss=_loss_bool("use_targeted_rightstep2_loss", False),
        target_bucket=str(cfg["loss"].get("target_bucket", "medium")),
        target_anchor_pattern=str(cfg["loss"].get("target_anchor_pattern", "two_anchor")),
        use_target_jump_loss=_loss_bool("use_target_jump_loss", True),
        use_target_curve_loss=_loss_bool("use_target_curve_loss", True),
        target_jump_lambda=_loss_value("target_jump_lambda", 0.0),
        target_curve_lambda=_loss_value("target_curve_lambda", 0.0),
        use_target_value_rightstep2_loss=_loss_bool("use_target_value_rightstep2_loss", False),
        target_value_lambda=_loss_value("target_value_lambda", 0.0),
        target_interp_lambda=float(cfg["loss"].get("target_interp_lambda", 0.5)),
        use_target_rightstep2_boundary_pull=_loss_bool("use_target_rightstep2_boundary_pull", False),
        target_rightstep2_boundary_pull_lambda=_loss_value("target_rightstep2_boundary_pull_lambda", 0.0),
        lambda_var_reg=_loss_value("lambda_var_reg", 0.0),
        var_reg_min_ratio=float(cfg["loss"].get("var_reg_min_ratio", 0.3)),
        aux_alt_loss_series=str(cfg["loss"].get("aux_alt_loss_series", "pred_pos")),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    resume_cfg = cfg["training"].get("resume", {}) if isinstance(cfg.get("training"), dict) else {}
    initial_history = None
    initial_best_val = None
    start_epoch = 1
    if bool(resume_cfg.get("enabled", False)):
        ckpt_path = Path(str(resume_cfg.get("checkpoint_path", "")))
        if not ckpt_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        strict = bool(resume_cfg.get("strict", True))
        model.load_state_dict(state["model_state_dict"], strict=strict)
        resume_best_path = Path(cfg["outputs"]["run_dir"]) / cfg["outputs"]["checkpoint_name"]
        if not resume_best_path.exists():
            shutil.copy2(ckpt_path, resume_best_path)
        start_epoch = int(resume_cfg.get("start_epoch", 1))
        history_path = Path(str(resume_cfg.get("history_path", "")))
        if history_path.exists():
            initial_history = json.loads(history_path.read_text(encoding="utf-8"))
            monitor_metric = str(cfg["training"].get("checkpoint_monitor_metric", "gap_horizontal_rmse_m"))
            prev_vals = []
            for row in list(initial_history.get("val", [])):
                if monitor_metric in row:
                    prev_vals.append(float(row[monitor_metric]))
                elif f"val_{monitor_metric}" in row:
                    prev_vals.append(float(row[f"val_{monitor_metric}"]))
            if prev_vals:
                initial_best_val = min(prev_vals)
        print(
            f"[resume] checkpoint={ckpt_path} start_epoch={start_epoch} "
            f"history={'yes' if initial_history else 'no'} best_val={initial_best_val}",
            flush=True,
        )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        run_dir=cfg["outputs"]["run_dir"],
        checkpoint_name=cfg["outputs"]["checkpoint_name"],
        grad_clip=float(cfg["training"]["grad_clip"]),
    )

    run_signature = {
        "config_path": str(args.config),
        "run_dir": str(cfg["outputs"]["run_dir"]),
        "seed": int(cfg.get("seed", 42)),
        "device": str(device),
        "data_samples_path": str(cfg["data"].get("samples_path", "")),
        "model_variant_config": cfg_variant,
        "model_variant_runtime": runtime_variant,
        "backbone_type": str(cfg["model"].get("backbone_type", "")),
        "alt_gate_enabled": bool(cfg["model"].get("alt_gate_enabled", False)),
        "alt_gate_hidden_size": int(cfg["model"].get("alt_gate_hidden_size", 32)),
        "alt_gate_mode": str(cfg["model"].get("alt_gate_mode", "learned")),
        "alt_gate_fixed_value": float(cfg["model"].get("alt_gate_fixed_value", 1.0)),
        "use_left_edge_directional_constraint": bool(cfg["model"].get("use_left_edge_directional_constraint", False)),
        "left_edge_direction_mode": str(cfg["model"].get("left_edge_direction_mode", "anchor_based")),
        "left_edge_width": int(cfg["model"].get("left_edge_width", 2)),
        "left_edge_direction_strength": float(cfg["model"].get("left_edge_direction_strength", 1.0)),
        "left_edge_clip_mode": str(cfg["model"].get("left_edge_clip_mode", "hard")),
        "use_segment_teacher": bool(cfg["training"].get("risk_aware", {}).get("use_segment_teacher", True)),
        "use_alt_baseline_residual": bool(cfg["training"].get("risk_aware", {}).get("use_alt_baseline_residual", True)),
        "segment_risk_rules_path": str(cfg["data"].get("segment_risk_rules_path", "")),
        "lambda_alt_gate_supervision": float(cfg["loss"].get("lambda_alt_gate_supervision", 0.0)),
        "lambda_alt_gate_risk_shrink": float(cfg["loss"].get("lambda_alt_gate_risk_shrink", 0.0)),
        "alt_gate_risk_target": float(cfg["loss"].get("alt_gate_risk_target", 0.35)),
        "anchor_gate_enabled": True,
        "alt_label_governance_enabled": bool(cfg["training"].get("alt_label_governance", {}).get("enabled", False)),
    }
    (Path(cfg["outputs"]["run_dir"]) / "train_run_signature.json").write_text(
        json.dumps(run_signature, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    curr_epoch_audit: dict[int, dict[str, float]] = {}

    def _build_train_loader_for_epoch(epoch: int):
        if not use_curriculum:
            sampler = None
            if bool(cfg["training"].get("failure_mode_reweighting", {}).get("use_failure_mode_reweighting", False)):
                sampler = _build_failure_mode_sampler(train_ds)
            return DataLoader(
                train_ds,
                batch_size=int(cfg["training"]["batch_size"]),
                shuffle=(sampler is None),
                sampler=sampler,
                num_workers=int(cfg["training"]["num_workers"]),
                collate_fn=trajectory_collate_fn,
            )
        assert stage_train_frames is not None
        weights = _pick_schedule_weights(epoch, epoch_schedule)
        mixed_df, audit = _materialize_mixed_epoch_frame(
            stage_train_frames=stage_train_frames,
            weights=weights,
            n_samples_total=int(curr_total_samples),
            seed=int(cfg.get("seed", 42)),
            epoch=epoch,
            stage_sampling_profiles=stage_sampling_profiles,
        )
        curr_epoch_audit[epoch] = audit
        ds = TrajectoryDataset(mixed_df, dcfg)
        _dump_reweight_coverage(
            ds,
            Path(cfg["outputs"]["run_dir"]) / "reweight_coverage_stats.csv",
            epoch=epoch,
        )
        sampler = None
        if bool(cfg["training"].get("failure_mode_reweighting", {}).get("use_failure_mode_reweighting", False)):
            sampler = _build_failure_mode_sampler(ds)
        return DataLoader(
            ds,
            batch_size=int(cfg["training"]["batch_size"]),
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=int(cfg["training"]["num_workers"]),
            collate_fn=trajectory_collate_fn,
        )

    def _epoch_ctx(epoch: int) -> dict:
        ctx = {}
        if use_curriculum:
            ctx.update(curr_epoch_audit.get(epoch, {}))
        beta_sched = cfg["training"].get("savca_beta_schedule", {})
        if bool(beta_sched.get("enabled", False)):
            ctx["savca_beta_max"] = _interp_scalar_schedule(
                epoch=int(epoch),
                schedule_epochs=[int(x) for x in beta_sched.get("epochs", [])],
                schedule_values=[float(x) for x in beta_sched.get("values", [])],
            )
        return ctx

    monitor_metric = str(cfg["training"].get("checkpoint_monitor_metric", "gap_horizontal_rmse_m"))
    deprecated_monitor = {"gap_rmse", "overall_rmse", "rmse", "gap_horizontal_rmse"}
    if monitor_metric in deprecated_monitor:
        print(
            f"[monitor] checkpoint_monitor_metric={monitor_metric} is mixed-unit/debug metric; "
            "switching to gap_horizontal_rmse_m for best-checkpoint selection."
        )
        monitor_metric = "gap_horizontal_rmse_m"

    try:
        trainer.fit(
            train_loader=_build_train_loader_for_epoch(1),
            val_loader=val_loader,
            epochs=int(cfg["training"]["epochs"]),
            start_epoch=int(start_epoch),
            teacher_forcing_ratio=float(cfg["training"]["teacher_forcing_ratio"]),
            teacher_forcing_decay=float(cfg["training"]["teacher_forcing_decay"]),
            coord_mode=str(cfg["model"].get("coord_mode", "latlon")),
            u_relative_anchor=bool(cfg["model"].get("u_relative_anchor", False)),
            en_relative_anchor=bool(cfg["model"].get("en_relative_anchor", True)),
            en_incremental=bool(cfg["model"].get("en_incremental", False)),
            long_gap_threshold=int(cfg["training"].get("long_gap_threshold", 20)),
            checkpoint_monitor_metric=monitor_metric,
            target_norm_stats=target_norm_stats,
            alt_target_transform_mode=str(cfg["training"].get("alt_target_robust", {}).get("mode", "none")).lower(),
            alt_target_clip_value=float(cfg["training"].get("alt_target_robust", {}).get("clip_value", 3000.0)),
            early_stopping_patience=(
                int(cfg["training"]["early_stopping"]["patience"])
                if "early_stopping" in cfg["training"] and cfg["training"]["early_stopping"].get("enabled", False)
                else None
            ),
            early_stopping_min_delta=(
                float(cfg["training"]["early_stopping"].get("min_delta", 0.0))
                if "early_stopping" in cfg["training"]
                else 0.0
            ),
            early_stopping_min_epochs=(
                int(cfg["training"]["early_stopping"].get("min_epochs", 0))
                if "early_stopping" in cfg["training"]
                else 0
            ),
            train_loader_factory=_build_train_loader_for_epoch,
            epoch_context_factory=_epoch_ctx,
            save_every_epoch=bool(cfg["training"].get("save_every_epoch", False)),
            save_epoch_interval=int(cfg["training"].get("save_epoch_interval", 1)),
            verbose_epoch_diagnostics=bool(cfg["training"].get("verbose_epoch_diagnostics", False)),
            verbose_diag_first_epoch_only=bool(cfg["training"].get("verbose_diag_first_epoch_only", True)),
            heartbeat_enabled=bool(cfg["training"].get("step_heartbeat", {}).get("enabled", True)),
            heartbeat_interval=int(cfg["training"].get("step_heartbeat", {}).get("interval", 200)),
            use_segment_teacher=bool(cfg["training"].get("risk_aware", {}).get("use_segment_teacher", True)),
            use_alt_baseline_residual=bool(cfg["training"].get("risk_aware", {}).get("use_alt_baseline_residual", True)),
            initial_history=initial_history,
            initial_best_val=initial_best_val,
        )
    except Exception as e:
        tb = traceback.format_exc()
        run_dir = Path(cfg["outputs"]["run_dir"])
        (run_dir / "train_fatal_traceback.log").write_text(tb, encoding="utf-8")
        (run_dir / "train_fatal_traceback.json").write_text(
            json.dumps(
                {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": tb,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        raise

    print(f"[ok] checkpoint={Path(cfg['outputs']['run_dir']) / cfg['outputs']['checkpoint_name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
