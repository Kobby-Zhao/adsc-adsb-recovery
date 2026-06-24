from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.datasets import DatasetConfig, TrajectoryDataset, trajectory_collate_fn
from src.losses import TrajectoryLoss
from src.models import TrajectoryRecoveryModel
from src.preprocessing.standardize import (
    apply_standardizer,
    fit_standardizer,
    load_standardizer,
    save_standardizer,
    select_continuous_feature_cols,
)
from src.training.coords import build_anchor_alt_tracks, build_anchor_pair_tracks, prepare_model_coordinates, restore_to_latlon
from src.training.altitude_governance import add_anchor_alt_features, add_vertical_v2_features
from src.training.alt_target import apply_alt_target_transform, invert_alt_target_transform
from src.training.target_norm import denormalize_coords, load_target_stats, normalize_coords
from src.training.utils import load_config, set_seed, split_by_flight_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate trajectory recovery model.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--plots-dir", default=None)
    parser.add_argument("--plot-count", type=int, default=3)
    parser.add_argument("--long-gap-threshold", type=int, default=20)
    parser.add_argument("--few-anchor-threshold", type=int, default=6)
    return parser


def _haversine_m(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return 6371000.0 * c


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


def _metric_mean(df: pd.DataFrame, col: str) -> float:
    if df.empty:
        return float("nan")
    return float(df[col].mean())


def _masked_mae_rmse_np(err: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if err.size == 0:
        return np.full((3,), np.nan, dtype=np.float64), np.full((3,), np.nan, dtype=np.float64)
    if mask is None:
        mask = np.ones((err.shape[0],), dtype=bool)
    mask = mask.astype(bool)
    if mask.sum() <= 0:
        return np.full((err.shape[1],), np.nan, dtype=np.float64), np.full((err.shape[1],), np.nan, dtype=np.float64)
    sel = err[mask]
    mae = np.mean(np.abs(sel), axis=0)
    rmse = np.sqrt(np.mean(sel**2, axis=0))
    return mae, rmse


def _masked_altrel_stats_np(pred_alt: np.ndarray, true_alt: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    mask = mask.astype(bool)
    if mask.sum() <= 1:
        return float("nan"), float("nan")
    p = pred_alt[mask]
    t = true_alt[mask]
    pstd = float(np.std(p))
    tstd = float(np.std(t))
    if pstd <= 1e-8 or tstd <= 1e-8:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(p, t)[0, 1])
    return pstd, corr


def _gap_len_bucket(v: float) -> str:
    if v < 15:
        return "short"
    if v < 45:
        return "medium"
    return "long"


def _delta_z_bucket(v: float) -> str:
    av = abs(float(v))
    if av <= 30:
        return "[0,30]"
    if av <= 100:
        return "(30,100]"
    if av <= 300:
        return "(100,300]"
    return "(300,+inf]"


def _boundary_alt_from_model_obs(
    obs_for_model: torch.Tensor,
    obs_mask: torch.Tensor,
    seq_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, t_len, _ = obs_for_model.shape
    left = torch.zeros((bsz,), device=obs_for_model.device, dtype=obs_for_model.dtype)
    right = torch.zeros((bsz,), device=obs_for_model.device, dtype=obs_for_model.dtype)
    obs_alt = obs_for_model[..., 2]
    for i in range(bsz):
        valid = seq_mask[i] > 0.5
        obs = (obs_mask[i] > 0.5) & valid
        valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
        obs_idx = torch.nonzero(obs, as_tuple=False).flatten()
        if obs_idx.numel() == 0:
            if valid_idx.numel() == 0:
                left[i] = 0.0
                right[i] = 0.0
            else:
                left[i] = obs_alt[i, valid_idx[0]]
                right[i] = obs_alt[i, valid_idx[-1]]
            continue
        gap = (~obs) & valid
        best_s, best_e, best_len = -1, -1, 0
        t = 0
        while t < t_len:
            if not bool(gap[t]):
                t += 1
                continue
            s = t
            while t < t_len and bool(gap[t]):
                t += 1
            e = t
            if (e - s) > best_len:
                best_s, best_e, best_len = s, e, e - s
        if best_len <= 0:
            left[i] = obs_alt[i, obs_idx[0]]
            right[i] = obs_alt[i, obs_idx[-1]]
            continue
        left_idx = best_s - 1 if (best_s - 1 >= 0 and bool(obs[best_s - 1])) else None
        right_idx = best_e if (best_e < t_len and bool(obs[best_e])) else None
        if left_idx is None:
            cand = obs_idx[obs_idx < best_s]
            if cand.numel() > 0:
                left_idx = int(cand[-1].item())
        if right_idx is None:
            cand = obs_idx[obs_idx >= best_e]
            if cand.numel() > 0:
                right_idx = int(cand[0].item())
        if left_idx is None and right_idx is None:
            left_idx = int(obs_idx[0].item())
            right_idx = int(obs_idx[-1].item())
        elif left_idx is None:
            left_idx = int(right_idx)  # type: ignore[arg-type]
        elif right_idx is None:
            right_idx = int(left_idx)
        left[i] = obs_alt[i, int(left_idx)]
        right[i] = obs_alt[i, int(right_idx)]
    return left, right


def _sample_has_anchor(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    obs = pd.to_numeric(frame.get("obs_mask", 0.0), errors="coerce").fillna(0.0)
    by = frame.assign(_obs_anchor=(obs > 0.5)).groupby("sample_id")["_obs_anchor"].any()
    return by.astype(bool)


def _split_main_and_no_anchor(frame: pd.DataFrame, split_name: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if frame.empty:
        raise RuntimeError(f"FATAL: split={split_name} is empty before anchor gating.")
    has_anchor = _sample_has_anchor(frame)
    total = int(len(has_anchor))
    no_anchor = has_anchor[~has_anchor]
    keep_ids = set(has_anchor[has_anchor].index.astype(str).tolist())
    drop_ids = set(no_anchor.index.astype(str).tolist())
    sid = frame["sample_id"].astype(str)
    main_df = frame[sid.isin(keep_ids)].copy()
    audit_df = frame[sid.isin(drop_ids)].copy()
    if main_df.empty:
        raise RuntimeError(
            f"FATAL: alt_rel main task has no valid samples in split={split_name} after has_anchor gating."
        )
    rem_main = _sample_has_anchor(main_df)
    if bool((~rem_main).any()):
        bad = rem_main[~rem_main]
        sid0 = str(bad.index.astype(str)[0])
        ex = main_df[main_df["sample_id"].astype(str).eq(sid0)].head(1)
        fid0 = str(ex["flight_id"].iloc[0]) if (not ex.empty and "flight_id" in ex.columns) else "unknown"
        raise RuntimeError(
            "FATAL: alt_rel main task contains has_anchor=false samples "
            f"in split={split_name}. count={int(len(bad))} / total={int(len(rem_main))}. "
            f"example_sample_id={sid0}, example_flight_id={fid0}. This violates current task definition."
        )
    info = {
        "split": split_name,
        "samples_total": total,
        "samples_has_anchor_true": int(has_anchor.sum()),
        "samples_has_anchor_false": int((~has_anchor).sum()),
        "ratio_has_anchor_false": float((~has_anchor).mean()) if total > 0 else 0.0,
    }
    if len(no_anchor) > 0:
        sid0 = str(no_anchor.index.astype(str)[0])
        ex = frame[frame["sample_id"].astype(str).eq(sid0)].head(1)
        fid0 = str(ex["flight_id"].iloc[0]) if (not ex.empty and "flight_id" in ex.columns) else "unknown"
        info["example_no_anchor_sample_id"] = sid0
        info["example_no_anchor_flight_id"] = fid0
    return main_df, audit_df, info


def _fatal_if_main_contains_no_anchor(sample_df: pd.DataFrame, split_name: str) -> None:
    if sample_df.empty:
        raise RuntimeError(f"FATAL: main_task sample_df is empty in split={split_name}.")
    bad = sample_df[sample_df.get("has_anchor", False) == False]
    if len(bad) > 0:
        sid0 = str(bad.iloc[0]["sample_id"])
        fid0 = str(bad.iloc[0]["flight_id"])
        total = int(len(sample_df))
        count_bad = int(len(bad))
        raise RuntimeError(
            "FATAL: alt_rel main task contains has_anchor=false samples "
            f"in split={split_name}. count={count_bad} / total={total}. "
            f"example_sample_id={sid0}, example_flight_id={fid0}. "
            "This violates current task definition."
        )


def _plot_subset_bars(out_path: Path, subset_stats: dict) -> None:
    keys = list(subset_stats.keys())
    values = [subset_stats[k]["haversine_m"] for k in keys]
    counts = [subset_stats[k]["count"] for k in keys]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(keys, values, color=["#34495e", "#2c7fb8", "#d95f0e", "#6a3d9a"])
    ax.set_ylabel("Mean Haversine Error (m)")
    ax.set_title("Subset Error Comparison")
    for i, (v, c) in enumerate(zip(values, counts)):
        label = "nan" if np.isnan(v) else f"{v:.1f}"
        ax.text(i, 0 if np.isnan(v) else v, f"{label}\n(n={c})", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_haversine_hist(out_path: Path, sample_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(sample_df["haversine_m"], bins=20, color="#1f78b4", alpha=0.85)
    ax.set_xlabel("Per-sample Mean Haversine Error (m)")
    ax.set_ylabel("Count")
    ax.set_title("Error Distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _track_group_from_sample_id(sample_id: str) -> str:
    sid = str(sample_id)
    sid = sid.split("__seg")[0]
    if sid.endswith("_tail"):
        return sid[:-5]
    m = re.match(r"^(.*)_\d+$", sid)
    if m:
        return m.group(1)
    return sid


def _build_flight_track(samples: list[dict]) -> dict:
    rows: dict[str, dict[str, list[float]]] = {}
    for s in samples:
        times = s["times"]
        pred = s["pred"]
        target = s["target"]
        obs_mask = s["obs_mask"]
        n = min(len(times), len(pred), len(target), len(obs_mask))
        for i in range(n):
            ts = str(times[i])
            rec = rows.setdefault(
                ts,
                {
                    "pred0": [], "pred1": [], "pred2": [],
                    "target0": [], "target1": [], "target2": [],
                    "obs": [],
                },
            )
            rec["pred0"].append(float(pred[i, 0]))
            rec["pred1"].append(float(pred[i, 1]))
            rec["pred2"].append(float(pred[i, 2]))
            rec["target0"].append(float(target[i, 0]))
            rec["target1"].append(float(target[i, 1]))
            rec["target2"].append(float(target[i, 2]))
            rec["obs"].append(float(obs_mask[i]))

    if not rows:
        return {"ok": False, "reason": "empty_rows"}

    keys = sorted(rows.keys())
    pred = []
    target = []
    obs = []
    for ts in keys:
        rec = rows[ts]
        pred.append([np.mean(rec["pred0"]), np.mean(rec["pred1"]), np.mean(rec["pred2"])])
        target.append([np.mean(rec["target0"]), np.mean(rec["target1"]), np.mean(rec["target2"])])
        obs.append(1.0 if np.mean(rec["obs"]) > 0.5 else 0.0)

    pred_np = np.asarray(pred, dtype=np.float64)
    target_np = np.asarray(target, dtype=np.float64)
    obs_np = np.asarray(obs, dtype=np.float64)

    anchor_idx = np.where(obs_np > 0.5)[0]
    if len(anchor_idx) < 2:
        return {
            "ok": False,
            "reason": "no_legal_anchor_interval",
            "anchor_count": int(len(anchor_idx)),
            "track_len": int(len(obs_np)),
        }

    s = int(anchor_idx[0])
    e = int(anchor_idx[-1]) + 1
    if e - s <= 1:
        return {
            "ok": False,
            "reason": "anchor_interval_too_short",
            "anchor_count": int(len(anchor_idx)),
            "track_len": int(len(obs_np)),
        }

    seg_obs = obs_np[s:e]
    return {
        "ok": True,
        "times": keys[s:e],
        "pred": pred_np[s:e],
        "target": target_np[s:e],
        "obs_mask": seg_obs,
        "start_idx": s,
        "end_idx": e,
        "window_len": int(e - s),
        "max_gap_minutes": int(_max_gap_len(seg_obs)),
    }


def _plot_flight_tracks(out_dir: Path, flights: list[dict], plot_count: int) -> None:
    skipped = []
    for item in flights[:plot_count]:
        fid = item["flight_id"]
        split_name = str(item.get("split", "unknown"))
        track_key = item.get("track_key", fid)
        merged = _build_flight_track(item["samples"])
        if not merged.get("ok", False):
            skipped.append(
                {
                    "split": split_name,
                    "flight_id": fid,
                    "track_key": track_key,
                    "reason": merged.get("reason", "unknown"),
                    "anchor_count": merged.get("anchor_count", np.nan),
                    "track_len": merged.get("track_len", np.nan),
                }
            )
            continue

        pred_seg = merged["pred"]
        target_seg = merged["target"]
        obs_seg = merged["obs_mask"]
        t = np.arange(merged["start_idx"], merged["end_idx"])
        local_anchor_idx = np.where(obs_seg > 0.5)[0]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(target_seg[:, 1], target_seg[:, 0], label="target", color="#1b9e77")
        axes[0].plot(pred_seg[:, 1], pred_seg[:, 0], label="pred", color="#d95f02")
        axes[0].scatter(
            target_seg[local_anchor_idx, 1],
            target_seg[local_anchor_idx, 0],
            s=14,
            color="#000000",
            alpha=0.7,
            label="obs/anchor",
        )
        axes[0].scatter(
            target_seg[local_anchor_idx[0], 1],
            target_seg[local_anchor_idx[0], 0],
            s=36,
            color="#1f78b4",
            marker="o",
            label="first_anchor",
        )
        axes[0].scatter(
            target_seg[local_anchor_idx[-1], 1],
            target_seg[local_anchor_idx[-1], 0],
            s=42,
            color="#6a3d9a",
            marker="X",
            label="last_anchor",
        )
        axes[0].set_xlabel("Lon")
        axes[0].set_ylabel("Lat")
        axes[0].set_title(
            f"Flight Overlay ({track_key}) split={split_name} has_anchor=1 "
            f"len={merged['window_len']} max_gap={merged['max_gap_minutes']}"
        )
        axes[0].legend()

        axes[1].plot(t, target_seg[:, 2], label="target_alt", color="#1b9e77")
        axes[1].plot(t, pred_seg[:, 2], label="pred_alt", color="#d95f02")
        axes[1].scatter(
            t[local_anchor_idx],
            target_seg[local_anchor_idx, 2],
            s=14,
            color="#000000",
            alpha=0.7,
            label="obs/anchor",
        )
        axes[1].scatter(
            t[local_anchor_idx[0]],
            target_seg[local_anchor_idx[0], 2],
            s=36,
            color="#1f78b4",
            marker="o",
            label="first_anchor",
        )
        axes[1].scatter(
            t[local_anchor_idx[-1]],
            target_seg[local_anchor_idx[-1], 2],
            s=42,
            color="#6a3d9a",
            marker="X",
            label="last_anchor",
        )
        axes[1].set_xlabel("Minute Index")
        axes[1].set_ylabel("Altitude")
        axes[1].set_title("Altitude Curve")
        axes[1].legend()

        fig.tight_layout()
        safe_name = str(track_key).replace('/', '_')
        fig.savefig(out_dir / f"flight_{safe_name}.png", dpi=150)
        plt.close(fig)

    if skipped:
        pd.DataFrame(skipped).to_csv(out_dir / "plot_skip_reasons.csv", index=False)
        print(f"[plot] skipped_tracks={len(skipped)} reasons_saved={out_dir / 'plot_skip_reasons.csv'}")


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    df = pd.read_parquet(cfg["data"]["samples_path"])
    splits = split_by_flight_id(
        df=df,
        flight_id_col=cfg["data"]["flight_id_col"],
        train_ratio=float(cfg["data"]["split"]["train_ratio"]),
        val_ratio=float(cfg["data"]["split"]["val_ratio"]),
        seed=int(cfg.get("seed", 42)),
    )
    for k in list(splits.keys()):
        splits[k] = add_anchor_alt_features(splits[k])
        splits[k] = add_vertical_v2_features(splits[k])

    # Main-task gating before evaluation: has_anchor=false must not enter main eval path.
    split_main_df, split_no_anchor_df, split_anchor_audit = _split_main_and_no_anchor(splits[args.split], args.split)
    train_main_df, _train_no_anchor_df, _train_anchor_audit = _split_main_and_no_anchor(splits["train"], "train_for_scaler")

    run_dir = Path(cfg["outputs"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = run_dir / "feature_standardizer.json"
    scaler_stats = load_standardizer(scaler_path)
    if scaler_stats is None:
        candidate_cols = list(
            dict.fromkeys(
                ["dt_prev", "dt_next"]
                + cfg["data"]["exo_cols"]
                + cfg["data"].get("vertical_exo_cols", [])
                + cfg["data"]["quality_cols"]
            )
        )
        continuous_cols = select_continuous_feature_cols(
            train_main_df,
            candidate_cols=candidate_cols,
            exclude_cols={cfg["data"]["obs_mask_col"]},
        )
        scaler_stats = fit_standardizer(train_main_df, feature_cols=continuous_cols)
        save_standardizer(scaler_stats, scaler_path)
        print(f"[norm] scaler missing, fitted from anchor-valid train split and saved to {scaler_path}")
    # Keep obs_lat/lon/alt in physical coordinates for ENU reference selection.
    scaler_stats = {k: v for k, v in scaler_stats.items() if k not in set(cfg["data"]["obs_cols"])}
    split_main_df = apply_standardizer(split_main_df, scaler_stats)

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
        segment_risk_rules_path=cfg["data"].get("segment_risk_rules_path"),
    )

    ds = TrajectoryDataset(split_main_df, dcfg)
    if len(ds) == 0:
        raise RuntimeError(f"FATAL: {args.split} main-task split is empty after has_anchor gating.")

    loader = DataLoader(
        ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"]["num_workers"]),
        collate_fn=trajectory_collate_fn,
    )

    device = torch.device(cfg["training"].get("device", "cpu"))
    abr_bounds = cfg["model"].get("alt_base_residual_bounds")
    if abr_bounds is None:
        bpath = Path(cfg["outputs"]["run_dir"]) / "alt_base_residual_bounds.json"
        if bpath.exists():
            with open(bpath, "r", encoding="utf-8") as f:
                abr_bounds = json.load(f).get("alt_base_residual_bounds")
    model = TrajectoryRecoveryModel(
        exo_dim=len(cfg["data"]["exo_cols"]),
        vertical_exo_dim=len(cfg["data"].get("vertical_exo_cols", [])),
        quality_dim=len(cfg["data"]["quality_cols"]),
        backbone_type=str(cfg["model"].get("backbone_type", "bilstm")),
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
        minimal_task_adapt_baseline=bool(cfg["model"].get("minimal_task_adapt_baseline", False)),
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
        alt_target_mode=str(
            cfg["model"].get(
                "alt_target_mode",
                "relative_to_left_anchor" if bool(cfg["model"].get("u_relative_anchor", False)) else "absolute",
            )
        ),
        proto_use_anchor_features=bool(cfg["model"].get("proto_use_anchor_features", True)),
        proto_include_exo_quality=bool(cfg["model"].get("proto_include_exo_quality", False)),
        proto_gap_len_ref_min=float(cfg["model"].get("proto_gap_len_ref_min", 180.0)),
        use_z_adapter=bool(cfg["model"].get("use_z_adapter", False)),
        z_adapter_ratio=float(cfg["model"].get("z_adapter_ratio", 0.25)),
        z_adapter_gamma_init=float(cfg["model"].get("z_adapter_gamma_init", 0.0)),
        recurrent_anchor_init=str(cfg["model"].get("recurrent_anchor_init", "none")),
        obs_anchor_feedback_update=bool(cfg["model"].get("obs_anchor_feedback_update", False)),
        alt_base_residual_hidden_size=int(cfg["model"].get("alt_base_residual_hidden_size", 64)),
        alt_base_residual_dropout=float(cfg["model"].get("alt_base_residual_dropout", 0.0)),
        alt_base_residual_bounds=abr_bounds,
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
    ).to(device)

    ckpt_path = args.checkpoint or str(Path(cfg["outputs"]["run_dir"]) / cfg["outputs"]["checkpoint_name"])
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])

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
        fusion_reg_lambda=float(cfg["loss"].get("fusion_reg_lambda", 0.0)),
        fusion_reg_long_gap_weight=float(cfg["loss"].get("fusion_reg_long_gap_weight", 1.0)),
        gap_alt_weight=float(cfg["loss"].get("gap_alt_weight", 1.0)),
        lambda_vertical_smooth=float(cfg["loss"].get("lambda_vertical_smooth", 0.0)),
        lambda_alt_residual=float(cfg["loss"].get("lambda_alt_residual", 0.0)),
        lambda_alt_absolute_aux=float(cfg["loss"].get("lambda_alt_absolute_aux", 0.0)),
        alt_edge_steps=int(cfg["loss"].get("alt_edge_steps", 0)),
        alt_edge_weight=float(cfg["loss"].get("alt_edge_weight", 1.0)),
        lambda_alt_edge_delta=float(cfg["loss"].get("lambda_alt_edge_delta", 0.0)),
        lambda_anchor_consistency=float(cfg["loss"].get("lambda_anchor_consistency", 0.0)),
        anchor_boundary_weight=float(cfg["loss"].get("anchor_boundary_weight", 2.0)),
        lambda_alt_edge_first_diff=float(cfg["loss"].get("lambda_alt_edge_first_diff", 0.0)),
        lambda_alt_edge_second_diff=float(cfg["loss"].get("lambda_alt_edge_second_diff", 0.0)),
        lambda_alt_segment_bound=float(cfg["loss"].get("lambda_alt_segment_bound", 0.0)),
        lambda_alt_vertical_rate_penalty=float(cfg["loss"].get("lambda_alt_vertical_rate_penalty", 0.0)),
        lambda_alt_boundary_anchor=float(cfg["loss"].get("lambda_alt_boundary_anchor", 0.0)),
        alt_vertical_rate_max=float(cfg["loss"].get("alt_vertical_rate_max", 300.0)),
        segment_boundary_short_len=int(cfg["loss"].get("segment_boundary_short_len", 15)),
        segment_disturbed_alt_std_threshold=float(cfg["loss"].get("segment_disturbed_alt_std_threshold", 120.0)),
        alt_residual_cap_stable=float(cfg["loss"].get("alt_residual_cap_stable", 300.0)),
        alt_residual_cap_disturbed=float(cfg["loss"].get("alt_residual_cap_disturbed", 180.0)),
        alt_residual_cap_boundary=float(cfg["loss"].get("alt_residual_cap_boundary", 120.0)),
        alt_residual_cap_short=float(cfg["loss"].get("alt_residual_cap_short", 100.0)),
        lambda_alt_aux=float(cfg["loss"].get("lambda_alt_aux", 0.0)),
        lambda_aux=float(cfg["loss"].get("lambda_aux", 0.0)),
        lambda_savca_alloc=float(cfg["loss"].get("lambda_savca_alloc", 0.0)),
        lambda_savca_state=float(cfg["loss"].get("lambda_savca_state", 0.0)),
        lambda_savca_smooth=float(cfg["loss"].get("lambda_savca_smooth", 0.0)),
        lambda_savca_center=float(cfg["loss"].get("lambda_savca_center", 0.0)),
        lambda_savca_final_shape=float(cfg["loss"].get("lambda_savca_final_shape", 0.0)),
        lambda_savca_nonlinear=float(cfg["loss"].get("lambda_savca_nonlinear", 0.0)),
        lambda_savca_change_score=float(cfg["loss"].get("lambda_savca_change_score", 0.0)),
        lambda_fltp_shape=float(cfg["loss"].get("lambda_fltp_shape", 0.0)),
        lambda_fltp_center=float(cfg["loss"].get("lambda_fltp_center", 0.0)),
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
        lambda_alt_gate_supervision=float(cfg["loss"].get("lambda_alt_gate_supervision", 0.0)),
        lambda_alt_gate_risk_shrink=float(cfg["loss"].get("lambda_alt_gate_risk_shrink", 0.0)),
        alt_gate_risk_target=float(cfg["loss"].get("alt_gate_risk_target", 0.35)),
        use_first_step_anchor_loss=bool(cfg["loss"].get("use_first_step_anchor_loss", False)),
        first_step_anchor_lambda=float(cfg["loss"].get("first_step_anchor_lambda", 0.0)),
        use_second_step_anchor_loss=bool(cfg["loss"].get("use_second_step_anchor_loss", False)),
        second_step_anchor_lambda=float(cfg["loss"].get("second_step_anchor_lambda", 0.0)),
        use_local_spike_loss=bool(cfg["loss"].get("use_local_spike_loss", False)),
        local_spike_target_bucket=str(cfg["loss"].get("local_spike_target_bucket", "medium")),
        local_spike_target_pattern=str(cfg["loss"].get("local_spike_target_pattern", "two_anchor")),
        local_spike_use_rightstep2=bool(cfg["loss"].get("local_spike_use_rightstep2", True)),
        local_spike_use_second_diff=bool(cfg["loss"].get("local_spike_use_second_diff", True)),
        local_spike_lambda_jump=float(cfg["loss"].get("local_spike_lambda_jump", 0.0)),
        local_spike_lambda_curve=float(cfg["loss"].get("local_spike_lambda_curve", 0.0)),
        use_targeted_rightstep2_loss=bool(cfg["loss"].get("use_targeted_rightstep2_loss", False)),
        target_bucket=str(cfg["loss"].get("target_bucket", "medium")),
        target_anchor_pattern=str(cfg["loss"].get("target_anchor_pattern", "two_anchor")),
        use_target_jump_loss=bool(cfg["loss"].get("use_target_jump_loss", True)),
        use_target_curve_loss=bool(cfg["loss"].get("use_target_curve_loss", True)),
        target_jump_lambda=float(cfg["loss"].get("target_jump_lambda", 0.0)),
        target_curve_lambda=float(cfg["loss"].get("target_curve_lambda", 0.0)),
        use_target_value_rightstep2_loss=bool(cfg["loss"].get("use_target_value_rightstep2_loss", False)),
        target_value_lambda=float(cfg["loss"].get("target_value_lambda", 0.0)),
        target_interp_lambda=float(cfg["loss"].get("target_interp_lambda", 0.5)),
        use_target_rightstep2_boundary_pull=bool(cfg["loss"].get("use_target_rightstep2_boundary_pull", False)),
        target_rightstep2_boundary_pull_lambda=float(cfg["loss"].get("target_rightstep2_boundary_pull_lambda", 0.0)),
        aux_alt_loss_series=str(cfg["loss"].get("aux_alt_loss_series", "pred_pos")),
    )

    model.eval()
    target_norm_cfg = cfg["training"].get("target_norm", {})
    target_norm_stats = None
    if bool(target_norm_cfg.get("enabled", False)):
        target_norm_stats = load_target_stats(Path(cfg["outputs"]["run_dir"]) / "target_model_scaler.json")
        if target_norm_stats is None:
            raise RuntimeError("target_norm is enabled but target_model_scaler.json is missing.")
    sample_rows: list[dict] = []
    sample_plot_cache: list[dict] = []
    branch_sample_rows: list[dict] = []
    branch_tau_rows: list[dict] = []
    total_loss = 0.0
    total_pos = 0.0
    total_smooth = 0.0
    n_batches = 0

    with torch.no_grad():
        use_segment_teacher = bool(cfg.get("training", {}).get("risk_aware", {}).get("use_segment_teacher", True))
        use_alt_baseline_residual = bool(cfg.get("training", {}).get("risk_aware", {}).get("use_alt_baseline_residual", True))
        for batch in loader:
            obs_pos = batch["obs_pos"].to(device)
            obs_mask = batch["obs_mask"].to(device)
            dt_prev = batch["dt_prev"].to(device)
            dt_next = batch["dt_next"].to(device)
            exo = batch["exo"].to(device)
            quality = batch["quality"].to(device)
            global_quality = batch["global_quality"].to(device)
            risk_flag = batch["risk_flag"].to(device) if "risk_flag" in batch else None
            risk_flag_teacher = batch["risk_flag_teacher"].to(device) if ("risk_flag_teacher" in batch and use_segment_teacher) else None
            teacher_scale = batch["teacher_scale"].to(device) if ("teacher_scale" in batch and use_segment_teacher) else None
            segment_bucket = batch["segment_bucket"].to(device) if "segment_bucket" in batch else None
            anchor_pattern = batch["anchor_pattern"].to(device) if "anchor_pattern" in batch else None
            edge_weight = batch["edge_weight"].to(device) if ("edge_weight" in batch and use_segment_teacher) else None
            residual_rmax_m = batch["residual_rmax_m"].to(device) if ("residual_rmax_m" in batch and use_alt_baseline_residual) else None
            residual_rmax_ft = batch["residual_rmax_ft"].to(device) if ("residual_rmax_ft" in batch and use_alt_baseline_residual) else None
            gate_bias = batch["gate_bias"].to(device) if ("gate_bias" in batch and use_segment_teacher) else None
            left_boundary_alt = batch["left_boundary_alt"].to(device) if "left_boundary_alt" in batch else None
            right_boundary_alt = batch["right_boundary_alt"].to(device) if "right_boundary_alt" in batch else None
            target_pos = batch["target_pos"].to(device)
            seq_mask = batch["seq_mask"].to(device)
            target_model, obs_model, coord_ctx = prepare_model_coordinates(
                target_pos=target_pos,
                obs_pos=obs_pos,
                obs_mask=obs_mask,
                seq_mask=seq_mask,
                mode=str(cfg["model"].get("coord_mode", "latlon")),
                u_relative_anchor=bool(cfg["model"].get("u_relative_anchor", False)),
                en_relative_anchor=bool(cfg["model"].get("en_relative_anchor", True)),
                en_incremental=bool(cfg["model"].get("en_incremental", False)),
            )
            alt_target_mode = str(cfg["training"].get("alt_target_robust", {}).get("mode", "none")).lower()
            alt_target_clip = float(cfg["training"].get("alt_target_robust", {}).get("clip_value", 3000.0))
            target_model_raw = target_model
            target_model = apply_alt_target_transform(
                target_model,
                mode=alt_target_mode,
                clip_value=alt_target_clip,
            )
            obs_model = apply_alt_target_transform(
                obs_model,
                mode=alt_target_mode,
                clip_value=alt_target_clip,
            )
            target_for_model = normalize_coords(target_model, target_norm_stats)
            obs_for_model = normalize_coords(obs_model, target_norm_stats)
            anchor_left_raw, anchor_right_raw = build_anchor_pair_tracks(
                obs_pos=obs_pos,
                obs_mask=obs_mask,
                seq_mask=seq_mask,
                ctx=coord_ctx,
            )
            anchor_left_model = normalize_coords(
                apply_alt_target_transform(anchor_left_raw, mode=alt_target_mode, clip_value=alt_target_clip),
                target_norm_stats,
            )
            anchor_right_model = normalize_coords(
                apply_alt_target_transform(anchor_right_raw, mode=alt_target_mode, clip_value=alt_target_clip),
                target_norm_stats,
            )
            left_boundary_alt_model, right_boundary_alt_model = _boundary_alt_from_model_obs(
                obs_for_model=obs_for_model,
                obs_mask=obs_mask,
                seq_mask=seq_mask,
            )
            anchor_alt = build_anchor_alt_tracks(obs_pos=obs_pos, obs_mask=obs_mask, seq_mask=seq_mask)

            out = model(
                obs_pos=obs_for_model,
                obs_mask=obs_mask,
                seq_mask=seq_mask,
                dt_prev=dt_prev,
                dt_next=dt_next,
                exo=exo,
                vertical_exo=batch["vertical_exo"].to(device) if "vertical_exo" in batch else None,
                quality=quality,
                global_quality=global_quality,
                anchor_alt=anchor_alt,
                risk_flag=risk_flag,
                teacher_scale=teacher_scale,
                risk_flag_teacher=risk_flag_teacher,
                segment_bucket=segment_bucket,
                edge_weight=edge_weight,
                residual_rmax_m=residual_rmax_m,
                residual_rmax_ft=residual_rmax_ft,
                gate_bias=gate_bias,
                left_boundary_alt=left_boundary_alt_model,
                right_boundary_alt=right_boundary_alt_model,
                anchor_left=anchor_left_model,
                anchor_right=anchor_right_model,
                target_pos=None,
                teacher_forcing_ratio=0.0,
            )
            pred_model_t = denormalize_coords(out["pred_pos"], target_norm_stats)
            pred_model = invert_alt_target_transform(
                pred_model_t,
                mode=alt_target_mode,
                clip_value=alt_target_clip,
            )
            pred_latlon = restore_to_latlon(pred_model, seq_mask=seq_mask, ctx=coord_ctx)
            pred_aux = out.get("pred_pos_aux")
            pred_aux_latlon = None
            pred_aux_model = None
            if pred_aux is not None:
                pred_aux_model_t = denormalize_coords(pred_aux, target_norm_stats)
                pred_aux_model = invert_alt_target_transform(
                    pred_aux_model_t,
                    mode=alt_target_mode,
                    clip_value=alt_target_clip,
                )
                pred_aux_latlon = restore_to_latlon(pred_aux_model, seq_mask=seq_mask, ctx=coord_ctx)
            mu_f_model_t = denormalize_coords(out["mu_f"], target_norm_stats)
            mu_b_model_t = denormalize_coords(out["mu_b"], target_norm_stats)
            mu_f_model = invert_alt_target_transform(mu_f_model_t, mode=alt_target_mode, clip_value=alt_target_clip)
            mu_b_model = invert_alt_target_transform(mu_b_model_t, mode=alt_target_mode, clip_value=alt_target_clip)
            pred_latlon_f = restore_to_latlon(mu_f_model, seq_mask=seq_mask, ctx=coord_ctx)
            pred_latlon_b = restore_to_latlon(mu_b_model, seq_mask=seq_mask, ctx=coord_ctx)
            loss_dict = criterion(
                pred_pos=out["pred_pos"],
                target_pos=target_for_model,
                obs_mask=obs_mask,
                seq_mask=seq_mask,
                dt_prev=dt_prev,
                dt_next=dt_next,
                logvar=out["logvar_f"],
                alt_base=out.get("alt_base"),
                residual_bound=out.get("residual_bound"),
                delta_alt_pred_norm=out.get("delta_alt_pred_norm"),
                alt_gate=out.get("alt_gate"),
                teacher_scale=teacher_scale,
                risk_flag=risk_flag,
                risk_flag_teacher=risk_flag_teacher,
                segment_bucket=segment_bucket,
                anchor_pattern=anchor_pattern,
                edge_weight=edge_weight,
                pred_pos_main=out.get("pred_pos_main"),
                pred_pos_aux=out.get("pred_pos_aux"),
                pred_pos_aux_supervise_dims=out.get("pred_pos_aux_supervise_dims"),
                left_boundary_alt=left_boundary_alt_model,
                right_boundary_alt=right_boundary_alt_model,
                mu_f=out.get("mu_f"),
                mu_b=out.get("mu_b"),
                savca_alloc_p=out.get("savca_alloc_p"),
                savca_state=out.get("savca_state"),
                savca_alloc_valid=out.get("savca_alloc_valid"),
                savca_target_alt_abs=target_pos[..., 2],
                savca_change_score=out.get("savca_change_score"),
                savca_beta=out.get("savca_beta"),
                savca_beta_floor_pred=out.get("savca_beta_floor_pred"),
                savca_g_linear=out.get("savca_g_linear"),
                savca_g_savca=out.get("savca_g_savca"),
                savca_g_final=out.get("savca_g_final"),
                savca_ref_linear_abs=out.get("savca_ref_linear_abs"),
                savca_ref_savca_abs=out.get("savca_ref_savca_abs"),
                savca_ref_final_abs=out.get("savca_ref_final_abs"),
                fltp_beta=out.get("fltp_beta"),
                fltp_c=out.get("fltp_c"),
                fltp_w=out.get("fltp_w"),
                fltp_g_linear=out.get("fltp_g_linear"),
                fltp_g_sig=out.get("fltp_g_sig"),
                fltp_g_final=out.get("fltp_g_final"),
                fltp_ref_linear_abs=out.get("fltp_ref_linear_abs"),
                fltp_ref_sig_abs=out.get("fltp_ref_sig_abs"),
                fltp_ref_final_abs=out.get("fltp_ref_final_abs"),
            )
            total_loss += float(loss_dict["loss"].detach().cpu())
            total_pos += float(loss_dict["loss_pos"].detach().cpu())
            total_smooth += float(loss_dict["loss_smooth"].detach().cpu())
            n_batches += 1

            pred_np = pred_latlon.detach().cpu().numpy()
            pred_f_np = pred_latlon_f.detach().cpu().numpy()
            pred_b_np = pred_latlon_b.detach().cpu().numpy()
            pred_aux_np = pred_aux_latlon.detach().cpu().numpy() if pred_aux_latlon is not None else None
            pred_model_np = pred_model.detach().cpu().numpy()
            pred_f_model_np = mu_f_model.detach().cpu().numpy()
            pred_b_model_np = mu_b_model.detach().cpu().numpy()
            pred_aux_model_np = pred_aux_model.detach().cpu().numpy() if pred_aux_model is not None else None
            target_model_raw_np = target_model_raw.detach().cpu().numpy()
            target_np = target_pos.detach().cpu().numpy()
            obs_mask_np = batch["obs_mask"].detach().cpu().numpy()
            seq_mask_np = batch["seq_mask"].detach().cpu().numpy()
            quality_np = batch["quality"].detach().cpu().numpy()
            dt_prev_np = batch["dt_prev"].detach().cpu().numpy()
            dt_next_np = batch["dt_next"].detach().cpu().numpy()
            weights_np = out.get("fusion_weights")
            weights_np = weights_np.detach().cpu().numpy() if weights_np is not None else None
            weights_detail_np = out.get("fusion_weights_detail")
            weights_detail_np = weights_detail_np.detach().cpu().numpy() if weights_detail_np is not None else None
            sample_ids = batch["sample_id"]
            flight_ids = batch["flight_id"]

            for i, sid in enumerate(sample_ids):
                valid = seq_mask_np[i] > 0.5
                pred_i = pred_np[i][valid]
                tgt_i = target_np[i][valid]
                obs_i = obs_mask_np[i][valid]
                q_i = quality_np[i][valid]
                gap_i = obs_i <= 0.5

                hav = _haversine_m(pred_i[:, 0], pred_i[:, 1], tgt_i[:, 0], tgt_i[:, 1])
                alt_abs = np.abs(pred_i[:, 2] - tgt_i[:, 2])
                mae = np.mean(np.abs(pred_i - tgt_i))
                rmse = np.sqrt(np.mean((pred_i - tgt_i) ** 2))
                err_i = pred_i - tgt_i
                dim_mae, dim_rmse = _masked_mae_rmse_np(err_i)
                gap_dim_mae, gap_dim_rmse = _masked_mae_rmse_np(err_i, mask=gap_i)
                gap_hav_mean = float(np.mean(hav[gap_i])) if gap_i.any() else float("nan")
                anchor_count = int((obs_i > 0.5).sum())
                gap_count = int((obs_i <= 0.5).sum())
                max_gap = int(_max_gap_len(obs_i))
                jump_rate = float(np.mean(q_i[:, -2])) if q_i.shape[1] >= 2 else 0.0
                smooth_proxy = float(np.mean(q_i[:, -1])) if q_i.shape[1] >= 1 else 0.0

                sample_rows.append(
                    {
                        "sample_id": sid,
                        "flight_id": flight_ids[i],
                        "mae": float(mae),
                        "rmse": float(rmse),
                        "dim0_mae": float(dim_mae[0]),
                        "dim1_mae": float(dim_mae[1]),
                        "dim2_mae": float(dim_mae[2]),
                        "dim0_rmse": float(dim_rmse[0]),
                        "dim1_rmse": float(dim_rmse[1]),
                        "dim2_rmse": float(dim_rmse[2]),
                        "lat_mae": float(dim_mae[0]),
                        "lon_mae": float(dim_mae[1]),
                        "alt_mae": float(dim_mae[2]),
                        "lat_rmse": float(dim_rmse[0]),
                        "lon_rmse": float(dim_rmse[1]),
                        "alt_rmse": float(dim_rmse[2]),
                        "gap_dim0_mae": float(gap_dim_mae[0]),
                        "gap_dim1_mae": float(gap_dim_mae[1]),
                        "gap_dim2_mae": float(gap_dim_mae[2]),
                        "gap_dim0_rmse": float(gap_dim_rmse[0]),
                        "gap_dim1_rmse": float(gap_dim_rmse[1]),
                        "gap_dim2_rmse": float(gap_dim_rmse[2]),
                        "gap_lat_mae": float(gap_dim_mae[0]),
                        "gap_lon_mae": float(gap_dim_mae[1]),
                        "gap_alt_mae": float(gap_dim_mae[2]),
                        "gap_lat_rmse": float(gap_dim_rmse[0]),
                        "gap_lon_rmse": float(gap_dim_rmse[1]),
                        "gap_alt_rmse": float(gap_dim_rmse[2]),
                        "haversine_m": float(np.mean(hav)),
                        "gap_haversine_m": gap_hav_mean,
                        "altitude_mae": float(np.mean(alt_abs)),
                        "anchor_count": anchor_count,
                        "gap_count": gap_count,
                        "max_gap_minutes": max_gap,
                        "has_anchor": bool(anchor_count > 0),
                        "jump_rate": jump_rate,
                        "smooth_proxy_mean": smooth_proxy,
                        "segment_bucket": int(batch["segment_bucket"][i].item()) if "segment_bucket" in batch else -1,
                        "segment_bucket_name": str(batch["segment_bucket_name"][i]) if "segment_bucket_name" in batch else "unknown",
                        "anchor_pattern": int(batch["anchor_pattern"][i].item()) if "anchor_pattern" in batch else -1,
                        "anchor_pattern_name": str(batch["anchor_pattern_name"][i]) if "anchor_pattern_name" in batch else "unknown",
                    }
                )
                pred_f_i = pred_f_np[i][valid]
                pred_b_i = pred_b_np[i][valid]
                pred_model_i = pred_model_np[i][valid]
                pred_f_model_i = pred_f_model_np[i][valid]
                pred_b_model_i = pred_b_model_np[i][valid]
                tau_i = (dt_prev_np[i][valid] / (dt_prev_np[i][valid] + dt_next_np[i][valid] + 1e-6)).astype(np.float64)
                anchor_idx = np.where(obs_i > 0.5)[0]
                delta_z_abs = float(abs(tgt_i[anchor_idx[-1], 2] - tgt_i[anchor_idx[0], 2])) if anchor_idx.size >= 2 else float("nan")
                branch_triplets = [
                    ("fused", pred_i, pred_model_i),
                    ("fwd", pred_f_i, pred_f_model_i),
                    ("bwd", pred_b_i, pred_b_model_i),
                ]
                if pred_aux_np is not None and pred_aux_model_np is not None:
                    branch_triplets.append(("aux", pred_aux_np[i][valid], pred_aux_model_np[i][valid]))
                for branch_name, branch_pred_i, branch_model_i in branch_triplets:
                    berr = branch_pred_i - tgt_i
                    b_gap_dim_mae, b_gap_dim_rmse = _masked_mae_rmse_np(berr, mask=gap_i)
                    b_pstd, b_corr = _masked_altrel_stats_np(branch_model_i[:, 2], target_model_raw_np[i][valid, 2], gap_i)
                    branch_sample_rows.append(
                        {
                            "sample_id": sid,
                            "flight_id": flight_ids[i],
                            "branch": branch_name,
                            "gap_len_bucket": _gap_len_bucket(max_gap),
                            "delta_z_abs": delta_z_abs,
                            "delta_z_bucket": _delta_z_bucket(delta_z_abs) if np.isfinite(delta_z_abs) else "unknown",
                            "max_gap_minutes": max_gap,
                            "gap_lat_rmse": float(b_gap_dim_rmse[0]),
                            "gap_lon_rmse": float(b_gap_dim_rmse[1]),
                            "gap_alt_rmse": float(b_gap_dim_rmse[2]),
                            "gap_alt_mae": float(b_gap_dim_mae[2]),
                            "altrel_pred_std": float(b_pstd),
                            "altrel_corr": float(b_corr),
                        }
                    )
                if weights_np is not None:
                    wf_i = weights_np[i][valid, 0]
                    branch_sample_rows.append(
                        {
                            "sample_id": sid,
                            "flight_id": flight_ids[i],
                            "branch": "fusion_weight",
                            "gap_len_bucket": _gap_len_bucket(max_gap),
                            "delta_z_abs": delta_z_abs,
                            "delta_z_bucket": _delta_z_bucket(delta_z_abs) if np.isfinite(delta_z_abs) else "unknown",
                            "max_gap_minutes": max_gap,
                            "gap_lat_rmse": float("nan"),
                            "gap_lon_rmse": float("nan"),
                            "gap_alt_rmse": float("nan"),
                            "gap_alt_mae": float("nan"),
                            "altrel_pred_std": float(np.std(wf_i[gap_i])) if gap_i.any() else float("nan"),
                            "altrel_corr": float(np.mean(wf_i[gap_i])) if gap_i.any() else float("nan"),
                        }
                    )
                    for region_name, region_mask in [
                        ("left", gap_i & (tau_i < 0.25)),
                        ("mid", gap_i & (tau_i >= 0.25) & (tau_i <= 0.75)),
                        ("right", gap_i & (tau_i > 0.75)),
                    ]:
                        if not region_mask.any():
                            continue
                        row = {
                            "sample_id": sid,
                            "flight_id": flight_ids[i],
                            "tau_region": region_name,
                            "wf_mean": float(np.mean(wf_i[region_mask])),
                            "wf_std": float(np.std(wf_i[region_mask])),
                            "fused_gap_alt_rmse": float(np.sqrt(np.mean((pred_i[region_mask, 2] - tgt_i[region_mask, 2]) ** 2))),
                            "fwd_gap_alt_rmse": float(np.sqrt(np.mean((pred_f_i[region_mask, 2] - tgt_i[region_mask, 2]) ** 2))),
                            "bwd_gap_alt_rmse": float(np.sqrt(np.mean((pred_b_i[region_mask, 2] - tgt_i[region_mask, 2]) ** 2))),
                            "fused_gap_lat_rmse": float(np.sqrt(np.mean((pred_i[region_mask, 0] - tgt_i[region_mask, 0]) ** 2))),
                            "fwd_gap_lat_rmse": float(np.sqrt(np.mean((pred_f_i[region_mask, 0] - tgt_i[region_mask, 0]) ** 2))),
                            "bwd_gap_lat_rmse": float(np.sqrt(np.mean((pred_b_i[region_mask, 0] - tgt_i[region_mask, 0]) ** 2))),
                            "fused_gap_lon_rmse": float(np.sqrt(np.mean((pred_i[region_mask, 1] - tgt_i[region_mask, 1]) ** 2))),
                            "fwd_gap_lon_rmse": float(np.sqrt(np.mean((pred_f_i[region_mask, 1] - tgt_i[region_mask, 1]) ** 2))),
                            "bwd_gap_lon_rmse": float(np.sqrt(np.mean((pred_b_i[region_mask, 1] - tgt_i[region_mask, 1]) ** 2))),
                        }
                        if weights_detail_np is not None:
                            wd = weights_detail_np[i][valid]
                            if wd.shape[1] == 2:
                                row["wf_xy_mean"] = float(np.mean(wd[region_mask, 0, 0]))
                                row["wf_xy_std"] = float(np.std(wd[region_mask, 0, 0]))
                                row["wf_z_mean"] = float(np.mean(wd[region_mask, 1, 0]))
                                row["wf_z_std"] = float(np.std(wd[region_mask, 1, 0]))
                            elif wd.shape[1] == 3:
                                row["wf_lat_mean"] = float(np.mean(wd[region_mask, 0, 0]))
                                row["wf_lon_mean"] = float(np.mean(wd[region_mask, 1, 0]))
                                row["wf_alt_mean"] = float(np.mean(wd[region_mask, 2, 0]))
                                row["wf_lat_std"] = float(np.std(wd[region_mask, 0, 0]))
                                row["wf_lon_std"] = float(np.std(wd[region_mask, 1, 0]))
                                row["wf_alt_std"] = float(np.std(wd[region_mask, 2, 0]))
                        branch_tau_rows.append(row)
                sample_plot_cache.append(
                    {
                        "sample_id": sid,
                        "flight_id": flight_ids[i],
                        "track_key": _track_group_from_sample_id(sid),
                        "times": batch["times"][i][: int(valid.sum())],
                        "pred": pred_i,
                        "target": tgt_i,
                        "obs_mask": obs_i,
                    }
                )

    sample_df = pd.DataFrame(sample_rows).sort_values("haversine_m", ascending=False).reset_index(drop=True)
    if sample_df.empty:
        raise RuntimeError(f"FATAL: no evaluated samples in split={args.split} after main-task gating.")
    _fatal_if_main_contains_no_anchor(sample_df, split_name=args.split)
    branch_df = pd.DataFrame(branch_sample_rows)
    tau_df = pd.DataFrame(branch_tau_rows)

    smooth_q75 = float(sample_df["smooth_proxy_mean"].quantile(0.75)) if not sample_df.empty else 0.0
    sample_df["is_long_gap"] = sample_df["max_gap_minutes"] >= int(args.long_gap_threshold)
    sample_df["is_few_anchor"] = sample_df["anchor_count"] <= int(args.few_anchor_threshold)
    sample_df["is_low_quality"] = (sample_df["jump_rate"] > 0.0) | (sample_df["smooth_proxy_mean"] >= smooth_q75)

    main_df = sample_df.copy()

    # Audit table is data-level no-anchor set, never mixed into main metrics.
    no_anchor_sample_df = (
        split_no_anchor_df.groupby(["sample_id", "flight_id"], as_index=False)
        .agg(
            rows=("sample_id", "size"),
            max_gap_minutes=("gap_len", lambda x: float(pd.to_numeric(x, errors="coerce").fillna(0.0).max())),
            max_abs_altrel=("alt_rel_prev_anchor", lambda x: float(np.abs(pd.to_numeric(x, errors="coerce").fillna(0.0)).max())),
        )
        .sort_values("max_abs_altrel", ascending=False)
        .reset_index(drop=True)
        if not split_no_anchor_df.empty
        else pd.DataFrame(columns=["sample_id", "flight_id", "rows", "max_gap_minutes", "max_abs_altrel"])
    )

    overall = {
        "loss": total_loss / max(1, n_batches),
        "loss_pos": total_pos / max(1, n_batches),
        "loss_smooth": total_smooth / max(1, n_batches),
        "mae": _metric_mean(main_df, "mae"),
        "rmse": _metric_mean(main_df, "rmse"),
        "dim0_mae": _metric_mean(main_df, "dim0_mae"),
        "dim1_mae": _metric_mean(main_df, "dim1_mae"),
        "dim2_mae": _metric_mean(main_df, "dim2_mae"),
        "dim0_rmse": _metric_mean(main_df, "dim0_rmse"),
        "dim1_rmse": _metric_mean(main_df, "dim1_rmse"),
        "dim2_rmse": _metric_mean(main_df, "dim2_rmse"),
        "lat_mae": _metric_mean(main_df, "lat_mae"),
        "lon_mae": _metric_mean(main_df, "lon_mae"),
        "alt_mae": _metric_mean(main_df, "alt_mae"),
        "lat_rmse": _metric_mean(main_df, "lat_rmse"),
        "lon_rmse": _metric_mean(main_df, "lon_rmse"),
        "alt_rmse": _metric_mean(main_df, "alt_rmse"),
        "gap_dim0_mae": _metric_mean(main_df, "gap_dim0_mae"),
        "gap_dim1_mae": _metric_mean(main_df, "gap_dim1_mae"),
        "gap_dim2_mae": _metric_mean(main_df, "gap_dim2_mae"),
        "gap_dim0_rmse": _metric_mean(main_df, "gap_dim0_rmse"),
        "gap_dim1_rmse": _metric_mean(main_df, "gap_dim1_rmse"),
        "gap_dim2_rmse": _metric_mean(main_df, "gap_dim2_rmse"),
        "gap_lat_mae": _metric_mean(main_df, "gap_lat_mae"),
        "gap_lon_mae": _metric_mean(main_df, "gap_lon_mae"),
        "gap_alt_mae": _metric_mean(main_df, "gap_alt_mae"),
        "gap_lat_rmse": _metric_mean(main_df, "gap_lat_rmse"),
        "gap_lon_rmse": _metric_mean(main_df, "gap_lon_rmse"),
        "gap_alt_rmse": _metric_mean(main_df, "gap_alt_rmse"),
        "haversine_m": _metric_mean(main_df, "haversine_m"),
        "gap_haversine_m": _metric_mean(main_df, "gap_haversine_m"),
        "altitude_mae": _metric_mean(main_df, "altitude_mae"),
    }

    subsets = {
        "overall": main_df,
        "long_gap": main_df[main_df["is_long_gap"]],
        "few_anchor": main_df[main_df["is_few_anchor"]],
        "low_quality": main_df[main_df["is_low_quality"]],
    }
    subset_stats = {
        name: {
            "count": int(len(sdf)),
            "mae": _metric_mean(sdf, "mae"),
            "rmse": _metric_mean(sdf, "rmse"),
            "dim0_mae": _metric_mean(sdf, "dim0_mae"),
            "dim1_mae": _metric_mean(sdf, "dim1_mae"),
            "dim2_mae": _metric_mean(sdf, "dim2_mae"),
            "dim0_rmse": _metric_mean(sdf, "dim0_rmse"),
            "dim1_rmse": _metric_mean(sdf, "dim1_rmse"),
            "dim2_rmse": _metric_mean(sdf, "dim2_rmse"),
            "lat_mae": _metric_mean(sdf, "lat_mae"),
            "lon_mae": _metric_mean(sdf, "lon_mae"),
            "alt_mae": _metric_mean(sdf, "alt_mae"),
            "lat_rmse": _metric_mean(sdf, "lat_rmse"),
            "lon_rmse": _metric_mean(sdf, "lon_rmse"),
            "alt_rmse": _metric_mean(sdf, "alt_rmse"),
            "gap_dim0_mae": _metric_mean(sdf, "gap_dim0_mae"),
            "gap_dim1_mae": _metric_mean(sdf, "gap_dim1_mae"),
            "gap_dim2_mae": _metric_mean(sdf, "gap_dim2_mae"),
            "gap_dim0_rmse": _metric_mean(sdf, "gap_dim0_rmse"),
            "gap_dim1_rmse": _metric_mean(sdf, "gap_dim1_rmse"),
            "gap_dim2_rmse": _metric_mean(sdf, "gap_dim2_rmse"),
            "gap_lat_mae": _metric_mean(sdf, "gap_lat_mae"),
            "gap_lon_mae": _metric_mean(sdf, "gap_lon_mae"),
            "gap_alt_mae": _metric_mean(sdf, "gap_alt_mae"),
            "gap_lat_rmse": _metric_mean(sdf, "gap_lat_rmse"),
            "gap_lon_rmse": _metric_mean(sdf, "gap_lon_rmse"),
            "gap_alt_rmse": _metric_mean(sdf, "gap_alt_rmse"),
            "haversine_m": _metric_mean(sdf, "haversine_m"),
            "gap_haversine_m": _metric_mean(sdf, "gap_haversine_m"),
            "altitude_mae": _metric_mean(sdf, "altitude_mae"),
        }
        for name, sdf in subsets.items()
    }

    stats = {
        **overall,
        "variables": {
            "dim0": "lat",
            "dim1": "lon",
            "dim2": "alt",
            "space": "restored_latlon",
            "masking": "seq_mask(valid) and obs_mask(anchor/gap split)",
        },
        "subsets": subset_stats,
    }

    out_dir = Path(cfg["outputs"]["run_dir"])
    main_metrics_json = out_dir / f"main_task_metrics_{args.split}.json"
    main_per_sample_csv = out_dir / f"main_task_metrics_{args.split}_per_sample.csv"
    main_metrics_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    sample_df.to_csv(main_per_sample_csv, index=False)

    if not branch_df.empty:
        branch_rows_csv = out_dir / f"branch_diagnostics_{args.split}_per_sample.csv"
        branch_summary_csv = out_dir / f"branch_diagnostics_{args.split}_summary.csv"
        branch_summary_json = out_dir / f"branch_diagnostics_{args.split}_summary.json"
        branch_df.to_csv(branch_rows_csv, index=False)
        branch_summary = (
            branch_df[branch_df["branch"].isin(["fused", "fwd", "bwd"])]
            .groupby(["branch", "gap_len_bucket", "delta_z_bucket"], dropna=False)
            .agg(
                count=("sample_id", "size"),
                gap_lat_rmse=("gap_lat_rmse", "mean"),
                gap_lon_rmse=("gap_lon_rmse", "mean"),
                gap_alt_rmse=("gap_alt_rmse", "mean"),
                gap_alt_mae=("gap_alt_mae", "mean"),
                altrel_pred_std=("altrel_pred_std", "mean"),
                altrel_corr=("altrel_corr", "mean"),
            )
            .reset_index()
        )
        branch_summary.to_csv(branch_summary_csv, index=False)
        branch_summary_json.write_text(branch_summary.to_json(orient="records", indent=2), encoding="utf-8")

        fused_df = branch_df[branch_df["branch"] == "fused"][["sample_id", "gap_alt_rmse"]].rename(columns={"gap_alt_rmse": "err_p"})
        fwd_df = branch_df[branch_df["branch"] == "fwd"][["sample_id", "gap_alt_rmse"]].rename(columns={"gap_alt_rmse": "err_f"})
        bwd_df = branch_df[branch_df["branch"] == "bwd"][["sample_id", "gap_alt_rmse"]].rename(columns={"gap_alt_rmse": "err_b"})
        eff = fused_df.merge(fwd_df, on="sample_id").merge(bwd_df, on="sample_id")
        best = np.minimum(eff["err_f"].to_numpy(), eff["err_b"].to_numpy())
        worst = np.maximum(eff["err_f"].to_numpy(), eff["err_b"].to_numpy())
        eff_summary = {
            "fused_better_than_best_branch_rate": float(np.mean(eff["err_p"].to_numpy() <= best)),
            "fused_worse_than_both_rate": float(np.mean(eff["err_p"].to_numpy() > worst)),
            "fused_between_two_branches_rate": float(np.mean((eff["err_p"].to_numpy() > best) & (eff["err_p"].to_numpy() <= worst))),
        }
        (out_dir / f"fusion_effectiveness_{args.split}.json").write_text(json.dumps(eff_summary, indent=2), encoding="utf-8")

        fwd_s = branch_df[branch_df["branch"] == "fwd"][["sample_id", "gap_lat_rmse", "gap_lon_rmse", "gap_alt_rmse"]]
        bwd_s = branch_df[branch_df["branch"] == "bwd"][["sample_id", "gap_lat_rmse", "gap_lon_rmse", "gap_alt_rmse"]]
        fb = fwd_s.merge(bwd_s, on="sample_id", suffixes=("_f", "_b"))
        oracle = {
            "oracle_dimwise_best_gap_lat_rmse": float(np.mean(np.minimum(fb["gap_lat_rmse_f"], fb["gap_lat_rmse_b"]))),
            "oracle_dimwise_best_gap_lon_rmse": float(np.mean(np.minimum(fb["gap_lon_rmse_f"], fb["gap_lon_rmse_b"]))),
            "oracle_dimwise_best_gap_alt_rmse": float(np.mean(np.minimum(fb["gap_alt_rmse_f"], fb["gap_alt_rmse_b"]))),
        }
        (out_dir / f"oracle_dimwise_best_{args.split}.json").write_text(json.dumps(oracle, indent=2), encoding="utf-8")

    if not tau_df.empty:
        tau_csv = out_dir / f"branch_tau_regions_{args.split}.csv"
        tau_summary_csv = out_dir / f"branch_tau_regions_{args.split}_summary.csv"
        tau_df.to_csv(tau_csv, index=False)
        tau_summary = tau_df.groupby("tau_region", dropna=False).agg(
            count=("sample_id", "size"),
            wf_mean=("wf_mean", "mean"),
            wf_std=("wf_std", "mean"),
            fused_gap_alt_rmse=("fused_gap_alt_rmse", "mean"),
            fwd_gap_alt_rmse=("fwd_gap_alt_rmse", "mean"),
            bwd_gap_alt_rmse=("bwd_gap_alt_rmse", "mean"),
            fused_gap_lat_rmse=("fused_gap_lat_rmse", "mean"),
            fwd_gap_lat_rmse=("fwd_gap_lat_rmse", "mean"),
            bwd_gap_lat_rmse=("bwd_gap_lat_rmse", "mean"),
            fused_gap_lon_rmse=("fused_gap_lon_rmse", "mean"),
            fwd_gap_lon_rmse=("fwd_gap_lon_rmse", "mean"),
            bwd_gap_lon_rmse=("bwd_gap_lon_rmse", "mean"),
        ).reset_index()
        tau_summary.to_csv(tau_summary_csv, index=False)

    audit_no_anchor_summary = {
        **split_anchor_audit,
        "rows_no_anchor": int(len(split_no_anchor_df)),
        "samples_no_anchor": int(no_anchor_sample_df["sample_id"].nunique()) if not no_anchor_sample_df.empty else 0,
    }
    audit_no_anchor_json = out_dir / f"audit_has_no_anchor_{args.split}_summary.json"
    audit_no_anchor_rows_csv = out_dir / f"audit_has_no_anchor_{args.split}_rows.csv"
    audit_no_anchor_topk_csv = out_dir / f"audit_has_no_anchor_{args.split}_topk.csv"
    audit_no_anchor_json.write_text(json.dumps(audit_no_anchor_summary, indent=2), encoding="utf-8")
    split_no_anchor_df.to_csv(audit_no_anchor_rows_csv, index=False)
    no_anchor_sample_df.head(20).to_csv(audit_no_anchor_topk_csv, index=False)

    model_name = Path(ckpt_path).stem
    summary_dim = pd.DataFrame(
        [
            {
                "model": model_name,
                "split": args.split,
                "mae_dim0": overall["dim0_mae"],
                "mae_dim1": overall["dim1_mae"],
                "mae_dim2": overall["dim2_mae"],
                "rmse_dim0": overall["dim0_rmse"],
                "rmse_dim1": overall["dim1_rmse"],
                "rmse_dim2": overall["dim2_rmse"],
                "gap_mae_dim0": overall["gap_dim0_mae"],
                "gap_mae_dim1": overall["gap_dim1_mae"],
                "gap_mae_dim2": overall["gap_dim2_mae"],
                "gap_rmse_dim0": overall["gap_dim0_rmse"],
                "gap_rmse_dim1": overall["gap_dim1_rmse"],
                "gap_rmse_dim2": overall["gap_dim2_rmse"],
            }
        ]
    )
    summary_latlon = pd.DataFrame(
        [
            {
                "model": model_name,
                "split": args.split,
                "mae_lat": overall["lat_mae"],
                "mae_lon": overall["lon_mae"],
                "mae_alt": overall["alt_mae"],
                "rmse_lat": overall["lat_rmse"],
                "rmse_lon": overall["lon_rmse"],
                "rmse_alt": overall["alt_rmse"],
                "gap_mae_lat": overall["gap_lat_mae"],
                "gap_mae_lon": overall["gap_lon_mae"],
                "gap_mae_alt": overall["gap_alt_mae"],
                "gap_rmse_lat": overall["gap_lat_rmse"],
                "gap_rmse_lon": overall["gap_lon_rmse"],
                "gap_rmse_alt": overall["gap_alt_rmse"],
            }
        ]
    )
    # Main table for paper-style reporting: prioritize per-variable overall/gap MAE/RMSE.
    summary_main = pd.DataFrame(
        [
            {
                "model": model_name,
                "split": args.split,
                "lat_mae": overall["lat_mae"],
                "lon_mae": overall["lon_mae"],
                "alt_mae": overall["alt_mae"],
                "lat_rmse": overall["lat_rmse"],
                "lon_rmse": overall["lon_rmse"],
                "alt_rmse": overall["alt_rmse"],
                "gap_lat_mae": overall["gap_lat_mae"],
                "gap_lon_mae": overall["gap_lon_mae"],
                "gap_alt_mae": overall["gap_alt_mae"],
                "gap_lat_rmse": overall["gap_lat_rmse"],
                "gap_lon_rmse": overall["gap_lon_rmse"],
                "gap_alt_rmse": overall["gap_alt_rmse"],
            }
        ]
    )
    summary_dim_csv = out_dir / f"main_task_metrics_{args.split}_summary_dim.csv"
    summary_dim_json = out_dir / f"main_task_metrics_{args.split}_summary_dim.json"
    summary_latlon_csv = out_dir / f"main_task_metrics_{args.split}_summary_latlon.csv"
    summary_latlon_json = out_dir / f"main_task_metrics_{args.split}_summary_latlon.json"
    summary_main_csv = out_dir / f"main_task_metrics_{args.split}_main_table.csv"
    summary_main_json = out_dir / f"main_task_metrics_{args.split}_main_table.json"
    summary_dim.to_csv(summary_dim_csv, index=False)
    summary_dim_json.write_text(summary_dim.to_json(orient="records", indent=2), encoding="utf-8")
    summary_latlon.to_csv(summary_latlon_csv, index=False)
    summary_latlon_json.write_text(summary_latlon.to_json(orient="records", indent=2), encoding="utf-8")
    summary_main.to_csv(summary_main_csv, index=False)
    summary_main_json.write_text(summary_main.to_json(orient="records", indent=2), encoding="utf-8")

    # Segment-bucket / anchor-pattern grouped table for fair risk-aware comparison.
    if "segment_bucket_name" in sample_df.columns:
        by_bucket = (
            sample_df.groupby("segment_bucket_name", as_index=False)
            .agg(
                count=("sample_id", "count"),
                lat_rmse=("lat_rmse", "mean"),
                lon_rmse=("lon_rmse", "mean"),
                alt_rmse=("alt_rmse", "mean"),
                gap_alt_rmse=("gap_alt_rmse", "mean"),
            )
            .sort_values("count", ascending=False)
        )
    else:
        by_bucket = pd.DataFrame(columns=["segment_bucket_name", "count", "lat_rmse", "lon_rmse", "alt_rmse", "gap_alt_rmse"])
    by_bucket.to_csv(out_dir / f"main_task_metrics_{args.split}_by_segment_bucket.csv", index=False)
    if "anchor_pattern_name" in sample_df.columns:
        by_pattern = (
            sample_df.groupby("anchor_pattern_name", as_index=False)
            .agg(
                count=("sample_id", "count"),
                alt_rmse=("alt_rmse", "mean"),
                gap_alt_rmse=("gap_alt_rmse", "mean"),
            )
            .sort_values("count", ascending=False)
        )
    else:
        by_pattern = pd.DataFrame(columns=["anchor_pattern_name", "count", "alt_rmse", "gap_alt_rmse"])
    by_pattern.to_csv(out_dir / f"main_task_metrics_{args.split}_by_anchor_pattern.csv", index=False)

    plots_dir = Path(args.plots_dir) if args.plots_dir else Path(cfg["outputs"]["run_dir"]) / f"plots_{args.split}"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot_subset_bars(plots_dir / "subset_haversine_bar.png", subset_stats)
    _plot_haversine_hist(plots_dir / "haversine_hist.png", sample_df)
    sample_df["track_key"] = sample_df["sample_id"].map(_track_group_from_sample_id)

    track_cache: dict[str, list[dict]] = {}
    track_to_flight: dict[str, str] = {}
    for x in sample_plot_cache:
        key = str(x.get("track_key", _track_group_from_sample_id(x["sample_id"])))
        track_cache.setdefault(key, []).append(x)
        track_to_flight.setdefault(key, str(x["flight_id"]))

    track_rank = (
        sample_df.groupby("track_key", as_index=False)["haversine_m"].mean()
        .sort_values("haversine_m", ascending=False)
        .reset_index(drop=True)
    )
    worst_tracks = []
    for key in track_rank["track_key"].head(max(0, int(args.plot_count))).tolist():
        if key in track_cache:
            worst_tracks.append({
                "flight_id": track_to_flight.get(key, "unknown"),
                "track_key": key,
                "samples": track_cache[key],
                "split": args.split,
            })
    _plot_flight_tracks(plots_dir, worst_tracks, plot_count=max(0, int(args.plot_count)))

    print(json.dumps(stats, indent=2))
    print(f"[ok] main_task_metrics_json={main_metrics_json}")
    print(f"[ok] main_task_per_sample_csv={main_per_sample_csv}")
    print(f"[ok] main_task_summary_dim_csv={summary_dim_csv}")
    print(f"[ok] main_task_summary_dim_json={summary_dim_json}")
    print(f"[ok] main_task_summary_latlon_csv={summary_latlon_csv}")
    print(f"[ok] main_task_summary_latlon_json={summary_latlon_json}")
    print(f"[ok] main_task_summary_main_csv={summary_main_csv}")
    print(f"[ok] main_task_summary_main_json={summary_main_json}")
    print(f"[ok] audit_no_anchor_summary_json={audit_no_anchor_json}")
    print(f"[ok] audit_no_anchor_rows_csv={audit_no_anchor_rows_csv}")
    print(f"[ok] audit_no_anchor_topk_csv={audit_no_anchor_topk_csv}")
    print(f"[ok] plots_dir={plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
