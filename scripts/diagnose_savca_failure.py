from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.datasets import DatasetConfig, TrajectoryDataset, trajectory_collate_fn
from src.losses import TrajectoryLoss
from src.models import TrajectoryRecoveryModel
from src.preprocessing.standardize import apply_standardizer, load_standardizer
from src.training.alt_target import apply_alt_target_transform, invert_alt_target_transform
from src.training.altitude_governance import (
    add_anchor_alt_features,
    add_vertical_v2_features,
    apply_alt_label_governance,
)
from src.training.coords import build_anchor_alt_tracks, prepare_model_coordinates, restore_to_latlon
from src.training.target_norm import denormalize_coords, load_target_stats, normalize_coords
from src.training.utils import load_config, set_seed, split_by_flight_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose SAVCA failure modes.")
    parser.add_argument(
        "--config",
        default="outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_savca_supervised_fixlabel_v1/configs/savca_supervised.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_savca_supervised_fixlabel_v1/savca_supervised/best.pt",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/runs/0523/savca_failure_diagnosis_fixlabel_v1",
    )
    parser.add_argument("--case-count", type=int, default=6)
    parser.add_argument("--grad-batches", type=int, default=6)
    parser.add_argument(
        "--model-splits",
        default="val,test",
        help="Comma-separated splits for full model forward diagnosis. Train split difficulty/audit stats are still computed separately.",
    )
    return parser


def _normalize_stage_weights(weights: dict[str, float]) -> dict[str, float]:
    out = {str(k): max(0.0, float(v)) for k, v in (weights or {}).items()}
    s = float(sum(out.values()))
    if s <= 0.0:
        return {"stage1": 1.0, "stage2": 0.0, "stage3": 0.0}
    return {k: v / s for k, v in out.items()}


def _pick_schedule_weights(epoch: int, schedule: list[dict]) -> dict[str, float]:
    if not schedule:
        return {"stage1": 1.0, "stage2": 0.0, "stage3": 0.0}
    for item in schedule:
        if int(epoch) <= int(item.get("end_epoch", 0)):
            return _normalize_stage_weights(item.get("weights", {}))
    return _normalize_stage_weights(schedule[-1].get("weights", {}))


def _sample_ids(ids: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0 or len(ids) == 0:
        return np.array([], dtype=object)
    replace = n > len(ids)
    return rng.choice(ids, size=n, replace=replace)


def _sample_has_anchor(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    obs = pd.to_numeric(frame["obs_mask"], errors="coerce").fillna(0.0)
    by = frame.assign(_obs_anchor=(obs > 0.5)).groupby("sample_id")["_obs_anchor"].any()
    return by.astype(bool)


def _anchor_gate_filter(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    has_anchor = _sample_has_anchor(frame)
    keep_ids = set(has_anchor[has_anchor].index.astype(str).tolist())
    return frame[frame["sample_id"].astype(str).isin(keep_ids)].copy()


def _build_training_like_splits(cfg: dict) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    curriculum_cfg = cfg["training"].get("curriculum", {})
    use_curriculum = bool(curriculum_cfg.get("enabled", False))
    split_cfg = cfg["data"]["split"]
    seed = int(cfg.get("seed", 42))
    flight_id_col = cfg["data"]["flight_id_col"]
    if not use_curriculum:
        df = pd.read_parquet(cfg["data"]["samples_path"])
        splits = split_by_flight_id(
            df=df,
            flight_id_col=flight_id_col,
            train_ratio=float(split_cfg["train_ratio"]),
            val_ratio=float(split_cfg["val_ratio"]),
            seed=seed,
        )
        for sp in ["train", "val", "test"]:
            splits[sp]["audit_stage"] = "single"
        return splits, {}

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
    train_frames: list[pd.DataFrame] = []
    stage_train_frames: dict[str, pd.DataFrame] = {}
    for stage_name, stage_df in stage_frames.items():
        fid = stage_df[flight_id_col].astype(str)
        train_part = stage_df[fid.isin(train_ids)].copy()
        train_part["audit_stage"] = stage_name
        train_frames.append(train_part)
        stage_train_frames[stage_name] = train_part.copy()
    val_df = stage_frames[val_stage]
    val_fid = val_df[flight_id_col].astype(str)
    splits = {
        "train": pd.concat(train_frames, ignore_index=True),
        "val": val_df[val_fid.isin(val_ids)].copy(),
        "test": val_df[val_fid.isin(test_ids)].copy(),
    }
    splits["val"]["audit_stage"] = val_stage
    splits["test"]["audit_stage"] = val_stage
    return splits, stage_train_frames


def _apply_training_ready_preprocess(
    cfg: dict,
    splits_raw: dict[str, pd.DataFrame],
    stage_train_frames_raw: dict[str, pd.DataFrame],
    run_dir: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    splits_ready = {k: add_vertical_v2_features(add_anchor_alt_features(v)) for k, v in splits_raw.items()}
    stage_train_ready = {k: add_vertical_v2_features(add_anchor_alt_features(v)) for k, v in stage_train_frames_raw.items()}

    splits_ready = {k: _anchor_gate_filter(v) for k, v in splits_ready.items()}
    keep_train_ids = set(splits_ready["train"]["sample_id"].astype(str).unique().tolist())
    stage_train_ready = {
        k: v[v["sample_id"].astype(str).isin(keep_train_ids)].copy()
        for k, v in stage_train_ready.items()
    }

    lg_cfg = cfg["training"].get("alt_label_governance", {})
    train_post, _ = apply_alt_label_governance(splits_ready["train"], lg_cfg, out_dir=run_dir)
    splits_ready["train"] = train_post
    keep_train_ids = set(splits_ready["train"]["sample_id"].astype(str).unique().tolist())
    stage_train_ready = {
        k: v[v["sample_id"].astype(str).isin(keep_train_ids)].copy()
        for k, v in stage_train_ready.items()
    }

    scaler = load_standardizer(run_dir / "feature_standardizer.json")
    if scaler is None:
        raise RuntimeError(f"Missing feature_standardizer.json under {run_dir}")
    splits_std = {k: apply_standardizer(v, scaler) for k, v in splits_ready.items()}
    return splits_ready, splits_std, stage_train_ready


def _build_dataset_cfg(cfg: dict) -> DatasetConfig:
    return DatasetConfig(
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
    )


def _build_model(cfg: dict, device: torch.device) -> TrajectoryRecoveryModel:
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
        fusion_position_prior_enabled=bool(cfg["model"].get("fusion_position_prior_enabled", False)),
        fusion_position_prior_deviation=float(cfg["model"].get("fusion_position_prior_deviation", 0.30)),
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
        main_rmax_m=float(cfg["model"].get("main_rmax_m", 0.0)),
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
    return model


def _build_criterion(cfg: dict) -> TrajectoryLoss:
    return TrajectoryLoss(
        anchor_weight=float(cfg["loss"]["anchor_weight"]),
        gap_weight=float(cfg["loss"]["gap_weight"]),
        lambda_pos=float(cfg["loss"]["lambda_pos"]),
        lambda_smooth=float(cfg["loss"]["lambda_smooth"]),
        lambda_unc=float(cfg["loss"]["lambda_unc"]),
        dim_weights=cfg["loss"].get("dim_weights"),
        alpha_vertical=float(cfg["loss"].get("alpha_vertical", 1.0)),
        exo_feature_names=list(cfg["data"].get("exo_cols", [])),
        lambda_cruise_phys=float(cfg["loss"].get("lambda_cruise_phys", 0.0)),
        fusion_reg_lambda=float(cfg["loss"].get("fusion_reg_lambda", 0.0)),
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
        lambda_alt_aux=float(cfg["loss"].get("lambda_alt_aux", 0.0)),
        lambda_savca_alloc=float(cfg["loss"].get("lambda_savca_alloc", 0.0)),
        lambda_savca_state=float(cfg["loss"].get("lambda_savca_state", 0.0)),
        lambda_savca_smooth=float(cfg["loss"].get("lambda_savca_smooth", 0.0)),
        savca_alloc_min_anchor_delta_m=float(cfg["loss"].get("savca_alloc_min_anchor_delta_m", 30.0)),
        savca_state_min_anchor_delta_m=float(cfg["loss"].get("savca_state_min_anchor_delta_m", 30.0)),
        savca_active_min_anchor_delta_m=float(cfg["loss"].get("savca_active_min_anchor_delta_m", 30.0)),
        savca_change_deadband_m=float(cfg["loss"].get("savca_change_deadband_m", 3.0)),
        savca_label_median_window=int(cfg["loss"].get("savca_label_median_window", 5)),
        savca_active_ratio_to_max=float(cfg["loss"].get("savca_active_ratio_to_max", 0.25)),
        savca_active_min_abs_change_m=float(cfg["loss"].get("savca_active_min_abs_change_m", 10.0)),
        savca_active_expand_steps=int(cfg["loss"].get("savca_active_expand_steps", 1)),
        lambda_alt_gate_supervision=float(cfg["loss"].get("lambda_alt_gate_supervision", 0.0)),
        lambda_alt_gate_risk_shrink=float(cfg["loss"].get("lambda_alt_gate_risk_shrink", 0.0)),
        alt_gate_risk_target=float(cfg["loss"].get("alt_gate_risk_target", 0.35)),
        aux_alt_loss_series=str(cfg["loss"].get("aux_alt_loss_series", "pred_pos")),
    )


def _find_intervals(anchor_mask_1d: np.ndarray, valid_mask_1d: np.ndarray) -> list[tuple[int, int]]:
    anchors = np.where((anchor_mask_1d > 0.5) & valid_mask_1d)[0]
    out: list[tuple[int, int]] = []
    if anchors.size < 2:
        return out
    for left, right in zip(anchors[:-1], anchors[1:]):
        if right - left >= 2:
            out.append((int(left), int(right)))
    return out


def _median_smooth_1d(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or x.size <= 2:
        return x.copy()
    half = window // 2
    out = np.empty_like(x, dtype=float)
    for i in range(x.size):
        lo = max(0, i - half)
        hi = min(x.size, i + half + 1)
        out[i] = float(np.median(x[lo:hi]))
    return out


def _label_active_q(
    z_seg_abs: np.ndarray,
    anchor_delta_abs_m: float,
    cfg_loss: dict,
) -> tuple[np.ndarray, np.ndarray, float]:
    if z_seg_abs.size < 2:
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=bool), 0.0
    deadband_m = float(cfg_loss.get("savca_change_deadband_m", 3.0))
    median_window = int(cfg_loss.get("savca_label_median_window", 5))
    active_min_anchor_delta_m = float(cfg_loss.get("savca_active_min_anchor_delta_m", 30.0))
    active_ratio_to_max = float(cfg_loss.get("savca_active_ratio_to_max", 0.25))
    active_min_abs_change_m = float(cfg_loss.get("savca_active_min_abs_change_m", 10.0))
    active_expand_steps = int(cfg_loss.get("savca_active_expand_steps", 1))

    z_s = _median_smooth_1d(np.asarray(z_seg_abs, dtype=float), median_window)
    diffs = np.abs(np.diff(z_s))
    if deadband_m > 0.0:
        diffs = np.where(diffs >= deadband_m, diffs, 0.0)
    active_mask = np.zeros_like(diffs, dtype=bool)
    diff_sum = float(diffs.sum())
    diff_max = float(diffs.max()) if diffs.size else 0.0
    if anchor_delta_abs_m >= active_min_anchor_delta_m and diff_sum > 1e-6 and diff_max > 1e-6:
        active_thr = max(active_min_abs_change_m, deadband_m, active_ratio_to_max * diff_max)
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
    q = active_diffs / (active_sum + 1e-6) if active_sum > 1e-6 else np.zeros_like(active_diffs)
    return q.astype(float), active_mask, active_sum


def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p + 1e-12)).sum())


def _normalized_entropy(p: np.ndarray) -> float:
    n = int(len(p))
    if n <= 1:
        return 0.0
    return float(_entropy(p) / math.log(n))


def _effective_support(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    s = float((p * p).sum())
    if s <= 1e-12:
        return 0.0
    return float(1.0 / s)


def _summary(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"count": 0}
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "p50": float(np.quantile(x, 0.5)),
        "p75": float(np.quantile(x, 0.75)),
        "p90": float(np.quantile(x, 0.9)),
        "p95": float(np.quantile(x, 0.95)),
        "max": float(np.max(x)),
    }


def _assign_bucket(v: float, edges: list[float]) -> str:
    for lo, hi in zip(edges[:-1], edges[1:]):
        if v >= lo and v < hi:
            if math.isinf(hi):
                return f"[{int(lo)},inf)"
            return f"[{int(lo)},{int(hi)})"
    lo = edges[-2]
    hi = edges[-1]
    return f"[{int(lo)},inf)" if math.isinf(hi) else f"[{int(lo)},{int(hi)})"


def _group_summary(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    if df.empty:
        return pd.DataFrame(rows)
    for keys, g in df.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        for vc in value_cols:
            vals = pd.to_numeric(g[vc], errors="coerce").dropna().to_numpy(dtype=float)
            st = _summary(vals)
            for k, v in st.items():
                row[f"{vc}_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def _sample_stage_stats(stage_train_ready: dict[str, pd.DataFrame], cfg_loss: dict) -> pd.DataFrame:
    rows: list[dict] = []
    delta_edges = [0.0, 30.0, 100.0, 300.0, float("inf")]
    for stage, df in stage_train_ready.items():
        for sample_id, g in df.groupby("sample_id", sort=False):
            z = pd.to_numeric(g["alt"], errors="coerce").to_numpy(dtype=float)
            obs = pd.to_numeric(g["obs_mask"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            valid = np.isfinite(z)
            intervals = _find_intervals(anchor_mask_1d=obs > 0.5, valid_mask_1d=valid)
            max_anchor_delta = 0.0
            max_gap_len = 0.0
            active_ratio_max = 0.0
            active_len_max = 0.0
            for left, right in intervals:
                interval = np.arange(left + 1, right + 1)
                anchor_delta = float(abs(z[right] - z[left]))
                max_anchor_delta = max(max_anchor_delta, anchor_delta)
                max_gap_len = max(max_gap_len, float(right - left))
                q, active_mask, _ = _label_active_q(z[left : right + 1], anchor_delta, cfg_loss)
                active_ratio_max = max(active_ratio_max, float(active_mask.mean()) if active_mask.size else 0.0)
                active_len_max = max(active_len_max, float(active_mask.sum()))
            rows.append(
                {
                    "stage": stage,
                    "sample_id": str(sample_id),
                    "stage_sample_key": f"{stage}::{sample_id}",
                    "max_anchor_delta_abs_m": max_anchor_delta,
                    "max_gap_len": max_gap_len,
                    "active_ratio_max": active_ratio_max,
                    "active_len_max": active_len_max,
                    "delta_bucket": _assign_bucket(max_anchor_delta, delta_edges),
                }
            )
    return pd.DataFrame(rows)


def _simulate_epoch_sampling(cfg: dict, stage_train_ready: dict[str, pd.DataFrame], sample_stats: pd.DataFrame) -> pd.DataFrame:
    curriculum_cfg = cfg["training"].get("curriculum", {})
    if not bool(curriculum_cfg.get("enabled", False)):
        return pd.DataFrame()
    schedule = curriculum_cfg.get("schedule", [])
    n_samples_total = int(curriculum_cfg.get("train_samples_per_epoch", 0))
    seed = int(cfg.get("seed", 42))
    stage_order = ["stage1", "stage2", "stage3"]
    rows: list[dict] = []
    stat_map = sample_stats.set_index("stage_sample_key").to_dict(orient="index")
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        weights = _pick_schedule_weights(epoch, schedule)
        w = np.array([float(weights.get(s, 0.0)) for s in stage_order], dtype=np.float64)
        if w.sum() <= 0:
            w = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        w = w / w.sum()
        counts = np.floor(w * int(n_samples_total)).astype(int)
        while counts.sum() < int(n_samples_total):
            counts[int(np.argmax(w - counts / max(1, n_samples_total)))] += 1
        rng = np.random.default_rng(seed + epoch * 1009)
        chosen_ids: list[str] = []
        chosen_stage: list[str] = []
        for idx, stage in enumerate(stage_order):
            sdf = stage_train_ready.get(stage, pd.DataFrame())
            if sdf.empty:
                continue
            ids = sdf["sample_id"].astype(str).drop_duplicates().to_numpy()
            picked = _sample_ids(ids, int(counts[idx]), rng)
            chosen_ids.extend([str(x) for x in picked.tolist()])
            chosen_stage.extend([stage] * len(picked))
        if not chosen_ids:
            continue
        chosen_rows = []
        for sid, stage in zip(chosen_ids, chosen_stage):
            x = dict(stat_map.get(f"{stage}::{sid}", {}))
            x["sample_id"] = sid
            x["stage"] = stage
            chosen_rows.append(x)
        cdf = pd.DataFrame(chosen_rows)
        rows.append(
            {
                "epoch": epoch,
                "sample_count": int(len(cdf)),
                "anchor_delta_mean": float(pd.to_numeric(cdf["max_anchor_delta_abs_m"], errors="coerce").mean()),
                "anchor_delta_p50": float(pd.to_numeric(cdf["max_anchor_delta_abs_m"], errors="coerce").quantile(0.5)),
                "anchor_delta_p90": float(pd.to_numeric(cdf["max_anchor_delta_abs_m"], errors="coerce").quantile(0.9)),
                "gap_len_mean": float(pd.to_numeric(cdf["max_gap_len"], errors="coerce").mean()),
                "gap_len_p90": float(pd.to_numeric(cdf["max_gap_len"], errors="coerce").quantile(0.9)),
                "active_ratio_mean": float(pd.to_numeric(cdf["active_ratio_max"], errors="coerce").mean()),
                "large_delta_long_gap_ratio_100_30": float(
                    (
                        (pd.to_numeric(cdf["max_anchor_delta_abs_m"], errors="coerce") >= 100.0)
                        & (pd.to_numeric(cdf["max_gap_len"], errors="coerce") >= 30.0)
                    ).mean()
                ),
                "large_delta_long_gap_ratio_300_30": float(
                    (
                        (pd.to_numeric(cdf["max_anchor_delta_abs_m"], errors="coerce") >= 300.0)
                        & (pd.to_numeric(cdf["max_gap_len"], errors="coerce") >= 30.0)
                    ).mean()
                ),
                "stage1_ratio": float((cdf["stage"].astype(str) == "stage1").mean()),
                "stage2_ratio": float((cdf["stage"].astype(str) == "stage2").mean()),
                "stage3_ratio": float((cdf["stage"].astype(str) == "stage3").mean()),
            }
        )
    return pd.DataFrame(rows)


def _tensor_grad_vector(model: torch.nn.Module, name_filter=None) -> torch.Tensor:
    parts = []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if name_filter is not None and (not name_filter(name)):
            continue
        parts.append(p.grad.detach().reshape(-1))
    if not parts:
        return torch.zeros(1)
    return torch.cat(parts)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    na = torch.linalg.norm(a)
    nb = torch.linalg.norm(b)
    if float(na) <= 1e-12 or float(nb) <= 1e-12:
        return float("nan")
    return float(torch.dot(a, b) / (na * nb))


def _build_model_batch(batch: dict, cfg: dict, target_norm_stats) -> tuple[dict, dict]:
    device = batch["obs_pos"].device
    obs_pos = batch["obs_pos"]
    obs_mask = batch["obs_mask"]
    seq_mask = batch["seq_mask"]
    target_pos = batch["target_pos"]
    target_model_raw, obs_model_raw, coord_ctx = prepare_model_coordinates(
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
    target_model = apply_alt_target_transform(target_model_raw, mode=alt_target_mode, clip_value=alt_target_clip)
    obs_model = apply_alt_target_transform(obs_model_raw, mode=alt_target_mode, clip_value=alt_target_clip)
    target_for_model = normalize_coords(target_model, target_norm_stats)
    obs_for_model = normalize_coords(obs_model, target_norm_stats)
    anchor_alt = build_anchor_alt_tracks(obs_pos=obs_pos, obs_mask=obs_mask, seq_mask=seq_mask)
    model_batch = dict(batch)
    model_batch["obs_pos_model"] = obs_for_model
    model_batch["target_pos_model"] = target_for_model
    model_batch["target_pos_model_raw"] = target_model_raw
    model_batch["coord_ctx"] = coord_ctx
    model_batch["anchor_alt"] = anchor_alt
    return model_batch, {"alt_target_mode": alt_target_mode, "alt_target_clip": alt_target_clip}


def _run_split_diagnostics(
    split_name: str,
    frame_raw: pd.DataFrame,
    frame_std: pd.DataFrame,
    cfg: dict,
    model: TrajectoryRecoveryModel,
    target_norm_stats,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dcfg = _build_dataset_cfg(cfg)
    ds = TrajectoryDataset(frame_std, dcfg)
    loader = DataLoader(
        ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"]["num_workers"]),
        collate_fn=trajectory_collate_fn,
    )
    rows_seg: list[dict] = []
    rows_sample: list[dict] = []
    raw_meta = frame_raw.groupby("sample_id", sort=False).agg(
        flight_id=("flight_id", "first"),
        audit_stage=("audit_stage", "first"),
    )
    loss_cfg = cfg.get("loss", {})
    delta_edges = [0.0, 30.0, 100.0, 300.0, float("inf")]
    gap_edges = [0.0, 15.0, 30.0, 60.0, float("inf")]
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            model_batch, meta = _build_model_batch(batch, cfg, target_norm_stats)
            out = model(
                obs_pos=model_batch["obs_pos_model"],
                obs_mask=batch["obs_mask"],
                dt_prev=batch["dt_prev"],
                dt_next=batch["dt_next"],
                exo=batch["exo"],
                vertical_exo=batch["vertical_exo"] if "vertical_exo" in batch else None,
                quality=batch["quality"],
                global_quality=batch["global_quality"],
                seq_mask=batch["seq_mask"],
                anchor_alt=model_batch["anchor_alt"],
                target_pos=None,
                teacher_forcing_ratio=0.0,
            )
            pred_model_t = denormalize_coords(out["pred_pos"], target_norm_stats)
            pred_model = invert_alt_target_transform(
                pred_model_t,
                mode=meta["alt_target_mode"],
                clip_value=meta["alt_target_clip"],
            )
            pred_latlon = restore_to_latlon(pred_model, seq_mask=batch["seq_mask"], ctx=model_batch["coord_ctx"]).detach().cpu().numpy()
            truth_latlon = batch["target_pos"].detach().cpu().numpy()
            obs_mask_np = batch["obs_mask"].detach().cpu().numpy()
            seq_mask_np = batch["seq_mask"].detach().cpu().numpy()
            p_all = out["savca_alloc_p"].detach().cpu().numpy()
            r_all = out["savca_state"].detach().cpu().numpy()
            valid_all = out["savca_alloc_valid"].detach().cpu().numpy()
            dt_prev_np = batch["dt_prev"].detach().cpu().numpy()
            dt_next_np = batch["dt_next"].detach().cpu().numpy()
            times_list = batch["times"]
            sids = batch["sample_id"]
            fids = batch["flight_id"]
            for i, sid in enumerate(sids):
                valid = seq_mask_np[i] > 0.5
                anchor = (obs_mask_np[i] > 0.5) & valid
                z_true = truth_latlon[i, :, 2]
                z_pred = pred_latlon[i, :, 2]
                alpha_all = dt_prev_np[i] / (dt_prev_np[i] + dt_next_np[i] + 1e-6)
                gap_mask = (obs_mask_np[i] <= 0.5) & valid
                z_a1 = np.zeros_like(z_true)
                z_a1[:] = np.nan
                sample_seg_rows = []
                intervals = _find_intervals(anchor.astype(float), valid)
                for seg_idx, (left, right) in enumerate(intervals):
                    interval = np.arange(left + 1, right + 1)
                    interior = np.arange(left + 1, right)
                    z_left = float(z_true[left])
                    z_right = float(z_true[right])
                    anchor_delta_abs = abs(z_right - z_left)
                    z_seg_truth = z_true[left : right + 1]
                    q, active_mask, active_sum = _label_active_q(z_seg_truth, anchor_delta_abs, loss_cfg)
                    p = p_all[i, interval].astype(float)
                    r = r_all[i, interval].astype(float)
                    p_valid = valid_all[i, interval].astype(float)
                    p = np.where(np.isfinite(p), p, 0.0)
                    if p.sum() > 1e-8:
                        p = p / (p.sum() + 1e-8)
                    tau = alpha_all[interval].astype(float)
                    c_p = float((tau * p).sum()) if p.size else float("nan")
                    c_q = float((tau * q).sum()) if q.size else float("nan")
                    p_max = float(p.max()) if p.size else 0.0
                    q_max = float(q.max()) if q.size else 0.0
                    p_ent = _entropy(p)
                    p_ent_norm = _normalized_entropy(p)
                    q_ent = _entropy(q)
                    q_ent_norm = _normalized_entropy(q)
                    p_eff = _effective_support(p)
                    q_eff = _effective_support(q)
                    p_halfmax_len = int((p >= 0.5 * p_max).sum()) if p.size and p_max > 0 else 0
                    q_halfmax_len = int((q >= 0.5 * q_max).sum()) if q.size and q_max > 0 else 0
                    active_vals = r[active_mask] if active_mask.size else np.array([], dtype=float)
                    non_active_vals = r[~active_mask] if active_mask.size else np.array([], dtype=float)
                    state_gap = float(np.mean(active_vals) - np.mean(non_active_vals)) if active_vals.size and non_active_vals.size else float("nan")
                    savca_gap_pred = z_pred[interior]
                    a1_gap_pred = z_left + alpha_all[interior] * (z_right - z_left)
                    z_a1[interior] = a1_gap_pred
                    truth_gap = z_true[interior]
                    if truth_gap.size == 0:
                        continue
                    savca_err = savca_gap_pred - truth_gap
                    a1_err = a1_gap_pred - truth_gap
                    row = {
                        "split": split_name,
                        "sample_id": str(sid),
                        "flight_id": str(fids[i]),
                        "audit_stage": str(raw_meta.loc[str(sid), "audit_stage"]) if str(sid) in raw_meta.index else "",
                        "segment_index": int(seg_idx),
                        "left_idx": int(left),
                        "right_idx": int(right),
                        "left_ts": str(times_list[i][left]),
                        "right_ts": str(times_list[i][right]),
                        "gap_len": int(right - left),
                        "gap_bucket": _assign_bucket(float(right - left), gap_edges),
                        "anchor_delta_abs_m": float(anchor_delta_abs),
                        "delta_bucket": _assign_bucket(float(anchor_delta_abs), delta_edges),
                        "active_len": int(active_mask.sum()),
                        "active_ratio": float(active_mask.mean()) if active_mask.size else 0.0,
                        "p_entropy": p_ent,
                        "p_entropy_norm": p_ent_norm,
                        "q_entropy": q_ent,
                        "q_entropy_norm": q_ent_norm,
                        "p_max": p_max,
                        "q_max": q_max,
                        "p_effective_support": p_eff,
                        "q_effective_support": q_eff,
                        "p_halfmax_len": p_halfmax_len,
                        "q_halfmax_len": q_halfmax_len,
                        "c_p": c_p,
                        "c_q": c_q,
                        "center_shift_abs": abs(c_p - c_q) if np.isfinite(c_p) and np.isfinite(c_q) else float("nan"),
                        "peak_shift_abs_idx": abs(int(np.argmax(p)) - int(np.argmax(q))) if p.size and q.size else float("nan"),
                        "state_active_mean": float(np.mean(active_vals)) if active_vals.size else float("nan"),
                        "state_active_max": float(np.max(active_vals)) if active_vals.size else float("nan"),
                        "state_active_var": float(np.var(active_vals)) if active_vals.size else float("nan"),
                        "state_nonactive_mean": float(np.mean(non_active_vals)) if non_active_vals.size else float("nan"),
                        "state_nonactive_max": float(np.max(non_active_vals)) if non_active_vals.size else float("nan"),
                        "state_nonactive_var": float(np.var(non_active_vals)) if non_active_vals.size else float("nan"),
                        "state_gap_mean": state_gap,
                        "savca_gap_alt_rmse": float(np.sqrt(np.mean(np.square(savca_err)))),
                        "savca_gap_alt_mae": float(np.mean(np.abs(savca_err))),
                        "a1_gap_alt_rmse": float(np.sqrt(np.mean(np.square(a1_err)))),
                        "a1_gap_alt_mae": float(np.mean(np.abs(a1_err))),
                        "savca_minus_a1_gap_rmse": float(np.sqrt(np.mean(np.square(savca_err))) - np.sqrt(np.mean(np.square(a1_err)))),
                        "alloc_valid_mean": float(np.mean(p_valid)) if p_valid.size else 0.0,
                        "transition_near_left": int(np.isfinite(c_q) and c_q < 0.25),
                        "transition_near_right": int(np.isfinite(c_q) and c_q > 0.75),
                    }
                    rows_seg.append(row)
                    sample_seg_rows.append(row)
                if sample_seg_rows:
                    sdf = pd.DataFrame(sample_seg_rows)
                    rows_sample.append(
                        {
                            "split": split_name,
                            "sample_id": str(sid),
                            "flight_id": str(fids[i]),
                            "audit_stage": str(raw_meta.loc[str(sid), "audit_stage"]) if str(sid) in raw_meta.index else "",
                            "segment_count": int(len(sdf)),
                            "max_gap_len": float(sdf["gap_len"].max()),
                            "max_anchor_delta_abs_m": float(sdf["anchor_delta_abs_m"].max()),
                            "mean_active_ratio": float(sdf["active_ratio"].mean()),
                            "mean_center_shift_abs": float(sdf["center_shift_abs"].mean()),
                            "max_center_shift_abs": float(sdf["center_shift_abs"].max()),
                            "mean_state_gap_mean": float(sdf["state_gap_mean"].mean()),
                            "savca_gap_alt_rmse": float(np.sqrt(np.mean(np.square(np.concatenate([
                                (z_pred[np.arange(r["left_idx"] + 1, r["right_idx"])] - z_true[np.arange(r["left_idx"] + 1, r["right_idx"])])
                                for _, r in sdf.iterrows()
                            ]))))),
                            "a1_gap_alt_rmse": float(np.sqrt(np.mean(np.square(np.concatenate([
                                ((z_true[int(r["left_idx"])] + alpha_all[np.arange(int(r["left_idx"]) + 1, int(r["right_idx"]))] * (z_true[int(r["right_idx"])] - z_true[int(r["left_idx"])])) - z_true[np.arange(int(r["left_idx"]) + 1, int(r["right_idx"]))])
                                for _, r in sdf.iterrows()
                            ]))))),
                        }
                    )
    sample_df = pd.DataFrame(rows_sample)
    if not sample_df.empty:
        sample_df["savca_minus_a1_gap_rmse"] = sample_df["savca_gap_alt_rmse"] - sample_df["a1_gap_alt_rmse"]
    return pd.DataFrame(rows_seg), sample_df


def _compute_gradient_diagnostics(
    cfg: dict,
    frame_std_train: pd.DataFrame,
    checkpoint_path: Path,
    target_norm_stats,
    grad_batches: int,
) -> pd.DataFrame:
    device = torch.device("cpu")
    dcfg = _build_dataset_cfg(cfg)
    ds = TrajectoryDataset(frame_std_train, dcfg)
    loader = DataLoader(
        ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=trajectory_collate_fn,
    )
    model = _build_model(cfg, device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.train()
    criterion = _build_criterion(cfg)
    rows: list[dict] = []
    name_filter = lambda n: n.startswith("forward_net") or n.startswith("backward_net") or n.startswith("fusion") or n.startswith("hidden_fusion")
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= int(grad_batches):
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        model_batch, meta = _build_model_batch(batch, cfg, target_norm_stats)
        out = model(
            obs_pos=model_batch["obs_pos_model"],
            obs_mask=batch["obs_mask"],
            dt_prev=batch["dt_prev"],
            dt_next=batch["dt_next"],
            exo=batch["exo"],
            vertical_exo=batch["vertical_exo"] if "vertical_exo" in batch else None,
            quality=batch["quality"],
            global_quality=batch["global_quality"],
            seq_mask=batch["seq_mask"],
            anchor_alt=model_batch["anchor_alt"],
            target_pos=model_batch["target_pos_model"],
            teacher_forcing_ratio=0.0,
        )
        loss_dict = criterion(
            pred_pos=out["pred_pos"],
            target_pos=model_batch["target_pos_model"],
            obs_mask=batch["obs_mask"],
            seq_mask=batch["seq_mask"],
            exo=batch["exo"],
            quality=batch["quality"],
            fusion_weights=out.get("fusion_weights"),
            dt_prev=batch["dt_prev"],
            dt_next=batch["dt_next"],
            logvar=out.get("logvar", out["logvar_f"]),
            long_gap_threshold=int(cfg["training"].get("long_gap_threshold", 20)),
            alt_base=out.get("alt_base"),
            residual_bound=out.get("residual_bound"),
            delta_alt_pred_norm=out.get("delta_alt_pred_norm"),
            alt_gate=out.get("alt_gate"),
            teacher_scale=batch["teacher_scale"],
            risk_flag=batch["risk_flag"],
            risk_flag_teacher=batch["risk_flag_teacher"],
            segment_bucket=batch["segment_bucket"],
            anchor_pattern=batch["anchor_pattern"],
            edge_weight=batch["edge_weight"],
            pred_pos_main=out.get("pred_pos_main"),
            left_boundary_alt=batch["left_boundary_alt"],
            right_boundary_alt=batch["right_boundary_alt"],
            mu_f=out.get("mu_f"),
            mu_b=out.get("mu_b"),
            savca_alloc_p=out.get("savca_alloc_p"),
            savca_state=out.get("savca_state"),
            savca_alloc_valid=out.get("savca_alloc_valid"),
            savca_target_alt_abs=batch["target_pos"][..., 2],
        )
        rec = loss_dict["loss_pos"] + float(cfg["loss"].get("lambda_smooth", 0.0)) * loss_dict["loss_smooth"]
        alloc = float(cfg["loss"].get("lambda_savca_alloc", 0.0)) * loss_dict["savca_alloc_loss"]
        state_l = float(cfg["loss"].get("lambda_savca_state", 0.0)) * loss_dict["savca_state_loss"]
        smooth_l = float(cfg["loss"].get("lambda_savca_smooth", 0.0)) * loss_dict["savca_smooth_loss"]

        def grad_vec(loss_scalar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            model.zero_grad(set_to_none=True)
            loss_scalar.backward(retain_graph=True)
            return _tensor_grad_vector(model), _tensor_grad_vector(model, name_filter=name_filter)

        g_rec_all, g_rec_shared = grad_vec(rec)
        g_alloc_all, g_alloc_shared = grad_vec(alloc)
        g_state_all, g_state_shared = grad_vec(state_l)
        g_smooth_all, g_smooth_shared = grad_vec(smooth_l)
        rows.append(
            {
                "batch_idx": batch_idx,
                "L_rec": float(rec.detach().cpu()),
                "L_alloc_weighted": float(alloc.detach().cpu()),
                "L_state_weighted": float(state_l.detach().cpu()),
                "L_savca_smooth_weighted": float(smooth_l.detach().cpu()),
                "grad_norm_rec_all": float(torch.linalg.norm(g_rec_all)),
                "grad_norm_alloc_all": float(torch.linalg.norm(g_alloc_all)),
                "grad_norm_state_all": float(torch.linalg.norm(g_state_all)),
                "grad_norm_smooth_all": float(torch.linalg.norm(g_smooth_all)),
                "grad_norm_rec_shared": float(torch.linalg.norm(g_rec_shared)),
                "grad_norm_alloc_shared": float(torch.linalg.norm(g_alloc_shared)),
                "grad_norm_state_shared": float(torch.linalg.norm(g_state_shared)),
                "grad_norm_smooth_shared": float(torch.linalg.norm(g_smooth_shared)),
                "cos_rec_alloc_shared": _cosine(g_rec_shared, g_alloc_shared),
                "cos_rec_state_shared": _cosine(g_rec_shared, g_state_shared),
                "cos_rec_smooth_shared": _cosine(g_rec_shared, g_smooth_shared),
                "savca_alloc_loss_raw": float(loss_dict["savca_alloc_loss"].detach().cpu()),
                "savca_state_loss_raw": float(loss_dict["savca_state_loss"].detach().cpu()),
                "savca_smooth_loss_raw": float(loss_dict["savca_smooth_loss"].detach().cpu()),
            }
        )
        model.zero_grad(set_to_none=True)
    return pd.DataFrame(rows)


def _plot_case(case_row: pd.Series, seg_df: pd.DataFrame, case_dir: Path, split_frames: dict[str, pd.DataFrame]) -> None:
    split_name = str(case_row["split"])
    sid = str(case_row["sample_id"])
    seg_idx = int(case_row["segment_index"])
    frame = split_frames[split_name]
    g = frame[frame["sample_id"].astype(str).eq(sid)].sort_values("minute_ts").reset_index(drop=True)
    if g.empty:
        return
    row = seg_df[
        (seg_df["split"].astype(str) == split_name)
        & (seg_df["sample_id"].astype(str) == sid)
        & (pd.to_numeric(seg_df["segment_index"], errors="coerce") == seg_idx)
    ]
    if row.empty:
        return
    r = row.iloc[0]
    left = int(r["left_idx"])
    right = int(r["right_idx"])
    interval = np.arange(left + 1, right + 1)
    interior = np.arange(left + 1, right)
    z_true = pd.to_numeric(g["alt"], errors="coerce").to_numpy(dtype=float)
    obs = pd.to_numeric(g["obs_mask"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    dt_prev = pd.to_numeric(g["dt_prev"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    dt_next = pd.to_numeric(g["dt_next"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    alpha = dt_prev / (dt_prev + dt_next + 1e-6)
    pred_csv = case_dir.parent / "segment_predictions_full.csv"
    if not pred_csv.exists():
        return
    pred_df = pd.read_csv(pred_csv)
    pg = pred_df[
        (pred_df["split"].astype(str) == split_name)
        & (pred_df["sample_id"].astype(str) == sid)
        & (pd.to_numeric(pred_df["segment_index"], errors="coerce") == seg_idx)
    ].sort_values("minute_ts")
    if pg.empty:
        return
    z_pred = pg["savca_pred_alt_m"].to_numpy(dtype=float)
    z_a1 = pg["a1_alt_m"].to_numpy(dtype=float)
    p = pg["p_t"].to_numpy(dtype=float)
    q = pg["q_t"].to_numpy(dtype=float)
    rr = pg["r_t"].to_numpy(dtype=float)
    tau = pg["tau_t"].to_numpy(dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    x = np.arange(len(z_true))
    axes[0].plot(x, z_true, color="black", lw=1.8, label="ADS-B true")
    axes[0].plot(interior, z_a1[:-1], color="#1f77b4", lw=1.6, label="A1")
    axes[0].plot(interior, z_pred[:-1], color="#d62728", lw=1.8, label="SAVCA")
    anchor_idx = np.where(obs > 0.5)[0]
    axes[0].scatter(anchor_idx, z_true[anchor_idx], marker="*", s=90, color="black", label="ADS-C anchors")
    axes[0].axvline(left, color="#777777", ls="--", lw=1.0)
    axes[0].axvline(right, color="#777777", ls="--", lw=1.0)
    axes[0].set_ylabel("Altitude (m)")
    axes[0].legend(loc="best", fontsize=9)
    axes[0].set_title(f"{sid} {split_name} seg{seg_idx} | c_p={r['c_p']:.3f}, c_q={r['c_q']:.3f}")

    axes[1].plot(tau, p, color="#d62728", lw=1.8, label="p_t")
    axes[1].plot(tau, q, color="#1f77b4", lw=1.8, label="q_t")
    axes[1].axvline(float(r["c_p"]), color="#d62728", ls="--", lw=1.0)
    axes[1].axvline(float(r["c_q"]), color="#1f77b4", ls="--", lw=1.0)
    axes[1].set_ylabel("Allocation")
    axes[1].legend(loc="best", fontsize=9)

    axes[2].plot(tau, rr, color="#2ca02c", lw=1.8, label="r_t")
    axes[2].plot(tau, (q > 0).astype(float), color="#666666", lw=1.2, ls="--", label="active(q)")
    axes[2].set_xlabel("gap position ratio")
    axes[2].set_ylabel("State")
    axes[2].legend(loc="best", fontsize=9)
    fig.savefig(case_dir / f"{sid}_seg{seg_idx}_savca_case.png", dpi=180)
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(cfg["outputs"]["run_dir"])
    checkpoint_path = Path(args.checkpoint)
    model_splits = [x.strip() for x in str(args.model_splits).split(",") if x.strip()]

    splits_raw, stage_train_frames_raw = _build_training_like_splits(cfg)
    splits_ready_raw, splits_ready_std, stage_train_ready = _apply_training_ready_preprocess(
        cfg=cfg,
        splits_raw=splits_raw,
        stage_train_frames_raw=stage_train_frames_raw,
        run_dir=run_dir,
    )
    device = torch.device("cpu")
    model = _build_model(cfg, device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    target_norm_stats = None
    if bool(cfg["training"].get("target_norm", {}).get("enabled", False)):
        target_norm_stats = load_target_stats(run_dir / "target_model_scaler.json")

    split_seg_parts = []
    split_sample_parts = []
    for split_name in model_splits:
        if split_name not in splits_ready_raw:
            continue
        seg_df, sample_df = _run_split_diagnostics(
            split_name=split_name,
            frame_raw=splits_ready_raw[split_name],
            frame_std=splits_ready_std[split_name],
            cfg=cfg,
            model=model,
            target_norm_stats=target_norm_stats,
            device=device,
        )
        seg_df.to_csv(out_dir / f"{split_name}_segment_diagnostics.csv", index=False)
        sample_df.to_csv(out_dir / f"{split_name}_sample_diagnostics.csv", index=False)
        split_seg_parts.append(seg_df)
        split_sample_parts.append(sample_df)

    seg_all = pd.concat(split_seg_parts, ignore_index=True) if split_seg_parts else pd.DataFrame()
    sample_all = pd.concat(split_sample_parts, ignore_index=True) if split_sample_parts else pd.DataFrame()
    seg_all.to_csv(out_dir / "savca_segment_diagnostics_all.csv", index=False)
    sample_all.to_csv(out_dir / "savca_sample_diagnostics_all.csv", index=False)

    # Train/val/test difficulty distributions from ground-truth-derived labels, independent of model forward.
    dist_rows = []
    for split_name, frame in splits_ready_raw.items():
        for sample_id, g in frame.groupby("sample_id", sort=False):
            z = pd.to_numeric(g["alt"], errors="coerce").to_numpy(dtype=float)
            obs = pd.to_numeric(g["obs_mask"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            valid = np.isfinite(z)
            for left, right in _find_intervals(obs > 0.5, valid):
                q, active_mask, _ = _label_active_q(z[left : right + 1], abs(z[right] - z[left]), cfg.get("loss", {}))
                dist_rows.append(
                    {
                        "split": split_name,
                        "sample_id": str(sample_id),
                        "audit_stage": str(g["audit_stage"].iloc[0]) if "audit_stage" in g.columns else "",
                        "gap_len": int(right - left),
                        "anchor_delta_abs_m": float(abs(z[right] - z[left])),
                        "active_len": int(active_mask.sum()),
                        "active_ratio": float(active_mask.mean()) if active_mask.size else 0.0,
                        "q_entropy_norm": _normalized_entropy(q),
                        "q_max": float(q.max()) if q.size else 0.0,
                    }
                )
    dist_df = pd.DataFrame(dist_rows)
    dist_df.to_csv(out_dir / "ground_truth_difficulty_segments.csv", index=False)

    # Key grouped summaries
    _group_summary(
        seg_all,
        group_cols=["split"],
        value_cols=["center_shift_abs", "p_entropy_norm", "p_max", "active_ratio", "state_gap_mean", "savca_minus_a1_gap_rmse"],
    ).to_csv(out_dir / "summary_by_split.csv", index=False)
    _group_summary(
        seg_all,
        group_cols=["split", "gap_bucket"],
        value_cols=["center_shift_abs", "p_entropy_norm", "p_max", "savca_gap_alt_rmse", "a1_gap_alt_rmse", "savca_minus_a1_gap_rmse"],
    ).to_csv(out_dir / "summary_by_gap_bucket.csv", index=False)
    _group_summary(
        seg_all,
        group_cols=["split", "delta_bucket"],
        value_cols=["center_shift_abs", "p_entropy_norm", "p_max", "savca_gap_alt_rmse", "a1_gap_alt_rmse", "savca_minus_a1_gap_rmse"],
    ).to_csv(out_dir / "summary_by_delta_bucket.csv", index=False)
    _group_summary(
        seg_all,
        group_cols=["split", "audit_stage"],
        value_cols=["center_shift_abs", "p_entropy_norm", "p_max", "savca_gap_alt_rmse", "a1_gap_alt_rmse", "savca_minus_a1_gap_rmse"],
    ).to_csv(out_dir / "summary_by_stage.csv", index=False)

    # Dataset difficulty distributions
    dataset_dist = _group_summary(
        dist_df,
        group_cols=["split"],
        value_cols=["anchor_delta_abs_m", "gap_len", "active_len", "active_ratio", "q_entropy_norm", "q_max"],
    )
    dataset_dist.to_csv(out_dir / "dataset_difficulty_by_split.csv", index=False)

    # Actual sampled epoch distribution
    if stage_train_ready:
        stage_stats = _sample_stage_stats(stage_train_ready, cfg.get("loss", {}))
        stage_stats.to_csv(out_dir / "train_stage_sample_stats.csv", index=False)
        epoch_sampling = _simulate_epoch_sampling(cfg, stage_train_ready, stage_stats)
        epoch_sampling.to_csv(out_dir / "epoch_actual_sampling_summary.csv", index=False)

    # Gradient diagnostics
    grad_df = _compute_gradient_diagnostics(
        cfg=cfg,
        frame_std_train=splits_ready_std["train"],
        checkpoint_path=checkpoint_path,
        target_norm_stats=target_norm_stats,
        grad_batches=int(args.grad_batches),
    )
    grad_df.to_csv(out_dir / "gradient_component_diagnostics.csv", index=False)

    # Predicted point tables for selected cases
    pred_rows: list[dict] = []
    if not seg_all.empty:
        top_fail = seg_all.sort_values("savca_minus_a1_gap_rmse", ascending=False).head(int(args.case_count)).copy()
        top_fail.to_csv(out_dir / "top_savca_fail_segments.csv", index=False)
        cases_dir = out_dir / "cases"
        cases_dir.mkdir(parents=True, exist_ok=True)
        # Build detailed per-step tables from diagnostics tables already computed.
        # Re-run the relevant segments only from raw frames.
        for _, row in top_fail.iterrows():
            split_name = str(row["split"])
            sid = str(row["sample_id"])
            seg_idx = int(row["segment_index"])
            g = splits_ready_raw[split_name]
            sg = g[g["sample_id"].astype(str).eq(sid)].sort_values("minute_ts").reset_index(drop=True)
            if sg.empty:
                continue
            z_true = pd.to_numeric(sg["alt"], errors="coerce").to_numpy(dtype=float)
            obs = pd.to_numeric(sg["obs_mask"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            dt_prev = pd.to_numeric(sg["dt_prev"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            dt_next = pd.to_numeric(sg["dt_next"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            alpha = dt_prev / (dt_prev + dt_next + 1e-6)
            left = int(row["left_idx"])
            right = int(row["right_idx"])
            interval = np.arange(left + 1, right + 1)
            interior = np.arange(left + 1, right)
            # use summary row stats only; detailed prediction table will be rebuilt from recorded segment diagnostics file not model rerun.
            q, active_mask, _ = _label_active_q(z_true[left : right + 1], abs(z_true[right] - z_true[left]), cfg.get("loss", {}))
            # placeholder arrays will be filled by second pass using current model directly on this sample below.
            case_dir = cases_dir / f"{split_name}_{sid}_seg{seg_idx}"
            case_dir.mkdir(parents=True, exist_ok=True)
            # Build one-sample batch through dataset pipeline.
            dcfg = _build_dataset_cfg(cfg)
            sg_std = apply_standardizer(sg, load_standardizer(run_dir / "feature_standardizer.json") or {})
            sds = TrajectoryDataset(sg_std, dcfg)
            if len(sds) == 0:
                continue
            one = trajectory_collate_fn([sds[0]])
            one = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in one.items()}
            model_batch, meta = _build_model_batch(one, cfg, target_norm_stats)
            with torch.no_grad():
                out = model(
                    obs_pos=model_batch["obs_pos_model"],
                    obs_mask=one["obs_mask"],
                    dt_prev=one["dt_prev"],
                    dt_next=one["dt_next"],
                    exo=one["exo"],
                    vertical_exo=one["vertical_exo"] if "vertical_exo" in one else None,
                    quality=one["quality"],
                    global_quality=one["global_quality"],
                    seq_mask=one["seq_mask"],
                    anchor_alt=model_batch["anchor_alt"],
                    target_pos=None,
                    teacher_forcing_ratio=0.0,
                )
                pred_model_t = denormalize_coords(out["pred_pos"], target_norm_stats)
                pred_model = invert_alt_target_transform(pred_model_t, mode=meta["alt_target_mode"], clip_value=meta["alt_target_clip"])
                pred_latlon = restore_to_latlon(pred_model, seq_mask=one["seq_mask"], ctx=model_batch["coord_ctx"]).detach().cpu().numpy()[0]
                p = out["savca_alloc_p"].detach().cpu().numpy()[0, interval]
                r_t = out["savca_state"].detach().cpu().numpy()[0, interval]
            for k, t in enumerate(interval):
                pred_rows.append(
                    {
                        "split": split_name,
                        "sample_id": sid,
                        "segment_index": seg_idx,
                        "minute_ts": str(sg["minute_ts"].iloc[t]),
                        "tau_t": float(alpha[t]),
                        "truth_alt_m": float(z_true[t]),
                        "savca_pred_alt_m": float(pred_latlon[t, 2]),
                        "a1_alt_m": float(z_true[left] + alpha[t] * (z_true[right] - z_true[left])),
                        "p_t": float(p[k]) if k < len(p) else float("nan"),
                        "q_t": float(q[k]) if k < len(q) else float("nan"),
                        "r_t": float(r_t[k]) if k < len(r_t) else float("nan"),
                        "active_t": int(active_mask[k]) if k < len(active_mask) else 0,
                        "is_anchor": int(obs[t] > 0.5),
                    }
                )
            _plot_case(row, seg_all, case_dir, splits_ready_raw)
        if pred_rows:
            pd.DataFrame(pred_rows).to_csv(out_dir / "segment_predictions_full.csv", index=False)

    summary = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "out_dir": str(out_dir),
        "segment_count": int(len(seg_all)),
        "sample_count": int(len(sample_all)),
        "best_trained_run_dir": str(run_dir),
    }
    (out_dir / "diagnosis_metadata.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] wrote SAVCA diagnosis to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
