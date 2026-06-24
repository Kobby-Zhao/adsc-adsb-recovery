from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.datasets import DatasetConfig, TrajectoryDataset, trajectory_collate_fn
from src.models import TrajectoryRecoveryModel
from src.preprocessing.standardize import apply_standardizer, load_standardizer
from src.training.alt_target import apply_alt_target_transform
from src.training.altitude_governance import (
    add_anchor_alt_features,
    add_vertical_v2_features,
    apply_alt_label_governance,
)
from src.training.coords import build_anchor_alt_tracks, prepare_model_coordinates
from src.training.target_norm import load_target_stats, normalize_coords
from src.training.utils import load_config, split_by_flight_id


def _anchor_gate_filter(frame: pd.DataFrame) -> pd.DataFrame:
    obs = pd.to_numeric(frame["obs_mask"], errors="coerce").fillna(0.0)
    by = frame.assign(_obs_anchor=(obs > 0.5)).groupby("sample_id")["_obs_anchor"].any().astype(bool)
    keep_ids = set(by[by].index.astype(str).tolist())
    return frame[frame["sample_id"].astype(str).isin(keep_ids)].copy()


def _build_val_frame(cfg: dict, run_dir: Path, seed: int) -> pd.DataFrame:
    stage_paths = cfg["training"]["curriculum"]["stage_paths"]
    stage_frames = {k: pd.read_parquet(v) for k, v in stage_paths.items()}
    ref = stage_frames["stage1"]
    split_cfg = cfg["data"]["split"]
    ref_splits = split_by_flight_id(
        df=ref,
        flight_id_col=cfg["data"]["flight_id_col"],
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        seed=seed,
    )
    train_ids = set(ref_splits["train"][cfg["data"]["flight_id_col"]].astype(str).unique().tolist())
    val_ids = set(ref_splits["val"][cfg["data"]["flight_id_col"]].astype(str).unique().tolist())

    val_stage = cfg["training"]["curriculum"].get("val_stage", "stage3")
    val = stage_frames[val_stage]
    val = val[val[cfg["data"]["flight_id_col"]].astype(str).isin(val_ids)].copy()

    train_parts = []
    for _, stage_df in stage_frames.items():
        train_parts.append(stage_df[stage_df[cfg["data"]["flight_id_col"]].astype(str).isin(train_ids)].copy())
    train = pd.concat(train_parts, ignore_index=True)

    train = add_vertical_v2_features(add_anchor_alt_features(train))
    val = add_vertical_v2_features(add_anchor_alt_features(val))
    train = _anchor_gate_filter(train)
    val = _anchor_gate_filter(val)
    train, _ = apply_alt_label_governance(train, cfg["training"].get("alt_label_governance", {}), out_dir=run_dir)
    scaler = load_standardizer(run_dir / "feature_standardizer.json")
    return apply_standardizer(val, scaler)


def _build_model(cfg: dict) -> TrajectoryRecoveryModel:
    return TrajectoryRecoveryModel(
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
        savca_beta_enabled=bool(cfg["model"].get("savca_beta_enabled", False)),
        savca_beta_hidden_size=int(cfg["model"].get("savca_beta_hidden_size", 32)),
        savca_beta_init_bias=float(cfg["model"].get("savca_beta_init_bias", -2.0)),
        savca_beta_default_max=float(cfg["model"].get("savca_beta_default_max", 1.0)),
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
    )


def _beta_max_at_epoch(cfg: dict, epoch: int) -> float | None:
    sched = cfg["training"].get("savca_beta_schedule", {})
    if not sched.get("enabled", False):
        return None
    epochs = sched.get("epochs", [])
    values = sched.get("values", [])
    if not epochs or not values:
        return None
    if epoch <= epochs[0]:
        return float(values[0])
    current = float(values[-1])
    for i in range(len(epochs) - 1):
        e0, e1 = int(epochs[i]), int(epochs[i + 1])
        v0, v1 = float(values[i]), float(values[i + 1])
        if epoch <= e1:
            ratio = 0.0 if e1 == e0 else (epoch - e0) / float(e1 - e0)
            current = v0 + ratio * (v1 - v0)
            break
    return current


def _find_intervals(anchor_mask: np.ndarray, valid_mask: np.ndarray) -> list[tuple[int, int]]:
    anchors = np.where((anchor_mask > 0.5) & valid_mask)[0]
    return [(int(l), int(r)) for l, r in zip(anchors[:-1], anchors[1:]) if r - l >= 2]


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


def _label_active_q(cfg_loss: dict, z_seg_abs: np.ndarray, anchor_delta_abs_m: float) -> tuple[np.ndarray, np.ndarray]:
    deadband = float(cfg_loss.get("savca_change_deadband_m", 3.0))
    win = int(cfg_loss.get("savca_label_median_window", 5))
    min_anchor = float(cfg_loss.get("savca_active_min_anchor_delta_m", 30.0))
    ratio = float(cfg_loss.get("savca_active_ratio_to_max", 0.25))
    min_abs = float(cfg_loss.get("savca_active_min_abs_change_m", 10.0))
    expand = int(cfg_loss.get("savca_active_expand_steps", 1))
    z_s = _median_smooth_1d(np.asarray(z_seg_abs, dtype=float), win)
    diffs = np.abs(np.diff(z_s))
    if deadband > 0:
        diffs = np.where(diffs >= deadband, diffs, 0.0)
    active = np.zeros_like(diffs, dtype=bool)
    diff_sum = float(diffs.sum())
    diff_max = float(diffs.max()) if diffs.size else 0.0
    if anchor_delta_abs_m >= min_anchor and diff_sum > 1e-6 and diff_max > 1e-6:
        thr = max(min_abs, deadband, ratio * diff_max)
        active = diffs >= thr
        if expand > 0 and active.any():
            mask = active.copy()
            for _ in range(expand):
                prev = np.concatenate([mask[:1], mask[:-1]])
                nxt = np.concatenate([mask[1:], mask[-1:]])
                mask = mask | prev | nxt
            active = mask
    active_diffs = np.where(active, diffs, 0.0)
    denom = float(active_diffs.sum())
    q = active_diffs / (denom + 1e-6) if denom > 1e-6 else np.zeros_like(active_diffs)
    return q.astype(float), active


def _norm_entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    n = len(p)
    p = p[p > 0]
    if n <= 1 or p.size == 0:
        return 0.0
    return float((-(p * np.log(p + 1e-12)).sum()) / math.log(n))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose A1+SAVCA beta fusion behavior.")
    parser.add_argument(
        "--config",
        default="outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_savca_beta_v1/configs/savca_beta.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_savca_beta_v1/savca_beta/best.pt",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/runs/0524/savca_beta_diagnosis",
    )
    parser.add_argument(
        "--epoch-for-beta-schedule",
        type=int,
        default=10,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(str(ROOT / args.config))
    ckpt_path = ROOT / args.checkpoint
    run_dir = ckpt_path.parent
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("seed", 42))

    val_frame = _build_val_frame(cfg, run_dir, seed)
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
    )
    loader = DataLoader(
        TrajectoryDataset(val_frame, dcfg),
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=trajectory_collate_fn,
    )

    model = _build_model(cfg).to("cpu")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    beta_max = _beta_max_at_epoch(cfg, epoch=int(args.epoch_for_beta_schedule))
    if beta_max is not None:
        model.set_runtime_savca_beta_max(beta_max)

    coord_mode = str(cfg["model"].get("coord_mode", "latlon"))
    u_relative_anchor = bool(cfg["model"].get("u_relative_anchor", False))
    en_relative_anchor = bool(cfg["model"].get("en_relative_anchor", True))
    en_incremental = bool(cfg["model"].get("en_incremental", False))
    alt_target_mode = str(cfg["training"].get("alt_target_robust", {}).get("mode", "none")).lower()
    alt_target_clip = float(cfg["training"].get("alt_target_robust", {}).get("clip_value", 3000.0))
    target_norm_stats = load_target_stats(run_dir / "target_norm_stats.json")

    rows: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            obs_pos = batch["obs_pos"]
            obs_mask = batch["obs_mask"]
            dt_prev = batch["dt_prev"]
            dt_next = batch["dt_next"]
            exo = batch["exo"]
            quality = batch["quality"]
            global_quality = batch["global_quality"]
            target_pos = batch["target_pos"]
            seq_mask = batch["seq_mask"]
            risk_flag = batch.get("risk_flag")
            risk_flag_teacher = batch.get("risk_flag_teacher")
            teacher_scale = batch.get("teacher_scale")
            segment_bucket = batch.get("segment_bucket")
            edge_weight = batch.get("edge_weight")
            residual_rmax_m = batch.get("residual_rmax_m")
            residual_rmax_ft = batch.get("residual_rmax_ft")
            gate_bias = batch.get("gate_bias")

            target_model_raw, obs_model_raw, _ = prepare_model_coordinates(
                target_pos=target_pos,
                obs_pos=obs_pos,
                obs_mask=obs_mask,
                seq_mask=seq_mask,
                mode=coord_mode,
                u_relative_anchor=u_relative_anchor,
                en_relative_anchor=en_relative_anchor,
                en_incremental=en_incremental,
            )
            target_model = apply_alt_target_transform(target_model_raw, mode=alt_target_mode, clip_value=alt_target_clip)
            obs_model = apply_alt_target_transform(obs_model_raw, mode=alt_target_mode, clip_value=alt_target_clip)
            target_for_model = normalize_coords(target_model, target_norm_stats)
            obs_for_model = normalize_coords(obs_model, target_norm_stats)
            anchor_alt = build_anchor_alt_tracks(obs_pos=obs_pos, obs_mask=obs_mask, seq_mask=seq_mask)

            out = model(
                obs_pos=obs_for_model,
                obs_mask=obs_mask,
                seq_mask=seq_mask,
                dt_prev=dt_prev,
                dt_next=dt_next,
                exo=exo,
                vertical_exo=batch.get("vertical_exo"),
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
                target_pos=target_for_model,
                teacher_forcing_ratio=0.0,
                return_vertical_tune_weights=False,
            )

            p_all = out["savca_alloc_p"].detach().cpu().numpy()
            r_all = out["savca_state"].detach().cpu().numpy()
            beta_all = out["savca_beta"].detach().cpu().numpy()
            target_alt_abs = target_pos[..., 2].detach().cpu().numpy()
            obs_mask_np = obs_mask.detach().cpu().numpy()
            valid_np = seq_mask.detach().cpu().numpy() > 0.5

            for i, sid in enumerate(batch["sample_id"]):
                intervals = _find_intervals(obs_mask_np[i], valid_np[i])
                if not intervals:
                    continue
                point_beta = []
                savca_sq = []
                savca_abs = []
                center_shifts = []
                p_ents = []
                p_maxs = []
                state_seps = []
                active_ratios = []
                active_lens = []
                anchor_deltas = []
                gap_lens = []
                total_gap_pts = 0

                for left, right in intervals:
                    interval = np.arange(left + 1, right + 1)
                    gap_pts = np.arange(left + 1, right)
                    if gap_pts.size == 0:
                        continue
                    z_l = float(target_alt_abs[i, left])
                    z_r = float(target_alt_abs[i, right])
                    dz = z_r - z_l
                    p = np.clip(p_all[i, interval].astype(float), 0.0, None)
                    s = p.sum()
                    p = p / s if s > 1e-9 else np.full_like(p, 1.0 / len(p))
                    p_gap = p[:-1]
                    cdf = np.cumsum(p)
                    pred_seg = z_l + dz * cdf[:-1]
                    true_seg = target_alt_abs[i, gap_pts].astype(float)
                    savca_sq.extend(((pred_seg - true_seg) ** 2).tolist())
                    savca_abs.extend(np.abs(pred_seg - true_seg).tolist())
                    total_gap_pts += len(gap_pts)
                    point_beta.extend(beta_all[i, interval].astype(float).tolist())

                    anchor_delta = abs(z_r - z_l)
                    q, active_mask = _label_active_q(cfg["loss"], target_alt_abs[i, left : right + 1].astype(float), anchor_delta)
                    tau = np.arange(1, len(p) + 1, dtype=float) / max(1, len(p))
                    c_p = float((tau * p).sum()) if len(p) else np.nan
                    c_q = float((tau * q).sum()) if len(q) and q.sum() > 1e-9 else np.nan
                    if np.isfinite(c_p) and np.isfinite(c_q):
                        center_shifts.append(abs(c_p - c_q))
                    p_ents.append(_norm_entropy(p))
                    p_maxs.append(float(np.max(p)) if len(p) else 0.0)
                    if active_mask.size and active_mask.any():
                        r_active = r_all[i, interval][active_mask]
                        r_nonactive = r_all[i, interval][~active_mask]
                        state_sep = float(np.mean(r_active) - np.mean(r_nonactive)) if r_nonactive.size else float(np.mean(r_active))
                    else:
                        state_sep = np.nan
                    state_seps.append(state_sep)
                    active_ratios.append(float(active_mask.mean()) if active_mask.size else 0.0)
                    active_lens.append(float(active_mask.sum()))
                    anchor_deltas.append(anchor_delta)
                    gap_lens.append(float(right - left - 1))

                if total_gap_pts == 0:
                    continue
                rows.append(
                    {
                        "sample_id": str(sid),
                        "savca_gap_alt_rmse": float(np.sqrt(np.mean(np.asarray(savca_sq, dtype=float)))),
                        "savca_gap_alt_mae": float(np.mean(np.asarray(savca_abs, dtype=float))),
                        "beta_mean": float(np.mean(point_beta)) if point_beta else np.nan,
                        "beta_p50": float(np.quantile(point_beta, 0.5)) if point_beta else np.nan,
                        "beta_p75": float(np.quantile(point_beta, 0.75)) if point_beta else np.nan,
                        "beta_p90": float(np.quantile(point_beta, 0.9)) if point_beta else np.nan,
                        "center_shift_abs": float(np.nanmean(center_shifts)) if center_shifts else np.nan,
                        "p_entropy": float(np.nanmean(p_ents)) if p_ents else np.nan,
                        "p_max": float(np.nanmean(p_maxs)) if p_maxs else np.nan,
                        "state_sep": float(np.nanmean(state_seps)) if state_seps else np.nan,
                        "active_ratio": float(np.nanmean(active_ratios)) if active_ratios else np.nan,
                        "active_len": float(np.nanmean(active_lens)) if active_lens else np.nan,
                        "anchor_delta_abs_m": float(np.nanmax(anchor_deltas)) if anchor_deltas else np.nan,
                        "gap_len": float(np.nanmax(gap_lens)) if gap_lens else np.nan,
                        "gap_points": int(total_gap_pts),
                    }
                )

    diag = pd.DataFrame(rows)
    fused = pd.read_csv(run_dir / "main_task_metrics_val_per_sample.csv")
    a1 = pd.read_csv(ROOT / "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_physical_time_ablation_v1/a1_linear_alt_baseline/main_task_metrics_val_per_sample.csv")
    merged = diag.merge(
        fused[["sample_id", "gap_alt_rmse", "gap_alt_mae", "segment_bucket_name", "segment_bucket", "anchor_count", "gap_count", "max_gap_minutes"]],
        on="sample_id",
    ).merge(
        a1[["sample_id", "gap_alt_rmse", "gap_alt_mae"]],
        on="sample_id",
        suffixes=("_fused", "_a1"),
    )
    merged["savca_minus_a1"] = merged["savca_gap_alt_rmse"] - merged["gap_alt_rmse_a1"]
    merged["fused_minus_a1"] = merged["gap_alt_rmse_fused"] - merged["gap_alt_rmse_a1"]
    merged["oracle_pick_savca"] = merged["savca_gap_alt_rmse"] < merged["gap_alt_rmse_a1"]

    def aggregate(mode: str) -> tuple[float, float]:
        w = pd.to_numeric(merged["gap_count"], errors="coerce").fillna(merged["gap_points"]).to_numpy(dtype=float)
        if mode == "a1":
            rmse_sq = merged["gap_alt_rmse_a1"].to_numpy(dtype=float) ** 2
            mae = merged["gap_alt_mae_a1"].to_numpy(dtype=float)
        elif mode == "savca":
            rmse_sq = merged["savca_gap_alt_rmse"].to_numpy(dtype=float) ** 2
            mae = merged["savca_gap_alt_mae"].to_numpy(dtype=float)
        elif mode == "fused":
            rmse_sq = merged["gap_alt_rmse_fused"].to_numpy(dtype=float) ** 2
            mae = merged["gap_alt_mae_fused"].to_numpy(dtype=float)
        else:
            pick = merged["oracle_pick_savca"].to_numpy(dtype=bool)
            rmse_sq = np.where(
                pick,
                merged["savca_gap_alt_rmse"].to_numpy(dtype=float) ** 2,
                merged["gap_alt_rmse_a1"].to_numpy(dtype=float) ** 2,
            )
            mae = np.where(
                pick,
                merged["savca_gap_alt_mae"].to_numpy(dtype=float),
                merged["gap_alt_mae_a1"].to_numpy(dtype=float),
            )
        return float(np.sqrt((rmse_sq * w).sum() / w.sum())), float((mae * w).sum() / w.sum())

    oracle_rows = []
    for mode in ["a1", "savca", "fused", "oracle"]:
        rmse, mae = aggregate(mode)
        oracle_rows.append({"method": mode, "gap_alt_rmse": rmse, "gap_alt_mae": mae})
    pd.DataFrame(oracle_rows).to_csv(out_dir / "oracle_gate_summary.csv", index=False)

    records = []
    for group_name, group_df in [("all", merged)] + [(f"bucket::{b}", g) for b, g in merged.groupby("segment_bucket_name")]:
        rec = {
            "group": group_name,
            "count": len(group_df),
            "savca_better_ratio": float((group_df["savca_minus_a1"] < 0).mean()),
            "savca_worse_gt5_ratio": float((group_df["savca_minus_a1"] > 5).mean()),
            "fused_better_ratio": float((group_df["fused_minus_a1"] < 0).mean()),
            "fused_worse_gt5_ratio": float((group_df["fused_minus_a1"] > 5).mean()),
        }
        for col in [
            "beta_mean",
            "beta_p50",
            "beta_p75",
            "beta_p90",
            "center_shift_abs",
            "p_entropy",
            "p_max",
            "state_sep",
            "active_ratio",
            "anchor_delta_abs_m",
            "gap_len",
            "savca_minus_a1",
            "fused_minus_a1",
        ]:
            x = pd.to_numeric(group_df[col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(x):
                rec[f"{col}_mean"] = float(np.mean(x))
                rec[f"{col}_p50"] = float(np.quantile(x, 0.5))
                rec[f"{col}_p75"] = float(np.quantile(x, 0.75))
                rec[f"{col}_p90"] = float(np.quantile(x, 0.9))
        records.append(rec)
    pd.DataFrame(records).to_csv(out_dir / "beta_group_summary.csv", index=False)

    merged["delta_bucket"] = pd.cut(
        merged["anchor_delta_abs_m"],
        bins=[-1, 30, 100, 300, 1e9],
        labels=["[0,30]", "(30,100]", "(100,300]", "(300,+inf]"],
    )
    bucket_rows = []
    for key, group_df in merged.groupby(["segment_bucket_name", "delta_bucket"], dropna=False):
        bucket_rows.append(
            {
                "segment_bucket_name": key[0],
                "delta_bucket": str(key[1]),
                "count": len(group_df),
                "beta_mean": float(pd.to_numeric(group_df["beta_mean"], errors="coerce").mean()),
                "beta_p90": float(pd.to_numeric(group_df["beta_mean"], errors="coerce").quantile(0.9)),
                "center_shift_abs_mean": float(pd.to_numeric(group_df["center_shift_abs"], errors="coerce").mean()),
                "state_sep_mean": float(pd.to_numeric(group_df["state_sep"], errors="coerce").mean()),
                "p_entropy_mean": float(pd.to_numeric(group_df["p_entropy"], errors="coerce").mean()),
                "p_max_mean": float(pd.to_numeric(group_df["p_max"], errors="coerce").mean()),
                "savca_minus_a1_mean": float(pd.to_numeric(group_df["savca_minus_a1"], errors="coerce").mean()),
                "fused_minus_a1_mean": float(pd.to_numeric(group_df["fused_minus_a1"], errors="coerce").mean()),
            }
        )
    pd.DataFrame(bucket_rows).to_csv(out_dir / "beta_bucket_delta_summary.csv", index=False)

    merged["reliability_group"] = pd.cut(
        merged["savca_minus_a1"],
        bins=[-1e9, -5, 5, 1e9],
        labels=["savca_better", "similar", "savca_worse"],
    )
    rel_rows = []
    for key, group_df in merged.groupby("reliability_group", dropna=False):
        rel_rows.append(
            {
                "reliability_group": str(key),
                "count": len(group_df),
                "beta_mean": float(pd.to_numeric(group_df["beta_mean"], errors="coerce").mean()),
                "center_shift_abs_mean": float(pd.to_numeric(group_df["center_shift_abs"], errors="coerce").mean()),
                "p_entropy_mean": float(pd.to_numeric(group_df["p_entropy"], errors="coerce").mean()),
                "p_max_mean": float(pd.to_numeric(group_df["p_max"], errors="coerce").mean()),
                "state_sep_mean": float(pd.to_numeric(group_df["state_sep"], errors="coerce").mean()),
                "fused_minus_a1_mean": float(pd.to_numeric(group_df["fused_minus_a1"], errors="coerce").mean()),
            }
        )
    pd.DataFrame(rel_rows).to_csv(out_dir / "beta_reliability_summary.csv", index=False)
    merged.to_csv(out_dir / "val_sample_beta_diagnostics.csv", index=False)
    print(pd.DataFrame(oracle_rows).to_string(index=False))
    print("rows", len(merged))
    print(merged[["beta_mean", "savca_minus_a1", "fused_minus_a1", "center_shift_abs", "p_entropy", "p_max", "state_sep"]].describe().to_string())
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
