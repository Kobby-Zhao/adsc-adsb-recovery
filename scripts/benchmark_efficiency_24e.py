#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.datasets import DatasetConfig, TrajectoryDataset, trajectory_collate_fn
from src.models import TrajectoryRecoveryModel
from src.preprocessing.standardize import apply_standardizer, load_standardizer
from src.training.alt_target import apply_alt_target_transform, invert_alt_target_transform
from src.training.altitude_governance import add_anchor_alt_features, add_vertical_v2_features
from src.training.coords import (
    build_anchor_alt_tracks,
    build_anchor_pair_tracks,
    prepare_model_coordinates,
    restore_to_latlon,
)
from src.training.target_norm import denormalize_coords, load_target_stats, normalize_coords
from src.training.utils import load_config, set_seed, split_by_flight_id

from scripts.eval_interpolation_baselines import _interp_linear_gapwise
from scripts.eval_rts_kalman_baseline import _smooth_sample


ROOT = Path("/home/jj/workspace/data-0313")
FINAL_CFG_ROOT = ROOT / "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs"
FINAL_RUN_ROOT = ROOT / "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum"

MODEL_SPECS = [
    {
        "model": "本文方案",
        "kind": "neural",
        "config": FINAL_CFG_ROOT / "formal_24ep_gapaware_small.yaml",
        "checkpoint": FINAL_RUN_ROOT / "formal_24ep_gapaware_small/best.pt",
        "history": FINAL_RUN_ROOT / "formal_24ep_gapaware_small/history.json",
    },
    {
        "model": "分段线性插值",
        "kind": "piecewise_linear",
    },
    {
        "model": "Kalman Filter",
        "kind": "kalman",
    },
    {
        "model": "Mamba",
        "kind": "neural",
        "config": FINAL_CFG_ROOT / "formal_24ep_mamba_proto.yaml",
        "checkpoint": FINAL_RUN_ROOT / "formal_24ep_mamba_proto/best.pt",
        "history": FINAL_RUN_ROOT / "formal_24ep_mamba_proto/history.json",
    },
    {
        "model": "Bi-Mamba",
        "kind": "neural",
        "config": FINAL_CFG_ROOT / "formal_24ep_bimamba.yaml",
        "checkpoint": FINAL_RUN_ROOT / "formal_24ep_bimamba/best.pt",
        "history": FINAL_RUN_ROOT / "formal_24ep_bimamba/history.json",
    },
    {
        "model": "LSTM",
        "kind": "neural",
        "config": FINAL_CFG_ROOT / "formal_24ep_unilstm_proto.yaml",
        "checkpoint": FINAL_RUN_ROOT / "formal_24ep_unilstm_proto/best.pt",
        "history": FINAL_RUN_ROOT / "formal_24ep_unilstm_proto/history.json",
    },
    {
        "model": "BiLSTM",
        "kind": "neural",
        "config": FINAL_CFG_ROOT / "formal_24ep_bilstm_proto.yaml",
        "checkpoint": FINAL_RUN_ROOT / "formal_24ep_bilstm_proto/best.pt",
        "history": FINAL_RUN_ROOT / "formal_24ep_bilstm_proto/history.json",
    },
    {
        "model": "CNN+LSTM",
        "kind": "neural",
        "config": FINAL_CFG_ROOT / "formal_24ep_cnnlstm_proto.yaml",
        "checkpoint": FINAL_RUN_ROOT / "formal_24ep_cnnlstm_proto/best.pt",
        "history": FINAL_RUN_ROOT / "formal_24ep_cnnlstm_proto/history.json",
    },
    {
        "model": "Transformer",
        "kind": "neural",
        "config": FINAL_CFG_ROOT / "formal_24ep_transformer_proto.yaml",
        "checkpoint": FINAL_RUN_ROOT / "formal_24ep_transformer_proto/best.pt",
        "history": FINAL_RUN_ROOT / "formal_24ep_transformer_proto/history.json",
    },
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark model efficiency for final 24-epoch baselines.")
    p.add_argument(
        "--samples",
        default="outputs/mvp_merged_250_20260514_clean/stage3_clean/samples.parquet",
        help="Benchmark split source parquet; defaults to final S3 clean split.",
    )
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--max-samples", type=int, default=64, help="Number of anchor-valid samples to benchmark.")
    p.add_argument("--warmup", type=int, default=8, help="Warmup samples before timing.")
    p.add_argument("--repeat", type=int, default=3, help="Repeat count per sample for latency averaging.")
    p.add_argument("--device", default="cuda", help="cuda or cpu; CUDA strongly recommended.")
    p.add_argument(
        "--out-csv",
        default="outputs/analysis/efficiency_benchmark_24e.csv",
        help="Where to write the benchmark CSV.",
    )
    return p


def _sample_has_anchor(frame: pd.DataFrame, sample_id_col: str, obs_mask_col: str) -> pd.Series:
    obs = pd.to_numeric(frame.get(obs_mask_col, 0.0), errors="coerce").fillna(0.0)
    by = frame.assign(_obs_anchor=(obs > 0.5)).groupby(sample_id_col)["_obs_anchor"].any()
    return by.astype(bool)


def _split_main_frame(cfg: dict, split_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    split_df = splits[split_name]
    train_df = splits["train"]

    has_anchor = _sample_has_anchor(split_df, cfg["data"]["sample_id_col"], cfg["data"]["obs_mask_col"])
    keep = set(has_anchor[has_anchor].index.astype(str).tolist())
    split_main_df = split_df[split_df[cfg["data"]["sample_id_col"]].astype(str).isin(keep)].copy()

    has_anchor_train = _sample_has_anchor(train_df, cfg["data"]["sample_id_col"], cfg["data"]["obs_mask_col"])
    keep_train = set(has_anchor_train[has_anchor_train].index.astype(str).tolist())
    train_main_df = train_df[train_df[cfg["data"]["sample_id_col"]].astype(str).isin(keep_train)].copy()
    return split_main_df, train_main_df


def _make_dataset(cfg: dict, split_name: str) -> tuple[TrajectoryDataset, dict | None]:
    split_main_df, _train_main_df = _split_main_frame(cfg, split_name)
    run_dir = Path(cfg["outputs"]["run_dir"])
    scaler_path = run_dir / "feature_standardizer.json"
    scaler_stats = load_standardizer(scaler_path)
    if scaler_stats is not None:
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
    target_norm_stats = None
    if bool(cfg["training"].get("target_norm", {}).get("enabled", False)):
        target_norm_stats = load_target_stats(run_dir / "target_model_scaler.json")
    return ds, target_norm_stats


def _build_model(cfg: dict, checkpoint: Path, device: torch.device) -> TrajectoryRecoveryModel:
    abr_bounds = cfg["model"].get("alt_base_residual_bounds")
    if abr_bounds is None:
        bpath = Path(cfg["outputs"]["run_dir"]) / "alt_base_residual_bounds.json"
        if bpath.exists():
            abr_bounds = json.loads(bpath.read_text()).get("alt_base_residual_bounds")
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
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


def _sample_to_frame(sample: dict) -> pd.DataFrame:
    target = sample["target_pos"].detach().cpu().numpy()
    obs_mask = sample["obs_mask"].detach().cpu().numpy()
    return pd.DataFrame(
        {
            "minute_ts": sample["times"],
            "obs_mask": obs_mask.astype(float),
            "lat": target[:, 0].astype(float),
            "lon": target[:, 1].astype(float),
            "alt": target[:, 2].astype(float),
        }
    )


def _prepare_forward_inputs(batch: dict, cfg: dict, device: torch.device, target_norm_stats: dict | None) -> dict:
    obs_pos = batch["obs_pos"].to(device)
    obs_mask = batch["obs_mask"].to(device)
    dt_prev = batch["dt_prev"].to(device)
    dt_next = batch["dt_next"].to(device)
    exo = batch["exo"].to(device)
    quality = batch["quality"].to(device)
    global_quality = batch["global_quality"].to(device)
    vertical_exo = batch["vertical_exo"].to(device) if "vertical_exo" in batch else None
    seq_mask = batch["seq_mask"].to(device)
    target_pos = batch["target_pos"].to(device)

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
    target_model = apply_alt_target_transform(target_model, mode=alt_target_mode, clip_value=alt_target_clip)
    obs_model = apply_alt_target_transform(obs_model, mode=alt_target_mode, clip_value=alt_target_clip)
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
    anchor_alt = build_anchor_alt_tracks(obs_pos=obs_pos, obs_mask=obs_mask, seq_mask=seq_mask)
    return {
        "obs_for_model": obs_for_model,
        "obs_mask": obs_mask,
        "seq_mask": seq_mask,
        "dt_prev": dt_prev,
        "dt_next": dt_next,
        "exo": exo,
        "vertical_exo": vertical_exo,
        "quality": quality,
        "global_quality": global_quality,
        "anchor_alt": anchor_alt,
        "anchor_left": anchor_left_model,
        "anchor_right": anchor_right_model,
        "coord_ctx": coord_ctx,
        "alt_target_mode": alt_target_mode,
        "alt_target_clip": alt_target_clip,
        "target_norm_stats": target_norm_stats,
    }


def _run_neural_once(model: TrajectoryRecoveryModel, prepared: dict) -> torch.Tensor:
    out = model(
        obs_pos=prepared["obs_for_model"],
        obs_mask=prepared["obs_mask"],
        seq_mask=prepared["seq_mask"],
        dt_prev=prepared["dt_prev"],
        dt_next=prepared["dt_next"],
        exo=prepared["exo"],
        vertical_exo=prepared["vertical_exo"],
        quality=prepared["quality"],
        global_quality=prepared["global_quality"],
        anchor_alt=prepared["anchor_alt"],
        risk_flag=None,
        teacher_scale=None,
        risk_flag_teacher=None,
        segment_bucket=None,
        edge_weight=None,
        residual_rmax_m=None,
        residual_rmax_ft=None,
        gate_bias=None,
        left_boundary_alt=None,
        right_boundary_alt=None,
        anchor_left=prepared["anchor_left"],
        anchor_right=prepared["anchor_right"],
        target_pos=None,
        teacher_forcing_ratio=0.0,
    )
    pred_model_t = denormalize_coords(out["pred_pos"], prepared["target_norm_stats"])
    pred_model = invert_alt_target_transform(
        pred_model_t,
        mode=prepared["alt_target_mode"],
        clip_value=prepared["alt_target_clip"],
    )
    pred_latlon = restore_to_latlon(pred_model, seq_mask=prepared["seq_mask"], ctx=prepared["coord_ctx"])
    return pred_latlon


def _history_stats(history_path: Path) -> dict[str, float | int]:
    hist = json.loads(history_path.read_text())
    vals = [r["epoch_sec"] for r in hist["train"] if "epoch_sec" in r]
    total = float(sum(vals))
    return {
        "epochs": int(len(vals)),
        "train_total_sec": total,
        "train_total_min": total / 60.0,
        "train_avg_epoch_sec": total / max(len(vals), 1),
    }


def _benchmark_neural(spec: dict, args: argparse.Namespace, raw_cfg: dict) -> dict:
    device = torch.device(args.device)
    cfg = dict(raw_cfg)
    cfg["data"]["samples_path"] = str(Path(args.samples))
    ds, target_norm_stats = _make_dataset(cfg, args.split)
    samples = [s for s in ds.samples if float(s["obs_mask"].sum().item()) > 0.5][: args.max_samples]
    if not samples:
        raise RuntimeError(f"No anchor-valid samples found for {spec['model']}.")

    model = _build_model(cfg, Path(spec["checkpoint"]), device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    prepared_batches = []
    for sample in samples:
        batch = trajectory_collate_fn([sample])
        prepared_batches.append(_prepare_forward_inputs(batch, cfg, device, target_norm_stats))

    warmup = min(args.warmup, len(prepared_batches))
    with torch.no_grad():
        for i in range(warmup):
            _ = _run_neural_once(model, prepared_batches[i])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        elapsed = 0.0
        measured = 0
        for prepared in prepared_batches:
            for _ in range(args.repeat):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                _ = _run_neural_once(model, prepared)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed += time.perf_counter() - t0
                measured += 1

    peak_mb = float("nan")
    if device.type == "cuda":
        peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))

    row = {
        "model": spec["model"],
        "params": total_params,
        "trainable_params": trainable_params,
        "params_m": total_params / 1e6,
        "inference_ms_per_sample": (elapsed / max(measured, 1)) * 1000.0,
        "peak_gpu_mem_mb": peak_mb,
        "flops_gmacs": np.nan,
        "notes": "FLOPs/MACs not filled; custom kernels need dedicated profiler.",
    }
    row.update(_history_stats(Path(spec["history"])))
    return row


def _benchmark_piecewise(args: argparse.Namespace, cfg: dict) -> dict:
    cfg = dict(cfg)
    cfg["data"]["samples_path"] = str(Path(args.samples))
    ds, _ = _make_dataset(cfg, args.split)
    samples = [s for s in ds.samples if float(s["obs_mask"].sum().item()) > 0.5][: args.max_samples]
    warmup = min(args.warmup, len(samples))
    for i in range(warmup):
        _interp_linear_gapwise(_sample_to_frame(samples[i]))
    elapsed = 0.0
    measured = 0
    for sample in samples:
        sdf = _sample_to_frame(sample)
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            _interp_linear_gapwise(sdf)
            elapsed += time.perf_counter() - t0
            measured += 1
    return {
        "model": "分段线性插值",
        "params": np.nan,
        "trainable_params": np.nan,
        "params_m": np.nan,
        "train_total_sec": np.nan,
        "train_total_min": np.nan,
        "train_avg_epoch_sec": np.nan,
        "epochs": 0,
        "inference_ms_per_sample": (elapsed / max(measured, 1)) * 1000.0,
        "peak_gpu_mem_mb": np.nan,
        "flops_gmacs": np.nan,
        "notes": "No training; CPU interpolation baseline.",
    }


def _benchmark_kalman(args: argparse.Namespace, cfg: dict) -> dict:
    cfg = dict(cfg)
    cfg["data"]["samples_path"] = str(Path(args.samples))
    ds, _ = _make_dataset(cfg, args.split)
    samples = [s for s in ds.samples if float(s["obs_mask"].sum().item()) > 0.5][: args.max_samples]
    warmup = min(args.warmup, len(samples))
    for i in range(warmup):
        _smooth_sample(_sample_to_frame(samples[i]))
    elapsed = 0.0
    measured = 0
    for sample in samples:
        sdf = _sample_to_frame(sample)
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            _smooth_sample(sdf)
            elapsed += time.perf_counter() - t0
            measured += 1
    return {
        "model": "Kalman Filter",
        "params": np.nan,
        "trainable_params": np.nan,
        "params_m": np.nan,
        "train_total_sec": np.nan,
        "train_total_min": np.nan,
        "train_avg_epoch_sec": np.nan,
        "epochs": 0,
        "inference_ms_per_sample": (elapsed / max(measured, 1)) * 1000.0,
        "peak_gpu_mem_mb": np.nan,
        "flops_gmacs": np.nan,
        "notes": "No training; standalone RTS smoother baseline.",
    }


def main() -> int:
    args = build_parser().parse_args()
    set_seed(42)
    device = torch.device(args.device)

    rows = []
    base_cfg = load_config(str(FINAL_CFG_ROOT / "formal_24ep_gapaware_small.yaml"))

    for spec in MODEL_SPECS:
        print(f"[benchmark] {spec['model']}")
        if spec["kind"] == "piecewise_linear":
            rows.append(_benchmark_piecewise(args, base_cfg))
            continue
        if spec["kind"] == "kalman":
            rows.append(_benchmark_kalman(args, base_cfg))
            continue

        cfg = load_config(str(spec["config"]))
        backbone = str(cfg["model"].get("backbone_type", "")).lower()
        if device.type != "cuda" and backbone in {
            "mamba_proto",
            "bimamba",
            "bimamba_context_xyaux_zlinear",
            "bimamba_context_xyaux_zlinear_zadapter_gapaware_small",
        }:
            raise RuntimeError(f"{spec['model']} requires CUDA for fair benchmarking; got device={device}.")
        rows.append(_benchmark_neural(spec, args, cfg))

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[done] saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
