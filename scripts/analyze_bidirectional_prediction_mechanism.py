from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.evaluate import _boundary_alt_from_model_obs, _max_gap_len, _split_main_and_no_anchor
from src.datasets import DatasetConfig, TrajectoryDataset, trajectory_collate_fn
from src.models import TrajectoryRecoveryModel
from src.preprocessing.standardize import apply_standardizer, load_standardizer
from src.training.alt_target import apply_alt_target_transform, invert_alt_target_transform
from src.training.altitude_governance import add_anchor_alt_features, add_vertical_v2_features
from src.training.coords import build_anchor_alt_tracks, build_anchor_pair_tracks, prepare_model_coordinates, restore_to_latlon
from src.training.target_norm import denormalize_coords, load_target_stats, normalize_coords
from src.training.utils import load_config, set_seed, split_by_flight_id


ROOT = Path(__file__).resolve().parents[1]


MODEL_KEYS = {
    "Backbone-only": "ours_backbone_absolute",
    "Ours-A3": "a3_risk_routed",
    "BiLSTM-clean": "bilstm_clean_absolute",
}


def _model_specs(run_tag: str) -> dict[str, dict[str, str]]:
    base = f"outputs/experiments/obs_conditioned_gaponly/{run_tag}"
    return {
        label: {
            "config": f"{base}/configs/{key}.yaml",
            "checkpoint": f"{base}/{key}/best.pt",
        }
        for label, key in MODEL_KEYS.items()
    }


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _build_model(cfg: dict, checkpoint: Path, device: torch.device) -> TrajectoryRecoveryModel:
    abr_bounds = cfg["model"].get("alt_base_residual_bounds")
    if abr_bounds is None:
        bpath = _resolve(cfg["outputs"]["run_dir"]) / "alt_base_residual_bounds.json"
        if bpath.exists():
            with bpath.open("r", encoding="utf-8") as f:
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


def _prepare_dataset(cfg: dict, split_name: str) -> TrajectoryDataset:
    df = pd.read_parquet(_resolve(cfg["data"]["samples_path"]))
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

    split_main_df, _, _ = _split_main_and_no_anchor(splits[split_name], split_name)
    scaler_stats = load_standardizer(_resolve(cfg["outputs"]["run_dir"]) / "feature_standardizer.json")
    if scaler_stats is None:
        raise RuntimeError(f"feature_standardizer.json is missing in {cfg['outputs']['run_dir']}")
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
    return TrajectoryDataset(split_main_df, dcfg)


def _sample_summary(ds: TrajectoryDataset) -> pd.DataFrame:
    rows: list[dict] = []
    for sample in ds.samples:
        obs = sample["obs_mask"].numpy()
        target = sample["target_pos"].numpy()
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "flight_id": sample["flight_id"],
                "length": int(len(obs)),
                "anchor_count": int((obs > 0.5).sum()),
                "gap_count": int((obs <= 0.5).sum()),
                "max_gap": int(_max_gap_len(obs)),
                "alt_range_m": float(np.nanmax(target[:, 2]) - np.nanmin(target[:, 2])),
            }
        )
    return pd.DataFrame(rows)


def _choose_two_samples(summary: pd.DataFrame) -> pd.DataFrame:
    sparse = summary[(summary["anchor_count"] < 5) & (summary["gap_count"] > 20)].copy()
    dense = summary[(summary["anchor_count"].between(8, 10)) & (summary["gap_count"] > 20)].copy()
    if sparse.empty or dense.empty:
        raise RuntimeError("Cannot find both sparse-anchor and 8-10-anchor samples.")

    # Prefer a matched underlying flight so the only major change is anchor density.
    for _, row in sparse.sort_values(["max_gap", "alt_range_m"], ascending=False).iterrows():
        d = dense[dense["flight_id"].eq(row["flight_id"])]
        if not d.empty:
            pick_dense = d.sort_values(["max_gap", "alt_range_m"], ascending=False).iloc[0]
            return pd.DataFrame(
                [
                    {"scenario": "sparse_anchor_lt5", **row.to_dict()},
                    {"scenario": "anchor_8_to_10", **pick_dense.to_dict()},
                ]
            )

    pick_sparse = sparse.sort_values(["max_gap", "alt_range_m"], ascending=False).iloc[0]
    pick_dense = dense.sort_values(["max_gap", "alt_range_m"], ascending=False).iloc[0]
    return pd.DataFrame(
        [
            {"scenario": "sparse_anchor_lt5", **pick_sparse.to_dict()},
            {"scenario": "anchor_8_to_10", **pick_dense.to_dict()},
        ]
    )


def _gap_position_from_obs_mask(obs_mask: np.ndarray) -> np.ndarray:
    """Return true [0, 1] position inside each contiguous gap run.

    Model inputs may standardize dt_prev/dt_next, so plotting dt_prev/(dt_prev+dt_next)
    directly is not a valid physical gap coordinate. This diagnostic coordinate is
    computed from the observed/missing pattern only.
    """
    obs = np.asarray(obs_mask, dtype=float) > 0.5
    out = np.full(len(obs), np.nan, dtype=np.float64)
    t = 0
    while t < len(obs):
        if obs[t]:
            t += 1
            continue
        s = t
        while t < len(obs) and not obs[t]:
            t += 1
        e = t
        n = e - s
        if n > 0:
            out[s:e] = (np.arange(n, dtype=np.float64) + 1.0) / (n + 1.0)
    return out


def _restore_series(
    series_model: torch.Tensor,
    *,
    seq_mask: torch.Tensor,
    coord_ctx: dict,
    target_norm_stats: dict | None,
    alt_target_mode: str,
    alt_target_clip: float,
) -> np.ndarray:
    den = denormalize_coords(series_model, target_norm_stats)
    inv = invert_alt_target_transform(den, mode=alt_target_mode, clip_value=alt_target_clip)
    latlon = restore_to_latlon(inv, seq_mask=seq_mask, ctx=coord_ctx)
    return latlon.detach().cpu().numpy()


def _run_model_for_samples(
    model_name: str,
    model_specs: dict[str, dict[str, str]],
    selected_ids: set[str],
    split_name: str,
    device: torch.device,
) -> dict[str, dict]:
    spec = model_specs[model_name]
    cfg = load_config(str(_resolve(spec["config"])))
    ds = _prepare_dataset(cfg, split_name=split_name)
    ds.samples = [s for s in ds.samples if str(s["sample_id"]) in selected_ids]
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=trajectory_collate_fn)
    model = _build_model(cfg, _resolve(spec["checkpoint"]), device)

    target_norm_cfg = cfg["training"].get("target_norm", {})
    target_norm_stats = None
    if bool(target_norm_cfg.get("enabled", False)):
        target_norm_stats = load_target_stats(_resolve(cfg["outputs"]["run_dir"]) / "target_model_scaler.json")
    alt_target_mode = str(cfg["training"].get("alt_target_robust", {}).get("mode", "none")).lower()
    alt_target_clip = float(cfg["training"].get("alt_target_robust", {}).get("clip_value", 3000.0))
    use_segment_teacher = bool(cfg.get("training", {}).get("risk_aware", {}).get("use_segment_teacher", True))
    use_alt_baseline_residual = bool(cfg.get("training", {}).get("risk_aware", {}).get("use_alt_baseline_residual", True))

    results: dict[str, dict] = {}
    with torch.no_grad():
        for batch in loader:
            obs_pos = batch["obs_pos"].to(device)
            obs_mask = batch["obs_mask"].to(device)
            dt_prev = batch["dt_prev"].to(device)
            dt_next = batch["dt_next"].to(device)
            exo = batch["exo"].to(device)
            quality = batch["quality"].to(device)
            global_quality = batch["global_quality"].to(device)
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
            left_alt, right_alt = _boundary_alt_from_model_obs(obs_for_model, obs_mask, seq_mask)
            anchor_alt = build_anchor_alt_tracks(obs_pos=obs_pos, obs_mask=obs_mask, seq_mask=seq_mask)

            out = model(
                obs_pos=obs_for_model,
                obs_mask=obs_mask,
                dt_prev=dt_prev,
                dt_next=dt_next,
                exo=exo,
                vertical_exo=batch["vertical_exo"].to(device) if "vertical_exo" in batch else None,
                quality=quality,
                global_quality=global_quality,
                anchor_alt=anchor_alt,
                risk_flag=batch["risk_flag"].to(device) if "risk_flag" in batch else None,
                teacher_scale=batch["teacher_scale"].to(device) if ("teacher_scale" in batch and use_segment_teacher) else None,
                risk_flag_teacher=batch["risk_flag_teacher"].to(device)
                if ("risk_flag_teacher" in batch and use_segment_teacher)
                else None,
                segment_bucket=batch["segment_bucket"].to(device) if "segment_bucket" in batch else None,
                edge_weight=batch["edge_weight"].to(device) if ("edge_weight" in batch and use_segment_teacher) else None,
                residual_rmax_m=batch["residual_rmax_m"].to(device)
                if ("residual_rmax_m" in batch and use_alt_baseline_residual)
                else None,
                residual_rmax_ft=batch["residual_rmax_ft"].to(device)
                if ("residual_rmax_ft" in batch and use_alt_baseline_residual)
                else None,
                gate_bias=batch["gate_bias"].to(device) if ("gate_bias" in batch and use_segment_teacher) else None,
                left_boundary_alt=left_alt,
                right_boundary_alt=right_alt,
                anchor_left=anchor_left_model,
                anchor_right=anchor_right_model,
                target_pos=None,
                teacher_forcing_ratio=0.0,
            )

            sid = str(batch["sample_id"][0])
            valid = batch["seq_mask"][0].detach().cpu().numpy() > 0.5
            restored = {
                "final": _restore_series(
                    out["pred_pos"],
                    seq_mask=seq_mask,
                    coord_ctx=coord_ctx,
                    target_norm_stats=target_norm_stats,
                    alt_target_mode=alt_target_mode,
                    alt_target_clip=alt_target_clip,
                )[0][valid],
                "mu_f": _restore_series(
                    out["mu_f"],
                    seq_mask=seq_mask,
                    coord_ctx=coord_ctx,
                    target_norm_stats=target_norm_stats,
                    alt_target_mode=alt_target_mode,
                    alt_target_clip=alt_target_clip,
                )[0][valid],
                "mu_b": _restore_series(
                    out["mu_b"],
                    seq_mask=seq_mask,
                    coord_ctx=coord_ctx,
                    target_norm_stats=target_norm_stats,
                    alt_target_mode=alt_target_mode,
                    alt_target_clip=alt_target_clip,
                )[0][valid],
                "weights": out["fusion_weights"].detach().cpu().numpy()[0][valid],
            }
            results[sid] = restored
    return results


def _plot_scenario(case_dir: Path, scenario_name: str, table: pd.DataFrame) -> None:
    gap = table["obs_mask"].to_numpy() <= 0.5
    anchor = table["obs_mask"].to_numpy() > 0.5
    x = table["minute_index"].to_numpy()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(x, table["target_adsb_alt_m"], color="#111111", lw=2.2, label="ADS-B truth")
    ax.scatter(x[anchor], table.loc[anchor, "anchor_alt_m"], s=28, color="#1b9e77", zorder=4, label="Observed anchors")
    ax.plot(x, table["Backbone_fusion_alt_m"], color="#1f77b4", lw=1.8, label="Backbone fusion")
    ax.plot(x, table["Ours_A3_alt_m"], color="#d62728", lw=2.2, label="Ours-A3")
    ax.plot(x, table["BiLSTM_alt_m"], color="#666666", lw=1.8, ls="--", label="BiLSTM")
    ax.fill_between(x, table["target_adsb_alt_m"].min(), table["target_adsb_alt_m"].max(), where=gap, color="#f1f1f1", alpha=0.45, label="Gap points")
    ax.set_title(f"{scenario_name}: altitude recovery")
    ax.set_xlabel("Minute index")
    ax.set_ylabel("Altitude (m)")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(case_dir / "01_altitude_recovery_curve.png", dpi=180)
    plt.close(fig)

    gap_table = table[gap].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(gap_table["gap_pos_ratio"], gap_table["Backbone_forward_abs_err_m"], color="#2ca02c", lw=1.6, label="Ours forward branch")
    axes[0].plot(gap_table["gap_pos_ratio"], gap_table["Backbone_backward_abs_err_m"], color="#ff7f0e", lw=1.6, label="Ours backward branch")
    axes[0].plot(gap_table["gap_pos_ratio"], gap_table["Backbone_fusion_abs_err_m"], color="#1f77b4", lw=2.0, label="Backbone fusion")
    axes[0].plot(gap_table["gap_pos_ratio"], gap_table["Ours_A3_abs_err_m"], color="#d62728", lw=2.0, label="Ours-A3")
    axes[0].plot(gap_table["gap_pos_ratio"], gap_table["BiLSTM_abs_err_m"], color="#666666", lw=1.8, ls="--", label="BiLSTM")
    axes[0].set_xlabel("Normalized gap position")
    axes[0].set_ylabel("Altitude absolute error (m)")
    axes[0].set_title("Forward/backward/fusion error")
    axes[0].legend(fontsize=7)

    axes[1].plot(gap_table["gap_pos_ratio"], gap_table["Backbone_fusion_w_forward"], color="#1f77b4", lw=2.0, label="Backbone forward weight")
    axes[1].plot(gap_table["gap_pos_ratio"], gap_table["Backbone_fusion_w_backward"], color="#1f77b4", lw=2.0, ls=":", label="Backbone backward weight")
    axes[1].plot(gap_table["gap_pos_ratio"], gap_table["Ours_A3_fusion_w_forward"], color="#d62728", lw=1.8, label="A3 forward weight")
    axes[1].plot(gap_table["gap_pos_ratio"], gap_table["BiLSTM_fusion_w_forward"], color="#666666", lw=1.8, ls="--", label="BiLSTM forward weight")
    axes[1].set_xlabel("Normalized gap position")
    axes[1].set_ylabel("Fusion weight")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("Fusion weight vs gap position")
    axes[1].legend(fontsize=7)

    fig.suptitle(f"{scenario_name}: bidirectional mechanism analysis", y=1.02)
    fig.tight_layout()
    fig.savefig(case_dir / "02_bidirectional_error_and_fusion_weight.png", dpi=180)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument(
        "--run-tag",
        default="obscons_gaponly_physical_time_ablation_v1",
        help="Experiment run tag under outputs/experiments/obs_conditioned_gaponly.",
    )
    ap.add_argument(
        "--out-dir",
        default="outputs/experiments/obs_conditioned_gaponly/bidirectional_mechanism_analysis_physical_time_20260518",
    )
    args = ap.parse_args()

    set_seed(42)
    device = torch.device("cpu")
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_specs = _model_specs(args.run_tag)
    for model_name, spec in model_specs.items():
        for field in ["config", "checkpoint"]:
            path = _resolve(spec[field])
            if not path.exists():
                raise FileNotFoundError(f"Missing {field} for {model_name}: {path}")

    base_cfg = load_config(str(_resolve(model_specs["Backbone-only"]["config"])))
    base_ds = _prepare_dataset(base_cfg, split_name=args.split)
    summary = _sample_summary(base_ds)
    selection = _choose_two_samples(summary)
    selected_ids = set(selection["sample_id"].astype(str))
    selection.to_csv(out_dir / "selected_scenarios.csv", index=False)
    summary.to_csv(out_dir / "candidate_sample_summary.csv", index=False)

    all_results = {
        name: _run_model_for_samples(
            name,
            model_specs=model_specs,
            selected_ids=selected_ids,
            split_name=args.split,
            device=device,
        )
        for name in model_specs
    }

    base_samples = {str(s["sample_id"]): s for s in base_ds.samples if str(s["sample_id"]) in selected_ids}
    case_rows = []
    for _, sel in selection.iterrows():
        sid = str(sel["sample_id"])
        scenario = str(sel["scenario"])
        sample = base_samples[sid]
        target = sample["target_pos"].numpy()
        obs_mask = sample["obs_mask"].numpy()
        gap_pos = _gap_position_from_obs_mask(obs_mask)
        times = sample["times"]
        n = len(obs_mask)

        b = all_results["Backbone-only"][sid]
        a3 = all_results["Ours-A3"][sid]
        bi = all_results["BiLSTM-clean"][sid]
        table = pd.DataFrame(
            {
                "scenario": scenario,
                "sample_id": sid,
                "flight_id": sample["flight_id"],
                "minute_index": np.arange(n, dtype=int),
                "time_utc": times,
                "obs_mask": obs_mask.astype(int),
                "gap_pos_ratio": gap_pos,
                "target_adsb_alt_m": target[:, 2],
                "anchor_alt_m": np.where(obs_mask > 0.5, target[:, 2], np.nan),
                "Backbone_forward_alt_m": b["mu_f"][:, 2],
                "Backbone_backward_alt_m": b["mu_b"][:, 2],
                "Backbone_fusion_alt_m": b["final"][:, 2],
                "Ours_A3_alt_m": a3["final"][:, 2],
                "BiLSTM_alt_m": bi["final"][:, 2],
                "Backbone_fusion_w_forward": b["weights"][:, 0],
                "Backbone_fusion_w_backward": b["weights"][:, 1],
                "Ours_A3_fusion_w_forward": a3["weights"][:, 0],
                "Ours_A3_fusion_w_backward": a3["weights"][:, 1],
                "BiLSTM_fusion_w_forward": bi["weights"][:, 0],
                "BiLSTM_fusion_w_backward": bi["weights"][:, 1],
            }
        )
        for col in [
            "Backbone_forward",
            "Backbone_backward",
            "Backbone_fusion",
            "Ours_A3",
            "BiLSTM",
        ]:
            table[f"{col}_abs_err_m"] = (table[f"{col}_alt_m"] - table["target_adsb_alt_m"]).abs()
        for col in table.columns:
            if col.endswith("_m") or col.endswith("_ratio") or col.endswith("_forward") or col.endswith("_backward"):
                table[col] = pd.to_numeric(table[col], errors="coerce").round(6)

        case_dir = out_dir / scenario
        case_dir.mkdir(parents=True, exist_ok=True)
        table_path = case_dir / "bidirectional_mechanism_points.csv"
        table.to_csv(table_path, index=False)
        _plot_scenario(case_dir, scenario, table)

        gap = table["obs_mask"] <= 0.5
        case_rows.append(
            {
                "scenario": scenario,
                "sample_id": sid,
                "flight_id": sample["flight_id"],
                "length": int(n),
                "anchor_count": int((obs_mask > 0.5).sum()),
                "gap_count": int((obs_mask <= 0.5).sum()),
                "max_gap": int(_max_gap_len(obs_mask)),
                "alt_range_m": float(target[:, 2].max() - target[:, 2].min()),
                "Backbone_forward_gap_mae_m": float(table.loc[gap, "Backbone_forward_abs_err_m"].mean()),
                "Backbone_backward_gap_mae_m": float(table.loc[gap, "Backbone_backward_abs_err_m"].mean()),
                "Backbone_fusion_gap_mae_m": float(table.loc[gap, "Backbone_fusion_abs_err_m"].mean()),
                "Ours_A3_gap_mae_m": float(table.loc[gap, "Ours_A3_abs_err_m"].mean()),
                "BiLSTM_gap_mae_m": float(table.loc[gap, "BiLSTM_abs_err_m"].mean()),
                "table_csv": str(table_path.relative_to(ROOT)),
                "plot_recovery": str((case_dir / "01_altitude_recovery_curve.png").relative_to(ROOT)),
                "plot_mechanism": str((case_dir / "02_bidirectional_error_and_fusion_weight.png").relative_to(ROOT)),
            }
        )

    report = pd.DataFrame(case_rows)
    report.to_csv(out_dir / "bidirectional_mechanism_summary.csv", index=False)
    print(out_dir / "bidirectional_mechanism_summary.csv")
    print(report.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
