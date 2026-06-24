from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.datasets import DatasetConfig, TrajectoryDataset, trajectory_collate_fn
from src.models import TrajectoryRecoveryModel
from src.preprocessing.standardize import (
    apply_standardizer,
    fit_standardizer,
    load_standardizer,
    save_standardizer,
    select_continuous_feature_cols,
)
from src.training.alt_target import apply_alt_target_transform, invert_alt_target_transform
from src.training.coords import build_anchor_alt_tracks, prepare_model_coordinates, restore_to_latlon
from src.training.target_norm import denormalize_coords, load_target_stats, normalize_coords
from src.training.utils import load_config, set_seed
from scripts.evaluate import _boundary_alt_from_model_obs, _split_main_and_no_anchor
from src.training.altitude_governance import add_anchor_alt_features, add_vertical_v2_features


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", required=True)
    p.add_argument("--selected", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def default_models() -> list[dict[str, str]]:
    base = "outputs/experiments/curriculum_20260415_exp4cmp_s2v2"
    cfg = "configs/alt_focus/curriculum_20260415_exp4cmp_s2v2"
    return [
        {"name": "ourmethod", "config": f"{cfg}/exp_cur_proposed_24e.yaml", "ckpt": f"{base}/exp_cur_proposed_24e/best.pt"},
        {"name": "unilstm", "config": f"{cfg}/exp_cur_unilstm_baseline_24e.yaml", "ckpt": f"{base}/exp_cur_unilstm_baseline_24e/best.pt"},
        {"name": "bilstm", "config": f"{cfg}/exp_cur_bilstm_baseline_24e.yaml", "ckpt": f"{base}/exp_cur_bilstm_baseline_24e/best.pt"},
        {"name": "cnnlstm", "config": f"{cfg}/exp_cur_cnnlstm_baseline_24e.yaml", "ckpt": f"{base}/exp_cur_cnnlstm_baseline_24e/best.pt"},
        {"name": "transformer", "config": f"{cfg}/exp_cur_transformer_baseline_24e.yaml", "ckpt": f"{base}/exp_cur_transformer_baseline_24e/best.pt"},
        {"name": "kalman", "config": f"{cfg}/exp_cur_kalman_filter_baseline_24e.yaml", "ckpt": f"{base}/exp_cur_kalman_filter_baseline_24e/best.pt"},
    ]


def build_model(cfg: dict[str, Any], device: torch.device) -> TrajectoryRecoveryModel:
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
        alt_main_mode=str(cfg["model"].get("alt_main_mode", "absolute")),
        main_rmax_ft=float(cfg["model"].get("main_rmax_ft", 500.0)),
        v3_anchor_hard_consistency=bool(cfg["model"].get("v3_anchor_hard_consistency", True)),
        v3_edge_residual_damp_enabled=bool(cfg["model"].get("v3_edge_residual_damp_enabled", True)),
        v3_edge_residual_damp_strength=float(cfg["model"].get("v3_edge_residual_damp_strength", 0.7)),
        v3_edge_residual_damp_steps=int(cfg["model"].get("v3_edge_residual_damp_steps", 2)),
    ).to(device)
    return model


def infer_model(model_name: str, cfg_path: str, ckpt_path: str, in_df: pd.DataFrame, device: torch.device) -> tuple[dict[str, dict[str, np.ndarray]], pd.DataFrame]:
    cfg = load_config(cfg_path)
    cfg["data"]["samples_path"] = "<in_memory>"

    df = add_anchor_alt_features(in_df.copy())
    df = add_vertical_v2_features(df)
    split_main_df, _, _ = _split_main_and_no_anchor(df, f"{model_name}_all")

    run_dir = Path(cfg["outputs"]["run_dir"])
    scaler_path = run_dir / "feature_standardizer.json"
    scaler_stats = load_standardizer(scaler_path)
    if scaler_stats is None:
        candidate_cols = list(dict.fromkeys(["dt_prev", "dt_next"] + cfg["data"]["exo_cols"] + cfg["data"].get("vertical_exo_cols", []) + cfg["data"]["quality_cols"]))
        continuous_cols = select_continuous_feature_cols(split_main_df, candidate_cols=candidate_cols, exclude_cols={cfg["data"]["obs_mask_col"]})
        scaler_stats = fit_standardizer(split_main_df, feature_cols=continuous_cols)
        save_standardizer(scaler_stats, scaler_path)
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
    loader = DataLoader(ds, batch_size=min(64, int(cfg["training"].get("batch_size", 32))), shuffle=False, num_workers=0, collate_fn=trajectory_collate_fn)

    model = build_model(cfg, device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()

    target_norm_cfg = cfg["training"].get("target_norm", {})
    target_norm_stats = None
    if bool(target_norm_cfg.get("enabled", False)):
        target_norm_stats = load_target_stats(Path(cfg["outputs"]["run_dir"]) / "target_model_scaler.json")

    rows = []
    curves: dict[str, dict[str, np.ndarray]] = {}
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
            edge_weight = batch["edge_weight"].to(device) if ("edge_weight" in batch and use_segment_teacher) else None
            residual_rmax_ft = batch["residual_rmax_ft"].to(device) if ("residual_rmax_ft" in batch and use_alt_baseline_residual) else None
            gate_bias = batch["gate_bias"].to(device) if ("gate_bias" in batch and use_segment_teacher) else None
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
            target_model = apply_alt_target_transform(target_model, mode=alt_target_mode, clip_value=alt_target_clip)
            obs_model = apply_alt_target_transform(obs_model, mode=alt_target_mode, clip_value=alt_target_clip)
            target_for_model = normalize_coords(target_model, target_norm_stats)
            obs_for_model = normalize_coords(obs_model, target_norm_stats)
            left_boundary_alt_model, right_boundary_alt_model = _boundary_alt_from_model_obs(obs_for_model=obs_for_model, obs_mask=obs_mask, seq_mask=seq_mask)
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
                risk_flag=risk_flag,
                teacher_scale=teacher_scale,
                risk_flag_teacher=risk_flag_teacher,
                segment_bucket=segment_bucket,
                edge_weight=edge_weight,
                residual_rmax_ft=residual_rmax_ft,
                gate_bias=gate_bias,
                left_boundary_alt=left_boundary_alt_model,
                right_boundary_alt=right_boundary_alt_model,
                target_pos=None,
                teacher_forcing_ratio=0.0,
            )

            pred_model_t = denormalize_coords(out["pred_pos"], target_norm_stats)
            pred_model = invert_alt_target_transform(pred_model_t, mode=alt_target_mode, clip_value=alt_target_clip)
            pred_latlon = restore_to_latlon(pred_model, seq_mask=seq_mask, ctx=coord_ctx)

            pred_np = pred_latlon.detach().cpu().numpy()
            target_np = target_pos.detach().cpu().numpy()
            obs_mask_np = batch["obs_mask"].detach().cpu().numpy()
            seq_mask_np = batch["seq_mask"].detach().cpu().numpy()

            for i, sid in enumerate(batch["sample_id"]):
                valid = seq_mask_np[i] > 0.5
                pred_i = pred_np[i][valid]
                tgt_i = target_np[i][valid]
                obs_i = obs_mask_np[i][valid]
                gap_i = obs_i <= 0.5
                gap_alt_rmse = float(np.sqrt(np.mean((pred_i[gap_i, 2] - tgt_i[gap_i, 2]) ** 2))) if gap_i.any() else np.nan
                alt_rmse = float(np.sqrt(np.mean((pred_i[:, 2] - tgt_i[:, 2]) ** 2)))
                fid = str(batch["flight_id"][i])
                curves[fid] = {
                    "pred": pred_i,
                    "target": tgt_i,
                    "obs_mask": obs_i,
                    "times": np.asarray(batch["times"][i][: int(valid.sum())]),
                }
                rows.append({"model": model_name, "sample_id": sid, "flight_id": fid, "alt_rmse": alt_rmse, "gap_alt_rmse": gap_alt_rmse})

    return curves, pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(42)

    selected = pd.read_csv(args.selected)
    flight_ids = set(selected["flight_id"].astype(str).tolist())
    df = pd.read_parquet(args.samples)
    df = df[df["flight_id"].astype(str).isin(flight_ids)].copy()

    device = torch.device(args.device)
    all_curves: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    all_metrics = []

    for spec in default_models():
        curves, mdf = infer_model(spec["name"], spec["config"], spec["ckpt"], df, device)
        all_curves[spec["name"]] = curves
        all_metrics.append(mdf)
        print(f"[ok] {spec['name']} inferred flights={len(curves)}")

    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(out_dir / "per_flight_metrics_all_models.csv", index=False)
    piv = metrics.pivot_table(index="flight_id", columns="model", values="gap_alt_rmse")
    piv.to_csv(out_dir / "gap_alt_rmse_compare.csv")

    colors = {
        "ourmethod": "#d62728",
        "unilstm": "#1f77b4",
        "bilstm": "#2ca02c",
        "cnnlstm": "#9467bd",
        "transformer": "#17becf",
        "kalman": "#8c564b",
    }

    for fid in sorted(flight_ids):
        base = all_curves["ourmethod"].get(fid)
        if base is None:
            continue
        target = base["target"]
        obs = base["obs_mask"]
        t = np.arange(len(target))
        anchor_idx = np.where(obs > 0.5)[0]

        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.plot(t, target[:, 2], color="black", linewidth=1.8, label="GT", alpha=0.9)
        if len(anchor_idx) > 0:
            ax.scatter(t[anchor_idx], target[anchor_idx, 2], s=14, color="black", alpha=0.7, label="Anchor")

        for m in ["ourmethod", "unilstm", "bilstm", "cnnlstm", "transformer", "kalman"]:
            c = all_curves.get(m, {}).get(fid)
            if c is None:
                continue
            ax.plot(t, c["pred"][:, 2], linestyle="--", linewidth=1.4, color=colors[m], label=m)

        ax.set_title(f"{fid} | minute-level recovery (pred dashed)")
        ax.set_xlabel("Minute index")
        ax.set_ylabel("Altitude")
        ax.legend(ncol=4, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"compare_alt_{fid}.png", dpi=160, transparent=True)
        plt.close(fig)

        fig2, ax2 = plt.subplots(1, 1, figsize=(5.5, 5.5))
        ax2.plot(target[:, 1], target[:, 0], color="black", linewidth=1.8, label="GT")
        if len(anchor_idx) > 0:
            ax2.scatter(target[anchor_idx, 1], target[anchor_idx, 0], s=12, color="black", alpha=0.7, label="Anchor")
        for m in ["ourmethod", "unilstm", "bilstm", "cnnlstm", "transformer", "kalman"]:
            c = all_curves.get(m, {}).get(fid)
            if c is None:
                continue
            ax2.plot(c["pred"][:, 1], c["pred"][:, 0], linestyle="--", linewidth=1.2, color=colors[m], label=m)
        ax2.set_title(f"{fid} | 2D trajectory (pred dashed)")
        ax2.set_xlabel("Lon")
        ax2.set_ylabel("Lat")
        ax2.legend(fontsize=7, ncol=2)
        fig2.tight_layout()
        fig2.savefig(out_dir / f"compare_2d_{fid}.png", dpi=160, transparent=True)
        plt.close(fig2)

    print(f"[ok] out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
