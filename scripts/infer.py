from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.datasets import DatasetConfig, TrajectoryDataset, trajectory_collate_fn
from src.inference import TrajectoryInferencer
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
from src.training.utils import load_config, set_seed, split_by_flight_id, validate_inference_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run trajectory inference.")
    parser.add_argument("--config", default="configs/infer.yaml")
    parser.add_argument("--checkpoint", default="outputs/runs/train_default/best.pt")
    parser.add_argument("--split", default=None, choices=["train", "val", "test"])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    split_name = args.split or cfg["inference"].get("split", "test")
    out_csv = Path(cfg["inference"]["output_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)

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
    run_dir = Path(cfg["outputs"]["run_dir"])
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
            splits["train"],
            candidate_cols=candidate_cols,
            exclude_cols={cfg["data"]["obs_mask_col"]},
        )
        scaler_stats = fit_standardizer(splits["train"], feature_cols=continuous_cols)
        save_standardizer(scaler_stats, scaler_path)
        print(f"[norm] scaler missing, fitted from train split and saved to {scaler_path}")
    scaler_stats = {k: v for k, v in scaler_stats.items() if k not in set(cfg["data"]["obs_cols"])}
    splits[split_name] = apply_standardizer(splits[split_name], scaler_stats)

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
    )

    infer_cols_raw = (
        [cfg["data"]["sample_id_col"], cfg["data"]["flight_id_col"], cfg["data"]["time_col"], cfg["data"]["obs_mask_col"]]
        + cfg["data"]["obs_cols"]
        + ["dt_prev", "dt_next"]
        + cfg["data"]["exo_cols"]
        + cfg["data"].get("vertical_exo_cols", [])
        + cfg["data"]["quality_cols"]
    )
    infer_cols = list(dict.fromkeys(infer_cols_raw))
    infer_frame = splits[split_name][infer_cols].copy()
    validate_inference_frame(infer_frame, cfg)

    ds = TrajectoryDataset(infer_frame, dcfg)
    if len(ds) == 0:
        raise RuntimeError(f"{split_name} split is empty.")

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
        minimal_task_adapt_baseline=bool(cfg["model"].get("minimal_task_adapt_baseline", False)),
        alt_bias_enabled=bool(cfg["model"].get("alt_bias_enabled", False)),
        alt_bias_hidden_size=int(cfg["model"].get("alt_bias_hidden_size", 32)),
        alt_bias_use_exo_quality=bool(cfg["model"].get("alt_bias_use_exo_quality", True)),
        vertical_projector_enabled=bool(cfg["model"].get("vertical_projector_enabled", False)),
        vertical_projector_hidden_size=int(cfg["model"].get("vertical_projector_hidden_size", 32)),
        vertical_projector_use_vertical_exo=bool(cfg["model"].get("vertical_projector_use_vertical_exo", True)),
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
        alt_anchor_hard_consistency=bool(cfg["model"].get("alt_anchor_hard_consistency", False)),
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
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    inferencer = TrajectoryInferencer(model=model, device=device)
    coord_mode = str(cfg["model"].get("coord_mode", "latlon"))
    target_norm_cfg = cfg["training"].get("target_norm", {})
    target_norm_stats = None
    if bool(target_norm_cfg.get("enabled", False)):
        target_norm_stats = load_target_stats(Path(cfg["outputs"]["run_dir"]) / "target_model_scaler.json")
        if target_norm_stats is None:
            raise RuntimeError("target_norm is enabled but target_model_scaler.json is missing.")

    rows = []
    for batch in loader:
        target_dummy = torch.zeros_like(batch["obs_pos"].to(device))
        seq_mask = batch["seq_mask"].to(device)
        _, obs_model_raw, coord_ctx = prepare_model_coordinates(
            target_pos=target_dummy,
            obs_pos=batch["obs_pos"].to(device),
            obs_mask=batch["obs_mask"].to(device),
            seq_mask=seq_mask,
            mode=coord_mode,
            allow_target_fallback=False,
            u_relative_anchor=bool(cfg["model"].get("u_relative_anchor", False)),
            en_relative_anchor=bool(cfg["model"].get("en_relative_anchor", True)),
            en_incremental=bool(cfg["model"].get("en_incremental", False)),
        )
        alt_target_mode = str(cfg["training"].get("alt_target_robust", {}).get("mode", "none")).lower()
        alt_target_clip = float(cfg["training"].get("alt_target_robust", {}).get("clip_value", 3000.0))
        obs_model_t = apply_alt_target_transform(
            obs_model_raw,
            mode=alt_target_mode,
            clip_value=alt_target_clip,
        )
        obs_model = normalize_coords(obs_model_t, target_norm_stats)
        anchor_left_raw, anchor_right_raw = build_anchor_pair_tracks(
            obs_pos=batch["obs_pos"].to(device),
            obs_mask=batch["obs_mask"].to(device),
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
        anchor_alt = build_anchor_alt_tracks(
            obs_pos=batch["obs_pos"].to(device),
            obs_mask=batch["obs_mask"].to(device),
            seq_mask=seq_mask,
        )
        batch_model = dict(batch)
        batch_model["obs_pos"] = obs_model
        batch_model["anchor_alt"] = anchor_alt
        batch_model["anchor_left"] = anchor_left_model
        batch_model["anchor_right"] = anchor_right_model
        pred_model = inferencer.predict_batch(batch_model, interpolate=True)["pred_pos"]
        pred_tensor = torch.tensor(pred_model, device=device)
        pred_tensor_t = denormalize_coords(pred_tensor, target_norm_stats)
        pred_tensor_raw = invert_alt_target_transform(
            pred_tensor_t,
            mode=alt_target_mode,
            clip_value=alt_target_clip,
        )
        pred_latlon = restore_to_latlon(pred_tensor_raw, seq_mask=seq_mask, ctx=coord_ctx).detach().cpu().numpy()
        lengths = batch["lengths"].tolist()
        for i, n in enumerate(lengths):
            sid = batch["sample_id"][i]
            fid = batch["flight_id"][i]
            times = batch["times"][i]
            for t in range(n):
                rows.append(
                    {
                        "sample_id": sid,
                        "flight_id": fid,
                        "minute_ts": times[t],
                        "pred_lat": float(pred_latlon[i, t, 0]),
                        "pred_lon": float(pred_latlon[i, t, 1]),
                        "pred_alt": float(pred_latlon[i, t, 2]),
                    }
                )

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[ok] wrote={out_csv} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
