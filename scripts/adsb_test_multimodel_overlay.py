from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate import (
    _boundary_alt_from_model_obs,
    _build_flight_track,
    _split_main_and_no_anchor,
    _track_group_from_sample_id,
)
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
from src.training.altitude_governance import add_anchor_alt_features, add_vertical_v2_features
from src.training.coords import build_anchor_alt_tracks, prepare_model_coordinates, restore_to_latlon
from src.training.target_norm import denormalize_coords, load_target_stats, normalize_coords
from src.training.utils import load_config, set_seed, split_by_flight_id


@dataclass
class ModelSpec:
    name: str
    config: str
    checkpoint: str


def _default_models() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="UniLSTM",
            config="configs/alt_focus/backbone_fair_20260331/train_adsb3864_unilstm_baseline_e10_fair_20260331.yaml",
            checkpoint="outputs/runs/fair_train_adsb3864_unilstm_baseline_e10_20260331/best.pt",
        ),
        ModelSpec(
            name="BiLSTM_Baseline",
            config="configs/alt_focus/backbone_fair_20260331/train_adsb3864_bilstm_baseline_e10_fair_20260331.yaml",
            checkpoint="outputs/runs/fair_train_adsb3864_bilstm_baseline_e10_20260331/best.pt",
        ),
        ModelSpec(
            name="Transformer",
            config="configs/alt_focus/backbone_fair_20260331/train_adsb3864_transformer_baseline_e10_fair_20260331.yaml",
            checkpoint="outputs/runs/fair_train_adsb3864_transformer_baseline_e10_20260331/best.pt",
        ),
        ModelSpec(
            name="CNN+LSTM",
            config="configs/alt_focus/backbone_fair_20260331/train_adsb3864_cnnlstm_baseline_e10_fair_20260402.yaml",
            checkpoint="outputs/runs/fair_train_adsb3864_cnnlstm_baseline_e10_20260402/best.pt",
        ),
        ModelSpec(
            name="OurMethod_BiLSTM",
            config="configs/alt_focus/backbone_fair_20260331/train_adsb3864_ourmethod_bilstm_e10_fair_20260331.yaml",
            checkpoint="outputs/runs/fair_train_adsb3864_ourmethod_bilstm_e10_20260331/best.pt",
        ),
    ]


def _parse_track_from_flight_plot(name: str) -> str:
    stem = str(name).strip()
    stem = re.sub(r"\.png$", "", stem)
    stem = re.sub(r"^flight_", "", stem)
    return stem


def _load_track_keys(track_csv: Path, max_tracks: int) -> list[str]:
    if track_csv.exists():
        df = pd.read_csv(track_csv)
        keys = [_parse_track_from_flight_plot(x) for x in df["flight_plot"].astype(str).tolist()]
        keys = [k for k in keys if k]
        if max_tracks > 0:
            keys = keys[:max_tracks]
        return keys
    return []


def _build_loader(cfg: dict, split: str, batch_size_override: int | None = None) -> DataLoader:
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

    split_main_df, _split_no_anchor_df, _split_anchor_audit = _split_main_and_no_anchor(splits[split], split)
    train_main_df, _train_no_anchor_df, _train_anchor_audit = _split_main_and_no_anchor(splits["train"], "train_for_scaler")

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
            train_main_df,
            candidate_cols=candidate_cols,
            exclude_cols={cfg["data"]["obs_mask_col"]},
        )
        scaler_stats = fit_standardizer(train_main_df, feature_cols=continuous_cols)
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
    if len(ds) == 0:
        raise RuntimeError(f"empty split={split} after has_anchor gating")
    bs = int(batch_size_override or cfg["training"]["batch_size"])
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=int(cfg["training"]["num_workers"]),
        collate_fn=trajectory_collate_fn,
    )


def _build_model(cfg: dict, device: torch.device) -> TrajectoryRecoveryModel:
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


def _infer_tracks_for_model(
    spec: ModelSpec,
    split: str,
    track_keys: set[str],
    batch_size_override: int | None = None,
) -> dict[str, dict]:
    cfg = load_config(spec.config)
    set_seed(int(cfg.get("seed", 42)))
    loader = _build_loader(cfg, split=split, batch_size_override=batch_size_override)
    device = torch.device(cfg["training"].get("device", "cpu"))
    model = _build_model(cfg, device=device)
    state = torch.load(spec.checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    target_norm_cfg = cfg["training"].get("target_norm", {})
    target_norm_stats = None
    if bool(target_norm_cfg.get("enabled", False)):
        target_norm_stats = load_target_stats(Path(cfg["outputs"]["run_dir"]) / "target_model_scaler.json")
        if target_norm_stats is None:
            raise RuntimeError("target_norm is enabled but target_model_scaler.json is missing.")
    alt_target_mode = str(cfg["training"].get("alt_target_robust", {}).get("mode", "none")).lower()
    alt_target_clip = float(cfg["training"].get("alt_target_robust", {}).get("clip_value", 3000.0))
    use_segment_teacher = bool(cfg.get("training", {}).get("risk_aware", {}).get("use_segment_teacher", True))
    use_alt_baseline_residual = bool(cfg.get("training", {}).get("risk_aware", {}).get("use_alt_baseline_residual", True))

    sample_plot_cache: dict[str, list[dict]] = {}
    with torch.no_grad():
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
            target_model = apply_alt_target_transform(target_model, mode=alt_target_mode, clip_value=alt_target_clip)
            obs_model = apply_alt_target_transform(obs_model, mode=alt_target_mode, clip_value=alt_target_clip)
            target_for_model = normalize_coords(target_model, target_norm_stats)
            obs_for_model = normalize_coords(obs_model, target_norm_stats)
            left_boundary_alt_model, right_boundary_alt_model = _boundary_alt_from_model_obs(
                obs_for_model=obs_for_model,
                obs_mask=obs_mask,
                seq_mask=seq_mask,
            )
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
            sample_ids = batch["sample_id"]
            flight_ids = batch["flight_id"]
            for i, sid in enumerate(sample_ids):
                track_key = _track_group_from_sample_id(sid)
                if track_keys and track_key not in track_keys:
                    continue
                valid = seq_mask_np[i] > 0.5
                n_valid = int(valid.sum())
                rec = {
                    "sample_id": sid,
                    "flight_id": flight_ids[i],
                    "track_key": track_key,
                    "times": batch["times"][i][:n_valid],
                    "pred": pred_np[i][valid],
                    "target": target_np[i][valid],
                    "obs_mask": obs_mask_np[i][valid],
                }
                sample_plot_cache.setdefault(track_key, []).append(rec)

    merged = {}
    for key, samples in sample_plot_cache.items():
        flight_id = samples[0]["flight_id"] if samples else key
        merged[key] = {"flight_id": flight_id, "track_key": key, "merged": _build_flight_track(samples)}
    return merged


def _plot_overlay(out_dir: Path, track_key: str, by_model: dict[str, dict]) -> bool:
    available = {k: v for k, v in by_model.items() if v and v.get("merged", {}).get("ok", False)}
    if not available:
        return False
    ref = next(iter(available.values()))["merged"]
    ref_times = [str(x) for x in ref["times"]]
    target = ref["target"]
    obs = ref["obs_mask"]
    local_anchor_idx = np.where(obs > 0.5)[0]
    t = np.arange(len(ref_times))
    colors = {
        "UniLSTM": "#1f77b4",
        "BiLSTM_Baseline": "#2ca02c",
        "Transformer": "#ff7f0e",
        "CNN+LSTM": "#9467bd",
        "OurMethod_BiLSTM": "#d62728",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(target[:, 1], target[:, 0], color="#333333", lw=1.5, alpha=0.85, label="GT")
    for name, x in available.items():
        m = x["merged"]
        mt = [str(z) for z in m["times"]]
        idx_map = {ts: i for i, ts in enumerate(mt)}
        # Align every model to the same reference timeline to avoid pseudo-overlap.
        hit = [idx_map.get(ts, None) for ts in ref_times]
        if all(h is None for h in hit):
            continue
        pred = np.asarray(
            [m["pred"][h] if h is not None else [np.nan, np.nan, np.nan] for h in hit],
            dtype=np.float64,
        )
        axes[0].plot(pred[:, 1], pred[:, 0], lw=1.6, alpha=0.95, color=colors.get(name), label=name)
    if len(local_anchor_idx) > 0:
        axes[0].scatter(target[local_anchor_idx, 1], target[local_anchor_idx, 0], s=18, color="#000000", alpha=0.75, label="Anchor")
    axes[0].set_xlabel("Lon")
    axes[0].set_ylabel("Lat")
    axes[0].set_title("Lat/Lon Overlay")
    axes[0].legend(fontsize=8)

    axes[1].plot(t, target[:, 2], color="#333333", lw=1.5, alpha=0.85, label="GT Alt")
    for name, x in available.items():
        m = x["merged"]
        mt = [str(z) for z in m["times"]]
        idx_map = {ts: i for i, ts in enumerate(mt)}
        hit = [idx_map.get(ts, None) for ts in ref_times]
        if all(h is None for h in hit):
            continue
        pred = np.asarray(
            [m["pred"][h] if h is not None else [np.nan, np.nan, np.nan] for h in hit],
            dtype=np.float64,
        )
        axes[1].plot(t, pred[:, 2], lw=1.6, alpha=0.95, color=colors.get(name), label=name)
    if len(local_anchor_idx) > 0:
        axes[1].scatter(t[local_anchor_idx], target[local_anchor_idx, 2], s=18, color="#000000", alpha=0.75, label="Anchor")
    axes[1].set_xlabel("Minute Index")
    axes[1].set_ylabel("Altitude")
    axes[1].set_title("Altitude Overlay")
    axes[1].legend(fontsize=8)

    fig.suptitle(f"ADS-B Test Overlay | {track_key}", fontsize=11)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out_path = out_dir / f"overlay_{track_key.replace('/', '_')}.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return True


def main() -> int:
    p = argparse.ArgumentParser(description="Overlay multiple model recoveries on same ADS-B test samples.")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--out-dir", default="outputs/runs/adsb_test_multimodel_overlay_20260412")
    p.add_argument(
        "--track-csv",
        default="outputs/runs/adsb_test_baseline_gallery_20260412/common_flight_plots_for_compare.csv",
    )
    p.add_argument("--max-tracks", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()

    out_dir = ROOT / args.out_dir
    plots_dir = out_dir / "plots_overlay_same_axes"
    plots_dir.mkdir(parents=True, exist_ok=True)

    track_keys = _load_track_keys(ROOT / args.track_csv, max_tracks=int(args.max_tracks))
    if not track_keys:
        raise RuntimeError("no track keys found")
    track_set = set(track_keys)

    specs = _default_models()
    per_model_track = {}
    for spec in specs:
        print(f"[run] {spec.name}")
        per_model_track[spec.name] = _infer_tracks_for_model(
            spec,
            split=args.split,
            track_keys=track_set,
            batch_size_override=int(args.batch_size),
        )

    rows = []
    for tk in track_keys:
        by_model = {name: per_model_track.get(name, {}).get(tk, {}) for name in per_model_track}
        ok = _plot_overlay(plots_dir, tk, by_model)
        rows.append(
            {
                "track_key": tk,
                "plotted": bool(ok),
                "models_ok": int(
                    sum(1 for _name, x in by_model.items() if x and x.get("merged", {}).get("ok", False))
                ),
            }
        )

    pd.DataFrame(rows).to_csv(out_dir / "overlay_plot_index.csv", index=False)
    print(f"[done] plots_dir={plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
