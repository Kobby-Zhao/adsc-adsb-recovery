from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.datasets import DatasetConfig, TrajectoryDataset, trajectory_collate_fn
from src.models import TrajectoryRecoveryModel
from src.preprocessing.standardize import apply_standardizer, load_standardizer
from src.training.altitude_governance import add_anchor_alt_features, add_vertical_v2_features
from src.training.coords import build_anchor_alt_tracks, prepare_model_coordinates
from src.training.alt_target import apply_alt_target_transform, invert_alt_target_transform
from src.training.target_norm import denormalize_coords, load_target_stats, normalize_coords
from src.training.utils import load_config, set_seed, split_by_flight_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit altitude learnability and systematic bias patterns.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", default=None)
    return parser


def _find_complete_gaps(anchor_mask_1d: np.ndarray, valid_mask_1d: np.ndarray) -> list[tuple[int, int, int, int]]:
    gaps: list[tuple[int, int, int, int]] = []
    t_len = int(anchor_mask_1d.shape[0])
    t = 0
    while t < t_len:
        if (not bool(valid_mask_1d[t])) or bool(anchor_mask_1d[t]):
            t += 1
            continue
        s = t
        while t < t_len and bool(valid_mask_1d[t]) and (not bool(anchor_mask_1d[t])):
            t += 1
        e = t - 1
        l = s - 1
        r = e + 1
        if l >= 0 and r < t_len and bool(valid_mask_1d[l]) and bool(valid_mask_1d[r]) and bool(anchor_mask_1d[l]) and bool(anchor_mask_1d[r]):
            gaps.append((l, s, e, r))
    return gaps


def _bucket_name(glen: int) -> str:
    if glen <= 3:
        return "1_3"
    if glen <= 8:
        return "4_8"
    if glen <= 15:
        return "9_15"
    if glen <= 30:
        return "16_30"
    return "30_plus"


def _summary_stats(x: np.ndarray) -> dict:
    if x.size == 0:
        return {"count": 0}
    q = np.quantile(x, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "q01": float(q[0]),
        "q05": float(q[1]),
        "q25": float(q[2]),
        "q50": float(q[3]),
        "q75": float(q[4]),
        "q95": float(q[5]),
        "q99": float(q[6]),
        "max": float(np.max(x)),
    }


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
    splits = {k: add_vertical_v2_features(add_anchor_alt_features(v)) for k, v in splits.items()}
    run_dir = Path(cfg["outputs"]["run_dir"])
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / f"alt_audit_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    scaler_stats = load_standardizer(run_dir / "feature_standardizer.json")
    if scaler_stats is None:
        raise RuntimeError("feature_standardizer.json is missing. Run training/evaluation first.")
    scaler_stats = {k: v for k, v in scaler_stats.items() if k not in set(cfg["data"]["obs_cols"])}
    split_df = apply_standardizer(splits[args.split], scaler_stats)

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
    )
    ds = TrajectoryDataset(split_df, dcfg)
    loader = DataLoader(
        ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"]["num_workers"]),
        collate_fn=trajectory_collate_fn,
    )
    device = torch.device(cfg["training"].get("device", "cpu"))
    model = TrajectoryRecoveryModel(
        exo_dim=len(cfg["data"]["exo_cols"]),
        vertical_exo_dim=len(cfg["data"].get("vertical_exo_cols", [])),
        quality_dim=len(cfg["data"]["quality_cols"]),
        hidden_size=int(cfg["model"]["hidden_size"]),
        num_layers=int(cfg["model"].get("num_layers", 1)),
        dropout=float(cfg["model"].get("dropout", 0.0)),
        fusion_hidden_size=int(cfg["model"].get("fusion_hidden_size", 32)),
        fusion_use_exo_quality=bool(cfg["model"].get("fusion_use_exo_quality", False)),
        alt_bias_enabled=bool(cfg["model"].get("alt_bias_enabled", False)),
        alt_bias_hidden_size=int(cfg["model"].get("alt_bias_hidden_size", 32)),
        alt_bias_use_exo_quality=bool(cfg["model"].get("alt_bias_use_exo_quality", True)),
        vertical_projector_enabled=bool(cfg["model"].get("vertical_projector_enabled", False)),
        vertical_projector_hidden_size=int(cfg["model"].get("vertical_projector_hidden_size", 32)),
        vertical_projector_use_vertical_exo=bool(cfg["model"].get("vertical_projector_use_vertical_exo", True)),
        alt_anchor_hard_consistency=bool(cfg["model"].get("alt_anchor_hard_consistency", False)),
    ).to(device)
    ckpt_path = args.checkpoint or str(run_dir / cfg["outputs"]["checkpoint_name"])
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    target_norm_stats = None
    if bool(cfg["training"].get("target_norm", {}).get("enabled", False)):
        target_norm_stats = load_target_stats(run_dir / "target_model_scaler.json")
        if target_norm_stats is None:
            raise RuntimeError("target_norm is enabled but target_model_scaler.json is missing.")

    true_altrel_all: list[np.ndarray] = []
    pred_altrel_all: list[np.ndarray] = []
    err_altrel_all: list[np.ndarray] = []
    point_rows: list[dict] = []
    seg_rows: list[dict] = []

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

            target_model_raw, obs_model_raw, _ = prepare_model_coordinates(
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
            target_model = apply_alt_target_transform(
                target_model_raw,
                mode=alt_target_mode,
                clip_value=alt_target_clip,
            )
            obs_model = apply_alt_target_transform(
                obs_model_raw,
                mode=alt_target_mode,
                clip_value=alt_target_clip,
            )
            target_for_model = normalize_coords(target_model, target_norm_stats)
            obs_for_model = normalize_coords(obs_model, target_norm_stats)
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
                target_pos=None,
                teacher_forcing_ratio=0.0,
            )
            pred_model_t = denormalize_coords(out["pred_pos"], target_norm_stats)
            pred_model = invert_alt_target_transform(
                pred_model_t,
                mode=alt_target_mode,
                clip_value=alt_target_clip,
            )
            err_model = pred_model - target_model_raw

            pred_altrel = pred_model[..., 2].detach().cpu().numpy()
            true_altrel = target_model_raw[..., 2].detach().cpu().numpy()
            err_altrel = err_model[..., 2].detach().cpu().numpy()
            obs_mask_np = obs_mask.detach().cpu().numpy()
            seq_mask_np = seq_mask.detach().cpu().numpy()
            sample_ids = batch["sample_id"]

            for i, sid in enumerate(sample_ids):
                valid = seq_mask_np[i] > 0.5
                gap = (obs_mask_np[i] <= 0.5) & valid
                anchor = (obs_mask_np[i] > 0.5) & valid
                if np.any(gap):
                    true_altrel_all.append(true_altrel[i][gap])
                    pred_altrel_all.append(pred_altrel[i][gap])
                    err_altrel_all.append(err_altrel[i][gap])

                gaps = _find_complete_gaps(anchor_mask_1d=anchor, valid_mask_1d=valid)
                for (_, s, e, _) in gaps:
                    seg_err = err_altrel[i][s : e + 1]
                    if seg_err.size <= 0:
                        continue
                    seg_bias = float(np.mean(seg_err))
                    seg_std = float(np.std(seg_err))
                    seg_abs = np.abs(seg_err)
                    p95 = float(np.quantile(seg_abs, 0.95)) if seg_abs.size > 0 else 0.0
                    outlier_ratio = float(np.mean(seg_abs > max(1.0, p95)))
                    if abs(seg_bias) >= 1.5 * (seg_std + 1e-6):
                        pattern = "stable_bias"
                    elif outlier_ratio >= 0.2:
                        pattern = "spike_outlier"
                    elif seg_std >= 1.5 * (abs(seg_bias) + 1e-6):
                        pattern = "local_jitter"
                    else:
                        pattern = "mixed"
                    seg_rows.append(
                        {
                            "sample_id": sid,
                            "gap_start": int(s),
                            "gap_end": int(e),
                            "gap_len": int(e - s + 1),
                            "bucket": _bucket_name(int(e - s + 1)),
                            "altrel_bias_mean": seg_bias,
                            "altrel_jitter_std": seg_std,
                            "altrel_abs_mae": float(np.mean(seg_abs)),
                            "altrel_abs_rmse": float(np.sqrt(np.mean(seg_err**2))),
                            "pattern": pattern,
                        }
                    )
                    bname = _bucket_name(int(e - s + 1))
                    for t in range(int(s), int(e + 1)):
                        point_rows.append(
                            {
                                "sample_id": sid,
                                "t": int(t),
                                "gap_len": int(e - s + 1),
                                "bucket": bname,
                                "true_alt_rel": float(true_altrel[i, t]),
                                "pred_alt_rel": float(pred_altrel[i, t]),
                                "err_alt_rel": float(err_altrel[i, t]),
                            }
                        )

    true_v = np.concatenate(true_altrel_all) if true_altrel_all else np.array([], dtype=np.float32)
    pred_v = np.concatenate(pred_altrel_all) if pred_altrel_all else np.array([], dtype=np.float32)
    err_v = np.concatenate(err_altrel_all) if err_altrel_all else np.array([], dtype=np.float32)

    corr = float(np.corrcoef(true_v, pred_v)[0, 1]) if true_v.size > 1 and np.std(true_v) > 1e-9 and np.std(pred_v) > 1e-9 else float("nan")
    point_df = pd.DataFrame(point_rows)
    seg_df = pd.DataFrame(seg_rows)
    if not point_df.empty:
        bucket_stats = (
            point_df.groupby("bucket")
            .apply(
                lambda g: pd.Series(
                    {
                        "count": int(len(g)),
                        "altrel_mae": float(np.mean(np.abs(g["err_alt_rel"]))),
                        "altrel_rmse": float(np.sqrt(np.mean(g["err_alt_rel"] ** 2))),
                        "altrel_bias_mean": float(np.mean(g["err_alt_rel"])),
                    }
                )
            )
            .reset_index()
        )
    else:
        bucket_stats = pd.DataFrame(columns=["bucket", "count", "altrel_mae", "altrel_rmse", "altrel_bias_mean"])

    pattern_counts = (
        seg_df["pattern"].value_counts(dropna=False).to_dict()
        if not seg_df.empty
        else {"stable_bias": 0, "local_jitter": 0, "spike_outlier": 0, "mixed": 0}
    )
    summary = {
        "config": args.config,
        "checkpoint": ckpt_path,
        "split": args.split,
        "true_alt_rel_distribution": _summary_stats(true_v),
        "pred_alt_rel_distribution": _summary_stats(pred_v),
        "err_alt_rel_distribution": _summary_stats(err_v),
        "pred_vs_true_alt_rel_corr": corr,
        "pattern_counts": {str(k): int(v) for k, v in pattern_counts.items()},
        "segment_count": int(len(seg_df)),
    }
    (out_dir / f"alt_learnability_summary_{args.split}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    point_df.to_csv(out_dir / f"alt_learnability_points_{args.split}.csv", index=False)
    seg_df.to_csv(out_dir / f"alt_learnability_segments_{args.split}.csv", index=False)
    bucket_stats.to_csv(out_dir / f"alt_learnability_gap_bucket_{args.split}.csv", index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[ok] out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
