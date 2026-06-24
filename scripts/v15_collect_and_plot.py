from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.datasets import DatasetConfig, TrajectoryDataset, trajectory_collate_fn
from src.models import TrajectoryRecoveryModel
from src.preprocessing.standardize import apply_standardizer, fit_standardizer, load_standardizer, save_standardizer, select_continuous_feature_cols
from src.training.alt_target import apply_alt_target_transform, invert_alt_target_transform
from src.training.altitude_governance import add_anchor_alt_features, add_vertical_v2_features
from src.training.coords import build_anchor_alt_tracks, prepare_model_coordinates, restore_to_latlon
from src.training.target_norm import denormalize_coords, load_target_stats, normalize_coords
from src.training.utils import load_config, set_seed, split_by_flight_id


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Collect alt-rel points and plots for v1/v1.5 comparison.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--tag", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-scatter-points", type=int, default=80000)
    return p


def _sample_has_anchor(frame: pd.DataFrame) -> pd.Series:
    obs = pd.to_numeric(frame.get("obs_mask", 0.0), errors="coerce").fillna(0.0)
    return frame.assign(_obs_anchor=(obs > 0.5)).groupby("sample_id")["_obs_anchor"].any().astype(bool)


def _main_anchor_gate(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    has_anchor = _sample_has_anchor(frame)
    keep_ids = set(has_anchor[has_anchor].index.astype(str).tolist())
    out = frame[frame["sample_id"].astype(str).isin(keep_ids)].copy()
    rem = _sample_has_anchor(out)
    if len(rem) == 0 or bool((~rem).any()):
        raise RuntimeError(f"FATAL: main task split={split_name} contains has_anchor=false after gating.")
    return out


def _max_gap(obs: np.ndarray) -> int:
    best = 0
    cur = 0
    for v in obs:
        if v < 0.5:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    main_split_df = _main_anchor_gate(splits[args.split], split_name=args.split)
    train_main_df = _main_anchor_gate(splits["train"], split_name="train_for_scaler")

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
            train_main_df, candidate_cols=candidate_cols, exclude_cols={cfg["data"]["obs_mask_col"]}
        )
        scaler_stats = fit_standardizer(train_main_df, feature_cols=continuous_cols)
        save_standardizer(scaler_stats, scaler_path)
    scaler_stats = {k: v for k, v in scaler_stats.items() if k not in set(cfg["data"]["obs_cols"])}
    main_split_df = apply_standardizer(main_split_df, scaler_stats)

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
    ds = TrajectoryDataset(main_split_df, dcfg)
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
        vertical_tune_enabled=bool(cfg["model"].get("vertical_tune_enabled", False)),
        vertical_tune_hidden_size=int(cfg["model"].get("vertical_tune_hidden_size", 16)),
        vertical_tune_temperature=float(cfg["model"].get("vertical_tune_temperature", 1.0)),
        vertical_tune_mode=str(cfg["model"].get("vertical_tune_mode", "combined")),
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    target_norm_cfg = cfg["training"].get("target_norm", {})
    target_norm_stats = None
    if bool(target_norm_cfg.get("enabled", False)):
        target_norm_stats = load_target_stats(Path(cfg["outputs"]["run_dir"]) / "target_model_scaler.json")
        if target_norm_stats is None:
            raise RuntimeError("target_norm enabled but scaler missing.")

    alt_target_mode = str(cfg["training"].get("alt_target_robust", {}).get("mode", "none")).lower()
    alt_target_clip = float(cfg["training"].get("alt_target_robust", {}).get("clip_value", 3000.0))

    point_rows: list[dict] = []
    rep_cache: list[dict] = []
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
            target_model_raw = target_model
            target_model_t = apply_alt_target_transform(target_model, mode=alt_target_mode, clip_value=alt_target_clip)
            obs_model_t = apply_alt_target_transform(obs_model, mode=alt_target_mode, clip_value=alt_target_clip)

            target_for_model = normalize_coords(target_model_t, target_norm_stats)
            obs_for_model = normalize_coords(obs_model_t, target_norm_stats)
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
                return_vertical_tune_weights=True,
            )
            pred_model_t = denormalize_coords(out["pred_pos"], target_norm_stats)
            pred_model = invert_alt_target_transform(pred_model_t, mode=alt_target_mode, clip_value=alt_target_clip)
            pred_latlon = restore_to_latlon(pred_model, seq_mask=seq_mask, ctx=coord_ctx)

            pred_rel_np = pred_model.detach().cpu().numpy()
            true_rel_np = target_model_raw.detach().cpu().numpy()
            pred_abs_np = pred_latlon.detach().cpu().numpy()
            true_abs_np = target_pos.detach().cpu().numpy()
            obs_mask_np = obs_mask.detach().cpu().numpy()
            seq_mask_np = seq_mask.detach().cpu().numpy()
            w_np = out.get("vertical_tune_weights")
            w_np = w_np.detach().cpu().numpy() if w_np is not None else None
            dt_prev_np = dt_prev.detach().cpu().numpy()
            dt_next_np = dt_next.detach().cpu().numpy()

            for i, sid in enumerate(batch["sample_id"]):
                valid = seq_mask_np[i] > 0.5
                obs_i = obs_mask_np[i][valid]
                gap_i = obs_i <= 0.5
                pred_rel_i = pred_rel_np[i][valid]
                true_rel_i = true_rel_np[i][valid]
                pred_abs_i = pred_abs_np[i][valid]
                true_abs_i = true_abs_np[i][valid]
                dtp_i = dt_prev_np[i][valid]
                dtn_i = dt_next_np[i][valid]
                max_gap = _max_gap(obs_i)
                rep_cache.append(
                    {
                        "sample_id": sid,
                        "flight_id": batch["flight_id"][i],
                        "obs": obs_i,
                        "pred_abs": pred_abs_i,
                        "true_abs": true_abs_i,
                        "weights": None if w_np is None else w_np[i][valid],
                        "max_gap": max_gap,
                    }
                )
                gap_idx = np.where(gap_i)[0]
                for t in gap_idx:
                    row = {
                        "tag": args.tag,
                        "sample_id": sid,
                        "flight_id": batch["flight_id"][i],
                        "t_idx": int(t),
                        "gap_len": float(dtp_i[t] + dtn_i[t]),
                        "pred_alt_rel": float(pred_rel_i[t, 2]),
                        "true_alt_rel": float(true_rel_i[t, 2]),
                        "abs_alt_error": float(abs(pred_abs_i[t, 2] - true_abs_i[t, 2])),
                    }
                    if w_np is not None:
                        row["w_left"] = float(w_np[i][valid][t, 0])
                        row["w_right"] = float(w_np[i][valid][t, 1])
                        row["w_pos"] = float(w_np[i][valid][t, 2])
                    point_rows.append(row)

    point_df = pd.DataFrame(point_rows)
    point_csv = out_dir / f"{args.tag}_altrel_points.csv"
    point_df.to_csv(point_csv, index=False)

    pred = point_df["pred_alt_rel"].to_numpy(dtype=np.float64)
    true = point_df["true_alt_rel"].to_numpy(dtype=np.float64)
    corr = float(np.corrcoef(pred, true)[0, 1]) if len(pred) > 1 else float("nan")
    pred_std = float(np.std(pred)) if len(pred) > 0 else float("nan")
    true_std = float(np.std(true)) if len(true) > 0 else float("nan")
    summary = {
        "tag": args.tag,
        "n_gap_points": int(len(point_df)),
        "altrel_corr": corr,
        "pred_alt_rel_std": pred_std,
        "true_alt_rel_std": true_std,
        "std_ratio": float(pred_std / (true_std + 1e-9)) if np.isfinite(true_std) else float("nan"),
    }
    (out_dir / f"{args.tag}_altrel_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # 1) scatter
    fig, ax = plt.subplots(figsize=(5, 5))
    if len(point_df) > args.max_scatter_points:
        sdf = point_df.sample(n=args.max_scatter_points, random_state=42)
    else:
        sdf = point_df
    ax.scatter(sdf["true_alt_rel"], sdf["pred_alt_rel"], s=3, alpha=0.25)
    lo = float(min(sdf["true_alt_rel"].min(), sdf["pred_alt_rel"].min()))
    hi = float(max(sdf["true_alt_rel"].max(), sdf["pred_alt_rel"].max()))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("true_alt_rel")
    ax.set_ylabel("pred_alt_rel")
    ax.set_title(f"{args.tag} corr={corr:.3f} std_ratio={pred_std/(true_std+1e-9):.3f}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{args.tag}_scatter_altrel.png", dpi=150)
    plt.close(fig)

    # 2) std bar (single run)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["pred_std", "true_std"], [pred_std, true_std], color=["#d95f02", "#1b9e77"])
    ax.set_title(f"{args.tag} alt_rel std")
    fig.tight_layout()
    fig.savefig(out_dir / f"{args.tag}_std_bar.png", dpi=150)
    plt.close(fig)

    # 3) representative 30+ profile
    cands = [x for x in rep_cache if x["max_gap"] >= 30]
    if cands:
        cands = sorted(cands, key=lambda x: x["max_gap"], reverse=True)
        rep = cands[0]
        t = np.arange(len(rep["obs"]))
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.plot(t, rep["true_abs"][:, 2], label="true_alt")
        ax.plot(t, rep["pred_abs"][:, 2], label="pred_alt")
        anc = np.where(rep["obs"] > 0.5)[0]
        if len(anc) > 0:
            ax.scatter(anc, rep["true_abs"][anc, 2], s=12, c="k", alpha=0.6, label="anchor")
        ax.set_title(f"{args.tag} rep30+ sample={rep['sample_id']} max_gap={rep['max_gap']}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{args.tag}_rep30p_alt_profile.png", dpi=150)
        plt.close(fig)

        # 4) gate weights along gap if available
        if rep["weights"] is not None:
            gap_idx = np.where(rep["obs"] <= 0.5)[0]
            if len(gap_idx) > 0:
                g = rep["weights"][gap_idx]
                r = np.linspace(0.0, 1.0, num=len(gap_idx))
                fig, ax = plt.subplots(figsize=(8, 3.2))
                ax.plot(r, g[:, 0], label="w_left")
                ax.plot(r, g[:, 1], label="w_right")
                ax.plot(r, g[:, 2], label="w_pos")
                ax.set_xlabel("relative gap position")
                ax.set_ylabel("weight")
                ax.set_title(f"{args.tag} adaptive weights on rep30+")
                ax.legend()
                fig.tight_layout()
                fig.savefig(out_dir / f"{args.tag}_rep30p_weight_curve.png", dpi=150)
                plt.close(fig)

    print(f"[ok] point_csv={point_csv}")
    print(f"[ok] summary_json={out_dir / f'{args.tag}_altrel_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
