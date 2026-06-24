from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.datasets import DatasetConfig, TrajectoryDataset, trajectory_collate_fn
from src.models import TrajectoryRecoveryModel
from src.preprocessing.standardize import apply_standardizer, fit_standardizer, select_continuous_feature_cols
from src.training import load_config, set_seed, split_by_flight_id
from src.training.coords import prepare_model_coordinates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report anchor/gap loss mix by batch.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", default="", help="Optional checkpoint path for model predictions.")
    parser.add_argument("--max-batches", type=int, default=0, help="0 means all batches.")
    parser.add_argument(
        "--out-csv",
        default="outputs/reports/batch_anchor_gap_ratio.csv",
        help="Where to save per-batch ratio report.",
    )
    return parser


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
    candidate_cols = list(dict.fromkeys(["dt_prev", "dt_next"] + cfg["data"]["exo_cols"] + cfg["data"]["quality_cols"]))
    continuous_cols = select_continuous_feature_cols(
        splits["train"],
        candidate_cols=candidate_cols,
        exclude_cols={cfg["data"]["obs_mask_col"]},
    )
    scaler_stats = fit_standardizer(splits["train"], feature_cols=continuous_cols)
    splits[args.split] = apply_standardizer(splits[args.split], scaler_stats)
    frame = splits[args.split]
    if frame.empty:
        raise RuntimeError(f"{args.split} split is empty.")

    dcfg = DatasetConfig(
        sample_id_col=cfg["data"]["sample_id_col"],
        flight_id_col=cfg["data"]["flight_id_col"],
        time_col=cfg["data"]["time_col"],
        target_cols=cfg["data"]["target_cols"],
        obs_cols=cfg["data"]["obs_cols"],
        obs_mask_col=cfg["data"]["obs_mask_col"],
        exo_cols=cfg["data"]["exo_cols"],
        quality_cols=cfg["data"]["quality_cols"],
    )
    ds = TrajectoryDataset(frame, dcfg)
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
        quality_dim=len(cfg["data"]["quality_cols"]),
        hidden_size=int(cfg["model"]["hidden_size"]),
        num_layers=int(cfg["model"].get("num_layers", 1)),
        dropout=float(cfg["model"].get("dropout", 0.0)),
        alt_anchor_hard_consistency=bool(cfg["model"].get("alt_anchor_hard_consistency", False)),
    ).to(device)
    model.eval()

    ckpt_path = args.checkpoint.strip()
    if ckpt_path:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        print(f"[info] loaded checkpoint: {ckpt_path}")
    else:
        print("[warn] no checkpoint provided, using randomly initialized model.")

    anchor_w = float(cfg["loss"]["anchor_weight"])
    gap_w = float(cfg["loss"]["gap_weight"])
    coord_mode = str(cfg["model"].get("coord_mode", "latlon"))
    smooth_l1 = torch.nn.SmoothL1Loss(reduction="none")

    rows: list[dict] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break

            obs_pos = batch["obs_pos"].to(device)
            obs_mask = batch["obs_mask"].to(device)
            dt_prev = batch["dt_prev"].to(device)
            dt_next = batch["dt_next"].to(device)
            exo = batch["exo"].to(device)
            quality = batch["quality"].to(device)
            global_quality = batch["global_quality"].to(device)
            target_pos = batch["target_pos"].to(device)
            seq_mask = batch["seq_mask"].to(device)

            target_model, obs_model, _ = prepare_model_coordinates(
                target_pos=target_pos,
                obs_pos=obs_pos,
                obs_mask=obs_mask,
                seq_mask=seq_mask,
                mode=coord_mode,
            )
            out = model(
                obs_pos=obs_model,
                obs_mask=obs_mask,
                dt_prev=dt_prev,
                dt_next=dt_next,
                exo=exo,
                quality=quality,
                global_quality=global_quality,
                target_pos=None,
                teacher_forcing_ratio=0.0,
            )
            per_t = smooth_l1(out["pred_pos"], target_model).mean(dim=-1)
            valid = seq_mask
            anchor_mask = (obs_mask > 0.5).float() * valid
            gap_mask = (obs_mask <= 0.5).float() * valid

            anchor_weighted = (per_t * anchor_mask * anchor_w).sum()
            gap_weighted = (per_t * gap_mask * gap_w).sum()
            total_weighted = anchor_weighted + gap_weighted + 1e-9

            anchor_raw = (per_t * anchor_mask).sum()
            gap_raw = (per_t * gap_mask).sum()
            total_raw = anchor_raw + gap_raw + 1e-9

            anchor_count = anchor_mask.sum()
            gap_count = gap_mask.sum()

            rows.append(
                {
                    "batch_idx": batch_idx,
                    "samples_in_batch": int(len(batch["sample_id"])),
                    "anchor_points": int(anchor_count.item()),
                    "gap_points": int(gap_count.item()),
                    "anchor_raw_loss_sum": float(anchor_raw.item()),
                    "gap_raw_loss_sum": float(gap_raw.item()),
                    "anchor_weighted_loss_sum": float(anchor_weighted.item()),
                    "gap_weighted_loss_sum": float(gap_weighted.item()),
                    "anchor_weighted_ratio": float((anchor_weighted / total_weighted).item()),
                    "gap_weighted_ratio": float((gap_weighted / total_weighted).item()),
                    "anchor_raw_ratio": float((anchor_raw / total_raw).item()),
                    "gap_raw_ratio": float((gap_raw / total_raw).item()),
                }
            )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(rows)
    report.to_csv(out_csv, index=False)

    if report.empty:
        print("[warn] no batch rows produced.")
        return 0

    agg = report[
        [
            "anchor_weighted_ratio",
            "gap_weighted_ratio",
            "anchor_raw_ratio",
            "gap_raw_ratio",
            "anchor_points",
            "gap_points",
        ]
    ].mean()
    print(
        "[ok] "
        f"batches={len(report)} "
        f"anchor_weighted_ratio_mean={agg['anchor_weighted_ratio']:.4f} "
        f"gap_weighted_ratio_mean={agg['gap_weighted_ratio']:.4f} "
        f"anchor_raw_ratio_mean={agg['anchor_raw_ratio']:.4f} "
        f"gap_raw_ratio_mean={agg['gap_raw_ratio']:.4f}"
    )
    print(f"[ok] report={out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
