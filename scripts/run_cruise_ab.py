from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.datasets import DatasetConfig, TrajectoryDataset, trajectory_collate_fn
from src.models import TrajectoryRecoveryModel
from src.preprocessing import CruiseSegmentFilter
from src.preprocessing.standardize import apply_standardizer, load_standardizer
from src.training.coords import prepare_model_coordinates, restore_to_latlon
from src.training.utils import load_config, split_by_flight_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run cruise-threshold A/B experiment.")
    parser.add_argument("--config", default="configs/cruise_ab.yaml")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    return parser


def _run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd))
    env = dict(os.environ)
    env.setdefault("PYTHONNOUSERSITE", "1")
    subprocess.run(cmd, check=True, env=env)


def _max_gap_len(mask: np.ndarray) -> int:
    best = 0
    cur = 0
    for v in mask:
        if v < 0.5:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def _stable_ratio(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(mask.mean())


def _haversine_m(pred_lat: np.ndarray, pred_lon: np.ndarray, tgt_lat: np.ndarray, tgt_lon: np.ndarray) -> np.ndarray:
    lat1 = np.deg2rad(pred_lat)
    lon1 = np.deg2rad(pred_lon)
    lat2 = np.deg2rad(tgt_lat)
    lon2 = np.deg2rad(tgt_lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return 6371000.0 * c


def _distribution(series: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return {}
    q = s.quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    return {
        "count": float(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "p10": float(q.loc[0.1]),
        "p25": float(q.loc[0.25]),
        "p50": float(q.loc[0.5]),
        "p75": float(q.loc[0.75]),
        "p90": float(q.loc[0.9]),
        "p95": float(q.loc[0.95]),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def _sample_window_stats(samples_df: pd.DataFrame, long_gap_threshold: int) -> tuple[dict, pd.DataFrame]:
    g = samples_df.groupby("sample_id", as_index=False).agg(
        flight_id=("flight_id", "first"),
        sample_len=("sample_id", "size"),
        anchor_count=("obs_mask", lambda x: int((pd.to_numeric(x, errors="coerce").fillna(0.0) > 0.5).sum())),
        stable_ratio=("is_cruise_candidate", lambda x: _stable_ratio(pd.to_numeric(x, errors="coerce").fillna(0.0).to_numpy())),
    )

    max_gap = []
    for _, sg in samples_df.groupby("sample_id"):
        obs = pd.to_numeric(sg["obs_mask"], errors="coerce").fillna(0.0).to_numpy()
        max_gap.append(_max_gap_len(obs))
    g["max_gap_len"] = max_gap
    g["is_long_gap"] = g["max_gap_len"] >= int(long_gap_threshold)

    anchor_dist = _distribution(g["anchor_count"])
    stable_dist = _distribution(g["stable_ratio"])
    sample_len_dist = _distribution(g["sample_len"])

    flights = int(g["flight_id"].astype(str).nunique())
    samples = int(g["sample_id"].nunique())
    avg_windows = float(samples / flights) if flights > 0 else 0.0

    stats = {
        "retained_flights": flights,
        "sample_count": samples,
        "avg_windows_per_flight": avg_windows,
        "sample_avg_length": float(g["sample_len"].mean()) if len(g) else 0.0,
        "long_gap_sample_ratio": float(g["is_long_gap"].mean()) if len(g) else 0.0,
        "anchor_count_distribution": anchor_dist,
        "window_anchor_count_distribution": anchor_dist,
        "sample_length_distribution": sample_len_dist,
        "sample_stable_ratio_distribution": stable_dist,
    }
    return stats, g


def _cruise_feature_stats(adsb_minute: pd.DataFrame, min_cruise_minutes: int, max_abs_vertical_rate: float, max_speed_delta: float, max_heading_rate: float) -> dict:
    cf = CruiseSegmentFilter(
        min_cruise_minutes=min_cruise_minutes,
        max_abs_vertical_rate=max_abs_vertical_rate,
        max_speed_delta=max_speed_delta,
        max_heading_rate=max_heading_rate,
    )
    marked = cf.mark_cruise(adsb_minute)
    kept = marked[marked["is_cruise"].eq(1)].copy()
    return {
        "kept_flight_count": int(kept["flight_id"].astype(str).nunique()),
        "vertical_speed_distribution": _distribution(kept["vertical_speed"]),
        "heading_rate_distribution": _distribution(kept["heading_rate"]),
        "speed_delta_distribution": _distribution(kept["speed_delta"]),
    }


def _compute_gap_errors(cfg: dict, checkpoint: Path, long_gap_threshold: int) -> dict[str, float]:
    df = pd.read_parquet(cfg["data"]["samples_path"])
    splits = split_by_flight_id(
        df=df,
        flight_id_col=cfg["data"]["flight_id_col"],
        train_ratio=float(cfg["data"]["split"]["train_ratio"]),
        val_ratio=float(cfg["data"]["split"]["val_ratio"]),
        seed=int(cfg.get("seed", 42)),
    )
    scaler = load_standardizer(Path(cfg["outputs"]["run_dir"]) / "feature_standardizer.json")
    if scaler:
        scaler = {k: v for k, v in scaler.items() if k not in set(cfg["data"]["obs_cols"])}
        splits["val"] = apply_standardizer(splits["val"], scaler)

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
    ds = TrajectoryDataset(splits["val"], dcfg)
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
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    gap_hav_vals: list[np.ndarray] = []
    long_gap_hav_vals: list[np.ndarray] = []
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

            target_model, obs_model, ctx = prepare_model_coordinates(
                target_pos=target_pos,
                obs_pos=obs_pos,
                obs_mask=obs_mask,
                seq_mask=seq_mask,
                mode=str(cfg["model"].get("coord_mode", "latlon")),
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
            pred = restore_to_latlon(out["pred_pos"], seq_mask=seq_mask, ctx=ctx).detach().cpu().numpy()
            tgt = target_pos.detach().cpu().numpy()
            obs = obs_mask.detach().cpu().numpy()
            seq = seq_mask.detach().cpu().numpy()
            for i in range(pred.shape[0]):
                valid = seq[i] > 0.5
                gap = (obs[i] < 0.5) & valid
                if not gap.any():
                    continue
                hav = _haversine_m(pred[i, :, 0], pred[i, :, 1], tgt[i, :, 0], tgt[i, :, 1])
                gap_hav_vals.append(hav[gap])
                if _max_gap_len(obs[i][valid]) >= int(long_gap_threshold):
                    long_gap_hav_vals.append(hav[gap])

    gap_mean = float(np.concatenate(gap_hav_vals).mean()) if gap_hav_vals else float("nan")
    long_gap_mean = float(np.concatenate(long_gap_hav_vals).mean()) if long_gap_hav_vals else float("nan")
    return {"gap_haversine_m": gap_mean, "long_gap_haversine_m": long_gap_mean}


def main() -> int:
    args = build_parser().parse_args()
    exp_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    base_train_config = str(exp_cfg["base_train_config"])
    prep_cfg = exp_cfg["prepare"]
    train_cfg = exp_cfg["training"]
    out_root = Path(exp_cfg["outputs"]["root_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    detail_by_threshold: dict[str, dict] = {}

    for min_cruise in prep_cfg["min_cruise_minutes_options"]:
        tag = f"min_cruise_{int(min_cruise)}"
        data_out = out_root / tag / "data"
        run_dir = out_root / tag / "run"
        data_out.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)

        prep_cmd = [
            sys.executable,
            "scripts/prepare_data.py",
            "--adsb-raw-dir",
            str(prep_cfg["adsb_raw_dir"]),
            "--adsc-decoded",
            str(prep_cfg["adsc_decoded"]),
            "--output-dir",
            str(data_out),
            "--window-size",
            str(prep_cfg["window_size"]),
            "--stride",
            str(prep_cfg["stride"]),
            "--num-augment-per-flight",
            str(prep_cfg["num_augment_per_flight"]),
            "--min-gap-minutes",
            str(prep_cfg["min_gap_minutes"]),
            "--max-gap-minutes",
            str(prep_cfg["max_gap_minutes"]),
            "--target-unique-flights",
            str(prep_cfg["target_unique_flights"]),
            "--min-cruise-minutes",
            str(min_cruise),
            "--max-abs-vertical-rate",
            str(prep_cfg["max_abs_vertical_rate"]),
            "--max-speed-delta",
            str(prep_cfg["max_speed_delta"]),
            "--max-heading-rate",
            str(prep_cfg["max_heading_rate"]),
        ]
        if args.force_prepare or not (data_out / "samples.parquet").exists():
            _run(prep_cmd)
        else:
            print(f"[skip] prepare exists: {data_out / 'samples.parquet'}")

        adsb_minute = pd.read_parquet(data_out / "adsb_minute.parquet")
        samples = pd.read_parquet(data_out / "samples.parquet")
        cruise_stats = _cruise_feature_stats(
            adsb_minute=adsb_minute,
            min_cruise_minutes=int(min_cruise),
            max_abs_vertical_rate=float(prep_cfg["max_abs_vertical_rate"]),
            max_speed_delta=float(prep_cfg["max_speed_delta"]),
            max_heading_rate=float(prep_cfg["max_heading_rate"]),
        )
        sample_stats, sample_by_id = _sample_window_stats(
            samples_df=samples,
            long_gap_threshold=int(train_cfg["long_gap_threshold"]),
        )

        merged_train_cfg = load_config(base_train_config)
        merged_train_cfg["data"]["samples_path"] = str(data_out / "samples.parquet")
        merged_train_cfg["outputs"]["run_dir"] = str(run_dir)
        merged_train_cfg["training"]["epochs"] = int(train_cfg["short_epochs"])
        train_cfg_path = out_root / tag / "train_config.yaml"
        train_cfg_path.write_text(yaml.safe_dump(merged_train_cfg, sort_keys=False), encoding="utf-8")

        ckpt = run_dir / merged_train_cfg["outputs"]["checkpoint_name"]
        history_path = run_dir / "history.json"
        if args.force_train or not (ckpt.exists() and history_path.exists()):
            _run([sys.executable, "scripts/train.py", "--config", str(train_cfg_path)])
        else:
            print(f"[skip] train exists: {ckpt}")

        hist = json.loads(history_path.read_text(encoding="utf-8"))
        train_last = hist["train"][-1]
        val_last = hist["val"][-1]
        gap_metrics = _compute_gap_errors(
            cfg=merged_train_cfg,
            checkpoint=ckpt,
            long_gap_threshold=int(train_cfg["long_gap_threshold"]),
        )

        detail = {
            "threshold": int(min_cruise),
            "sample_stats": sample_stats,
            "cruise_feature_stats": cruise_stats,
            "train_last": train_last,
            "val_last": val_last,
            "gap_metrics": gap_metrics,
            "artifacts": {
                "data_dir": str(data_out),
                "run_dir": str(run_dir),
                "train_config": str(train_cfg_path),
            },
        }
        detail_by_threshold[tag] = detail

        summary_rows.append(
            {
                "threshold": int(min_cruise),
                "retained_flights": sample_stats["retained_flights"],
                "sample_count": sample_stats["sample_count"],
                "avg_windows_per_flight": sample_stats["avg_windows_per_flight"],
                "sample_avg_length": sample_stats["sample_avg_length"],
                "long_gap_sample_ratio": sample_stats["long_gap_sample_ratio"],
                "train_loss": float(train_last.get("loss", float("nan"))),
                "val_loss": float(val_last.get("loss", float("nan"))),
                "val_haversine_m": float(val_last.get("haversine_m", float("nan"))),
                "gap_haversine_m": float(gap_metrics["gap_haversine_m"]),
                "long_gap_gap_haversine_m": float(gap_metrics["long_gap_haversine_m"]),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("threshold", ascending=False).reset_index(drop=True)
    summary_csv = out_root / "ab_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    detail_json = out_root / "ab_detail.json"
    detail_json.write_text(json.dumps(detail_by_threshold, indent=2), encoding="utf-8")

    print("[ok] summary_csv=", summary_csv)
    print(summary_df.to_string(index=False))
    print("[ok] detail_json=", detail_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
