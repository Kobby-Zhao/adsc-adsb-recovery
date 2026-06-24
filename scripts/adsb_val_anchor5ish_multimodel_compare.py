from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.adsb_test_multimodel_overlay_2d3d import (
    _default_models,
    _infer_tracks_for_model,
    _track_key_from_sample_id,
)
from src.training.utils import load_config, split_by_flight_id


def _select_tracks(anchor_min: int, anchor_max: int, max_tracks: int) -> pd.DataFrame:
    cfg = load_config("configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_proposed_24e.yaml")
    df = pd.read_parquet(cfg["data"]["samples_path"])
    splits = split_by_flight_id(
        df=df,
        flight_id_col=cfg["data"]["flight_id_col"],
        train_ratio=float(cfg["data"]["split"]["train_ratio"]),
        val_ratio=float(cfg["data"]["split"]["val_ratio"]),
        seed=int(cfg.get("seed", 42)),
    )
    val = splits["val"].copy()
    sample_id_col = cfg["data"]["sample_id_col"]
    flight_id_col = cfg["data"]["flight_id_col"]
    time_col = cfg["data"]["time_col"]
    obs_mask_col = cfg["data"]["obs_mask_col"]
    val["track_key"] = val[sample_id_col].astype(str).map(_track_key_from_sample_id)
    val["obs_mask_num"] = pd.to_numeric(val[obs_mask_col], errors="coerce").fillna(0.0)
    agg = (
        val.groupby("track_key", as_index=False)
        .agg(
            sample_id=(sample_id_col, "first"),
            flight_id=(flight_id_col, "first"),
            rows=(time_col, "size"),
            anchor_count=("obs_mask_num", lambda s: int((s > 0.5).sum())),
        )
        .sort_values(["anchor_count", "rows", "track_key"])
    )
    picked = agg[agg["anchor_count"].between(anchor_min, anchor_max)].copy()
    return picked.head(max_tracks).reset_index(drop=True)


def _load_val_with_track_key() -> tuple[dict, pd.DataFrame]:
    cfg = load_config("configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_proposed_24e.yaml")
    df = pd.read_parquet(cfg["data"]["samples_path"])
    splits = split_by_flight_id(
        df=df,
        flight_id_col=cfg["data"]["flight_id_col"],
        train_ratio=float(cfg["data"]["split"]["train_ratio"]),
        val_ratio=float(cfg["data"]["split"]["val_ratio"]),
        seed=int(cfg.get("seed", 42)),
    )
    val = splits["val"].copy()
    sample_id_col = cfg["data"]["sample_id_col"]
    val["track_key"] = val[sample_id_col].astype(str).map(_track_key_from_sample_id)
    return cfg, val


def _aligned_pred(merged: dict, ref_times: list[str]) -> np.ndarray:
    mt = [str(z) for z in merged["times"]]
    idx_map = {ts: i for i, ts in enumerate(mt)}
    hit = [idx_map.get(ts, None) for ts in ref_times]
    return np.asarray(
        [merged["pred"][h] if h is not None else [np.nan, np.nan, np.nan] for h in hit],
        dtype=np.float64,
    )


def main() -> int:
    p = argparse.ArgumentParser("Build val-set sparse-anchor multimodel comparison tables.")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--anchor-min", type=int, default=5)
    p.add_argument("--anchor-max", type=int, default=8)
    p.add_argument("--max-tracks", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--out-dir", default="outputs/runs/adsb_val_anchor5ish_multimodel_compare_20260430")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    per_track_dir = out_dir / "per_track_compare"
    per_track_full_dir = out_dir / "per_track_full_stage3"
    per_track_dir.mkdir(parents=True, exist_ok=True)
    per_track_full_dir.mkdir(parents=True, exist_ok=True)

    selected = _select_tracks(args.anchor_min, args.anchor_max, args.max_tracks)
    if selected.empty:
        raise RuntimeError("No validation tracks found for the requested anchor-count range.")
    selected.to_csv(out_dir / "selected_tracks.csv", index=False)
    track_keys = selected["track_key"].astype(str).tolist()
    track_set = set(track_keys)
    cfg, val = _load_val_with_track_key()
    time_col = cfg["data"]["time_col"]

    specs = _default_models()
    per_model_track: dict[str, dict[str, dict]] = {}
    for spec in specs:
        print(f"[run] {spec.name}")
        per_model_track[spec.name] = _infer_tracks_for_model(
            spec=spec,
            split=args.split,
            track_keys=track_set,
            batch_size_override=int(args.batch_size),
        )

    summary_rows = []
    combined_rows = []
    for tk in track_keys:
        by_model = {name: per_model_track.get(name, {}).get(tk, {}) for name in per_model_track}
        available = {k: v for k, v in by_model.items() if v and v.get("merged", {}).get("ok", False)}
        if not available:
            summary_rows.append({"track_key": tk, "status": "no_model_ok"})
            continue
        ref = next(iter(available.values()))["merged"]
        ref_times = [str(x) for x in ref["times"]]
        target = np.asarray(ref["target"], dtype=np.float64)
        obs = np.asarray(ref["obs_mask"], dtype=np.float64)
        gap_mask = obs <= 0.5

        base = pd.DataFrame(
            {
                "track_key": tk,
                "flight_id": ref.get("flight_id", ""),
                "minute_ts": ref_times,
                "is_anchor": obs > 0.5,
                "is_gap_non_anchor": gap_mask,
                "true_lat": target[:, 0],
                "true_lon": target[:, 1],
                "true_alt": target[:, 2],
            }
        )
        for name, x in available.items():
            pred = _aligned_pred(x["merged"], ref_times)
            prefix = name.lower().replace("+", "plus").replace("-", "_")
            base[f"{prefix}_pred_lat"] = pred[:, 0]
            base[f"{prefix}_pred_lon"] = pred[:, 1]
            base[f"{prefix}_pred_alt"] = pred[:, 2]
            base[f"{prefix}_abs_err_lat"] = np.abs(pred[:, 0] - target[:, 0])
            base[f"{prefix}_abs_err_lon"] = np.abs(pred[:, 1] - target[:, 1])
            base[f"{prefix}_abs_err_alt"] = np.abs(pred[:, 2] - target[:, 2])

        gap_only = base.loc[base["is_gap_non_anchor"]].copy().reset_index(drop=True)
        out_csv = per_track_dir / f"{tk}_non_anchor_compare.csv"
        gap_only.to_csv(out_csv, index=False)
        combined_rows.append(gap_only)

        # Full stage3 feature table in chronological order, with anchors + non-anchors retained.
        stage3_full = val.loc[val["track_key"].astype(str) == tk].copy()
        stage3_full[time_col] = stage3_full[time_col].astype(str)
        stage3_full = stage3_full.sort_values(time_col).reset_index(drop=True)
        full = stage3_full.merge(
            base.drop(columns=["flight_id"]),
            left_on=time_col,
            right_on="minute_ts",
            how="left",
        )
        full_out_csv = per_track_full_dir / f"{tk}_full_stage3_compare.csv"
        full.to_csv(full_out_csv, index=False)

        summary = {
            "track_key": tk,
            "flight_id": ref.get("flight_id", ""),
            "rows_total": int(len(base)),
            "gap_rows_non_anchor": int(gap_mask.sum()),
            "anchor_count": int((obs > 0.5).sum()),
            "per_track_csv": str(out_csv),
            "per_track_full_csv": str(full_out_csv),
        }
        for name, _x in available.items():
            prefix = name.lower().replace("+", "plus").replace("-", "_")
            summary[f"{prefix}_mean_abs_err_alt"] = float(gap_only[f"{prefix}_abs_err_alt"].mean())
        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "comparison_summary.csv", index=False)
    if combined_rows:
        pd.concat(combined_rows, ignore_index=True).to_csv(out_dir / "comparison_all_tracks.csv", index=False)
    full_paths = [out_dir / "per_track_full_stage3" / f"{tk}_full_stage3_compare.csv" for tk in track_keys]
    full_frames = [pd.read_csv(p) for p in full_paths if p.exists()]
    if full_frames:
        pd.concat(full_frames, ignore_index=True).to_csv(out_dir / "comparison_all_tracks_full_stage3.csv", index=False)
    print(f"[done] out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
