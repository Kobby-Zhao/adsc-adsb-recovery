from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.adsb_test_multimodel_overlay_2d3d import _default_models, _infer_tracks_for_model, _track_key_from_sample_id
from src.training.utils import load_config, split_by_flight_id


def _aligned_pred(merged: dict, ref_times: list[str]) -> np.ndarray:
    mt = [str(z) for z in merged["times"]]
    idx_map = {ts: i for i, ts in enumerate(mt)}
    hit = [idx_map.get(ts, None) for ts in ref_times]
    return np.asarray(
        [merged["pred"][h] if h is not None else [np.nan, np.nan, np.nan] for h in hit],
        dtype=np.float64,
    )


def _select_samples(
    anchor_min: int,
    anchor_max: int,
    gap_min: int | None,
    gap_max: int | None,
    max_samples: int,
    unique_by_flight: bool,
) -> pd.DataFrame:
    cfg = load_config("configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_proposed_24e.yaml")
    stage3_path = "outputs/mvp_merged_nostage_20260415/stage_datasets_20260415_s2v2/stage3/samples.parquet"
    df = pd.read_parquet(stage3_path)
    splits = split_by_flight_id(
        df=df,
        flight_id_col=cfg["data"]["flight_id_col"],
        train_ratio=float(cfg["data"]["split"]["train_ratio"]),
        val_ratio=float(cfg["data"]["split"]["val_ratio"]),
        seed=int(cfg.get("seed", 42)),
    )
    val = splits["val"].copy()
    val["obs_mask_num"] = pd.to_numeric(val[cfg["data"]["obs_mask_col"]], errors="coerce").fillna(0.0)
    agg = val.groupby(cfg["data"]["sample_id_col"], as_index=False).agg(
        flight_id=(cfg["data"]["flight_id_col"], "first"),
        rows=(cfg["data"]["time_col"], "size"),
        anchor_count=("obs_mask_num", lambda s: int((s > 0.5).sum())),
        gap_rows=("obs_mask_num", lambda s: int((s <= 0.5).sum())),
    )
    agg = agg.sort_values(["gap_rows", "anchor_count", "rows", cfg["data"]["sample_id_col"]], ascending=[False, True, True, True]).reset_index(drop=True)

    cond = agg["anchor_count"].between(anchor_min, anchor_max)
    if gap_min is not None:
        cond &= agg["gap_rows"] >= gap_min
    if gap_max is not None:
        cond &= agg["gap_rows"] <= gap_max
    cand = agg.loc[cond].copy()

    picked: list[dict] = []
    used_flights: set[str] = set()
    for _, r in cand.iterrows():
        if unique_by_flight and str(r["flight_id"]) in used_flights:
            continue
        rec = r.to_dict()
        rec["bucket"] = f"a{anchor_min}_{anchor_max}_g{gap_min if gap_min is not None else 'na'}_{gap_max if gap_max is not None else 'na'}"
        picked.append(rec)
        used_flights.add(str(r["flight_id"]))
        if len(picked) >= max_samples:
            break
    out = pd.DataFrame(picked)
    if out.empty:
        return out
    out["track_key"] = out["sample_id"].astype(str).map(_track_key_from_sample_id)
    return out


def main() -> int:
    p = argparse.ArgumentParser("Export per-flight stage3 val CSVs with model recovery columns.")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--out-dir", default="outputs/runs/adsb_stage3_val_selected10_fullcruise_multimodel_csvs_20260430")
    p.add_argument("--anchor-min", type=int, default=2)
    p.add_argument("--anchor-max", type=int, default=8)
    p.add_argument("--gap-min", type=int, default=None)
    p.add_argument("--gap-max", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=10)
    p.add_argument("--allow-duplicate-flight", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    per_flight_dir = out_dir / "per_flight_csv"
    per_flight_dir.mkdir(parents=True, exist_ok=True)

    selected = _select_samples(
        anchor_min=int(args.anchor_min),
        anchor_max=int(args.anchor_max),
        gap_min=args.gap_min,
        gap_max=args.gap_max,
        max_samples=int(args.max_samples),
        unique_by_flight=not args.allow_duplicate_flight,
    )
    if selected.empty:
        raise RuntimeError("No samples matched the requested anchor/gap filters.")
    selected.to_csv(out_dir / "selected_samples.csv", index=False)

    cfg = load_config("configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_proposed_24e.yaml")
    stage3_path = "outputs/mvp_merged_nostage_20260415/stage_datasets_20260415_s2v2/stage3/samples.parquet"
    cruise_path = "outputs/mvp_merged_nostage_20260415/adsb_cruise_merged.parquet"
    df = pd.read_parquet(stage3_path)
    splits = split_by_flight_id(
        df=df,
        flight_id_col=cfg["data"]["flight_id_col"],
        train_ratio=float(cfg["data"]["split"]["train_ratio"]),
        val_ratio=float(cfg["data"]["split"]["val_ratio"]),
        seed=int(cfg.get("seed", 42)),
    )
    val = splits["val"].copy()
    time_col = cfg["data"]["time_col"]
    sample_id_col = cfg["data"]["sample_id_col"]
    val["track_key"] = val[sample_id_col].astype(str).map(_track_key_from_sample_id)
    selected_flights = selected["flight_id"].astype(str).tolist()
    cruise_full = pd.read_parquet(cruise_path)
    cruise_full = cruise_full.loc[cruise_full["flight_id"].astype(str).isin(selected_flights)].copy()
    cruise_full[time_col] = cruise_full[time_col].astype(str)

    track_keys = set(selected["track_key"].astype(str).tolist())
    specs = _default_models()
    per_model_track: dict[str, dict[str, dict]] = {}
    for spec in specs:
        print(f"[run] {spec.name}")
        per_model_track[spec.name] = _infer_tracks_for_model(
            spec=spec,
            split=args.split,
            track_keys=track_keys,
            batch_size_override=int(args.batch_size),
        )

    rows = []
    combined = []
    for _, sel in selected.iterrows():
        sample_id = str(sel["sample_id"])
        flight_id = str(sel["flight_id"])
        track_key = str(sel["track_key"])
        # Use the full upstream cruise-minute ADS-B trajectory for the selected flight.
        stage3_full = (
            cruise_full.loc[cruise_full[cfg["data"]["flight_id_col"]].astype(str) == flight_id]
            .sort_values(time_col)
            .reset_index(drop=True)
        )

        by_model = {name: per_model_track.get(name, {}).get(track_key, {}) for name in per_model_track}
        available = {k: v for k, v in by_model.items() if v and v.get("merged", {}).get("ok", False)}
        if not available:
            rows.append({"sample_id": sample_id, "flight_id": flight_id, "track_key": track_key, "status": "no_model_ok"})
            continue
        sample_window = (
            val.loc[val[sample_id_col].astype(str) == sample_id, [time_col, cfg["data"]["obs_mask_col"]]]
            .copy()
            .sort_values(time_col)
        )
        sample_window[time_col] = sample_window[time_col].astype(str)
        sample_window["compare_is_anchor"] = (
            pd.to_numeric(sample_window[cfg["data"]["obs_mask_col"]], errors="coerce").fillna(0.0) > 0.5
        )
        sample_window["is_gap_non_anchor"] = ~sample_window["compare_is_anchor"]
        sample_window = sample_window[[time_col, "compare_is_anchor", "is_gap_non_anchor"]].rename(columns={time_col: "minute_ts"})
        ref = next(iter(available.values()))["merged"]
        ref_times = [str(x) for x in ref["times"]]
        target = np.asarray(ref["target"], dtype=np.float64)

        base = pd.DataFrame(
            {
                "minute_ts": ref_times,
                "window_true_lat": target[:, 0],
                "window_true_lon": target[:, 1],
                "window_true_alt": target[:, 2],
            }
        )
        base = base.merge(sample_window, on="minute_ts", how="left")
        base["selected_window_minute"] = base["compare_is_anchor"].notna() | base["is_gap_non_anchor"].notna()
        base["selected_anchor_lat"] = np.where(base["compare_is_anchor"].fillna(False), base["window_true_lat"], np.nan)
        base["selected_anchor_lon"] = np.where(base["compare_is_anchor"].fillna(False), base["window_true_lon"], np.nan)
        base["selected_anchor_alt"] = np.where(base["compare_is_anchor"].fillna(False), base["window_true_alt"], np.nan)
        for name, x in available.items():
            pred = _aligned_pred(x["merged"], ref_times)
            prefix = name.lower().replace("+", "plus").replace("-", "_")
            gap_only = base["is_gap_non_anchor"].fillna(False).to_numpy()
            base[f"{prefix}_pred_lat"] = np.where(gap_only, pred[:, 0], np.nan)
            base[f"{prefix}_pred_lon"] = np.where(gap_only, pred[:, 1], np.nan)
            base[f"{prefix}_pred_alt"] = np.where(gap_only, pred[:, 2], np.nan)
            base[f"{prefix}_abs_err_lat"] = np.where(gap_only, np.abs(pred[:, 0] - target[:, 0]), np.nan)
            base[f"{prefix}_abs_err_lon"] = np.where(gap_only, np.abs(pred[:, 1] - target[:, 1]), np.nan)
            base[f"{prefix}_abs_err_alt"] = np.where(gap_only, np.abs(pred[:, 2] - target[:, 2]), np.nan)
        # Some stitched inference outputs can contain duplicate minute_ts. Keep one row per minute before merging
        # back to the full cruise trajectory.
        base = base.sort_values("minute_ts").drop_duplicates(subset=["minute_ts"], keep="first").reset_index(drop=True)

        out = stage3_full.merge(base, left_on=time_col, right_on="minute_ts", how="left")
        out["selected_window_minute"] = out["selected_window_minute"].fillna(False)
        out["compare_is_anchor"] = out["compare_is_anchor"].fillna(False)
        out["is_gap_non_anchor"] = out["is_gap_non_anchor"].fillna(False)
        out["adsb_lat"] = out["lat"]
        out["adsb_lon"] = out["lon"]
        out["adsb_alt"] = out["alt"]
        out["true_lat"] = out["lat"]
        out["true_lon"] = out["lon"]
        out["true_alt"] = out["alt"]
        # The selected sparse sample defines which minutes are anchors/non-anchors for model comparison inside the full cruise trajectory.
        out["selected_sample_id"] = sample_id
        out["selected_bucket"] = str(sel["bucket"])
        out["selected_anchor_count"] = int(sel["anchor_count"])
        out["selected_gap_rows"] = int(sel["gap_rows"])
        out["selected_rows"] = int(sel["rows"])
        out["selected_window_start_ts"] = sample_window["minute_ts"].min()
        out["selected_window_end_ts"] = sample_window["minute_ts"].max()
        ordered_front = [
            "flight_id",
            time_col,
            "adsb_lat",
            "adsb_lon",
            "adsb_alt",
            "speed",
            "heading",
            "vertical_speed",
            "num_points_in_minute",
            "adsb_icao",
            "selected_sample_id",
            "selected_bucket",
            "selected_anchor_count",
            "selected_gap_rows",
            "selected_rows",
            "selected_window_start_ts",
            "selected_window_end_ts",
            "selected_window_minute",
            "compare_is_anchor",
            "is_gap_non_anchor",
            "selected_anchor_lat",
            "selected_anchor_lon",
            "selected_anchor_alt",
            "true_lat",
            "true_lon",
            "true_alt",
        ]
        front = [c for c in ordered_front if c in out.columns]
        rest = [c for c in out.columns if c not in front]
        out = out[front + rest]
        out_path = per_flight_dir / f"{flight_id}.csv"
        out.to_csv(out_path, index=False)
        combined.append(out)
        rows.append(
            {
                "sample_id": sample_id,
                "flight_id": flight_id,
                "track_key": track_key,
                "bucket": str(sel["bucket"]),
                "anchor_count": int(sel["anchor_count"]),
                "gap_rows": int(sel["gap_rows"]),
                "rows": int(sel["rows"]),
                "full_cruise_rows": int(len(out)),
                "selected_window_minutes": int(out["selected_window_minute"].sum()),
                "selected_anchor_minutes": int(out["compare_is_anchor"].sum()),
                "selected_gap_minutes": int(out["is_gap_non_anchor"].sum()),
                "csv_path": str(out_path),
            }
        )

    pd.DataFrame(rows).to_csv(out_dir / "export_summary.csv", index=False)
    if combined:
        pd.concat(combined, ignore_index=True).to_csv(out_dir / "combined_all_flights.csv", index=False)
    print(f"[done] out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
