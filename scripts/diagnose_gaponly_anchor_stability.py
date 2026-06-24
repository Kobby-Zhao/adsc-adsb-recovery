from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.real_adsc_replay_eval import _predict_on_frame
from scripts.train import load_config
from src.training.utils import split_by_flight_id


STAGES = {
    "S1": "outputs/mvp_merged_250_20260514_clean/stage1_clean/samples.parquet",
    "S2": "outputs/mvp_merged_250_20260514_clean/stage2_clean/samples.parquet",
    "S3": "outputs/mvp_merged_250_20260514_clean/stage3_clean/samples.parquet",
}

MODELS = {
    "BiLSTM-clean": (
        "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_bilstm_vs_backbone_v1/configs/bilstm_clean_absolute.yaml",
        "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_bilstm_vs_backbone_v1/bilstm_clean_absolute/best.pt",
    ),
    "Backbone-only": (
        "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_bilstm_vs_backbone_v1/configs/ours_backbone_absolute.yaml",
        "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_bilstm_vs_backbone_v1/ours_backbone_absolute/best.pt",
    ),
    "A1-linear-alt": (
        "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_bilstm_vs_backbone_v1/configs/a1_linear_alt_baseline.yaml",
        "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_bilstm_vs_backbone_v1/a1_linear_alt_baseline/best.pt",
    ),
    "Ours-A3": (
        "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_bilstm_vs_backbone_v1/configs/a3_risk_routed.yaml",
        "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_bilstm_vs_backbone_v1/a3_risk_routed/best.pt",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Diagnose hard-anchor and altitude-profile stability on gap-only test splits.")
    p.add_argument("--out-dir", default="outputs/experiments/obs_conditioned_gaponly/anchor_stability_diagnostics_20260517")
    p.add_argument("--long-gap-threshold", type=int, default=30)
    p.add_argument("--few-anchor-threshold", type=int, default=4)
    p.add_argument("--hard-alt-rmse-threshold-m", type=float, default=50.0)
    return p


def _max_gap_len(obs: np.ndarray) -> int:
    best = 0
    cur = 0
    for v in obs:
        if v <= 0.5:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _linear_ref_alt(true_alt: np.ndarray, obs: np.ndarray) -> np.ndarray:
    ref = true_alt.astype(float).copy()
    anchor_idx = np.where(obs > 0.5)[0]
    if len(anchor_idx) < 2:
        return ref
    for left, right in zip(anchor_idx[:-1], anchor_idx[1:]):
        if right <= left + 1:
            continue
        idx = np.arange(left + 1, right)
        frac = (idx - left) / float(right - left)
        ref[idx] = true_alt[left] + frac * (true_alt[right] - true_alt[left])
    return ref


def _safe_mean(x: list[float] | np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _safe_max(x: list[float] | np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.max()) if arr.size else float("nan")


def _sample_metrics(sample_id: str, base: pd.DataFrame, pred: pd.DataFrame) -> dict:
    b = base.sort_values("minute_ts").reset_index(drop=True).copy()
    p = pred.sort_values("minute_ts").reset_index(drop=True).copy()
    b["minute_ts"] = pd.to_datetime(b["minute_ts"], utc=True)
    p["minute_ts"] = pd.to_datetime(p["minute_ts"], utc=True)
    merged = b.merge(
        p[["minute_ts", "pred_lat", "pred_lon", "pred_alt"]],
        on="minute_ts",
        how="left",
    )
    obs = pd.to_numeric(merged["obs_mask"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    gap = obs <= 0.5
    anchor = obs > 0.5
    true = merged[["lat", "lon", "alt"]].to_numpy(dtype=float)
    pred_arr = merged[["pred_lat", "pred_lon", "pred_alt"]].to_numpy(dtype=float)
    err = pred_arr - true

    ref_alt = _linear_ref_alt(true[:, 2], obs)
    linear_gap_err = ref_alt[gap] - true[gap, 2]
    hard_alt_rmse = float(np.sqrt(np.mean(linear_gap_err**2))) if linear_gap_err.size else float("nan")

    anchor_err = np.abs(err[anchor]) if anchor.any() else np.empty((0, 3))
    gap_err = err[gap] if gap.any() else np.empty((0, 3))

    boundary_jumps = []
    edge_jumps = []
    first_diffs = []
    second_diffs = []
    ref_devs = []
    anchor_idx = np.where(anchor)[0]
    for left, right in zip(anchor_idx[:-1], anchor_idx[1:]):
        if right <= left + 1:
            continue
        gidx = np.arange(left + 1, right)
        if gidx.size == 0:
            continue
        alt_seg = pred_arr[left : right + 1, 2]
        boundary_jumps.append(abs(pred_arr[left + 1, 2] - pred_arr[left, 2]))
        boundary_jumps.append(abs(pred_arr[right, 2] - pred_arr[right - 1, 2]))
        edge_jumps.append(abs(pred_arr[left + 1, 2] - pred_arr[left, 2]))
        edge_jumps.append(abs(pred_arr[right, 2] - pred_arr[right - 1, 2]))
        if alt_seg.size >= 2:
            d1 = np.diff(alt_seg)
            first_diffs.extend(np.abs(d1).tolist())
            if d1.size >= 2:
                second_diffs.extend(np.abs(np.diff(d1)).tolist())
        ref_devs.extend(np.abs(pred_arr[gidx, 2] - ref_alt[gidx]).tolist())

    return {
        "sample_id": sample_id,
        "flight_id": str(merged["flight_id"].iloc[0]),
        "anchor_count": int(anchor.sum()),
        "gap_count": int(gap.sum()),
        "max_gap_minutes": int(_max_gap_len(obs)),
        "linear_ref_gap_alt_rmse_m": hard_alt_rmse,
        "gap_lon_rmse": float(np.sqrt(np.mean(gap_err[:, 1] ** 2))) if gap_err.size else float("nan"),
        "gap_lat_rmse": float(np.sqrt(np.mean(gap_err[:, 0] ** 2))) if gap_err.size else float("nan"),
        "gap_alt_rmse_m": float(np.sqrt(np.mean(gap_err[:, 2] ** 2))) if gap_err.size else float("nan"),
        "gap_alt_mae_m": float(np.mean(np.abs(gap_err[:, 2]))) if gap_err.size else float("nan"),
        "anchor_lat_max_abs_err": float(anchor_err[:, 0].max()) if anchor_err.size else float("nan"),
        "anchor_lon_max_abs_err": float(anchor_err[:, 1].max()) if anchor_err.size else float("nan"),
        "anchor_alt_max_abs_err_m": float(anchor_err[:, 2].max()) if anchor_err.size else float("nan"),
        "boundary_jump_mean_m": _safe_mean(boundary_jumps),
        "boundary_jump_max_m": _safe_max(boundary_jumps),
        "edge_jump_mean_m": _safe_mean(edge_jumps),
        "vertical_step_abs_mean_m": _safe_mean(first_diffs),
        "vertical_second_diff_abs_mean_m": _safe_mean(second_diffs),
        "anchor_ref_dev_mean_m": _safe_mean(ref_devs),
        "anchor_ref_dev_max_m": _safe_max(ref_devs),
    }


def _load_test_frame(samples_path: str, cfg: dict) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / samples_path)
    splits = split_by_flight_id(
        df=df,
        flight_id_col=cfg["data"]["flight_id_col"],
        train_ratio=float(cfg["data"]["split"]["train_ratio"]),
        val_ratio=float(cfg["data"]["split"]["val_ratio"]),
        seed=int(cfg.get("seed", 42)),
    )
    test = splits["test"].copy()
    # _predict_on_frame expects physical obs columns. Missing obs are allowed to remain zero.
    test["minute_ts"] = pd.to_datetime(test["minute_ts"], utc=True)
    return test.sort_values(["sample_id", "minute_ts"]).reset_index(drop=True)


def _summarize(df: pd.DataFrame, subset_name: str, mask: pd.Series) -> dict:
    s = df[mask].copy()
    out = {"subset": subset_name, "sample_count": int(len(s))}
    for col in [
        "gap_lon_rmse",
        "gap_lat_rmse",
        "gap_alt_rmse_m",
        "gap_alt_mae_m",
        "anchor_lat_max_abs_err",
        "anchor_lon_max_abs_err",
        "anchor_alt_max_abs_err_m",
        "boundary_jump_mean_m",
        "boundary_jump_max_m",
        "vertical_step_abs_mean_m",
        "vertical_second_diff_abs_mean_m",
        "anchor_ref_dev_mean_m",
        "anchor_ref_dev_max_m",
        "linear_ref_gap_alt_rmse_m",
    ]:
        out[col] = float(s[col].mean()) if len(s) else float("nan")
    return out


def main() -> int:
    args = build_parser().parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_sample_rows = []
    all_summary_rows = []

    for model_name, (cfg_rel, ckpt_rel) in MODELS.items():
        cfg = load_config(str(ROOT / cfg_rel))
        for stage, samples_path in STAGES.items():
            print(f"[diag] model={model_name} stage={stage}")
            frame = _load_test_frame(samples_path, cfg)
            pred = _predict_on_frame(cfg=cfg, checkpoint=ROOT / ckpt_rel, frame=frame, pred_key="pred_pos")
            pred.to_csv(out_dir / f"point_predictions_{model_name}_{stage}.csv", index=False)
            rows = []
            pred_groups = {str(k): g for k, g in pred.groupby("sample_id", sort=False)}
            for sid, base_g in frame.groupby("sample_id", sort=False):
                sid = str(sid)
                if sid not in pred_groups:
                    continue
                row = _sample_metrics(sid, base_g, pred_groups[sid])
                row["model"] = model_name
                row["stage"] = stage
                row["is_long_gap"] = row["max_gap_minutes"] >= int(args.long_gap_threshold)
                row["is_few_anchor"] = row["anchor_count"] <= int(args.few_anchor_threshold)
                row["is_hard_altitude"] = row["linear_ref_gap_alt_rmse_m"] >= float(args.hard_alt_rmse_threshold_m)
                row["is_adsc_like"] = bool(row["is_long_gap"] and row["is_few_anchor"])
                rows.append(row)
            sample_df = pd.DataFrame(rows)
            sample_df.to_csv(out_dir / f"sample_diagnostics_{model_name}_{stage}.csv", index=False)
            all_sample_rows.extend(rows)

            for subset, mask in [
                ("all", pd.Series(True, index=sample_df.index)),
                ("long_gap", sample_df["is_long_gap"]),
                ("few_anchor", sample_df["is_few_anchor"]),
                ("hard_altitude", sample_df["is_hard_altitude"]),
                ("adsc_like", sample_df["is_adsc_like"]),
            ]:
                summary = _summarize(sample_df, subset, mask)
                summary["model"] = model_name
                summary["stage"] = stage
                all_summary_rows.append(summary)

    all_samples = pd.DataFrame(all_sample_rows)
    all_summary = pd.DataFrame(all_summary_rows)
    all_samples.to_csv(out_dir / "all_sample_diagnostics.csv", index=False)
    all_summary.to_csv(out_dir / "stability_summary_by_subset.csv", index=False)

    report = {
        "out_dir": str(out_dir),
        "models": list(MODELS.keys()),
        "stages": STAGES,
        "thresholds": {
            "long_gap_minutes": args.long_gap_threshold,
            "few_anchor_count": args.few_anchor_threshold,
            "hard_altitude_linear_ref_rmse_m": args.hard_alt_rmse_threshold_m,
        },
    }
    (out_dir / "diagnostic_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[ok] samples={out_dir / 'all_sample_diagnostics.csv'}")
    print(f"[ok] summary={out_dir / 'stability_summary_by_subset.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
