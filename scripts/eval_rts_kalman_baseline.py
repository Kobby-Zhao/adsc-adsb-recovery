#!/usr/bin/env python3
"""
Standalone RTS Kalman smoother baseline for ADS-C sparse gap recovery.

This baseline is intentionally kept outside the shared neural framework:
- no training
- no shared full_model altitude heads
- no fusion / SAVCA / graph / residual modules

It uses a constant-velocity state-space model independently on
lat / lon / alt, applies a forward Kalman filter over observed anchors,
then runs Rauch-Tung-Striebel smoothing with all anchor observations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _load_test_split(
    samples_path: str,
    flight_id_col: str = "flight_id",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    df = pd.read_parquet(samples_path)
    flights = sorted(df[flight_id_col].astype(str).unique().tolist())
    rng = np.random.default_rng(seed)
    rng.shuffle(flights)

    n = len(flights)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    test_flights = set(flights[n_train + n_val :])
    return df[df[flight_id_col].astype(str).isin(test_flights)].copy()


def _compute_metrics(true: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    err = np.abs(true - pred)
    sq_err = err**2
    gap = ~mask.astype(bool)
    return {
        "mae": float(np.mean(err)),
        "rmse": float(np.sqrt(np.mean(sq_err))),
        "gap_mae": float(np.mean(err[gap])) if gap.any() else float("nan"),
        "gap_rmse": float(np.sqrt(np.mean(sq_err[gap]))) if gap.any() else float("nan"),
    }


def _safe_minutes(ts: pd.Series) -> np.ndarray:
    t = pd.to_datetime(ts, utc=True)
    mins = (t.astype("int64") / 60_000_000_000.0).to_numpy(dtype=np.float64)
    dt = np.diff(mins, prepend=mins[0])
    if len(dt) > 1:
        dt[0] = dt[1]
    dt = np.clip(dt, 1e-3, None)
    return mins


def _estimate_initial_velocity(times: np.ndarray, obs: np.ndarray, mask: np.ndarray) -> float:
    idx = np.flatnonzero(mask)
    if idx.size >= 2:
        dt = max(times[idx[1]] - times[idx[0]], 1e-3)
        return float((obs[idx[1]] - obs[idx[0]]) / dt)
    return 0.0


def _estimate_process_sigma(times: np.ndarray, obs: np.ndarray, mask: np.ndarray, floor: float) -> float:
    idx = np.flatnonzero(mask)
    if idx.size < 3:
        return floor
    vel = []
    for i in range(idx.size - 1):
        dt = max(times[idx[i + 1]] - times[idx[i]], 1e-3)
        vel.append((obs[idx[i + 1]] - obs[idx[i]]) / dt)
    vel = np.asarray(vel, dtype=np.float64)
    if vel.size < 2:
        return floor
    acc = np.diff(vel)
    sigma = float(np.median(np.abs(acc)) * 1.4826)
    return max(sigma, floor)


def _kalman_rts_1d(
    times: np.ndarray,
    obs: np.ndarray,
    mask: np.ndarray,
    obs_sigma: float,
    accel_sigma: float,
) -> np.ndarray:
    n = len(obs)
    h = np.array([[1.0, 0.0]], dtype=np.float64)
    i2 = np.eye(2, dtype=np.float64)

    x_filt = np.zeros((n, 2), dtype=np.float64)
    p_filt = np.zeros((n, 2, 2), dtype=np.float64)
    x_pred = np.zeros((n, 2), dtype=np.float64)
    p_pred = np.zeros((n, 2, 2), dtype=np.float64)
    a_hist = np.zeros((n, 2, 2), dtype=np.float64)

    first_idx = int(np.flatnonzero(mask)[0]) if mask.any() else 0
    x_prev = np.array([obs[first_idx], _estimate_initial_velocity(times, obs, mask)], dtype=np.float64)
    p_prev = np.diag([max(obs_sigma**2, 1e-6), max((10.0 * obs_sigma) ** 2, 1e-4)])

    for t in range(n):
        dt = max(times[t] - times[t - 1], 1e-3) if t > 0 else max(times[min(1, n - 1)] - times[0], 1e-3)
        a = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        q = (accel_sigma**2) * np.array(
            [[0.25 * dt**4, 0.5 * dt**3], [0.5 * dt**3, dt**2]],
            dtype=np.float64,
        )
        x_p = a @ x_prev
        p_p = a @ p_prev @ a.T + q

        x_pred[t] = x_p
        p_pred[t] = p_p
        a_hist[t] = a

        if mask[t]:
            z = np.array([[obs[t]]], dtype=np.float64)
            r = np.array([[obs_sigma**2]], dtype=np.float64)
            y = z - h @ x_p.reshape(2, 1)
            s = h @ p_p @ h.T + r
            k = p_p @ h.T @ np.linalg.inv(s)
            x_u = x_p.reshape(2, 1) + k @ y
            p_u = (i2 - k @ h) @ p_p
            x_prev = x_u[:, 0]
            p_prev = 0.5 * (p_u + p_u.T)
        else:
            x_prev = x_p
            p_prev = p_p

        x_filt[t] = x_prev
        p_filt[t] = p_prev

    x_smooth = x_filt.copy()
    p_smooth = p_filt.copy()
    for t in range(n - 2, -1, -1):
        c = p_filt[t] @ a_hist[t + 1].T @ np.linalg.pinv(p_pred[t + 1])
        x_smooth[t] = x_filt[t] + c @ (x_smooth[t + 1] - x_pred[t + 1])
        p_smooth[t] = p_filt[t] + c @ (p_smooth[t + 1] - p_pred[t + 1]) @ c.T

    out = x_smooth[:, 0]
    out[mask] = obs[mask]
    return out


def _smooth_sample(sample_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sdf = sample_df.sort_values("minute_ts").reset_index(drop=True)
    times = _safe_minutes(sdf["minute_ts"])
    mask = sdf["obs_mask"].to_numpy(dtype=bool)
    lat = sdf["lat"].to_numpy(dtype=np.float64)
    lon = sdf["lon"].to_numpy(dtype=np.float64)
    alt = sdf["alt"].to_numpy(dtype=np.float64)

    lat_pred = _kalman_rts_1d(times, lat, mask, obs_sigma=1e-6, accel_sigma=_estimate_process_sigma(times, lat, mask, 1e-5))
    lon_pred = _kalman_rts_1d(times, lon, mask, obs_sigma=1e-6, accel_sigma=_estimate_process_sigma(times, lon, mask, 1e-5))
    alt_pred = _kalman_rts_1d(times, alt, mask, obs_sigma=1.0, accel_sigma=_estimate_process_sigma(times, alt, mask, 5.0))

    # Hard-anchor protocol: preserve observed ADS-C points exactly.
    lat_pred[mask] = lat[mask]
    lon_pred[mask] = lon[mask]
    alt_pred[mask] = alt[mask]
    return lat_pred, lon_pred, alt_pred


def evaluate(df: pd.DataFrame) -> tuple[Dict[str, float], pd.DataFrame]:
    sample_ids = sorted(df["sample_id"].unique())
    rows = []
    all_lat_true, all_lon_true, all_alt_true = [], [], []
    all_lat_pred, all_lon_pred, all_alt_pred = [], [], []
    all_mask = []

    for sid in sample_ids:
        sdf = df[df["sample_id"] == sid].sort_values("minute_ts")
        if len(sdf) < 3:
            continue
        lat_pred, lon_pred, alt_pred = _smooth_sample(sdf)
        mask = sdf["obs_mask"].to_numpy(dtype=bool)
        lat_true = sdf["lat"].to_numpy(dtype=np.float64)
        lon_true = sdf["lon"].to_numpy(dtype=np.float64)
        alt_true = sdf["alt"].to_numpy(dtype=np.float64)

        lat_m = _compute_metrics(lat_true, lat_pred, mask)
        lon_m = _compute_metrics(lon_true, lon_pred, mask)
        alt_m = _compute_metrics(alt_true, alt_pred, mask)

        rows.append(
            {
                "sample_id": sid,
                "lat_rmse": lat_m["rmse"],
                "lat_mae": lat_m["mae"],
                "gap_lat_rmse": lat_m["gap_rmse"],
                "gap_lat_mae": lat_m["gap_mae"],
                "lon_rmse": lon_m["rmse"],
                "lon_mae": lon_m["mae"],
                "gap_lon_rmse": lon_m["gap_rmse"],
                "gap_lon_mae": lon_m["gap_mae"],
                "alt_rmse": alt_m["rmse"],
                "alt_mae": alt_m["mae"],
                "gap_alt_rmse": alt_m["gap_rmse"],
                "gap_alt_mae": alt_m["gap_mae"],
                "num_points": len(sdf),
                "gap_points": int((~mask).sum()),
                "anchor_points": int(mask.sum()),
            }
        )

        all_lat_true.append(lat_true)
        all_lon_true.append(lon_true)
        all_alt_true.append(alt_true)
        all_lat_pred.append(lat_pred)
        all_lon_pred.append(lon_pred)
        all_alt_pred.append(alt_pred)
        all_mask.append(mask)

    lat_m = _compute_metrics(np.concatenate(all_lat_true), np.concatenate(all_lat_pred), np.concatenate(all_mask))
    lon_m = _compute_metrics(np.concatenate(all_lon_true), np.concatenate(all_lon_pred), np.concatenate(all_mask))
    alt_m = _compute_metrics(np.concatenate(all_alt_true), np.concatenate(all_alt_pred), np.concatenate(all_mask))

    summary = {
        "model": "RTS_Kalman_Smoother",
        "split": "test",
        "mae_lat": lat_m["mae"],
        "mae_lon": lon_m["mae"],
        "mae_alt": alt_m["mae"],
        "rmse_lat": lat_m["rmse"],
        "rmse_lon": lon_m["rmse"],
        "rmse_alt": alt_m["rmse"],
        "gap_mae_lat": lat_m["gap_mae"],
        "gap_mae_lon": lon_m["gap_mae"],
        "gap_mae_alt": alt_m["gap_mae"],
        "gap_rmse_lat": lat_m["gap_rmse"],
        "gap_rmse_lon": lon_m["gap_rmse"],
        "gap_rmse_alt": alt_m["gap_rmse"],
    }
    return summary, pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate standalone RTS Kalman smoother baseline.")
    parser.add_argument(
        "--samples",
        default="outputs/mvp_merged_250_20260514_clean/stage3_clean/samples.parquet",
        help="Parquet samples path.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        default="outputs/experiments/obs_conditioned_gaponly/rts_kalman_smoother_baseline_v1",
        help="Output directory for summary JSON and per-sample CSV.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    samples_path = root / args.samples
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Loading test split from {samples_path} ...")
    df_test = _load_test_split(
        str(samples_path),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(
        f"      samples={df_test['sample_id'].nunique()} rows={len(df_test)} "
        f"anchors={int(df_test['obs_mask'].sum())} gaps={int((1 - df_test['obs_mask']).sum())}"
    )

    print("[2/3] Running RTS Kalman smoother baseline ...")
    summary, per_sample = evaluate(df_test)
    print(
        f"      gap_rmse_lat={summary['gap_rmse_lat']:.6f} "
        f"gap_rmse_lon={summary['gap_rmse_lon']:.6f} "
        f"gap_rmse_alt={summary['gap_rmse_alt']:.6f}"
    )

    print("[3/3] Saving outputs ...")
    summary_path = out_dir / "main_task_metrics_test_summary_latlon.json"
    per_sample_path = out_dir / "per_sample_metrics.csv"
    summary_path.write_text(json.dumps([summary], indent=2), encoding="utf-8")
    per_sample.to_csv(per_sample_path, index=False)
    print(f"[ok] summary={summary_path}")
    print(f"[ok] per_sample={per_sample_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
