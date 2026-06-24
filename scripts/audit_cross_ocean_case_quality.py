from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Audit cross-ocean replay cases for ADS-B freezing and speed-position contradictions.")
    p.add_argument(
        "--case-root",
        default="outputs/runs/current_cross_ocean_altitude_compare_20260517_three_models",
    )
    p.add_argument("--out-csv", default="")
    p.add_argument("--max-repeat-minutes", type=int, default=3)
    p.add_argument("--min-speed-for-freeze-mps", type=float, default=50.0)
    p.add_argument("--min-speed-position-ratio", type=float, default=0.25)
    return p


def _haversine_m(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371000.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _max_true_run(mask: np.ndarray) -> int:
    best = 0
    cur = 0
    for v in mask:
        if bool(v):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _audit_case(case_dir: Path, args: argparse.Namespace) -> dict:
    p = case_dir / "known_adsb_minute.csv"
    if not p.exists():
        return {"pair_id": case_dir.name, "status": "missing_known_adsb"}
    df = pd.read_csv(p)
    if df.empty:
        return {"pair_id": case_dir.name, "status": "empty_known_adsb"}
    df["minute_ts"] = pd.to_datetime(df["minute_ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["minute_ts", "lat", "lon", "alt"]).sort_values("minute_ts").reset_index(drop=True)
    lat = pd.to_numeric(df["lat"], errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(df["lon"], errors="coerce").to_numpy(dtype=float)
    alt = pd.to_numeric(df["alt"], errors="coerce").to_numpy(dtype=float)
    speed = pd.to_numeric(df.get("speed", pd.Series(np.nan, index=df.index)), errors="coerce").to_numpy(dtype=float)
    same = np.zeros(len(df), dtype=bool)
    if len(df) >= 2:
        same[1:] = (np.abs(np.diff(lat)) < 1e-10) & (np.abs(np.diff(lon)) < 1e-10) & (np.abs(np.diff(alt)) < 1e-6)
        dist = np.zeros(len(df), dtype=float)
        dist[1:] = _haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    else:
        dist = np.zeros(len(df), dtype=float)
    implied_speed = dist / 60.0
    freeze_with_speed = same & (speed > float(args.min_speed_for_freeze_mps))
    speed_ratio = np.divide(implied_speed, speed, out=np.full_like(implied_speed, np.nan), where=np.isfinite(speed) & (speed > 1e-6))
    contradiction = (speed > float(args.min_speed_for_freeze_mps)) & np.isfinite(speed_ratio) & (speed_ratio < float(args.min_speed_position_ratio))
    max_repeat = _max_true_run(same) + (1 if same.any() else 0)
    max_freeze_speed = _max_true_run(freeze_with_speed) + (1 if freeze_with_speed.any() else 0)
    pass_quality = bool(max_repeat <= int(args.max_repeat_minutes) and max_freeze_speed <= int(args.max_repeat_minutes))
    return {
        "pair_id": case_dir.name,
        "status": "ok",
        "known_adsb_rows": int(len(df)),
        "unique_lat_lon_alt": int(pd.DataFrame({"lat": lat, "lon": lon, "alt": alt}).drop_duplicates().shape[0]),
        "max_repeat_position_minutes": int(max_repeat),
        "max_freeze_with_speed_minutes": int(max_freeze_speed),
        "freeze_with_speed_rows": int(freeze_with_speed.sum()),
        "speed_position_contradiction_rows": int(contradiction.sum()),
        "median_speed_mps": float(np.nanmedian(speed)),
        "median_implied_speed_mps": float(np.nanmedian(implied_speed)),
        "quality_pass": pass_quality,
        "reason": "pass" if pass_quality else "repeated/frozen ADS-B minute positions",
    }


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.case_root)
    rows = []
    for case_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        rows.append(_audit_case(case_dir, args))
    out = pd.DataFrame(rows)
    out_csv = Path(args.out_csv) if args.out_csv else root / "cross_ocean_case_quality_audit.csv"
    out.to_csv(out_csv, index=False)
    print(f"[ok] audit={out_csv}")
    if "quality_pass" in out.columns:
        print(out[["pair_id", "quality_pass", "max_repeat_position_minutes", "max_freeze_with_speed_minutes", "reason"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
