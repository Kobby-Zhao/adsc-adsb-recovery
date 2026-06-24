from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import load_settings


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Refetch raw OpenSky ADS-B points for selected cross-ocean cases and audit freezes.")
    p.add_argument(
        "--overlay-summary-csv",
        default="outputs/runs/adsc_cross_ocean_top10_highest_anchor_20260429/overlay_summary.csv",
    )
    p.add_argument(
        "--old-adsb-minute-csv",
        default="outputs/runs/adsc_cross_ocean_top10_highest_anchor_20260429/top10_cross_ocean_highest_anchor_adsb_minute_full_flights.csv",
    )
    p.add_argument("--out-dir", default="outputs/runs/refetch_cross_ocean_adsb_raw_20260517")
    p.add_argument("--settings", default="config/settings.yaml")
    p.add_argument("--max-cases", type=int, default=10)
    p.add_argument("--pad-min", type=float, default=15.0)
    p.add_argument("--chunk-min", type=float, default=90.0)
    p.add_argument("--sleep-sec", type=float, default=3.0)
    p.add_argument("--cached", action="store_true", default=True)
    p.add_argument("--overwrite", action="store_true", help="Refetch even when the raw CSV already exists.")
    return p


def _to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _normalize_history(points: pd.DataFrame | None) -> pd.DataFrame:
    if points is None or len(points) == 0:
        return pd.DataFrame(columns=["timestamp", "icao24", "lat", "lon", "baroaltitude", "velocity", "heading"])
    d = points.copy()
    if "timestamp" not in d.columns:
        if "time" not in d.columns:
            raise RuntimeError(f"OpenSky history result has neither 'timestamp' nor 'time': {list(d.columns)}")
        if pd.api.types.is_numeric_dtype(d["time"]):
            d["timestamp"] = pd.to_datetime(d["time"], unit="s", utc=True, errors="coerce")
        else:
            d["timestamp"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
    else:
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    for col in ["lat", "lon", "baroaltitude", "velocity", "heading"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
        else:
            d[col] = np.nan
    if "icao24" not in d.columns:
        d["icao24"] = ""
    return d[["timestamp", "icao24", "lat", "lon", "baroaltitude", "velocity", "heading"]].dropna(
        subset=["timestamp"]
    ).sort_values("timestamp")


def minute_agg_adsb(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(
            columns=["minute_ts", "lat", "lon", "alt", "speed", "heading", "num_points_in_minute"]
        )
    d = raw.dropna(subset=["timestamp", "lat", "lon"]).copy()
    d["minute_ts"] = d["timestamp"].dt.floor("min")
    g = d.groupby("minute_ts", as_index=False).agg(
        lat=("lat", "median"),
        lon=("lon", "median"),
        alt=("baroaltitude", "median"),
        speed=("velocity", "median"),
        heading=("heading", "median"),
        num_points_in_minute=("timestamp", "size"),
    )
    return g.sort_values("minute_ts").reset_index(drop=True)


def _max_same_position_run(df: pd.DataFrame, pos_cols: list[str]) -> dict[str, object]:
    if df.empty:
        return {
            "max_frozen_run_min": 0,
            "frozen_start": "",
            "frozen_end": "",
            "frozen_lat": np.nan,
            "frozen_lon": np.nan,
            "frozen_alt": np.nan,
            "median_speed_during_run": np.nan,
        }
    d = df.sort_values("minute_ts").reset_index(drop=True).copy()
    rounded = d[pos_cols].round(6)
    same = rounded.eq(rounded.shift()).all(axis=1).to_numpy()
    best_len, best_start = 1, 0
    cur_len, cur_start = 1, 0
    for i in range(1, len(d)):
        if same[i]:
            cur_len += 1
        else:
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
            cur_len, cur_start = 1, i
    if cur_len > best_len:
        best_len, best_start = cur_len, cur_start
    seg = d.iloc[best_start : best_start + best_len]
    def first_float(col: str) -> float:
        if col not in seg:
            return np.nan
        vals = pd.to_numeric(seg[col], errors="coerce").dropna()
        return float(vals.iloc[0]) if len(vals) else np.nan

    return {
        "max_frozen_run_min": int(best_len),
        "frozen_start": str(seg["minute_ts"].iloc[0]),
        "frozen_end": str(seg["minute_ts"].iloc[-1]),
        "frozen_lat": first_float("lat"),
        "frozen_lon": first_float("lon"),
        "frozen_alt": first_float("alt"),
        "median_speed_during_run": float(pd.to_numeric(seg.get("speed"), errors="coerce").median())
        if "speed" in seg
        else np.nan,
    }


def _gap_stats(df: pd.DataFrame) -> dict[str, object]:
    if df.empty or len(df) < 2:
        return {"max_time_gap_min": np.nan, "gap_count_gt_1min": 0}
    dt = pd.to_datetime(df["minute_ts"], utc=True, errors="coerce").sort_values().diff().dt.total_seconds().div(60.0)
    return {
        "max_time_gap_min": float(dt.max()),
        "gap_count_gt_1min": int((dt > 1.5).sum()),
    }


def _load_case_windows(summary_csv: Path, max_cases: int) -> pd.DataFrame:
    summary = pd.read_csv(summary_csv)
    rows = []
    for _, row in summary.head(max_cases).iterrows():
        overlay = ROOT / str(row["overlay_csv"])
        if not overlay.exists():
            raise FileNotFoundError(f"Missing overlay CSV for {row['pair_id']}: {overlay}")
        odf = pd.read_csv(overlay, nrows=1)
        rows.append(
            {
                "pair_id": row["pair_id"],
                "icao24": str(row["icao24"]).lower(),
                "adsb_flight_id": row["adsb_flight_id"],
                "flight_start": _to_utc(odf["adsb_flight_start_ts"]).iloc[0],
                "flight_end": _to_utc(odf["adsb_flight_end_ts"]).iloc[0],
                "adsc_start": _to_utc(odf["adsc_start_ts"]).iloc[0],
                "adsc_end": _to_utc(odf["adsc_end_ts"]).iloc[0],
            }
        )
    return pd.DataFrame(rows)


def _fetch_history_chunked(trino, icao24: str, start: pd.Timestamp, stop: pd.Timestamp, chunk_min: float, cached: bool) -> pd.DataFrame:
    chunks = []
    cur = start
    step = pd.Timedelta(minutes=float(chunk_min))
    while cur < stop:
        nxt = min(cur + step, stop)
        points = trino.history(
            start=cur.to_pydatetime(),
            stop=nxt.to_pydatetime(),
            icao24=icao24,
            selected_columns=("time", "icao24", "lat", "lon", "baroaltitude", "velocity", "heading"),
            cached=bool(cached),
        )
        chunk = _normalize_history(points)
        if not chunk.empty:
            chunks.append(chunk)
        cur = nxt
    if not chunks:
        return _normalize_history(None)
    out = pd.concat(chunks, ignore_index=True)
    return out.drop_duplicates(subset=["timestamp", "icao24", "lat", "lon", "baroaltitude"]).sort_values("timestamp")


def main() -> int:
    args = build_parser().parse_args()
    out_dir = ROOT / args.out_dir
    raw_dir = out_dir / "raw_opensky"
    minute_dir = out_dir / "minute_refetched"
    raw_dir.mkdir(parents=True, exist_ok=True)
    minute_dir.mkdir(parents=True, exist_ok=True)

    settings = load_settings(args.settings)
    username = settings.get("opensky", {}).get("username") or os.getenv("OPENSKY_USERNAME")
    password = settings.get("opensky", {}).get("password") or os.getenv("OPENSKY_PASSWORD")
    if username:
        os.environ["OPENSKY_USERNAME"] = str(username).lower()
        os.environ["OPENSKY_TRINO_USERNAME"] = str(username).lower()
    if password:
        os.environ["OPENSKY_PASSWORD"] = str(password)
        os.environ["OPENSKY_TRINO_PASSWORD"] = str(password)

    from pyopensky.trino import Trino

    cases = _load_case_windows(ROOT / args.overlay_summary_csv, args.max_cases)
    cases.to_csv(out_dir / "refetch_plan.csv", index=False)
    old = pd.read_csv(ROOT / args.old_adsb_minute_csv, parse_dates=["minute_ts"])

    trino = Trino()
    rows = []
    pad = pd.Timedelta(minutes=float(args.pad_min))
    for i, case in cases.iterrows():
        pair_id = str(case["pair_id"])
        icao24 = str(case["icao24"]).lower()
        start = case["flight_start"] - pad
        stop = case["flight_end"] + pad
        print(f"[fetch] {i + 1}/{len(cases)} {pair_id} {icao24} {start}~{stop}", flush=True)
        raw_path = raw_dir / f"{pair_id}_raw_opensky.csv"
        try:
            if raw_path.exists() and not bool(args.overwrite):
                raw = pd.read_csv(raw_path)
                raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
                raw = _normalize_history(raw)
                print(f"[reuse] {pair_id} raw_rows={len(raw)}", flush=True)
            else:
                raw = _fetch_history_chunked(
                    trino=trino,
                    icao24=icao24,
                    start=start,
                    stop=stop,
                    chunk_min=float(args.chunk_min),
                    cached=bool(args.cached),
                )
            status = "ok"
            error = ""
        except Exception as exc:
            raw = _normalize_history(None)
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        raw.to_csv(raw_path, index=False)

        minute = minute_agg_adsb(raw)
        if not minute.empty:
            minute.insert(0, "pair_id", pair_id)
            minute.insert(1, "flight_id", case["adsb_flight_id"])
            minute.insert(2, "adsb_icao", icao24)
        minute_path = minute_dir / f"{pair_id}_minute_refetched.csv"
        minute.to_csv(minute_path, index=False)

        old_g = old[old["flight_id"].astype(str) == str(case["adsb_flight_id"])].copy()
        new_g = minute.copy()
        old_freeze = _max_same_position_run(old_g, ["lat", "lon", "alt"])
        new_freeze = _max_same_position_run(new_g, ["lat", "lon", "alt"])
        row = {
            "pair_id": pair_id,
            "icao24": icao24,
            "status": status,
            "error": error,
            "raw_rows": int(len(raw)),
            "old_minute_rows": int(len(old_g)),
            "new_minute_rows": int(len(new_g)),
            "raw_csv": str(raw_path.relative_to(ROOT)),
            "minute_csv": str(minute_path.relative_to(ROOT)),
        }
        row.update({f"old_{k}": v for k, v in old_freeze.items()})
        row.update({f"new_{k}": v for k, v in new_freeze.items()})
        row.update({f"old_{k}": v for k, v in _gap_stats(old_g).items()})
        row.update({f"new_{k}": v for k, v in _gap_stats(new_g).items()})
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / "refetch_quality_summary.csv", index=False)
        if float(args.sleep_sec) > 0 and i + 1 < len(cases):
            time.sleep(float(args.sleep_sec))

    print(f"[done] wrote {out_dir / 'refetch_quality_summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
