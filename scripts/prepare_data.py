from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.preprocessing import (
    ADSCGapPatternSampler,
    ADSCRawParser,
    ADSBMinuteAggregator,
    CruiseSegmentFilter,
    TrajectorySampleBuilder,
)

ICAO24_RE = re.compile(r"^[0-9a-fA-F]{6}$")


def _latest_dir(root: Path) -> Path | None:
    if not root.is_dir():
        return None
    dirs = sorted([p for p in root.iterdir() if p.is_dir() and list(p.glob("*.csv"))])
    return dirs[-1] if dirs else None


def _read_adsb_inputs(raw_dir: Path, limit_files: int | None) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*.csv"))
    if limit_files is not None and limit_files > 0:
        files = files[:limit_files]
    if not files:
        raise FileNotFoundError(f"No ADS-B CSV found in {raw_dir}")

    chunks = []
    used_ids = set()
    for path in files:
        df = pd.read_csv(path, low_memory=False)
        if "flight_id" not in df.columns:
            # Use filename stem as fallback flight_id to avoid collisions across files.
            df["flight_id"] = path.stem
        # Keep a standard ICAO field for ADS-C alignment.
        if "adsb_icao" not in df.columns:
            if "icao24" in df.columns:
                df["adsb_icao"] = df["icao24"].astype(str).str.lower()
            else:
                df["adsb_icao"] = ""
        # Keep one raw file per flight_id to enforce flight-level diversity.
        fid = str(df["flight_id"].iloc[0]) if len(df) else path.stem
        if fid in used_ids:
            continue
        used_ids.add(fid)
        chunks.append(df)
    if not chunks:
        raise RuntimeError("No valid ADS-B input files after flight_id deduplication.")
    return pd.concat(chunks, ignore_index=True)


def _extract_icao_from_flight_id(fid: str) -> str | None:
    parts = str(fid).split("-")
    if not parts:
        return None
    candidate = parts[-1].lower()
    if ICAO24_RE.match(candidate):
        return candidate
    return None


def _build_adsb_flight_icao_map(adsb_raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fid, g in adsb_raw.groupby("flight_id"):
        icao = None
        if "adsb_icao" in g.columns:
            vals = (
                g["adsb_icao"]
                .astype(str)
                .str.lower()
                .map(lambda x: x if ICAO24_RE.match(x) else None)
                .dropna()
                .unique()
                .tolist()
            )
            if vals:
                icao = vals[0]
        if icao is None:
            icao = _extract_icao_from_flight_id(str(fid))
        rows.append({"flight_id": str(fid), "adsb_icao": icao})
    return pd.DataFrame(rows)


def _summarize_aux_nonzero(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    position_nonzero = pd.to_numeric(frame.get("position_accuracy"), errors="coerce").fillna(0).gt(0)
    tag_any = (
        pd.to_numeric(frame.get("tag13_exists"), errors="coerce").fillna(0).gt(0)
        | pd.to_numeric(frame.get("tag14_exists"), errors="coerce").fillna(0).gt(0)
        | pd.to_numeric(frame.get("tag15_exists"), errors="coerce").fillna(0).gt(0)
        | pd.to_numeric(frame.get("tag16_exists"), errors="coerce").fillna(0).gt(0)
    )
    return float((position_nonzero | tag_any).mean())


def _parse_float_list(text: str) -> list[float]:
    vals = []
    for x in str(text).split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(float(x))
    return vals


def _parse_gap_buckets(text: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for seg in str(text).split(","):
        seg = seg.strip()
        if not seg:
            continue
        if "-" not in seg:
            v = int(seg)
            out.append((v, v))
            continue
        a, b = seg.split("-", 1)
        lo, hi = int(a), int(b)
        if hi < lo:
            lo, hi = hi, lo
        out.append((lo, hi))
    return out


def _gap_lengths(mask: pd.Series) -> list[int]:
    arr = pd.to_numeric(mask, errors="coerce").fillna(0.0).to_numpy()
    out: list[int] = []
    cur = 0
    for v in arr:
        if v < 0.5:
            cur += 1
        elif cur > 0:
            out.append(cur)
            cur = 0
    if cur > 0:
        out.append(cur)
    return out


def _angle_diff_deg(curr: pd.Series, prev: pd.Series) -> pd.Series:
    a = pd.to_numeric(curr, errors="coerce")
    b = pd.to_numeric(prev, errors="coerce")
    return ((a - b + 180.0) % 360.0) - 180.0


def _bearing_deg_from_latlon(lat: pd.Series, lon: pd.Series) -> pd.Series:
    lat2 = np.deg2rad(pd.to_numeric(lat, errors="coerce"))
    lon2 = np.deg2rad(pd.to_numeric(lon, errors="coerce"))
    lat1 = lat2.shift(1)
    lon1 = lon2.shift(1)
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    brg = (np.rad2deg(np.arctan2(x, y)) + 360.0) % 360.0
    return pd.Series(brg, index=lat.index)


def _clip_robust(series: pd.Series, q: float = 0.995) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    x = s.dropna()
    if x.empty:
        return s
    lo = float(x.quantile(1.0 - q))
    hi = float(x.quantile(q))
    return s.clip(lower=lo, upper=hi)


def _deterministic_sparse_mask(keys: pd.Series, rate: float) -> np.ndarray:
    p = float(np.clip(rate, 0.0, 1.0))
    if p <= 0.0:
        return np.zeros((len(keys),), dtype=bool)
    if p >= 1.0:
        return np.ones((len(keys),), dtype=bool)
    # Stable pseudo-randomness from row keys (no hidden RNG drift).
    h = pd.util.hash_pandas_object(keys.astype(str), index=False).to_numpy(dtype=np.uint64)
    u = (h % np.uint64(10_000_000)).astype(np.float64) / 10_000_000.0
    return u < p


def _compute_dt_since(mask: pd.Series, ts: pd.Series) -> pd.Series:
    m = mask.astype(bool).to_numpy()
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    out = np.full((len(m),), np.nan, dtype=np.float64)
    last = None
    for i in range(len(m)):
        if pd.isna(t.iloc[i]):
            continue
        if m[i]:
            out[i] = 0.0
            last = t.iloc[i]
        elif last is not None:
            out[i] = (t.iloc[i] - last).total_seconds() / 60.0
    return pd.Series(out, index=ts.index)


def _compute_dt_until(mask: pd.Series, ts: pd.Series) -> pd.Series:
    m = mask.astype(bool).to_numpy()
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    out = np.full((len(m),), np.nan, dtype=np.float64)
    nxt = None
    for i in range(len(m) - 1, -1, -1):
        if pd.isna(t.iloc[i]):
            continue
        if m[i]:
            out[i] = 0.0
            nxt = t.iloc[i]
        elif nxt is not None:
            out[i] = (nxt - t.iloc[i]).total_seconds() / 60.0
    return pd.Series(out, index=ts.index)


def _build_adsc_minute_table(adsc_df: pd.DataFrame) -> pd.DataFrame:
    if adsc_df.empty:
        return pd.DataFrame(
            columns=[
                "adsb_icao",
                "minute_ts",
                "adsc_minute_aligned",
                "position_accuracy_aligned",
                "tag13_exists_aligned",
                "tag14_exists_aligned",
                "tag15_exists_aligned",
                "tag16_exists_aligned",
            ]
        )
    x = adsc_df.copy()
    x["adsb_icao"] = x["flight_id"].astype(str).str.lower()
    x["minute_ts"] = pd.to_datetime(x["timestamp"], utc=True, errors="coerce").dt.floor("min")
    x = x.dropna(subset=["adsb_icao", "minute_ts"]).copy()
    out = (
        x.groupby(["adsb_icao", "minute_ts"], as_index=False)
        .agg(
            position_accuracy_aligned=("position_accuracy", "mean"),
            tag13_exists_aligned=("tag13_exists", "max"),
            tag14_exists_aligned=("tag14_exists", "max"),
            tag15_exists_aligned=("tag15_exists", "max"),
            tag16_exists_aligned=("tag16_exists", "max"),
        )
        .sort_values(["adsb_icao", "minute_ts"])
        .reset_index(drop=True)
    )
    out["adsc_minute_aligned"] = 1
    return out


def _inject_tag_proxy_features(adsb_cruise: pd.DataFrame, adsc_df: pd.DataFrame, stats: dict) -> tuple[pd.DataFrame, dict]:
    out = adsb_cruise.copy()
    out["minute_ts"] = pd.to_datetime(out["minute_ts"], utc=True, errors="coerce")
    out["adsb_icao"] = out["adsb_icao"].astype(str).str.lower()
    out = out.sort_values(["flight_id", "minute_ts"]).reset_index(drop=True)

    adsc_min = _build_adsc_minute_table(adsc_df)
    out = out.merge(adsc_min, on=["adsb_icao", "minute_ts"], how="left")
    out["adsc_minute_aligned"] = pd.to_numeric(out["adsc_minute_aligned"], errors="coerce").fillna(0).astype(int)
    aligned_mask = out["adsc_minute_aligned"].eq(1)

    # Keep backward-compatible raw columns, but now minute-aligned (not flight-level broadcast).
    for c in ["tag13_exists", "tag14_exists", "tag15_exists", "tag16_exists"]:
        ca = f"{c}_aligned"
        out[c] = pd.to_numeric(out.get(ca, np.nan), errors="coerce")
    out["position_accuracy"] = pd.to_numeric(out.get("position_accuracy_aligned", np.nan), errors="coerce")
    out["adsc_aligned"] = aligned_mask

    # Base proxy dynamics from ADS-B (minute-level).
    out["ground_speed_proxy"] = pd.to_numeric(out.get("speed", np.nan), errors="coerce")
    brg = pd.Series(np.nan, index=out.index, dtype=np.float64)
    for _, g in out.groupby("flight_id", sort=False):
        brg.loc[g.index] = _bearing_deg_from_latlon(g["lat"], g["lon"]).to_numpy()
    hd = pd.to_numeric(out.get("heading", np.nan), errors="coerce")
    out["track_proxy"] = hd.where(hd.notna(), brg)
    out["track_proxy"] = out["track_proxy"].mod(360.0)
    out["track_sin"] = np.sin(np.deg2rad(out["track_proxy"]))
    out["track_cos"] = np.cos(np.deg2rad(out["track_proxy"]))

    # Per-minute rates/deltas with robust clipping.
    dt_min = out.groupby("flight_id", sort=False)["minute_ts"].diff().dt.total_seconds().div(60.0)
    dt_min = dt_min.where(dt_min > 0, 1.0).fillna(1.0)
    d_alt = out.groupby("flight_id", sort=False)["alt"].diff()
    out["vertical_rate_proxy"] = _clip_robust(d_alt / dt_min)
    d_gs = out.groupby("flight_id", sort=False)["ground_speed_proxy"].diff() / dt_min
    out["delta_ground_speed_proxy"] = _clip_robust(d_gs)
    prev_track = out.groupby("flight_id", sort=False)["track_proxy"].shift(1)
    d_track = _angle_diff_deg(out["track_proxy"], prev_track) / dt_min
    out["delta_track_proxy"] = _clip_robust(d_track)
    d_vr = out.groupby("flight_id", sort=False)["vertical_rate_proxy"].diff() / dt_min
    out["delta_vertical_rate_proxy"] = _clip_robust(d_vr)

    # Optional heading proxies; currently supported from ADS-B heading.
    out["heading_proxy"] = hd.where(hd.notna(), out["track_proxy"]).mod(360.0)
    prev_heading = out.groupby("flight_id", sort=False)["heading_proxy"].shift(1)
    out["delta_heading_proxy"] = _clip_robust(_angle_diff_deg(out["heading_proxy"], prev_heading) / dt_min)
    out["mach_proxy"] = np.nan
    out["mach_proxy_reliable"] = 0

    # Sparse availability masks: prefer real minute-aligned Tag14/Tag15 presence.
    tag_rates = stats.get("tag_exist_rate", {}) if isinstance(stats, dict) else {}
    r14 = float(tag_rates.get("tag14_exists", 0.0))
    r15 = float(tag_rates.get("tag15_exists", 0.0))
    row_keys = out["flight_id"].astype(str) + "|" + out["minute_ts"].astype(str)
    sim14 = _deterministic_sparse_mask(row_keys + "|t14", rate=r14)
    sim15 = _deterministic_sparse_mask(row_keys + "|t15", rate=r15)
    real14 = pd.to_numeric(out.get("tag14_exists_aligned", np.nan), errors="coerce").fillna(0.0).gt(0.5).to_numpy()
    real15 = pd.to_numeric(out.get("tag15_exists_aligned", np.nan), errors="coerce").fillna(0.0).gt(0.5).to_numpy()
    am = aligned_mask.to_numpy()
    out["tag14_proxy_available"] = np.where(am, real14, sim14).astype(int)
    out["tag15_proxy_available"] = np.where(am, real15, sim15).astype(int)
    out["tag14_proxy_available_source"] = np.where(am, "aligned", "sim_rate")
    out["tag15_proxy_available_source"] = np.where(am, "aligned", "sim_rate")

    # Freshness / recency over sparse availability.
    out["dt_since_tag14_proxy_obs"] = np.nan
    out["dt_since_tag15_proxy_obs"] = np.nan
    out["dt_until_next_tag14_proxy_obs"] = np.nan
    out["dt_until_next_tag15_proxy_obs"] = np.nan
    for fid, g in out.groupby("flight_id", sort=False):
        idx = g.index
        out.loc[idx, "dt_since_tag14_proxy_obs"] = _compute_dt_since(g["tag14_proxy_available"] > 0, g["minute_ts"]).to_numpy()
        out.loc[idx, "dt_since_tag15_proxy_obs"] = _compute_dt_since(g["tag15_proxy_available"] > 0, g["minute_ts"]).to_numpy()
        out.loc[idx, "dt_until_next_tag14_proxy_obs"] = _compute_dt_until(g["tag14_proxy_available"] > 0, g["minute_ts"]).to_numpy()
        out.loc[idx, "dt_until_next_tag15_proxy_obs"] = _compute_dt_until(g["tag15_proxy_available"] > 0, g["minute_ts"]).to_numpy()

    # Tag14/Tag15 scoped proxy values (NaN when unavailable, mask keeps validity semantics explicit).
    tag14_value_cols = [
        "ground_speed_proxy",
        "track_proxy",
        "track_sin",
        "track_cos",
        "vertical_rate_proxy",
        "delta_ground_speed_proxy",
        "delta_track_proxy",
        "delta_vertical_rate_proxy",
    ]
    tag15_value_cols = ["heading_proxy", "delta_heading_proxy", "mach_proxy"]
    m14 = out["tag14_proxy_available"].eq(1)
    m15 = out["tag15_proxy_available"].eq(1)
    for c in tag14_value_cols:
        out[f"{c}_tag14"] = out[c].where(m14, np.nan)
    for c in tag15_value_cols:
        out[f"{c}_tag15"] = out[c].where(m15, np.nan)

    audit = {
        "tag14_rate_from_stats": r14,
        "tag15_rate_from_stats": r15,
        "rows_total": int(len(out)),
        "adsc_minute_aligned_ratio": float(aligned_mask.mean()) if len(out) else 0.0,
        "tag14_proxy_available_ratio": float(out["tag14_proxy_available"].mean()) if len(out) else 0.0,
        "tag15_proxy_available_ratio": float(out["tag15_proxy_available"].mean()) if len(out) else 0.0,
        "tag14_proxy_source_aligned_ratio": float((out["tag14_proxy_available_source"] == "aligned").mean()) if len(out) else 0.0,
        "tag15_proxy_source_aligned_ratio": float((out["tag15_proxy_available_source"] == "aligned").mean()) if len(out) else 0.0,
    }
    return out, audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare MVP training samples for ADS-B/ADS-C recovery.")
    parser.add_argument("--adsb-raw-dir", default=None, help="ADS-B raw points dir. default: latest under outputs/points/raw")
    parser.add_argument("--adsc-decoded", default="ads-c_data/adsc_decoded.txt", help="ADS-C decoded text file")
    parser.add_argument("--output-dir", default="outputs/mvp", help="Output root dir")
    parser.add_argument("--window-size", type=int, default=250)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--limit-files", type=int, default=-1, help="Use -1 to read all files")
    parser.add_argument("--num-augment-per-flight", type=int, default=3)
    parser.add_argument("--min-cruise-minutes", type=int, default=30)
    parser.add_argument("--max-abs-vertical-rate", type=float, default=300.0)
    parser.add_argument("--max-speed-delta", type=float, default=30.0)
    parser.add_argument("--max-heading-rate", type=float, default=5.0)
    parser.add_argument("--min-gap-minutes", type=int, default=3)
    parser.add_argument("--max-gap-minutes", type=int, default=180)
    parser.add_argument(
        "--max-time-gap-minutes",
        type=float,
        default=5.0,
        help="Split one flight into continuous segments when minute_ts jump exceeds this threshold.",
    )
    parser.add_argument("--target-unique-flights", type=int, default=1000)
    parser.add_argument(
        "--obs-sim-mode",
        default="stage1",
        choices=["stage1", "stage2", "stage2_medium", "stage2_irregular_medium", "stage3"],
    )
    # Stage defaults updated from latest ADS-C anchor-interval distribution
    # audit (icao24+day, dt<180, min2). See outputs/runs/adsc_interval_*.
    parser.add_argument("--stage1-mask-ratios", default="0.08,0.12,0.18")
    parser.add_argument("--stage1-gap-buckets", default="1-3,3-6,6-10")
    parser.add_argument("--stage2-mask-ratios", default="0.18,0.28,0.38")
    parser.add_argument("--stage2-gap-buckets", default="10-20,20-30,30-45")
    parser.add_argument("--stage2-medium-mask-ratios", default="0.60,0.72,0.85")
    parser.add_argument("--stage2-medium-gap-buckets", default="25-35,35-48,48-62")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.adsb_raw_dir:
        adsb_raw_dir = Path(args.adsb_raw_dir)
    else:
        latest = _latest_dir(Path("outputs/points/raw"))
        if latest is None:
            raise FileNotFoundError("No run directory found in outputs/points/raw")
        adsb_raw_dir = latest

    limit_files = None if int(args.limit_files) <= 0 else int(args.limit_files)
    adsb_raw = _read_adsb_inputs(adsb_raw_dir, limit_files)
    unique_flights = int(adsb_raw["flight_id"].astype(str).nunique())
    target_unique = int(args.target_unique_flights)
    if target_unique > 0 and unique_flights < target_unique:
        raise RuntimeError(
            f"Insufficient unique flights for training: got {unique_flights}, need at least {target_unique}. "
            "Run scripts/fetch_points_for_training.py to fetch more flight tracks."
        )
    aggregator = ADSBMinuteAggregator()
    adsb_minute = aggregator.aggregate(adsb_raw)

    # Build robust flight_id -> ICAO mapping used to align ADS-C features.
    flight_icao_map = _build_adsb_flight_icao_map(adsb_raw)
    adsb_minute = adsb_minute.merge(flight_icao_map, on="flight_id", how="left")
    adsb_minute.to_parquet(output_root / "adsb_minute.parquet", index=False)

    cruise_filter = CruiseSegmentFilter(
        min_cruise_minutes=args.min_cruise_minutes,
        max_abs_vertical_rate=args.max_abs_vertical_rate,
        max_speed_delta=args.max_speed_delta,
        max_heading_rate=args.max_heading_rate,
    )
    adsb_marked = cruise_filter.mark_cruise(adsb_minute)
    adsb_cruise = adsb_marked[adsb_marked["is_cruise"].eq(1)].copy()
    if adsb_cruise.empty:
        # Keep pipeline runnable even when strict thresholds yield no cruise rows.
        adsb_cruise = adsb_minute.copy()
        adsb_cruise["is_cruise"] = 0
        print("[warn] no_cruise_rows_found -> fallback_to_all_minute_rows")
    adsb_cruise.to_parquet(output_root / "adsb_cruise.parquet", index=False)

    adsc_parser = ADSCRawParser()
    adsc_df = adsc_parser.parse_file(args.adsc_decoded)
    adsc_df.to_parquet(output_root / "adsc_parsed.parquet", index=False)

    sampler = ADSCGapPatternSampler(
        min_gap_minutes=args.min_gap_minutes,
        max_gap_minutes=args.max_gap_minutes,
    )
    stats = sampler.fit(adsc_df)
    (output_root / "adsc_gap_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    sample_builder = TrajectorySampleBuilder(window_size=args.window_size, stride=args.stride)

    # Baseline (before fix): direct flight_id equality, expected to fail for most flights.
    legacy_match = adsb_cruise[["flight_id"]].drop_duplicates().copy()
    legacy_match["legacy_aligned"] = legacy_match["flight_id"].isin(set(adsc_df["flight_id"].astype(str)))
    legacy_ratio = float(legacy_match["legacy_aligned"].mean()) if len(legacy_match) else 0.0

    # Build minute-level ADS-C alignment + sparse Tag14/15-style proxy features.
    # Important: do NOT broadcast flight-level mean/max to all minutes.
    adsb_cruise, proxy_audit = _inject_tag_proxy_features(adsb_cruise, adsc_df, stats)
    (output_root / "tag_proxy_sparse_audit.json").write_text(
        json.dumps(proxy_audit, indent=2),
        encoding="utf-8",
    )

    sample_frames = []
    stage1_cfg = {
        "mask_ratios": _parse_float_list(args.stage1_mask_ratios),
        "gap_buckets": _parse_gap_buckets(args.stage1_gap_buckets),
    }
    stage2_cfg = {
        "mask_ratios": _parse_float_list(args.stage2_mask_ratios),
        "gap_buckets": _parse_gap_buckets(args.stage2_gap_buckets),
    }
    stage2_medium_cfg = {
        "mask_ratios": _parse_float_list(args.stage2_medium_mask_ratios),
        "gap_buckets": _parse_gap_buckets(str(args.stage2_medium_gap_buckets)),
    }
    for fid, g in adsb_cruise.groupby("flight_id"):
        g = g.sort_values("minute_ts").reset_index(drop=True)
        if len(g) < 10:
            continue
        for aug in range(max(1, int(args.num_augment_per_flight))):
            mask = sampler.simulate_observation(
                length=len(g),
                mode=str(args.obs_sim_mode),
                stats=stats,
                stage1_cfg=stage1_cfg,
                stage2_cfg=stage2_cfg,
                stage2_medium_cfg=stage2_medium_cfg,
            )
            samples = sample_builder.build_for_flight(
                g,
                mask,
                sample_prefix=f"{fid}_a{aug}",
                max_time_gap_minutes=float(args.max_time_gap_minutes),
            )
            sample_frames.extend(samples)

    if not sample_frames:
        raise RuntimeError(
            "No sample built. Try increasing --limit-files or check if ADS-B inputs have valid minute tracks."
        )

    samples_df = pd.concat(sample_frames, ignore_index=True)
    samples_df.to_parquet(output_root / "samples.parquet", index=False)

    # Auditing stats required by alignment fix task.
    sample_aux_ratio = (
        samples_df.groupby("sample_id", as_index=False)
        .apply(_summarize_aux_nonzero, include_groups=False)
        .rename(columns={None: "aux_nonzero_ratio"})
    )
    sample_aux_ratio.to_csv(output_root / "adsc_aux_nonzero_ratio_by_sample.csv", index=False)
    post_nonzero_mean = float(sample_aux_ratio["aux_nonzero_ratio"].mean()) if len(sample_aux_ratio) else 0.0
    proxy_cols = [
        "tag14_proxy_available",
        "tag15_proxy_available",
        "ground_speed_proxy_tag14",
        "track_proxy_tag14",
        "vertical_rate_proxy_tag14",
        "heading_proxy_tag15",
        "dt_since_tag14_proxy_obs",
        "dt_since_tag15_proxy_obs",
        "dt_until_next_tag14_proxy_obs",
        "dt_until_next_tag15_proxy_obs",
    ]
    proxy_rows = []
    for c in proxy_cols:
        if c not in samples_df.columns:
            continue
        s = pd.to_numeric(samples_df[c], errors="coerce")
        proxy_rows.append(
            {
                "column": c,
                "non_null_ratio": float(s.notna().mean()),
                "non_zero_ratio": float((s.fillna(0.0) != 0.0).mean()),
                "mean": float(s.mean()) if s.notna().any() else float("nan"),
                "std": float(s.std(ddof=0)) if s.notna().any() else float("nan"),
            }
        )
    pd.DataFrame(proxy_rows).to_csv(output_root / "tag_proxy_feature_summary.csv", index=False)

    # Observation simulation audit (mask ratio + contiguous gap behavior).
    sample_mask_ratio = (
        samples_df.groupby("sample_id")["obs_mask"]
        .apply(lambda x: float((pd.to_numeric(x, errors="coerce").fillna(0.0) < 0.5).mean()))
        .reset_index(name="missing_ratio")
    )
    gap_lens = []
    for _, sg in samples_df.groupby("sample_id"):
        gap_lens.extend(_gap_lengths(sg["obs_mask"]))
    sim_audit = {
        "mode": str(args.obs_sim_mode),
        "max_time_gap_minutes": float(args.max_time_gap_minutes),
        "sample_missing_ratio_mean": float(sample_mask_ratio["missing_ratio"].mean()) if len(sample_mask_ratio) else 0.0,
        "sample_missing_ratio_std": float(sample_mask_ratio["missing_ratio"].std(ddof=0)) if len(sample_mask_ratio) else 0.0,
        "gap_length_mean": float(pd.Series(gap_lens).mean()) if gap_lens else 0.0,
        "gap_length_q50": float(pd.Series(gap_lens).quantile(0.5)) if gap_lens else 0.0,
        "gap_length_q90": float(pd.Series(gap_lens).quantile(0.9)) if gap_lens else 0.0,
        "num_gaps": int(len(gap_lens)),
        "stage1_cfg": stage1_cfg,
        "stage2_cfg": stage2_cfg,
    }
    (output_root / "obs_sim_audit.json").write_text(json.dumps(sim_audit, indent=2), encoding="utf-8")

    # Time continuity audit per sample (after segmentation + windowing).
    sample_time_gap = (
        samples_df.sort_values(["sample_id", "minute_ts"])
        .groupby("sample_id")["minute_ts"]
        .apply(
            lambda x: float(
                pd.to_datetime(x, utc=True, errors="coerce")
                .diff()
                .dt.total_seconds()
                .div(60.0)
                .fillna(1.0)
                .max()
            )
        )
        .reset_index(name="max_step_minutes")
    )
    (output_root / "sample_time_gap_audit.csv").write_text(sample_time_gap.to_csv(index=False), encoding="utf-8")
    gap_over = int((sample_time_gap["max_step_minutes"] > float(args.max_time_gap_minutes)).sum())

    flight_align = (
        adsb_cruise.groupby("flight_id", as_index=False)
        .agg(
            adsb_icao=("adsb_icao", "first"),
            adsc_aligned=("adsc_aligned", "max"),
        )
    )
    flight_align.to_csv(output_root / "adsc_alignment_by_flight.csv", index=False)
    aligned_flight_ratio = float(flight_align["adsc_aligned"].mean()) if len(flight_align) else 0.0

    print("[align] examples_adsb_flight_id:", ", ".join(flight_align["flight_id"].astype(str).head(3).tolist()))
    print("[align] examples_adsb_icao:", ", ".join(flight_align["adsb_icao"].fillna("NA").astype(str).head(3).tolist()))
    print("[align] examples_adsc_flight_id:", ", ".join(adsc_df["flight_id"].astype(str).drop_duplicates().head(3).tolist()))
    print(f"[align][before] flight_id_direct_match_ratio={legacy_ratio:.4f}")
    print(f"[align][after] aligned_flight_ratio={aligned_flight_ratio:.4f}")
    print(f"[align][before] sample_aux_nonzero_ratio_mean=0.0000")
    print(f"[align][after] sample_aux_nonzero_ratio_mean={post_nonzero_mean:.4f}")
    print(
        "[tag_proxy] "
        f"aligned_row_ratio={proxy_audit.get('adsc_minute_aligned_ratio', 0.0):.4f} "
        f"tag14_available_ratio={proxy_audit.get('tag14_proxy_available_ratio', 0.0):.4f} "
        f"tag15_available_ratio={proxy_audit.get('tag15_proxy_available_ratio', 0.0):.4f} "
        f"tag14_source_aligned_ratio={proxy_audit.get('tag14_proxy_source_aligned_ratio', 0.0):.4f} "
        f"tag15_source_aligned_ratio={proxy_audit.get('tag15_proxy_source_aligned_ratio', 0.0):.4f}"
    )
    print(
        f"[obs_sim] mode={sim_audit['mode']} "
        f"missing_ratio_mean={sim_audit['sample_missing_ratio_mean']:.4f} "
        f"gap_len_mean={sim_audit['gap_length_mean']:.2f} "
        f"num_gaps={sim_audit['num_gaps']}"
    )
    print(
        f"[time_gap] threshold={float(args.max_time_gap_minutes):.1f} "
        f"samples_over_threshold={gap_over}/{len(sample_time_gap)} "
        f"max_step_minutes_max={float(sample_time_gap['max_step_minutes'].max()) if len(sample_time_gap) else 0.0:.1f}"
    )

    print(f"[ok] adsb_raw_dir={adsb_raw_dir}")
    print(f"[ok] adsb_minute_rows={len(adsb_minute)}")
    print(f"[ok] adsb_cruise_rows={len(adsb_cruise)}")
    print(f"[ok] adsc_rows={len(adsc_df)}")
    print(f"[ok] sample_rows={len(samples_df)} sample_count={samples_df['sample_id'].nunique()}")
    print(f"[ok] tag_proxy_sparse_audit={output_root / 'tag_proxy_sparse_audit.json'}")
    print(f"[ok] tag_proxy_feature_summary={output_root / 'tag_proxy_feature_summary.csv'}")
    print(f"[ok] output_dir={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
