from __future__ import annotations

import argparse
import math
import re
import os
import time
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class FlightFile:
    path: Path
    sample_id: str
    flight_id: str
    flight_date: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ADS-C vs ADS-B raw alignment sanity check (no model).")
    parser.add_argument(
        "--source-dir",
        type=str,
        default="outputs/adsc/adsc_flight_point/2026-01-13-2110/all",
        help="Directory containing per-flight CSV with mixed adsb/adsc rows.",
    )
    parser.add_argument(
        "--adsc-decoded-csv",
        type=str,
        default="",
        help="Optional ADS-C decoded CSV (icao24, day, timestamp, latitude, longitude, altitude_m).",
    )
    parser.add_argument(
        "--adsb-minute-parquet",
        type=str,
        default="outputs/mvp_adsb3864_20260322/adsb_minute.parquet",
        help="ADS-B minute parquet with adsb_icao and minute_ts.",
    )
    parser.add_argument(
        "--adsb-flight-matches-csv",
        type=str,
        default="",
        help="Optional CSV from OpenSky flightlist matching (day+icao with firstSeen/lastSeen).",
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=0,
        help="Limit number of matched rows to fetch from OpenSky (0=all).",
    )
    parser.add_argument(
        "--matches-offset",
        type=int,
        default=0,
        help="Skip first N rows from adsb_flight_matches_csv before limiting.",
    )
    parser.add_argument(
        "--adsb-fetch-pad-min",
        type=float,
        default=0.0,
        help="Padding minutes to extend OpenSky history fetch window around firstSeen/lastSeen.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/runs/adsc_adsb_raw_alignment_check_20260329",
        help="Output directory.",
    )
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to plot.")
    parser.add_argument("--min-adsc-points", type=int, default=2, help="Minimum ADS-C points required.")
    parser.add_argument("--adsb-gap-break-min", type=float, default=10.0, help="Break ADS-B line if gap >= N minutes.")
    parser.add_argument(
        "--local-window-min",
        type=float,
        default=90.0,
        help="Local window minutes around ADS-C span for focused plots.",
    )
    parser.add_argument(
        "--fetch-missing-adsb",
        action="store_true",
        help="Fetch ADS-B from OpenSky if no local candidate found (single-thread).",
    )
    parser.add_argument(
        "--opensky-pad-min",
        type=float,
        default=120.0,
        help="Padding minutes before/after ADS-C window when fetching ADS-B.",
    )
    parser.add_argument(
        "--max-accepted-plots",
        type=int,
        default=10,
        help="Number of high-confidence samples to plot.",
    )
    parser.add_argument(
        "--max-rejected-plots",
        type=int,
        default=5,
        help="Number of rejected samples to plot.",
    )
    parser.add_argument(
        "--skip-tier-filter",
        action="store_true",
        help="Skip tier filtering and plot top-N from all matched records.",
    )
    return parser.parse_args()


def parse_flight_meta(path: Path) -> FlightFile | None:
    m = re.match(r"^(\d{4}-\d{2}-\d{2})-(.+)-\d+\.csv$", path.name)
    if not m:
        return None
    return FlightFile(path=path, sample_id=path.stem, flight_id=m.group(2), flight_date=m.group(1))


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371000.0 * 2.0 * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    b = math.degrees(math.atan2(x, y))
    return (b + 360.0) % 360.0


def angle_diff_deg(a: float, b: float) -> float:
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def load_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp", "lat", "lon", "baroaltitude", "source"])
    out = out.sort_values("timestamp").reset_index(drop=True)
    adsc = out[out["source"] == "adsc"].copy()
    adsb = out[out["source"] == "adsb"].copy()
    return adsc, adsb


def minute_agg_adsb(adsb: pd.DataFrame) -> pd.DataFrame:
    d = adsb.copy()
    d["minute_ts"] = d["timestamp"].dt.floor("min")
    g = (
        d.groupby("minute_ts", as_index=False)[["lat", "lon", "baroaltitude"]]
        .median()
        .rename(columns={"baroaltitude": "alt"})
    )
    return g.sort_values("minute_ts").reset_index(drop=True)


def interval_overlap_min(
    a0: pd.Timestamp, a1: pd.Timestamp, b0: pd.Timestamp, b1: pd.Timestamp
) -> float:
    left = max(a0, b0)
    right = min(a1, b1)
    if right <= left:
        return 0.0
    return float((right - left).total_seconds() / 60.0)


def nearest_interval_gap_min(
    a0: pd.Timestamp, a1: pd.Timestamp, b0: pd.Timestamp, b1: pd.Timestamp
) -> float:
    if interval_overlap_min(a0, a1, b0, b1) > 0:
        return 0.0
    if a1 < b0:
        return float((b0 - a1).total_seconds() / 60.0)
    return float((a0 - b1).total_seconds() / 60.0)


def nearest_neighbor_checks(adsc: pd.DataFrame, adsb_min: pd.DataFrame) -> pd.DataFrame:
    x = adsc[["timestamp", "lat", "lon", "baroaltitude"]].copy()
    y = adsb_min.rename(columns={"minute_ts": "adsb_ts", "alt": "baroaltitude"}).copy()
    x_ts = pd.to_datetime(x["timestamp"], utc=True, errors="coerce").dt.tz_convert("UTC").dt.tz_localize(None)
    y_ts = pd.to_datetime(y["adsb_ts"], utc=True, errors="coerce").dt.tz_convert("UTC").dt.tz_localize(None)
    x["timestamp"] = x_ts.values.astype("datetime64[ns]")
    y["adsb_ts"] = y_ts.values.astype("datetime64[ns]")
    x = x.dropna(subset=["timestamp"]).sort_values("timestamp")
    y = y.dropna(subset=["adsb_ts"]).sort_values("adsb_ts")
    if x.empty or y.empty:
        return pd.DataFrame()
    merged = pd.merge_asof(
        x,
        y,
        left_on="timestamp",
        right_on="adsb_ts",
        suffixes=("_adsc", "_adsb"),
        direction="nearest",
    )
    merged = merged.dropna(subset=["lat_adsb", "lon_adsb", "baroaltitude_adsb"]).copy()
    if merged.empty:
        return merged
    merged["time_diff_min"] = (merged["timestamp"] - merged["adsb_ts"]).dt.total_seconds().abs() / 60.0
    merged["spatial_distance_m"] = [
        haversine_m(a, b, c, d)
        for a, b, c, d in zip(
            merged["lat_adsc"].to_numpy(),
            merged["lon_adsc"].to_numpy(),
            merged["lat_adsb"].to_numpy(),
            merged["lon_adsb"].to_numpy(),
        )
    ]
    merged["alt_diff"] = (merged["baroaltitude_adsc"] - merged["baroaltitude_adsb"]).abs()
    return merged


def summarize_numeric(nn: pd.DataFrame) -> dict:
    if nn.empty:
        return {
            "matched_points": 0,
            "mean_time_diff_minutes": np.nan,
            "p90_time_diff_minutes": np.nan,
            "mean_spatial_distance_m": np.nan,
            "p90_spatial_distance_m": np.nan,
            "mean_alt_diff": np.nan,
            "p90_alt_diff": np.nan,
        }
    return {
        "matched_points": int(len(nn)),
        "mean_time_diff_minutes": float(nn["time_diff_min"].mean()),
        "p90_time_diff_minutes": float(nn["time_diff_min"].quantile(0.90)),
        "mean_spatial_distance_m": float(nn["spatial_distance_m"].mean()),
        "p90_spatial_distance_m": float(nn["spatial_distance_m"].quantile(0.90)),
        "mean_alt_diff": float(nn["alt_diff"].mean()),
        "p90_alt_diff": float(nn["alt_diff"].quantile(0.90)),
    }


def confidence_from_metrics(overlap_min: float, nearest_gap: float, p90_dist_m: float) -> tuple[str, bool]:
    suspicious = False
    if overlap_min >= 20 and nearest_gap <= 5 and (np.isnan(p90_dist_m) or p90_dist_m < 100000):
        return "high", False
    if overlap_min >= 5 and nearest_gap <= 30 and (np.isnan(p90_dist_m) or p90_dist_m < 250000):
        return "medium", False
    suspicious = True
    return "low", suspicious


def visual_judgement_row(
    numeric: dict, confidence: str, suspicious: bool, chosen_same_id: bool, adsb_match_count: int
) -> dict:
    matched = int(numeric["matched_points"])
    p90_time = numeric["p90_time_diff_minutes"]
    p90_dist = numeric["p90_spatial_distance_m"]
    p90_alt = numeric["p90_alt_diff"]
    time_ok = bool(matched > 0 and (np.isnan(p90_time) or p90_time <= 240.0))
    planar_ok = bool(matched > 0 and (np.isnan(p90_dist) or p90_dist <= 250000.0))
    alt_ok = bool(matched > 0 and (np.isnan(p90_alt) or p90_alt <= 2500.0))
    same_flight = (chosen_same_id and adsb_match_count == 1) or (confidence in {"high", "medium"} and not suspicious)
    issue = ""
    if not time_ok:
        issue = "time_offset_or_low_overlap"
    elif not planar_ok:
        issue = "planar_mismatch"
    elif not alt_ok:
        issue = "altitude_mismatch_or_unit_issue"
    elif suspicious:
        issue = "low_confidence_match"
    usable = same_flight and time_ok and planar_ok
    return {
        "same_flight_likely": "yes" if same_flight else "no",
        "time_alignment_ok": "yes" if time_ok else "no",
        "planar_alignment_ok": "yes" if planar_ok else "no",
        "altitude_alignment_ok": "yes" if alt_ok else "no",
        "obvious_issue_type": issue,
        "usable_for_recovery_experiment": "yes" if usable else "no",
    }


def pick_samples(records: list[dict], n: int) -> list[dict]:
    d = pd.DataFrame(records)
    if d.empty:
        return []
    # prefer richer ADS-C anchors and longer ADS-C span
    d = d.copy()
    d["adsc_span_min"] = (
        (pd.to_datetime(d["adsc_end_time"]) - pd.to_datetime(d["adsc_start_time"])).dt.total_seconds() / 60.0
    )
    d = d.sort_values(["adsc_points", "adsc_span_min", "adsb_points"], ascending=[False, False, False])
    return d.head(n).to_dict(orient="records")


def minute_overlap_count(adsc: pd.DataFrame, adsb_min: pd.DataFrame) -> float:
    if adsc.empty or adsb_min.empty:
        return 0.0
    s, e = adsc["timestamp"].min(), adsc["timestamp"].max()
    return float(((adsb_min["minute_ts"] >= s.floor("min")) & (adsb_min["minute_ts"] <= e.floor("min"))).sum())


def _adsb_range(adsb_min: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp, float]:
    if adsb_min.empty:
        return pd.NaT, pd.NaT, 0.0
    s = adsb_min["minute_ts"].min()
    e = adsb_min["minute_ts"].max()
    dur = float((e - s).total_seconds() / 60.0) if pd.notna(s) and pd.notna(e) else 0.0
    return s, e, dur


def dedupe_candidates(candidates: list[FlightFile], adsb_cache: dict[str, pd.DataFrame]) -> tuple[list[FlightFile], list[dict]]:
    if len(candidates) <= 1:
        return candidates, []
    rows = []
    for c in candidates:
        adsb_min = adsb_cache.get(c.sample_id)
        if adsb_min is None:
            continue
        s, e, dur = _adsb_range(adsb_min)
        rows.append(
            {
                "sample_id": c.sample_id,
                "flight_id": c.flight_id,
                "date": c.flight_date,
                "start": s,
                "end": e,
                "duration_min": dur,
                "points": int(len(adsb_min)),
            }
        )
    if len(rows) <= 1:
        return candidates, []
    rows = sorted(rows, key=lambda r: (r["duration_min"], r["points"]), reverse=True)
    keep = []
    dropped = []
    for r in rows:
        keep_it = True
        for k in keep:
            # overlap ratio of shorter range
            left = max(r["start"], k["start"])
            right = min(r["end"], k["end"])
            overlap = max(0.0, (right - left).total_seconds() / 60.0) if pd.notna(left) and pd.notna(right) else 0.0
            shorter = min(r["duration_min"], k["duration_min"]) or 1.0
            ratio = overlap / shorter
            if ratio >= 0.8:
                keep_it = False
                dropped.append(
                    {
                        "dropped_sample_id": r["sample_id"],
                        "kept_sample_id": k["sample_id"],
                        "overlap_ratio": ratio,
                        "dropped_duration_min": r["duration_min"],
                        "kept_duration_min": k["duration_min"],
                        "dropped_points": r["points"],
                        "kept_points": k["points"],
                        "reason": "time_range_high_overlap",
                    }
                )
                break
        if keep_it:
            keep.append(r)
    keep_ids = {k["sample_id"] for k in keep}
    kept = [c for c in candidates if c.sample_id in keep_ids]
    return kept, dropped


def nearest_time_gap_min(adsc: pd.DataFrame, adsb_min: pd.DataFrame) -> float:
    if adsc.empty or adsb_min.empty:
        return float("nan")
    a = adsc["timestamp"].dt.floor("min").astype("int64").to_numpy()
    b = adsb_min["minute_ts"].astype("int64").to_numpy()
    # O(N*M) is acceptable here due small sample count.
    d = np.abs(a[:, None] - b[None, :]).min()
    return float(d / 1e9 / 60.0)


def detect_altitude_profile(adsb_min: pd.DataFrame) -> dict:
    if adsb_min.empty:
        return {
            "climb_phase_detected": False,
            "cruise_phase_detected": False,
            "descent_phase_detected": False,
            "cruise_altitude_level": np.nan,
            "cruise_duration_minutes": 0.0,
            "cruise_start": pd.NaT,
            "cruise_end": pd.NaT,
            "altitude_profile_complete_flag": False,
        }
    d = adsb_min.sort_values("minute_ts").copy()
    d["alt"] = d["alt"].astype(float)
    d["alt_smooth"] = d["alt"].rolling(window=5, center=True, min_periods=1).median()
    d["dalt"] = d["alt_smooth"].diff().fillna(0.0)
    # assume minutes
    climb = (d["dalt"] > 100.0)
    descent = (d["dalt"] < -100.0)
    climb_phase = climb.sum() >= 5
    descent_phase = descent.sum() >= 5
    # cruise candidate: low slope and high altitude (near top band)
    alt_hi = d["alt_smooth"].quantile(0.7)
    cruise_candidate = (d["alt_smooth"] >= alt_hi) & (d["dalt"].abs() <= 30.0)
    if cruise_candidate.any():
        groups = (cruise_candidate != cruise_candidate.shift()).cumsum()
        lengths = d.groupby(groups)["minute_ts"].agg(["count", "min", "max"])
        cruise_groups = lengths[cruise_candidate.groupby(groups).first()]
        if not cruise_groups.empty:
            g = cruise_groups.sort_values("count", ascending=False).iloc[0]
            cruise_start = g["min"]
            cruise_end = g["max"]
            cruise_duration = float((cruise_end - cruise_start).total_seconds() / 60.0)
            cruise_alt = float(d.loc[(d["minute_ts"] >= cruise_start) & (d["minute_ts"] <= cruise_end), "alt_smooth"].median())
            cruise_phase = cruise_duration >= 30.0
        else:
            cruise_start = pd.NaT
            cruise_end = pd.NaT
            cruise_duration = 0.0
            cruise_alt = float(d["alt_smooth"].median())
            cruise_phase = False
    else:
        cruise_start = pd.NaT
        cruise_end = pd.NaT
        cruise_duration = 0.0
        cruise_alt = float(d["alt_smooth"].median())
        cruise_phase = False

    profile_complete = bool(climb_phase and cruise_phase and descent_phase)
    return {
        "climb_phase_detected": bool(climb_phase),
        "cruise_phase_detected": bool(cruise_phase),
        "descent_phase_detected": bool(descent_phase),
        "cruise_altitude_level": cruise_alt,
        "cruise_duration_minutes": cruise_duration,
        "cruise_start": cruise_start,
        "cruise_end": cruise_end,
        "altitude_profile_complete_flag": profile_complete,
    }


def adsc_cruise_alignment(adsc: pd.DataFrame, adsb_min: pd.DataFrame, cruise_start, cruise_end, cruise_alt) -> dict:
    if adsc.empty or adsb_min.empty or pd.isna(cruise_start) or pd.isna(cruise_end):
        return {
            "adsc_in_cruise_window_ratio": 0.0,
            "mean_adsc_vs_cruise_alt_diff": np.nan,
            "max_adsc_vs_cruise_alt_diff": np.nan,
            "within_adsb_flight_time_flag": False,
            "within_adsb_cruise_time_flag": False,
        }
    adsc_ts = adsc["timestamp"]
    adsb_start = adsb_min["minute_ts"].min()
    adsb_end = adsb_min["minute_ts"].max()
    within_flight = (adsc_ts.min() >= adsb_start) and (adsc_ts.max() <= adsb_end)
    in_cruise = (adsc_ts >= cruise_start) & (adsc_ts <= cruise_end)
    ratio = float(in_cruise.mean()) if len(adsc_ts) else 0.0
    alt_diff = (adsc["baroaltitude"] - cruise_alt).abs()
    return {
        "adsc_in_cruise_window_ratio": ratio,
        "mean_adsc_vs_cruise_alt_diff": float(alt_diff.mean()) if len(alt_diff) else np.nan,
        "max_adsc_vs_cruise_alt_diff": float(alt_diff.max()) if len(alt_diff) else np.nan,
        "within_adsb_flight_time_flag": bool(within_flight),
        "within_adsb_cruise_time_flag": bool(ratio >= 0.6),
    }


def direction_consistency(adsc: pd.DataFrame, adsb_min: pd.DataFrame, cruise_start, cruise_end) -> tuple[bool, float]:
    if adsc.empty or adsb_min.empty:
        return False, np.nan
    adsc_sorted = adsc.sort_values("timestamp")
    adsb_sorted = adsb_min.sort_values("minute_ts")
    if pd.notna(cruise_start) and pd.notna(cruise_end):
        adsb_sorted = adsb_sorted[(adsb_sorted["minute_ts"] >= cruise_start) & (adsb_sorted["minute_ts"] <= cruise_end)]
    if len(adsc_sorted) < 2 or len(adsb_sorted) < 2:
        return False, np.nan
    adsc_bearing = bearing_deg(
        adsc_sorted.iloc[0]["lat"],
        adsc_sorted.iloc[0]["lon"],
        adsc_sorted.iloc[-1]["lat"],
        adsc_sorted.iloc[-1]["lon"],
    )
    adsb_bearing = bearing_deg(
        adsb_sorted.iloc[0]["lat"],
        adsb_sorted.iloc[0]["lon"],
        adsb_sorted.iloc[-1]["lat"],
        adsb_sorted.iloc[-1]["lon"],
    )
    diff = angle_diff_deg(adsc_bearing, adsb_bearing)
    return bool(diff <= 60.0), float(diff)


def plot_alignment(
    out_png: Path,
    adsc: pd.DataFrame,
    adsb_min: pd.DataFrame,
    sample_meta: dict,
    gap_break_min: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax0, ax1 = axes

    # Plot ADS-B as segmented lines to avoid connecting across missing gaps.
    adsb_min = adsb_min.sort_values("minute_ts").copy()
    adsb_min["dt_min"] = adsb_min["minute_ts"].diff().dt.total_seconds().div(60.0).fillna(0.0)
    adsb_min["seg_id"] = (adsb_min["dt_min"] >= gap_break_min).astype("int64").cumsum()
    first_seg = True
    for _, seg in adsb_min.groupby("seg_id"):
        if len(seg) < 2:
            continue
        ax0.plot(
            seg["lon"],
            seg["lat"],
            color="gray",
            lw=1.0,
            alpha=0.8,
            label="ADS-B minute agg" if first_seg else None,
        )
        first_seg = False
    ax0.scatter(adsc["lon"], adsc["lat"], color="#7b3294", s=22, label="ADS-C raw")
    ax0.plot(adsc["lon"], adsc["lat"], color="#7b3294", lw=1.2, alpha=0.8)
    ax0.set_xlabel("Lon")
    ax0.set_ylabel("Lat")
    ax0.set_title("Raw Alignment (Lat/Lon)")
    ax0.legend(loc="best", fontsize=8)

    first_seg = True
    for _, seg in adsb_min.groupby("seg_id"):
        if len(seg) < 2:
            continue
        ax1.plot(
            seg["minute_ts"],
            seg["alt"],
            color="gray",
            lw=1.0,
            alpha=0.8,
            label="ADS-B minute agg" if first_seg else None,
        )
        first_seg = False
    ax1.scatter(adsc["timestamp"], adsc["baroaltitude"], color="#7b3294", s=22, label="ADS-C raw")
    ax1.plot(adsc["timestamp"], adsc["baroaltitude"], color="#7b3294", lw=1.2, alpha=0.8)
    ax1.set_xlabel("Time (UTC)")
    ax1.set_ylabel("Altitude")
    ax1.set_title("Raw Alignment (Altitude vs Time)")
    ax1.legend(loc="best", fontsize=8)

    title = (
        f"{sample_meta['sample_id']} | flight={sample_meta['flight_id']} | date={sample_meta['flight_date']} | "
        f"ADS-C={sample_meta['adsc_start_time']}~{sample_meta['adsc_end_time']} | "
        f"ADS-B={sample_meta['adsb_start_time']}~{sample_meta['adsb_end_time']} | "
        f"rule={sample_meta['matching_rule']} | conf={sample_meta['alignment_confidence']} | suspicious={sample_meta['suspicious_flag']}"
    )
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_alignment_local(
    out_png: Path,
    adsc: pd.DataFrame,
    adsb_min: pd.DataFrame,
    sample_meta: dict,
    gap_break_min: float,
    local_window_min: float,
) -> None:
    if adsc.empty or adsb_min.empty:
        return
    adsc_start = adsc["timestamp"].min()
    adsc_end = adsc["timestamp"].max()
    pad = pd.Timedelta(minutes=local_window_min)
    t0 = adsc_start - pad
    t1 = adsc_end + pad
    adsb_local = adsb_min[(adsb_min["minute_ts"] >= t0) & (adsb_min["minute_ts"] <= t1)].copy()
    adsc_local = adsc[(adsc["timestamp"] >= t0) & (adsc["timestamp"] <= t1)].copy()
    if adsb_local.empty or adsc_local.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax0, ax1 = axes
    adsb_local = adsb_local.sort_values("minute_ts").copy()
    adsb_local["dt_min"] = adsb_local["minute_ts"].diff().dt.total_seconds().div(60.0).fillna(0.0)
    adsb_local["seg_id"] = (adsb_local["dt_min"] >= gap_break_min).astype("int64").cumsum()
    first_seg = True
    for _, seg in adsb_local.groupby("seg_id"):
        if len(seg) < 2:
            continue
        ax0.plot(
            seg["lon"],
            seg["lat"],
            color="gray",
            lw=1.0,
            alpha=0.8,
            label="ADS-B minute agg (local)" if first_seg else None,
        )
        first_seg = False
    ax0.scatter(adsc_local["lon"], adsc_local["lat"], color="#7b3294", s=22, label="ADS-C raw")
    ax0.plot(adsc_local["lon"], adsc_local["lat"], color="#7b3294", lw=1.2, alpha=0.8)
    ax0.set_xlabel("Lon")
    ax0.set_ylabel("Lat")
    ax0.set_title("Local Alignment (Lat/Lon)")
    ax0.legend(loc="best", fontsize=8)

    first_seg = True
    for _, seg in adsb_local.groupby("seg_id"):
        if len(seg) < 2:
            continue
        ax1.plot(
            seg["minute_ts"],
            seg["alt"],
            color="gray",
            lw=1.0,
            alpha=0.8,
            label="ADS-B minute agg (local)" if first_seg else None,
        )
        first_seg = False
    ax1.scatter(adsc_local["timestamp"], adsc_local["baroaltitude"], color="#7b3294", s=22, label="ADS-C raw")
    ax1.plot(adsc_local["timestamp"], adsc_local["baroaltitude"], color="#7b3294", lw=1.2, alpha=0.8)
    ax1.set_xlabel("Time (UTC)")
    ax1.set_ylabel("Altitude")
    ax1.set_title("Local Alignment (Altitude vs Time)")
    ax1.legend(loc="best", fontsize=8)

    title = (
        f"{sample_meta['sample_id']} | flight={sample_meta['flight_id']} | date={sample_meta['flight_date']} | "
        f"ADS-C={sample_meta['adsc_start_time']}~{sample_meta['adsc_end_time']} | "
        f"local_window=±{local_window_min}min | conf={sample_meta['alignment_confidence']}"
    )
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def build_alignment_series(adsc: pd.DataFrame, adsb_min: pd.DataFrame, gap_break_min: float) -> pd.DataFrame:
    rows = []
    if not adsb_min.empty:
        adsb_min = adsb_min.sort_values("minute_ts").copy()
        adsb_min["dt_min"] = adsb_min["minute_ts"].diff().dt.total_seconds().div(60.0).fillna(0.0)
        adsb_min["adsb_seg_id"] = (adsb_min["dt_min"] >= gap_break_min).astype("int64").cumsum()
        for r in adsb_min.itertuples(index=False):
            rows.append(
                {
                    "time": r.minute_ts,
                    "lat": r.lat,
                    "lon": r.lon,
                    "alt": r.alt,
                    "source": "adsb_minute",
                    "adsb_seg_id": int(r.adsb_seg_id),
                    "is_adsc": False,
                }
            )
    if not adsc.empty:
        adsc = adsc.sort_values("timestamp").copy()
        for r in adsc.itertuples(index=False):
            rows.append(
                {
                    "time": r.timestamp,
                    "lat": r.lat,
                    "lon": r.lon,
                    "alt": r.baroaltitude,
                    "source": "adsc_raw",
                    "adsb_seg_id": None,
                    "is_adsc": True,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["time", "lat", "lon", "alt", "source", "adsb_seg_id", "is_adsc"])
    df = pd.DataFrame(rows)
    df = df.sort_values(["time", "is_adsc"]).reset_index(drop=True)
    return df


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    out_dir = Path(args.output_dir)
    plots_dir = out_dir / "plots" / "adsc_adsb_raw_alignment_check"
    plots_local_dir = out_dir / "plots" / "adsc_adsb_raw_alignment_check_local"
    series_dir = out_dir / "aligned_series"
    series_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    plots_local_dir.mkdir(parents=True, exist_ok=True)

    flight_files: list[FlightFile] = []
    use_decoded = bool(args.adsc_decoded_csv)
    use_match_csv = bool(args.adsb_flight_matches_csv)
    adsc_groups = {}
    adsb_min_groups = {}
    adsb_index = {}
    adsb_cache: dict[str, pd.DataFrame] = {}
    fetch_logs: list[dict] = []

    if use_decoded:
        adsc_df = pd.read_csv(args.adsc_decoded_csv)
        adsc_df = adsc_df.rename(
            columns={
                "latitude": "lat",
                "longitude": "lon",
                "altitude_m": "baroaltitude",
            }
        )
        adsc_df["timestamp"] = pd.to_datetime(adsc_df["timestamp"], utc=True, errors="coerce")
        adsc_df = adsc_df.dropna(subset=["icao24", "day", "timestamp", "lat", "lon", "baroaltitude"])
        if use_match_csv:
            match_df = pd.read_csv(args.adsb_flight_matches_csv)
            if args.matches_offset and args.matches_offset > 0:
                match_df = match_df.iloc[args.matches_offset :].reset_index(drop=True)
            if args.max_matches and args.max_matches > 0:
                match_df = match_df.head(args.max_matches)
            match_df = match_df.dropna(subset=["icao24", "day"])
            match_df["icao24"] = match_df["icao24"].astype(str).str.lower()
            match_df["day"] = match_df["day"].astype(str)
            keep_ids = set(match_df["day"] + "-" + match_df["icao24"])
            for (icao, day), g in adsc_df.groupby(["icao24", "day"]):
                sample_id = f"{day}-{icao}"
                if sample_id not in keep_ids:
                    continue
                ff = FlightFile(path=Path(sample_id), sample_id=sample_id, flight_id=str(icao), flight_date=str(day))
                flight_files.append(ff)
                adsc_groups[sample_id] = g[["timestamp", "lat", "lon", "baroaltitude"]].copy()
        else:
            for (icao, day), g in adsc_df.groupby(["icao24", "day"]):
                sample_id = f"{day}-{icao}"
                ff = FlightFile(path=Path(sample_id), sample_id=sample_id, flight_id=str(icao), flight_date=str(day))
                flight_files.append(ff)
                adsc_groups[sample_id] = g[["timestamp", "lat", "lon", "baroaltitude"]].copy()

            # optional downsample to reduce API load
            if args.num_samples and args.skip_tier_filter:
                sizes = [(sid, len(df)) for sid, df in adsc_groups.items()]
                sizes.sort(key=lambda x: x[1], reverse=True)
                keep = set([sid for sid, _ in sizes[: args.num_samples]])
                flight_files = [f for f in flight_files if f.sample_id in keep]
                adsc_groups = {k: v for k, v in adsc_groups.items() if k in keep}

        if use_match_csv:
            # Build ADS-B minute tracks by fetching OpenSky history for matched flights.
            sys.path.append(str(Path(__file__).resolve().parents[1]))
            from src.io_utils import load_settings

            settings = load_settings("config/settings.yaml")
            username = settings["opensky"].get("username")
            password = settings["opensky"].get("password")
            if username:
                os.environ["OPENSKY_USERNAME"] = username.lower()
                os.environ["OPENSKY_TRINO_USERNAME"] = username.lower()
            if password:
                os.environ["OPENSKY_PASSWORD"] = password
                os.environ["OPENSKY_TRINO_PASSWORD"] = password
            from pyopensky.trino import Trino

            trino = Trino()
            rate_limit = float(settings["opensky"].get("rate_limit_seconds") or 10.0)

            match_df = pd.read_csv(args.adsb_flight_matches_csv)
            if args.matches_offset and args.matches_offset > 0:
                match_df = match_df.iloc[args.matches_offset :].reset_index(drop=True)
            if args.max_matches and args.max_matches > 0:
                match_df = match_df.head(args.max_matches)
            match_df["firstSeen"] = pd.to_datetime(match_df["firstSeen"], utc=True, errors="coerce")
            match_df["lastSeen"] = pd.to_datetime(match_df["lastSeen"], utc=True, errors="coerce")
            match_df = match_df.dropna(subset=["icao24", "day", "firstSeen", "lastSeen"])

            pad = pd.Timedelta(minutes=float(args.adsb_fetch_pad_min))
            for _, r in match_df.iterrows():
                icao = str(r["icao24"]).lower()
                day = str(r["day"])
                sample_id = f"{day}-{icao}"
                start = r["firstSeen"] - pad
                end = r["lastSeen"] + pad
                try:
                    points = trino.history(
                        start=start.to_pydatetime(),
                        stop=end.to_pydatetime(),
                        icao24=icao,
                        selected_columns=("time", "icao24", "lat", "lon", "baroaltitude", "velocity", "heading"),
                        cached=True,
                    )
                except Exception:
                    points = None
                fetched_rows = 0 if points is None else len(points)
                fetch_logs.append(
                    {
                        "sample_id": sample_id,
                        "icao24": icao,
                        "day": day,
                        "start": start,
                        "end": end,
                        "fetched_rows": fetched_rows,
                    }
                )
                if points is not None and len(points) > 0:
                    dfp = pd.DataFrame(points)
                    dfp = dfp.rename(columns={"time": "timestamp", "baroaltitude": "alt"})
                    dfp["timestamp"] = pd.to_datetime(dfp["timestamp"], utc=True, errors="coerce")
                    dfp = dfp.dropna(subset=["timestamp", "lat", "lon", "alt", "icao24"])
                    if not dfp.empty:
                        dfp["minute_ts"] = dfp["timestamp"].dt.floor("min")
                        g = dfp.groupby("minute_ts", as_index=False)[["lat", "lon", "alt"]].median()
                        g["adsb_icao"] = dfp["icao24"].iloc[0]
                        adsb_min_groups[sample_id] = g
                        adsb_cache[sample_id] = g
                        adsb_index.setdefault((icao, day), []).append(sample_id)
                if rate_limit > 0:
                    time.sleep(rate_limit)
        else:
            adsb_min = pd.read_parquet(args.adsb_minute_parquet)
            adsb_min["minute_ts"] = pd.to_datetime(adsb_min["minute_ts"], utc=True, errors="coerce")
            for fid, g in adsb_min.groupby("flight_id"):
                if g.empty:
                    continue
                g = g[["minute_ts", "lat", "lon", "alt", "adsb_icao"]].copy()
                g = g.dropna(subset=["minute_ts", "lat", "lon", "alt", "adsb_icao"])
                if g.empty:
                    continue
                adsb_min_groups[fid] = g
                icao = str(g["adsb_icao"].iloc[0])
                date = str(g["minute_ts"].iloc[0].date())
                adsb_index.setdefault((icao, date), []).append(fid)
            adsb_cache = adsb_min_groups
            # optional OpenSky fetch support
            if args.fetch_missing_adsb:
                sys.path.append(str(Path(__file__).resolve().parents[1]))
                from src.io_utils import load_settings

                settings = load_settings("config/settings.yaml")
                username = settings["opensky"].get("username")
                password = settings["opensky"].get("password")
                if username:
                    os.environ["OPENSKY_USERNAME"] = username.lower()
                    os.environ["OPENSKY_TRINO_USERNAME"] = username.lower()
                if password:
                    os.environ["OPENSKY_PASSWORD"] = password
                    os.environ["OPENSKY_TRINO_PASSWORD"] = password
                from pyopensky.trino import Trino

                trino = Trino()
                rate_limit = float(settings["opensky"].get("rate_limit_seconds") or 10.0)
    else:
        for p in sorted(source_dir.glob("*.csv")):
            meta = parse_flight_meta(p)
            if meta is not None:
                flight_files.append(meta)

    adsc_cache: dict[str, pd.DataFrame] = {}
    records: list[dict] = []
    skip_records: list[dict] = []
    dedup_records: list[dict] = []
    dup_flight_counts: list[dict] = []

    for meta in flight_files:
        if use_decoded:
            adsc = adsc_groups.get(meta.sample_id)
            if adsc is None:
                continue
            adsc = adsc.copy()
            adsc["source"] = "adsc"
            adsb = pd.DataFrame(columns=["timestamp", "lat", "lon", "baroaltitude", "source"])
        else:
            df = pd.read_csv(meta.path)
            adsc, adsb = load_split(df)
        if len(adsc) < args.min_adsc_points:
            skip_records.append(
                {
                    "sample_id": meta.sample_id,
                    "flight_id": meta.flight_id,
                    "flight_date": meta.flight_date,
                    "skip_reason": "insufficient_adsc_points",
                }
            )
            if not use_decoded:
                continue
        if (not use_decoded) and len(adsb) == 0:
            skip_records.append(
                {
                    "sample_id": meta.sample_id,
                    "flight_id": meta.flight_id,
                    "flight_date": meta.flight_date,
                    "skip_reason": "insufficient_adsc_or_adsb_points",
                }
            )
            if not use_decoded:
                continue
        adsb_min = minute_agg_adsb(adsb) if not use_decoded else pd.DataFrame()
        if (not use_decoded) and adsb_min.empty:
            skip_records.append(
                {
                    "sample_id": meta.sample_id,
                    "flight_id": meta.flight_id,
                    "flight_date": meta.flight_date,
                    "skip_reason": "empty_adsb_minute_after_agg",
                }
            )
            continue
        adsc_cache[meta.sample_id] = adsc
        if not use_decoded:
            adsb_cache[meta.sample_id] = adsb_min

        # same-day + same-flight candidates
        if use_decoded:
            if use_match_csv:
                # Prefer exact sample_id fetched from matches.
                cand_ids = [meta.sample_id] if meta.sample_id in adsb_min_groups else []
            else:
                cand_ids = adsb_index.get((meta.flight_id, meta.flight_date), [])
            if (not cand_ids) and args.fetch_missing_adsb and (not use_match_csv):
                # fetch from OpenSky for this ADS-C window
                adsc_start = adsc["timestamp"].min()
                adsc_end = adsc["timestamp"].max()
                pad = pd.Timedelta(minutes=float(args.opensky_pad_min))
                start = adsc_start - pad
                end = adsc_end + pad
                try:
                    points = trino.history(
                        start=start.to_pydatetime(),
                        stop=end.to_pydatetime(),
                        icao24=str(meta.flight_id).lower(),
                        selected_columns=("time", "icao24", "lat", "lon", "baroaltitude", "velocity", "heading"),
                        cached=True,
                    )
                except Exception:
                    points = None
                fetched_rows = 0 if points is None else len(points)
                fetch_logs.append(
                    {
                        "sample_id": meta.sample_id,
                        "icao24": meta.flight_id,
                        "date": meta.flight_date,
                        "start": start,
                        "end": end,
                        "fetched_rows": fetched_rows,
                    }
                )
                if points is not None and len(points) > 0:
                    dfp = pd.DataFrame(points)
                    dfp = dfp.rename(columns={"time": "timestamp", "baroaltitude": "alt"})
                    dfp["timestamp"] = pd.to_datetime(dfp["timestamp"], utc=True, errors="coerce")
                    dfp = dfp.dropna(subset=["timestamp", "lat", "lon", "alt", "icao24"])
                    if not dfp.empty:
                        dfp["minute_ts"] = dfp["timestamp"].dt.floor("min")
                        g = dfp.groupby("minute_ts", as_index=False)[["lat", "lon", "alt"]].median()
                        g["adsb_icao"] = dfp["icao24"].iloc[0]
                        fid = f"opensky_{meta.flight_date}_{meta.flight_id}"
                        adsb_min_groups[fid] = g
                        adsb_cache[fid] = g
                        adsb_index.setdefault((meta.flight_id, meta.flight_date), []).append(fid)
                if rate_limit > 0:
                    time.sleep(rate_limit)
                cand_ids = adsb_index.get((meta.flight_id, meta.flight_date), [])
            candidates = [FlightFile(path=Path(fid), sample_id=fid, flight_id=meta.flight_id, flight_date=meta.flight_date) for fid in cand_ids]
        else:
            candidates = [x for x in flight_files if x.flight_date == meta.flight_date and x.flight_id == meta.flight_id]
        dup_flight_counts.append(
            {
                "flight_id": meta.flight_id,
                "date": meta.flight_date,
                "candidate_count_raw": len(candidates),
            }
        )
        candidates, dropped = dedupe_candidates(candidates, adsb_cache)
        if dropped:
            for d in dropped:
                dedup_records.append(
                    {
                        "flight_id": meta.flight_id,
                        "date": meta.flight_date,
                        **d,
                    }
                )
        best = None
        best_score = -1e18
        adsc_start = adsc["timestamp"].min()
        adsc_end = adsc["timestamp"].max()

        if use_decoded and use_match_csv:
            if meta.sample_id not in adsb_min_groups:
                skip_records.append(
                    {
                        "sample_id": meta.sample_id,
                        "flight_id": meta.flight_id,
                        "flight_date": meta.flight_date,
                        "skip_reason": "matched_adsb_missing_after_fetch",
                    }
                )
                continue
            c_adsb_min = adsb_min_groups.get(meta.sample_id, pd.DataFrame())
            if c_adsb_min.empty:
                skip_records.append(
                    {
                        "sample_id": meta.sample_id,
                        "flight_id": meta.flight_id,
                        "flight_date": meta.flight_date,
                        "skip_reason": "matched_adsb_empty_after_fetch",
                    }
                )
                continue
            overlap = minute_overlap_count(adsc, c_adsb_min)
            nearest_gap = nearest_time_gap_min(adsc, c_adsb_min)
            nn = nearest_neighbor_checks(adsc, c_adsb_min)
            num = summarize_numeric(nn)
            best = {
                "candidate": FlightFile(path=Path(meta.sample_id), sample_id=meta.sample_id, flight_id=meta.flight_id, flight_date=meta.flight_date),
                "adsb_min": c_adsb_min,
                "overlap": overlap,
                "nearest_gap": nearest_gap,
                "numeric": num,
            }
        else:
            for c in candidates:
                if use_decoded:
                    c_adsb_min = adsb_min_groups.get(c.sample_id, pd.DataFrame())
                else:
                    if c.sample_id not in adsb_cache:
                        c_df = pd.read_csv(c.path)
                        _, c_adsb = load_split(c_df)
                        adsb_cache[c.sample_id] = minute_agg_adsb(c_adsb)
                    c_adsb_min = adsb_cache[c.sample_id]
                if c_adsb_min.empty:
                    continue
                overlap = minute_overlap_count(adsc, c_adsb_min)
                nearest_gap = nearest_time_gap_min(adsc, c_adsb_min)
                nn = nearest_neighbor_checks(adsc, c_adsb_min)
                num = summarize_numeric(nn)
                p90_dist = num["p90_spatial_distance_m"] if num["matched_points"] > 0 else 1e9
                score = overlap * 1000.0 - nearest_gap * 20.0 - (p90_dist / 1000.0)
                if score > best_score:
                    best_score = score
                    best = {
                        "candidate": c,
                        "adsb_min": c_adsb_min,
                        "overlap": overlap,
                        "nearest_gap": nearest_gap,
                        "numeric": num,
                    }

        if best is None:
            skip_records.append(
                {
                    "sample_id": meta.sample_id,
                    "flight_id": meta.flight_id,
                    "flight_date": meta.flight_date,
                    "skip_reason": "no_reliable_candidate",
                }
            )
            continue

        conf, suspicious = confidence_from_metrics(
            best["overlap"], best["nearest_gap"], best["numeric"]["p90_spatial_distance_m"]
        )
        if len(candidates) == 1 and best["candidate"].sample_id == meta.sample_id:
            conf = "high"
            suspicious = False
        profile = detect_altitude_profile(best["adsb_min"])
        cruise_align = adsc_cruise_alignment(adsc, best["adsb_min"], profile["cruise_start"], profile["cruise_end"], profile["cruise_altitude_level"])
        dir_ok, dir_diff = direction_consistency(adsc, best["adsb_min"], profile["cruise_start"], profile["cruise_end"])
        planar_score = float(np.exp(-(best["numeric"]["p90_spatial_distance_m"] or 1e9) / 200000.0))

        # Tiered confidence logic
        flight_level = (
            profile["altitude_profile_complete_flag"]
            and cruise_align["within_adsb_flight_time_flag"]
            and (cruise_align["mean_adsc_vs_cruise_alt_diff"] <= 1200.0 if not np.isnan(cruise_align["mean_adsc_vs_cruise_alt_diff"]) else False)
        )
        medium_level = (
            flight_level
            and (best["numeric"]["mean_time_diff_minutes"] <= 240.0 if not np.isnan(best["numeric"]["mean_time_diff_minutes"]) else False)
            and (best["numeric"]["p90_spatial_distance_m"] <= 1500000.0 if not np.isnan(best["numeric"]["p90_spatial_distance_m"]) else False)
            and (cruise_align["adsc_in_cruise_window_ratio"] >= 0.3)
            and dir_ok
        )
        strong_level = (
            profile["altitude_profile_complete_flag"]
            and cruise_align["within_adsb_cruise_time_flag"]
            and (cruise_align["adsc_in_cruise_window_ratio"] >= 0.6)
            and (cruise_align["mean_adsc_vs_cruise_alt_diff"] <= 600.0 if not np.isnan(cruise_align["mean_adsc_vs_cruise_alt_diff"]) else False)
            and (best["numeric"]["mean_time_diff_minutes"] <= 120.0 if not np.isnan(best["numeric"]["mean_time_diff_minutes"]) else False)
            and (best["numeric"]["p90_spatial_distance_m"] <= 300000.0 if not np.isnan(best["numeric"]["p90_spatial_distance_m"]) else False)
            and dir_ok
        )
        weak_reason = []
        if not profile["altitude_profile_complete_flag"]:
            weak_reason.append("no_complete_altitude_profile")
        if not cruise_align["within_adsb_flight_time_flag"]:
            weak_reason.append("adsc_outside_adsb_flight_time")
        if np.isnan(cruise_align["mean_adsc_vs_cruise_alt_diff"]) or cruise_align["mean_adsc_vs_cruise_alt_diff"] > 1200.0:
            weak_reason.append("adsc_alt_not_close_to_cruise")
        medium_reason = []
        if not flight_level:
            medium_reason.append("flight_level_failed")
        if np.isnan(best["numeric"]["mean_time_diff_minutes"]) or best["numeric"]["mean_time_diff_minutes"] > 240.0:
            medium_reason.append("time_diff_too_large")
        if np.isnan(best["numeric"]["p90_spatial_distance_m"]) or best["numeric"]["p90_spatial_distance_m"] > 1500000.0:
            medium_reason.append("spatial_distance_too_large")
        if cruise_align["adsc_in_cruise_window_ratio"] < 0.3:
            medium_reason.append("adsc_not_in_cruise_window")
        if not dir_ok:
            medium_reason.append("direction_inconsistent")
        if strong_level:
            confidence_level = "strong"
            reject_reason = ""
        elif medium_level:
            confidence_level = "medium"
            reject_reason = ""
        elif flight_level:
            confidence_level = "weak"
            reject_reason = ""
        else:
            confidence_level = "reject"
            reject_reason = "failed_flight_level_rules"
        rec = {
            "sample_id": meta.sample_id,
            "flight_id": meta.flight_id,
            "flight_date": meta.flight_date,
            "adsc_start_time": adsc_start,
            "adsc_end_time": adsc_end,
            "adsb_start_time": best["adsb_min"]["minute_ts"].min(),
            "adsb_end_time": best["adsb_min"]["minute_ts"].max(),
            "adsc_points": int(len(adsc)),
            "adsb_points": int(len(best["adsb_min"])),
            "adsb_match_count": int(len(candidates)),
            "chosen_adsb_match_id": best["candidate"].sample_id,
            "matching_rule": "same_day+same_flight_then_best_time_overlap_and_path_consistency",
            "time_overlap_minutes": float(best["overlap"]),
            "nearest_time_gap_minutes": float(best["nearest_gap"]),
            "alignment_confidence": conf,
            "suspicious_flag": bool(suspicious),
            "notes": "",
            "numeric": best["numeric"],
            "altitude_profile": profile,
            "cruise_alignment": cruise_align,
            "direction_consistency_flag": dir_ok,
            "direction_diff_deg": dir_diff,
            "planar_alignment_score": planar_score,
            "flight_level_consistency_flag": bool(flight_level),
            "segment_level_consistency_flag": bool(medium_level),
            "local_strong_match_flag": bool(strong_level),
            "weak_confidence_reason": "" if flight_level else ";".join(weak_reason) if weak_reason else "unknown_weak_fail",
            "medium_confidence_reason": "" if medium_level else ";".join(medium_reason) if medium_reason else "unknown_medium_fail",
            "confidence_level": confidence_level,
            "reject_reason": reject_reason,
        }
        records.append(rec)

    # Tiered selection or full plotting
    records_df = pd.DataFrame(records)
    if args.skip_tier_filter:
        chosen = pick_samples(records_df.to_dict(orient="records"), args.max_accepted_plots)
        chosen_medium = []
        chosen_weak = []
        rejected = []
    else:
        strong_df = records_df[records_df["local_strong_match_flag"] == True]
        medium_df = records_df[records_df["segment_level_consistency_flag"] == True]
        weak_df = records_df[records_df["flight_level_consistency_flag"] == True]
        reject_df = records_df[records_df["confidence_level"] == "reject"]
        chosen = pick_samples(strong_df.to_dict(orient="records"), args.max_accepted_plots)
        chosen_medium = pick_samples(medium_df.to_dict(orient="records"), args.max_accepted_plots)
        chosen_weak = pick_samples(weak_df.to_dict(orient="records"), args.max_accepted_plots)
        rejected = pick_samples(reject_df.to_dict(orient="records"), args.max_rejected_plots)

    audit_rows = []
    numeric_rows = []
    visual_rows = []

    for idx, rec in enumerate(chosen, start=1):
        sid = rec["sample_id"]
        adsc = adsc_cache[sid]
        adsb_min = adsb_cache[rec["chosen_adsb_match_id"]]
        num = rec["numeric"]
        vj = visual_judgement_row(
            num,
            rec["alignment_confidence"],
            bool(rec["suspicious_flag"]),
            chosen_same_id=(rec["sample_id"] == rec["chosen_adsb_match_id"]),
            adsb_match_count=int(rec["adsb_match_count"]),
        )
        # local segment availability check
        adsc_start = adsc["timestamp"].min()
        adsc_end = adsc["timestamp"].max()
        local_pad = pd.Timedelta(minutes=args.local_window_min)
        adsb_local = adsb_min[
            (adsb_min["minute_ts"] >= adsc_start - local_pad)
            & (adsb_min["minute_ts"] <= adsc_end + local_pad)
        ].copy()
        if not adsb_local.empty:
            adsb_local = adsb_local.sort_values("minute_ts")
            adsb_local["dt_min"] = adsb_local["minute_ts"].diff().dt.total_seconds().div(60.0).fillna(0.0)
            gap_flags = (adsb_local["dt_min"] >= args.adsb_gap_break_min).astype("int64")
            max_run = adsb_local.groupby(gap_flags.cumsum()).size().max()
        else:
            max_run = 0
        local_segment_ok = bool(max_run and max_run >= 2)

        audit_rows.append(
            {
                "sample_id": rec["sample_id"],
                "flight_id": rec["flight_id"],
                "flight_date": rec["flight_date"],
                "adsc_start_time": rec["adsc_start_time"],
                "adsc_end_time": rec["adsc_end_time"],
                "adsb_start_time": rec["adsb_start_time"],
                "adsb_end_time": rec["adsb_end_time"],
                "adsb_match_count": rec["adsb_match_count"],
                "chosen_adsb_match_id": rec["chosen_adsb_match_id"],
                "matching_rule": rec["matching_rule"],
                "time_overlap_minutes": rec["time_overlap_minutes"],
                "nearest_time_gap_minutes": rec["nearest_time_gap_minutes"],
                "alignment_confidence": rec["alignment_confidence"],
                "suspicious_flag": rec["suspicious_flag"],
                "climb_phase_detected": rec["altitude_profile"]["climb_phase_detected"],
                "cruise_phase_detected": rec["altitude_profile"]["cruise_phase_detected"],
                "descent_phase_detected": rec["altitude_profile"]["descent_phase_detected"],
                "cruise_altitude_level": rec["altitude_profile"]["cruise_altitude_level"],
                "cruise_duration_minutes": rec["altitude_profile"]["cruise_duration_minutes"],
                "adsc_in_cruise_window_ratio": rec["cruise_alignment"]["adsc_in_cruise_window_ratio"],
                "mean_adsc_vs_cruise_alt_diff": rec["cruise_alignment"]["mean_adsc_vs_cruise_alt_diff"],
                "max_adsc_vs_cruise_alt_diff": rec["cruise_alignment"]["max_adsc_vs_cruise_alt_diff"],
                "within_adsb_flight_time_flag": rec["cruise_alignment"]["within_adsb_flight_time_flag"],
                "within_adsb_cruise_time_flag": rec["cruise_alignment"]["within_adsb_cruise_time_flag"],
                "direction_consistency_flag": rec["direction_consistency_flag"],
                "direction_diff_deg": rec["direction_diff_deg"],
                "planar_alignment_score": rec["planar_alignment_score"],
                "flight_level_consistency_flag": rec["flight_level_consistency_flag"],
                "segment_level_consistency_flag": rec["segment_level_consistency_flag"],
                "local_strong_match_flag": rec["local_strong_match_flag"],
                "weak_confidence_reason": rec["weak_confidence_reason"],
                "medium_confidence_reason": rec["medium_confidence_reason"],
                "confidence_level": rec["confidence_level"],
                "reject_reason": rec["reject_reason"],
                "notes": rec["notes"],
            }
        )
        numeric_rows.append(
            {
                "sample_id": rec["sample_id"],
                "flight_id": rec["flight_id"],
                "matched_points": num["matched_points"],
                "mean_time_diff_minutes": num["mean_time_diff_minutes"],
                "p90_time_diff_minutes": num["p90_time_diff_minutes"],
                "mean_spatial_distance_m": num["mean_spatial_distance_m"],
                "p90_spatial_distance_m": num["p90_spatial_distance_m"],
                "mean_alt_diff": num["mean_alt_diff"],
                "p90_alt_diff": num["p90_alt_diff"],
                "suspicious_flag": rec["suspicious_flag"],
                "local_segment_available": local_segment_ok,
                "notes": "",
            }
        )
        visual_rows.append(
            {
                "sample_id": rec["sample_id"],
                "flight_id": rec["flight_id"],
                **vj,
                "local_segment_available": "yes" if local_segment_ok else "no",
                "comments": "",
            }
        )

        png_name = f"{idx:02d}_{rec['sample_id']}_{rec['flight_id']}_{rec['flight_date']}_alignment_check.png"
        series_path = series_dir / f"{rec['sample_id']}_aligned.csv"
        series_df = build_alignment_series(adsc=adsc, adsb_min=adsb_min, gap_break_min=args.adsb_gap_break_min)
        series_df.to_csv(series_path, index=False)
        plot_alignment(
            plots_dir / png_name,
            adsc=adsc,
            adsb_min=adsb_min,
            sample_meta={
                "sample_id": rec["sample_id"],
                "flight_id": rec["flight_id"],
                "flight_date": rec["flight_date"],
                "adsc_start_time": rec["adsc_start_time"],
                "adsc_end_time": rec["adsc_end_time"],
                "adsb_start_time": rec["adsb_start_time"],
                "adsb_end_time": rec["adsb_end_time"],
                "matching_rule": rec["matching_rule"],
                "alignment_confidence": rec["alignment_confidence"],
                "suspicious_flag": rec["suspicious_flag"],
            },
            gap_break_min=args.adsb_gap_break_min,
        )
        plot_alignment_local(
            plots_local_dir / png_name,
            adsc=adsc,
            adsb_min=adsb_min,
            sample_meta={
                "sample_id": rec["sample_id"],
                "flight_id": rec["flight_id"],
                "flight_date": rec["flight_date"],
                "adsc_start_time": rec["adsc_start_time"],
                "adsc_end_time": rec["adsc_end_time"],
                "alignment_confidence": rec["alignment_confidence"],
            },
            gap_break_min=args.adsb_gap_break_min,
            local_window_min=args.local_window_min,
        )

    # Plot medium/weak tiers
    for tier_name, tier_rows in [
        ("medium", chosen_medium),
        ("weak", chosen_weak),
    ]:
        for idx, rec in enumerate(tier_rows, start=1):
            sid = rec["sample_id"]
            adsc = adsc_cache[sid]
            adsb_min = adsb_cache[rec["chosen_adsb_match_id"]]
            png_name = f"{tier_name}_{idx:02d}_{rec['sample_id']}_{rec['flight_id']}_{rec['flight_date']}_alignment_check.png"
            series_path = series_dir / f"{rec['sample_id']}_aligned.csv"
            series_df = build_alignment_series(adsc=adsc, adsb_min=adsb_min, gap_break_min=args.adsb_gap_break_min)
            series_df.to_csv(series_path, index=False)
            plot_alignment(
                plots_dir / png_name,
                adsc=adsc,
                adsb_min=adsb_min,
                sample_meta={
                    "sample_id": rec["sample_id"],
                    "flight_id": rec["flight_id"],
                    "flight_date": rec["flight_date"],
                    "adsc_start_time": rec["adsc_start_time"],
                    "adsc_end_time": rec["adsc_end_time"],
                    "adsb_start_time": rec["adsb_start_time"],
                    "adsb_end_time": rec["adsb_end_time"],
                    "matching_rule": rec["matching_rule"],
                    "alignment_confidence": rec["alignment_confidence"],
                    "suspicious_flag": rec["suspicious_flag"],
                },
                gap_break_min=args.adsb_gap_break_min,
            )
            plot_alignment_local(
                plots_local_dir / png_name,
                adsc=adsc,
                adsb_min=adsb_min,
                sample_meta={
                    "sample_id": rec["sample_id"],
                    "flight_id": rec["flight_id"],
                    "flight_date": rec["flight_date"],
                    "adsc_start_time": rec["adsc_start_time"],
                    "adsc_end_time": rec["adsc_end_time"],
                    "alignment_confidence": rec["alignment_confidence"],
                },
                gap_break_min=args.adsb_gap_break_min,
                local_window_min=args.local_window_min,
            )

    # Plot a few rejected examples for contrast
    for idx, rec in enumerate(rejected, start=1):
        sid = rec["sample_id"]
        adsc = adsc_cache[sid]
        adsb_min = adsb_cache[rec["chosen_adsb_match_id"]]
        png_name = f"reject_{idx:02d}_{rec['sample_id']}_{rec['flight_id']}_{rec['flight_date']}_alignment_check.png"
        series_path = series_dir / f"{rec['sample_id']}_aligned.csv"
        series_df = build_alignment_series(adsc=adsc, adsb_min=adsb_min, gap_break_min=args.adsb_gap_break_min)
        series_df.to_csv(series_path, index=False)
        plot_alignment(
            plots_dir / png_name,
            adsc=adsc,
            adsb_min=adsb_min,
            sample_meta={
                "sample_id": rec["sample_id"],
                "flight_id": rec["flight_id"],
                "flight_date": rec["flight_date"],
                "adsc_start_time": rec["adsc_start_time"],
                "adsc_end_time": rec["adsc_end_time"],
                "adsb_start_time": rec["adsb_start_time"],
                "adsb_end_time": rec["adsb_end_time"],
                "matching_rule": rec["matching_rule"],
                "alignment_confidence": rec["alignment_confidence"],
                "suspicious_flag": rec["suspicious_flag"],
            },
            gap_break_min=args.adsb_gap_break_min,
        )
        plot_alignment_local(
            plots_local_dir / png_name,
            adsc=adsc,
            adsb_min=adsb_min,
            sample_meta={
                "sample_id": rec["sample_id"],
                "flight_id": rec["flight_id"],
                "flight_date": rec["flight_date"],
                "adsc_start_time": rec["adsc_start_time"],
                "adsc_end_time": rec["adsc_end_time"],
                "alignment_confidence": rec["alignment_confidence"],
            },
            gap_break_min=args.adsb_gap_break_min,
            local_window_min=args.local_window_min,
        )

    pd.DataFrame(audit_rows).to_csv(out_dir / "adsc_adsb_alignment_audit.csv", index=False)
    pd.DataFrame(visual_rows).to_csv(out_dir / "adsc_adsb_alignment_visual_judgement.csv", index=False)
    pd.DataFrame(numeric_rows).to_csv(out_dir / "adsc_adsb_alignment_numeric_check.csv", index=False)
    # Full high-confidence summary (flattened)
    high_rows = []
    for rec in records:
        num = rec["numeric"]
        prof = rec["altitude_profile"]
        cruise = rec["cruise_alignment"]
        high_rows.append(
            {
                "sample_id": rec["sample_id"],
                "flight_id": rec["flight_id"],
                "date": rec["flight_date"],
                "adsb_match_id": rec["chosen_adsb_match_id"],
                "climb_phase_detected": prof["climb_phase_detected"],
                "cruise_phase_detected": prof["cruise_phase_detected"],
                "descent_phase_detected": prof["descent_phase_detected"],
                "cruise_altitude_level": prof["cruise_altitude_level"],
                "cruise_duration_minutes": prof["cruise_duration_minutes"],
                "adsc_in_cruise_window_ratio": cruise["adsc_in_cruise_window_ratio"],
                "mean_adsc_vs_cruise_alt_diff": cruise["mean_adsc_vs_cruise_alt_diff"],
                "max_adsc_vs_cruise_alt_diff": cruise["max_adsc_vs_cruise_alt_diff"],
                "within_adsb_flight_time_flag": cruise["within_adsb_flight_time_flag"],
                "within_adsb_cruise_time_flag": cruise["within_adsb_cruise_time_flag"],
                "mean_nearest_time_diff_min": num["mean_time_diff_minutes"],
                "p90_nearest_time_diff_min": num["p90_time_diff_minutes"],
                "mean_nearest_spatial_distance": num["mean_spatial_distance_m"],
                "p90_nearest_spatial_distance": num["p90_spatial_distance_m"],
                "direction_consistency_flag": rec["direction_consistency_flag"],
                "planar_alignment_score": rec["planar_alignment_score"],
                "flight_level_consistency_flag": rec["flight_level_consistency_flag"],
                "segment_level_consistency_flag": rec["segment_level_consistency_flag"],
                "local_strong_match_flag": rec["local_strong_match_flag"],
                "confidence_tier": rec["confidence_level"],
                "weak_confidence_reason": rec["weak_confidence_reason"],
                "medium_confidence_reason": rec["medium_confidence_reason"],
                "reject_reason": rec["reject_reason"],
            }
        )
    if high_rows:
        pd.DataFrame(high_rows).to_csv(out_dir / "adsc_adsb_high_confidence_alignment.csv", index=False)
    if skip_records:
        pd.DataFrame(skip_records).to_csv(out_dir / "adsc_adsb_alignment_skipped.csv", index=False)
    if fetch_logs:
        pd.DataFrame(fetch_logs).to_csv(out_dir / "adsc_adsb_opensky_fetch_log.csv", index=False)
    if dedup_records:
        pd.DataFrame(dedup_records).to_csv(out_dir / "adsc_adsb_alignment_dedup_log.csv", index=False)
    if dup_flight_counts:
        df_dup = pd.DataFrame(dup_flight_counts).drop_duplicates(subset=["flight_id", "date"])
        df_dup.to_csv(out_dir / "adsc_adsb_alignment_duplicate_counts.csv", index=False)

    print(f"[done] selected_strong={len(chosen)} medium={len(chosen_medium)} weak={len(chosen_weak)} rejected={len(rejected)} plots={plots_dir}")
    print(f"[done] tier_counts strong={sum(1 for r in records if r['local_strong_match_flag'])} medium={sum(1 for r in records if r['segment_level_consistency_flag'])} weak={sum(1 for r in records if r['flight_level_consistency_flag'])}")
    if dup_flight_counts:
        df_dup = pd.DataFrame(dup_flight_counts).drop_duplicates(subset=["flight_id", "date"])
        dup_multi = int((df_dup["candidate_count_raw"] > 1).sum())
        print(f"[done] duplicate_flights_raw={dup_multi} / {len(df_dup)}")
    print(f"[done] audit={out_dir / 'adsc_adsb_alignment_audit.csv'}")


if __name__ == "__main__":
    main()
