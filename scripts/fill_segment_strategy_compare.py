from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.real_adsc_replay_eval import (
    _build_fill_frame,
    _build_fill_intervals,
    _build_known_blocks,
    _extract_block_boundary_state,
    _predict_on_frame,
    _resample_minute,
)
from src.training.utils import load_config


def _safe_ratio(num: float, denom: float, eps: float = 1e-3) -> float:
    return float(num) / float(max(abs(denom), eps))


def _bucket_len(m: float) -> str:
    if m <= 15:
        return "<=15"
    if m <= 60:
        return "15-60"
    if m <= 180:
        return "60-180"
    return ">180"


def _bucket_len5(m: float) -> str:
    if m <= 15:
        return "<=15"
    if m <= 30:
        return "15-30"
    if m <= 60:
        return "30-60"
    if m <= 180:
        return "60-180"
    return ">180"


def _quality_class(row: pd.Series) -> tuple[str, str]:
    if row["shape_abnormal_flag"]:
        return "abnormal", row["abnormal_reason"]
    if row["overshoot_flag"] or row["edge_spike_flag"]:
        return "warn", "overshoot_or_edge_spike"
    return "keep", "ok"


def _edge_weights(n: int, mode: str = "hard") -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=float)
    w = np.ones(n, dtype=float)
    if mode == "hard":
        if n >= 1:
            w[0] = 0.0
            w[-1] = 0.0
        if n >= 2:
            w[1] = 0.0
            w[-2] = 0.0
    else:
        # smooth: 0,0.25,0.5,...,0.5,0.25,0
        if n >= 1:
            w[0] = 0.0
            w[-1] = 0.0
        if n >= 2:
            w[1] = 0.25
            w[-2] = 0.25
        if n >= 3:
            w[2] = 0.5
            w[-3] = 0.5
    return w


def _edge_weights_soft_custom(n: int, steps: list[float]) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=float)
    w = np.ones(n, dtype=float)
    k = min(len(steps), max(1, n // 2 + (n % 2)))
    for i in range(k):
        val = float(max(0.0, min(1.0, steps[i])))
        if i < n:
            w[i] = min(w[i], val)
            w[n - 1 - i] = min(w[n - 1 - i], val)
    return w


def _parse_float_list(text: str) -> list[float]:
    vals: list[float] = []
    for s in str(text).split(","):
        s = s.strip()
        if not s:
            continue
        vals.append(float(s))
    return vals


def _in_bucket(v: float, bucket: str) -> bool:
    if bucket == "<=15":
        return v <= 15
    if bucket == "15-30":
        return (v > 15) and (v <= 30)
    if bucket == "30-60":
        return (v > 30) and (v <= 60)
    return False


def _match_bucket(v: float, bucket: str) -> bool:
    if bucket in ("*", "any"):
        return True
    return _in_bucket(v, bucket) or _bucket_len(v) == bucket or _bucket_len5(v) == bucket


def _build_step_caps(
    n: int,
    fill_minutes: float,
    cap_mode: str,
    fixed_cap: float,
    cap_le15: float,
    cap_15_60: float,
    cap_gt60: float,
    edge_cap_factor: float,
) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=float)
    if cap_mode == "none":
        return np.full(n, np.inf, dtype=float)
    if cap_mode == "fixed":
        base_cap = float(fixed_cap)
    else:
        if fill_minutes <= 15:
            base_cap = float(cap_le15)
        elif fill_minutes <= 60:
            base_cap = float(cap_15_60)
        else:
            base_cap = float(cap_gt60)
    caps = np.full(n, base_cap, dtype=float)
    if cap_mode == "edge_strict":
        ef = float(max(0.0, min(1.0, edge_cap_factor)))
        if n >= 1:
            caps[0] = min(caps[0], base_cap * ef)
            caps[-1] = min(caps[-1], base_cap * ef)
        if n >= 2:
            caps[1] = min(caps[1], base_cap * ef)
            caps[-2] = min(caps[-2], base_cap * ef)
    return caps


def _resolve_policy(strategy: dict, fill_type: str, fill_minutes: float) -> dict:
    cfg = {
        "disable": False,
        "base_scale": float(strategy.get("base_scale", 1.0)),
        "taper_mode": strategy.get("taper_mode", "none"),
        "soft_steps": strategy.get("soft_steps", [0.0, 0.25, 0.5]),
        "cap_mode": strategy.get("cap_mode", "none"),
        "fixed_cap": strategy.get("fixed_cap", 300.0),
        "cap_le15": strategy.get("cap_le15", 120.0),
        "cap_15_60": strategy.get("cap_15_60", 220.0),
        "cap_gt60": strategy.get("cap_gt60", 350.0),
        "edge_cap_factor": strategy.get("edge_cap_factor", 0.6),
    }
    table = strategy.get("policy_table", []) or []
    lb5 = _bucket_len5(fill_minutes)
    lb4 = _bucket_len(fill_minutes)
    for row in table:
        rt = str(row.get("fill_type", "*"))
        rb = str(row.get("length_bucket", "*"))
        if (rt not in ("*", fill_type)) or (rb not in ("*", lb5, lb4)):
            continue
        for k in cfg.keys():
            if k in row:
                cfg[k] = row[k]
    return cfg


def _apply_residual_strategy(
    on_alt: np.ndarray,
    off_alt: np.ndarray,
    fill_type: str,
    fill_minutes: float,
    strategy: dict,
    args: argparse.Namespace,
) -> tuple[np.ndarray, bool]:
    n = len(on_alt)
    cfg = _resolve_policy(strategy, fill_type, fill_minutes)
    if bool(cfg.get("disable", False)):
        return off_alt.copy(), False

    taper_mode = str(cfg.get("taper_mode", "none"))
    if taper_mode == "none":
        w = np.ones(n, dtype=float)
    elif taper_mode == "hard":
        w = _edge_weights(n, mode="hard")
    elif taper_mode == "smooth":
        w = _edge_weights(n, mode="smooth")
    else:
        w = _edge_weights_soft_custom(n, list(cfg.get("soft_steps", [0.0, 0.25, 0.5])))

    base_scale = float(cfg.get("base_scale", 1.0))
    delta = (on_alt - off_alt) * base_scale * w
    cap_mode = str(cfg.get("cap_mode", "none"))
    caps = _build_step_caps(
        n=n,
        fill_minutes=fill_minutes,
        cap_mode=cap_mode,
        fixed_cap=float(cfg.get("fixed_cap", args.fixed_cap)),
        cap_le15=float(cfg.get("cap_le15", args.cap_le15)),
        cap_15_60=float(cfg.get("cap_15_60", args.cap_15_60)),
        cap_gt60=float(cfg.get("cap_gt60", args.cap_gt60)),
        edge_cap_factor=float(cfg.get("edge_cap_factor", args.edge_cap_factor)),
    )
    delta = np.clip(delta, -caps, caps)
    return off_alt + delta, True


def _segment_metrics(meta: dict, seg: pd.DataFrame, strategy_name: str, residual_enabled: bool) -> dict:
    seg = seg.sort_values("minute_ts").reset_index(drop=True)
    alt = seg["pred_alt"].to_numpy(dtype=float)
    left_alt = float(meta["left_boundary_alt"])
    right_alt = float(meta["right_boundary_alt"])
    gap = right_alt - left_alt
    alt_min = float(np.min(alt))
    alt_max = float(np.max(alt))
    peak_idx = int(np.argmax(alt))
    trough_idx = int(np.argmin(alt))
    overshoot_up = alt_max - max(left_alt, right_alt)
    undershoot_down = min(left_alt, right_alt) - alt_min
    overshoot_ratio = _safe_ratio(overshoot_up, gap)
    undershoot_ratio = _safe_ratio(undershoot_down, gap)
    diffs = np.diff(alt)
    first_jump = float(diffs[0]) if len(diffs) > 0 else 0.0
    second_jump = float(diffs[1]) if len(diffs) > 1 else 0.0
    last_jump = float(diffs[-1]) if len(diffs) > 0 else 0.0
    second_last_jump = float(diffs[-2]) if len(diffs) > 1 else 0.0
    first_two_peak = peak_idx <= 2 or trough_idx <= 2
    last_two_peak = peak_idx >= len(alt) - 3 or trough_idx >= len(alt) - 3
    max_vr = float(np.max(np.abs(diffs))) if len(diffs) else 0.0
    mean_vr = float(np.mean(np.abs(diffs))) if len(diffs) else 0.0
    smoothness = float(np.mean(np.abs(np.diff(alt, n=2)))) if len(alt) > 2 else 0.0
    sign_changes = int(np.sum(np.diff(np.sign(diffs)) != 0)) if len(diffs) > 1 else 0

    overshoot_flag = overshoot_up > 200.0
    undershoot_flag = undershoot_down > 200.0
    edge_spike_flag = (first_two_peak or last_two_peak) and (max_vr > 300.0)
    peak_then_return_flag = first_two_peak or last_two_peak
    shape_abnormal_flag = (overshoot_flag or undershoot_flag) and (edge_spike_flag or peak_then_return_flag)
    if first_two_peak:
        anomaly_position = "first_two_steps"
    elif last_two_peak:
        anomaly_position = "last_two_steps"
    elif overshoot_flag or undershoot_flag:
        anomaly_position = "middle"
    else:
        anomaly_position = "none"
    abnormal_reason = "overshoot_edge" if shape_abnormal_flag else ""

    row = {
        "strategy_name": strategy_name,
        "sample_id": meta["sample_id"],
        "fill_id": meta["fill_id"],
        "fill_type": f"{meta['left_block_type']}->{meta['right_block_type']}",
        "left_block_type": meta["left_block_type"],
        "right_block_type": meta["right_block_type"],
        "fill_minutes": meta["fill_minutes"],
        "length_bucket": _bucket_len(meta["fill_minutes"]),
        "point_count": int(len(seg)),
        "recovery_mode": "fill_intervals",
        "residual_enabled": residual_enabled,
        "left_boundary_alt": left_alt,
        "right_boundary_alt": right_alt,
        "boundary_alt_gap": gap,
        "boundary_alt_mean": (left_alt + right_alt) / 2.0,
        "segment_alt_min": alt_min,
        "segment_alt_max": alt_max,
        "segment_alt_range": alt_max - alt_min,
        "peak_index": peak_idx,
        "trough_index": trough_idx,
        "peak_time_offset_min": peak_idx,
        "trough_time_offset_min": trough_idx,
        "overshoot_up": overshoot_up,
        "undershoot_down": undershoot_down,
        "overshoot_ratio": overshoot_ratio,
        "undershoot_ratio": undershoot_ratio,
        "first_step_alt_jump": first_jump,
        "second_step_alt_jump": second_jump,
        "last_step_alt_jump": last_jump,
        "second_last_step_alt_jump": second_last_jump,
        "first_two_step_peak_flag": first_two_peak,
        "last_two_step_peak_flag": last_two_peak,
        "max_vertical_rate_inside": max_vr,
        "mean_vertical_rate_inside": mean_vr,
        "altitude_smoothness_score": smoothness,
        "sign_change_count_in_alt_diff": sign_changes,
        "overshoot_flag": overshoot_flag,
        "undershoot_flag": undershoot_flag,
        "peak_then_return_flag": peak_then_return_flag,
        "edge_spike_flag": edge_spike_flag,
        "shape_abnormal_flag": shape_abnormal_flag,
        "abnormal_reason": abnormal_reason,
        "anomaly_position": anomaly_position,
    }
    q, reason = _quality_class(pd.Series(row))
    row["quality_class"] = q
    row["quality_reason"] = reason
    return row


def _enforce_segment_endpoint_anchors(seg: pd.DataFrame, meta: dict) -> pd.DataFrame:
    if seg is None or seg.empty:
        return seg
    s = seg.sort_values("minute_ts").reset_index(drop=True).copy()
    # Hard endpoint constraint: recovered segment must pass both boundary anchors.
    s.at[0, "pred_lat"] = float(meta["left_boundary_lat"])
    s.at[0, "pred_lon"] = float(meta["left_boundary_lon"])
    s.at[0, "pred_alt"] = float(meta["left_boundary_alt"])
    s.at[len(s) - 1, "pred_lat"] = float(meta["right_boundary_lat"])
    s.at[len(s) - 1, "pred_lon"] = float(meta["right_boundary_lon"])
    s.at[len(s) - 1, "pred_alt"] = float(meta["right_boundary_alt"])
    return s


def _build_fill_dataset(audit: pd.DataFrame, adsc_dir: Path, context_minutes: int) -> tuple[list[dict], pd.DataFrame]:
    csv_files = sorted(adsc_dir.glob("*.csv"))
    flight_adsb_cache: dict[str, pd.DataFrame] = {}
    flight_adsc_cache: dict[str, pd.DataFrame] = {}
    for fp in csv_files:
        fdf = pd.read_csv(fp)
        if "source" not in fdf.columns:
            continue
        fdf["source"] = fdf["source"].astype(str).str.lower()
        if "time" in fdf.columns and "timestamp" not in fdf.columns:
            fdf = fdf.rename(columns={"time": "timestamp"})
        if "alt" in fdf.columns and "baroaltitude" not in fdf.columns:
            fdf = fdf.rename(columns={"alt": "baroaltitude"})
        src_adsb = fdf["source"].str.contains("adsb", na=False)
        src_adsc = fdf["source"].str.contains("adsc", na=False)
        adsb_min = _resample_minute(fdf[src_adsb].copy())
        adsc_raw = fdf[src_adsc].copy()
        if "timestamp" in adsc_raw.columns:
            adsc_raw["timestamp"] = pd.to_datetime(adsc_raw["timestamp"], utc=True, errors="coerce")
            adsc_raw = adsc_raw.dropna(subset=["timestamp", "lat", "lon", "baroaltitude"])
        if len(adsb_min):
            flight_adsb_cache[fp.stem] = adsb_min
        if len(adsc_raw):
            flight_adsc_cache[fp.stem] = adsc_raw

    fill_meta: list[dict] = []
    frames: list[pd.DataFrame] = []
    for _, row in audit.iterrows():
        sample_id = str(row["sample_id"])
        base = sample_id.split("_a")[0]
        adsb_all = flight_adsb_cache.get(base, pd.DataFrame(columns=["minute_ts", "lat", "lon", "alt"])).copy()
        adsc_raw = flight_adsc_cache.get(base, pd.DataFrame(columns=["timestamp", "lat", "lon", "baroaltitude"])).copy()
        if len(adsb_all):
            adsb_all["minute_ts"] = pd.to_datetime(adsb_all["minute_ts"], utc=True)
        if len(adsc_raw):
            adsc_raw["timestamp"] = pd.to_datetime(adsc_raw["timestamp"], utc=True)

        if len(adsb_all):
            w_start = adsb_all["minute_ts"].min()
            w_end = adsb_all["minute_ts"].max()
        else:
            w_start = pd.to_datetime(row["adsc_anchor_start_time"], utc=True)
            w_end = pd.to_datetime(row["adsc_anchor_end_time"], utc=True)
        if len(adsc_raw):
            w_start = min(w_start, adsc_raw["timestamp"].min())
            w_end = max(w_end, adsc_raw["timestamp"].max())

        blocks = _build_known_blocks(adsb_all, adsc_raw, w_start, w_end, gap_break_min=10.0)
        fills = _build_fill_intervals(blocks, min_gap_min=2.0)
        for fi, f in enumerate(fills, start=1):
            left_block = blocks[f["left_block_idx"]]
            right_block = blocks[f["right_block_idx"]]
            left_state = _extract_block_boundary_state(left_block, "end")
            right_state = _extract_block_boundary_state(right_block, "start")
            fill_id = f"f{fi:02d}"
            fill_sid = f"{sample_id}__{fill_id}"
            frame, _, _ = _build_fill_frame(
                minute_all_adsb=adsb_all,
                fill_start=f["fill_start_time"],
                fill_end=f["fill_end_time"],
                left_state=left_state,
                right_state=right_state,
                task_type="adsc_plus_local_adsb",
                context_minutes=context_minutes,
            )
            if frame.empty:
                continue
            frame["sample_id"] = fill_sid
            frame["flight_id"] = row.get("flight_id", "")
            frames.append(frame)
            fill_meta.append(
                {
                    "sample_id": sample_id,
                    "fill_id": fill_id,
                    "fill_sid": fill_sid,
                    "left_block_type": f["left_block_type"],
                    "right_block_type": f["right_block_type"],
                    "fill_minutes": f["fill_minutes"],
                    "fill_start_time": f["fill_start_time"],
                    "fill_end_time": f["fill_end_time"],
                    "left_boundary_lat": float(left_state["lat"]),
                    "left_boundary_lon": float(left_state["lon"]),
                    "left_boundary_alt": float(left_state["alt"]),
                    "right_boundary_lat": float(right_state["lat"]),
                    "right_boundary_lon": float(right_state["lon"]),
                    "right_boundary_alt": float(right_state["alt"]),
                }
            )
    if not frames:
        raise RuntimeError("No fill frames built.")
    return fill_meta, pd.concat(frames, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--adsc-dir", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--edge-taper-mode", choices=["hard", "smooth"], default="hard")
    ap.add_argument("--soft-taper-steps", type=str, default="0,0.25,0.5")
    ap.add_argument("--fixed-cap", type=float, default=300.0)
    ap.add_argument("--cap-le15", type=float, default=120.0)
    ap.add_argument("--cap-15-60", type=float, default=220.0)
    ap.add_argument("--cap-gt60", type=float, default=350.0)
    ap.add_argument("--edge-cap-factor", type=float, default=0.6)
    ap.add_argument("--gallery-count", type=int, default=20)
    ap.add_argument("--disable-endpoint-anchor-enforce", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(args.audit).head(args.max_samples)

    fill_meta, infer_frame = _build_fill_dataset(
        audit=audit,
        adsc_dir=Path(args.adsc_dir),
        context_minutes=int(cfg.get("data", {}).get("context_minutes", 5)),
    )

    pred_on = _predict_on_frame(cfg, Path(args.checkpoint), infer_frame, pred_key="pred_pos")
    pred_off = _predict_on_frame(cfg, Path(args.checkpoint), infer_frame, pred_key="pred_pos_main")
    pred_on["minute_ts"] = pd.to_datetime(pred_on["minute_ts"], utc=True)
    pred_off["minute_ts"] = pd.to_datetime(pred_off["minute_ts"], utc=True)

    enforce_endpoint_anchors = not bool(args.disable_endpoint_anchor_enforce)

    soft_steps = _parse_float_list(args.soft_taper_steps)
    if not soft_steps:
        soft_steps = [0.0, 0.25, 0.5]

    base_table = [
        {"fill_type": "*", "length_bucket": "<=15", "disable": True},
        {"fill_type": "adsc_anchor->adsc_anchor", "length_bucket": "15-30", "disable": True},
        {"fill_type": "adsc_anchor->adsc_anchor", "length_bucket": "30-60", "disable": True},
        {"fill_type": "adsc_anchor->adsb_segment", "length_bucket": "15-30", "disable": True},
        {"fill_type": "adsc_anchor->adsb_segment", "length_bucket": "30-60", "disable": True},
        {"fill_type": "adsb_segment->adsc_anchor", "length_bucket": "15-30", "disable": True},
        {"fill_type": "adsb_segment->adsc_anchor", "length_bucket": "30-60", "disable": True},
    ]
    conservative_ad2as_gt180 = [
        {"fill_type": "adsb_segment->adsc_anchor", "length_bucket": ">180", "base_scale": 0.5, "taper_mode": "custom_soft", "soft_steps": [0.0, 0.1, 0.3], "cap_mode": "edge_strict", "edge_cap_factor": 0.45},
    ]

    strategies: list[dict] = [
        {"name": "baseline", "baseline": True},
        {"name": "strategy_short_disable", "policy_table": base_table[:-2]},  # without ad2as60
        {"name": "strategy_short_disable_plus_ad2as60", "policy_table": base_table},
        {"name": "default_soft_taper", "policy_table": base_table, "taper_mode": "custom_soft", "soft_steps": soft_steps},
        {"name": "default_soft_taper_plus_len_bucket_cap", "policy_table": base_table, "taper_mode": "custom_soft", "soft_steps": soft_steps, "cap_mode": "length_bucket"},
        {"name": "adsb2adsb_15_60_disable", "policy_table": base_table + [{"fill_type": "adsb_segment->adsb_segment", "length_bucket": "15-30", "disable": True}, {"fill_type": "adsb_segment->adsb_segment", "length_bucket": "30-60", "disable": True}]},
        {"name": "adsb2adsb_15_60_scale05", "policy_table": base_table + [{"fill_type": "adsb_segment->adsb_segment", "length_bucket": "15-30", "base_scale": 0.5}, {"fill_type": "adsb_segment->adsb_segment", "length_bucket": "30-60", "base_scale": 0.5}]},
        {"name": "adsb2adsb_15_60_scale03", "policy_table": base_table + [{"fill_type": "adsb_segment->adsb_segment", "length_bucket": "15-30", "base_scale": 0.3}, {"fill_type": "adsb_segment->adsb_segment", "length_bucket": "30-60", "base_scale": 0.3}]},
        {"name": "adsb2adsb_15_30_scale05", "policy_table": base_table + [{"fill_type": "adsb_segment->adsb_segment", "length_bucket": "15-30", "base_scale": 0.5}]},
        {"name": "adsb2adsb_30_60_scale05", "policy_table": base_table + [{"fill_type": "adsb_segment->adsb_segment", "length_bucket": "30-60", "base_scale": 0.5}]},
        {"name": "ad2as_gt180_conservative_scale05", "policy_table": base_table + conservative_ad2as_gt180},
        {"name": "ad2as_gt180_conservative_scale03", "policy_table": base_table + [{"fill_type": "adsb_segment->adsc_anchor", "length_bucket": ">180", "base_scale": 0.3, "taper_mode": "custom_soft", "soft_steps": [0.0, 0.1, 0.25], "cap_mode": "edge_strict", "edge_cap_factor": 0.4}]},
        {"name": "default_soft_taper_with_fallback", "policy_table": base_table, "taper_mode": "custom_soft", "soft_steps": soft_steps, "fallback_policy_table": base_table + [{"fill_type": "adsb_segment->adsc_anchor", "length_bucket": ">180", "base_scale": 0.2, "taper_mode": "custom_soft", "soft_steps": [0.0, 0.05, 0.2], "cap_mode": "edge_strict", "edge_cap_factor": 0.35}, {"fill_type": "adsb_segment->adsb_segment", "length_bucket": "15-30", "base_scale": 0.3}, {"fill_type": "adsb_segment->adsb_segment", "length_bucket": "30-60", "base_scale": 0.3}]},
        {
            "name": "production_chain_v1",
            # default layer
            "policy_table": base_table
            + [
                # rule override 1: adsb->adsb 15~60 disable
                {"fill_type": "adsb_segment->adsb_segment", "length_bucket": "15-30", "disable": True},
                {"fill_type": "adsb_segment->adsb_segment", "length_bucket": "30-60", "disable": True},
                # rule override 2: adsb->adsc >180 conservative (not disable)
                {"fill_type": "adsb_segment->adsc_anchor", "length_bucket": ">180", "disable": False, "base_scale": 0.5, "taper_mode": "custom_soft", "soft_steps": [0.0, 0.1, 0.3], "cap_mode": "edge_strict", "edge_cap_factor": 0.45},
            ],
            "taper_mode": "custom_soft",
            "soft_steps": soft_steps,
            "cap_mode": "length_bucket",
            # fallback layer (post-check)
            "fallback_policy_table": base_table
            + [
                {"fill_type": "adsb_segment->adsb_segment", "length_bucket": "15-30", "disable": True},
                {"fill_type": "adsb_segment->adsb_segment", "length_bucket": "30-60", "disable": True},
                {"fill_type": "adsb_segment->adsc_anchor", "length_bucket": ">180", "disable": False, "base_scale": 0.2, "taper_mode": "custom_soft", "soft_steps": [0.0, 0.05, 0.2], "cap_mode": "edge_strict", "edge_cap_factor": 0.35},
                # broad fallback tightening
                {"fill_type": "adsb_segment->adsc_anchor", "length_bucket": "60-180", "disable": False, "base_scale": 0.35, "taper_mode": "custom_soft", "soft_steps": [0.0, 0.1, 0.25], "cap_mode": "edge_strict", "edge_cap_factor": 0.45},
            ],
        },
    ]
    strategy_rows: list[dict] = []
    strategy_segments: dict[str, dict[str, pd.DataFrame]] = {s["name"]: {} for s in strategies}

    for meta in fill_meta:
        sid = meta["fill_sid"]
        on = pred_on[pred_on["sample_id"].astype(str).eq(sid)].copy().sort_values("minute_ts").reset_index(drop=True)
        off = pred_off[pred_off["sample_id"].astype(str).eq(sid)].copy().sort_values("minute_ts").reset_index(drop=True)
        if len(on) < 2 or len(off) < 2:
            continue
        fill_type = f"{meta['left_block_type']}->{meta['right_block_type']}"
        on_alt = on["pred_alt"].to_numpy(dtype=float)
        off_alt = off["pred_alt"].to_numpy(dtype=float)
        for s in strategies:
            sname = s["name"]
            fallback_triggered = False
            if s.get("baseline", False):
                seg = on.copy()
                enabled = True
            else:
                final_alt, enabled = _apply_residual_strategy(
                    on_alt=on_alt,
                    off_alt=off_alt,
                    fill_type=fill_type,
                    fill_minutes=float(meta["fill_minutes"]),
                    strategy=s,
                    args=args,
                )
                seg = off.copy()
                seg["pred_alt"] = final_alt
                row_tmp = _segment_metrics(meta, seg, sname, enabled)
                if row_tmp["shape_abnormal_flag"] and s.get("fallback_policy_table"):
                    fb = {
                        "policy_table": s.get("fallback_policy_table", []),
                        "base_scale": s.get("base_scale", 1.0),
                        "taper_mode": s.get("taper_mode", "none"),
                        "soft_steps": s.get("soft_steps", [0.0, 0.25, 0.5]),
                        "cap_mode": s.get("cap_mode", "none"),
                        "fixed_cap": s.get("fixed_cap", args.fixed_cap),
                        "cap_le15": s.get("cap_le15", args.cap_le15),
                        "cap_15_60": s.get("cap_15_60", args.cap_15_60),
                        "cap_gt60": s.get("cap_gt60", args.cap_gt60),
                        "edge_cap_factor": s.get("edge_cap_factor", args.edge_cap_factor),
                    }
                    fb_alt, fb_enabled = _apply_residual_strategy(
                        on_alt=on_alt,
                        off_alt=off_alt,
                        fill_type=fill_type,
                        fill_minutes=float(meta["fill_minutes"]),
                        strategy=fb,
                        args=args,
                    )
                    seg = off.copy()
                    seg["pred_alt"] = fb_alt
                    enabled = fb_enabled
                    fallback_triggered = True
            if enforce_endpoint_anchors:
                seg = _enforce_segment_endpoint_anchors(seg, meta)
            strategy_segments[sname][sid] = seg
            rowm = _segment_metrics(meta, seg, sname, enabled)
            rowm["fallback_triggered"] = fallback_triggered
            strategy_rows.append(rowm)

    detail = pd.DataFrame(strategy_rows)
    detail.to_csv(out_dir / "fill_segment_strategy_detail.csv", index=False)

    # Global compare
    cmp_rows = []
    for sname, g in detail.groupby("strategy_name"):
        vc = g["quality_class"].value_counts().to_dict()
        cmp_rows.append(
            {
                "strategy_name": sname,
                "segment_count": int(len(g)),
                "abnormal_ratio": float(g["shape_abnormal_flag"].mean()),
                "overshoot_flag_ratio": float(g["overshoot_flag"].mean()),
                "edge_spike_flag_ratio": float(g["edge_spike_flag"].mean()),
                "mean_overshoot": float(g["overshoot_up"].mean()),
                "p90_overshoot": float(g["overshoot_up"].quantile(0.9)),
                "keep_count": int(vc.get("keep", 0)),
                "warn_count": int(vc.get("warn", 0)),
                "abnormal_count": int(vc.get("abnormal", 0)),
                "acceptable_ratio": float((g["quality_class"].isin(["keep", "warn"])).mean()),
                "fallback_trigger_count": int(g.get("fallback_triggered", pd.Series(False)).sum()) if "fallback_triggered" in g.columns else 0,
            }
        )
    cmp_df = pd.DataFrame(cmp_rows)
    base_row = cmp_df[cmp_df["strategy_name"].eq("baseline")].iloc[0]
    ref_row = cmp_df[cmp_df["strategy_name"].eq("strategy_short_disable")]
    ref_row = ref_row.iloc[0] if len(ref_row) else base_row
    cmp_df["delta_abnormal_vs_baseline"] = cmp_df["abnormal_ratio"] - float(base_row["abnormal_ratio"])
    cmp_df["delta_overshoot_vs_baseline"] = cmp_df["overshoot_flag_ratio"] - float(base_row["overshoot_flag_ratio"])
    cmp_df["delta_edge_spike_vs_baseline"] = cmp_df["edge_spike_flag_ratio"] - float(base_row["edge_spike_flag_ratio"])
    cmp_df["delta_abnormal_vs_short_disable"] = cmp_df["abnormal_ratio"] - float(ref_row["abnormal_ratio"])
    cmp_df.to_csv(out_dir / "fill_segment_strategy_compare.csv", index=False)

    # by type + length
    grp_rows = []
    for (sname, ftype, lb), g in detail.groupby(["strategy_name", "fill_type", "length_bucket"]):
        grp_rows.append(
            {
                "strategy_name": sname,
                "fill_type": ftype,
                "length_bucket": lb,
                "count": int(len(g)),
                "abnormal_ratio": float(g["shape_abnormal_flag"].mean()),
                "overshoot_flag_ratio": float(g["overshoot_flag"].mean()),
                "edge_spike_flag_ratio": float(g["edge_spike_flag"].mean()),
                "mean_overshoot": float(g["overshoot_up"].mean()),
                "p90_overshoot": float(g["overshoot_up"].quantile(0.9)),
                "mean_max_vertical_rate_inside": float(g["max_vertical_rate_inside"].mean()),
            }
        )
    grp_df = pd.DataFrame(grp_rows)
    grp_df.to_csv(out_dir / "fill_segment_strategy_by_type.csv", index=False)
    risk_rank = (
        grp_df[grp_df["count"] >= 5]
        .sort_values(["abnormal_ratio", "edge_spike_flag_ratio", "mean_overshoot"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    risk_rank.to_csv(out_dir / "fill_segment_risk_rank.csv", index=False)

    policy_rows = []
    for s in strategies:
        sname = s["name"]
        if s.get("baseline", False):
            policy_rows.append({"strategy_name": sname, "fill_type": "*", "length_bucket": "*", "disable": False, "base_scale": 1.0, "taper_mode": "none", "cap_mode": "none", "position_bucket": "all"})
            continue
        table = s.get("policy_table", []) or []
        if not table:
            policy_rows.append({"strategy_name": sname, "fill_type": "*", "length_bucket": "*", "disable": False, "base_scale": s.get("base_scale", 1.0), "taper_mode": s.get("taper_mode", "none"), "cap_mode": s.get("cap_mode", "none"), "position_bucket": "all"})
        for r in table:
            policy_rows.append(
                {
                    "strategy_name": sname,
                    "fill_type": r.get("fill_type", "*"),
                    "length_bucket": r.get("length_bucket", "*"),
                    "disable": r.get("disable", False),
                    "base_scale": r.get("base_scale", s.get("base_scale", 1.0)),
                    "taper_mode": r.get("taper_mode", s.get("taper_mode", "none")),
                    "cap_mode": r.get("cap_mode", s.get("cap_mode", "none")),
                    "position_bucket": "edge+middle",
                }
            )
    pd.DataFrame(policy_rows).to_csv(out_dir / "residual_policy_table.csv", index=False)

    # Compact action table for production chain (disable / conservative / default)
    prod = next((s for s in strategies if s["name"] == "production_chain_v1"), None)
    if prod is not None:
        compact_rows = []
        # Baseline default action
        compact_rows.append(
            {
                "chain_name": "production_chain_v1",
                "layer": "default",
                "fill_type": "*",
                "length_bucket": "*",
                "action": "default",
                "base_scale": prod.get("base_scale", 1.0),
                "taper_mode": prod.get("taper_mode", "custom_soft"),
                "soft_steps": ",".join([str(x) for x in prod.get("soft_steps", [])]),
                "cap_mode": prod.get("cap_mode", "length_bucket"),
            }
        )
        for r in prod.get("policy_table", []) or []:
            act = "disable" if r.get("disable", False) else "default"
            if (not r.get("disable", False)) and (float(r.get("base_scale", 1.0)) < 1.0 or str(r.get("cap_mode", "none")) != "none"):
                act = "conservative"
            compact_rows.append(
                {
                    "chain_name": "production_chain_v1",
                    "layer": "override",
                    "fill_type": r.get("fill_type", "*"),
                    "length_bucket": r.get("length_bucket", "*"),
                    "action": act,
                    "base_scale": r.get("base_scale", prod.get("base_scale", 1.0)),
                    "taper_mode": r.get("taper_mode", prod.get("taper_mode", "custom_soft")),
                    "soft_steps": ",".join([str(x) for x in r.get("soft_steps", prod.get("soft_steps", []))]) if (r.get("soft_steps", prod.get("soft_steps", []))) else "",
                    "cap_mode": r.get("cap_mode", prod.get("cap_mode", "length_bucket")),
                }
            )
        for r in prod.get("fallback_policy_table", []) or []:
            act = "disable" if r.get("disable", False) else "default"
            if (not r.get("disable", False)) and (float(r.get("base_scale", 1.0)) < 1.0 or str(r.get("cap_mode", "none")) != "none"):
                act = "conservative"
            compact_rows.append(
                {
                    "chain_name": "production_chain_v1",
                    "layer": "fallback",
                    "fill_type": r.get("fill_type", "*"),
                    "length_bucket": r.get("length_bucket", "*"),
                    "action": act,
                    "base_scale": r.get("base_scale", 1.0),
                    "taper_mode": r.get("taper_mode", prod.get("taper_mode", "custom_soft")),
                    "soft_steps": ",".join([str(x) for x in r.get("soft_steps", prod.get("soft_steps", []))]) if (r.get("soft_steps", prod.get("soft_steps", []))) else "",
                    "cap_mode": r.get("cap_mode", prod.get("cap_mode", "length_bucket")),
                }
            )
        pd.DataFrame(compact_rows).to_csv(out_dir / "residual_policy_actions_compact.csv", index=False)

    # Fallback trigger audit for production chain
    if "fallback_triggered" in detail.columns:
        prod_detail = detail[detail["strategy_name"].eq("production_chain_v1")].copy()
        fb = prod_detail[prod_detail["fallback_triggered"].eq(True)].copy()
        if len(fb):
            fb["fill_type"] = fb["left_block_type"] + "->" + fb["right_block_type"]
            fb["length_bucket5"] = fb["fill_minutes"].apply(_bucket_len5)
            dist_type = (
                fb.groupby("fill_type", as_index=False)
                .size()
                .rename(columns={"size": "count"})
                .sort_values("count", ascending=False)
            )
            dist_len = (
                fb.groupby("length_bucket5", as_index=False)
                .size()
                .rename(columns={"size": "count"})
                .sort_values("count", ascending=False)
            )
            dist_pos = (
                fb.groupby("anomaly_position", as_index=False)
                .size()
                .rename(columns={"size": "count"})
                .sort_values("count", ascending=False)
            )
            fb.to_csv(out_dir / "fallback_trigger_segments.csv", index=False)
            dist_type.to_csv(out_dir / "fallback_trigger_distribution_fill_type.csv", index=False)
            dist_len.to_csv(out_dir / "fallback_trigger_distribution_length_bucket.csv", index=False)
            dist_pos.to_csv(out_dir / "fallback_trigger_distribution_anomaly_position.csv", index=False)

    # gallery (3-column compare)
    gal_dir = out_dir / "fill_segment_strategy_gallery"
    triple_dir = gal_dir / "triple_compare"
    main_dir = gal_dir / "main_display"
    abnormal_dir = gal_dir / "abnormal_gallery"
    triple_dir.mkdir(parents=True, exist_ok=True)
    main_dir.mkdir(parents=True, exist_ok=True)
    abnormal_dir.mkdir(parents=True, exist_ok=True)

    base = detail[detail["strategy_name"].eq("baseline")].sort_values("overshoot_up", ascending=False).head(args.gallery_count)
    triple_show = [
        "baseline",
        "strategy_short_disable_plus_ad2as60",
        "default_soft_taper_plus_len_bucket_cap",
    ]
    for _, r in base.iterrows():
        sid = f"{r['sample_id']}__{r['fill_id']}"
        fig, axes = plt.subplots(1, 3, figsize=(15, 3.4), sharey=True)
        for j, sname in enumerate(triple_show):
            seg = strategy_segments[sname].get(sid)
            if seg is None or seg.empty:
                axes[j].set_title(f"{sname}\\nmissing")
                continue
            row = detail[
                (detail["strategy_name"].eq(sname))
                & (detail["sample_id"].eq(r["sample_id"]))
                & (detail["fill_id"].eq(r["fill_id"]))
            ].iloc[0]
            axes[j].plot(seg["minute_ts"], seg["pred_alt"], color="#e7298a", lw=1.6)
            axes[j].scatter([seg["minute_ts"].iloc[0], seg["minute_ts"].iloc[-1]], [row["left_boundary_alt"], row["right_boundary_alt"]], c="#1f78b4", s=25)
            axes[j].set_title(
                f"{sname}\nq={row['quality_class']} ov={row['overshoot_up']:.1f}"
            )
        fig.suptitle(
            f"{r['sample_id']} {r['fill_id']} {r['fill_type']} {int(r['fill_minutes'])}m",
            fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(triple_dir / f"{r['sample_id']}_{r['fill_id']}_triple.png", dpi=140)
        plt.close(fig)

    # Main display rule: keep+warn only, abnormal excluded
    for sname in sorted(strategy_segments.keys()):
        s_main = main_dir / sname
        s_abn = abnormal_dir / sname
        s_main.mkdir(parents=True, exist_ok=True)
        s_abn.mkdir(parents=True, exist_ok=True)
        sub = detail[detail["strategy_name"].eq(sname)].copy()
        for _, r in sub.sort_values("overshoot_up", ascending=False).head(args.gallery_count).iterrows():
            sid = f"{r['sample_id']}__{r['fill_id']}"
            seg = strategy_segments[sname].get(sid)
            if seg is None or seg.empty:
                continue
            fig, ax = plt.subplots(1, 1, figsize=(8, 3))
            ax.plot(seg["minute_ts"], seg["pred_alt"], color="#e7298a", lw=1.6)
            ax.scatter([seg["minute_ts"].iloc[0], seg["minute_ts"].iloc[-1]], [r["left_boundary_alt"], r["right_boundary_alt"]], c="#1f78b4", s=25)
            ax.set_title(
                f"{r['sample_id']} {r['fill_id']} {r['fill_type']} {int(r['fill_minutes'])}m | "
                f"q={r['quality_class']} | ov={r['overshoot_up']:.1f}"
            )
            fig.tight_layout()
            outp = (s_main if r["quality_class"] in {"keep", "warn"} else s_abn) / f"{r['sample_id']}_{r['fill_id']}.png"
            fig.savefig(outp, dpi=140)
            plt.close(fig)

    # Summary
    cmp = cmp_df.sort_values("abnormal_ratio")
    lines = [
        "# Fill Segment Strategy Compare Summary",
        "",
        f"- strategies: {', '.join(cmp['strategy_name'].tolist())}",
        f"- soft_taper_steps: {soft_steps}",
        f"- cap(fixed={args.fixed_cap}, le15={args.cap_le15}, 15_60={args.cap_15_60}, gt60={args.cap_gt60}, edge_factor={args.edge_cap_factor})",
        "",
    ]
    for _, r in cmp.iterrows():
        lines.append(
            f"- {r['strategy_name']}: abnormal={r['abnormal_ratio']:.4f}, "
            f"overshoot={r['overshoot_flag_ratio']:.4f}, edge_spike={r['edge_spike_flag_ratio']:.4f}, "
            f"keep/warn/abn={int(r['keep_count'])}/{int(r['warn_count'])}/{int(r['abnormal_count'])}"
        )
    (out_dir / "fill_segment_strategy_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
