from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.datasets import DatasetConfig, TrajectoryDataset, trajectory_collate_fn
from src.inference.segment_alt_policy import SegmentResidualPolicy
from src.models import TrajectoryRecoveryModel
from src.preprocessing.feature_builder import FeatureBuilder
from src.preprocessing.standardize import apply_standardizer, load_standardizer
from src.training.altitude_governance import add_anchor_alt_features, add_vertical_v2_features
from src.training.alt_target import apply_alt_target_transform, invert_alt_target_transform
from src.training.coords import build_anchor_alt_tracks, build_anchor_pair_tracks, prepare_model_coordinates, restore_to_latlon
from src.training.target_norm import denormalize_coords, load_target_stats, normalize_coords
from src.training.utils import load_config, set_seed, validate_inference_frame


@dataclass
class ReplaySample:
    sample_id: str
    flight_id: str
    flight_date: str
    anchor_start_time: pd.Timestamp
    anchor_end_time: pd.Timestamp
    gap_minutes: int
    anchor_start_lat: float
    anchor_start_lon: float
    anchor_start_alt: float
    anchor_end_lat: float
    anchor_end_lon: float
    anchor_end_alt: float
    left_known_time: pd.Timestamp
    right_known_time: pd.Timestamp
    left_fill_minutes: int
    middle_gap_minutes: int
    right_fill_minutes: int
    pre_gap_adsb_context_len: int
    post_gap_adsb_context_len: int
    pre_gap_adsb_context_complete: bool
    post_gap_adsb_context_complete: bool
    gap_inner_adsb_used: bool
    matched_adsb_flight: str
    adsb_match_confidence_rule: str
    task_type: str
    frame: pd.DataFrame


def _extract_block_boundary_state(block: dict, side: str) -> dict:
    if side not in {"start", "end"}:
        raise ValueError("side must be 'start' or 'end'")
    return {
        "time": block["start_time"] if side == "start" else block["end_time"],
        "lat": block["start_lat"] if side == "start" else block["end_lat"],
        "lon": block["start_lon"] if side == "start" else block["end_lon"],
        "alt": block["start_alt"] if side == "start" else block["end_alt"],
        "source_type": block["block_type"],
        "is_anchor": block["block_type"] == "adsc_anchor",
    }


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371000.0 * 2 * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    b = math.degrees(math.atan2(x, y))
    return (b + 360.0) % 360.0


def _angle_diff_deg(a: float, b: float) -> float:
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def _segment_bucket3(fill_minutes: float) -> str:
    m = float(fill_minutes)
    if m <= 15.0:
        return "short"
    if m <= 60.0:
        return "medium"
    return "long"


def _anchor_pattern(left_block_type: str, right_block_type: str) -> str:
    left_anchor = str(left_block_type) == "adsc_anchor"
    right_anchor = str(right_block_type) == "adsc_anchor"
    if left_anchor and right_anchor:
        return "two_anchor"
    if left_anchor or right_anchor:
        return "asymmetric"
    return "sparse_context"


def _quality_flags_from_altitude(
    alt: np.ndarray,
    left_boundary_alt: float,
    right_boundary_alt: float,
) -> dict:
    if alt is None or len(alt) == 0:
        return {
            "overshoot_up": 0.0,
            "undershoot_down": 0.0,
            "overshoot_flag": False,
            "edge_spike_flag": False,
            "abnormal_flag": False,
            "warn_flag": False,
            "keep_flag": True,
            "max_vertical_rate_inside": 0.0,
            "anomaly_position": "none",
        }
    left_alt = float(left_boundary_alt)
    right_alt = float(right_boundary_alt)
    alt_min = float(np.min(alt))
    alt_max = float(np.max(alt))
    peak_idx = int(np.argmax(alt))
    trough_idx = int(np.argmin(alt))
    overshoot_up = alt_max - max(left_alt, right_alt)
    undershoot_down = min(left_alt, right_alt) - alt_min
    diffs = np.diff(alt)
    max_vr = float(np.max(np.abs(diffs))) if len(diffs) else 0.0
    first_two_peak = peak_idx <= 2 or trough_idx <= 2
    last_two_peak = peak_idx >= len(alt) - 3 or trough_idx >= len(alt) - 3
    peak_then_return = first_two_peak or last_two_peak
    overshoot_flag = bool(overshoot_up > 200.0)
    undershoot_flag = bool(undershoot_down > 200.0)
    edge_spike_flag = bool((first_two_peak or last_two_peak) and (max_vr > 300.0))
    abnormal_flag = bool((overshoot_flag or undershoot_flag) and (edge_spike_flag or peak_then_return))
    warn_flag = bool((overshoot_flag or edge_spike_flag) and (not abnormal_flag))
    keep_flag = bool((not abnormal_flag) and (not warn_flag))
    if first_two_peak:
        anomaly_pos = "first_two_steps"
    elif last_two_peak:
        anomaly_pos = "last_two_steps"
    elif overshoot_flag or undershoot_flag:
        anomaly_pos = "middle"
    else:
        anomaly_pos = "none"
    return {
        "overshoot_up": float(overshoot_up),
        "undershoot_down": float(undershoot_down),
        "overshoot_flag": overshoot_flag,
        "edge_spike_flag": edge_spike_flag,
        "abnormal_flag": abnormal_flag,
        "warn_flag": warn_flag,
        "keep_flag": keep_flag,
        "max_vertical_rate_inside": float(max_vr),
        "anomaly_position": anomaly_pos,
    }


def _overshoot_position_and_cause(
    final_alt: np.ndarray,
    baseline_alt: np.ndarray,
    left_boundary_alt: float,
    right_boundary_alt: float,
    overshoot_thresh: float = 200.0,
) -> dict:
    """Decompose overshoot by reference and position.

    Notes:
    - In real ADS-C replay there is no guaranteed gap-inner minute-level GT.
    - `overshoot_vs_gt_range_max` is therefore emitted as NaN by default.
    """
    if final_alt is None or len(final_alt) == 0:
        return {
            "overshoot_vs_baseline_max": 0.0,
            "overshoot_vs_anchor_envelope_max": 0.0,
            "overshoot_vs_gt_range_max": float("nan"),
            "overshoot_edge_left_max": 0.0,
            "overshoot_edge_right_max": 0.0,
            "overshoot_middle_max": 0.0,
            "overshoot_left_edge_flag": False,
            "overshoot_right_edge_flag": False,
            "overshoot_middle_flag": False,
            "overshoot_peak_pos_ratio": float("nan"),
            "baseline_overshoot_max": 0.0,
            "final_overshoot_max": 0.0,
            "baseline_overshoot_flag": False,
            "final_overshoot_flag": False,
            "cause_label": "unclear",
            "overshoot_reference_note": "anchor_envelope_threshold_200",
        }

    n = int(len(final_alt))
    edge_w = max(1, min(2, n))
    upper = max(float(left_boundary_alt), float(right_boundary_alt))
    final_arr = np.asarray(final_alt, dtype=float)
    base_arr = np.asarray(baseline_alt, dtype=float) if baseline_alt is not None and len(baseline_alt) == n else final_arr.copy()

    over_anchor = np.maximum(final_arr - upper, 0.0)
    over_base = np.maximum(final_arr - base_arr, 0.0)
    base_over_anchor = np.maximum(base_arr - upper, 0.0)
    final_over_anchor = np.maximum(final_arr - upper, 0.0)

    left_max = float(np.max(over_anchor[:edge_w])) if n > 0 else 0.0
    right_max = float(np.max(over_anchor[-edge_w:])) if n > 0 else 0.0
    if n > 2 * edge_w:
        middle_max = float(np.max(over_anchor[edge_w:-edge_w]))
    else:
        middle_max = 0.0

    peak_idx = int(np.argmax(over_anchor)) if n > 0 else 0
    peak_ratio = float(peak_idx / max(1, n - 1))

    baseline_overshoot_max = float(np.max(base_over_anchor))
    final_overshoot_max = float(np.max(final_over_anchor))
    baseline_overshoot_flag = bool(baseline_overshoot_max > overshoot_thresh)
    final_overshoot_flag = bool(final_overshoot_max > overshoot_thresh)

    delta_used_abs = float(np.mean(np.abs(final_arr - base_arr)))
    if baseline_overshoot_flag and final_overshoot_flag and delta_used_abs <= 50.0:
        cause = "baseline_driven"
    elif (not baseline_overshoot_flag) and final_overshoot_flag:
        cause = "residual_driven"
    elif baseline_overshoot_flag and final_overshoot_flag and delta_used_abs > 50.0:
        cause = "mixed"
    else:
        cause = "unclear"

    return {
        "overshoot_vs_baseline_max": float(np.max(over_base)),
        "overshoot_vs_anchor_envelope_max": float(np.max(over_anchor)),
        "overshoot_vs_gt_range_max": float("nan"),
        "overshoot_edge_left_max": left_max,
        "overshoot_edge_right_max": right_max,
        "overshoot_middle_max": middle_max,
        "overshoot_left_edge_flag": bool(left_max > overshoot_thresh),
        "overshoot_right_edge_flag": bool(right_max > overshoot_thresh),
        "overshoot_middle_flag": bool(middle_max > overshoot_thresh),
        "overshoot_peak_pos_ratio": peak_ratio,
        "baseline_overshoot_max": baseline_overshoot_max,
        "final_overshoot_max": final_overshoot_max,
        "baseline_overshoot_flag": baseline_overshoot_flag,
        "final_overshoot_flag": final_overshoot_flag,
        "cause_label": cause,
        "overshoot_reference_note": "anchor_envelope_threshold_200",
    }


def _apply_left_edge_projection(
    alt_main: np.ndarray,
    left_boundary_alt: float,
    right_boundary_alt: float,
    enabled: bool,
    mode: str = "envelope",
    steps: int = 2,
    left_local_band: float = 200.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Project left-edge steps to a safe band.

    Returns:
      projected_alt, projection_delta (projected - raw)
    """
    x = np.asarray(alt_main, dtype=float).copy()
    d = np.zeros_like(x, dtype=float)
    if (not enabled) or x.size == 0:
        return x, d
    k = max(0, min(int(steps), int(x.size)))
    if k <= 0:
        return x, d

    lb = float(left_boundary_alt)
    rb = float(right_boundary_alt)
    mode_s = str(mode).lower()
    for t in range(k):
        raw = float(x[t])
        if mode_s == "left_local_band":
            lo, hi = lb - float(left_local_band), lb + float(left_local_band)
        else:
            lo, hi = min(lb, rb), max(lb, rb)
        proj = float(np.clip(raw, lo, hi))
        x[t] = proj
        d[t] = proj - raw
    return x, d


def _apply_left_edge_smoothing(
    alt_final: np.ndarray,
    left_boundary_alt: float,
    enabled: bool,
    mode: str = "left_blend",
    steps: int = 3,
    blend_betas: list[float] | None = None,
    slope_cap: float = 300.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply minimal replay-side smoothing only on left edge."""
    x = np.asarray(alt_final, dtype=float).copy()
    d = np.zeros_like(x, dtype=float)
    if (not enabled) or x.size == 0:
        return x, d
    k = max(0, min(int(steps), int(x.size)))
    if k <= 0:
        return x, d

    m = str(mode).lower()
    lb = float(left_boundary_alt)
    if m == "slope_cap":
        cap = float(max(1e-6, slope_cap))
        prev = lb
        for t in range(k):
            raw = float(x[t])
            lo, hi = prev - cap, prev + cap
            sm = float(np.clip(raw, lo, hi))
            x[t] = sm
            d[t] = sm - raw
            prev = sm
        return x, d

    # default: left_blend
    betas = list(blend_betas or [0.5, 0.3, 0.1])
    if len(betas) < k:
        betas = betas + [betas[-1] if len(betas) else 0.1] * (k - len(betas))
    for t in range(k):
        b = float(max(0.0, min(1.0, betas[t])))
        raw = float(x[t])
        sm = (1.0 - b) * raw + b * lb
        x[t] = sm
        d[t] = sm - raw
    return x, d


def _apply_right_edge_smoothing(
    alt_final: np.ndarray,
    right_boundary_alt: float,
    enabled: bool,
    mode: str = "right_blend",
    steps: int = 2,
    blend_betas: list[float] | None = None,
    right_local_band: float = 200.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply minimal replay-side smoothing only on right edge."""
    x = np.asarray(alt_final, dtype=float).copy()
    d = np.zeros_like(x, dtype=float)
    if (not enabled) or x.size == 0:
        return x, d
    # step2/step3 near right boundary; do not alter last boundary step
    k = max(0, min(int(steps), max(0, int(x.size) - 1)))
    if k <= 0:
        return x, d

    m = str(mode).lower()
    rb = float(right_boundary_alt)
    if m == "right_local_band":
        eps = float(max(1e-6, right_local_band))
        lo, hi = rb - eps, rb + eps
        for j in range(k):
            t = int(x.size - 2 - j)
            raw = float(x[t])
            sm = float(np.clip(raw, lo, hi))
            x[t] = sm
            d[t] = sm - raw
        return x, d

    # default: right_blend; j=0 is step2 from right (index -2), j=1 is step3 (index -3)
    betas = list(blend_betas or [0.5, 0.2])
    if len(betas) < k:
        betas = betas + [betas[-1] if len(betas) else 0.2] * (k - len(betas))
    for j in range(k):
        t = int(x.size - 2 - j)
        b = float(max(0.0, min(1.0, betas[j])))
        raw = float(x[t])
        sm = (1.0 - b) * raw + b * rb
        x[t] = sm
        d[t] = sm - raw
    return x, d


def _apply_conditional_rightstep2_fuse(
    alt_final: np.ndarray,
    *,
    segment_bucket: str,
    anchor_pattern: str,
    right_boundary_alt: float,
    enabled: bool,
    target_bucket: str = "medium",
    target_pattern: str = "two_anchor",
    tau_jump: float = 200.0,
    mode: str = "local_interp",
    fuse_lambda: float = 0.5,
    use_second_condition: bool = False,
    tau_curve: float = 200.0,
    right_local_band: float = 200.0,
) -> tuple[np.ndarray, np.ndarray, bool, float, float]:
    """Conditionally repair only right_step2 (index -2) for medium+two_anchor.

    Returns:
      fused_alt, fuse_delta(full-length), triggered_flag, abs_jump_tminus2_to_tminus1, abs_second_diff_right
    """
    x = np.asarray(alt_final, dtype=float).copy()
    d = np.zeros_like(x, dtype=float)
    n = int(x.size)
    if (not enabled) or n < 3:
        return x, d, False, float("nan"), float("nan")

    sb = str(segment_bucket)
    ap = str(anchor_pattern)
    if sb != str(target_bucket) or ap != str(target_pattern):
        return x, d, False, float("nan"), float("nan")

    # Geometry probes near right boundary:
    # t-2 = x[-2], t-1 = x[-1], t-3 = x[-3]
    jump = float(x[-2] - x[-1])  # jump_tminus2_to_tminus1
    second_diff = float((x[-1] - x[-2]) - (x[-2] - x[-3]))  # second_diff_right
    cond = bool(abs(jump) > float(tau_jump))
    if bool(use_second_condition):
        cond = cond and bool(abs(second_diff) > float(tau_curve))
    if not cond:
        return x, d, False, abs(jump), abs(second_diff)

    raw = float(x[-2])
    m = str(mode).lower()
    if m == "right_local_band":
        eps = float(max(1e-6, right_local_band))
        fused = float(np.clip(raw, float(right_boundary_alt) - eps, float(right_boundary_alt) + eps))
    else:
        # default: local interpolation between t-3 and right boundary anchor
        lam = float(max(0.0, min(1.0, fuse_lambda)))
        fused = lam * float(x[-3]) + (1.0 - lam) * float(right_boundary_alt)
    x[-2] = fused
    d[-2] = fused - raw
    return x, d, True, abs(jump), abs(second_diff)


def _edge_spike_position_stats(
    alt: np.ndarray,
    left_boundary_alt: float,
    right_boundary_alt: float,
    jump_thresh: float = 300.0,
) -> dict:
    arr = np.asarray(alt, dtype=float) if alt is not None else np.zeros((0,), dtype=float)
    n = int(arr.size)
    if n == 0:
        return {
            "left_step1_spike_flag": False,
            "left_step2_spike_flag": False,
            "right_step1_spike_flag": False,
            "right_step2_spike_flag": False,
            "left_step1_jump": float("nan"),
            "left_step2_jump": float("nan"),
            "right_step1_jump": float("nan"),
            "right_step2_jump": float("nan"),
        }
    lb = float(left_boundary_alt)
    rb = float(right_boundary_alt)
    left_step1 = float(arr[0] - lb)
    left_step2 = float(arr[1] - arr[0]) if n >= 2 else float("nan")
    right_step1 = float(rb - arr[-1])
    right_step2 = float(arr[-1] - arr[-2]) if n >= 2 else float("nan")

    def _flag(v: float) -> bool:
        return bool(np.isfinite(v) and abs(v) > float(jump_thresh))

    return {
        "left_step1_spike_flag": _flag(left_step1),
        "left_step2_spike_flag": _flag(left_step2),
        "right_step1_spike_flag": _flag(right_step1),
        "right_step2_spike_flag": _flag(right_step2),
        "left_step1_jump": left_step1,
        "left_step2_jump": left_step2,
        "right_step1_jump": right_step1,
        "right_step2_jump": right_step2,
    }


def _boundary_alt_from_model_obs(
    obs_for_model: torch.Tensor,
    obs_mask: torch.Tensor,
    seq_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, t_len, _ = obs_for_model.shape
    left = torch.zeros((bsz,), device=obs_for_model.device, dtype=obs_for_model.dtype)
    right = torch.zeros((bsz,), device=obs_for_model.device, dtype=obs_for_model.dtype)
    obs_alt = obs_for_model[..., 2]
    for i in range(bsz):
        valid = seq_mask[i] > 0.5
        obs = (obs_mask[i] > 0.5) & valid
        valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
        obs_idx = torch.nonzero(obs, as_tuple=False).flatten()
        if obs_idx.numel() == 0:
            if valid_idx.numel() == 0:
                left[i] = 0.0
                right[i] = 0.0
            else:
                left[i] = obs_alt[i, valid_idx[0]]
                right[i] = obs_alt[i, valid_idx[-1]]
            continue
        gap = (~obs) & valid
        best_s, best_e, best_len = -1, -1, 0
        t = 0
        while t < t_len:
            if not bool(gap[t]):
                t += 1
                continue
            s = t
            while t < t_len and bool(gap[t]):
                t += 1
            e = t
            if (e - s) > best_len:
                best_s, best_e, best_len = s, e, e - s
        if best_len <= 0:
            left[i] = obs_alt[i, obs_idx[0]]
            right[i] = obs_alt[i, obs_idx[-1]]
            continue
        left_idx = best_s - 1 if (best_s - 1 >= 0 and bool(obs[best_s - 1])) else None
        right_idx = best_e if (best_e < t_len and bool(obs[best_e])) else None
        if left_idx is None:
            cand = obs_idx[obs_idx < best_s]
            if cand.numel() > 0:
                left_idx = int(cand[-1].item())
        if right_idx is None:
            cand = obs_idx[obs_idx >= best_e]
            if cand.numel() > 0:
                right_idx = int(cand[0].item())
        if left_idx is None and right_idx is None:
            left_idx = int(obs_idx[0].item())
            right_idx = int(obs_idx[-1].item())
        elif left_idx is None:
            left_idx = int(right_idx)  # type: ignore[arg-type]
        elif right_idx is None:
            right_idx = int(left_idx)
        left[i] = obs_alt[i, int(left_idx)]
        right[i] = obs_alt[i, int(right_idx)]
    return left, right


def _boundary_alt_from_batch_meta(
    left_boundary_alt: torch.Tensor,
    right_boundary_alt: torch.Tensor,
    *,
    u_relative_anchor: bool,
    target_norm_stats: dict | None,
    alt_target_transform_mode: str,
    alt_target_clip_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if u_relative_anchor:
        left_raw = torch.zeros_like(left_boundary_alt)
        right_raw = right_boundary_alt - left_boundary_alt
    else:
        left_raw = left_boundary_alt
        right_raw = right_boundary_alt

    def _map(raw_alt: torch.Tensor) -> torch.Tensor:
        z = torch.zeros((raw_alt.shape[0], 1, 3), device=raw_alt.device, dtype=raw_alt.dtype)
        z[..., 2] = raw_alt.view(-1, 1)
        z = apply_alt_target_transform(
            z,
            mode=alt_target_transform_mode,
            clip_value=alt_target_clip_value,
        )
        z = normalize_coords(z, target_norm_stats)
        return z[:, 0, 2]

    return _map(left_raw), _map(right_raw)


def _signed_overflow_vs_band(arr: np.ndarray, lower: float, upper: float) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    out = np.zeros_like(x, dtype=float)
    hi = x > float(upper)
    lo = x < float(lower)
    out[hi] = x[hi] - float(upper)
    out[lo] = x[lo] - float(lower)
    return out


def _series_overflow_stats(
    arr: np.ndarray,
    lower: float,
    upper: float,
    thresh: float = 200.0,
) -> dict:
    if arr is None or len(arr) == 0:
        return {
            "flag": False,
            "max_abs": 0.0,
            "trigger_step": -1,
            "trigger_value": 0.0,
            "left_1_delta": float("nan"),
            "left_2_delta": float("nan"),
            "right_1_delta": float("nan"),
            "right_2_delta": float("nan"),
        }
    overflow = _signed_overflow_vs_band(arr, lower=lower, upper=upper)
    abs_over = np.abs(overflow)
    trig_idx = np.where(abs_over > float(thresh))[0]
    trigger_step = int(trig_idx[0]) if len(trig_idx) else -1
    trigger_value = float(overflow[trigger_step]) if trigger_step >= 0 else 0.0
    n = int(len(arr))
    return {
        "flag": bool((abs_over > float(thresh)).any()),
        "max_abs": float(np.max(abs_over)),
        "trigger_step": trigger_step,
        "trigger_value": trigger_value,
        "left_1_delta": float(overflow[0]) if n >= 1 else float("nan"),
        "left_2_delta": float(overflow[1]) if n >= 2 else float("nan"),
        "right_1_delta": float(overflow[-1]) if n >= 1 else float("nan"),
        "right_2_delta": float(overflow[-2]) if n >= 2 else float("nan"),
    }


def _build_segment_level_audit(
    samples: list[ReplaySample],
    fill_meta_by_sample: dict[str, list[dict]],
    pred_fill_on: pd.DataFrame,
    pred_fill_off: pd.DataFrame,
    segment_policy: SegmentResidualPolicy,
    task_type: str,
    projection_cfg: dict | None = None,
    smoothing_cfg: dict | None = None,
    replay_alt_series_source: str = "policy_post_alt",
) -> pd.DataFrame:
    rows: list[dict] = []
    point_rows: list[dict] = []
    sample_to_flight = {s.sample_id: s.flight_id for s in samples}
    for s in samples:
        fills = fill_meta_by_sample.get(s.sample_id, [])
        for f in fills:
            fill_sid = str(f["fill_sid"])
            seg_on = pred_fill_on[pred_fill_on["sample_id"].astype(str).eq(fill_sid)].copy()
            if seg_on.empty:
                continue
            seg_on["minute_ts"] = pd.to_datetime(seg_on["minute_ts"], utc=True)
            seg_on = seg_on.sort_values("minute_ts").reset_index(drop=True)
            seg_off = pred_fill_off[pred_fill_off["sample_id"].astype(str).eq(fill_sid)].copy()
            if len(seg_off):
                seg_off["minute_ts"] = pd.to_datetime(seg_off["minute_ts"], utc=True)
                seg_off = seg_off.sort_values("minute_ts").reset_index(drop=True)

            core_start = pd.to_datetime(f["fill_start_time"], utc=True)
            core_end = pd.to_datetime(f["fill_end_time"], utc=True)
            core_mask = (seg_on["minute_ts"] >= core_start) & (seg_on["minute_ts"] <= core_end)
            seg_core = seg_on.loc[core_mask].copy()
            if seg_core.empty:
                seg_core = seg_on.copy()

            on_alt = seg_core["pred_alt"].to_numpy(dtype=float)
            on_alt_main = seg_core["pred_alt_main"].to_numpy(dtype=float) if "pred_alt_main" in seg_core.columns else on_alt.copy()
            gate_arr = seg_core["gate"].to_numpy(dtype=float) if "gate" in seg_core.columns else np.full_like(on_alt, np.nan)
            dms_cand_arr = seg_core["delta_candidate"].to_numpy(dtype=float) if "delta_candidate" in seg_core.columns else np.full_like(on_alt, np.nan)
            dms_used_arr = seg_core["delta_used"].to_numpy(dtype=float) if "delta_used" in seg_core.columns else np.full_like(on_alt, np.nan)
            left_wrong_arr = (
                seg_core["left_edge_wrong_direction"].to_numpy(dtype=float)
                if "left_edge_wrong_direction" in seg_core.columns
                else np.full_like(on_alt, np.nan)
            )
            risk_level_row = str(seg_on["risk_level"].iloc[0]) if "risk_level" in seg_on.columns else "unknown"
            matched_rule_row = str(seg_on["matched_risk_rule"].iloc[0]) if "matched_risk_rule" in seg_on.columns else "unknown"
            risk_flag_teacher_row = float(seg_on["risk_flag_teacher"].iloc[0]) if "risk_flag_teacher" in seg_on.columns else 0.0
            teacher_scale_row = float(seg_on["teacher_scale"].iloc[0]) if "teacher_scale" in seg_on.columns else 1.0
            residual_rmax_row = float(seg_on["residual_rmax_ft"].iloc[0]) if "residual_rmax_ft" in seg_on.columns else float("nan")
            if len(seg_off):
                core_mask_off = (seg_off["minute_ts"] >= core_start) & (seg_off["minute_ts"] <= core_end)
                seg_off_core = seg_off.loc[core_mask_off].copy()
                if seg_off_core.empty:
                    seg_off_core = seg_off.copy()
            else:
                seg_off_core = seg_off

            if len(seg_off_core) == len(seg_core):
                off_alt = seg_off_core["pred_alt"].to_numpy(dtype=float)
            else:
                off_alt = on_alt.copy()

            proj_enabled = bool((projection_cfg or {}).get("use_left_edge_projection", False))
            proj_mode = str((projection_cfg or {}).get("left_edge_projection_mode", "envelope"))
            proj_steps = int((projection_cfg or {}).get("left_edge_projection_steps", 2))
            proj_band = float((projection_cfg or {}).get("left_local_band_ft", 200.0))
            pred_alt_main_series_raw = on_alt_main.copy()
            pred_alt_main_series, proj_delta = _apply_left_edge_projection(
                alt_main=pred_alt_main_series_raw,
                left_boundary_alt=float(f.get("left_boundary_alt", pred_alt_main_series_raw[0] if len(pred_alt_main_series_raw) else 0.0)),
                right_boundary_alt=float(f.get("right_boundary_alt", pred_alt_main_series_raw[-1] if len(pred_alt_main_series_raw) else 0.0)),
                enabled=proj_enabled,
                mode=proj_mode,
                steps=proj_steps,
                left_local_band=proj_band,
            )
            pred_alt_final_series = pred_alt_main_series.copy()
            if segment_policy.enabled and len(seg_off) == len(seg_on):
                final_alt, meta = segment_policy.apply(
                    on_alt=pred_alt_main_series,
                    off_alt=off_alt,
                    fill_minutes=float(f["fill_minutes"]),
                    left_block_type=str(f["left_block_type"]),
                    right_block_type=str(f["right_block_type"]),
                )
            else:
                final_alt = pred_alt_main_series.copy()
                meta = None
            pred_alt_final_raw_series = final_alt.copy()
            smooth_enabled = bool((smoothing_cfg or {}).get("use_edge_smoothing_projection", False))
            smooth_mode = str((smoothing_cfg or {}).get("edge_smoothing_mode", "left_blend"))
            smooth_steps = int((smoothing_cfg or {}).get("left_edge_smoothing_steps", 3))
            smooth_betas = (smoothing_cfg or {}).get("left_blend_betas", [0.5, 0.3, 0.1])
            smooth_cap = float((smoothing_cfg or {}).get("left_slope_cap_ft", 300.0))
            policy_post_alt, smooth_delta = _apply_left_edge_smoothing(
                alt_final=pred_alt_final_raw_series,
                left_boundary_alt=float(f.get("left_boundary_alt", pred_alt_final_raw_series[0] if len(pred_alt_final_raw_series) else 0.0)),
                enabled=smooth_enabled,
                mode=smooth_mode,
                steps=smooth_steps,
                blend_betas=list(smooth_betas) if isinstance(smooth_betas, (list, tuple)) else [0.5, 0.3, 0.1],
                slope_cap=smooth_cap,
            )
            right_smooth_cfg = (smoothing_cfg or {}).get("right_edge_smoothing", {})
            right_enabled = bool(right_smooth_cfg.get("use_right_edge_smoothing", False))
            right_mode = str(right_smooth_cfg.get("right_edge_smoothing_mode", "right_blend"))
            right_steps = int(right_smooth_cfg.get("right_edge_smoothing_steps", 2))
            right_betas = right_smooth_cfg.get("right_blend_betas", [0.2, 0.5])
            right_band = float(right_smooth_cfg.get("right_local_band_ft", 200.0))
            policy_post_alt, right_smooth_delta = _apply_right_edge_smoothing(
                alt_final=policy_post_alt,
                right_boundary_alt=float(f.get("right_boundary_alt", policy_post_alt[-1] if len(policy_post_alt) else 0.0)),
                enabled=right_enabled,
                mode=right_mode,
                steps=right_steps,
                blend_betas=list(right_betas) if isinstance(right_betas, (list, tuple)) else [0.2, 0.5],
                right_local_band=right_band,
            )
            cond_fuse_cfg = (smoothing_cfg or {}).get("conditional_rightstep2_fuse", {})
            cond_enabled = bool(cond_fuse_cfg.get("use_conditional_rightstep2_fuse", False))
            cond_bucket = str(cond_fuse_cfg.get("conditional_target_bucket", "medium"))
            cond_pattern = str(cond_fuse_cfg.get("conditional_target_pattern", "two_anchor"))
            cond_tau_jump = float(cond_fuse_cfg.get("tau_jump", 200.0))
            cond_mode = str(cond_fuse_cfg.get("fuse_mode", "local_interp"))
            cond_lambda = float(cond_fuse_cfg.get("fuse_lambda", 0.5))
            cond_use_second = bool(cond_fuse_cfg.get("use_second_condition", False))
            cond_tau_curve = float(cond_fuse_cfg.get("tau_curve", 200.0))
            cond_band = float(cond_fuse_cfg.get("right_local_band_ft", 200.0))
            policy_post_alt, cond_fuse_delta, cond_triggered, cond_jump_abs, cond_curve_abs = _apply_conditional_rightstep2_fuse(
                policy_post_alt,
                segment_bucket=str(_segment_bucket3(float(f["fill_minutes"]))),
                anchor_pattern=str(_anchor_pattern(str(f["left_block_type"]), str(f["right_block_type"]))),
                right_boundary_alt=float(f.get("right_boundary_alt", policy_post_alt[-1] if len(policy_post_alt) else 0.0)),
                enabled=cond_enabled,
                target_bucket=cond_bucket,
                target_pattern=cond_pattern,
                tau_jump=cond_tau_jump,
                mode=cond_mode,
                fuse_lambda=cond_lambda,
                use_second_condition=cond_use_second,
                tau_curve=cond_tau_curve,
                right_local_band=cond_band,
            )
            final_alt = policy_post_alt.copy()
            source = str(replay_alt_series_source or "policy_post_alt").lower()
            if source in {"pred_pos", "pred_alt", "model_final"}:
                quality_alt = on_alt.copy()
                quality_series_name = "pred_alt(model_final)"
            elif source in {"pred_pos_main", "pred_alt_main", "main"}:
                quality_alt = pred_alt_main_series.copy()
                quality_series_name = "pred_alt_main_series"
            elif source in {"pred_alt_final_series", "final_merge", "pre_policy"}:
                quality_alt = pred_alt_final_raw_series.copy()
                quality_series_name = "pred_alt_final_raw_series"
            else:
                quality_alt = final_alt.copy()
                quality_series_name = "policy_post_alt"

            seg_len = float(f["fill_minutes"])
            seg_bucket = _segment_bucket3(seg_len)
            anchor_pat = _anchor_pattern(str(f["left_block_type"]), str(f["right_block_type"]))
            risk_flag = int((seg_bucket == "short") or (anchor_pat != "two_anchor"))

            left_b = float(f.get("left_boundary_alt", final_alt[0] if len(final_alt) else 0.0))
            right_b = float(f.get("right_boundary_alt", final_alt[-1] if len(final_alt) else 0.0))
            ref_lower = min(left_b, right_b)
            ref_upper = max(left_b, right_b)
            q = _quality_flags_from_altitude(
                alt=quality_alt,
                left_boundary_alt=left_b,
                right_boundary_alt=right_b,
            )
            edge_pos = _edge_spike_position_stats(
                alt=quality_alt,
                left_boundary_alt=left_b,
                right_boundary_alt=right_b,
                jump_thresh=300.0,
            )
            o = _overshoot_position_and_cause(
                final_alt=quality_alt,
                baseline_alt=on_alt_main,
                left_boundary_alt=left_b,
                right_boundary_alt=right_b,
                overshoot_thresh=200.0,
            )
            st_base = _series_overflow_stats(on_alt_main, ref_lower, ref_upper, thresh=200.0)
            st_main = _series_overflow_stats(pred_alt_main_series, ref_lower, ref_upper, thresh=200.0)
            st_final = _series_overflow_stats(pred_alt_final_series, ref_lower, ref_upper, thresh=200.0)
            st_policy = _series_overflow_stats(policy_post_alt, ref_lower, ref_upper, thresh=200.0)

            # Determine first trigger stage along pipeline.
            candidates = []
            if st_main["flag"] and st_main["trigger_step"] >= 0:
                candidates.append(("main", int(st_main["trigger_step"])))
            if st_final["flag"] and st_final["trigger_step"] >= 0:
                candidates.append(("final", int(st_final["trigger_step"])))
            if st_policy["flag"] and st_policy["trigger_step"] >= 0:
                candidates.append(("policy_post", int(st_policy["trigger_step"])))
            if candidates:
                candidates = sorted(candidates, key=lambda x: x[1])
                trigger_stage, trigger_step = candidates[0]
            else:
                trigger_stage, trigger_step = "none", -1

            if st_main["flag"]:
                ref_consistency_cause = "main_driven"
            elif (not st_main["flag"]) and st_final["flag"]:
                ref_consistency_cause = "final_merge_driven"
            elif (not st_main["flag"]) and (not st_final["flag"]) and st_policy["flag"]:
                ref_consistency_cause = "policy_post_driven"
            elif st_base["flag"] and (not st_main["flag"]) and (not st_final["flag"]) and (not st_policy["flag"]):
                ref_consistency_cause = "baseline_ref_mismatch"
            else:
                ref_consistency_cause = "unclear"

            anchor_interp = np.linspace(left_b, right_b, num=max(1, len(final_alt)), dtype=float)
            baseline_mae = float(np.mean(np.abs(on_alt_main - anchor_interp))) if len(on_alt_main) else float("nan")
            final_mae = float(np.mean(np.abs(final_alt - anchor_interp))) if len(final_alt) else float("nan")
            rows.append(
                {
                    "segment_id": fill_sid,
                    "sample_id": s.sample_id,
                    "fill_id": str(f["fill_id"]),
                    "task_type": task_type,
                    "flight_id": sample_to_flight.get(s.sample_id, ""),
                    "segment_len": seg_len,
                    "segment_bucket": seg_bucket,
                    "anchor_pattern": anchor_pat,
                    "risk_flag": risk_flag,
                    "risk_level": risk_level_row,
                    "matched_risk_rule": matched_rule_row,
                    "risk_flag_teacher": risk_flag_teacher_row,
                    "teacher_scale": teacher_scale_row,
                    "residual_rmax_ft": residual_rmax_row,
                    "policy_mode": "disabled" if meta is None else str(meta.segment_type),
                    "policy_scale": 1.0 if meta is None else float(meta.residual_scale),
                    "left_block_type": str(f["left_block_type"]),
                    "right_block_type": str(f["right_block_type"]),
                    "pred_alt_main_mean": float(np.mean(on_alt_main)),
                    "pred_alt_main_std": float(np.std(on_alt_main)),
                    "pred_alt_main_min": float(np.min(on_alt_main)),
                    "pred_alt_main_max": float(np.max(on_alt_main)),
                    "pred_alt_main_raw_series_mean": float(np.mean(pred_alt_main_series_raw)),
                    "pred_alt_main_raw_series_std": float(np.std(pred_alt_main_series_raw)),
                    "pred_alt_main_raw_series_min": float(np.min(pred_alt_main_series_raw)),
                    "pred_alt_main_raw_series_max": float(np.max(pred_alt_main_series_raw)),
                    "pred_alt_main_series_mean": float(np.mean(pred_alt_main_series)),
                    "pred_alt_main_series_std": float(np.std(pred_alt_main_series)),
                    "pred_alt_main_series_min": float(np.min(pred_alt_main_series)),
                    "pred_alt_main_series_max": float(np.max(pred_alt_main_series)),
                    "pred_alt_final_series_mean": float(np.mean(pred_alt_final_series)),
                    "pred_alt_final_series_std": float(np.std(pred_alt_final_series)),
                    "pred_alt_final_series_min": float(np.min(pred_alt_final_series)),
                    "pred_alt_final_series_max": float(np.max(pred_alt_final_series)),
                    "policy_post_alt_mean": float(np.mean(policy_post_alt)),
                    "policy_post_alt_std": float(np.std(policy_post_alt)),
                    "policy_post_alt_min": float(np.min(policy_post_alt)),
                    "policy_post_alt_max": float(np.max(policy_post_alt)),
                    "replay_alt_series_source": quality_series_name,
                    "alt_baseline_left_1": float(on_alt_main[0]) if len(on_alt_main) >= 1 else float("nan"),
                    "alt_baseline_left_2": float(on_alt_main[1]) if len(on_alt_main) >= 2 else float("nan"),
                    "alt_baseline_right_1": float(on_alt_main[-1]) if len(on_alt_main) >= 1 else float("nan"),
                    "alt_baseline_right_2": float(on_alt_main[-2]) if len(on_alt_main) >= 2 else float("nan"),
                    "pred_alt_main_left_1": float(pred_alt_main_series[0]) if len(pred_alt_main_series) >= 1 else float("nan"),
                    "pred_alt_main_left_2": float(pred_alt_main_series[1]) if len(pred_alt_main_series) >= 2 else float("nan"),
                    "pred_alt_main_right_1": float(pred_alt_main_series[-1]) if len(pred_alt_main_series) >= 1 else float("nan"),
                    "pred_alt_main_right_2": float(pred_alt_main_series[-2]) if len(pred_alt_main_series) >= 2 else float("nan"),
                    "pred_alt_main_raw_left_1": float(pred_alt_main_series_raw[0]) if len(pred_alt_main_series_raw) >= 1 else float("nan"),
                    "pred_alt_main_raw_left_2": float(pred_alt_main_series_raw[1]) if len(pred_alt_main_series_raw) >= 2 else float("nan"),
                    "pred_alt_final_left_1": float(pred_alt_final_series[0]) if len(pred_alt_final_series) >= 1 else float("nan"),
                    "pred_alt_final_left_2": float(pred_alt_final_series[1]) if len(pred_alt_final_series) >= 2 else float("nan"),
                    "pred_alt_final_right_1": float(pred_alt_final_series[-1]) if len(pred_alt_final_series) >= 1 else float("nan"),
                    "pred_alt_final_right_2": float(pred_alt_final_series[-2]) if len(pred_alt_final_series) >= 2 else float("nan"),
                    "policy_post_left_1": float(policy_post_alt[0]) if len(policy_post_alt) >= 1 else float("nan"),
                    "policy_post_left_2": float(policy_post_alt[1]) if len(policy_post_alt) >= 2 else float("nan"),
                    "policy_post_right_1": float(policy_post_alt[-1]) if len(policy_post_alt) >= 1 else float("nan"),
                    "policy_post_right_2": float(policy_post_alt[-2]) if len(policy_post_alt) >= 2 else float("nan"),
                    "overshoot_ref_left_1_min": float(ref_lower),
                    "overshoot_ref_left_1_max": float(ref_upper),
                    "overshoot_ref_left_2_min": float(ref_lower),
                    "overshoot_ref_left_2_max": float(ref_upper),
                    "overshoot_ref_right_1_min": float(ref_lower),
                    "overshoot_ref_right_1_max": float(ref_upper),
                    "overshoot_ref_right_2_min": float(ref_lower),
                    "overshoot_ref_right_2_max": float(ref_upper),
                    "overshoot_ref_global_min": float(ref_lower),
                    "overshoot_ref_global_max": float(ref_upper),
                    "main_vs_ref_left_1_delta": float(st_main["left_1_delta"]),
                    "main_vs_ref_left_2_delta": float(st_main["left_2_delta"]),
                    "final_vs_ref_left_1_delta": float(st_final["left_1_delta"]),
                    "final_vs_ref_left_2_delta": float(st_final["left_2_delta"]),
                    "baseline_vs_ref_left_1_delta": float(st_base["left_1_delta"]),
                    "baseline_vs_ref_left_2_delta": float(st_base["left_2_delta"]),
                    "policy_post_vs_ref_left_1_delta": float(st_policy["left_1_delta"]),
                    "policy_post_vs_ref_left_2_delta": float(st_policy["left_2_delta"]),
                    "projection_applied_rate_step1": float(1.0 if (len(proj_delta) >= 1 and abs(proj_delta[0]) > 1e-9) else 0.0),
                    "projection_applied_rate_step2": float(1.0 if (len(proj_delta) >= 2 and abs(proj_delta[1]) > 1e-9) else 0.0),
                    "projection_delta_step1": float(proj_delta[0]) if len(proj_delta) >= 1 else float("nan"),
                    "projection_delta_step2": float(proj_delta[1]) if len(proj_delta) >= 2 else float("nan"),
                    "projection_mode": proj_mode if proj_enabled else "disabled",
                    "smoothing_mode": smooth_mode if smooth_enabled else "disabled",
                    "smoothing_applied_rate_step1": float(1.0 if (len(smooth_delta) >= 1 and abs(smooth_delta[0]) > 1e-9) else 0.0),
                    "smoothing_applied_rate_step2": float(1.0 if (len(smooth_delta) >= 2 and abs(smooth_delta[1]) > 1e-9) else 0.0),
                    "smoothing_delta_step1": float(smooth_delta[0]) if len(smooth_delta) >= 1 else float("nan"),
                    "smoothing_delta_step2": float(smooth_delta[1]) if len(smooth_delta) >= 2 else float("nan"),
                    "right_smoothing_mode": right_mode if right_enabled else "disabled",
                    "right_smoothing_applied_rate_step2": float(1.0 if (len(right_smooth_delta) >= 2 and abs(right_smooth_delta[-2]) > 1e-9) else 0.0),
                    "right_smoothing_applied_rate_step3": float(1.0 if (len(right_smooth_delta) >= 3 and abs(right_smooth_delta[-3]) > 1e-9) else 0.0),
                    "right_smoothing_delta_step2": float(right_smooth_delta[-2]) if len(right_smooth_delta) >= 2 else float("nan"),
                    "right_smoothing_delta_step3": float(right_smooth_delta[-3]) if len(right_smooth_delta) >= 3 else float("nan"),
                    "conditional_fuse_mode": cond_mode if cond_enabled else "disabled",
                    "conditional_fuse_applied": float(1.0 if cond_triggered else 0.0),
                    "conditional_fuse_applied_rate": float(1.0 if cond_triggered else 0.0),
                    "conditional_fuse_delta_rightstep2": float(cond_fuse_delta[-2]) if len(cond_fuse_delta) >= 2 else 0.0,
                    "mean_fuse_delta_rightstep2": float(abs(cond_fuse_delta[-2])) if len(cond_fuse_delta) >= 2 else 0.0,
                    "max_fuse_delta_rightstep2": float(abs(cond_fuse_delta[-2])) if len(cond_fuse_delta) >= 2 else 0.0,
                    "conditional_fuse_jump_abs": float(cond_jump_abs) if np.isfinite(cond_jump_abs) else float("nan"),
                    "conditional_fuse_curve_abs": float(cond_curve_abs) if np.isfinite(cond_curve_abs) else float("nan"),
                    "overshoot_trigger_step": int(trigger_step),
                    "overshoot_trigger_stage": str(trigger_stage),
                    "overshoot_trigger_series": str(trigger_stage),
                    "reference_consistency_cause": str(ref_consistency_cause),
                    "pred_alt_final_mean": float(np.mean(quality_alt)),
                    "pred_alt_final_std": float(np.std(quality_alt)),
                    "pred_alt_final_min": float(np.min(quality_alt)),
                    "pred_alt_final_max": float(np.max(quality_alt)),
                    "gate_mean": float(np.nanmean(gate_arr)) if np.isfinite(gate_arr).any() else float("nan"),
                    "bounded_residual_mean_abs": float(np.nanmean(np.abs(dms_used_arr))) if np.isfinite(dms_used_arr).any() else float("nan"),
                    "bounded_residual_max_abs": float(np.nanmax(np.abs(dms_used_arr))) if np.isfinite(dms_used_arr).any() else float("nan"),
                    "delta_candidate_mean_abs": float(np.nanmean(np.abs(dms_cand_arr))) if np.isfinite(dms_cand_arr).any() else float("nan"),
                    "delta_used_mean_abs": float(np.nanmean(np.abs(dms_used_arr))) if np.isfinite(dms_used_arr).any() else float("nan"),
                    "delta_used_signed_mean": float(np.nanmean(dms_used_arr)) if np.isfinite(dms_used_arr).any() else float("nan"),
                    "left_edge_wrong_direction_ratio": (
                        float(np.nanmean(left_wrong_arr)) if np.isfinite(left_wrong_arr).any() else float("nan")
                    ),
                    "alt_baseline_mae": baseline_mae,
                    "alt_final_mae": final_mae,
                    **q,
                    **edge_pos,
                    **o,
                }
            )

            # point-level rows for overshoot case plots
            ts_arr = pd.to_datetime(seg_core["minute_ts"], utc=True).tolist()
            npt = len(ts_arr)
            d_used_plot = dms_used_arr if len(dms_used_arr) == npt else np.full((npt,), np.nan, dtype=float)
            left_wrong_plot = left_wrong_arr if len(left_wrong_arr) == npt else np.full((npt,), np.nan, dtype=float)
            for pi in range(npt):
                point_rows.append(
                    {
                        "segment_id": fill_sid,
                        "sample_id": s.sample_id,
                        "fill_id": str(f["fill_id"]),
                        "minute_ts": ts_arr[pi].isoformat(),
                        "point_idx": int(pi),
                        "point_ratio": float(pi / max(1, npt - 1)),
                        "pred_alt_final": float(quality_alt[pi]),
                        "pred_alt_main": float(on_alt_main[pi]),
                        "pred_alt_main_raw_series": float(pred_alt_main_series_raw[pi]),
                        "pred_alt_main_series": float(pred_alt_main_series[pi]),
                        "pred_alt_final_raw_series": float(pred_alt_final_raw_series[pi]),
                        "pred_alt_final_series": float(pred_alt_final_series[pi]),
                        "policy_post_alt": float(policy_post_alt[pi]),
                        "projection_delta": float(proj_delta[pi]) if len(proj_delta) == npt else float("nan"),
                        "smoothing_delta": float(smooth_delta[pi]) if len(smooth_delta) == npt else float("nan"),
                        "right_smoothing_delta": float(right_smooth_delta[pi]) if len(right_smooth_delta) == npt else float("nan"),
                        "conditional_fuse_delta": float(cond_fuse_delta[pi]) if len(cond_fuse_delta) == npt else float("nan"),
                        "delta_used": float(d_used_plot[pi]) if np.isfinite(d_used_plot[pi]) else float("nan"),
                        "left_edge_wrong_direction": (
                            float(left_wrong_plot[pi]) if np.isfinite(left_wrong_plot[pi]) else float("nan")
                        ),
                        "overshoot_ref_min": float(ref_lower),
                        "overshoot_ref_max": float(ref_upper),
                        "anchor_interp_alt": float(anchor_interp[pi]),
                        "gt_alt": float("nan"),
                        "left_boundary_alt": left_b,
                        "right_boundary_alt": right_b,
                    }
                )
    if not rows:
        empty_seg = pd.DataFrame(
            columns=[
                "sample_id",
                "flight_id",
                "minute_ts",
                "obs_mask",
                "pred_lat",
                "pred_lon",
                "pred_alt",
                "pred_alt_main",
                "pred_alt_main_series",
                "pred_alt_final_raw_series",
                "pred_alt_final_series",
                "policy_post_alt",
                "risk_level",
                "matched_risk_rule",
                "risk_flag_teacher",
                "teacher_scale",
                "residual_rmax_ft",
                "gate",
                "delta_candidate",
                "delta_used",
                "left_edge_wrong_direction",
                "conditional_fuse_delta",
            ]
        )
        return empty_seg, pd.DataFrame(
            columns=[
                "segment_id",
                "sample_id",
                "fill_id",
                "minute_ts",
                "point_idx",
                "point_ratio",
                "pred_alt_final",
                "pred_alt_main",
                "pred_alt_main_series",
                "pred_alt_final_raw_series",
                "pred_alt_final_series",
                "policy_post_alt",
                "conditional_fuse_delta",
                "delta_used",
                "left_edge_wrong_direction",
                "overshoot_ref_min",
                "overshoot_ref_max",
                "anchor_interp_alt",
                "gt_alt",
                "left_boundary_alt",
                "right_boundary_alt",
            ]
        )
    return pd.DataFrame(rows), pd.DataFrame(point_rows)


def _export_overshoot_audit_tables(
    segment_df: pd.DataFrame,
    point_df: pd.DataFrame,
    out_dir: Path,
    experiment_name: str,
) -> None:
    audit_dir = out_dir / "overshoot_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    root_dir = Path("outputs/overshoot_audit") / experiment_name
    root_dir.mkdir(parents=True, exist_ok=True)

    seg = segment_df.copy()
    if seg.empty:
        pd.DataFrame().to_csv(audit_dir / "overshoot_segment_audit.csv", index=False)
        pd.DataFrame().to_csv(root_dir / "overshoot_segment_audit.csv", index=False)
        return

    seg.to_csv(audit_dir / "overshoot_segment_audit.csv", index=False)
    seg.to_csv(root_dir / "overshoot_segment_audit.csv", index=False)
    if point_df is not None and len(point_df):
        point_df.to_csv(audit_dir / "overshoot_point_audit.csv", index=False)
        point_df.to_csv(root_dir / "overshoot_point_audit.csv", index=False)

    def _agg(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
        g = (
            df.groupby(group_cols, dropna=False, as_index=False)
            .agg(
                total=("segment_id", "count"),
                overshoot_rate=("overshoot_flag", "mean"),
                left_edge_ratio=("overshoot_left_edge_flag", "mean"),
                right_edge_ratio=("overshoot_right_edge_flag", "mean"),
                middle_ratio=("overshoot_middle_flag", "mean"),
                overshoot_vs_baseline_max_mean=("overshoot_vs_baseline_max", "mean"),
                overshoot_vs_anchor_envelope_max_mean=("overshoot_vs_anchor_envelope_max", "mean"),
                overshoot_vs_gt_range_max_mean=("overshoot_vs_gt_range_max", "mean"),
                gate_mean=("gate_mean", "mean"),
                bounded_residual_mean_abs=("bounded_residual_mean_abs", "mean"),
                left_edge_wrong_direction_ratio=("left_edge_wrong_direction_ratio", "mean"),
                baseline_driven_ratio=("cause_label", lambda s: float((s == "baseline_driven").mean())),
                residual_driven_ratio=("cause_label", lambda s: float((s == "residual_driven").mean())),
                mixed_ratio=("cause_label", lambda s: float((s == "mixed").mean())),
            )
        )
        return g

    by_rule = _agg(seg, ["matched_risk_rule"])
    by_rule.to_csv(audit_dir / "overshoot_by_matched_risk_rule.csv", index=False)
    by_rule.to_csv(root_dir / "overshoot_by_matched_risk_rule.csv", index=False)

    by_bucket_pattern = _agg(seg, ["segment_bucket", "anchor_pattern"])
    by_bucket_pattern.to_csv(audit_dir / "overshoot_by_segment_bucket_anchor_pattern.csv", index=False)
    by_bucket_pattern.to_csv(root_dir / "overshoot_by_segment_bucket_anchor_pattern.csv", index=False)

    by_risk_level = _agg(seg, ["risk_level"])
    by_risk_level.to_csv(audit_dir / "overshoot_by_risk_level.csv", index=False)
    by_risk_level.to_csv(root_dir / "overshoot_by_risk_level.csv", index=False)

    pos_rows = []
    for pos in ["left_edge", "right_edge", "middle_zone"]:
        col = {
            "left_edge": "overshoot_left_edge_flag",
            "right_edge": "overshoot_right_edge_flag",
            "middle_zone": "overshoot_middle_flag",
        }[pos]
        pos_rows.append(
            {
                "experiment_name": experiment_name,
                "position_bucket": pos,
                "overshoot_hit_count": int(seg[col].sum()),
                "overshoot_hit_rate": float(seg[col].mean()),
            }
        )
    pos_df = pd.DataFrame(pos_rows)
    pos_df.to_csv(audit_dir / "overshoot_by_position.csv", index=False)
    pos_df.to_csv(root_dir / "overshoot_by_position.csv", index=False)

    cause_df = (
        seg.groupby("cause_label", dropna=False, as_index=False)
        .agg(total=("segment_id", "count"))
        .sort_values("total", ascending=False)
    )
    cause_df["ratio"] = cause_df["total"] / max(1, len(seg))
    cause_df.to_csv(audit_dir / "overshoot_cause_summary.csv", index=False)
    cause_df.to_csv(root_dir / "overshoot_cause_summary.csv", index=False)

    # Top-N risky structures and warnings.
    top_abn = by_bucket_pattern.sort_values(["overshoot_rate", "middle_ratio", "total"], ascending=[False, False, False]).head(10)
    top_abn.to_csv(audit_dir / "overshoot_top_modes.csv", index=False)
    top_abn.to_csv(root_dir / "overshoot_top_modes.csv", index=False)

    # Case gallery (20 segments, prioritized by requested patterns + highest overshoot)
    if point_df is not None and len(point_df):
        wanted = [
            ("short", "two_anchor"),
            ("medium", "asymmetric"),
            ("medium", "sparse_context"),
            ("long", "sparse_context"),
            ("long", "asymmetric"),
        ]
        seg_rank = seg.sort_values(["overshoot_vs_anchor_envelope_max", "edge_spike_flag"], ascending=[False, False]).copy()
        chosen = []
        for b, a in wanted:
            sub = seg_rank[(seg_rank["segment_bucket"] == b) & (seg_rank["anchor_pattern"] == a)]
            for _, r in sub.head(5).iterrows():
                sid = str(r["segment_id"])
                if sid not in chosen:
                    chosen.append(sid)
        for _, r in seg_rank.iterrows():
            if len(chosen) >= 20:
                break
            sid = str(r["segment_id"])
            if sid not in chosen:
                chosen.append(sid)

        gallery_dir = audit_dir / "overshoot_case_plots"
        gallery_root = root_dir / "overshoot_case_plots"
        gallery_dir.mkdir(parents=True, exist_ok=True)
        gallery_root.mkdir(parents=True, exist_ok=True)
        for sid in chosen[:20]:
            srow = seg[seg["segment_id"].astype(str) == sid]
            prow = point_df[point_df["segment_id"].astype(str) == sid].copy()
            if srow.empty or prow.empty:
                continue
            srow = srow.iloc[0]
            prow["minute_ts"] = pd.to_datetime(prow["minute_ts"], utc=True)
            prow = prow.sort_values("minute_ts").reset_index(drop=True)
            x = np.arange(len(prow))
            fig, ax1 = plt.subplots(figsize=(10.5, 4.5))
            gt = pd.to_numeric(prow["gt_alt"], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(gt).any():
                ax1.plot(x, gt, color="k", lw=1.2, label="gt_alt")
            ax1.plot(x, prow["anchor_interp_alt"].to_numpy(dtype=float), color="#777", lw=1.0, ls="--", label="anchor_interp")
            ax1.plot(x, prow["pred_alt_main"].to_numpy(dtype=float), color="#1f77b4", lw=1.4, label="alt_baseline")
            ax1.plot(x, prow["pred_alt_final"].to_numpy(dtype=float), color="#d62728", lw=1.6, label="pred_alt_final")
            left_b = float(pd.to_numeric(prow["left_boundary_alt"], errors="coerce").iloc[0])
            right_b = float(pd.to_numeric(prow["right_boundary_alt"], errors="coerce").iloc[0])
            peak_i = int(np.argmax(np.maximum(prow["pred_alt_final"].to_numpy(dtype=float) - max(left_b, right_b), 0.0)))
            ax1.axvline(peak_i, color="#d62728", ls=":", lw=1.0)
            ax1.set_xlabel("segment minute index")
            ax1.set_ylabel("altitude")
            ax2 = ax1.twinx()
            ax2.plot(x, pd.to_numeric(prow["delta_used"], errors="coerce").to_numpy(dtype=float), color="#2ca02c", lw=1.0, alpha=0.8, label="delta_used")
            ax2.set_ylabel("delta_used")
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
            title = (
                f"{sid} | bucket={srow['segment_bucket']} | pattern={srow['anchor_pattern']} | "
                f"rule={srow['matched_risk_rule']} | risk={srow['risk_level']} | "
                f"gate={srow['gate_mean']:.3f} | t_scale={srow['teacher_scale']:.3f} | "
                f"cause={srow['cause_label']} | peak_ratio={srow['overshoot_peak_pos_ratio']:.3f}"
            )
            ax1.set_title(title, fontsize=9)
            fig.tight_layout()
            fig.savefig(gallery_dir / f"{sid}_overshoot_case.png", dpi=160)
            fig.savefig(gallery_root / f"{sid}_overshoot_case.png", dpi=160)
            plt.close(fig)

    # concise summary for the required Q/A
    left_rate = float(seg["overshoot_left_edge_flag"].mean())
    right_rate = float(seg["overshoot_right_edge_flag"].mean())
    mid_rate = float(seg["overshoot_middle_flag"].mean())
    top_rule = by_rule.sort_values("overshoot_rate", ascending=False).head(3)
    summary_lines = [
        f"experiment={experiment_name}",
        "overshoot_definition=final_alt compared against anchor envelope upper bound (max(left_boundary_alt,right_boundary_alt)); threshold=200",
        "overshoot_reference_note=overshoot_vs_gt_range_max is NaN in real ADS-C replay (no guaranteed gap-inner GT)",
        f"position_hit_rate:left_edge={left_rate:.4f},right_edge={right_rate:.4f},middle={mid_rate:.4f}",
        f"cause_ratio_baseline_driven={float((seg['cause_label']=='baseline_driven').mean()):.4f}",
        f"cause_ratio_residual_driven={float((seg['cause_label']=='residual_driven').mean()):.4f}",
        f"cause_ratio_mixed={float((seg['cause_label']=='mixed').mean()):.4f}",
        f"left_edge_wrong_direction_ratio={float(pd.to_numeric(seg.get('left_edge_wrong_direction_ratio'), errors='coerce').mean()):.4f}",
    ]
    if len(top_rule):
        summary_lines.append("top_rules_by_overshoot=" + ";".join([f"{r.matched_risk_rule}:{r.overshoot_rate:.4f}" for _, r in top_rule.iterrows()]))
    (audit_dir / "overshoot_audit_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    (root_dir / "overshoot_audit_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")


def _export_edge_spike_audit_tables(
    segment_df: pd.DataFrame,
    point_df: pd.DataFrame,
    out_dir: Path,
    experiment_name: str,
) -> None:
    audit_dir = out_dir / "edge_spike_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    root_dir = Path("outputs/edge_spike_audit") / experiment_name
    root_dir.mkdir(parents=True, exist_ok=True)

    seg = segment_df.copy()
    if seg.empty:
        pd.DataFrame().to_csv(audit_dir / "edge_spike_by_position.csv", index=False)
        pd.DataFrame().to_csv(root_dir / "edge_spike_by_position.csv", index=False)
        return

    by_pos = pd.DataFrame(
        [
            {
                "experiment": experiment_name,
                "left_step1_spike_rate": float(seg["left_step1_spike_flag"].mean()),
                "left_step2_spike_rate": float(seg["left_step2_spike_flag"].mean()),
                "right_step1_spike_rate": float(seg["right_step1_spike_flag"].mean()),
                "right_step2_spike_rate": float(seg["right_step2_spike_flag"].mean()),
            }
        ]
    )
    by_pos.to_csv(audit_dir / "edge_spike_by_position.csv", index=False)
    by_pos.to_csv(root_dir / "edge_spike_by_position.csv", index=False)

    by_bucket_pattern = (
        seg.groupby(["segment_bucket", "anchor_pattern"], as_index=False)
        .agg(total=("segment_id", "count"), edge_spike_rate=("edge_spike_flag", "mean"))
        .sort_values(["edge_spike_rate", "total"], ascending=[False, False])
    )
    by_bucket_pattern.to_csv(audit_dir / "edge_spike_by_segment_bucket_anchor_pattern.csv", index=False)
    by_bucket_pattern.to_csv(root_dir / "edge_spike_by_segment_bucket_anchor_pattern.csv", index=False)

    by_rule = (
        seg.groupby(["matched_risk_rule"], as_index=False)
        .agg(total=("segment_id", "count"), edge_spike_rate=("edge_spike_flag", "mean"))
        .sort_values(["edge_spike_rate", "total"], ascending=[False, False])
    )
    by_rule.to_csv(audit_dir / "edge_spike_by_matched_risk_rule.csv", index=False)
    by_rule.to_csv(root_dir / "edge_spike_by_matched_risk_rule.csv", index=False)

    signed_rows = []
    for side, step1_col, step2_col, step1_flag, step2_flag in [
        ("left", "left_step1_jump", "left_step2_jump", "left_step1_spike_flag", "left_step2_spike_flag"),
        ("right", "right_step1_jump", "right_step2_jump", "right_step1_spike_flag", "right_step2_spike_flag"),
    ]:
        for step_name, jump_col, flag_col in [("step1", step1_col, step1_flag), ("step2", step2_col, step2_flag)]:
            vals = pd.to_numeric(seg[jump_col], errors="coerce")
            flags = seg[flag_col].astype(bool)
            vv = vals[flags & vals.notna()]
            signed_rows.append(
                {
                    "experiment": experiment_name,
                    "edge_side": side,
                    "step": step_name,
                    "up_spike_rate": float((vv > 0).mean()) if len(vv) else 0.0,
                    "down_spike_rate": float((vv < 0).mean()) if len(vv) else 0.0,
                    "mean_signed_jump": float(vv.mean()) if len(vv) else 0.0,
                    "max_signed_jump": float(vv.max()) if len(vv) else 0.0,
                }
            )
    signed_df = pd.DataFrame(signed_rows)
    signed_df.to_csv(audit_dir / "edge_spike_signed_direction.csv", index=False)
    signed_df.to_csv(root_dir / "edge_spike_signed_direction.csv", index=False)

    smooth_stats = pd.DataFrame(
        [
            {
                "experiment": experiment_name,
                "smoothing_applied_rate_step1": float(pd.to_numeric(seg.get("smoothing_applied_rate_step1", pd.Series([0.0])), errors="coerce").fillna(0.0).mean()),
                "smoothing_applied_rate_step2": float(pd.to_numeric(seg.get("smoothing_applied_rate_step2", pd.Series([0.0])), errors="coerce").fillna(0.0).mean()),
                "mean_smoothing_delta_step1": float(pd.to_numeric(seg.get("smoothing_delta_step1", pd.Series([0.0])), errors="coerce").fillna(0.0).mean()),
                "mean_smoothing_delta_step2": float(pd.to_numeric(seg.get("smoothing_delta_step2", pd.Series([0.0])), errors="coerce").fillna(0.0).mean()),
                "right_smoothing_applied_rate_step2": float(pd.to_numeric(seg.get("right_smoothing_applied_rate_step2", pd.Series([0.0])), errors="coerce").fillna(0.0).mean()),
                "right_smoothing_applied_rate_step3": float(pd.to_numeric(seg.get("right_smoothing_applied_rate_step3", pd.Series([0.0])), errors="coerce").fillna(0.0).mean()),
                "mean_right_smoothing_delta_step2": float(pd.to_numeric(seg.get("right_smoothing_delta_step2", pd.Series([0.0])), errors="coerce").fillna(0.0).mean()),
                "mean_right_smoothing_delta_step3": float(pd.to_numeric(seg.get("right_smoothing_delta_step3", pd.Series([0.0])), errors="coerce").fillna(0.0).mean()),
                "conditional_fuse_applied_rate": float(pd.to_numeric(seg.get("conditional_fuse_applied_rate", pd.Series([0.0])), errors="coerce").fillna(0.0).mean()),
                "mean_fuse_delta_rightstep2": float(pd.to_numeric(seg.get("mean_fuse_delta_rightstep2", pd.Series([0.0])), errors="coerce").fillna(0.0).mean()),
                "max_fuse_delta_rightstep2": float(pd.to_numeric(seg.get("max_fuse_delta_rightstep2", pd.Series([0.0])), errors="coerce").fillna(0.0).max()),
            }
        ]
    )
    smooth_stats.to_csv(audit_dir / "edge_smoothing_application_stats.csv", index=False)
    smooth_stats.to_csv(root_dir / "edge_smoothing_application_stats.csv", index=False)

    if point_df is not None and len(point_df):
        p = point_df.copy()
        cand = seg[seg["right_step2_spike_flag"].astype(bool)].copy().sort_values(["segment_bucket", "anchor_pattern", "segment_len"])
        if cand.empty:
            cand = seg[seg["edge_spike_flag"].astype(bool)].copy().sort_values(["segment_bucket", "anchor_pattern", "segment_len"])
        chosen = cand["segment_id"].astype(str).tolist()[:20]
        if len(chosen):
            gallery_dir = audit_dir / "edge_spike_case_plots_20"
            gallery_root = root_dir / "edge_spike_case_plots_20"
            gallery_dir.mkdir(parents=True, exist_ok=True)
            gallery_root.mkdir(parents=True, exist_ok=True)
            for sid in chosen:
                srow = seg[seg["segment_id"].astype(str) == sid]
                prow = p[p["segment_id"].astype(str) == sid].copy()
                if srow.empty or prow.empty:
                    continue
                srow = srow.iloc[0]
                prow["minute_ts"] = pd.to_datetime(prow["minute_ts"], utc=True, errors="coerce")
                prow = prow.dropna(subset=["minute_ts"]).sort_values("minute_ts").reset_index(drop=True)
                x = np.arange(len(prow))
                fig, ax = plt.subplots(figsize=(10.5, 4.3))
                gt = pd.to_numeric(prow.get("gt_alt"), errors="coerce").to_numpy(dtype=float)
                if np.isfinite(gt).any():
                    ax.plot(x, gt, color="k", lw=1.1, label="gt_alt")
                raw = pd.to_numeric(prow.get("pred_alt_final_raw_series"), errors="coerce").to_numpy(dtype=float)
                sm = pd.to_numeric(prow.get("pred_alt_final"), errors="coerce").to_numpy(dtype=float)
                ax.plot(x, raw, color="#ff7f0e", lw=1.2, label="pred_alt_final_raw")
                ax.plot(x, sm, color="#d62728", lw=1.5, label="pred_alt_final_smoothed")
                lb = float(pd.to_numeric(prow["left_boundary_alt"], errors="coerce").iloc[0])
                rb = float(pd.to_numeric(prow["right_boundary_alt"], errors="coerce").iloc[0])
                lo, hi = min(lb, rb), max(lb, rb)
                ax.axhline(lb, color="#1f77b4", ls="--", lw=1.0, alpha=0.8, label="left_boundary")
                ax.axhline(rb, color="#2ca02c", ls="--", lw=1.0, alpha=0.8, label="right_boundary")
                ax.fill_between(x, lo, hi, color="#999", alpha=0.08, label="anchor_envelope")
                ax.axvline(0, color="#666", ls=":", lw=0.9)
                ax.axvline(1, color="#666", ls=":", lw=0.9)
                if len(x) >= 2:
                    ax.axvline(len(x) - 2, color="#9467bd", ls=":", lw=1.0, label="right_step2")
                ax.set_xlabel("segment minute index")
                ax.set_ylabel("altitude")
                ax.set_title(
                    f"{sid} | bucket={srow['segment_bucket']} | pattern={srow['anchor_pattern']} | "
                    f"rule={srow['matched_risk_rule']} | Lmode={srow.get('smoothing_mode','disabled')} | "
                    f"Rmode={srow.get('right_smoothing_mode','disabled')}",
                    fontsize=9,
                )
                ax.legend(loc="best", fontsize=8)
                fig.tight_layout()
                fig.savefig(gallery_dir / f"{sid}_edge_spike_raw_vs_smoothed.png", dpi=160)
                fig.savefig(gallery_root / f"{sid}_edge_spike_raw_vs_smoothed.png", dpi=160)
                plt.close(fig)


def _export_reference_consistency_audit(
    segment_df: pd.DataFrame,
    point_df: pd.DataFrame,
    out_dir: Path,
    experiment_name: str,
) -> None:
    audit_dir = out_dir / "reference_consistency_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    root_dir = Path("outputs/reference_consistency_audit") / experiment_name
    root_dir.mkdir(parents=True, exist_ok=True)

    if segment_df is None or segment_df.empty:
        pd.DataFrame().to_csv(audit_dir / "trajectory_alignment_audit.csv", index=False)
        pd.DataFrame().to_csv(root_dir / "trajectory_alignment_audit.csv", index=False)
        return

    seg = segment_df.copy()
    seg.to_csv(audit_dir / "trajectory_alignment_audit.csv", index=False)
    seg.to_csv(root_dir / "trajectory_alignment_audit.csv", index=False)

    target_series = (
        str(seg["replay_alt_series_source"].dropna().iloc[0])
        if ("replay_alt_series_source" in seg.columns and seg["replay_alt_series_source"].notna().any())
        else "policy_post_alt"
    )
    definition = {
        "overshoot_target_series": f"{target_series} (stored as pred_alt_final in replay segment quality flags)",
        "overshoot_reference_type": "anchor_envelope_band[min(left_boundary_alt,right_boundary_alt), max(left_boundary_alt,right_boundary_alt)]",
        "overshoot_stage": "post_policy_or_final_merge",
        "code_path_notes": "overshoot is computed in _quality_flags_from_altitude using configured replay_alt_series_source.",
    }
    (audit_dir / "overshoot_definition_summary.json").write_text(json.dumps(definition, indent=2), encoding="utf-8")
    (root_dir / "overshoot_definition_summary.json").write_text(json.dumps(definition, indent=2), encoding="utf-8")

    def _cause_ratio(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        rows = []
        for keys, g in df.groupby(cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            rec = {c: keys[i] for i, c in enumerate(cols)}
            n = len(g)
            rec.update(
                {
                    "total": int(n),
                    "main_driven_ratio": float((g["reference_consistency_cause"] == "main_driven").mean()),
                    "final_merge_driven_ratio": float((g["reference_consistency_cause"] == "final_merge_driven").mean()),
                    "policy_post_driven_ratio": float((g["reference_consistency_cause"] == "policy_post_driven").mean()),
                    "baseline_ref_mismatch_ratio": float((g["reference_consistency_cause"] == "baseline_ref_mismatch").mean()),
                    "unclear_ratio": float((g["reference_consistency_cause"] == "unclear").mean()),
                }
            )
            rows.append(rec)
        return pd.DataFrame(rows)

    by_bucket_pattern = _cause_ratio(seg, ["segment_bucket", "anchor_pattern"])
    by_bucket_pattern.to_csv(audit_dir / "reference_consistency_by_segment_bucket_anchor_pattern.csv", index=False)
    by_bucket_pattern.to_csv(root_dir / "reference_consistency_by_segment_bucket_anchor_pattern.csv", index=False)

    by_rule = _cause_ratio(seg, ["matched_risk_rule"])
    by_rule.to_csv(audit_dir / "reference_consistency_by_matched_risk_rule.csv", index=False)
    by_rule.to_csv(root_dir / "reference_consistency_by_matched_risk_rule.csv", index=False)

    by_risk = _cause_ratio(seg, ["risk_level"])
    by_risk.to_csv(audit_dir / "reference_consistency_by_risk_level.csv", index=False)
    by_risk.to_csv(root_dir / "reference_consistency_by_risk_level.csv", index=False)

    # left-edge overflow table
    rows = []
    for series_name, c1, c2 in [
        ("baseline", "baseline_vs_ref_left_1_delta", "baseline_vs_ref_left_2_delta"),
        ("main", "main_vs_ref_left_1_delta", "main_vs_ref_left_2_delta"),
        ("final", "final_vs_ref_left_1_delta", "final_vs_ref_left_2_delta"),
        ("policy_post", "policy_post_vs_ref_left_1_delta", "policy_post_vs_ref_left_2_delta"),
    ]:
        for step, col in [(1, c1), (2, c2)]:
            v = pd.to_numeric(seg.get(col), errors="coerce")
            ov = v.where(v > 0.0, 0.0)
            rows.append(
                {
                    "experiment": experiment_name,
                    "series_name": series_name,
                    "left_step": int(step),
                    "overflow_rate": float((ov > 0.0).mean()),
                    "mean_signed_overflow": float(v.mean()),
                    "max_signed_overflow": float(v.max()),
                }
            )
    left_df = pd.DataFrame(rows)
    left_df.to_csv(audit_dir / "left_edge_series_overflow.csv", index=False)
    left_df.to_csv(root_dir / "left_edge_series_overflow.csv", index=False)

    cause_dist = (
        seg.groupby("reference_consistency_cause", as_index=False)
        .agg(total=("segment_id", "count"))
        .sort_values("total", ascending=False)
    )
    cause_dist["ratio"] = cause_dist["total"] / max(1, int(len(seg)))
    cause_dist.to_csv(audit_dir / "reference_consistency_cause_distribution.csv", index=False)
    cause_dist.to_csv(root_dir / "reference_consistency_cause_distribution.csv", index=False)

    # Case plots with reference band and trigger markers.
    if point_df is not None and len(point_df):
        p = point_df.copy()
        p["minute_ts"] = pd.to_datetime(p["minute_ts"], utc=True, errors="coerce")
        p = p.dropna(subset=["minute_ts"]).reset_index(drop=True)
        top = seg.sort_values(["overshoot_vs_anchor_envelope_max", "segment_len"], ascending=[False, False]).head(20)
        gallery_dir = audit_dir / "reference_consistency_case_plots"
        gallery_root = root_dir / "reference_consistency_case_plots"
        gallery_dir.mkdir(parents=True, exist_ok=True)
        gallery_root.mkdir(parents=True, exist_ok=True)
        for _, row in top.iterrows():
            sid = str(row["segment_id"])
            g = p[p["segment_id"].astype(str) == sid].copy()
            if g.empty:
                continue
            g = g.sort_values("minute_ts").reset_index(drop=True)
            x = np.arange(len(g))
            base = pd.to_numeric(g.get("pred_alt_main"), errors="coerce").to_numpy(dtype=float)
            main = pd.to_numeric(g.get("pred_alt_main_series"), errors="coerce").to_numpy(dtype=float)
            final = pd.to_numeric(g.get("pred_alt_final_series"), errors="coerce").to_numpy(dtype=float)
            post = pd.to_numeric(g.get("policy_post_alt"), errors="coerce").to_numpy(dtype=float)
            lo = pd.to_numeric(g.get("overshoot_ref_min"), errors="coerce").to_numpy(dtype=float)
            hi = pd.to_numeric(g.get("overshoot_ref_max"), errors="coerce").to_numpy(dtype=float)
            gt = pd.to_numeric(g.get("gt_alt"), errors="coerce").to_numpy(dtype=float)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.8), sharex=True)
            ax1.fill_between(x, lo, hi, color="#cccccc", alpha=0.35, label="overshoot_ref_band")
            if np.isfinite(gt).any():
                ax1.plot(x, gt, color="k", lw=1.0, label="gt_alt")
            ax1.plot(x, base, color="#1f77b4", lw=1.2, label="alt_baseline")
            ax1.plot(x, main, color="#2ca02c", lw=1.2, label="pred_alt_main")
            ax1.plot(x, final, color="#ff7f0e", lw=1.2, label="pred_alt_final")
            ax1.plot(x, post, color="#d62728", lw=1.4, label="policy_post_alt")
            tstep = int(row.get("overshoot_trigger_step", -1))
            if tstep >= 0 and tstep < len(x):
                ax1.axvline(tstep, color="#d62728", ls=":", lw=1.0, label="overshoot_trigger_step")
            ax1.set_ylabel("Altitude")
            ax1.legend(loc="best", fontsize=8)

            d_used = pd.to_numeric(g.get("delta_used"), errors="coerce").to_numpy(dtype=float)
            ax2.plot(x, d_used, color="#9467bd", lw=1.1, label="delta_used")
            ax2.axhline(0.0, color="k", lw=0.8, alpha=0.5)
            ax2.set_ylabel("Delta")
            ax2.set_xlabel("Segment minute index")
            ax2.legend(loc="best", fontsize=8)

            task = str(row.get("task_type", "unknown"))
            title = (
                f"{sid} | bucket={row.get('segment_bucket')} | pattern={row.get('anchor_pattern')} | "
                f"rule={row.get('matched_risk_rule')} | risk={row.get('risk_level')} | "
                f"cause={row.get('reference_consistency_cause')} | "
                f"trigger_stage={row.get('overshoot_trigger_stage')} | trigger_series={row.get('overshoot_trigger_series')}"
            )
            ax1.set_title(title, fontsize=9)
            fig.tight_layout()
            fname = f"{sid}__{task}_reference_consistency.png"
            fig.savefig(gallery_dir / fname, dpi=160)
            fig.savefig(gallery_root / fname, dpi=160)
            plt.close(fig)

    # 10-question compact summary.
    q = []
    q.append(f"1) overshoot target series: {definition['overshoot_target_series']}")
    q.append(f"2) overshoot reference: {definition['overshoot_reference_type']}")
    q.append(f"3) overshoot stage: {definition['overshoot_stage']}")
    lead = (
        seg.groupby("overshoot_trigger_stage", as_index=False)
        .agg(n=("segment_id", "count"))
        .sort_values("n", ascending=False)
    )
    q.append("4) earliest trigger stage distribution: " + "; ".join([f"{r.overshoot_trigger_stage}:{int(r.n)}" for _, r in lead.iterrows()]))
    cause_top = (
        seg.groupby("reference_consistency_cause", as_index=False)
        .agg(ratio=("segment_id", lambda s: len(s) / max(1, len(seg))))
        .sort_values("ratio", ascending=False)
    )
    q.append("5) dominant cause: " + "; ".join([f"{r.reference_consistency_cause}:{float(r.ratio):.4f}" for _, r in cause_top.head(3).iterrows()]))
    q.append("6) baseline_ref_mismatch evidence: " + ("present" if float((seg['reference_consistency_cause'] == 'baseline_ref_mismatch').mean()) > 0 else "not_observed"))
    q.append("7) directional constraint limitation: if trigger stage remains main/policy while wrong-direction residual shrinks, issue is reference/series consistency.")
    q.append("8) two_anchor recommendation: verify main/final/reference coupling before further residual tuning.")
    q.append("9) main contradiction likely: reference consistency rather than residual magnitude.")
    q.append("10) next priority: inspect main/final merge and overshoot reference definition, then policy stage.")
    (audit_dir / "reference_consistency_summary.txt").write_text("\n".join(q), encoding="utf-8")
    (root_dir / "reference_consistency_summary.txt").write_text("\n".join(q), encoding="utf-8")

def _safe_group_stats(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if df.empty or group_col not in df.columns:
        return pd.DataFrame(
            columns=[
                group_col,
                "count",
                "keep_count",
                "warn_count",
                "abnormal_count",
                "keep_ratio",
                "warn_ratio",
                "abnormal_ratio",
                "overshoot_rate",
                "edge_spike_rate",
                "gate_mean",
                "bounded_residual_mean_abs",
                "bounded_residual_max_abs",
                "delta_candidate_mean_abs",
                "delta_used_mean_abs",
                "left_edge_wrong_direction_ratio",
            ]
        )
    rows: list[dict] = []
    for k, g in df.groupby(group_col, dropna=False):
        n = int(len(g))
        keep_n = int(g["keep_flag"].sum())
        warn_n = int(g["warn_flag"].sum())
        abnormal_n = int(g["abnormal_flag"].sum())
        rows.append(
            {
                group_col: k,
                "count": n,
                "keep_count": keep_n,
                "warn_count": warn_n,
                "abnormal_count": abnormal_n,
                "keep_ratio": keep_n / max(1, n),
                "warn_ratio": warn_n / max(1, n),
                "abnormal_ratio": abnormal_n / max(1, n),
                "overshoot_rate": float(g["overshoot_flag"].mean()),
                "edge_spike_rate": float(g["edge_spike_flag"].mean()),
                "gate_mean": float(pd.to_numeric(g.get("gate_mean"), errors="coerce").mean()) if "gate_mean" in g.columns else float("nan"),
                "bounded_residual_mean_abs": float(pd.to_numeric(g.get("bounded_residual_mean_abs"), errors="coerce").mean()) if "bounded_residual_mean_abs" in g.columns else float("nan"),
                "bounded_residual_max_abs": float(pd.to_numeric(g.get("bounded_residual_max_abs"), errors="coerce").mean()) if "bounded_residual_max_abs" in g.columns else float("nan"),
                "delta_candidate_mean_abs": float(pd.to_numeric(g.get("delta_candidate_mean_abs"), errors="coerce").mean()) if "delta_candidate_mean_abs" in g.columns else float("nan"),
                "delta_used_mean_abs": float(pd.to_numeric(g.get("delta_used_mean_abs"), errors="coerce").mean()) if "delta_used_mean_abs" in g.columns else float("nan"),
                "left_edge_wrong_direction_ratio": float(pd.to_numeric(g.get("left_edge_wrong_direction_ratio"), errors="coerce").mean()) if "left_edge_wrong_direction_ratio" in g.columns else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["abnormal_ratio", "warn_ratio"], ascending=[False, False])


def _stratified_pick(df: pd.DataFrame, flag_col: str, n: int, group_cols: list[str]) -> pd.DataFrame:
    if df.empty or flag_col not in df.columns:
        return pd.DataFrame(columns=df.columns)
    sub = df[df[flag_col].astype(bool)].copy()
    if sub.empty:
        return sub.head(0)
    sub["severity"] = sub["overshoot_up"].clip(lower=0.0) + 0.1 * sub["max_vertical_rate_inside"].clip(lower=0.0)
    sub = sub.sort_values("severity", ascending=False)

    chosen_idx: list[int] = []
    for _, g in sub.groupby(group_cols, dropna=False):
        chosen_idx.append(int(g.index[0]))
        if len(chosen_idx) >= n:
            break
    if len(chosen_idx) < n:
        for idx in sub.index:
            if int(idx) in chosen_idx:
                continue
            chosen_idx.append(int(idx))
            if len(chosen_idx) >= n:
                break
    out = sub.loc[chosen_idx].copy()
    return out.drop(columns=["severity"], errors="ignore").reset_index(drop=True)

def _speed_m_per_min(p1: tuple[float, float], p2: tuple[float, float], dt_min: float = 1.0) -> float:
    if dt_min <= 0:
        return 0.0
    return _haversine_m(p1[0], p1[1], p2[0], p2[1]) / dt_min


def _parse_flight_meta(path: Path) -> tuple[str, str]:
    stem = path.stem
    # Example: 2024-05-01-AEA098-1714527893
    m = re.match(r"^(\d{4}-\d{2}-\d{2})-(.+)-\d+$", stem)
    if m:
        return m.group(2), m.group(1)
    return stem, "unknown"


def _resample_minute(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    d = d.dropna(subset=["timestamp", "lat", "lon", "baroaltitude"])
    d["minute_ts"] = d["timestamp"].dt.floor("min")
    g = (
        d.groupby("minute_ts", as_index=False)[["lat", "lon", "baroaltitude"]]
        .median()
        .rename(columns={"baroaltitude": "alt"})
    )
    return g


def _build_minute_series(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start=start.floor("min"), end=end.floor("min"), freq="1min", tz="UTC")


def _build_task_frame(
    minute_all_adsb: pd.DataFrame,
    start_t: pd.Timestamp,
    end_t: pd.Timestamp,
    start_anchor: tuple[float, float, float],
    end_anchor: tuple[float, float, float],
    task_type: str,
    context_minutes: int,
    max_side_fill_minutes: int,
    left_context_max_points: int = 5,
) -> tuple[pd.DataFrame, int, int, pd.Timestamp, pd.Timestamp, int, int, int]:
    """Build frame for three-stage recovery:
    left fill (left ADS-B end -> first ADS-C anchor),
    middle recovery (first ADS-C anchor -> last ADS-C anchor),
    right fill (last ADS-C anchor -> right ADS-B start).
    """
    left_candidates = minute_all_adsb[minute_all_adsb["minute_ts"] < start_t.floor("min")].sort_values("minute_ts")
    right_candidates = minute_all_adsb[minute_all_adsb["minute_ts"] > end_t.floor("min")].sort_values("minute_ts")

    # nearest boundary points first
    left_row = left_candidates.iloc[-1] if len(left_candidates) else None
    right_row = right_candidates.iloc[0] if len(right_candidates) else None
    left_t_raw = pd.to_datetime(left_row["minute_ts"], utc=True).floor("min") if left_row is not None else pd.NaT
    right_t_raw = pd.to_datetime(right_row["minute_ts"], utc=True).floor("min") if right_row is not None else pd.NaT

    # Cap side-fill range to avoid unrealistic very long bridge segments.
    max_side = max(0, int(max_side_fill_minutes))
    left_t = start_t.floor("min")
    right_t = end_t.floor("min")
    use_left_fill = False
    use_right_fill = False
    if pd.notna(left_t_raw):
        left_gap = int((start_t.floor("min") - left_t_raw).total_seconds() // 60)
        if 0 < left_gap <= max_side:
            left_t = left_t_raw
            use_left_fill = True
    if pd.notna(right_t_raw):
        right_gap = int((right_t_raw - end_t.floor("min")).total_seconds() // 60)
        if 0 < right_gap <= max_side:
            right_t = right_t_raw
            use_right_fill = True

    if task_type == "pure_adsc":
        idx = _build_minute_series(left_t, right_t)
    elif task_type == "adsc_plus_left_adsb":
        # Extend index start to include the nearest valid ADS-B points before left ADS-C anchor.
        pre_valid_candidates = left_candidates.dropna(subset=["lat", "lon", "alt"]).copy()
        context_start = start_t.floor("min")
        if len(pre_valid_candidates):
            pre_ts = pd.to_datetime(pre_valid_candidates["minute_ts"], utc=True, errors="coerce").dropna().sort_values()
            if len(pre_ts):
                if int(left_context_max_points) > 0:
                    keep_n = int(left_context_max_points)
                    context_start = pre_ts.iloc[max(0, len(pre_ts) - keep_n)].floor("min")
                else:
                    context_start = pre_ts.iloc[0].floor("min")
        idx = _build_minute_series(context_start, right_t)
    else:
        idx = _build_minute_series(left_t - pd.Timedelta(minutes=context_minutes), right_t + pd.Timedelta(minutes=context_minutes))

    out = pd.DataFrame({"minute_ts": idx})
    adsb = minute_all_adsb.rename(columns={"lat": "adsb_lat", "lon": "adsb_lon", "alt": "adsb_alt"})
    out = out.merge(adsb, on="minute_ts", how="left")

    # Default latent trajectory seed: linear interpolation across full recover span.
    full_total = max(1, int((right_t - left_t).total_seconds() // 60))
    r = ((out["minute_ts"] - left_t).dt.total_seconds() / 60.0) / float(full_total)
    r = r.clip(lower=0.0, upper=1.0).to_numpy(dtype=float)
    # endpoints for seeding
    left_seed = (
        float(left_row["lat"]) if (left_row is not None and use_left_fill) else float(start_anchor[0]),
        float(left_row["lon"]) if (left_row is not None and use_left_fill) else float(start_anchor[1]),
        float(left_row["alt"]) if (left_row is not None and use_left_fill) else float(start_anchor[2]),
    )
    right_seed = (
        float(right_row["lat"]) if (right_row is not None and use_right_fill) else float(end_anchor[0]),
        float(right_row["lon"]) if (right_row is not None and use_right_fill) else float(end_anchor[1]),
        float(right_row["alt"]) if (right_row is not None and use_right_fill) else float(end_anchor[2]),
    )

    out["lat"] = left_seed[0] + r * (right_seed[0] - left_seed[0])
    out["lon"] = left_seed[1] + r * (right_seed[1] - left_seed[1])
    out["alt"] = left_seed[2] + r * (right_seed[2] - left_seed[2])

    # Use ADS-B known values where available outside middle ADS-C anchor interval.
    middle_core = (out["minute_ts"] >= start_t.floor("min")) & (out["minute_ts"] <= end_t.floor("min"))
    outside_middle = ~middle_core
    out.loc[outside_middle & out["adsb_lat"].notna(), "lat"] = out.loc[outside_middle & out["adsb_lat"].notna(), "adsb_lat"]
    out.loc[outside_middle & out["adsb_lon"].notna(), "lon"] = out.loc[outside_middle & out["adsb_lon"].notna(), "adsb_lon"]
    out.loc[outside_middle & out["adsb_alt"].notna(), "alt"] = out.loc[outside_middle & out["adsb_alt"].notna(), "adsb_alt"]

    out["obs_mask"] = 0.0
    out["obs_lat"] = 0.0
    out["obs_lon"] = 0.0
    out["obs_alt"] = 0.0

    # Real ADS-C hard anchors (middle segment boundaries).
    s_idx = out.index[out["minute_ts"] == start_t.floor("min")]
    e_idx = out.index[out["minute_ts"] == end_t.floor("min")]
    if len(s_idx) == 0 or len(e_idx) == 0:
        return pd.DataFrame(), 0, 0, pd.NaT, pd.NaT, 0, 0, 0
    s_i, e_i = int(s_idx[0]), int(e_idx[0])
    for i, (la, lo, al) in [(s_i, start_anchor), (e_i, end_anchor)]:
        out.at[i, "obs_mask"] = 1.0
        out.at[i, "obs_lat"] = float(la)
        out.at[i, "obs_lon"] = float(lo)
        out.at[i, "obs_alt"] = float(al)

    # Left/right known ADS-B boundary points as anchors for side fill (optional).
    if use_left_fill:
        l_idx = out.index[out["minute_ts"] == left_t]
        if len(l_idx):
            li = int(l_idx[0])
            out.at[li, "obs_mask"] = 1.0
            out.at[li, "obs_lat"] = float(left_row["lat"])
            out.at[li, "obs_lon"] = float(left_row["lon"])
            out.at[li, "obs_alt"] = float(left_row["alt"])
    if use_right_fill:
        r_idx = out.index[out["minute_ts"] == right_t]
        if len(r_idx):
            ri = int(r_idx[0])
            out.at[ri, "obs_mask"] = 1.0
            out.at[ri, "obs_lat"] = float(right_row["lat"])
            out.at[ri, "obs_lon"] = float(right_row["lon"])
            out.at[ri, "obs_alt"] = float(right_row["alt"])

    # Experiment B: use only pre/post-gap ADS-B context (strictly outside gap).
    pre_count = 0
    post_count = 0
    if task_type == "adsc_plus_local_adsb":
        pre_zone = (out["minute_ts"] >= left_t - pd.Timedelta(minutes=context_minutes)) & (out["minute_ts"] < left_t)
        post_zone = (out["minute_ts"] > right_t) & (out["minute_ts"] <= right_t + pd.Timedelta(minutes=context_minutes))
        pre_valid = pre_zone & out["adsb_lat"].notna() & out["adsb_lon"].notna() & out["adsb_alt"].notna()
        post_valid = post_zone & out["adsb_lat"].notna() & out["adsb_lon"].notna() & out["adsb_alt"].notna()
        pre_count = int(pre_valid.sum())
        post_count = int(post_valid.sum())
        for m in [pre_valid, post_valid]:
            out.loc[m, "obs_mask"] = 1.0
            out.loc[m, "obs_lat"] = out.loc[m, "adsb_lat"]
            out.loc[m, "obs_lon"] = out.loc[m, "adsb_lon"]
            out.loc[m, "obs_alt"] = out.loc[m, "adsb_alt"]
    elif task_type == "adsc_plus_left_adsb":
        # Left-context: nearest valid ADS-B points before the left ADS-C anchor (no time-window limit).
        pre_valid = (out["minute_ts"] < start_t) & out["adsb_lat"].notna() & out["adsb_lon"].notna() & out["adsb_alt"].notna()
        pre_idx = out.index[pre_valid].tolist()
        if int(left_context_max_points) > 0 and len(pre_idx) > int(left_context_max_points):
            pre_idx = pre_idx[-int(left_context_max_points):]
        pre_sel = np.zeros(len(out), dtype=bool)
        pre_sel[pre_idx] = True
        pre_mask = pd.Series(pre_sel, index=out.index)
        pre_count = int(pre_mask.sum())
        post_count = 0
        out.loc[pre_mask, "obs_mask"] = 1.0
        out.loc[pre_mask, "obs_lat"] = out.loc[pre_mask, "adsb_lat"]
        out.loc[pre_mask, "obs_lon"] = out.loc[pre_mask, "adsb_lon"]
        out.loc[pre_mask, "obs_alt"] = out.loc[pre_mask, "adsb_alt"]
    else:
        pre_zone = (out["minute_ts"] >= left_t - pd.Timedelta(minutes=context_minutes)) & (out["minute_ts"] < left_t)
        post_zone = (out["minute_ts"] > right_t) & (out["minute_ts"] <= right_t + pd.Timedelta(minutes=context_minutes))
        pre_count = int((pre_zone & out["adsb_lat"].notna()).sum())
        post_count = int((post_zone & out["adsb_lat"].notna()).sum())

    if task_type == "adsc_plus_left_adsb":
        # Coverage for left-context branch should reflect selected nearest-point availability.
        pre_valid_all = (out["minute_ts"] < start_t) & out["adsb_lat"].notna() & out["adsb_lon"].notna() & out["adsb_alt"].notna()
        pre_count = int(pre_valid_all.sum())
        if int(left_context_max_points) > 0:
            pre_count = int(min(pre_count, int(left_context_max_points)))
        post_count = 0

    # Strict no ADS-B inside middle ADS-C interval as model input (except ADS-C anchors).
    in_middle_inner = (out["minute_ts"] > start_t.floor("min")) & (out["minute_ts"] < end_t.floor("min"))
    out.loc[in_middle_inner, ["obs_mask", "obs_lat", "obs_lon", "obs_alt"]] = [0.0, 0.0, 0.0, 0.0]

    # Derive speed/heading from seeded trajectory for feature builder.
    lat = out["lat"].to_numpy(dtype=float)
    lon = out["lon"].to_numpy(dtype=float)
    alt = out["alt"].to_numpy(dtype=float)
    speed = np.zeros(len(out), dtype=float)
    heading = np.zeros(len(out), dtype=float)
    for i in range(1, len(out)):
        speed[i] = _speed_m_per_min((lat[i - 1], lon[i - 1]), (lat[i], lon[i]), dt_min=1.0)
        heading[i] = _bearing_deg(lat[i - 1], lon[i - 1], lat[i], lon[i])
    if len(out) > 1:
        heading[0] = heading[1]
    out["speed"] = speed
    out["heading"] = heading

    left_fill_minutes = int((start_t.floor("min") - left_t).total_seconds() // 60)
    middle_gap_minutes = int((end_t.floor("min") - start_t.floor("min")).total_seconds() // 60)
    right_fill_minutes = int((right_t - end_t.floor("min")).total_seconds() // 60)
    return out, pre_count, post_count, left_t, right_t, left_fill_minutes, middle_gap_minutes, right_fill_minutes


def _build_fill_frame(
    minute_all_adsb: pd.DataFrame,
    fill_start: pd.Timestamp,
    fill_end: pd.Timestamp,
    left_state: dict,
    right_state: dict,
    task_type: str,
    context_minutes: int,
    left_context_max_points: int = 5,
) -> tuple[pd.DataFrame, int, int]:
    if fill_end <= fill_start:
        return pd.DataFrame(), 0, 0

    if task_type == "adsc_plus_local_adsb":
        idx = _build_minute_series(fill_start - pd.Timedelta(minutes=context_minutes), fill_end + pd.Timedelta(minutes=context_minutes))
    elif task_type == "adsc_plus_left_adsb":
        # Include nearest valid ADS-B points before fill start (no fixed time-window).
        pre_candidates = minute_all_adsb[minute_all_adsb["minute_ts"] < fill_start.floor("min")].dropna(subset=["lat", "lon", "alt"]).copy()
        context_start = fill_start.floor("min")
        if len(pre_candidates):
            pre_ts = pd.to_datetime(pre_candidates["minute_ts"], utc=True, errors="coerce").dropna().sort_values()
            if len(pre_ts):
                if int(left_context_max_points) > 0:
                    keep_n = int(left_context_max_points)
                    context_start = pre_ts.iloc[max(0, len(pre_ts) - keep_n)].floor("min")
                else:
                    context_start = pre_ts.iloc[0].floor("min")
        idx = _build_minute_series(context_start, fill_end)
    else:
        idx = _build_minute_series(fill_start, fill_end)

    out = pd.DataFrame({"minute_ts": idx})
    adsb = minute_all_adsb.rename(columns={"lat": "adsb_lat", "lon": "adsb_lon", "alt": "adsb_alt"})
    out = out.merge(adsb, on="minute_ts", how="left")

    total = max(1, int((fill_end.floor("min") - fill_start.floor("min")).total_seconds() // 60))
    r = ((out["minute_ts"] - fill_start.floor("min")).dt.total_seconds() / 60.0) / float(total)
    r = r.clip(lower=0.0, upper=1.0).to_numpy(dtype=float)

    left_seed = (float(left_state["lat"]), float(left_state["lon"]), float(left_state["alt"]))
    right_seed = (float(right_state["lat"]), float(right_state["lon"]), float(right_state["alt"]))
    out["lat"] = left_seed[0] + r * (right_seed[0] - left_seed[0])
    out["lon"] = left_seed[1] + r * (right_seed[1] - left_seed[1])
    out["alt"] = left_seed[2] + r * (right_seed[2] - left_seed[2])

    out["obs_mask"] = 0.0
    out["obs_lat"] = 0.0
    out["obs_lon"] = 0.0
    out["obs_alt"] = 0.0

    s_idx = out.index[out["minute_ts"] == fill_start.floor("min")]
    e_idx = out.index[out["minute_ts"] == fill_end.floor("min")]
    if len(s_idx) == 0 or len(e_idx) == 0:
        return pd.DataFrame(), 0, 0
    s_i, e_i = int(s_idx[0]), int(e_idx[0])
    out.at[s_i, "obs_mask"] = 1.0
    out.at[s_i, "obs_lat"] = left_seed[0]
    out.at[s_i, "obs_lon"] = left_seed[1]
    out.at[s_i, "obs_alt"] = left_seed[2]
    out.at[e_i, "obs_mask"] = 1.0
    out.at[e_i, "obs_lat"] = right_seed[0]
    out.at[e_i, "obs_lon"] = right_seed[1]
    out.at[e_i, "obs_alt"] = right_seed[2]

    pre_count = 0
    post_count = 0
    if task_type == "adsc_plus_local_adsb" and context_minutes > 0:
        pre_zone = (out["minute_ts"] >= fill_start - pd.Timedelta(minutes=context_minutes)) & (out["minute_ts"] < fill_start)
        post_zone = (out["minute_ts"] > fill_end) & (out["minute_ts"] <= fill_end + pd.Timedelta(minutes=context_minutes))
        pre_valid = pre_zone & out["adsb_lat"].notna() & out["adsb_lon"].notna() & out["adsb_alt"].notna()
        post_valid = post_zone & out["adsb_lat"].notna() & out["adsb_lon"].notna() & out["adsb_alt"].notna()
        pre_count = int(pre_valid.sum())
        post_count = int(post_valid.sum())
        for m in [pre_valid, post_valid]:
            out.loc[m, "obs_mask"] = 1.0
            out.loc[m, "obs_lat"] = out.loc[m, "adsb_lat"]
            out.loc[m, "obs_lon"] = out.loc[m, "adsb_lon"]
            out.loc[m, "obs_alt"] = out.loc[m, "adsb_alt"]
    elif task_type == "adsc_plus_left_adsb":
        # Left-context: nearest valid ADS-B points before fill start (no time-window limit).
        pre_valid = (out["minute_ts"] < fill_start) & out["adsb_lat"].notna() & out["adsb_lon"].notna() & out["adsb_alt"].notna()
        pre_idx = out.index[pre_valid].tolist()
        if int(left_context_max_points) > 0 and len(pre_idx) > int(left_context_max_points):
            pre_idx = pre_idx[-int(left_context_max_points):]
        pre_sel = np.zeros(len(out), dtype=bool)
        pre_sel[pre_idx] = True
        pre_mask = pd.Series(pre_sel, index=out.index)
        pre_count = int(pre_mask.sum())
        post_count = 0
        out.loc[pre_mask, "obs_mask"] = 1.0
        out.loc[pre_mask, "obs_lat"] = out.loc[pre_mask, "adsb_lat"]
        out.loc[pre_mask, "obs_lon"] = out.loc[pre_mask, "adsb_lon"]
        out.loc[pre_mask, "obs_alt"] = out.loc[pre_mask, "adsb_alt"]

    in_fill = (out["minute_ts"] > fill_start.floor("min")) & (out["minute_ts"] < fill_end.floor("min"))
    out.loc[in_fill, ["obs_mask", "obs_lat", "obs_lon", "obs_alt"]] = [0.0, 0.0, 0.0, 0.0]

    lat = out["lat"].to_numpy(dtype=float)
    lon = out["lon"].to_numpy(dtype=float)
    alt = out["alt"].to_numpy(dtype=float)
    speed = np.zeros(len(out), dtype=float)
    heading = np.zeros(len(out), dtype=float)
    for i in range(1, len(out)):
        speed[i] = _speed_m_per_min((lat[i - 1], lon[i - 1]), (lat[i], lon[i]), dt_min=1.0)
        heading[i] = _bearing_deg(lat[i - 1], lon[i - 1], lat[i], lon[i])
    if len(out) > 1:
        heading[0] = heading[1]
    out["speed"] = speed
    out["heading"] = heading

    return out, pre_count, post_count


def _predict_on_frame(cfg: dict, checkpoint: Path, frame: pd.DataFrame, pred_key: str = "pred_pos") -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "flight_id",
                "minute_ts",
                "obs_mask",
                "pred_lat",
                "pred_lon",
                "pred_alt",
                "pred_alt_main",
                "risk_level",
                "matched_risk_rule",
                "risk_flag_teacher",
                "teacher_scale",
                "residual_rmax_ft",
                "gate",
                "delta_candidate",
                "delta_used",
            ]
        )

    run_dir = Path(cfg["outputs"]["run_dir"])
    scaler_path = run_dir / "feature_standardizer.json"
    scaler_stats = load_standardizer(scaler_path)
    if scaler_stats is None:
        raise RuntimeError(f"Missing scaler: {scaler_path}")

    f = frame.copy()
    # Ensure core exogenous structure fields exist before altitude governance.
    fb = FeatureBuilder()
    built = []
    sid_col = cfg["data"]["sample_id_col"]
    for _, g in f.groupby(sid_col, sort=False):
        built.append(fb.build(g))
    f = pd.concat(built, ignore_index=True) if built else f
    f = add_anchor_alt_features(f)
    f = add_vertical_v2_features(f)
    scaler_stats = {k: v for k, v in scaler_stats.items() if k not in set(cfg["data"]["obs_cols"])}
    f = apply_standardizer(f, scaler_stats)

    infer_cols = list(
        dict.fromkeys(
            [cfg["data"]["sample_id_col"], cfg["data"]["flight_id_col"], cfg["data"]["time_col"], cfg["data"]["obs_mask_col"]]
            + cfg["data"]["target_cols"]
            + cfg["data"]["obs_cols"]
            + ["dt_prev", "dt_next"]
            + cfg["data"]["exo_cols"]
            + cfg["data"].get("vertical_exo_cols", [])
            + cfg["data"]["quality_cols"]
        )
    )
    infer_frame = f[infer_cols].copy()
    validate_inference_frame(infer_frame, cfg)

    dcfg = DatasetConfig(
        sample_id_col=cfg["data"]["sample_id_col"],
        flight_id_col=cfg["data"]["flight_id_col"],
        time_col=cfg["data"]["time_col"],
        target_cols=cfg["data"]["target_cols"],
        obs_cols=cfg["data"]["obs_cols"],
        obs_mask_col=cfg["data"]["obs_mask_col"],
        exo_cols=cfg["data"]["exo_cols"],
        vertical_exo_cols=cfg["data"].get("vertical_exo_cols", []),
        quality_cols=cfg["data"]["quality_cols"],
        segment_risk_rules_path=cfg["data"].get("segment_risk_rules_path"),
    )
    ds = TrajectoryDataset(infer_frame, dcfg)
    loader = DataLoader(
        ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=trajectory_collate_fn,
    )

    device = torch.device(cfg["training"].get("device", "cpu"))
    abr_bounds = cfg["model"].get("alt_base_residual_bounds")
    if abr_bounds is None:
        bpath = run_dir / "alt_base_residual_bounds.json"
        if bpath.exists():
            with open(bpath, "r", encoding="utf-8") as f:
                abr_bounds = json.load(f).get("alt_base_residual_bounds")
    model = TrajectoryRecoveryModel(
        exo_dim=len(cfg["data"]["exo_cols"]),
        vertical_exo_dim=len(cfg["data"].get("vertical_exo_cols", [])),
        quality_dim=len(cfg["data"]["quality_cols"]),
        backbone_type=str(cfg["model"].get("backbone_type", "bilstm")),
        hidden_size=int(cfg["model"]["hidden_size"]),
        num_layers=int(cfg["model"].get("num_layers", 1)),
        dropout=float(cfg["model"].get("dropout", 0.0)),
        transformer_num_heads=int(cfg["model"].get("transformer_num_heads", 4)),
        transformer_ff_multiplier=int(cfg["model"].get("transformer_ff_multiplier", 4)),
        fusion_hidden_size=int(cfg["model"].get("fusion_hidden_size", 32)),
        fusion_use_exo_quality=bool(cfg["model"].get("fusion_use_exo_quality", False)),
        alt_bias_enabled=bool(cfg["model"].get("alt_bias_enabled", False)),
        alt_bias_hidden_size=int(cfg["model"].get("alt_bias_hidden_size", 32)),
        alt_bias_use_exo_quality=bool(cfg["model"].get("alt_bias_use_exo_quality", True)),
        vertical_projector_enabled=bool(cfg["model"].get("vertical_projector_enabled", False)),
        vertical_projector_hidden_size=int(cfg["model"].get("vertical_projector_hidden_size", 32)),
        vertical_projector_use_vertical_exo=bool(cfg["model"].get("vertical_projector_use_vertical_exo", True)),
        vertical_tune_enabled=bool(cfg["model"].get("vertical_tune_enabled", False)),
        vertical_tune_hidden_size=int(cfg["model"].get("vertical_tune_hidden_size", 16)),
        vertical_tune_temperature=float(cfg["model"].get("vertical_tune_temperature", 1.0)),
        vertical_tune_mode=str(cfg["model"].get("vertical_tune_mode", "combined")),
        model_variant=str(cfg["model"].get("model_variant", "default")),
        dms_refiner_hidden_size=int(cfg["model"].get("dms_refiner_hidden_size", 64)),
        dms_refiner_latent_dim=int(cfg["model"].get("dms_refiner_latent_dim", 32)),
        dms_refiner_num_heads=int(cfg["model"].get("dms_refiner_num_heads", 2)),
        dms_refiner_ff_multiplier=int(cfg["model"].get("dms_refiner_ff_multiplier", 2)),
        dms_refiner_dropout=float(cfg["model"].get("dms_refiner_dropout", 0.0)),
        alt_base_builder_type=str(cfg["model"].get("alt_base_builder_type", "auto")),
        alt_base_residual_hidden_size=int(cfg["model"].get("alt_base_residual_hidden_size", 64)),
        alt_base_residual_dropout=float(cfg["model"].get("alt_base_residual_dropout", 0.0)),
        alt_base_residual_bounds=abr_bounds,
        alt_base_residual_bound_enabled=bool(cfg["model"].get("alt_base_residual_bound_enabled", True)),
        alt_gate_enabled=bool(cfg["model"].get("alt_gate_enabled", False)),
        alt_gate_hidden_size=int(cfg["model"].get("alt_gate_hidden_size", 32)),
        alt_gate_mode=str(cfg["model"].get("alt_gate_mode", "learned")),
        alt_gate_fixed_value=float(cfg["model"].get("alt_gate_fixed_value", 1.0)),
        alt_anchor_hard_consistency=bool(cfg["model"].get("alt_anchor_hard_consistency", False)),
        use_left_edge_directional_constraint=bool(cfg["model"].get("use_left_edge_directional_constraint", False)),
        left_edge_direction_mode=str(cfg["model"].get("left_edge_direction_mode", "anchor_based")),
        left_edge_width=int(cfg["model"].get("left_edge_width", 2)),
        left_edge_direction_strength=float(cfg["model"].get("left_edge_direction_strength", 1.0)),
        left_edge_clip_mode=str(cfg["model"].get("left_edge_clip_mode", "hard")),
        alt_main_mode=str(cfg["model"].get("alt_main_mode", "absolute")),
        alt_anchor_reference_mode=str(cfg["model"].get("alt_anchor_reference_mode", "local_linear")),
        main_rmax_m=float(cfg["model"].get("main_rmax_m", float(cfg["model"].get("main_rmax_ft", 500.0)) * 0.3048)),
        main_rmax_min_m=float(cfg["model"].get("main_rmax_min_m", 91.44)),
        main_rmax_slope_m_per_min=float(cfg["model"].get("main_rmax_slope_m_per_min", 4.572)),
        main_rmax_max_m=float(cfg["model"].get("main_rmax_max_m", 365.76)),
        alt_residual_anchor_delta_gate_enabled=bool(cfg["model"].get("alt_residual_anchor_delta_gate_enabled", False)),
        alt_residual_anchor_delta_gate_low_m=float(cfg["model"].get("alt_residual_anchor_delta_gate_low_m", 60.0)),
        alt_residual_anchor_delta_gate_high_m=float(cfg["model"].get("alt_residual_anchor_delta_gate_high_m", 180.0)),
        alt_residual_anchor_delta_gate_min_scale=float(cfg["model"].get("alt_residual_anchor_delta_gate_min_scale", 0.0)),
        alt_residual_edge_taper_enabled=bool(cfg["model"].get("alt_residual_edge_taper_enabled", False)),
        alt_residual_edge_taper_steps=float(cfg["model"].get("alt_residual_edge_taper_steps", 3.0)),
        alt_anchor_graph_min_step_gap_min=float(cfg["model"].get("alt_anchor_graph_min_step_gap_min", 8.0)),
        alt_anchor_graph_step_center_ratio=float(cfg["model"].get("alt_anchor_graph_step_center_ratio", 0.5)),
        savca_hidden_size=int(cfg["model"].get("savca_hidden_size", 32)),
        savca_min_uniform=float(cfg["model"].get("savca_min_uniform", 0.05)),
        savca_state_eps=float(cfg["model"].get("savca_state_eps", 0.05)),
        alt_transition_hidden_size=int(cfg["model"].get("alt_transition_hidden_size", 32)),
        alt_transition_logit_rmax=float(cfg["model"].get("alt_transition_logit_rmax", 6.0)),
        alt_dms_route_mode=str(cfg["model"].get("alt_dms_route_mode", "none")),
        alt_dms_route_gap_threshold_min=float(cfg["model"].get("alt_dms_route_gap_threshold_min", 9.0)),
        alt_dms_route_low_risk_scale=float(cfg["model"].get("alt_dms_route_low_risk_scale", 0.0)),
        alt_dms_route_high_risk_scale=float(cfg["model"].get("alt_dms_route_high_risk_scale", 1.0)),
        v3_anchor_hard_consistency=bool(cfg["model"].get("v3_anchor_hard_consistency", True)),
        v3_edge_residual_damp_enabled=bool(cfg["model"].get("v3_edge_residual_damp_enabled", True)),
        v3_edge_residual_damp_strength=float(cfg["model"].get("v3_edge_residual_damp_strength", 0.7)),
        v3_edge_residual_damp_steps=int(cfg["model"].get("v3_edge_residual_damp_steps", 2)),
    ).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    target_norm_cfg = cfg["training"].get("target_norm", {})
    target_norm_stats = None
    if bool(target_norm_cfg.get("enabled", False)):
        target_norm_stats = load_target_stats(run_dir / "target_model_scaler.json")
        if target_norm_stats is None:
            raise RuntimeError("target_norm is enabled but target_model_scaler.json missing.")
    alt_target_mode = str(cfg["training"].get("alt_target_robust", {}).get("mode", "none")).lower()
    alt_target_clip = float(cfg["training"].get("alt_target_robust", {}).get("clip_value", 3000.0))

    rows = []
    with torch.no_grad():
        use_segment_teacher = bool(cfg.get("training", {}).get("risk_aware", {}).get("use_segment_teacher", True))
        use_alt_baseline_residual = bool(cfg.get("training", {}).get("risk_aware", {}).get("use_alt_baseline_residual", True))
        for batch in loader:
            obs_pos = batch["obs_pos"].to(device)
            obs_mask = batch["obs_mask"].to(device)
            seq_mask = batch["seq_mask"].to(device)
            target_dummy = torch.zeros_like(obs_pos)
            _, obs_model_raw, coord_ctx = prepare_model_coordinates(
                target_pos=target_dummy,
                obs_pos=obs_pos,
                obs_mask=obs_mask,
                seq_mask=seq_mask,
                mode=str(cfg["model"].get("coord_mode", "enu")),
                allow_target_fallback=False,
                u_relative_anchor=bool(cfg["model"].get("u_relative_anchor", False)),
                en_relative_anchor=bool(cfg["model"].get("en_relative_anchor", True)),
                en_incremental=bool(cfg["model"].get("en_incremental", False)),
            )
            obs_model = normalize_coords(obs_model_raw, target_norm_stats)
            if "left_boundary_alt" in batch and "right_boundary_alt" in batch:
                left_boundary_alt_model, right_boundary_alt_model = _boundary_alt_from_batch_meta(
                    left_boundary_alt=batch["left_boundary_alt"].to(device),
                    right_boundary_alt=batch["right_boundary_alt"].to(device),
                    u_relative_anchor=bool(cfg["model"].get("u_relative_anchor", False)),
                    target_norm_stats=target_norm_stats,
                    alt_target_transform_mode=alt_target_mode,
                    alt_target_clip_value=alt_target_clip,
                )
            else:
                left_boundary_alt_model, right_boundary_alt_model = _boundary_alt_from_model_obs(
                    obs_for_model=obs_model,
                    obs_mask=obs_mask,
                    seq_mask=seq_mask,
                )
            anchor_left_raw, anchor_right_raw = build_anchor_pair_tracks(
                obs_pos=obs_pos,
                obs_mask=obs_mask,
                seq_mask=seq_mask,
                ctx=coord_ctx,
            )
            anchor_left_model = normalize_coords(
                apply_alt_target_transform(anchor_left_raw, mode=alt_target_mode, clip_value=alt_target_clip),
                target_norm_stats,
            )
            anchor_right_model = normalize_coords(
                apply_alt_target_transform(anchor_right_raw, mode=alt_target_mode, clip_value=alt_target_clip),
                target_norm_stats,
            )
            anchor_alt = build_anchor_alt_tracks(obs_pos=obs_pos, obs_mask=obs_mask, seq_mask=seq_mask)
            out = model(
                obs_pos=obs_model,
                obs_mask=obs_mask,
                dt_prev=batch["dt_prev"].to(device),
                dt_next=batch["dt_next"].to(device),
                exo=batch["exo"].to(device),
                vertical_exo=batch["vertical_exo"].to(device) if "vertical_exo" in batch else None,
                quality=batch["quality"].to(device),
                global_quality=batch["global_quality"].to(device),
                anchor_alt=anchor_alt,
                risk_flag=batch["risk_flag"].to(device) if "risk_flag" in batch else None,
                teacher_scale=batch["teacher_scale"].to(device) if ("teacher_scale" in batch and use_segment_teacher) else None,
                risk_flag_teacher=batch["risk_flag_teacher"].to(device) if ("risk_flag_teacher" in batch and use_segment_teacher) else None,
                segment_bucket=batch["segment_bucket"].to(device) if "segment_bucket" in batch else None,
                edge_weight=batch["edge_weight"].to(device) if ("edge_weight" in batch and use_segment_teacher) else None,
                residual_rmax_m=batch["residual_rmax_m"].to(device) if ("residual_rmax_m" in batch and use_alt_baseline_residual) else None,
                residual_rmax_ft=batch["residual_rmax_ft"].to(device) if ("residual_rmax_ft" in batch and use_alt_baseline_residual) else None,
                gate_bias=batch["gate_bias"].to(device) if ("gate_bias" in batch and use_segment_teacher) else None,
                left_boundary_alt=left_boundary_alt_model,
                right_boundary_alt=right_boundary_alt_model,
                anchor_left=anchor_left_model,
                anchor_right=anchor_right_model,
                target_pos=None,
                teacher_forcing_ratio=0.0,
            )
            pred_tensor = out[pred_key] if pred_key in out else out["pred_pos"]
            pred_model = denormalize_coords(pred_tensor, target_norm_stats)
            pred_model = invert_alt_target_transform(pred_model, mode=alt_target_mode, clip_value=alt_target_clip)
            pred_latlon = restore_to_latlon(pred_model, seq_mask=seq_mask, ctx=coord_ctx).cpu().numpy()
            pred_main_latlon = None
            if "pred_pos_main" in out and out["pred_pos_main"] is not None:
                pred_main_model = denormalize_coords(out["pred_pos_main"], target_norm_stats)
                pred_main_model = invert_alt_target_transform(pred_main_model, mode=alt_target_mode, clip_value=alt_target_clip)
                pred_main_latlon = restore_to_latlon(pred_main_model, seq_mask=seq_mask, ctx=coord_ctx).cpu().numpy()
            gate_np = out["alt_gate"].detach().cpu().numpy() if out.get("alt_gate") is not None else None
            fusion_np = out["fusion_weights"].detach().cpu().numpy() if out.get("fusion_weights") is not None else None
            dms_cand_np = out["dms_alt_delta_candidate"].detach().cpu().numpy() if out.get("dms_alt_delta_candidate") is not None else None
            dms_used_np = out["dms_alt_delta"].detach().cpu().numpy() if out.get("dms_alt_delta") is not None else None
            left_wrong_np = (
                out["left_edge_wrong_direction_mask"].detach().cpu().numpy()
                if out.get("left_edge_wrong_direction_mask") is not None
                else None
            )

            lengths = batch["lengths"].tolist()
            for i, n in enumerate(lengths):
                sid = batch["sample_id"][i]
                fid = batch["flight_id"][i]
                ts = batch["times"][i]
                obs_np = batch["obs_mask"][i, :n].cpu().numpy()
                risk_level_i = str(batch["risk_level"][i]) if "risk_level" in batch else "unknown"
                matched_rule_i = str(batch["matched_risk_rule"][i]) if "matched_risk_rule" in batch else "unknown"
                risk_flag_teacher_i = float(batch["risk_flag_teacher"][i].item()) if "risk_flag_teacher" in batch else 0.0
                teacher_scale_i = float(batch["teacher_scale"][i].item()) if "teacher_scale" in batch else 1.0
                residual_rmax_i = float(batch["residual_rmax_ft"][i].item()) if "residual_rmax_ft" in batch else float("nan")
                for t in range(n):
                    rows.append(
                        {
                            "sample_id": sid,
                            "flight_id": fid,
                            "minute_ts": ts[t],
                            "obs_mask": float(obs_np[t]),
                            "pred_lat": float(pred_latlon[i, t, 0]),
                            "pred_lon": float(pred_latlon[i, t, 1]),
                            "pred_alt": float(pred_latlon[i, t, 2]),
                            "pred_alt_main": float(pred_main_latlon[i, t, 2]) if pred_main_latlon is not None else float(pred_latlon[i, t, 2]),
                            "risk_level": risk_level_i,
                            "matched_risk_rule": matched_rule_i,
                            "risk_flag_teacher": risk_flag_teacher_i,
                            "teacher_scale": teacher_scale_i,
                            "residual_rmax_ft": residual_rmax_i,
                            "fusion_w_forward": (
                                float(fusion_np[i, t, 0]) if fusion_np is not None else float("nan")
                            ),
                            "fusion_w_backward": (
                                float(fusion_np[i, t, 1]) if fusion_np is not None else float("nan")
                            ),
                            "gate": (float(gate_np[i, t]) if gate_np is not None else float("nan")),
                            "delta_candidate": (float(dms_cand_np[i, t]) if dms_cand_np is not None else float("nan")),
                            "delta_used": (float(dms_used_np[i, t]) if dms_used_np is not None else float("nan")),
                            "left_edge_wrong_direction": (
                                float(left_wrong_np[i, t]) if left_wrong_np is not None else float("nan")
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def _build_v15_segment_policy(cfg: dict) -> SegmentResidualPolicy:
    model_variant = str(cfg.get("model", {}).get("model_variant", "default")).lower()
    inf_cfg = cfg.get("inference", {}).get("segment_policy", {})
    enabled = bool(inf_cfg.get("enabled", False)) and (
        model_variant in {
            "bilstm_alt_dms_refiner_v1_1_5",
            "bilstm_alt_dms_refiner_v2",
            "bilstm_alt_dms_refiner_v2_1",
            "bilstm_alt_dms_refiner_v3",
        }
    )
    return SegmentResidualPolicy(
        enabled=enabled,
        stable_std_threshold=float(inf_cfg.get("stable_std_threshold", 120.0)),
        boundary_short_minutes=int(inf_cfg.get("boundary_short_minutes", 15)),
        policy_table=inf_cfg.get("policy_table", []),
    )


def _smoothness_metrics(lat: np.ndarray, lon: np.ndarray, alt: np.ndarray) -> tuple[float, float, float, float]:
    if len(lat) < 4:
        return float("nan"), float("nan"), float("nan"), float("nan")
    headings = []
    speeds = []
    for i in range(1, len(lat)):
        headings.append(_bearing_deg(lat[i - 1], lon[i - 1], lat[i], lon[i]))
        speeds.append(_speed_m_per_min((lat[i - 1], lon[i - 1]), (lat[i], lon[i]), 1.0))
    headings = np.asarray(headings, dtype=float)
    speeds = np.asarray(speeds, dtype=float)
    turn_rate = np.asarray([_angle_diff_deg(headings[i], headings[i - 1]) for i in range(1, len(headings))], dtype=float)
    vz = np.diff(alt)
    alt_smooth = np.diff(alt, n=2)
    return (
        float(np.nanmean(np.abs(turn_rate))),
        float(np.nanmean(np.abs(alt_smooth))),
        float(np.nanmax(np.abs(turn_rate))) if len(turn_rate) else 0.0,
        float(np.nanmax(np.abs(vz))) if len(vz) else 0.0,
    )


def _compute_boundary_metrics(pred_seg: pd.DataFrame, pre_adsb: pd.DataFrame, post_adsb: pd.DataFrame) -> dict:
    out = {
        "pre_boundary_heading_gap": np.nan,
        "post_boundary_heading_gap": np.nan,
        "pre_boundary_speed_gap": np.nan,
        "post_boundary_speed_gap": np.nan,
        "pre_boundary_vertical_rate_gap": np.nan,
        "post_boundary_vertical_rate_gap": np.nan,
        "pre_boundary_alt_slope_gap": np.nan,
        "post_boundary_alt_slope_gap": np.nan,
        "boundary_planar_continuity_score": np.nan,
        "boundary_alt_continuity_score": np.nan,
    }
    if len(pred_seg) < 4:
        return out
    p = pred_seg.sort_values("minute_ts").reset_index(drop=True)
    # Pred start/end local direction
    h_start = _bearing_deg(p.loc[0, "pred_lat"], p.loc[0, "pred_lon"], p.loc[1, "pred_lat"], p.loc[1, "pred_lon"])
    h_end = _bearing_deg(
        p.loc[len(p) - 2, "pred_lat"], p.loc[len(p) - 2, "pred_lon"], p.loc[len(p) - 1, "pred_lat"], p.loc[len(p) - 1, "pred_lon"]
    )
    s_start = _speed_m_per_min((p.loc[0, "pred_lat"], p.loc[0, "pred_lon"]), (p.loc[1, "pred_lat"], p.loc[1, "pred_lon"]))
    s_end = _speed_m_per_min(
        (p.loc[len(p) - 2, "pred_lat"], p.loc[len(p) - 2, "pred_lon"]),
        (p.loc[len(p) - 1, "pred_lat"], p.loc[len(p) - 1, "pred_lon"]),
    )
    vz_start = float(p.loc[1, "pred_alt"] - p.loc[0, "pred_alt"])
    vz_end = float(p.loc[len(p) - 1, "pred_alt"] - p.loc[len(p) - 2, "pred_alt"])

    if len(pre_adsb) >= 2:
        pre = pre_adsb.sort_values("minute_ts").reset_index(drop=True)
        h_pre = _bearing_deg(pre.loc[len(pre) - 2, "lat"], pre.loc[len(pre) - 2, "lon"], pre.loc[len(pre) - 1, "lat"], pre.loc[len(pre) - 1, "lon"])
        s_pre = _speed_m_per_min((pre.loc[len(pre) - 2, "lat"], pre.loc[len(pre) - 2, "lon"]), (pre.loc[len(pre) - 1, "lat"], pre.loc[len(pre) - 1, "lon"]))
        vz_pre = float(pre.loc[len(pre) - 1, "alt"] - pre.loc[len(pre) - 2, "alt"])
        out["pre_boundary_heading_gap"] = _angle_diff_deg(h_start, h_pre)
        out["pre_boundary_speed_gap"] = abs(s_start - s_pre)
        out["pre_boundary_vertical_rate_gap"] = abs(vz_start - vz_pre)
        out["pre_boundary_alt_slope_gap"] = abs(vz_start - vz_pre)

    if len(post_adsb) >= 2:
        post = post_adsb.sort_values("minute_ts").reset_index(drop=True)
        h_post = _bearing_deg(post.loc[0, "lat"], post.loc[0, "lon"], post.loc[1, "lat"], post.loc[1, "lon"])
        s_post = _speed_m_per_min((post.loc[0, "lat"], post.loc[0, "lon"]), (post.loc[1, "lat"], post.loc[1, "lon"]))
        vz_post = float(post.loc[1, "alt"] - post.loc[0, "alt"])
        out["post_boundary_heading_gap"] = _angle_diff_deg(h_end, h_post)
        out["post_boundary_speed_gap"] = abs(s_end - s_post)
        out["post_boundary_vertical_rate_gap"] = abs(vz_end - vz_post)
        out["post_boundary_alt_slope_gap"] = abs(vz_end - vz_post)

    heading_vals = [out["pre_boundary_heading_gap"], out["post_boundary_heading_gap"]]
    speed_vals = [out["pre_boundary_speed_gap"], out["post_boundary_speed_gap"]]
    alt_vals = [out["pre_boundary_alt_slope_gap"], out["post_boundary_alt_slope_gap"]]
    heading_mean = float(np.nanmean(heading_vals)) if np.any(np.isfinite(heading_vals)) else np.nan
    speed_mean = float(np.nanmean(speed_vals)) if np.any(np.isfinite(speed_vals)) else np.nan
    alt_mean = float(np.nanmean(alt_vals)) if np.any(np.isfinite(alt_vals)) else np.nan
    if np.isfinite(heading_mean) and np.isfinite(speed_mean):
        out["boundary_planar_continuity_score"] = float(math.exp(-(heading_mean / 30.0 + speed_mean / 150.0)))
    if np.isfinite(alt_mean):
        out["boundary_alt_continuity_score"] = float(math.exp(-(alt_mean / 300.0)))
    return out


def _build_full_recovered_track(adsb_all: pd.DataFrame, pred_seg: pd.DataFrame, gap_start: pd.Timestamp, gap_end: pd.Timestamp) -> pd.DataFrame:
    # Build a full-flight recovered curve:
    # keep ADS-B minute track outside gap, replace gap interval with recovered segment.
    base_cols = ["minute_ts", "lat", "lon", "alt"]
    adsb_track = adsb_all[base_cols].copy() if set(base_cols).issubset(adsb_all.columns) else pd.DataFrame(columns=base_cols)
    if len(adsb_track):
        adsb_track["minute_ts"] = pd.to_datetime(adsb_track["minute_ts"], utc=True)
    seg = pred_seg.copy()
    seg["minute_ts"] = pd.to_datetime(seg["minute_ts"], utc=True)
    seg = seg[(seg["minute_ts"] >= gap_start) & (seg["minute_ts"] <= gap_end)].copy()
    if len(seg):
        seg = seg.rename(columns={"pred_lat": "lat", "pred_lon": "lon", "pred_alt": "alt"})[base_cols]
    outside = adsb_track[(adsb_track["minute_ts"] < gap_start) | (adsb_track["minute_ts"] > gap_end)].copy()
    full = pd.concat([outside, seg], ignore_index=True)
    if full.empty:
        return full
    full = full.sort_values("minute_ts").drop_duplicates("minute_ts", keep="last").reset_index(drop=True)
    return full


def _plot_adsb_segments_latlon(ax, adsb_all: pd.DataFrame, gap_break_min: float, label: str) -> None:
    if adsb_all.empty:
        return
    adsb_all = adsb_all.sort_values("minute_ts").copy()
    adsb_all["dt_min"] = adsb_all["minute_ts"].diff().dt.total_seconds().div(60.0).fillna(0.0)
    adsb_all["seg_id"] = (adsb_all["dt_min"] >= gap_break_min).astype("int64").cumsum()
    first = True
    for _, seg in adsb_all.groupby("seg_id"):
        if len(seg) < 2:
            continue
        ax.plot(seg["lon"], seg["lat"], color="#7f7f7f", lw=1.0, alpha=0.7, label=label if first else None)
        first = False


def _plot_adsb_segments_alt(ax, adsb_all: pd.DataFrame, gap_break_min: float, label: str) -> None:
    if adsb_all.empty:
        return
    adsb_all = adsb_all.sort_values("minute_ts").copy()
    adsb_all["dt_min"] = adsb_all["minute_ts"].diff().dt.total_seconds().div(60.0).fillna(0.0)
    adsb_all["seg_id"] = (adsb_all["dt_min"] >= gap_break_min).astype("int64").cumsum()
    first = True
    for _, seg in adsb_all.groupby("seg_id"):
        if len(seg) < 2:
            continue
        ax.plot(seg["minute_ts"], seg["alt"], color="#7f7f7f", lw=1.0, alpha=0.7, label=label if first else None)
        first = False


def _plot_recovered_missing(
    ax,
    pred_seg: pd.DataFrame,
    adsb_all: pd.DataFrame,
    gap_break_min: float,
    value_cols: tuple[str, str],
    label: str,
    color: str,
) -> None:
    if pred_seg.empty:
        return
    d = pred_seg.copy()
    d["minute_ts"] = pd.to_datetime(d["minute_ts"], utc=True)
    adsb_ts = set()
    if len(adsb_all):
        adsb_ts = set(pd.to_datetime(adsb_all["minute_ts"], utc=True))
    miss = d[~d["minute_ts"].isin(adsb_ts)].copy()
    if miss.empty:
        return
    miss = miss.sort_values("minute_ts")
    miss["dt_min"] = miss["minute_ts"].diff().dt.total_seconds().div(60.0).fillna(0.0)
    miss["seg_id"] = (miss["dt_min"] >= gap_break_min).astype("int64").cumsum()
    first = True
    for _, seg in miss.groupby("seg_id"):
        if len(seg) < 2:
            continue
        x = seg[value_cols[0]]
        y = seg[value_cols[1]]
        ax.plot(x, y, color=color, lw=1.6, ls="--", alpha=0.9, label=label if first else None)
        first = False


def _build_known_blocks(
    adsb_all: pd.DataFrame,
    adsc_raw: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    gap_break_min: float,
) -> list[dict]:
    blocks: list[dict] = []
    if len(adsb_all):
        adsb_all = adsb_all.sort_values("minute_ts").copy()
        adsb_all = adsb_all[(adsb_all["minute_ts"] >= window_start) & (adsb_all["minute_ts"] <= window_end)]
        if len(adsb_all):
            adsb_all["dt_min"] = adsb_all["minute_ts"].diff().dt.total_seconds().div(60.0).fillna(0.0)
            adsb_all["seg_id"] = (adsb_all["dt_min"] >= gap_break_min).astype("int64").cumsum()
            for seg_id, seg in adsb_all.groupby("seg_id"):
                if seg.empty:
                    continue
                s = seg.iloc[0]
                e = seg.iloc[-1]
                blocks.append(
                    {
                        "block_type": "adsb_segment",
                        "start_time": pd.to_datetime(s["minute_ts"], utc=True),
                        "end_time": pd.to_datetime(e["minute_ts"], utc=True),
                        "start_lat": float(s["lat"]),
                        "start_lon": float(s["lon"]),
                        "start_alt": float(s["alt"]),
                        "end_lat": float(e["lat"]),
                        "end_lon": float(e["lon"]),
                        "end_alt": float(e["alt"]),
                    }
                )
    if len(adsc_raw):
        adsc_raw = adsc_raw.copy()
        adsc_raw["timestamp"] = pd.to_datetime(adsc_raw["timestamp"], utc=True)
        adsc_raw = adsc_raw[(adsc_raw["timestamp"] >= window_start) & (adsc_raw["timestamp"] <= window_end)]
        for r in adsc_raw.itertuples(index=False):
            t = pd.to_datetime(r.timestamp, utc=True)
            blocks.append(
                {
                    "block_type": "adsc_anchor",
                    "start_time": t,
                    "end_time": t,
                    "start_lat": float(r.lat),
                    "start_lon": float(r.lon),
                    "start_alt": float(r.baroaltitude),
                    "end_lat": float(r.lat),
                    "end_lon": float(r.lon),
                    "end_alt": float(r.baroaltitude),
                }
            )
    blocks.sort(key=lambda x: (x["start_time"], x["end_time"]))
    return blocks


def _build_fill_intervals(blocks: list[dict], min_gap_min: float) -> list[dict]:
    fills: list[dict] = []
    if len(blocks) < 2:
        return fills
    for i in range(len(blocks) - 1):
        a = blocks[i]
        b = blocks[i + 1]
        gap_min = (b["start_time"] - a["end_time"]).total_seconds() / 60.0
        if gap_min <= min_gap_min:
            continue
        fills.append(
            {
                "left_block_idx": i,
                "right_block_idx": i + 1,
                "left_block_type": a["block_type"],
                "right_block_type": b["block_type"],
                "fill_start_time": a["end_time"],
                "fill_end_time": b["start_time"],
                "fill_minutes": float(gap_min),
            }
        )
    return fills


def _plot_raw_full(sample_meta: ReplaySample, adsb_all: pd.DataFrame, adsc_raw: pd.DataFrame, out_dir: Path, gap_break_min: float = 10.0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    adsc = adsc_raw.copy()
    if len(adsc):
        adsc["timestamp"] = pd.to_datetime(adsc["timestamp"], utc=True)
    _plot_adsb_segments_latlon(axes[0], adsb_all, gap_break_min, "ADS-B minute agg")
    if len(adsc):
        axes[0].scatter(adsc["lon"], adsc["lat"], color="#7b3294", s=20, label="ADS-C raw")
        axes[0].plot(adsc["lon"], adsc["lat"], color="#7b3294", lw=1.2, alpha=0.8)
    axes[0].set_xlabel("Lon")
    axes[0].set_ylabel("Lat")
    axes[0].set_title("Raw Fused (Lat/Lon)")
    axes[0].legend(fontsize=8)

    _plot_adsb_segments_alt(axes[1], adsb_all, gap_break_min, "ADS-B minute agg")
    if len(adsc):
        axes[1].scatter(adsc["timestamp"], adsc["baroaltitude"], color="#7b3294", s=20, label="ADS-C raw")
        axes[1].plot(adsc["timestamp"], adsc["baroaltitude"], color="#7b3294", lw=1.2, alpha=0.8)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Altitude")
    axes[1].set_title("Raw Fused (Altitude)")
    axes[1].legend(fontsize=8)

    fig.suptitle(
        f"{sample_meta.sample_id} | flight={sample_meta.flight_id} | raw_fused",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"{sample_meta.sample_id}.png", dpi=150)
    plt.close(fig)


def _plot_raw_boundary(sample_meta: ReplaySample, adsb_all: pd.DataFrame, adsc_raw: pd.DataFrame, out_dir: Path, gap_break_min: float = 10.0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    adsc = adsc_raw.copy()
    if adsc.empty or adsb_all.empty:
        return
    adsc["timestamp"] = pd.to_datetime(adsc["timestamp"], utc=True)
    gap_start = sample_meta.anchor_start_time.floor("min")
    gap_end = sample_meta.anchor_end_time.floor("min")
    pad = pd.Timedelta(minutes=10)
    t0 = gap_start - pad
    t1 = gap_end + pad
    adsb_local = adsb_all[(adsb_all["minute_ts"] >= t0) & (adsb_all["minute_ts"] <= t1)].copy()
    adsc_local = adsc[(adsc["timestamp"] >= t0) & (adsc["timestamp"] <= t1)].copy()
    if adsb_local.empty and adsc_local.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    _plot_adsb_segments_latlon(axes[0], adsb_local, gap_break_min, "ADS-B minute agg (local)")
    if len(adsc_local):
        axes[0].scatter(adsc_local["lon"], adsc_local["lat"], color="#7b3294", s=20, label="ADS-C raw")
        axes[0].plot(adsc_local["lon"], adsc_local["lat"], color="#7b3294", lw=1.2, alpha=0.8)
    axes[0].set_xlabel("Lon")
    axes[0].set_ylabel("Lat")
    axes[0].set_title("Raw Fused Local (Lat/Lon)")
    axes[0].legend(fontsize=8)

    _plot_adsb_segments_alt(axes[1], adsb_local, gap_break_min, "ADS-B minute agg (local)")
    if len(adsc_local):
        axes[1].scatter(adsc_local["timestamp"], adsc_local["baroaltitude"], color="#7b3294", s=20, label="ADS-C raw")
        axes[1].plot(adsc_local["timestamp"], adsc_local["baroaltitude"], color="#7b3294", lw=1.2, alpha=0.8)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Altitude")
    axes[1].set_title("Raw Fused Local (Altitude)")
    axes[1].legend(fontsize=8)

    fig.suptitle(
        f"{sample_meta.sample_id} | flight={sample_meta.flight_id} | raw_fused_local",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"{sample_meta.sample_id}.png", dpi=150)
    plt.close(fig)


def _plot_full(
    sample_meta: ReplaySample,
    pred_seg: pd.DataFrame,
    adsb_all: pd.DataFrame,
    adsc_raw: pd.DataFrame,
    fills: list[dict],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    d = pred_seg.sort_values("minute_ts").reset_index(drop=True)
    d["minute_ts"] = pd.to_datetime(d["minute_ts"], utc=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    _plot_adsb_segments_latlon(axes[0], adsb_all, 10.0, "ADS-B minute agg")
    adsc = adsc_raw.copy()
    if len(adsc):
        axes[0].scatter(adsc["lon"], adsc["lat"], color="#7b3294", s=20, label="ADS-C raw")
    palette = ["#1f78b4", "#e7298a", "#33a02c", "#ff7f00", "#6a3d9a", "#a6cee3"]
    for i, f in enumerate(fills):
        seg = d[(d["minute_ts"] >= f["fill_start_time"]) & (d["minute_ts"] <= f["fill_end_time"])].copy()
        if seg.empty:
            continue
        color = palette[i % len(palette)]
        label = f"recovered {f['left_block_type']}->{f['right_block_type']} ({int(round(f['fill_minutes']))}m)"
        axes[0].plot(seg["pred_lon"], seg["pred_lat"], color=color, lw=1.8, label=label if i == 0 else None)
    axes[0].scatter(
        [sample_meta.anchor_start_lon, sample_meta.anchor_end_lon],
        [sample_meta.anchor_start_lat, sample_meta.anchor_end_lat],
        c="#6a3d9a",
        s=40,
        marker="X",
        label="ADS-C anchors",
    )
    axes[0].set_xlabel("Lon")
    axes[0].set_ylabel("Lat")
    axes[0].set_title("Full Recovery (Lat/Lon)")
    axes[0].legend(fontsize=8)

    _plot_adsb_segments_alt(axes[1], adsb_all, 10.0, "ADS-B minute agg")
    if len(adsc):
        adsc["timestamp"] = pd.to_datetime(adsc["timestamp"], utc=True)
        axes[1].scatter(adsc["timestamp"], adsc["baroaltitude"], color="#7b3294", s=20, label="ADS-C raw")
    for i, f in enumerate(fills):
        seg = d[(d["minute_ts"] >= f["fill_start_time"]) & (d["minute_ts"] <= f["fill_end_time"])].copy()
        if seg.empty:
            continue
        color = palette[i % len(palette)]
        label = f"recovered {f['left_block_type']}->{f['right_block_type']} ({int(round(f['fill_minutes']))}m)"
        axes[1].plot(seg["minute_ts"], seg["pred_alt"], color=color, lw=1.8, label=label if i == 0 else None)
    axes[1].scatter(
        [sample_meta.anchor_start_time.floor("min"), sample_meta.anchor_end_time.floor("min")],
        [sample_meta.anchor_start_alt, sample_meta.anchor_end_alt],
        c="#6a3d9a",
        s=40,
        marker="X",
        label="ADS-C anchors",
    )
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Altitude")
    axes[1].set_title("Full Recovery (Altitude)")
    axes[1].legend(fontsize=8)

    fill_minutes = [int(round(f["fill_minutes"])) for f in fills]
    fill_breakdown = "+".join(str(v) for v in fill_minutes) if fill_minutes else "0"
    fig.suptitle(
        f"{sample_meta.sample_id} | flight={sample_meta.flight_id} | task={sample_meta.task_type} | "
        f"blocks={len(fills)+1} | fills={len(fills)} | total_fill={sum(fill_minutes)}m | fill_breakdown={fill_breakdown}",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"{sample_meta.sample_id}.png", dpi=150)
    plt.close(fig)


def _plot_boundary(sample_meta: ReplaySample, pred_seg: pd.DataFrame, adsb_all: pd.DataFrame, adsc_raw: pd.DataFrame, metrics: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    d = pred_seg.sort_values("minute_ts").reset_index(drop=True)
    d["minute_ts"] = pd.to_datetime(d["minute_ts"], utc=True)
    gap_start = sample_meta.anchor_start_time.floor("min")
    gap_end = sample_meta.anchor_end_time.floor("min")
    pre_adsb = adsb_all[(adsb_all["minute_ts"] <= gap_start) & (adsb_all["minute_ts"] >= gap_start - pd.Timedelta(minutes=5))]
    post_adsb = adsb_all[(adsb_all["minute_ts"] >= gap_end) & (adsb_all["minute_ts"] <= gap_end + pd.Timedelta(minutes=5))]
    pred_pre = d[(d["minute_ts"] >= gap_start - pd.Timedelta(minutes=5)) & (d["minute_ts"] <= gap_start + pd.Timedelta(minutes=5))]
    pred_post = d[(d["minute_ts"] >= gap_end - pd.Timedelta(minutes=5)) & (d["minute_ts"] <= gap_end + pd.Timedelta(minutes=5))]
    adsc = adsc_raw.copy()
    if len(adsc):
        adsc["timestamp"] = pd.to_datetime(adsc["timestamp"], utc=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    _plot_adsb_segments_latlon(axes[0, 0], adsb_all, 10.0, "ADS-B minute agg")
    axes[0, 0].plot(pred_pre["pred_lon"], pred_pre["pred_lat"], color="#d95f02", lw=1.6, label="recovered(local)")
    if len(pre_adsb):
        axes[0, 0].plot(pre_adsb["lon"], pre_adsb["lat"], color="#1f78b4", lw=1.2, label="pre ADS-B")
    if len(adsc):
        axes[0, 0].scatter(adsc["lon"], adsc["lat"], color="#7b3294", s=18, label="ADS-C raw")
    axes[0, 0].scatter([sample_meta.anchor_start_lon], [sample_meta.anchor_start_lat], c="k", marker="X", s=35, label="start anchor")
    axes[0, 0].set_title("Start Boundary Planar")
    axes[0, 0].legend(fontsize=8)

    _plot_adsb_segments_latlon(axes[0, 1], adsb_all, 10.0, "ADS-B minute agg")
    axes[0, 1].plot(pred_post["pred_lon"], pred_post["pred_lat"], color="#d95f02", lw=1.6, label="recovered(local)")
    if len(post_adsb):
        axes[0, 1].plot(post_adsb["lon"], post_adsb["lat"], color="#33a02c", lw=1.2, label="post ADS-B")
    if len(adsc):
        axes[0, 1].scatter(adsc["lon"], adsc["lat"], color="#7b3294", s=18, label="ADS-C raw")
    axes[0, 1].scatter([sample_meta.anchor_end_lon], [sample_meta.anchor_end_lat], c="k", marker="X", s=35, label="end anchor")
    axes[0, 1].set_title("End Boundary Planar")
    axes[0, 1].legend(fontsize=8)

    _plot_adsb_segments_alt(axes[1, 0], adsb_all, 10.0, "ADS-B minute agg")
    axes[1, 0].plot(pred_pre["minute_ts"], pred_pre["pred_alt"], color="#d95f02", lw=1.6, label="recovered(local)")
    if len(pre_adsb):
        axes[1, 0].plot(pre_adsb["minute_ts"], pre_adsb["alt"], color="#1f78b4", lw=1.2, label="pre ADS-B alt")
    if len(adsc):
        axes[1, 0].scatter(adsc["timestamp"], adsc["baroaltitude"], color="#7b3294", s=18, label="ADS-C raw")
    axes[1, 0].scatter([gap_start], [sample_meta.anchor_start_alt], c="k", marker="X", s=35)
    axes[1, 0].set_title("Start Boundary Altitude")
    axes[1, 0].legend(fontsize=8)

    _plot_adsb_segments_alt(axes[1, 1], adsb_all, 10.0, "ADS-B minute agg")
    axes[1, 1].plot(pred_post["minute_ts"], pred_post["pred_alt"], color="#d95f02", lw=1.6, label="recovered(local)")
    if len(post_adsb):
        axes[1, 1].plot(post_adsb["minute_ts"], post_adsb["alt"], color="#33a02c", lw=1.2, label="post ADS-B alt")
    if len(adsc):
        axes[1, 1].scatter(adsc["timestamp"], adsc["baroaltitude"], color="#7b3294", s=18, label="ADS-C raw")
    axes[1, 1].scatter([gap_end], [sample_meta.anchor_end_alt], c="k", marker="X", s=35)
    axes[1, 1].set_title("End Boundary Altitude")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle(
        f"{sample_meta.sample_id} | {sample_meta.task_type} | "
        f"pre_hdg_gap={metrics.get('pre_boundary_heading_gap', np.nan):.2f} | "
        f"post_hdg_gap={metrics.get('post_boundary_heading_gap', np.nan):.2f} | "
        f"pre_alt_slope_gap={metrics.get('pre_boundary_alt_slope_gap', np.nan):.2f} | "
        f"post_alt_slope_gap={metrics.get('post_boundary_alt_slope_gap', np.nan):.2f}",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"{sample_meta.sample_id}.png", dpi=150)
    plt.close(fig)


def _plot_aux_compare(
    sample_meta: ReplaySample,
    pred_pure: pd.DataFrame,
    pred_plus: pd.DataFrame,
    adsb_all: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Compare before/after auxiliary information on the same sample.

    before: pure_adsc
    after:  adsc_plus_local_adsb (pre/post ADS-B context + ADS-C exogenous path)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    p0 = pred_pure.sort_values("minute_ts").copy()
    p1 = pred_plus.sort_values("minute_ts").copy()
    p0["minute_ts"] = pd.to_datetime(p0["minute_ts"], utc=True)
    p1["minute_ts"] = pd.to_datetime(p1["minute_ts"], utc=True)

    gap_start = sample_meta.anchor_start_time.floor("min")
    gap_end = sample_meta.anchor_end_time.floor("min")
    left_t = sample_meta.left_known_time.floor("min")
    right_t = sample_meta.right_known_time.floor("min")
    p0_zone = p0[(p0["minute_ts"] >= left_t) & (p0["minute_ts"] <= right_t)].copy()
    p1_zone = p1[(p1["minute_ts"] >= left_t) & (p1["minute_ts"] <= right_t)].copy()

    adsb_all = adsb_all.copy()
    if len(adsb_all):
        adsb_all["minute_ts"] = pd.to_datetime(adsb_all["minute_ts"], utc=True)
    adsb_left_known = adsb_all[(adsb_all["minute_ts"] <= left_t) & (adsb_all["minute_ts"] >= left_t - pd.Timedelta(minutes=5))]
    adsb_right_known = adsb_all[(adsb_all["minute_ts"] >= right_t) & (adsb_all["minute_ts"] <= right_t + pd.Timedelta(minutes=5))]
    adsb_ctx = adsb_all[
        ((adsb_all["minute_ts"] < gap_start) & (adsb_all["minute_ts"] >= gap_start - pd.Timedelta(minutes=5)))
        | ((adsb_all["minute_ts"] > gap_end) & (adsb_all["minute_ts"] <= gap_end + pd.Timedelta(minutes=5)))
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

    # Lat/Lon panel
    if len(adsb_left_known):
        axes[0].plot(adsb_left_known["lon"], adsb_left_known["lat"], color="#7f7f7f", lw=1.1, alpha=0.9, label="left known track")
    if len(adsb_right_known):
        axes[0].plot(adsb_right_known["lon"], adsb_right_known["lat"], color="#7f7f7f", lw=1.1, alpha=0.9, label="right known track")
    if len(adsb_ctx):
        axes[0].scatter(
            adsb_ctx["lon"],
            adsb_ctx["lat"],
            s=18,
            c="#1f78b4",
            marker="o",
            alpha=0.9,
            label="ADS-B context anchors (pre/post 5m)",
        )
    if len(p0_zone):
        axes[0].plot(p0_zone["pred_lon"], p0_zone["pred_lat"], color="#e7298a", lw=1.8, ls="--", label="before aux (pure_adsc)")
    if len(p1_zone):
        axes[0].plot(
            p1_zone["pred_lon"],
            p1_zone["pred_lat"],
            color="#d95f02",
            lw=2.0,
            label="after aux (left+middle+right recovery)",
        )
    axes[0].scatter(
        [sample_meta.anchor_start_lon, sample_meta.anchor_end_lon],
        [sample_meta.anchor_start_lat, sample_meta.anchor_end_lat],
        c=["#6a3d9a", "#6a3d9a"],
        s=45,
        marker="X",
        label="ADS-C anchors",
    )
    axes[0].set_xlabel("Lon")
    axes[0].set_ylabel("Lat")
    axes[0].set_title("Auxiliary Before/After (Lat/Lon)")
    axes[0].legend(fontsize=8)

    # Altitude panel
    if len(adsb_left_known):
        axes[1].plot(adsb_left_known["minute_ts"], adsb_left_known["alt"], color="#7f7f7f", lw=1.1, alpha=0.9, label="left known track")
    if len(adsb_right_known):
        axes[1].plot(adsb_right_known["minute_ts"], adsb_right_known["alt"], color="#7f7f7f", lw=1.1, alpha=0.9, label="right known track")
    if len(adsb_ctx):
        axes[1].scatter(
            adsb_ctx["minute_ts"],
            adsb_ctx["alt"],
            s=18,
            c="#1f78b4",
            marker="o",
            alpha=0.9,
            label="ADS-B context anchors",
        )
    if len(p0_zone):
        axes[1].plot(p0_zone["minute_ts"], p0_zone["pred_alt"], color="#e7298a", lw=1.8, ls="--", label="before aux (pure_adsc)")
    if len(p1_zone):
        axes[1].plot(
            p1_zone["minute_ts"],
            p1_zone["pred_alt"],
            color="#d95f02",
            lw=2.0,
            label="after aux (left+middle+right recovery)",
        )
    axes[1].scatter(
        [gap_start, gap_end],
        [sample_meta.anchor_start_alt, sample_meta.anchor_end_alt],
        c=["#6a3d9a", "#6a3d9a"],
        s=45,
        marker="X",
        label="ADS-C anchors",
    )
    axes[1].axvline(gap_start, color="k", ls="--", lw=0.8)
    axes[1].axvline(gap_end, color="k", ls="--", lw=0.8)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Altitude")
    axes[1].set_title("Auxiliary Before/After (Altitude)")
    axes[1].legend(fontsize=8)

    fig.suptitle(
        f"{sample_meta.sample_id} | flight={sample_meta.flight_id} | task={sample_meta.task_type} | "
        f"left_fill={sample_meta.left_fill_minutes}m | middle_gap={sample_meta.middle_gap_minutes}m | right_fill={sample_meta.right_fill_minutes}m",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"{sample_meta.sample_id}.png", dpi=150)
    plt.close(fig)


def _plot_left_context_compare(
    sample_meta: ReplaySample,
    pred_base: pd.DataFrame,
    pred_left_aug: pd.DataFrame,
    frame_left_aug: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Baseline (pure) vs left-context augmentation compare."""
    out_dir.mkdir(parents=True, exist_ok=True)
    p0 = pred_base.sort_values("minute_ts").copy()
    p1 = pred_left_aug.sort_values("minute_ts").copy()
    p0["minute_ts"] = pd.to_datetime(p0["minute_ts"], utc=True)
    p1["minute_ts"] = pd.to_datetime(p1["minute_ts"], utc=True)

    gap_start = sample_meta.anchor_start_time.floor("min")
    gap_end = sample_meta.anchor_end_time.floor("min")
    left_t = sample_meta.left_known_time.floor("min")
    right_t = sample_meta.right_known_time.floor("min")
    p0_zone = p0[(p0["minute_ts"] >= left_t) & (p0["minute_ts"] <= right_t)].copy()
    p1_zone = p1[(p1["minute_ts"] >= left_t) & (p1["minute_ts"] <= right_t)].copy()

    fr = frame_left_aug.copy()
    fr["minute_ts"] = pd.to_datetime(fr["minute_ts"], utc=True)
    left_ctx = fr[(fr["minute_ts"] < sample_meta.anchor_start_time.floor("min")) & (fr["obs_mask"] > 0.5)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    axes[0].plot(p0_zone["pred_lon"], p0_zone["pred_lat"], color="#7570b3", lw=1.8, ls="--", label="baseline pure_adsc")
    axes[0].plot(p1_zone["pred_lon"], p1_zone["pred_lat"], color="#d95f02", lw=2.0, label="left_context_5min")
    if len(left_ctx):
        axes[0].scatter(left_ctx["obs_lon"], left_ctx["obs_lat"], s=20, c="#1f78b4", label="left ADS-B context points")
    axes[0].scatter(
        [sample_meta.anchor_start_lon, sample_meta.anchor_end_lon],
        [sample_meta.anchor_start_lat, sample_meta.anchor_end_lat],
        c="#6a3d9a",
        s=45,
        marker="X",
        label="ADS-C anchors",
    )
    axes[0].set_xlabel("Lon")
    axes[0].set_ylabel("Lat")
    axes[0].set_title("Baseline vs Left-context (Lat/Lon)")
    axes[0].legend(fontsize=8)

    axes[1].plot(p0_zone["minute_ts"], p0_zone["pred_alt"], color="#7570b3", lw=1.8, ls="--", label="baseline pure_adsc")
    axes[1].plot(p1_zone["minute_ts"], p1_zone["pred_alt"], color="#d95f02", lw=2.0, label="left_context_5min")
    if len(left_ctx):
        axes[1].scatter(left_ctx["minute_ts"], left_ctx["obs_alt"], s=20, c="#1f78b4", label="left ADS-B context points")
    axes[1].scatter(
        [gap_start, gap_end],
        [sample_meta.anchor_start_alt, sample_meta.anchor_end_alt],
        c="#6a3d9a",
        s=45,
        marker="X",
        label="ADS-C anchors",
    )
    axes[1].axvline(gap_start, color="k", ls="--", lw=0.8)
    axes[1].axvline(gap_end, color="k", ls="--", lw=0.8)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Altitude")
    axes[1].set_title("Baseline vs Left-context (Altitude)")
    axes[1].legend(fontsize=8)
    fig.suptitle(
        f"{sample_meta.sample_id} | flight={sample_meta.flight_id} | left_ctx_points={int(len(left_ctx))}",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"{sample_meta.sample_id}.png", dpi=150)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Real ADS-C replay + local ADS-B context evaluation")
    p.add_argument("--config", default="configs/alt_focus/v15_20260327/train_v15_adaptive_right_only_e10_20260328_fix.yaml")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--adsc-dir", default="outputs/adsc/adsc_flight_point/2026-01-13-2110/all")
    p.add_argument("--out-dir", default="outputs/runs/real_adsc_replay_20260329")
    p.add_argument("--context-minutes", type=int, default=5)
    p.add_argument("--left-context-minutes", type=int, default=5)
    p.add_argument("--left-context-max-points", type=int, default=5)
    p.add_argument("--min-gap-minutes", type=int, default=5)
    p.add_argument("--max-gap-minutes", type=int, default=120)
    p.add_argument("--max-side-fill-minutes", type=int, default=30)
    p.add_argument("--min-adsc-anchors", type=int, default=6)
    p.add_argument("--max-samples", type=int, default=120)
    p.add_argument("--plot-count", type=int, default=40)
    return p


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.checkpoint) if args.checkpoint else (Path(cfg["outputs"]["run_dir"]) / "best.pt")
    if not ckpt.exists():
        raise RuntimeError(f"checkpoint not found: {ckpt}")

    adsc_dir = Path(args.adsc_dir)
    csv_files = sorted([p for p in adsc_dir.glob("*.csv") if p.is_file()])
    if not csv_files:
        raise RuntimeError(f"No csv in {adsc_dir}")

    audit_rows = []
    samples_a: list[ReplaySample] = []
    samples_b: list[ReplaySample] = []
    samples_c: list[ReplaySample] = []

    for fp in csv_files:
        fdf = pd.read_csv(fp)
        if "source" not in fdf.columns:
            continue
        fdf["source"] = fdf["source"].astype(str).str.lower()
        adsb_raw = fdf[fdf["source"].eq("adsb")].copy()
        adsc_raw = fdf[fdf["source"].eq("adsc")].copy()
        if len(adsc_raw) < 2:
            continue
        flight_id, flight_date = _parse_flight_meta(fp)
        adsb_min = _resample_minute(adsb_raw)
        adsc_min = _resample_minute(adsc_raw)
        if len(adsc_min) < 2:
            continue
        if len(adsc_min) < args.min_adsc_anchors:
            continue
        adsc_min = adsc_min.sort_values("minute_ts").drop_duplicates("minute_ts")
        for i in range(len(adsc_min) - 1):
            if len(samples_a) >= args.max_samples:
                break
            s = adsc_min.iloc[i]
            e = adsc_min.iloc[i + 1]
            start_t = pd.to_datetime(s["minute_ts"], utc=True)
            end_t = pd.to_datetime(e["minute_ts"], utc=True)
            gap_m = int((end_t - start_t).total_seconds() // 60)
            sample_id = f"{fp.stem}_a{i:03d}"
            if gap_m < args.min_gap_minutes or gap_m > args.max_gap_minutes:
                audit_rows.append(
                    {
                        "sample_id": sample_id,
                        "flight_id": flight_id,
                        "flight_date": flight_date,
                        "adsc_anchor_start_time": start_t.isoformat(),
                        "adsc_anchor_end_time": end_t.isoformat(),
                        "gap_minutes": gap_m,
                        "matched_adsb_flight": fp.stem,
                        "adsb_match_confidence_rule": "same_file_source_match",
                        "pre_gap_adsb_context_len": 0,
                        "post_gap_adsb_context_len": 0,
                        "pre_gap_adsb_context_complete": False,
                        "post_gap_adsb_context_complete": False,
                        "gap_inner_adsb_used": False,
                        "anchor_consistency_ok": False,
                        "skip_reason": "gap_out_of_range",
                    }
                )
                continue

            start_anchor = (float(s["lat"]), float(s["lon"]), float(s["alt"]))
            end_anchor = (float(e["lat"]), float(e["lon"]), float(e["alt"]))
            frame_a, pre_len_a, post_len_a, left_t, right_t, left_fill_m, middle_m, right_fill_m = _build_task_frame(
                minute_all_adsb=adsb_min,
                start_t=start_t,
                end_t=end_t,
                start_anchor=start_anchor,
                end_anchor=end_anchor,
                task_type="pure_adsc",
                context_minutes=args.context_minutes,
                max_side_fill_minutes=args.max_side_fill_minutes,
                left_context_max_points=args.left_context_max_points,
            )
            if frame_a.empty:
                audit_rows.append(
                    {
                        "sample_id": sample_id,
                        "flight_id": flight_id,
                        "flight_date": flight_date,
                        "adsc_anchor_start_time": start_t.isoformat(),
                        "adsc_anchor_end_time": end_t.isoformat(),
                        "gap_minutes": gap_m,
                        "matched_adsb_flight": fp.stem,
                        "adsb_match_confidence_rule": "same_file_source_match",
                        "pre_gap_adsb_context_len": 0,
                        "post_gap_adsb_context_len": 0,
                        "pre_gap_adsb_context_complete": False,
                        "post_gap_adsb_context_complete": False,
                        "left_fill_minutes": 0,
                        "middle_gap_minutes": int(gap_m),
                        "right_fill_minutes": 0,
                        "left_known_time": "",
                        "right_known_time": "",
                        "gap_inner_adsb_used": False,
                        "anchor_consistency_ok": False,
                        "skip_reason": "missing_left_or_right_boundary_adsb",
                    }
                )
                continue
            frame_a["sample_id"] = sample_id
            frame_a["flight_id"] = flight_id

            frame_b, pre_len_b, post_len_b, _, _, _, _, _ = _build_task_frame(
                minute_all_adsb=adsb_min,
                start_t=start_t,
                end_t=end_t,
                start_anchor=start_anchor,
                end_anchor=end_anchor,
                task_type="adsc_plus_local_adsb",
                context_minutes=args.context_minutes,
                max_side_fill_minutes=args.max_side_fill_minutes,
                left_context_max_points=args.left_context_max_points,
            )
            if frame_b.empty:
                continue
            frame_b["sample_id"] = sample_id
            frame_b["flight_id"] = flight_id
            frame_c, pre_len_c, post_len_c, _, _, _, _, _ = _build_task_frame(
                minute_all_adsb=adsb_min,
                start_t=start_t,
                end_t=end_t,
                start_anchor=start_anchor,
                end_anchor=end_anchor,
                task_type="adsc_plus_left_adsb",
                context_minutes=args.left_context_minutes,
                max_side_fill_minutes=args.max_side_fill_minutes,
                left_context_max_points=args.left_context_max_points,
            )
            if frame_c.empty:
                continue
            frame_c["sample_id"] = sample_id
            frame_c["flight_id"] = flight_id

            sp_a = ReplaySample(
                sample_id=sample_id,
                flight_id=flight_id,
                flight_date=flight_date,
                anchor_start_time=start_t,
                anchor_end_time=end_t,
                gap_minutes=gap_m,
                anchor_start_lat=start_anchor[0],
                anchor_start_lon=start_anchor[1],
                anchor_start_alt=start_anchor[2],
                anchor_end_lat=end_anchor[0],
                anchor_end_lon=end_anchor[1],
                anchor_end_alt=end_anchor[2],
                left_known_time=left_t,
                right_known_time=right_t,
                left_fill_minutes=left_fill_m,
                middle_gap_minutes=middle_m,
                right_fill_minutes=right_fill_m,
                pre_gap_adsb_context_len=pre_len_a,
                post_gap_adsb_context_len=post_len_a,
                pre_gap_adsb_context_complete=(pre_len_a >= args.context_minutes),
                post_gap_adsb_context_complete=(post_len_a >= args.context_minutes),
                gap_inner_adsb_used=False,
                matched_adsb_flight=fp.stem,
                adsb_match_confidence_rule="same_file_source_match",
                task_type="pure_adsc",
                frame=frame_a,
            )
            sp_b = ReplaySample(
                sample_id=sample_id,
                flight_id=flight_id,
                flight_date=flight_date,
                anchor_start_time=start_t,
                anchor_end_time=end_t,
                gap_minutes=gap_m,
                anchor_start_lat=start_anchor[0],
                anchor_start_lon=start_anchor[1],
                anchor_start_alt=start_anchor[2],
                anchor_end_lat=end_anchor[0],
                anchor_end_lon=end_anchor[1],
                anchor_end_alt=end_anchor[2],
                left_known_time=left_t,
                right_known_time=right_t,
                left_fill_minutes=left_fill_m,
                middle_gap_minutes=middle_m,
                right_fill_minutes=right_fill_m,
                pre_gap_adsb_context_len=pre_len_b,
                post_gap_adsb_context_len=post_len_b,
                pre_gap_adsb_context_complete=(pre_len_b >= args.context_minutes),
                post_gap_adsb_context_complete=(post_len_b >= args.context_minutes),
                gap_inner_adsb_used=False,
                matched_adsb_flight=fp.stem,
                adsb_match_confidence_rule="same_file_source_match",
                task_type="adsc_plus_local_adsb",
                frame=frame_b,
            )
            samples_a.append(sp_a)
            samples_b.append(sp_b)
            sp_c = ReplaySample(
                sample_id=sample_id,
                flight_id=flight_id,
                flight_date=flight_date,
                anchor_start_time=start_t,
                anchor_end_time=end_t,
                gap_minutes=gap_m,
                anchor_start_lat=start_anchor[0],
                anchor_start_lon=start_anchor[1],
                anchor_start_alt=start_anchor[2],
                anchor_end_lat=end_anchor[0],
                anchor_end_lon=end_anchor[1],
                anchor_end_alt=end_anchor[2],
                left_known_time=left_t,
                right_known_time=right_t,
                left_fill_minutes=left_fill_m,
                middle_gap_minutes=middle_m,
                right_fill_minutes=right_fill_m,
                pre_gap_adsb_context_len=pre_len_c,
                post_gap_adsb_context_len=post_len_c,
                pre_gap_adsb_context_complete=(pre_len_c >= args.left_context_max_points),
                post_gap_adsb_context_complete=False,
                gap_inner_adsb_used=False,
                matched_adsb_flight=fp.stem,
                adsb_match_confidence_rule="same_file_source_match",
                task_type="adsc_plus_left_adsb",
                frame=frame_c,
            )
            samples_c.append(sp_c)
            audit_rows.append(
                {
                    "sample_id": sample_id,
                    "flight_id": flight_id,
                    "flight_date": flight_date,
                    "adsc_anchor_start_time": start_t.isoformat(),
                    "adsc_anchor_end_time": end_t.isoformat(),
                    "gap_minutes": gap_m,
                    "matched_adsb_flight": fp.stem,
                    "adsb_match_confidence_rule": "same_file_source_match",
                    "pre_gap_adsb_context_len": pre_len_b,
                    "post_gap_adsb_context_len": post_len_b,
                    "pre_gap_adsb_context_complete": pre_len_b >= args.context_minutes,
                    "post_gap_adsb_context_complete": post_len_b >= args.context_minutes,
                    "left_context_pre_len_c": pre_len_c,
                    "left_context_post_len_c": post_len_c,
                    "left_fill_minutes": left_fill_m,
                    "middle_gap_minutes": middle_m,
                    "right_fill_minutes": right_fill_m,
                    "left_known_time": left_t.isoformat(),
                    "right_known_time": right_t.isoformat(),
                    "gap_inner_adsb_used": False,
                    "anchor_consistency_ok": True,
                    "skip_reason": "",
                }
            )
        if len(samples_a) >= args.max_samples:
            break

    if not samples_a:
        raise RuntimeError("No valid real ADS-C replay samples built.")

    # Data audit first.
    audit_df = pd.DataFrame(audit_rows)
    audit_path = out_dir / "real_adsc_replay_data_audit.csv"
    audit_df.to_csv(audit_path, index=False)

    # Predict for A and B (legacy anchor-gap frames for metrics).
    frame_a_all = pd.concat([s.frame for s in samples_a], ignore_index=True)
    frame_b_all = pd.concat([s.frame for s in samples_b], ignore_index=True)
    frame_c_all = pd.concat([s.frame for s in samples_c], ignore_index=True)
    pred_a = _predict_on_frame(cfg, ckpt, frame_a_all)
    pred_b = _predict_on_frame(cfg, ckpt, frame_b_all)
    pred_c = _predict_on_frame(cfg, ckpt, frame_c_all)

    # Index ADS-B minute tables for eval / plots.
    flight_adsb_cache: dict[str, pd.DataFrame] = {}
    flight_adsc_cache: dict[str, pd.DataFrame] = {}
    for fp in csv_files:
        fdf = pd.read_csv(fp)
        if "source" not in fdf.columns:
            continue
        fdf["source"] = fdf["source"].astype(str).str.lower()
        adsb_min = _resample_minute(fdf[fdf["source"].eq("adsb")].copy())
        adsc_raw = fdf[fdf["source"].eq("adsc")].copy()
        if len(adsb_min):
            flight_adsb_cache[fp.stem] = adsb_min
        if len(adsc_raw):
            flight_adsc_cache[fp.stem] = adsc_raw

    def evaluate(samples: list[ReplaySample], pred_df: pd.DataFrame, task_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        qual_rows = []
        b_rows = []
        for s in samples:
            ps = pred_df[pred_df["sample_id"].astype(str).eq(s.sample_id)].copy()
            if ps.empty:
                continue
            ps["minute_ts"] = pd.to_datetime(ps["minute_ts"], utc=True)
            ps = ps.sort_values("minute_ts")
            gap_start = s.anchor_start_time.floor("min")
            gap_end = s.anchor_end_time.floor("min")
            seg = ps[(ps["minute_ts"] >= gap_start) & (ps["minute_ts"] <= gap_end)].copy()
            if len(seg) < 2:
                continue
            # Hard-anchor consistency.
            srow = seg.iloc[0]
            erow = seg.iloc[-1]
            start_err = _haversine_m(float(srow["pred_lat"]), float(srow["pred_lon"]), s.anchor_start_lat, s.anchor_start_lon)
            end_err = _haversine_m(float(erow["pred_lat"]), float(erow["pred_lon"]), s.anchor_end_lat, s.anchor_end_lon)
            start_alt_err = abs(float(srow["pred_alt"]) - s.anchor_start_alt)
            end_alt_err = abs(float(erow["pred_alt"]) - s.anchor_end_alt)
            # Numerical restore may keep tiny residuals; use practical hard-consistency tolerance.
            anchor_ok = (start_err <= 1.0 and end_err <= 1.0 and start_alt_err <= 5.0 and end_alt_err <= 5.0)

            turn_mean, alt_smooth, max_turn, max_vr = _smoothness_metrics(
                seg["pred_lat"].to_numpy(dtype=float),
                seg["pred_lon"].to_numpy(dtype=float),
                seg["pred_alt"].to_numpy(dtype=float),
            )
            abnormal = bool((max_turn > 45.0) or (max_vr > 1200.0))
            alt_arr = seg["pred_alt"].to_numpy(dtype=float)
            q = _quality_flags_from_altitude(
                alt=alt_arr,
                left_boundary_alt=s.anchor_start_alt,
                right_boundary_alt=s.anchor_end_alt,
            )
            epos = _edge_spike_position_stats(
                alt=alt_arr,
                left_boundary_alt=s.anchor_start_alt,
                right_boundary_alt=s.anchor_end_alt,
                jump_thresh=300.0,
            )
            seg_bucket = _segment_bucket3(float(s.gap_minutes))
            anchor_pat = "two_anchor"
            if "matched_risk_rule" in seg.columns and len(seg):
                matched_rule = str(seg["matched_risk_rule"].iloc[0])
            else:
                matched_rule = "unknown"

            adsb_all = flight_adsb_cache.get(s.matched_adsb_flight, pd.DataFrame(columns=["minute_ts", "lat", "lon", "alt"]))
            if len(adsb_all):
                adsb_all = adsb_all.copy()
                adsb_all["minute_ts"] = pd.to_datetime(adsb_all["minute_ts"], utc=True)
            pre_adsb = adsb_all[(adsb_all["minute_ts"] < gap_start) & (adsb_all["minute_ts"] >= gap_start - pd.Timedelta(minutes=5))]
            post_adsb = adsb_all[(adsb_all["minute_ts"] > gap_end) & (adsb_all["minute_ts"] <= gap_end + pd.Timedelta(minutes=5))]
            bm = _compute_boundary_metrics(seg, pre_adsb, post_adsb)

            qual_rows.append(
                {
                    "run_name": ckpt.stem,
                    "task_type": task_type,
                    "sample_id": s.sample_id,
                    "flight_id": s.flight_id,
                    "anchor_consistency_ok": anchor_ok,
                    "anchor_start_error": start_err + start_alt_err,
                    "anchor_end_error": end_err + end_alt_err,
                    "gap_minutes": s.gap_minutes,
                    "recovered_minutes": int(len(seg) - 2),
                    "planar_path_smoothness": turn_mean,
                    "altitude_smoothness": alt_smooth,
                    "max_turn_rate": max_turn,
                    "max_vertical_rate": max_vr,
                    "abnormal_flag": abnormal,
                    "overshoot_flag": bool(q["overshoot_flag"]),
                    "edge_spike_flag": bool(q["edge_spike_flag"]),
                    "left_step1_spike_flag": bool(epos["left_step1_spike_flag"]),
                    "left_step2_spike_flag": bool(epos["left_step2_spike_flag"]),
                    "right_step1_spike_flag": bool(epos["right_step1_spike_flag"]),
                    "right_step2_spike_flag": bool(epos["right_step2_spike_flag"]),
                    "segment_bucket": seg_bucket,
                    "anchor_pattern": anchor_pat,
                    "matched_risk_rule": matched_rule,
                    "pre_context_minutes": s.pre_gap_adsb_context_len,
                    "post_context_minutes": s.post_gap_adsb_context_len,
                }
            )
            b_rows.append(
                {
                    "run_name": ckpt.stem,
                    "task_type": task_type,
                    "sample_id": s.sample_id,
                    "flight_id": s.flight_id,
                    "pre_context_minutes": s.pre_gap_adsb_context_len,
                    "post_context_minutes": s.post_gap_adsb_context_len,
                    **bm,
                }
            )
        return pd.DataFrame(qual_rows), pd.DataFrame(b_rows)

    qual_a, bnd_a = evaluate(samples_a, pred_a, "pure_adsc")
    qual_b, bnd_b = evaluate(samples_b, pred_b, "adsc_plus_local_adsb")
    qual_c, bnd_c = evaluate(samples_c, pred_c, "adsc_plus_left_adsb")

    # Aggregate csv with requested fields.
    def aggregate_qual(df: pd.DataFrame, task_type: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(
                [
                    {
                        "run_name": ckpt.stem,
                        "task_type": task_type,
                        "sample_count": 0,
                    }
                ]
            )
        return pd.DataFrame(
            [
                {
                    "run_name": ckpt.stem,
                    "task_type": task_type,
                    "sample_count": int(len(df)),
                    "anchor_consistency_ok_rate": float(df["anchor_consistency_ok"].mean()),
                    "anchor_start_error": float(df["anchor_start_error"].mean()),
                    "anchor_end_error": float(df["anchor_end_error"].mean()),
                    "mean_gap_minutes": float(df["gap_minutes"].mean()),
                    "mean_recovered_minutes": float(df["recovered_minutes"].mean()),
                    "planar_path_smoothness": float(df["planar_path_smoothness"].mean()),
                    "altitude_smoothness": float(df["altitude_smoothness"].mean()),
                    "max_turn_rate": float(df["max_turn_rate"].mean()),
                    "max_vertical_rate": float(df["max_vertical_rate"].mean()),
                    "abnormal_flag_rate": float(df["abnormal_flag"].mean()),
                    "overshoot_rate": float(df["overshoot_flag"].mean()) if "overshoot_flag" in df.columns else float("nan"),
                    "edge_spike_rate": float(df["edge_spike_flag"].mean()) if "edge_spike_flag" in df.columns else float("nan"),
                    "left_step1_spike_rate": float(df["left_step1_spike_flag"].mean()) if "left_step1_spike_flag" in df.columns else float("nan"),
                    "left_step2_spike_rate": float(df["left_step2_spike_flag"].mean()) if "left_step2_spike_flag" in df.columns else float("nan"),
                    "notes": "No gap-inner ground-truth RMSE; qualitative replay only.",
                }
            ]
        )

    def aggregate_boundary(df: pd.DataFrame, task_type: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame([{"run_name": ckpt.stem, "task_type": task_type, "sample_count": 0}])
        cols = [
            "pre_context_minutes",
            "post_context_minutes",
            "pre_boundary_heading_gap",
            "post_boundary_heading_gap",
            "pre_boundary_speed_gap",
            "post_boundary_speed_gap",
            "pre_boundary_vertical_rate_gap",
            "post_boundary_vertical_rate_gap",
            "pre_boundary_alt_slope_gap",
            "post_boundary_alt_slope_gap",
            "boundary_planar_continuity_score",
            "boundary_alt_continuity_score",
        ]
        row = {c: float(pd.to_numeric(df[c], errors="coerce").mean()) for c in cols}
        row.update(
            {
                "run_name": ckpt.stem,
                "task_type": task_type,
                "sample_count": int(len(df)),
                "notes": "Boundary consistency with ADS-B pre/post context only.",
            }
        )
        return pd.DataFrame([row])

    agg_qa = aggregate_qual(qual_a, "pure_adsc")
    agg_qb = aggregate_qual(qual_b, "adsc_plus_local_adsb")
    agg_qc = aggregate_qual(qual_c, "adsc_plus_left_adsb")
    agg_ba = aggregate_boundary(bnd_a, "pure_adsc")
    agg_bb = aggregate_boundary(bnd_b, "adsc_plus_local_adsb")
    agg_bc = aggregate_boundary(bnd_c, "adsc_plus_left_adsb")

    (out_dir / "real_adsc_replay_qualitative_eval.csv").write_text(agg_qa.to_csv(index=False), encoding="utf-8")
    (out_dir / "real_adsc_plus_local_adsb_qualitative_eval.csv").write_text(agg_qb.to_csv(index=False), encoding="utf-8")
    (out_dir / "real_adsc_plus_left_adsb_qualitative_eval.csv").write_text(agg_qc.to_csv(index=False), encoding="utf-8")
    (out_dir / "real_adsc_replay_boundary_consistency_eval.csv").write_text(agg_ba.to_csv(index=False), encoding="utf-8")
    (out_dir / "real_adsc_plus_local_adsb_boundary_consistency_eval.csv").write_text(agg_bb.to_csv(index=False), encoding="utf-8")
    (out_dir / "real_adsc_plus_left_adsb_boundary_consistency_eval.csv").write_text(agg_bc.to_csv(index=False), encoding="utf-8")

    # Left-context coverage summary for the new branch.
    lc = pd.DataFrame(
        {
            "sample_id": [s.sample_id for s in samples_c],
            "pre_context_len": [int(s.pre_gap_adsb_context_len) for s in samples_c],
            "post_context_len": [int(s.post_gap_adsb_context_len) for s in samples_c],
        }
    )
    if len(lc):
        bins_rows = [
            {"bucket": "len_0", "count": int((lc["pre_context_len"] == 0).sum())},
            {"bucket": "len_1_4", "count": int(((lc["pre_context_len"] >= 1) & (lc["pre_context_len"] <= 4)).sum())},
            {"bucket": "len_5_plus", "count": int((lc["pre_context_len"] >= 5).sum())},
        ]
        total = max(1, int(len(lc)))
        for r in bins_rows:
            r["ratio"] = float(r["count"] / total)
        pd.DataFrame(bins_rows).to_csv(out_dir / "left_context_coverage_stats.csv", index=False)
    else:
        pd.DataFrame(columns=["bucket", "count", "ratio"]).to_csv(out_dir / "left_context_coverage_stats.csv", index=False)

    def _bucket_stats(df: pd.DataFrame, task_name: str, by_cols: list[str], out_name: str) -> None:
        if df.empty:
            pd.DataFrame(columns=by_cols + ["task_type", "count", "abnormal_ratio", "overshoot_rate", "edge_spike_rate", "left_step1_spike_rate", "left_step2_spike_rate"]).to_csv(out_dir / out_name, index=False)
            return
        g = (
            df.groupby(by_cols, dropna=False)
            .agg(
                count=("sample_id", "count"),
                abnormal_ratio=("abnormal_flag", "mean"),
                overshoot_rate=("overshoot_flag", "mean"),
                edge_spike_rate=("edge_spike_flag", "mean"),
                left_step1_spike_rate=("left_step1_spike_flag", "mean"),
                left_step2_spike_rate=("left_step2_spike_flag", "mean"),
            )
            .reset_index()
        )
        g["task_type"] = task_name
        g.to_csv(out_dir / out_name, index=False)

    _bucket_stats(qual_a, "pure_adsc", ["segment_bucket"], "bucket_stats_pure_adsc_by_segment_bucket.csv")
    _bucket_stats(qual_b, "adsc_plus_local_adsb", ["segment_bucket"], "bucket_stats_plus_local_adsb_by_segment_bucket.csv")
    _bucket_stats(qual_c, "adsc_plus_left_adsb", ["segment_bucket"], "bucket_stats_plus_left_adsb_by_segment_bucket.csv")
    _bucket_stats(qual_a, "pure_adsc", ["segment_bucket", "anchor_pattern"], "bucket_stats_pure_adsc_by_bucket_pattern.csv")
    _bucket_stats(qual_b, "adsc_plus_local_adsb", ["segment_bucket", "anchor_pattern"], "bucket_stats_plus_local_adsb_by_bucket_pattern.csv")
    _bucket_stats(qual_c, "adsc_plus_left_adsb", ["segment_bucket", "anchor_pattern"], "bucket_stats_plus_left_adsb_by_bucket_pattern.csv")
    _bucket_stats(qual_a, "pure_adsc", ["matched_risk_rule"], "bucket_stats_pure_adsc_by_rule.csv")
    _bucket_stats(qual_b, "adsc_plus_local_adsb", ["matched_risk_rule"], "bucket_stats_plus_local_adsb_by_rule.csv")
    _bucket_stats(qual_c, "adsc_plus_left_adsb", ["matched_risk_rule"], "bucket_stats_plus_left_adsb_by_rule.csv")

    # Build fill-interval recovery frames for plotting.
    fill_meta_by_sample: dict[str, list[dict]] = {}
    fill_frames_a: list[pd.DataFrame] = []
    fill_frames_b: list[pd.DataFrame] = []
    fill_frames_c: list[pd.DataFrame] = []

    for s in samples_a:
        adsb_all = flight_adsb_cache.get(s.matched_adsb_flight, pd.DataFrame(columns=["minute_ts", "lat", "lon", "alt"])).copy()
        adsc_raw = flight_adsc_cache.get(s.matched_adsb_flight, pd.DataFrame(columns=["timestamp", "lat", "lon", "baroaltitude"])).copy()
        if len(adsb_all):
            adsb_all["minute_ts"] = pd.to_datetime(adsb_all["minute_ts"], utc=True)
        if len(adsc_raw):
            adsc_raw["timestamp"] = pd.to_datetime(adsc_raw["timestamp"], utc=True)
        if len(adsb_all):
            w_start = adsb_all["minute_ts"].min()
            w_end = adsb_all["minute_ts"].max()
        else:
            w_start = s.anchor_start_time.floor("min")
            w_end = s.anchor_end_time.floor("min")
        if len(adsc_raw):
            w_start = min(w_start, adsc_raw["timestamp"].min())
            w_end = max(w_end, adsc_raw["timestamp"].max())

        blocks = _build_known_blocks(adsb_all, adsc_raw, w_start, w_end, gap_break_min=10.0)
        fills = _build_fill_intervals(blocks, min_gap_min=2.0)
        fill_meta: list[dict] = []
        for fi, f in enumerate(fills, start=1):
            left_block = blocks[f["left_block_idx"]]
            right_block = blocks[f["right_block_idx"]]
            left_state = _extract_block_boundary_state(left_block, "end")
            right_state = _extract_block_boundary_state(right_block, "start")
            fill_id = f"f{fi:02d}"
            fill_sid = f"{s.sample_id}__{fill_id}"
            frame_a, _, _ = _build_fill_frame(
                minute_all_adsb=adsb_all,
                fill_start=f["fill_start_time"],
                fill_end=f["fill_end_time"],
                left_state=left_state,
                right_state=right_state,
                task_type="pure_adsc",
                context_minutes=args.context_minutes,
            )
            if not frame_a.empty:
                frame_a["sample_id"] = fill_sid
                frame_a["flight_id"] = s.flight_id
                fill_frames_a.append(frame_a)
            frame_b, _, _ = _build_fill_frame(
                minute_all_adsb=adsb_all,
                fill_start=f["fill_start_time"],
                fill_end=f["fill_end_time"],
                left_state=left_state,
                right_state=right_state,
                task_type="adsc_plus_local_adsb",
                context_minutes=args.context_minutes,
                left_context_max_points=args.left_context_max_points,
            )
            if not frame_b.empty:
                frame_b["sample_id"] = fill_sid
                frame_b["flight_id"] = s.flight_id
                fill_frames_b.append(frame_b)
            frame_c, _, _ = _build_fill_frame(
                minute_all_adsb=adsb_all,
                fill_start=f["fill_start_time"],
                fill_end=f["fill_end_time"],
                left_state=left_state,
                right_state=right_state,
                task_type="adsc_plus_left_adsb",
                context_minutes=args.left_context_minutes,
                left_context_max_points=args.left_context_max_points,
            )
            if not frame_c.empty:
                frame_c["sample_id"] = fill_sid
                frame_c["flight_id"] = s.flight_id
                fill_frames_c.append(frame_c)
            fill_meta.append(
                {
                    "fill_id": fill_id,
                    "fill_sid": fill_sid,
                    "fill_start_time": f["fill_start_time"],
                    "fill_end_time": f["fill_end_time"],
                    "fill_minutes": f["fill_minutes"],
                    "left_block_type": f["left_block_type"],
                    "right_block_type": f["right_block_type"],
                    "left_boundary_lat": float(left_state["lat"]),
                    "left_boundary_lon": float(left_state["lon"]),
                    "left_boundary_alt": float(left_state["alt"]),
                    "right_boundary_lat": float(right_state["lat"]),
                    "right_boundary_lon": float(right_state["lon"]),
                    "right_boundary_alt": float(right_state["alt"]),
                }
            )
        fill_meta_by_sample[s.sample_id] = fill_meta

    pred_fill_a = _predict_on_frame(cfg, ckpt, pd.concat(fill_frames_a, ignore_index=True)) if fill_frames_a else pd.DataFrame()
    pred_fill_b = _predict_on_frame(cfg, ckpt, pd.concat(fill_frames_b, ignore_index=True)) if fill_frames_b else pd.DataFrame()
    pred_fill_c = _predict_on_frame(cfg, ckpt, pd.concat(fill_frames_c, ignore_index=True)) if fill_frames_c else pd.DataFrame()
    # V1.5 segment policy requires main-branch prediction as residual-off reference.
    segment_policy = _build_v15_segment_policy(cfg)
    projection_cfg = cfg.get("inference", {}).get("left_edge_projection", {})
    smoothing_cfg = cfg.get("inference", {}).get("left_edge_smoothing", {})
    replay_alt_series_source = str(cfg.get("inference", {}).get("replay_alt_series_source", "policy_post_alt"))
    pred_fill_main_a = (
        _predict_on_frame(cfg, ckpt, pd.concat(fill_frames_a, ignore_index=True), pred_key="pred_pos_main")
        if (fill_frames_a and segment_policy.enabled)
        else _predict_on_frame(cfg, ckpt, pd.DataFrame(), pred_key="pred_pos_main")
    )
    pred_fill_main_b = (
        _predict_on_frame(cfg, ckpt, pd.concat(fill_frames_b, ignore_index=True), pred_key="pred_pos_main")
        if (fill_frames_b and segment_policy.enabled)
        else _predict_on_frame(cfg, ckpt, pd.DataFrame(), pred_key="pred_pos_main")
    )
    pred_fill_main_c = (
        _predict_on_frame(cfg, ckpt, pd.concat(fill_frames_c, ignore_index=True), pred_key="pred_pos_main")
        if (fill_frames_c and segment_policy.enabled)
        else _predict_on_frame(cfg, ckpt, pd.DataFrame(), pred_key="pred_pos_main")
    )

    # Plots (4 images per sample: raw full/local + recovered full/boundary)
    raw_full = out_dir / "plots/raw_fused/full_recovery"
    raw_bnd = out_dir / "plots/raw_fused/boundary_consistency"
    rec_full = out_dir / "plots/recovered/full_recovery"
    rec_bnd = out_dir / "plots/recovered/boundary_consistency"
    compare_aux = out_dir / "plots/compare_aux_before_after"
    compare_left_ctx = out_dir / "plots/compare_left_context_5min"
    known_blocks_rows = []
    fill_rows = []
    known_blocks_after_rows = []
    fill_after_rows = []
    recovered_after_rows = []
    plot_after_rows = []
    segment_policy_rows = []
    strategy_variant_name = str(cfg.get("model", {}).get("model_variant", "default"))

    plot_n = min(args.plot_count, len(samples_a), len(samples_b), len(samples_c))
    bnd_a_map = {r["sample_id"]: r for r in bnd_a.to_dict(orient="records")}
    bnd_b_map = {r["sample_id"]: r for r in bnd_b.to_dict(orient="records")}
    for i in range(plot_n):
        sa = samples_a[i]
        sb = samples_b[i]
        sc = samples_c[i]
        adsb_all = flight_adsb_cache.get(sa.matched_adsb_flight, pd.DataFrame(columns=["minute_ts", "lat", "lon", "alt"])).copy()
        adsc_raw = flight_adsc_cache.get(sa.matched_adsb_flight, pd.DataFrame(columns=["timestamp", "lat", "lon", "baroaltitude"])).copy()
        if len(adsb_all):
            adsb_all["minute_ts"] = pd.to_datetime(adsb_all["minute_ts"], utc=True)
        if len(adsc_raw):
            adsc_raw["timestamp"] = pd.to_datetime(adsc_raw["timestamp"], utc=True)
        psa = pred_a[pred_a["sample_id"].astype(str).eq(sa.sample_id)].copy()
        psb = pred_b[pred_b["sample_id"].astype(str).eq(sb.sample_id)].copy()
        psc = pred_c[pred_c["sample_id"].astype(str).eq(sc.sample_id)].copy()
        fills_meta = fill_meta_by_sample.get(sa.sample_id, [])
        pred_fill_seg_a = []
        pred_fill_seg_b = []
        for f in fills_meta:
            seg_a = pred_fill_a[pred_fill_a["sample_id"].astype(str).eq(f["fill_sid"])].copy()
            seg_a_main = pred_fill_main_a[pred_fill_main_a["sample_id"].astype(str).eq(f["fill_sid"])].copy()
            if len(seg_a):
                seg_a["minute_ts"] = pd.to_datetime(seg_a["minute_ts"], utc=True)
                proj_enabled = bool(projection_cfg.get("use_left_edge_projection", False))
                proj_mode = str(projection_cfg.get("left_edge_projection_mode", "envelope"))
                proj_steps = int(projection_cfg.get("left_edge_projection_steps", 2))
                proj_band = float(projection_cfg.get("left_local_band_ft", 200.0))
                seg_a = seg_a.sort_values("minute_ts").reset_index(drop=True)
                seg_a_proj, _ = _apply_left_edge_projection(
                    alt_main=seg_a["pred_alt"].to_numpy(dtype=float),
                    left_boundary_alt=float(f.get("left_boundary_alt", seg_a["pred_alt"].iloc[0])),
                    right_boundary_alt=float(f.get("right_boundary_alt", seg_a["pred_alt"].iloc[-1])),
                    enabled=proj_enabled,
                    mode=proj_mode,
                    steps=proj_steps,
                    left_local_band=proj_band,
                )
                seg_a["pred_alt"] = seg_a_proj
                if segment_policy.enabled and len(seg_a_main) == len(seg_a):
                    seg_a_main["minute_ts"] = pd.to_datetime(seg_a_main["minute_ts"], utc=True)
                    seg_a_main = seg_a_main.sort_values("minute_ts").reset_index(drop=True)
                    adj_alt, meta = segment_policy.apply(
                        on_alt=seg_a["pred_alt"].to_numpy(dtype=float),
                        off_alt=seg_a_main["pred_alt"].to_numpy(dtype=float),
                        fill_minutes=float(f["fill_minutes"]),
                        left_block_type=str(f["left_block_type"]),
                        right_block_type=str(f["right_block_type"]),
                    )
                    seg_a["pred_alt"] = adj_alt
                    segment_policy_rows.append(
                        {
                            "sample_id": sa.sample_id,
                            "fill_id": f["fill_id"],
                            "task_type": "pure_adsc",
                            "strategy_variant": strategy_variant_name,
                            **meta.to_dict(),
                        }
                    )
                smooth_enabled = bool(smoothing_cfg.get("use_edge_smoothing_projection", False))
                smooth_mode = str(smoothing_cfg.get("edge_smoothing_mode", "left_blend"))
                smooth_steps = int(smoothing_cfg.get("left_edge_smoothing_steps", 3))
                smooth_betas = smoothing_cfg.get("left_blend_betas", [0.5, 0.3, 0.1])
                smooth_cap = float(smoothing_cfg.get("left_slope_cap_ft", 300.0))
                seg_a_sm, _ = _apply_left_edge_smoothing(
                    alt_final=seg_a["pred_alt"].to_numpy(dtype=float),
                    left_boundary_alt=float(f.get("left_boundary_alt", seg_a["pred_alt"].iloc[0])),
                    enabled=smooth_enabled,
                    mode=smooth_mode,
                    steps=smooth_steps,
                    blend_betas=list(smooth_betas) if isinstance(smooth_betas, (list, tuple)) else [0.5, 0.3, 0.1],
                    slope_cap=smooth_cap,
                )
                seg_a["pred_alt"] = seg_a_sm
                right_smooth_cfg = smoothing_cfg.get("right_edge_smoothing", {}) if isinstance(smoothing_cfg, dict) else {}
                seg_a_sm_r, _ = _apply_right_edge_smoothing(
                    alt_final=seg_a["pred_alt"].to_numpy(dtype=float),
                    right_boundary_alt=float(f.get("right_boundary_alt", seg_a["pred_alt"].iloc[-1])),
                    enabled=bool(right_smooth_cfg.get("use_right_edge_smoothing", False)),
                    mode=str(right_smooth_cfg.get("right_edge_smoothing_mode", "right_blend")),
                    steps=int(right_smooth_cfg.get("right_edge_smoothing_steps", 2)),
                    blend_betas=list(right_smooth_cfg.get("right_blend_betas", [0.2, 0.5])),
                    right_local_band=float(right_smooth_cfg.get("right_local_band_ft", 200.0)),
                )
                cond_cfg = smoothing_cfg.get("conditional_rightstep2_fuse", {}) if isinstance(smoothing_cfg, dict) else {}
                seg_a_fused, _, _, _, _ = _apply_conditional_rightstep2_fuse(
                    seg_a_sm_r,
                    segment_bucket=str(_segment_bucket3(float(f["fill_minutes"]))),
                    anchor_pattern=str(_anchor_pattern(str(f["left_block_type"]), str(f["right_block_type"]))),
                    right_boundary_alt=float(f.get("right_boundary_alt", seg_a["pred_alt"].iloc[-1])),
                    enabled=bool(cond_cfg.get("use_conditional_rightstep2_fuse", False)),
                    target_bucket=str(cond_cfg.get("conditional_target_bucket", "medium")),
                    target_pattern=str(cond_cfg.get("conditional_target_pattern", "two_anchor")),
                    tau_jump=float(cond_cfg.get("tau_jump", 200.0)),
                    mode=str(cond_cfg.get("fuse_mode", "local_interp")),
                    fuse_lambda=float(cond_cfg.get("fuse_lambda", 0.5)),
                    use_second_condition=bool(cond_cfg.get("use_second_condition", False)),
                    tau_curve=float(cond_cfg.get("tau_curve", 200.0)),
                    right_local_band=float(cond_cfg.get("right_local_band_ft", 200.0)),
                )
                seg_a["pred_alt"] = seg_a_fused
                pred_fill_seg_a.append(seg_a)
                recovered_after_rows.append(
                    {
                        "sample_id": sa.sample_id,
                        "fill_id": f["fill_id"],
                        "fill_start_time": f["fill_start_time"].isoformat(),
                        "fill_end_time": f["fill_end_time"].isoformat(),
                        "fill_minutes": f["fill_minutes"],
                        "left_block_type": f["left_block_type"],
                        "right_block_type": f["right_block_type"],
                        "segment_len": int(len(seg_a)),
                    }
                )
            seg_b = pred_fill_b[pred_fill_b["sample_id"].astype(str).eq(f["fill_sid"])].copy()
            seg_b_main = pred_fill_main_b[pred_fill_main_b["sample_id"].astype(str).eq(f["fill_sid"])].copy()
            if len(seg_b):
                seg_b["minute_ts"] = pd.to_datetime(seg_b["minute_ts"], utc=True)
                proj_enabled = bool(projection_cfg.get("use_left_edge_projection", False))
                proj_mode = str(projection_cfg.get("left_edge_projection_mode", "envelope"))
                proj_steps = int(projection_cfg.get("left_edge_projection_steps", 2))
                proj_band = float(projection_cfg.get("left_local_band_ft", 200.0))
                seg_b = seg_b.sort_values("minute_ts").reset_index(drop=True)
                seg_b_proj, _ = _apply_left_edge_projection(
                    alt_main=seg_b["pred_alt"].to_numpy(dtype=float),
                    left_boundary_alt=float(f.get("left_boundary_alt", seg_b["pred_alt"].iloc[0])),
                    right_boundary_alt=float(f.get("right_boundary_alt", seg_b["pred_alt"].iloc[-1])),
                    enabled=proj_enabled,
                    mode=proj_mode,
                    steps=proj_steps,
                    left_local_band=proj_band,
                )
                seg_b["pred_alt"] = seg_b_proj
                if segment_policy.enabled and len(seg_b_main) == len(seg_b):
                    seg_b_main["minute_ts"] = pd.to_datetime(seg_b_main["minute_ts"], utc=True)
                    seg_b_main = seg_b_main.sort_values("minute_ts").reset_index(drop=True)
                    adj_alt, meta = segment_policy.apply(
                        on_alt=seg_b["pred_alt"].to_numpy(dtype=float),
                        off_alt=seg_b_main["pred_alt"].to_numpy(dtype=float),
                        fill_minutes=float(f["fill_minutes"]),
                        left_block_type=str(f["left_block_type"]),
                        right_block_type=str(f["right_block_type"]),
                    )
                    seg_b["pred_alt"] = adj_alt
                    segment_policy_rows.append(
                        {
                            "sample_id": sa.sample_id,
                            "fill_id": f["fill_id"],
                            "task_type": "adsc_plus_local_adsb",
                            "strategy_variant": strategy_variant_name,
                            **meta.to_dict(),
                        }
                    )
                smooth_enabled = bool(smoothing_cfg.get("use_edge_smoothing_projection", False))
                smooth_mode = str(smoothing_cfg.get("edge_smoothing_mode", "left_blend"))
                smooth_steps = int(smoothing_cfg.get("left_edge_smoothing_steps", 3))
                smooth_betas = smoothing_cfg.get("left_blend_betas", [0.5, 0.3, 0.1])
                smooth_cap = float(smoothing_cfg.get("left_slope_cap_ft", 300.0))
                seg_b_sm, _ = _apply_left_edge_smoothing(
                    alt_final=seg_b["pred_alt"].to_numpy(dtype=float),
                    left_boundary_alt=float(f.get("left_boundary_alt", seg_b["pred_alt"].iloc[0])),
                    enabled=smooth_enabled,
                    mode=smooth_mode,
                    steps=smooth_steps,
                    blend_betas=list(smooth_betas) if isinstance(smooth_betas, (list, tuple)) else [0.5, 0.3, 0.1],
                    slope_cap=smooth_cap,
                )
                seg_b["pred_alt"] = seg_b_sm
                right_smooth_cfg = smoothing_cfg.get("right_edge_smoothing", {}) if isinstance(smoothing_cfg, dict) else {}
                seg_b_sm_r, _ = _apply_right_edge_smoothing(
                    alt_final=seg_b["pred_alt"].to_numpy(dtype=float),
                    right_boundary_alt=float(f.get("right_boundary_alt", seg_b["pred_alt"].iloc[-1])),
                    enabled=bool(right_smooth_cfg.get("use_right_edge_smoothing", False)),
                    mode=str(right_smooth_cfg.get("right_edge_smoothing_mode", "right_blend")),
                    steps=int(right_smooth_cfg.get("right_edge_smoothing_steps", 2)),
                    blend_betas=list(right_smooth_cfg.get("right_blend_betas", [0.2, 0.5])),
                    right_local_band=float(right_smooth_cfg.get("right_local_band_ft", 200.0)),
                )
                cond_cfg = smoothing_cfg.get("conditional_rightstep2_fuse", {}) if isinstance(smoothing_cfg, dict) else {}
                seg_b_fused, _, _, _, _ = _apply_conditional_rightstep2_fuse(
                    seg_b_sm_r,
                    segment_bucket=str(_segment_bucket3(float(f["fill_minutes"]))),
                    anchor_pattern=str(_anchor_pattern(str(f["left_block_type"]), str(f["right_block_type"]))),
                    right_boundary_alt=float(f.get("right_boundary_alt", seg_b["pred_alt"].iloc[-1])),
                    enabled=bool(cond_cfg.get("use_conditional_rightstep2_fuse", False)),
                    target_bucket=str(cond_cfg.get("conditional_target_bucket", "medium")),
                    target_pattern=str(cond_cfg.get("conditional_target_pattern", "two_anchor")),
                    tau_jump=float(cond_cfg.get("tau_jump", 200.0)),
                    mode=str(cond_cfg.get("fuse_mode", "local_interp")),
                    fuse_lambda=float(cond_cfg.get("fuse_lambda", 0.5)),
                    use_second_condition=bool(cond_cfg.get("use_second_condition", False)),
                    tau_curve=float(cond_cfg.get("tau_curve", 200.0)),
                    right_local_band=float(cond_cfg.get("right_local_band_ft", 200.0)),
                )
                seg_b["pred_alt"] = seg_b_fused
                pred_fill_seg_b.append(seg_b)
        pred_fill_a_all = pd.concat(pred_fill_seg_a, ignore_index=True) if pred_fill_seg_a else pd.DataFrame()
        pred_fill_b_all = pd.concat(pred_fill_seg_b, ignore_index=True) if pred_fill_seg_b else pd.DataFrame()

        if len(adsb_all):
            window_start = adsb_all["minute_ts"].min()
            window_end = adsb_all["minute_ts"].max()
        else:
            window_start = sa.anchor_start_time.floor("min")
            window_end = sa.anchor_end_time.floor("min")
        if len(adsc_raw):
            window_start = min(window_start, adsc_raw["timestamp"].min())
            window_end = max(window_end, adsc_raw["timestamp"].max())

        blocks = _build_known_blocks(adsb_all, adsc_raw, window_start, window_end, gap_break_min=10.0)
        fills = _build_fill_intervals(blocks, min_gap_min=2.0)
        for bi, b in enumerate(blocks, start=1):
            known_blocks_rows.append(
                {
                    "sample_id": sa.sample_id,
                    "block_id": f"b{bi:02d}",
                    "block_type": b["block_type"],
                    "start_time": b["start_time"].isoformat(),
                    "end_time": b["end_time"].isoformat(),
                }
            )
        for fi, f in enumerate(fills, start=1):
            fill_rows.append(
                {
                    "sample_id": sa.sample_id,
                    "fill_id": f"f{fi:02d}",
                    "left_block_type": f["left_block_type"],
                    "right_block_type": f["right_block_type"],
                    "fill_start_time": f["fill_start_time"].isoformat(),
                    "fill_end_time": f["fill_end_time"].isoformat(),
                    "fill_minutes": f["fill_minutes"],
                }
            )
        for bi, b in enumerate(blocks, start=1):
            known_blocks_after_rows.append(
                {
                    "sample_id": sa.sample_id,
                    "block_id": f"b{bi:02d}",
                    "block_type": b["block_type"],
                    "start_time": b["start_time"].isoformat(),
                    "end_time": b["end_time"].isoformat(),
                }
            )
        for fi, f in enumerate(fills_meta, start=1):
            fill_after_rows.append(
                {
                    "sample_id": sa.sample_id,
                    "fill_id": f["fill_id"],
                    "left_block_type": f["left_block_type"],
                    "right_block_type": f["right_block_type"],
                    "fill_start_time": f["fill_start_time"].isoformat(),
                    "fill_end_time": f["fill_end_time"].isoformat(),
                    "fill_minutes": f["fill_minutes"],
                }
            )
            plot_after_rows.append(
                {
                    "sample_id": sa.sample_id,
                    "segment_id": f["fill_id"],
                    "segment_type": f"{f['left_block_type']}->{f['right_block_type']}",
                    "start_time": f["fill_start_time"].isoformat(),
                    "end_time": f["fill_end_time"].isoformat(),
                    "plotted_flag": True,
                    "drop_reason": "",
                }
            )

        _plot_raw_full(sa, adsb_all, adsc_raw, raw_full)
        _plot_raw_boundary(sa, adsb_all, adsc_raw, raw_bnd)
        if len(pred_fill_b_all):
            _plot_full(sb, pred_fill_b_all, adsb_all, adsc_raw, fills_meta, rec_full)
            _plot_boundary(sb, pred_fill_b_all, adsb_all, adsc_raw, bnd_b_map.get(sb.sample_id, {}), rec_bnd)
        if len(pred_fill_a_all) and len(pred_fill_b_all):
            _plot_aux_compare(sa, pred_fill_a_all, pred_fill_b_all, adsb_all, compare_aux)
        if len(psa) and len(psc):
            _plot_left_context_compare(
                sample_meta=sc,
                pred_base=psa,
                pred_left_aug=psc,
                frame_left_aug=sc.frame,
                out_dir=compare_left_ctx,
            )

    # Segment-level audit and bucket statistics (non-intrusive, export-only).
    seg_audit_a, seg_points_a = _build_segment_level_audit(
        samples=samples_a,
        fill_meta_by_sample=fill_meta_by_sample,
        pred_fill_on=pred_fill_a,
        pred_fill_off=pred_fill_main_a if len(pred_fill_main_a) else pred_fill_a,
        segment_policy=segment_policy,
        task_type="pure_adsc",
        projection_cfg=projection_cfg,
        smoothing_cfg=smoothing_cfg,
        replay_alt_series_source=replay_alt_series_source,
    )
    seg_audit_b, seg_points_b = _build_segment_level_audit(
        samples=samples_b,
        fill_meta_by_sample=fill_meta_by_sample,
        pred_fill_on=pred_fill_b,
        pred_fill_off=pred_fill_main_b if len(pred_fill_main_b) else pred_fill_b,
        segment_policy=segment_policy,
        task_type="adsc_plus_local_adsb",
        projection_cfg=projection_cfg,
        smoothing_cfg=smoothing_cfg,
        replay_alt_series_source=replay_alt_series_source,
    )
    seg_audit_c, seg_points_c = _build_segment_level_audit(
        samples=samples_c,
        fill_meta_by_sample=fill_meta_by_sample,
        pred_fill_on=pred_fill_c,
        pred_fill_off=pred_fill_main_c if len(pred_fill_main_c) else pred_fill_c,
        segment_policy=segment_policy,
        task_type="adsc_plus_left_adsb",
        projection_cfg=projection_cfg,
        smoothing_cfg=smoothing_cfg,
        replay_alt_series_source=replay_alt_series_source,
    )
    segment_audit_df = pd.concat([seg_audit_a, seg_audit_b, seg_audit_c], ignore_index=True) if (len(seg_audit_a) or len(seg_audit_b) or len(seg_audit_c)) else pd.DataFrame()
    segment_point_df = pd.concat([seg_points_a, seg_points_b, seg_points_c], ignore_index=True) if (len(seg_points_a) or len(seg_points_b) or len(seg_points_c)) else pd.DataFrame()
    segment_audit_path = out_dir / "production_chain_v1_segment_audit.csv"
    if len(segment_audit_df):
        segment_audit_df.to_csv(segment_audit_path, index=False)
    else:
        pd.DataFrame(
            columns=[
                "segment_id",
                "sample_id",
                "fill_id",
                "task_type",
                "flight_id",
                "segment_len",
                "segment_bucket",
                "anchor_pattern",
                "risk_flag",
                "risk_level",
                "matched_risk_rule",
                "risk_flag_teacher",
                "teacher_scale",
                "residual_rmax_ft",
                "policy_mode",
                "policy_scale",
                "gate_mean",
                "bounded_residual_mean_abs",
                "bounded_residual_max_abs",
                "delta_candidate_mean_abs",
                "delta_used_mean_abs",
                "left_edge_wrong_direction_ratio",
                "abnormal_flag",
                "warn_flag",
                "keep_flag",
                "overshoot_flag",
                "edge_spike_flag",
            ]
        ).to_csv(segment_audit_path, index=False)

    # Overshoot specialized audit exports (reference/position/cause decomposition).
    _export_overshoot_audit_tables(
        segment_df=segment_audit_df,
        point_df=segment_point_df,
        out_dir=out_dir,
        experiment_name=(Path(out_dir).parent.name if Path(out_dir).name == "replay" else Path(out_dir).name),
    )
    _export_edge_spike_audit_tables(
        segment_df=segment_audit_df,
        point_df=segment_point_df,
        out_dir=out_dir,
        experiment_name=(Path(out_dir).parent.name if Path(out_dir).name == "replay" else Path(out_dir).name),
    )
    _export_reference_consistency_audit(
        segment_df=segment_audit_df,
        point_df=segment_point_df,
        out_dir=out_dir,
        experiment_name=(Path(out_dir).parent.name if Path(out_dir).name == "replay" else Path(out_dir).name),
    )

    # Keep / warn / abnormal summary.
    if len(segment_audit_df):
        total_n = int(len(segment_audit_df))
        keep_n = int(segment_audit_df["keep_flag"].sum())
        warn_n = int(segment_audit_df["warn_flag"].sum())
        abnormal_n = int(segment_audit_df["abnormal_flag"].sum())
        summary_df = pd.DataFrame(
            [
                {"label": "keep", "count": keep_n, "ratio": keep_n / max(1, total_n)},
                {"label": "warn", "count": warn_n, "ratio": warn_n / max(1, total_n)},
                {"label": "abnormal", "count": abnormal_n, "ratio": abnormal_n / max(1, total_n)},
            ]
        )
    else:
        summary_df = pd.DataFrame(columns=["label", "count", "ratio"])
    summary_df.to_csv(out_dir / "production_chain_v1_keep_warn_abnormal_summary.csv", index=False)

    by_bucket_df = _safe_group_stats(segment_audit_df, "segment_bucket")
    by_bucket_df.to_csv(out_dir / "production_chain_v1_by_segment_bucket.csv", index=False)
    by_pattern_df = _safe_group_stats(segment_audit_df, "anchor_pattern")
    by_pattern_df.to_csv(out_dir / "production_chain_v1_by_anchor_pattern.csv", index=False)
    by_risk_df = _safe_group_stats(segment_audit_df, "risk_flag")
    by_risk_df.to_csv(out_dir / "production_chain_v1_by_risk_flag.csv", index=False)
    by_risk_level_df = _safe_group_stats(segment_audit_df, "risk_level")
    by_risk_level_df.to_csv(out_dir / "production_chain_v1_by_risk_level.csv", index=False)
    by_rule_df = _safe_group_stats(segment_audit_df, "matched_risk_rule")
    by_rule_df.to_csv(out_dir / "production_chain_v1_by_matched_risk_rule.csv", index=False)

    # Top-N remaining high-risk modes.
    if len(segment_audit_df):
        gp = (
            segment_audit_df.groupby(["segment_bucket", "anchor_pattern"], as_index=False)
            .agg(
                count=("segment_id", "count"),
                abnormal_ratio=("abnormal_flag", "mean"),
                warn_ratio=("warn_flag", "mean"),
                edge_spike_ratio=("edge_spike_flag", "mean"),
                overshoot_ratio=("overshoot_flag", "mean"),
            )
            .sort_values(["abnormal_ratio", "warn_ratio", "edge_spike_ratio"], ascending=[False, False, False])
        )
        top_n = max(1, int(min(10, len(gp))))
        top_abn = gp.sort_values(["abnormal_ratio", "count"], ascending=[False, False]).head(top_n).copy()
        top_abn["rank_type"] = "top_abnormal"
        top_warn = gp.sort_values(["warn_ratio", "count"], ascending=[False, False]).head(top_n).copy()
        top_warn["rank_type"] = "top_warn"
        top_edge = gp.sort_values(["edge_spike_ratio", "count"], ascending=[False, False]).head(top_n).copy()
        top_edge["rank_type"] = "top_edge_spike"
        risk_top_df = pd.concat([top_abn, top_warn, top_edge], ignore_index=True)
    else:
        risk_top_df = pd.DataFrame(
            columns=[
                "segment_bucket",
                "anchor_pattern",
                "count",
                "abnormal_ratio",
                "warn_ratio",
                "edge_spike_ratio",
                "overshoot_ratio",
                "rank_type",
            ]
        )
    risk_top_df.to_csv(out_dir / "production_chain_v1_failure_mode_topN.csv", index=False)

    # Stratified case export for manual review.
    warn_cases = _stratified_pick(segment_audit_df, "warn_flag", n=20, group_cols=["segment_bucket", "anchor_pattern"])
    abnormal_cases = _stratified_pick(segment_audit_df, "abnormal_flag", n=20, group_cols=["segment_bucket", "anchor_pattern"])
    warn_cases.to_csv(out_dir / "production_chain_v1_warn_samples_20.csv", index=False)
    abnormal_cases.to_csv(out_dir / "production_chain_v1_abnormal_samples_20.csv", index=False)

    # Summary markdown.
    summary = [
        "# Real ADS-C Replay + Local ADS-B Context Summary",
        "",
        f"- config: `{args.config}`",
        f"- checkpoint: `{ckpt}`",
        f"- adsc_dir: `{adsc_dir}`",
        f"- total_valid_samples: {len(samples_a)}",
        "",
        "## Key Findings",
        "",
        "1. Pure ADS-C replay can be evaluated by anchor-consistency, trajectory shape plausibility, and boundary continuity; "
        "it should not be interpreted as gap-inner minute-level truth error.",
        "2. Both experiments enforce hard ADS-C anchor consistency; recovered paths are constrained to pass the real start/end ADS-C anchors.",
        "3. Experiment B differs from A only by adding pre/post-gap (5-minute) ADS-B context; no gap-inner ADS-B is used.",
        "4. Experiment C differs from A by adding only left-side (pre-anchor) 5-minute ADS-B context; no right-side context and no gap-inner ADS-B is used.",
        "5. Boundary consistency metrics focus on heading/speed/vertical-slope continuity against pre/post ADS-B segments.",
        "",
        "## Important Evaluation Constraint",
        "",
        "- Cross-ocean gap inner minute-level MAE/RMSE is **not reported** because there is no guaranteed gap-inner truth in real ADS-C replay.",
        "- Reported metrics are qualitative trajectory plausibility and boundary-neighborhood consistency.",
        "",
        "## Output Files",
        "",
        "- `real_adsc_replay_data_audit.csv`",
        "- `real_adsc_replay_qualitative_eval.csv`",
        "- `real_adsc_plus_local_adsb_qualitative_eval.csv`",
        "- `real_adsc_plus_left_adsb_qualitative_eval.csv`",
        "- `real_adsc_replay_boundary_consistency_eval.csv`",
        "- `real_adsc_plus_local_adsb_boundary_consistency_eval.csv`",
        "- `real_adsc_plus_left_adsb_boundary_consistency_eval.csv`",
        "- `left_context_coverage_stats.csv`",
        "- `bucket_stats_plus_left_adsb_by_*.csv`",
        "- `plots/raw_fused/full_recovery/`",
        "- `plots/raw_fused/boundary_consistency/`",
        "- `plots/recovered/full_recovery/`",
        "- `plots/recovered/boundary_consistency/`",
        "- `plots/compare_left_context_5min/`",
    ]
    (out_dir / "real_adsc_replay_and_local_adsb_context_summary.md").write_text("\n".join(summary), encoding="utf-8")
    if known_blocks_rows:
        pd.DataFrame(known_blocks_rows).to_csv(out_dir / "known_blocks.csv", index=False)
    if fill_rows:
        pd.DataFrame(fill_rows).to_csv(out_dir / "fill_intervals.csv", index=False)
    if known_blocks_after_rows:
        pd.DataFrame(known_blocks_after_rows).to_csv(out_dir / "known_blocks_after_refactor.csv", index=False)
    if fill_after_rows:
        pd.DataFrame(fill_after_rows).to_csv(out_dir / "fill_intervals_after_refactor.csv", index=False)
    if recovered_after_rows:
        pd.DataFrame(recovered_after_rows).to_csv(out_dir / "recovered_segments_after_refactor.csv", index=False)
    if plot_after_rows:
        pd.DataFrame(plot_after_rows).to_csv(out_dir / "plot_segments_after_refactor.csv", index=False)
    if segment_policy_rows:
        pd.DataFrame(segment_policy_rows).to_csv(out_dir / "segment_policy_decisions.csv", index=False)

    print(f"[ok] audit={audit_path}")
    print(f"[ok] qualitative_A={out_dir / 'real_adsc_replay_qualitative_eval.csv'}")
    print(f"[ok] qualitative_B={out_dir / 'real_adsc_plus_local_adsb_qualitative_eval.csv'}")
    print(f"[ok] qualitative_C={out_dir / 'real_adsc_plus_left_adsb_qualitative_eval.csv'}")
    print(f"[ok] boundary_A={out_dir / 'real_adsc_replay_boundary_consistency_eval.csv'}")
    print(f"[ok] boundary_B={out_dir / 'real_adsc_plus_local_adsb_boundary_consistency_eval.csv'}")
    print(f"[ok] boundary_C={out_dir / 'real_adsc_plus_left_adsb_boundary_consistency_eval.csv'}")
    print(f"[ok] left_context_coverage={out_dir / 'left_context_coverage_stats.csv'}")
    print(f"[ok] plots_raw_full={raw_full}")
    print(f"[ok] plots_raw_bnd={raw_bnd}")
    print(f"[ok] plots_rec_full={rec_full}")
    print(f"[ok] plots_rec_bnd={rec_bnd}")
    print(f"[ok] segment_audit={segment_audit_path}")
    print(f"[ok] summary_md={out_dir / 'real_adsc_replay_and_local_adsb_context_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
