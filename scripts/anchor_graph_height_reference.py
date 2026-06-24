from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AnchorGraphParams:
    small_delta_m: float = 60.0
    step_delta_m: float = 120.0
    stable_std_m: float = 45.0
    context_tol_m: float = 90.0
    min_step_gap_min: int = 8
    transition_m_per_min: float = 150.0
    min_transition_min: int = 2
    max_transition_min: int = 8
    context_radius: int = 2
    soft_temperature: float = 0.75


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _nanmedian_or(values: np.ndarray, fallback: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float(fallback)
    return float(np.median(values))


def _nanstd(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) <= 1:
        return 0.0
    return float(np.std(values))


def _trend_consistent(anchor_alt: np.ndarray, center_edge: int, params: AnchorGraphParams) -> bool:
    lo = max(0, center_edge - params.context_radius)
    hi = min(len(anchor_alt) - 1, center_edge + params.context_radius + 1)
    dz = np.diff(anchor_alt[lo : hi + 1])
    dz = dz[np.isfinite(dz)]
    if len(dz) < 2:
        return False
    # A trend gap should not mix strong positive and negative level changes.
    return bool(np.all(dz >= -params.small_delta_m) or np.all(dz <= params.small_delta_m))


def _infer_step_center(raw: np.ndarray, z_left: float, z_right: float, n: int) -> int:
    midpoint = (float(z_left) + float(z_right)) / 2.0
    fallback = max(1, min(n - 2, n // 2))
    raw = np.asarray(raw, dtype=float)
    if len(raw) != n or not np.isfinite(raw).any():
        return int(fallback)
    finite = np.where(np.isfinite(raw), raw, np.nan)
    if z_right >= z_left:
        candidates = np.where(finite >= midpoint)[0]
    else:
        candidates = np.where(finite <= midpoint)[0]
    candidates = candidates[(candidates > 0) & (candidates < n - 1)]
    if len(candidates) > 0:
        return int(candidates[0])
    return int(np.nanargmin(np.abs(finite - midpoint))) if np.isfinite(finite).any() else int(fallback)


def _step_profile(z_left: float, z_right: float, n: int, raw: np.ndarray, params: AnchorGraphParams) -> np.ndarray:
    dz = float(z_right - z_left)
    center = _infer_step_center(raw=raw, z_left=z_left, z_right=z_right, n=n)
    duration = int(np.clip(round(abs(dz) / float(params.transition_m_per_min)), params.min_transition_min, params.max_transition_min))
    start = int(np.clip(center - duration // 2, 1, max(1, n - duration - 1)))
    end = int(np.clip(start + duration, start + 1, n - 1))
    out = np.full(n, float(z_left), dtype=float)
    u = np.linspace(0.0, 1.0, end - start + 1)
    out[start : end + 1] = float(z_left) + dz * smoothstep(u)
    out[end + 1 :] = float(z_right)
    out[0] = float(z_left)
    out[-1] = float(z_right)
    return out


def build_reference_candidates(
    z_left: float,
    z_right: float,
    n: int,
    raw: np.ndarray | None = None,
    params: AnchorGraphParams | None = None,
) -> dict[str, np.ndarray]:
    """Build candidate reference curves for one bounded gap.

    The candidates all satisfy endpoint consistency. They are deliberately simple:
    trend is the old local linear assumption, switch represents level-change behavior,
    and hold is a conservative stable-level candidate that degenerates to trend when
    the endpoint altitude difference is small.
    """
    params = params or AnchorGraphParams()
    raw_arr = np.asarray(raw if raw is not None else np.full(n, np.nan), dtype=float)
    trend = np.linspace(float(z_left), float(z_right), int(n))
    switch = _step_profile(float(z_left), float(z_right), int(n), raw_arr, params)
    if abs(float(z_right) - float(z_left)) <= params.small_delta_m:
        hold = trend.copy()
    else:
        # Conservative hold candidate: keep the first level for most of the interval,
        # then satisfy the right anchor with a short boundary transition.
        hold = _step_profile(float(z_left), float(z_right), int(n), np.full(int(n), float(z_left)), params)
    return {"hold": hold, "switch": switch, "trend": trend}


def anchor_graph_gap_features(
    anchor_alt: np.ndarray,
    edge_i: int,
    gap_len: int,
    z_left: float,
    z_right: float,
    params: AnchorGraphParams | None = None,
) -> dict[str, float]:
    params = params or AnchorGraphParams()
    anchor_alt = np.asarray(anchor_alt, dtype=float)
    left_ctx = anchor_alt[max(0, edge_i - params.context_radius) : edge_i + 1]
    right_ctx = anchor_alt[edge_i + 1 : min(len(anchor_alt), edge_i + 2 + params.context_radius)]
    prev_delta = float(anchor_alt[edge_i] - anchor_alt[edge_i - 1]) if edge_i >= 1 else 0.0
    next_delta = float(anchor_alt[edge_i + 2] - anchor_alt[edge_i + 1]) if edge_i + 2 < len(anchor_alt) else 0.0
    return {
        "anchor_count": float(len(anchor_alt)),
        "gap_len": float(gap_len),
        "delta_z": float(z_right - z_left),
        "abs_delta_z": float(abs(z_right - z_left)),
        "left_context_std": float(_nanstd(left_ctx)),
        "right_context_std": float(_nanstd(right_ctx)),
        "left_context_median_delta": float(z_left - _nanmedian_or(left_ctx, z_left)),
        "right_context_median_delta": float(z_right - _nanmedian_or(right_ctx, z_right)),
        "prev_delta_z": float(prev_delta),
        "next_delta_z": float(next_delta),
        "gap_index_ratio": float(edge_i / max(len(anchor_alt) - 2, 1)),
    }


def _edge_profiles(z_left: float, z_right: float, n: int, raw: np.ndarray, params: AnchorGraphParams) -> dict[str, np.ndarray]:
    trend = np.linspace(float(z_left), float(z_right), int(n))
    switch = _step_profile(z_left=z_left, z_right=z_right, n=int(n), raw=raw, params=params)
    # Hold is a near-level reference. For small endpoint differences it is almost
    # identical to trend; for large differences the soft gate should suppress it.
    hold = np.full(int(n), float(z_left), dtype=float)
    if int(n) >= 2:
        hold[-1] = float(z_right)
        if abs(float(z_right - z_left)) <= params.small_delta_m:
            hold = trend.copy()
    return {"hold": hold, "switch": switch, "trend": trend}


def _softmax_scores(scores: dict[str, float], temperature: float) -> dict[str, float]:
    keys = list(scores.keys())
    vals = np.array([float(scores[k]) for k in keys], dtype=float) / max(float(temperature), 1e-6)
    vals = vals - np.max(vals)
    exp = np.exp(vals)
    weights = exp / np.sum(exp)
    return {k: float(w) for k, w in zip(keys, weights)}


def _gate_scores(
    dz: float,
    gap_len: int,
    left_std: float,
    right_std: float,
    left_stable: bool,
    right_stable: bool,
    trend_consistent: bool,
    params: AnchorGraphParams,
) -> dict[str, float]:
    abs_dz = abs(float(dz))
    stable_bonus = float(left_stable) + float(right_stable)
    context_std = 0.5 * (float(left_std) + float(right_std))
    small = max(float(params.small_delta_m), 1e-6)
    step = max(float(params.step_delta_m), 1e-6)
    # Scores are intentionally simple and auditable. They can later be replaced
    # by a trainable MLP with the same inputs.
    hold_score = 2.0 - abs_dz / small - context_std / max(float(params.stable_std_m), 1e-6)
    switch_score = (
        abs_dz / step
        + 0.8 * stable_bonus
        + 0.4 * float(gap_len >= params.min_step_gap_min)
        - 0.7 * float(trend_consistent)
    )
    trend_score = 0.8 * float(trend_consistent) + 0.8 * min(abs_dz / step, 2.0) - 0.3 * stable_bonus
    return {"hold": hold_score, "switch": switch_score, "trend": trend_score}


def build_anchor_graph_reference(
    time_index: np.ndarray,
    anchor_mask: np.ndarray,
    anchor_alt_or_truth: np.ndarray,
    raw_alt: np.ndarray | None = None,
    params: AnchorGraphParams | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Generate a flight-level anchor-graph height reference.

    Only observed anchor positions and the optional raw model curve are used. Hidden truth
    at non-anchor timestamps must not be passed as anchors.
    """
    params = params or AnchorGraphParams()
    t = np.asarray(time_index, dtype=float)
    anchor_mask = np.asarray(anchor_mask, dtype=bool)
    alt = np.asarray(anchor_alt_or_truth, dtype=float)
    raw = np.asarray(raw_alt if raw_alt is not None else np.full_like(alt, np.nan), dtype=float)
    ref = np.asarray(raw, dtype=float).copy()
    if ref.shape != alt.shape:
        ref = np.full_like(alt, np.nan, dtype=float)
    anchors = np.where(anchor_mask & np.isfinite(alt))[0]
    decisions: list[dict] = []
    if len(anchors) < 2:
        return ref, decisions
    anchor_alt = alt[anchors]
    for edge_i, (left, right) in enumerate(zip(anchors[:-1], anchors[1:])):
        n = int(right - left + 1)
        if n <= 1:
            continue
        z_left = float(alt[left])
        z_right = float(alt[right])
        dz = z_right - z_left
        gap_len = int(right - left - 1)
        left_ctx = anchor_alt[max(0, edge_i - params.context_radius) : edge_i + 1]
        right_ctx = anchor_alt[edge_i + 1 : min(len(anchor_alt), edge_i + 2 + params.context_radius)]
        left_med = _nanmedian_or(left_ctx, z_left)
        right_med = _nanmedian_or(right_ctx, z_right)
        left_stable = (_nanstd(left_ctx) <= params.stable_std_m) and (abs(z_left - left_med) <= params.context_tol_m)
        right_stable = (_nanstd(right_ctx) <= params.stable_std_m) and (abs(z_right - right_med) <= params.context_tol_m)
        mode = "trend"
        candidates = build_reference_candidates(z_left, z_right, n, raw[left : right + 1], params)
        if abs(dz) <= params.small_delta_m:
            mode = "hold"
            profile = candidates["hold"]
        elif (
            abs(dz) >= params.step_delta_m
            and gap_len >= params.min_step_gap_min
            and left_stable
            and right_stable
            and abs(left_med - right_med) >= params.step_delta_m
        ):
            mode = "level_switch"
            profile = candidates["switch"]
        elif _trend_consistent(anchor_alt, edge_i, params):
            mode = "global_trend"
            profile = candidates["trend"]
        else:
            mode = "local_linear_fallback"
            profile = candidates["trend"]
        ref[left : right + 1] = profile
        decisions.append(
            {
                "left_index": int(left),
                "right_index": int(right),
                "left_time": float(t[left]) if len(t) == len(alt) else float(left),
                "right_time": float(t[right]) if len(t) == len(alt) else float(right),
                "gap_len": int(gap_len),
                "z_left_m": float(z_left),
                "z_right_m": float(z_right),
                "delta_z_m": float(dz),
                "left_context_median_m": float(left_med),
                "right_context_median_m": float(right_med),
                "left_context_std_m": float(_nanstd(left_ctx)),
                "right_context_std_m": float(_nanstd(right_ctx)),
                "left_stable": bool(left_stable),
                "right_stable": bool(right_stable),
                "mode": mode,
            }
        )
    ref[anchors] = alt[anchors]
    return ref, decisions


def build_anchor_graph_soft_reference(
    time_index: np.ndarray,
    anchor_mask: np.ndarray,
    anchor_alt_or_truth: np.ndarray,
    raw_alt: np.ndarray | None = None,
    params: AnchorGraphParams | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Soft-gated version of the flight-level anchor-graph reference."""
    params = params or AnchorGraphParams()
    t = np.asarray(time_index, dtype=float)
    anchor_mask = np.asarray(anchor_mask, dtype=bool)
    alt = np.asarray(anchor_alt_or_truth, dtype=float)
    raw = np.asarray(raw_alt if raw_alt is not None else np.full_like(alt, np.nan), dtype=float)
    ref = np.asarray(raw, dtype=float).copy()
    if ref.shape != alt.shape:
        ref = np.full_like(alt, np.nan, dtype=float)
    anchors = np.where(anchor_mask & np.isfinite(alt))[0]
    decisions: list[dict] = []
    if len(anchors) < 2:
        return ref, decisions
    anchor_alt = alt[anchors]
    for edge_i, (left, right) in enumerate(zip(anchors[:-1], anchors[1:])):
        n = int(right - left + 1)
        if n <= 1:
            continue
        z_left = float(alt[left])
        z_right = float(alt[right])
        dz = z_right - z_left
        gap_len = int(right - left - 1)
        left_ctx = anchor_alt[max(0, edge_i - params.context_radius) : edge_i + 1]
        right_ctx = anchor_alt[edge_i + 1 : min(len(anchor_alt), edge_i + 2 + params.context_radius)]
        left_med = _nanmedian_or(left_ctx, z_left)
        right_med = _nanmedian_or(right_ctx, z_right)
        left_std = _nanstd(left_ctx)
        right_std = _nanstd(right_ctx)
        left_stable = (left_std <= params.stable_std_m) and (abs(z_left - left_med) <= params.context_tol_m)
        right_stable = (right_std <= params.stable_std_m) and (abs(z_right - right_med) <= params.context_tol_m)
        trend_consistent = _trend_consistent(anchor_alt, edge_i, params)
        profiles = _edge_profiles(z_left, z_right, n, raw[left : right + 1], params)
        scores = _gate_scores(
            dz=dz,
            gap_len=gap_len,
            left_std=left_std,
            right_std=right_std,
            left_stable=left_stable,
            right_stable=right_stable,
            trend_consistent=trend_consistent,
            params=params,
        )
        weights = _softmax_scores(scores, temperature=params.soft_temperature)
        profile = (
            weights["hold"] * profiles["hold"]
            + weights["switch"] * profiles["switch"]
            + weights["trend"] * profiles["trend"]
        )
        ref[left : right + 1] = profile
        mode = max(weights, key=weights.get)
        decisions.append(
            {
                "left_index": int(left),
                "right_index": int(right),
                "left_time": float(t[left]) if len(t) == len(alt) else float(left),
                "right_time": float(t[right]) if len(t) == len(alt) else float(right),
                "gap_len": int(gap_len),
                "z_left_m": float(z_left),
                "z_right_m": float(z_right),
                "delta_z_m": float(dz),
                "left_context_median_m": float(left_med),
                "right_context_median_m": float(right_med),
                "left_context_std_m": float(left_std),
                "right_context_std_m": float(right_std),
                "left_stable": bool(left_stable),
                "right_stable": bool(right_stable),
                "trend_consistent": bool(trend_consistent),
                "score_hold": float(scores["hold"]),
                "score_switch": float(scores["switch"]),
                "score_trend": float(scores["trend"]),
                "weight_hold": float(weights["hold"]),
                "weight_switch": float(weights["switch"]),
                "weight_trend": float(weights["trend"]),
                "mode": f"soft_{mode}",
            }
        )
    ref[anchors] = alt[anchors]
    return ref, decisions


def build_anchor_graph_score_reference(
    time_index: np.ndarray,
    anchor_mask: np.ndarray,
    anchor_alt_or_truth: np.ndarray,
    raw_alt: np.ndarray | None = None,
    params: AnchorGraphParams | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Score-gated one-hot reference using the same scores as soft gating."""
    params = params or AnchorGraphParams()
    t = np.asarray(time_index, dtype=float)
    anchor_mask = np.asarray(anchor_mask, dtype=bool)
    alt = np.asarray(anchor_alt_or_truth, dtype=float)
    raw = np.asarray(raw_alt if raw_alt is not None else np.full_like(alt, np.nan), dtype=float)
    ref = np.asarray(raw, dtype=float).copy()
    if ref.shape != alt.shape:
        ref = np.full_like(alt, np.nan, dtype=float)
    anchors = np.where(anchor_mask & np.isfinite(alt))[0]
    decisions: list[dict] = []
    if len(anchors) < 2:
        return ref, decisions
    anchor_alt = alt[anchors]
    for edge_i, (left, right) in enumerate(zip(anchors[:-1], anchors[1:])):
        n = int(right - left + 1)
        if n <= 1:
            continue
        z_left = float(alt[left])
        z_right = float(alt[right])
        dz = z_right - z_left
        gap_len = int(right - left - 1)
        left_ctx = anchor_alt[max(0, edge_i - params.context_radius) : edge_i + 1]
        right_ctx = anchor_alt[edge_i + 1 : min(len(anchor_alt), edge_i + 2 + params.context_radius)]
        left_med = _nanmedian_or(left_ctx, z_left)
        right_med = _nanmedian_or(right_ctx, z_right)
        left_std = _nanstd(left_ctx)
        right_std = _nanstd(right_ctx)
        left_stable = (left_std <= params.stable_std_m) and (abs(z_left - left_med) <= params.context_tol_m)
        right_stable = (right_std <= params.stable_std_m) and (abs(z_right - right_med) <= params.context_tol_m)
        trend_consistent = _trend_consistent(anchor_alt, edge_i, params)
        profiles = _edge_profiles(z_left, z_right, n, raw[left : right + 1], params)
        scores = _gate_scores(
            dz=dz,
            gap_len=gap_len,
            left_std=left_std,
            right_std=right_std,
            left_stable=left_stable,
            right_stable=right_stable,
            trend_consistent=trend_consistent,
            params=params,
        )
        weights = _softmax_scores(scores, temperature=params.soft_temperature)
        selected = max(scores, key=scores.get)
        ref[left : right + 1] = profiles[selected]
        decisions.append(
            {
                "left_index": int(left),
                "right_index": int(right),
                "left_time": float(t[left]) if len(t) == len(alt) else float(left),
                "right_time": float(t[right]) if len(t) == len(alt) else float(right),
                "gap_len": int(gap_len),
                "z_left_m": float(z_left),
                "z_right_m": float(z_right),
                "delta_z_m": float(dz),
                "left_context_median_m": float(left_med),
                "right_context_median_m": float(right_med),
                "left_context_std_m": float(left_std),
                "right_context_std_m": float(right_std),
                "left_stable": bool(left_stable),
                "right_stable": bool(right_stable),
                "trend_consistent": bool(trend_consistent),
                "score_hold": float(scores["hold"]),
                "score_switch": float(scores["switch"]),
                "score_trend": float(scores["trend"]),
                "weight_hold": float(weights["hold"]),
                "weight_switch": float(weights["switch"]),
                "weight_trend": float(weights["trend"]),
                "mode": f"score_{selected}",
            }
        )
    ref[anchors] = alt[anchors]
    return ref, decisions
