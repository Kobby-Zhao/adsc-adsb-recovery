from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _length_bucket(fill_minutes: float) -> str:
    m = float(fill_minutes)
    if m <= 15:
        return "<=15"
    if m <= 60:
        return "15-60"
    return ">60"


def _edge_weights(n: int, edge_steps: int, edge_damp: float) -> np.ndarray:
    w = np.ones(max(0, int(n)), dtype=float)
    if n <= 0:
        return w
    k = max(0, min(int(edge_steps), n // 2))
    d = float(max(0.0, min(1.0, edge_damp)))
    for i in range(k):
        w[i] = min(w[i], d)
        w[n - 1 - i] = min(w[n - 1 - i], d)
    return w


def _resolve_policy_row(policy_table: list[dict[str, Any]], segment_type: str, length_bucket: str) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "residual_mode": "scale",
        "residual_scale": 1.0,
        "edge_mode": "none",
        "edge_steps": 2,
        "edge_damp": 0.3,
        "clip_value": 250.0,
        "output_weight": 1.0,
    }
    for row in policy_table or []:
        s = str(row.get("segment_type", "*"))
        b = str(row.get("length_bucket", "*"))
        if s not in {"*", segment_type}:
            continue
        if b not in {"*", length_bucket}:
            continue
        cfg.update(row)
    return cfg


@dataclass
class SegmentPolicyMeta:
    segment_type: str
    length_bucket: str
    residual_enabled: bool
    residual_scale: float
    edge_mode: str
    clip_value: float
    output_weight: float
    left_anchor_complete: bool
    right_anchor_complete: bool
    off_std: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_type": self.segment_type,
            "length_bucket": self.length_bucket,
            "residual_enabled": self.residual_enabled,
            "residual_scale": self.residual_scale,
            "edge_mode": self.edge_mode,
            "clip_value": self.clip_value,
            "output_weight": self.output_weight,
            "left_anchor_complete": self.left_anchor_complete,
            "right_anchor_complete": self.right_anchor_complete,
            "off_std": self.off_std,
        }


class SegmentResidualPolicy:
    """Table-driven segment-level altitude residual policy for V1.5 inference.

    This module is inference-only and does not alter model training behavior.
    """

    def __init__(
        self,
        enabled: bool = False,
        stable_std_threshold: float = 120.0,
        boundary_short_minutes: int = 15,
        policy_table: list[dict[str, Any]] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.stable_std_threshold = float(stable_std_threshold)
        self.boundary_short_minutes = int(boundary_short_minutes)
        self.policy_table = list(policy_table or [])

    def classify_segment(
        self,
        *,
        fill_minutes: float,
        left_block_type: str,
        right_block_type: str,
        alt_std_off: float,
    ) -> str:
        left_anchor = str(left_block_type) == "adsc_anchor"
        right_anchor = str(right_block_type) == "adsc_anchor"
        full_anchor = left_anchor and right_anchor
        short_seg = float(fill_minutes) <= float(self.boundary_short_minutes)
        disturbed = float(alt_std_off) > self.stable_std_threshold
        if short_seg or (not full_anchor):
            return "Boundary" if short_seg or ((not left_anchor) or (not right_anchor)) else "Disturbed"
        if disturbed:
            return "Disturbed"
        return "Stable"

    def apply(
        self,
        *,
        on_alt: np.ndarray,
        off_alt: np.ndarray,
        fill_minutes: float,
        left_block_type: str,
        right_block_type: str,
    ) -> tuple[np.ndarray, SegmentPolicyMeta]:
        if (not self.enabled) or len(on_alt) != len(off_alt):
            meta = SegmentPolicyMeta(
                segment_type="disabled",
                length_bucket=_length_bucket(fill_minutes),
                residual_enabled=True,
                residual_scale=1.0,
                edge_mode="none",
                clip_value=0.0,
                output_weight=1.0,
                left_anchor_complete=str(left_block_type) == "adsc_anchor",
                right_anchor_complete=str(right_block_type) == "adsc_anchor",
                off_std=float(np.nanstd(off_alt)) if len(off_alt) else 0.0,
            )
            return on_alt.copy(), meta

        off_std = float(np.nanstd(off_alt)) if len(off_alt) else 0.0
        seg_type = self.classify_segment(
            fill_minutes=float(fill_minutes),
            left_block_type=str(left_block_type),
            right_block_type=str(right_block_type),
            alt_std_off=off_std,
        )
        lb = _length_bucket(fill_minutes)
        cfg = _resolve_policy_row(self.policy_table, seg_type, lb)

        residual_mode = str(cfg.get("residual_mode", "scale")).lower()
        residual_scale = float(cfg.get("residual_scale", 1.0))
        edge_mode = str(cfg.get("edge_mode", "none")).lower()
        edge_steps = int(cfg.get("edge_steps", 2))
        edge_damp = float(cfg.get("edge_damp", 0.3))
        clip_value = float(cfg.get("clip_value", 250.0))
        output_weight = float(cfg.get("output_weight", 1.0))
        output_weight = max(0.0, min(1.0, output_weight))

        if residual_mode == "off":
            final = off_alt.copy()
            enabled = False
        else:
            delta = (on_alt - off_alt) * residual_scale
            if edge_mode == "soft_damp":
                delta = delta * _edge_weights(len(delta), edge_steps=edge_steps, edge_damp=edge_damp)
            elif edge_mode == "clip":
                delta = np.clip(delta, -clip_value, clip_value)
            elif edge_mode == "clip_damp":
                delta = delta * _edge_weights(len(delta), edge_steps=edge_steps, edge_damp=edge_damp)
                delta = np.clip(delta, -clip_value, clip_value)
            final = off_alt + output_weight * delta
            enabled = True

        meta = SegmentPolicyMeta(
            segment_type=seg_type,
            length_bucket=lb,
            residual_enabled=enabled,
            residual_scale=residual_scale if enabled else 0.0,
            edge_mode=edge_mode,
            clip_value=clip_value,
            output_weight=output_weight,
            left_anchor_complete=str(left_block_type) == "adsc_anchor",
            right_anchor_complete=str(right_block_type) == "adsc_anchor",
            off_std=off_std,
        )
        return final, meta
