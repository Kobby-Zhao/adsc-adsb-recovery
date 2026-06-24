from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

import numpy as np
import pandas as pd


@dataclass
class ADSCGapPatternSampler:
    min_gap_minutes: int = 3
    max_gap_minutes: int = 60
    force_endpoints: bool = True
    random_seed: int = 42
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.random_seed)

    def fit(self, adsc_df: pd.DataFrame, flight_id_col: str = "flight_id") -> dict:
        if adsc_df.empty:
            return {
                "anchor_count_distribution": [],
                "gap_distribution_minutes": [10],
                "position_accuracy_distribution": [],
                "tag_exist_rate": {"tag13_exists": 0.0, "tag14_exists": 0.0, "tag15_exists": 0.0, "tag16_exists": 0.0},
                "coverage": {"position_accuracy": 0.0},
            }

        df = adsc_df.sort_values([flight_id_col, "timestamp"]).copy()
        gaps = (
            df.groupby(flight_id_col)["timestamp"]
            .diff()
            .dt.total_seconds()
            .div(60.0)
            .dropna()
        )
        gaps = gaps[(gaps >= self.min_gap_minutes) & (gaps <= self.max_gap_minutes)]
        if gaps.empty:
            gaps = pd.Series([10.0])

        anchor_counts = df.groupby(flight_id_col).size().tolist()
        acc = pd.to_numeric(df.get("position_accuracy"), errors="coerce")

        tag_keys = ["tag13_exists", "tag14_exists", "tag15_exists", "tag16_exists"]
        tag_rates = {}
        for key in tag_keys:
            if key in df.columns:
                tag_rates[key] = float(pd.to_numeric(df[key], errors="coerce").fillna(0).mean())
            else:
                tag_rates[key] = 0.0

        return {
            "anchor_count_distribution": anchor_counts,
            "gap_distribution_minutes": [float(x) for x in gaps.tolist()],
            "position_accuracy_distribution": [float(x) for x in acc.dropna().tolist()],
            "tag_exist_rate": tag_rates,
            "coverage": {"position_accuracy": float(acc.notna().mean())},
        }

    def sample_anchor_mask(self, length: int, stats: dict) -> np.ndarray:
        mask = np.zeros(length, dtype=np.int64)
        if length <= 0:
            return mask

        gaps = stats.get("gap_distribution_minutes") or [10]
        pos = 0
        mask[pos] = 1
        while pos < length - 1:
            # Weighted sampling: longer gaps get higher selection probability
            # (α=0.5 strikes a balance between preserving the ADS-C shape and
            #  boosting long-tail coverage for stage3).
            gap_weights = np.array(gaps, dtype=np.float64) ** 0.5
            gap_weights = gap_weights / gap_weights.sum()
            step = int(round(float(self._rng.choice(gaps, p=gap_weights))))
            step = max(self.min_gap_minutes, min(self.max_gap_minutes, step))
            pos += step
            if pos < length:
                mask[pos] = 1

        if self.force_endpoints:
            mask[0] = 1
            mask[-1] = 1
        return mask

    def _sample_contiguous_gap_mask(
        self,
        length: int,
        mask_ratio: float,
        gap_buckets: list[tuple[int, int]],
    ) -> np.ndarray:
        mask = np.ones(length, dtype=np.int64)
        if length <= 0:
            return mask
        if self.force_endpoints and length >= 1:
            mask[0] = 1
            mask[-1] = 1
        if length <= 2:
            return mask

        ratio = float(np.clip(mask_ratio, 0.0, 0.95))
        target_missing = int(round(length * ratio))
        max_missing = max(0, length - 2) if self.force_endpoints else max(0, length - 1)
        target_missing = min(target_missing, max_missing)
        if target_missing <= 0:
            return mask

        buckets = [(max(1, int(a)), max(1, int(b))) for a, b in gap_buckets if int(b) >= int(a)]
        if not buckets:
            buckets = [(1, 3), (4, 8), (9, 15)]
        starts_min = 1 if self.force_endpoints else 0
        starts_max = length - 1 if self.force_endpoints else length

        missing_now = 0
        tries = 0
        while missing_now < target_missing and tries < 2000:
            tries += 1
            b_lo, b_hi = buckets[int(self._rng.integers(0, len(buckets)))]
            gap_len = int(self._rng.integers(b_lo, b_hi + 1))
            if gap_len <= 0:
                continue
            if starts_max - starts_min <= gap_len:
                continue
            start = int(self._rng.integers(starts_min, starts_max - gap_len + 1))
            end = start + gap_len
            if self.force_endpoints and (start <= 0 or end >= length):
                continue
            # Keep contiguous-gap semantics: skip if this segment already has missing values.
            if np.any(mask[start:end] == 0):
                continue
            mask[start:end] = 0
            missing_now = int((mask == 0).sum())

        # Fine-tune to hit target ratio as closely as possible without breaking endpoint anchors.
        if missing_now > target_missing:
            zero_idx = np.where(mask == 0)[0]
            self._rng.shuffle(zero_idx)
            for i in zero_idx[: max(0, missing_now - target_missing)]:
                mask[int(i)] = 1
        elif missing_now < target_missing:
            one_idx = np.where(mask == 1)[0]
            one_idx = one_idx[(one_idx > 0) & (one_idx < length - 1)] if self.force_endpoints else one_idx
            self._rng.shuffle(one_idx)
            for i in one_idx[: max(0, target_missing - missing_now)]:
                mask[int(i)] = 0

        # Guarantee at least two anchors.
        if int((mask == 1).sum()) < 2 and length >= 2:
            mask[0] = 1
            mask[-1] = 1
        return mask

    def simulate_observation(
        self,
        length: int,
        mode: str,
        stats: dict | None = None,
        stage1_cfg: dict | None = None,
        stage2_cfg: dict | None = None,
        stage2_medium_cfg: dict | None = None,
    ) -> np.ndarray:
        mode_n = str(mode).strip().lower()
        if mode_n in {"stage1", "s1"}:
            cfg = stage1_cfg or {}
            ratios = cfg.get("mask_ratios") or [0.08, 0.12, 0.18]
            ratio = float(self._rng.choice(np.asarray(ratios, dtype=np.float32)))
            buckets = cfg.get("gap_buckets") or [(1, 3), (3, 6), (6, 10)]
            return self._sample_contiguous_gap_mask(length=length, mask_ratio=ratio, gap_buckets=buckets)
        if mode_n in {"stage2_medium", "s2_medium"}:
            cfg = stage2_medium_cfg or {}
            ratios = cfg.get("mask_ratios") or [0.60, 0.72, 0.85]
            ratio = float(self._rng.choice(np.asarray(ratios, dtype=np.float32)))
            buckets = cfg.get("gap_buckets") or [(25, 35), (35, 48), (48, 62)]
            return self._sample_contiguous_gap_mask(length=length, mask_ratio=ratio, gap_buckets=buckets)
        if mode_n in {"stage2", "s2", "stage2_irregular_medium"}:
            cfg = stage2_cfg or {}
            ratios = cfg.get("mask_ratios") or [0.18, 0.28, 0.38]
            ratio = float(self._rng.choice(np.asarray(ratios, dtype=np.float32)))
            buckets = cfg.get("gap_buckets") or [(10, 20), (20, 30), (30, 45)]
            return self._sample_contiguous_gap_mask(length=length, mask_ratio=ratio, gap_buckets=buckets)
        # Stage3 keeps existing ADS-C interval simulation.
        return self.sample_anchor_mask(length=length, stats=stats or {})
