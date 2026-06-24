from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing.feature_builder import FeatureBuilder


class TrajectorySampleBuilder:
    """Build sequence samples with sparse observations and features."""

    def __init__(self, window_size: int = 180, stride: int | None = None) -> None:
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size
        self.feature_builder = FeatureBuilder()

    @staticmethod
    def _segment_ranges_by_time_gap(
        base: pd.DataFrame,
        max_time_gap_minutes: float,
    ) -> list[tuple[int, int]]:
        times = pd.to_datetime(base["minute_ts"], utc=True, errors="coerce")
        if len(times) <= 1:
            return [(0, len(base))]

        dt_min = times.diff().dt.total_seconds().div(60.0)
        # Split segment at large timestamp jumps or invalid timestamps.
        split_points = [0]
        for i in range(1, len(base)):
            d = dt_min.iloc[i]
            if pd.isna(d) or float(d) > float(max_time_gap_minutes):
                split_points.append(i)
        split_points.append(len(base))

        out: list[tuple[int, int]] = []
        for i in range(len(split_points) - 1):
            s, e = int(split_points[i]), int(split_points[i + 1])
            if e > s:
                out.append((s, e))
        return out

    @staticmethod
    def _apply_obs_mask(df: pd.DataFrame, obs_mask: np.ndarray) -> pd.DataFrame:
        out = df.copy()
        out["obs_mask"] = obs_mask.astype(int)
        out["obs_lat"] = np.where(out["obs_mask"].eq(1), out["lat"], 0.0)
        out["obs_lon"] = np.where(out["obs_mask"].eq(1), out["lon"], 0.0)
        out["obs_alt"] = np.where(out["obs_mask"].eq(1), out["alt"], 0.0)
        return out

    def build_for_flight(
        self,
        flight_df: pd.DataFrame,
        obs_mask: np.ndarray,
        flight_id_col: str = "flight_id",
        sample_prefix: str | None = None,
        max_time_gap_minutes: float | None = None,
        min_segment_len: int = 10,
    ) -> list[pd.DataFrame]:
        base = flight_df.sort_values("minute_ts").reset_index(drop=True)
        if len(base) != len(obs_mask):
            raise RuntimeError("obs_mask length mismatch")

        ranges = [(0, len(base))]
        if max_time_gap_minutes is not None and float(max_time_gap_minutes) > 0:
            ranges = self._segment_ranges_by_time_gap(base=base, max_time_gap_minutes=float(max_time_gap_minutes))

        samples: list[pd.DataFrame] = []
        for seg_id, (s, e) in enumerate(ranges):
            seg = base.iloc[s:e].copy().reset_index(drop=True)
            seg_mask = np.asarray(obs_mask[s:e], dtype=np.int64)
            if len(seg) < int(min_segment_len):
                continue
            # For each continuous segment, keep boundary states known.
            if len(seg_mask) >= 1:
                seg_mask[0] = 1
                seg_mask[-1] = 1
            seg = self._apply_obs_mask(seg, seg_mask)
            seg = self.feature_builder.build(seg)

            seg_prefix_raw = sample_prefix or str(seg[flight_id_col].iloc[0])
            seg_prefix = f"{seg_prefix_raw}_seg{seg_id}"

            if len(seg) < self.window_size:
                chunk = seg.copy()
                chunk["sample_id"] = f"{seg_prefix}_0"
                samples.append(chunk)
                continue

            step = max(1, self.stride)
            for i, start in enumerate(range(0, len(seg) - self.window_size + 1, step)):
                chunk = seg.iloc[start : start + self.window_size].copy()
                chunk["sample_id"] = f"{seg_prefix}_{i}"
                samples.append(chunk)

            # Ensure the tail segment is covered for long flights.
            if (len(seg) - self.window_size) % step != 0:
                chunk = seg.iloc[-self.window_size :].copy()
                chunk["sample_id"] = f"{seg_prefix}_tail"
                samples.append(chunk)

        return samples
