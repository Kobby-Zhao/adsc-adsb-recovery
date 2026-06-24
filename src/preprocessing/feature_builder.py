from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureBuilder:
    """Build MVP exogenous and quality proxy features."""

    @staticmethod
    def _heading_delta_deg(series: pd.Series) -> pd.Series:
        delta = (pd.to_numeric(series, errors="coerce").diff() + 180.0) % 360.0 - 180.0
        return delta

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.sort_values("minute_ts").copy()
        out["is_anchor"] = out["obs_mask"].astype(int)
        times = pd.to_datetime(out["minute_ts"], utc=True, errors="coerce")
        idx = np.arange(len(out))
        anchor_idx = idx[out["obs_mask"].values == 1]
        prev_anchor = np.full(len(out), -1, dtype=int)
        next_anchor = np.full(len(out), -1, dtype=int)

        if len(anchor_idx):
            j = 0
            last = -1
            for i in range(len(out)):
                while j < len(anchor_idx) and anchor_idx[j] <= i:
                    last = anchor_idx[j]
                    j += 1
                prev_anchor[i] = last
            j = len(anchor_idx) - 1
            nxt = -1
            for i in range(len(out) - 1, -1, -1):
                while j >= 0 and anchor_idx[j] >= i:
                    nxt = anchor_idx[j]
                    j -= 1
                next_anchor[i] = nxt

        dt_prev = np.zeros(len(out), dtype=float)
        dt_next = np.zeros(len(out), dtype=float)
        for i in range(len(out)):
            if prev_anchor[i] >= 0 and pd.notna(times.iloc[i]) and pd.notna(times.iloc[prev_anchor[i]]):
                dt_prev[i] = (times.iloc[i] - times.iloc[prev_anchor[i]]).total_seconds() / 60.0
            if next_anchor[i] >= 0 and pd.notna(times.iloc[i]) and pd.notna(times.iloc[next_anchor[i]]):
                dt_next[i] = (times.iloc[next_anchor[i]] - times.iloc[i]).total_seconds() / 60.0

        out["dt_prev"] = dt_prev
        out["dt_next"] = dt_next
        out["gap_len"] = out["dt_prev"] + out["dt_next"]
        out["gap_pos_ratio"] = np.where(out["gap_len"] > 0, out["dt_prev"] / out["gap_len"], 0.0)

        out["vertical_speed"] = out["alt"].diff().fillna(0.0)
        out["speed_delta"] = out["speed"].diff().fillna(0.0)
        out["heading_delta"] = self._heading_delta_deg(out["heading"]).fillna(0.0)
        out["turn_rate"] = out["heading_delta"]

        win = 5
        out["local_speed_std"] = out["speed"].rolling(win, min_periods=1, center=True).std().fillna(0.0)
        out["local_heading_std"] = out["heading_delta"].rolling(win, min_periods=1, center=True).std().fillna(0.0)
        out["local_alt_std"] = out["alt"].rolling(win, min_periods=1, center=True).std().fillna(0.0)

        step_dist = np.sqrt(out["lat"].diff().fillna(0.0) ** 2 + out["lon"].diff().fillna(0.0) ** 2)
        out["jump_flag"] = (step_dist > step_dist.quantile(0.95)).astype(int)
        smooth_lat = out["lat"].rolling(3, min_periods=1, center=True).mean()
        smooth_lon = out["lon"].rolling(3, min_periods=1, center=True).mean()
        out["smooth_residual_proxy"] = np.sqrt((out["lat"] - smooth_lat) ** 2 + (out["lon"] - smooth_lon) ** 2)

        for key in ["tag13_exists", "tag14_exists", "tag15_exists", "tag16_exists"]:
            if key not in out.columns:
                out[key] = 0
        if "position_accuracy" not in out.columns:
            out["position_accuracy"] = np.nan

        return out
