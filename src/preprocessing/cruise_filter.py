from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CruiseSegmentFilter:
    min_cruise_minutes: int = 40
    max_abs_vertical_rate: float = 300.0  # m/min
    max_speed_delta: float = 30.0  # m/s per min
    max_heading_rate: float = 5.0  # deg per min

    @staticmethod
    def _heading_delta_deg(series: pd.Series) -> pd.Series:
        a = pd.to_numeric(series, errors="coerce")
        delta = (a.diff() + 180.0) % 360.0 - 180.0
        return delta.abs()

    def mark_cruise(self, df: pd.DataFrame, flight_id_col: str = "flight_id") -> pd.DataFrame:
        out = df.sort_values([flight_id_col, "minute_ts"]).copy()
        dt_min = out.groupby(flight_id_col)["minute_ts"].diff().dt.total_seconds().div(60.0)
        dt_min = dt_min.replace(0, np.nan)

        out["vertical_speed"] = out.groupby(flight_id_col)["alt"].diff().div(dt_min)
        out["speed_delta"] = out.groupby(flight_id_col)["speed"].diff().abs().div(dt_min)
        out["heading_delta"] = out.groupby(flight_id_col)["heading"].transform(self._heading_delta_deg)
        out["heading_rate"] = out["heading_delta"].div(dt_min)

        stable = (
            out["vertical_speed"].abs().fillna(0.0).le(self.max_abs_vertical_rate)
            & out["speed_delta"].fillna(0.0).le(self.max_speed_delta)
            & out["heading_rate"].fillna(0.0).le(self.max_heading_rate)
        )

        out["is_cruise_candidate"] = stable.astype(int)
        out["is_cruise"] = 0

        for _, idx in out.groupby(flight_id_col).groups.items():
            g = out.loc[idx].copy()
            run_id = (g["is_cruise_candidate"].ne(g["is_cruise_candidate"].shift())).cumsum()
            run_len = g.groupby(run_id)["is_cruise_candidate"].transform("size")
            keep = (g["is_cruise_candidate"].eq(1)) & run_len.ge(self.min_cruise_minutes)
            out.loc[g.index[keep], "is_cruise"] = 1

        return out

    def filter(self, df: pd.DataFrame, flight_id_col: str = "flight_id") -> pd.DataFrame:
        marked = self.mark_cruise(df, flight_id_col=flight_id_col)
        return marked[marked["is_cruise"].eq(1)].copy()
