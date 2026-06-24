from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ADSBMinuteAggregator:
    """Clean ADS-B points and aggregate to minute-level tracks."""

    flight_id_col: str = "flight_id"
    timestamp_col: str = "timestamp"
    lat_col: str = "lat"
    lon_col: str = "lon"
    alt_col: str = "alt"
    speed_col: str = "speed"
    heading_col: str = "heading"

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        mapping = {
            "icao24": self.flight_id_col,
            "time": self.timestamp_col,
            "baroaltitude": self.alt_col,
            "velocity": self.speed_col,
            "latitude": self.lat_col,
            "longitude": self.lon_col,
        }
        rename = {k: v for k, v in mapping.items() if k in df.columns and v not in df.columns}
        if rename:
            df = df.rename(columns=rename)
        return df

    @staticmethod
    def _circular_mean_deg(values: pd.Series) -> float:
        vals = pd.to_numeric(values, errors="coerce").dropna()
        if vals.empty:
            return np.nan
        radians = np.deg2rad(vals % 360)
        mean_sin = np.sin(radians).mean()
        mean_cos = np.cos(radians).mean()
        angle = np.rad2deg(np.arctan2(mean_sin, mean_cos))
        if np.isnan(angle):
            return np.nan
        return float((angle + 360.0) % 360.0)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._normalize_columns(df.copy())
        required = [
            self.flight_id_col,
            self.timestamp_col,
            self.lat_col,
            self.lon_col,
            self.alt_col,
            self.speed_col,
            self.heading_col,
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise RuntimeError(f"Missing ADS-B columns: {missing}")

        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col], errors="coerce", utc=True)
        for col in [self.lat_col, self.lon_col, self.alt_col, self.speed_col, self.heading_col]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=[self.flight_id_col, self.timestamp_col, self.lat_col, self.lon_col])
        df = df[df[self.lat_col].between(-90, 90) & df[self.lon_col].between(-180, 180)]
        df = df[df[self.alt_col].between(0, 20000)]
        df = df[df[self.speed_col].between(0, 350)]
        df = df[df[self.heading_col].between(0, 360)]

        df = df.sort_values([self.flight_id_col, self.timestamp_col])
        df = df.drop_duplicates(subset=[self.flight_id_col, self.timestamp_col])
        return df

    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.clean(df)
        df = df.copy()
        df["minute_ts"] = df[self.timestamp_col].dt.floor("min")

        grouped = (
            df.groupby([self.flight_id_col, "minute_ts"], as_index=False)
            .agg(
                lat=(self.lat_col, "mean"),
                lon=(self.lon_col, "mean"),
                alt=(self.alt_col, "mean"),
                speed=(self.speed_col, "mean"),
                num_points_in_minute=(self.timestamp_col, "count"),
            )
            .sort_values([self.flight_id_col, "minute_ts"])
        )

        heading = (
            df.groupby([self.flight_id_col, "minute_ts"])[self.heading_col]
            .apply(self._circular_mean_deg)
            .reset_index(name="heading")
        )
        out = grouped.merge(heading, on=[self.flight_id_col, "minute_ts"], how="left")
        return out[[self.flight_id_col, "minute_ts", "lat", "lon", "alt", "speed", "heading", "num_points_in_minute"]]
