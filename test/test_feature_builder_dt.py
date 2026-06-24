from __future__ import annotations

import unittest

import pandas as pd

from src.preprocessing.feature_builder import FeatureBuilder


class TestFeatureBuilderDt(unittest.TestCase):
    def test_dt_uses_real_minutes_for_irregular_timestamps(self) -> None:
        df = pd.DataFrame(
            {
                "minute_ts": [
                    "2024-05-01T00:00:00Z",
                    "2024-05-01T00:01:00Z",
                    "2024-05-01T00:04:00Z",
                    "2024-05-01T00:10:00Z",
                    "2024-05-01T00:13:00Z",
                ],
                "obs_mask": [1, 0, 0, 1, 0],
                "lat": [35.0, 35.1, 35.2, 35.3, 35.4],
                "lon": [120.0, 120.1, 120.2, 120.3, 120.4],
                "alt": [10000, 10010, 10020, 10030, 10040],
                "speed": [250, 251, 252, 253, 254],
                "heading": [90, 91, 92, 93, 94],
                "num_points_in_minute": [4, 4, 3, 4, 3],
            }
        )

        out = FeatureBuilder().build(df)

        self.assertListEqual(out["dt_prev"].round(3).tolist(), [0.0, 1.0, 4.0, 0.0, 3.0])
        self.assertListEqual(out["dt_next"].round(3).tolist(), [0.0, 9.0, 6.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()

