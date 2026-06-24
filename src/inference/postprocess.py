from __future__ import annotations

import numpy as np


def linear_interpolate_track(track: np.ndarray) -> np.ndarray:
    out = track.copy()
    for dim in range(out.shape[1]):
        col = out[:, dim]
        idx = np.arange(len(col))
        mask = ~np.isnan(col)
        if mask.sum() < 2:
            continue
        out[:, dim] = np.interp(idx, idx[mask], col[mask])
    return out
