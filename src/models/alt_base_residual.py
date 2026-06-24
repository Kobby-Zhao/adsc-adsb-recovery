from __future__ import annotations

import torch
from torch import nn


def _bucket_id_from_gap_len(gap_len: torch.Tensor) -> torch.Tensor:
    # 0:<=15, 1:(15,30], 2:(30,60], 3:(60,180], 4:>180
    b0 = (gap_len <= 15.0).long()
    b1 = ((gap_len > 15.0) & (gap_len <= 30.0)).long() * 1
    b2 = ((gap_len > 30.0) & (gap_len <= 60.0)).long() * 2
    b3 = ((gap_len > 60.0) & (gap_len <= 180.0)).long() * 3
    b4 = (gap_len > 180.0).long() * 4
    return b0 + b1 + b2 + b3 + b4


class ResidualRangeNormalizer(nn.Module):
    """Bucket-wise residual range for bounded residual prediction.

    Residual bound must come from train-only statistics and reused in val/test.
    """

    def __init__(self, bounds: list[float] | tuple[float, ...] | None = None, min_bound: float = 10.0) -> None:
        super().__init__()
        b = list(bounds) if bounds is not None else [80.0, 120.0, 180.0, 300.0, 500.0]
        if len(b) != 5:
            raise ValueError(f"residual bounds must have 5 buckets, got {len(b)}")
        bb = [max(float(x), float(min_bound)) for x in b]
        self.register_buffer("bounds", torch.tensor(bb, dtype=torch.float32))

    def forward(self, gap_len: torch.Tensor) -> torch.Tensor:
        bid = _bucket_id_from_gap_len(gap_len).clamp(min=0, max=4)
        return self.bounds.to(gap_len.device, dtype=gap_len.dtype)[bid]


class AltitudeBaselineBuilder(nn.Module):
    """Build baseline altitude curve from anchors/known observations.

    baseline_type:
    - linear_anchor: two-anchor linear interpolation when both anchors exist.
    - local_diff: one-sided extrapolation from local vertical slope.
    """

    def __init__(self, baseline_type: str = "auto") -> None:
        super().__init__()
        self.baseline_type = str(baseline_type).lower()

    def _build_prev_next_anchor(
        self,
        obs_alt: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, tlen = obs_alt.shape
        prev_alt = torch.zeros_like(obs_alt)
        next_alt = torch.zeros_like(obs_alt)
        prev_idx = torch.zeros_like(obs_alt)
        next_idx = torch.zeros_like(obs_alt)
        has_prev = torch.zeros_like(obs_alt)
        has_next = torch.zeros_like(obs_alt)
        for b in range(bsz):
            last_i = -1
            last_alt = 0.0
            for t in range(tlen):
                if obs_mask[b, t] > 0.5:
                    last_i = t
                    last_alt = float(obs_alt[b, t].item())
                if last_i >= 0:
                    prev_alt[b, t] = last_alt
                    prev_idx[b, t] = float(last_i)
                    has_prev[b, t] = 1.0
            next_i = -1
            next_alt_v = 0.0
            for t in range(tlen - 1, -1, -1):
                if obs_mask[b, t] > 0.5:
                    next_i = t
                    next_alt_v = float(obs_alt[b, t].item())
                if next_i >= 0:
                    next_alt[b, t] = next_alt_v
                    next_idx[b, t] = float(next_i)
                    has_next[b, t] = 1.0
        return prev_alt, next_alt, prev_idx, next_idx, has_prev, has_next

    def _build_local_rate(self, obs_alt: torch.Tensor, obs_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, tlen = obs_alt.shape
        rate_prev = torch.zeros_like(obs_alt)
        rate_next = torch.zeros_like(obs_alt)
        for b in range(bsz):
            valid_idx = [i for i in range(tlen) if obs_mask[b, i] > 0.5]
            for t in range(tlen):
                prev = [i for i in valid_idx if i <= t]
                if len(prev) >= 2:
                    i2, i1 = prev[-2], prev[-1]
                    dt = max(i1 - i2, 1)
                    rate_prev[b, t] = (obs_alt[b, i1] - obs_alt[b, i2]) / float(dt)
                nxt = [i for i in valid_idx if i >= t]
                if len(nxt) >= 2:
                    i1, i2 = nxt[0], nxt[1]
                    dt = max(i2 - i1, 1)
                    rate_next[b, t] = (obs_alt[b, i2] - obs_alt[b, i1]) / float(dt)
        return rate_prev, rate_next

    def forward(
        self,
        obs_alt: torch.Tensor,
        obs_mask: torch.Tensor,
        dt_prev: torch.Tensor,
        dt_next: torch.Tensor,
    ) -> torch.Tensor:
        prev_alt, next_alt, prev_idx, next_idx, has_prev, has_next = self._build_prev_next_anchor(
            obs_alt=obs_alt,
            obs_mask=obs_mask,
        )
        gap_len = dt_prev + dt_next
        ratio = dt_prev / (gap_len + 1e-6)
        linear = prev_alt + ratio * (next_alt - prev_alt)

        rate_prev, rate_next = self._build_local_rate(obs_alt=obs_alt, obs_mask=obs_mask)
        local_prev = prev_alt + rate_prev * dt_prev
        local_next = next_alt - rate_next * dt_next

        base = torch.where((has_prev > 0.5) & (has_next > 0.5), linear, torch.zeros_like(obs_alt))
        prev_only = (has_prev > 0.5) & (has_next <= 0.5)
        next_only = (has_prev <= 0.5) & (has_next > 0.5)
        base = torch.where(prev_only, local_prev, base)
        base = torch.where(next_only, local_next, base)
        # At observed points force baseline to known observation value.
        base = torch.where(obs_mask > 0.5, obs_alt, base)
        return base


class AltitudeResidualHead(nn.Module):
    """Predict normalized bounded residual in [-1, 1]."""

    def __init__(self, in_dim: int, hidden_size: int = 64, dropout: float = 0.0, use_tanh: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(int(in_dim), int(hidden_size)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size), 1),
        ]
        if bool(use_tanh):
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
