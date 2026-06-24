from __future__ import annotations

import torch


def apply_alt_target_transform(
    coords: torch.Tensor,
    mode: str = "none",
    clip_value: float = 3000.0,
) -> torch.Tensor:
    if mode == "none":
        return coords
    out = coords.clone()
    x = out[..., 2]
    if mode == "clipped":
        c = float(max(1e-6, clip_value))
        out[..., 2] = torch.clamp(x, min=-c, max=c)
        return out
    if mode == "signed_log":
        out[..., 2] = torch.sign(x) * torch.log1p(torch.abs(x))
        return out
    raise ValueError(f"Unsupported alt target transform mode: {mode}")


def invert_alt_target_transform(
    coords: torch.Tensor,
    mode: str = "none",
    clip_value: float = 3000.0,
) -> torch.Tensor:
    if mode == "none":
        return coords
    out = coords.clone()
    x = out[..., 2]
    if mode == "clipped":
        c = float(max(1e-6, clip_value))
        out[..., 2] = torch.clamp(x, min=-c, max=c)
        return out
    if mode == "signed_log":
        out[..., 2] = torch.sign(x) * torch.expm1(torch.abs(x))
        return out
    raise ValueError(f"Unsupported alt target transform mode: {mode}")
