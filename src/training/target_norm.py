from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def compute_target_stats_from_loader(
    loader,
    coord_mode: str,
    apply_dims: list[int] | None = None,
    center: bool = True,
    center_per_dim: list[bool] | None = None,
    u_relative_anchor: bool = False,
    en_relative_anchor: bool = True,
    en_incremental: bool = False,
) -> dict[str, list[float]]:
    from src.training.coords import prepare_model_coordinates

    acc: list[np.ndarray] = []
    for batch in loader:
        target_pos = batch["target_pos"]
        obs_pos = batch["obs_pos"]
        obs_mask = batch["obs_mask"]
        seq_mask = batch["seq_mask"]
        target_model, _, _ = prepare_model_coordinates(
            target_pos=target_pos,
            obs_pos=obs_pos,
            obs_mask=obs_mask,
            seq_mask=seq_mask,
            mode=coord_mode,
            u_relative_anchor=u_relative_anchor,
            en_relative_anchor=en_relative_anchor,
            en_incremental=en_incremental,
        )
        valid = seq_mask > 0.5
        vals = target_model.detach().cpu().numpy()[valid.detach().cpu().numpy()]
        if vals.size:
            acc.append(vals)

    if not acc:
        return {"enabled": False, "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}

    all_vals = np.concatenate(acc, axis=0)
    mean = all_vals.mean(axis=0)
    std = all_vals.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    if center_per_dim is not None:
        center_mask = np.ones_like(mean, dtype=bool)
        for d in range(min(len(center_per_dim), int(mean.shape[0]))):
            center_mask[d] = bool(center_per_dim[d])
        mean = np.where(center_mask, mean, 0.0)
    elif not center:
        mean = np.zeros_like(mean)
    if apply_dims is not None:
        dim_set = {int(d) for d in apply_dims if 0 <= int(d) < int(mean.shape[0])}
        for d in range(int(mean.shape[0])):
            if d not in dim_set:
                mean[d] = 0.0
                std[d] = 1.0
    return {
        "enabled": True,
        "mean": [float(x) for x in mean.tolist()],
        "std": [float(x) for x in std.tolist()],
        "count": int(all_vals.shape[0]),
        "apply_dims": [int(d) for d in apply_dims] if apply_dims is not None else None,
        "center": bool(center),
        "center_per_dim": [bool(x) for x in center_per_dim] if center_per_dim is not None else None,
    }


def normalize_coords(x: torch.Tensor, stats: dict | None) -> torch.Tensor:
    if not stats or not bool(stats.get("enabled", False)):
        return x
    mean = torch.tensor(stats["mean"], device=x.device, dtype=x.dtype).view(1, 1, -1)
    std = torch.tensor(stats["std"], device=x.device, dtype=x.dtype).view(1, 1, -1)
    return (x - mean) / std


def denormalize_coords(x: torch.Tensor, stats: dict | None) -> torch.Tensor:
    if not stats or not bool(stats.get("enabled", False)):
        return x
    mean = torch.tensor(stats["mean"], device=x.device, dtype=x.dtype).view(1, 1, -1)
    std = torch.tensor(stats["std"], device=x.device, dtype=x.dtype).view(1, 1, -1)
    return x * std + mean


def save_target_stats(stats: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def load_target_stats(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))
