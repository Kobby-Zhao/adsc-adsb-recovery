from __future__ import annotations

import torch


def _haversine_m(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    lat1 = torch.deg2rad(pred[..., 0])
    lon1 = torch.deg2rad(pred[..., 1])
    lat2 = torch.deg2rad(target[..., 0])
    lon2 = torch.deg2rad(target[..., 1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.arcsin(torch.sqrt(torch.clamp(a, 0.0, 1.0)))
    return 6371000.0 * c


def _masked_mae_rmse(err: torch.Tensor, mask_bt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask_bt = mask_bt.float()
    count = mask_bt.sum()
    denom = count * err.shape[-1] + 1e-6
    mae = (torch.abs(err) * mask_bt.unsqueeze(-1)).sum() / denom
    rmse = torch.sqrt(((err**2) * mask_bt.unsqueeze(-1)).sum() / denom)
    return mae, rmse, count


def _per_dim_stats(
    pred_pos: torch.Tensor,
    target_pos: torch.Tensor,
    mask_bt: torch.Tensor,
    region_name: str,
) -> dict[str, float]:
    mask = mask_bt.float().unsqueeze(-1)
    count = mask_bt.float().sum()
    out: dict[str, float] = {f"{region_name}_count": float(count.detach().cpu())}

    for d in range(pred_pos.shape[-1]):
        pred_d = pred_pos[..., d]
        tgt_d = target_pos[..., d]
        err_d = pred_d - tgt_d
        m = mask_bt.float()
        denom = m.sum() + 1e-6

        mae = (torch.abs(err_d) * m).sum() / denom
        rmse = torch.sqrt(((err_d**2) * m).sum() / denom)
        bias = (err_d * m).sum() / denom

        pred_mean = (pred_d * m).sum() / denom
        tgt_mean = (tgt_d * m).sum() / denom
        pred_var = (((pred_d - pred_mean) ** 2) * m).sum() / denom
        tgt_var = (((tgt_d - tgt_mean) ** 2) * m).sum() / denom
        pred_std = torch.sqrt(torch.clamp(pred_var, min=0.0))
        tgt_std = torch.sqrt(torch.clamp(tgt_var, min=0.0))
        std_ratio = pred_std / (tgt_std + 1e-6)

        prefix = f"{region_name}_dim{d}"
        out[f"{prefix}_mae"] = float(mae.detach().cpu())
        out[f"{prefix}_rmse"] = float(rmse.detach().cpu())
        out[f"{prefix}_bias"] = float(bias.detach().cpu())
        out[f"{prefix}_pred_std"] = float(pred_std.detach().cpu())
        out[f"{prefix}_target_std"] = float(tgt_std.detach().cpu())
        out[f"{prefix}_pred_over_target_std"] = float(std_ratio.detach().cpu())
    return out


def _build_long_gap_mask(obs_mask: torch.Tensor, seq_mask: torch.Tensor, long_gap_threshold: int) -> tuple[torch.Tensor, torch.Tensor]:
    valid = seq_mask > 0.5
    gap = (obs_mask <= 0.5) & valid
    long_gap = torch.zeros_like(gap, dtype=torch.bool)
    if long_gap_threshold <= 1:
        return gap, torch.zeros_like(gap, dtype=torch.bool)

    bsz, t_len = gap.shape
    for i in range(bsz):
        t = 0
        while t < t_len:
            if not valid[i, t]:
                t += 1
                continue
            if not gap[i, t]:
                t += 1
                continue
            start = t
            while t < t_len and valid[i, t] and gap[i, t]:
                t += 1
            if (t - start) >= long_gap_threshold:
                long_gap[i, start:t] = True
    short_gap = gap & (~long_gap)
    return long_gap, short_gap


def compute_metrics(
    pred_pos: torch.Tensor,
    target_pos: torch.Tensor,
    seq_mask: torch.Tensor,
    obs_mask: torch.Tensor,
    long_gap_threshold: int = 20,
) -> dict:
    err = pred_pos - target_pos
    valid = seq_mask > 0.5
    anchor = (obs_mask > 0.5) & valid
    gap = (obs_mask <= 0.5) & valid
    long_gap, short_gap = _build_long_gap_mask(obs_mask=obs_mask, seq_mask=seq_mask, long_gap_threshold=long_gap_threshold)

    mae, rmse, valid_count = _masked_mae_rmse(err=err, mask_bt=valid)
    anchor_mae, anchor_rmse, anchor_count = _masked_mae_rmse(err=err, mask_bt=anchor)
    gap_mae, gap_rmse, gap_count = _masked_mae_rmse(err=err, mask_bt=gap)
    long_gap_mae, long_gap_rmse, long_gap_count = _masked_mae_rmse(err=err, mask_bt=long_gap)
    short_gap_mae, short_gap_rmse, short_gap_count = _masked_mae_rmse(err=err, mask_bt=short_gap)

    hav = _haversine_m(pred_pos, target_pos)
    hav_mean = (hav * seq_mask).sum() / (seq_mask.sum() + 1e-6)

    alt_mae = (torch.abs(err[..., 2]) * seq_mask).sum() / (seq_mask.sum() + 1e-6)
    alt_rmse = torch.sqrt((((err[..., 2] ** 2) * seq_mask).sum()) / (seq_mask.sum() + 1e-6))
    anchor_alt_mae = (torch.abs(err[..., 2]) * anchor.float()).sum() / (anchor.float().sum() + 1e-6)
    anchor_alt_rmse = torch.sqrt((((err[..., 2] ** 2) * anchor.float()).sum()) / (anchor.float().sum() + 1e-6))
    gap_alt_mae = (torch.abs(err[..., 2]) * gap.float()).sum() / (gap.float().sum() + 1e-6)
    gap_alt_rmse = torch.sqrt((((err[..., 2] ** 2) * gap.float()).sum()) / (gap.float().sum() + 1e-6))
    anchor_hav = hav[anchor].mean() if anchor.any() else torch.tensor(0.0, device=pred_pos.device)
    gap_hav = hav[gap].mean() if gap.any() else torch.tensor(0.0, device=pred_pos.device)
    long_gap_hav = hav[long_gap].mean() if long_gap.any() else torch.tensor(0.0, device=pred_pos.device)
    short_gap_hav = hav[short_gap].mean() if short_gap.any() else torch.tensor(0.0, device=pred_pos.device)

    out = {
        "mae": float(mae.detach().cpu()),
        "rmse": float(rmse.detach().cpu()),
        "overall_mae": float(mae.detach().cpu()),
        "overall_rmse": float(rmse.detach().cpu()),
        "anchor_mae": float(anchor_mae.detach().cpu()),
        "anchor_rmse": float(anchor_rmse.detach().cpu()),
        "gap_mae": float(gap_mae.detach().cpu()),
        "gap_rmse": float(gap_rmse.detach().cpu()),
        "long_gap_mae": float(long_gap_mae.detach().cpu()),
        "long_gap_rmse": float(long_gap_rmse.detach().cpu()),
        "short_gap_mae": float(short_gap_mae.detach().cpu()),
        "short_gap_rmse": float(short_gap_rmse.detach().cpu()),
        "haversine_m": float(hav_mean.detach().cpu()),
        "altitude_mae": float(alt_mae.detach().cpu()),
        "altitude_rmse": float(alt_rmse.detach().cpu()),
        "anchor_altitude_mae": float(anchor_alt_mae.detach().cpu()),
        "anchor_altitude_rmse": float(anchor_alt_rmse.detach().cpu()),
        "gap_altitude_mae": float(gap_alt_mae.detach().cpu()),
        "gap_altitude_rmse": float(gap_alt_rmse.detach().cpu()),
        "anchor_haversine_m": float(anchor_hav.detach().cpu()),
        "gap_haversine_m": float(gap_hav.detach().cpu()),
        "long_gap_haversine_m": float(long_gap_hav.detach().cpu()),
        "short_gap_haversine_m": float(short_gap_hav.detach().cpu()),
        "valid_count": float(valid_count.detach().cpu()),
        "anchor_count": float(anchor_count.detach().cpu()),
        "gap_count": float(gap_count.detach().cpu()),
        "long_gap_count": float(long_gap_count.detach().cpu()),
        "short_gap_count": float(short_gap_count.detach().cpu()),
    }
    out.update(_per_dim_stats(pred_pos=pred_pos, target_pos=target_pos, mask_bt=valid, region_name="overall"))
    out.update(_per_dim_stats(pred_pos=pred_pos, target_pos=target_pos, mask_bt=anchor, region_name="anchor"))
    out.update(_per_dim_stats(pred_pos=pred_pos, target_pos=target_pos, mask_bt=gap, region_name="gap"))
    out.update(_per_dim_stats(pred_pos=pred_pos, target_pos=target_pos, mask_bt=long_gap, region_name="long_gap"))
    return out
