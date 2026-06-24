from __future__ import annotations

from collections import defaultdict
import time

import numpy as np
import torch

from src.metrics import compute_metrics
from src.training.alt_target import apply_alt_target_transform, invert_alt_target_transform
from src.training.coords import _to_enu, build_anchor_alt_tracks, build_anchor_pair_tracks, prepare_model_coordinates, restore_to_latlon
from src.training.target_norm import denormalize_coords, normalize_coords


def _build_long_gap_mask(obs_mask: torch.Tensor, seq_mask: torch.Tensor, long_gap_threshold: int) -> torch.Tensor:
    valid = seq_mask > 0.5
    gap = (obs_mask <= 0.5) & valid
    long_gap = torch.zeros_like(gap, dtype=torch.bool)
    if long_gap_threshold <= 1:
        return gap
    bsz, t_len = gap.shape
    for i in range(bsz):
        t = 0
        while t < t_len:
            if not valid[i, t] or not gap[i, t]:
                t += 1
                continue
            start = t
            while t < t_len and valid[i, t] and gap[i, t]:
                t += 1
            if (t - start) >= long_gap_threshold:
                long_gap[i, start:t] = True
    return long_gap


def _masked_horizontal_rmse(pred_model: torch.Tensor, target_model: torch.Tensor, mask_bt: torch.Tensor) -> float:
    d_e = pred_model[..., 0] - target_model[..., 0]
    d_n = pred_model[..., 1] - target_model[..., 1]
    d_h = torch.sqrt(d_e**2 + d_n**2 + 1e-6)
    m = mask_bt.float()
    return float(torch.sqrt(((d_h**2) * m).sum() / (m.sum() + 1e-6)).detach().cpu())


def _masked_horizontal_rmse_m(pred_enu: torch.Tensor, target_enu: torch.Tensor, mask_bt: torch.Tensor) -> float:
    d_e = pred_enu[..., 0] - target_enu[..., 0]
    d_n = pred_enu[..., 1] - target_enu[..., 1]
    d_h = torch.sqrt(d_e**2 + d_n**2 + 1e-6)
    m = mask_bt.float()
    return float(torch.sqrt(((d_h**2) * m).sum() / (m.sum() + 1e-6)).detach().cpu())


def _masked_horizontal_mean_m(pred_enu: torch.Tensor, target_enu: torch.Tensor, mask_bt: torch.Tensor) -> float:
    d_e = pred_enu[..., 0] - target_enu[..., 0]
    d_n = pred_enu[..., 1] - target_enu[..., 1]
    d_h = torch.sqrt(d_e**2 + d_n**2 + 1e-6)
    m = mask_bt.float()
    return float(((d_h * m).sum() / (m.sum() + 1e-6)).detach().cpu())


def _masked_std_ratio(pred_d: torch.Tensor, tgt_d: torch.Tensor, mask_bt: torch.Tensor) -> tuple[float, float, float]:
    m = mask_bt.float()
    denom = m.sum() + 1e-6
    pred_mean = (pred_d * m).sum() / denom
    tgt_mean = (tgt_d * m).sum() / denom
    pred_std = torch.sqrt((((pred_d - pred_mean) ** 2) * m).sum() / denom)
    tgt_std = torch.sqrt((((tgt_d - tgt_mean) ** 2) * m).sum() / denom)
    ratio = pred_std / (tgt_std + 1e-6)
    return float(pred_std.detach().cpu()), float(tgt_std.detach().cpu()), float(ratio.detach().cpu())


def _masked_mean_std(x: torch.Tensor, mask_bt: torch.Tensor) -> tuple[float, float]:
    mask = mask_bt.unsqueeze(-1) if x.ndim == 3 else mask_bt
    denom = mask.sum().clamp_min(1.0)
    mean = (x * mask).sum() / denom
    var = (((x - mean) ** 2) * mask).sum() / denom
    return float(mean.detach().cpu()), float(torch.sqrt(torch.clamp(var, min=0.0)).detach().cpu())


def _find_complete_gaps(anchor_mask_1d: torch.Tensor, valid_mask_1d: torch.Tensor) -> list[tuple[int, int, int, int]]:
    gaps: list[tuple[int, int, int, int]] = []
    t_len = int(anchor_mask_1d.shape[0])
    t = 0
    while t < t_len:
        if not bool(valid_mask_1d[t]) or bool(anchor_mask_1d[t]):
            t += 1
            continue
        s = t
        while t < t_len and bool(valid_mask_1d[t]) and (not bool(anchor_mask_1d[t])):
            t += 1
        e = t - 1
        l = s - 1
        r = e + 1
        if l >= 0 and r < t_len and bool(valid_mask_1d[l]) and bool(valid_mask_1d[r]) and bool(anchor_mask_1d[l]) and bool(anchor_mask_1d[r]):
            gaps.append((l, s, e, r))
    return gaps


def _gap_bucket_name(gap_len: int) -> str:
    if gap_len <= 3:
        return "1_3"
    if gap_len <= 8:
        return "4_8"
    if gap_len <= 15:
        return "9_15"
    if gap_len <= 30:
        return "16_30"
    return "30_plus"


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _boundary_alt_from_model_obs(
    obs_for_model: torch.Tensor,
    obs_mask: torch.Tensor,
    seq_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract dominant-gap boundaries directly in model space.

    This keeps boundary-conditioned main altitude head in the same coordinate
    space as model inputs/targets (ENU-relative/normalized if enabled).
    """
    bsz, t_len, _ = obs_for_model.shape
    left = torch.zeros((bsz,), device=obs_for_model.device, dtype=obs_for_model.dtype)
    right = torch.zeros((bsz,), device=obs_for_model.device, dtype=obs_for_model.dtype)
    obs_alt = obs_for_model[..., 2]
    for i in range(bsz):
        valid = seq_mask[i] > 0.5
        obs = (obs_mask[i] > 0.5) & valid
        valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
        obs_idx = torch.nonzero(obs, as_tuple=False).flatten()
        if obs_idx.numel() == 0:
            if valid_idx.numel() == 0:
                left[i] = 0.0
                right[i] = 0.0
            else:
                left[i] = obs_alt[i, valid_idx[0]]
                right[i] = obs_alt[i, valid_idx[-1]]
            continue

        gap = (~obs) & valid
        # Find dominant contiguous gap [s, e)
        best_s, best_e, best_len = -1, -1, 0
        t = 0
        while t < t_len:
            if not bool(gap[t]):
                t += 1
                continue
            s = t
            while t < t_len and bool(gap[t]):
                t += 1
            e = t
            if (e - s) > best_len:
                best_s, best_e, best_len = s, e, e - s
        if best_len <= 0:
            left[i] = obs_alt[i, obs_idx[0]]
            right[i] = obs_alt[i, obs_idx[-1]]
            continue

        left_idx = best_s - 1 if (best_s - 1 >= 0 and bool(obs[best_s - 1])) else None
        right_idx = best_e if (best_e < t_len and bool(obs[best_e])) else None
        if left_idx is None:
            cand = obs_idx[obs_idx < best_s]
            if cand.numel() > 0:
                left_idx = int(cand[-1].item())
        if right_idx is None:
            cand = obs_idx[obs_idx >= best_e]
            if cand.numel() > 0:
                right_idx = int(cand[0].item())
        if left_idx is None and right_idx is None:
            left_idx = int(obs_idx[0].item())
            right_idx = int(obs_idx[-1].item())
        elif left_idx is None:
            left_idx = int(right_idx)  # type: ignore[arg-type]
        elif right_idx is None:
            right_idx = int(left_idx)
        left[i] = obs_alt[i, int(left_idx)]
        right[i] = obs_alt[i, int(right_idx)]
    return left, right


def _boundary_alt_from_batch_meta(
    left_boundary_alt: torch.Tensor,
    right_boundary_alt: torch.Tensor,
    *,
    u_relative_anchor: bool,
    target_norm_stats: dict | None,
    alt_target_transform_mode: str,
    alt_target_clip_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map raw boundary altitudes into model-space altitude coordinates."""
    if u_relative_anchor:
        left_raw = torch.zeros_like(left_boundary_alt)
        right_raw = right_boundary_alt - left_boundary_alt
    else:
        left_raw = left_boundary_alt
        right_raw = right_boundary_alt

    def _map(raw_alt: torch.Tensor) -> torch.Tensor:
        z = torch.zeros((raw_alt.shape[0], 1, 3), device=raw_alt.device, dtype=raw_alt.dtype)
        z[..., 2] = raw_alt.view(-1, 1)
        z = apply_alt_target_transform(
            z,
            mode=alt_target_transform_mode,
            clip_value=alt_target_clip_value,
        )
        z = normalize_coords(z, target_norm_stats)
        return z[:, 0, 2]

    return _map(left_raw), _map(right_raw)


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    teacher_forcing_ratio: float,
    train: bool,
    grad_clip: float,
    coord_mode: str = "latlon",
    u_relative_anchor: bool = False,
    en_relative_anchor: bool = True,
    en_incremental: bool = False,
    long_gap_threshold: int = 20,
    target_norm_stats: dict | None = None,
    alt_target_transform_mode: str = "none",
    alt_target_clip_value: float = 3000.0,
    enable_verbose_diag: bool = False,
    heartbeat_enabled: bool = False,
    heartbeat_interval: int = 200,
    use_segment_teacher: bool = True,
    use_alt_baseline_residual: bool = True,
):
    if not hasattr(loader, "__iter__"):
        raise TypeError(f"FATAL: loader must be iterable, got type={type(loader).__name__}")
    if heartbeat_enabled and heartbeat_interval > 0 and (not hasattr(loader, "__len__")):
        # Keep heartbeat usable for generic iterables (e.g., debug batch lists),
        # but avoid assumptions tied to DataLoader semantics.
        heartbeat_interval = max(1, int(heartbeat_interval))

    if train:
        model.train()
    else:
        model.eval()

    totals = defaultdict(float)
    count = 0
    grad_snapshot: dict[str, float] = {}
    grad_logged = False
    rel_sample_logged = False
    gap_audit_logged = False
    seg_acc: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    seg_err_points: dict[str, list[float]] = defaultdict(list)
    seg_err_end: dict[str, list[float]] = defaultdict(list)
    fusion_rel_bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    fusion_rel_acc: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    relpos_err_acc: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    prev_batch_end = time.perf_counter()

    for step_idx, batch in enumerate(loader, start=1):
        batch_start = time.perf_counter()
        data_time = batch_start - prev_batch_end
        model_t0 = batch_start
        forward_time = 0.0
        loss_time = 0.0
        metrics_time = 0.0
        obs_pos = batch["obs_pos"].to(device)
        obs_mask = batch["obs_mask"].to(device)
        dt_prev = batch["dt_prev"].to(device)
        dt_next = batch["dt_next"].to(device)
        exo = batch["exo"].to(device)
        quality = batch["quality"].to(device)
        global_quality = batch["global_quality"].to(device)
        risk_flag = batch["risk_flag"].to(device) if "risk_flag" in batch else None
        risk_flag_teacher = batch["risk_flag_teacher"].to(device) if ("risk_flag_teacher" in batch and use_segment_teacher) else None
        teacher_scale = batch["teacher_scale"].to(device) if ("teacher_scale" in batch and use_segment_teacher) else None
        segment_bucket = batch["segment_bucket"].to(device) if "segment_bucket" in batch else None
        anchor_pattern = batch["anchor_pattern"].to(device) if "anchor_pattern" in batch else None
        edge_weight = batch["edge_weight"].to(device) if ("edge_weight" in batch and use_segment_teacher) else None
        residual_rmax_m = batch["residual_rmax_m"].to(device) if ("residual_rmax_m" in batch and use_alt_baseline_residual) else None
        residual_rmax_ft = batch["residual_rmax_ft"].to(device) if ("residual_rmax_ft" in batch and use_alt_baseline_residual) else None
        gate_bias = batch["gate_bias"].to(device) if ("gate_bias" in batch and use_segment_teacher) else None
        left_boundary_alt = batch["left_boundary_alt"].to(device) if "left_boundary_alt" in batch else None
        right_boundary_alt = batch["right_boundary_alt"].to(device) if "right_boundary_alt" in batch else None
        target_pos = batch["target_pos"].to(device)
        seq_mask = batch["seq_mask"].to(device)
        target_model_raw, obs_model_raw, coord_ctx = prepare_model_coordinates(
            target_pos=target_pos,
            obs_pos=obs_pos,
            obs_mask=obs_mask,
            seq_mask=seq_mask,
            mode=coord_mode,
            u_relative_anchor=u_relative_anchor,
            en_relative_anchor=en_relative_anchor,
            en_incremental=en_incremental,
        )
        target_model = apply_alt_target_transform(
            target_model_raw,
            mode=alt_target_transform_mode,
            clip_value=alt_target_clip_value,
        )
        obs_model = apply_alt_target_transform(
            obs_model_raw,
            mode=alt_target_transform_mode,
            clip_value=alt_target_clip_value,
        )
        target_for_model = normalize_coords(target_model, target_norm_stats)
        obs_for_model = normalize_coords(obs_model, target_norm_stats)
        anchor_left_raw, anchor_right_raw = build_anchor_pair_tracks(
            obs_pos=obs_pos,
            obs_mask=obs_mask,
            seq_mask=seq_mask,
            ctx=coord_ctx,
        )
        anchor_left_model = normalize_coords(
            apply_alt_target_transform(
                anchor_left_raw,
                mode=alt_target_transform_mode,
                clip_value=alt_target_clip_value,
            ),
            target_norm_stats,
        )
        anchor_right_model = normalize_coords(
            apply_alt_target_transform(
                anchor_right_raw,
                mode=alt_target_transform_mode,
                clip_value=alt_target_clip_value,
            ),
            target_norm_stats,
        )
        if left_boundary_alt is not None and right_boundary_alt is not None:
            left_boundary_alt_model, right_boundary_alt_model = _boundary_alt_from_batch_meta(
                left_boundary_alt=left_boundary_alt,
                right_boundary_alt=right_boundary_alt,
                u_relative_anchor=bool(u_relative_anchor),
                target_norm_stats=target_norm_stats,
                alt_target_transform_mode=alt_target_transform_mode,
                alt_target_clip_value=alt_target_clip_value,
            )
        else:
            left_boundary_alt_model, right_boundary_alt_model = _boundary_alt_from_model_obs(
                obs_for_model=obs_for_model,
                obs_mask=obs_mask,
                seq_mask=seq_mask,
            )
        anchor_alt = build_anchor_alt_tracks(obs_pos=obs_pos, obs_mask=obs_mask, seq_mask=seq_mask)
        q_true = None
        q_mask = None
        q_res_true = None
        q_res_mask = None
        if left_boundary_alt is not None and right_boundary_alt is not None:
            gap = (obs_mask <= 0.5) & (seq_mask > 0.5)
            delta_z_raw = right_boundary_alt - left_boundary_alt
            active_seg = (torch.abs(delta_z_raw) > float(getattr(criterion, "vprog_enable_abs_dz_min", 100.0))).unsqueeze(-1)
            q_true = (target_pos[..., 2] - left_boundary_alt.unsqueeze(-1)) / (delta_z_raw.unsqueeze(-1) + 1e-6)
            q_true = torch.clamp(q_true, 0.0, 1.0)
            q_mask = gap.float() * active_seg.float()
            active_seg_res = (torch.abs(delta_z_raw) > float(getattr(criterion, "vprog_res_enable_abs_dz_min", 300.0))).unsqueeze(-1)
            gap_len = torch.clamp(dt_prev + dt_next, min=1e-6)
            r_t = torch.clamp(dt_prev / gap_len, min=0.0, max=1.0)
            q_res_true = q_true - r_t
            q_res_mask = gap.float() * active_seg_res.float()
        savca_beta_floor_mask = None
        if train and hasattr(criterion, "build_savca_beta_floor_mask"):
            savca_beta_floor_mask = criterion.build_savca_beta_floor_mask(
                target_alt_abs=target_pos[..., 2],
                obs_mask=obs_mask,
                seq_mask=seq_mask,
            )

        with torch.set_grad_enabled(train):
            _sync_if_cuda(device)
            t_fw0 = time.perf_counter()
            out = model(
                obs_pos=obs_for_model,
                obs_mask=obs_mask,
                seq_mask=seq_mask,
                dt_prev=dt_prev,
                dt_next=dt_next,
                exo=exo,
                vertical_exo=batch["vertical_exo"].to(device) if "vertical_exo" in batch else None,
                quality=quality,
                global_quality=global_quality,
                anchor_alt=anchor_alt,
                risk_flag=risk_flag,
                teacher_scale=teacher_scale,
                risk_flag_teacher=risk_flag_teacher,
                segment_bucket=segment_bucket,
                edge_weight=edge_weight,
                residual_rmax_m=residual_rmax_m,
                residual_rmax_ft=residual_rmax_ft,
                gate_bias=gate_bias,
                left_boundary_alt=left_boundary_alt_model,
                right_boundary_alt=right_boundary_alt_model,
                anchor_left=anchor_left_model,
                anchor_right=anchor_right_model,
                target_pos=target_for_model,
                savca_beta_floor_mask=savca_beta_floor_mask,
                teacher_forcing_ratio=teacher_forcing_ratio,
                return_vertical_tune_weights=False,
            )
            _sync_if_cuda(device)
            t_fw1 = time.perf_counter()
            # ---- shape logging (first batch only) ----
            if step_idx == 1:
                bsz, t_len = obs_pos.shape[0], obs_pos.shape[1]
                mu_f = out.get("mu_f")
                mu_b = out.get("mu_b")
                logvar_f = out.get("logvar_f", out.get("logvar"))
                logvar_b = out.get("logvar_b")
                pred = out.get("pred_pos")
                fusion_weights = out.get("fusion_weights")
                def _shape_str(tensor, name, note=""):
                    if tensor is None:
                        return f"{name}=None(placeholder){note}"
                    return f"{name}={tuple(tensor.shape)}{note}"
                parts = [
                    _shape_str(mu_f, "mu_f"),
                    _shape_str(mu_b, "mu_b", " (placeholder)" if mu_b is None else ""),
                    _shape_str(logvar_f, "logvar_f"),
                    _shape_str(logvar_b, "logvar_b", " (placeholder)" if logvar_b is None else ""),
                    _shape_str(pred, "pred"),
                    _shape_str(fusion_weights, "fusion_weights", " (placeholder)" if fusion_weights is None else ""),
                ]
                print(f"[shape] batch={bsz}x{t_len} " + " ".join(parts), flush=True)
            # ---- end shape logging ----
            _sync_if_cuda(device)
            t_loss0 = time.perf_counter()
            loss_dict = criterion(
                pred_pos=out["pred_pos"],
                target_pos=target_for_model,
                obs_mask=obs_mask,
                seq_mask=seq_mask,
                exo=exo,
                quality=quality,
                fusion_weights=out.get("fusion_weights"),
                dt_prev=dt_prev,
                dt_next=dt_next,
                logvar=out.get("logvar", out["logvar_f"]),
                long_gap_threshold=long_gap_threshold,
                alt_base=out.get("alt_base"),
                residual_bound=out.get("residual_bound"),
                delta_alt_pred_norm=out.get("delta_alt_pred_norm"),
                alt_gate=out.get("alt_gate"),
                teacher_scale=teacher_scale,
                risk_flag=risk_flag,
                risk_flag_teacher=risk_flag_teacher,
                segment_bucket=segment_bucket,
                anchor_pattern=anchor_pattern,
                edge_weight=edge_weight,
                pred_pos_main=out.get("pred_pos_main"),
                pred_pos_aux=out.get("pred_pos_aux"),
                pred_pos_aux_supervise_dims=out.get("pred_pos_aux_supervise_dims"),
                left_boundary_alt=left_boundary_alt_model,
                right_boundary_alt=right_boundary_alt_model,
                mu_f=out.get("mu_f"),
                mu_b=out.get("mu_b"),
                savca_alloc_p=out.get("savca_alloc_p"),
                savca_state=out.get("savca_state"),
                savca_alloc_valid=out.get("savca_alloc_valid"),
                savca_change_score=out.get("savca_change_score"),
                savca_beta=out.get("savca_beta"),
                savca_beta_floor_pred=out.get("savca_beta_floor_pred"),
                savca_target_alt_abs=target_pos[..., 2],
                savca_g_linear=out.get("savca_g_linear"),
                savca_g_savca=out.get("savca_g_savca"),
                savca_g_final=out.get("savca_g_final"),
                savca_ref_linear_abs=out.get("savca_ref_linear_abs"),
                savca_ref_savca_abs=out.get("savca_ref_savca_abs"),
                savca_ref_final_abs=out.get("savca_ref_final_abs"),
                fltp_beta=out.get("fltp_beta"),
                fltp_c=out.get("fltp_c"),
                fltp_w=out.get("fltp_w"),
                fltp_g_linear=out.get("fltp_g_linear"),
                fltp_g_sig=out.get("fltp_g_sig"),
                fltp_g_final=out.get("fltp_g_final"),
                fltp_ref_linear_abs=out.get("fltp_ref_linear_abs"),
                fltp_ref_sig_abs=out.get("fltp_ref_sig_abs"),
                fltp_ref_final_abs=out.get("fltp_ref_final_abs"),
                ssvr_pi_L=out.get("ssvr_pi_L"),
                ssvr_pi_T=out.get("ssvr_pi_T"),
                ssvr_pi_R=out.get("ssvr_pi_R"),
                ssvr_rho=out.get("ssvr_rho"),
                ssvr_state_logits=out.get("ssvr_state_logits"),
                ssvr_z_hat=out.get("ssvr_z_hat"),
                ssvr_z_linear=out.get("ssvr_z_linear"),
                alt_fwd=out.get("alt_fwd"),
                alt_bwd=out.get("alt_bwd"),
                q_pred=out.get("q_pred"),
                q_true=q_true,
                q_mask=q_mask,
                q_res_pred=out.get("q_res_pred"),
                q_res_true=q_res_true,
                q_res_mask=q_res_mask,
            )

            # ---- NaN / loss explosion detection ----
            loss_val = float(loss_dict["loss"].detach().cpu())
            if not (np.isfinite(loss_val) and loss_val < 1e9):
                split_name = "train" if train else "val"
                raise RuntimeError(
                    f"[FATAL][{split_name}] step={step_idx} loss={loss_val:.6f} "
                    f"NaN/inf/explosion detected. Aborting this model."
                )
            # ----
            if (not rel_sample_logged) and bool(getattr(model, "is_minimal_task_baseline", False)):
                valid = seq_mask > 0.5
                gap = (obs_mask <= 0.5) & valid
                gap_len = torch.clamp(dt_prev + dt_next, min=1e-6)
                tau = torch.clamp(dt_prev / gap_len, min=0.0, max=1.0)
                d_left_norm = tau
                d_right_norm = 1.0 - tau
                gap_len_ref = float(getattr(model, "proto_gap_len_ref_min", 180.0))
                gap_len_norm = torch.clamp(
                    torch.log1p(gap_len)
                    / torch.log1p(torch.tensor(gap_len_ref, device=gap_len.device, dtype=gap_len.dtype)),
                    min=0.0,
                    max=1.0,
                )
                anchor_valid = (obs_mask > 0.5) & valid
                obs_mean, obs_std = _masked_mean_std(obs_for_model, anchor_valid)
                obs_z_mean, obs_z_std = _masked_mean_std(obs_for_model[..., 2], anchor_valid)
                yl_mean, yl_std = _masked_mean_std(anchor_left_model[..., 2], valid)
                yr_mean, yr_std = _masked_mean_std(anchor_right_model[..., 2], valid)
                dz_mean, dz_std = _masked_mean_std(anchor_right_model[..., 2] - anchor_left_model[..., 2], valid)
                pred_altrel_std, true_altrel_std, _ = _masked_std_ratio(
                    out["pred_pos"][..., 2], target_for_model[..., 2], gap
                )
                lengths = seq_mask.sum(dim=1)
                print(
                    "[proto_audit] "
                    f"alt_target_mode={getattr(model, 'alt_target_mode', 'unknown')} "
                    f"obs_mean={obs_mean:.4f} obs_std={obs_std:.4f} "
                    f"obs_z_mean={obs_z_mean:.4f} obs_z_std={obs_z_std:.4f} "
                    f"y_left_z_mean={yl_mean:.4f} y_left_z_std={yl_std:.4f} "
                    f"y_right_z_mean={yr_mean:.4f} y_right_z_std={yr_std:.4f} "
                    f"delta_z_mean={dz_mean:.4f} delta_z_std={dz_std:.4f} "
                    f"tau_min={float(tau[valid].min().detach().cpu()):.4f} tau_max={float(tau[valid].max().detach().cpu()):.4f} "
                    f"d_left_norm_min={float(d_left_norm[valid].min().detach().cpu()):.4f} d_left_norm_max={float(d_left_norm[valid].max().detach().cpu()):.4f} "
                    f"d_right_norm_min={float(d_right_norm[valid].min().detach().cpu()):.4f} d_right_norm_max={float(d_right_norm[valid].max().detach().cpu()):.4f} "
                    f"gap_len_norm_mean={float(gap_len_norm[valid].mean().detach().cpu()):.4f} "
                    f"gap_len_norm_std={float(gap_len_norm[valid].std(unbiased=False).detach().cpu()):.4f} "
                    f"seq_len_min={int(lengths.min().item())} seq_len_mean={float(lengths.float().mean().item()):.2f} seq_len_max={int(lengths.max().item())} "
                    f"pred_altrel_std={pred_altrel_std:.4f} true_altrel_std={true_altrel_std:.4f}"
                )
                rel_sample_logged = True

            if train:
                optimizer.zero_grad()
                loss_dict["loss"].backward()
                # NaN gradient check after backward
                for _n, _p in model.named_parameters():
                    if _p.grad is not None and (not torch.isfinite(_p.grad).all()):
                        raise RuntimeError(
                            f"[FATAL][train] step={step_idx} param={_n} has NaN/inf grad. Aborting."
                        )
                captured_grad = False
                if (not grad_logged) and hasattr(model, "forward_net") and hasattr(model, "backward_net"):
                    fnet = model.forward_net
                    bnet = model.backward_net
                    if hasattr(fnet, "planar_head") and hasattr(fnet, "vertical_head"):
                        bw = bnet if bnet is not None else fnet
                        heads = [
                            (0, fnet.planar_head, bw.planar_head, 0),
                            (1, fnet.planar_head, bw.planar_head, 1),
                            (2, fnet.vertical_head, bw.vertical_head, 0),
                        ]
                    else:
                        bw = bnet if bnet is not None else fnet
                        if hasattr(fnet, "mu_horiz_head") and hasattr(bw, "mu_horiz_head"):
                            heads = [
                                (0, fnet.mu_horiz_head, bw.mu_horiz_head, 0),
                                (1, fnet.mu_horiz_head, bw.mu_horiz_head, 1),
                                (2, fnet.mu_alt_head, bw.mu_alt_head, 0),
                            ]
                        else:
                            heads = [
                                (0, fnet.mu_head, bw.mu_head, 0),
                                (1, fnet.mu_head, bw.mu_head, 1),
                                (2, fnet.mu_head, bw.mu_head, 2),
                            ]
                    all_ok = True
                    for _, fw_head, bw_head, row_idx in heads:
                        fw_w = getattr(fw_head, "weight", None)
                        bw_w = getattr(bw_head, "weight", None)
                        if fw_w is None or bw_w is None or fw_w.grad is None or bw_w.grad is None:
                            all_ok = False
                            break
                    if all_ok:
                        for d, fw_head, bw_head, row_idx in heads:
                            fw_w = fw_head.weight
                            bw_w = bw_head.weight
                            fw_b = getattr(fw_head, "bias", None)
                            bw_b = getattr(bw_head, "bias", None)
                            fw_row = float(torch.norm(fw_w.grad[row_idx], p=2).detach().cpu())
                            bw_row = float(torch.norm(bw_w.grad[row_idx], p=2).detach().cpu())
                            fw_bias = (
                                float(torch.abs(fw_b.grad[row_idx]).detach().cpu())
                                if fw_b is not None and fw_b.grad is not None
                                else 0.0
                            )
                            bw_bias = (
                                float(torch.abs(bw_b.grad[row_idx]).detach().cpu())
                                if bw_b is not None and bw_b.grad is not None
                                else 0.0
                            )
                            grad_snapshot[f"grad_sample_fw_mu_head_dim{d}_w_l2"] = fw_row
                            grad_snapshot[f"grad_sample_bw_mu_head_dim{d}_w_l2"] = bw_row
                            grad_snapshot[f"grad_sample_fw_mu_head_dim{d}_b_abs"] = fw_bias
                            grad_snapshot[f"grad_sample_bw_mu_head_dim{d}_b_abs"] = bw_bias
                            grad_snapshot[f"grad_sample_mu_head_dim{d}_w_l2_sum"] = fw_row + bw_row
                        captured_grad = True
                if (not grad_logged) and hasattr(model, "fusion") and hasattr(model.fusion, "mlp"):
                    fusion_total = 0.0
                    fusion_cnt = 0
                    for name, p in model.fusion.mlp.named_parameters():
                        if p.grad is None:
                            continue
                        gnorm = float(torch.norm(p.grad.detach(), p=2).cpu())
                        grad_snapshot[f"grad_fusion_{name.replace('.', '_')}_l2"] = gnorm
                        fusion_total += gnorm
                        fusion_cnt += 1
                    if fusion_cnt > 0:
                        grad_snapshot["grad_fusion_mlp_l2_total"] = fusion_total
                        grad_snapshot["grad_fusion_mlp_l2_mean"] = fusion_total / fusion_cnt
                        captured_grad = True
                if captured_grad:
                    grad_logged = True
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            _sync_if_cuda(device)
            t_loss1 = time.perf_counter()
            forward_time = t_fw1 - t_fw0
            # loss_time includes loss computation + backward + optimizer step.
            loss_time = t_loss1 - t_loss0

        _sync_if_cuda(device)
        t_metrics0 = time.perf_counter()
        pred_model_t = denormalize_coords(out["pred_pos"], target_norm_stats)
        pred_model = invert_alt_target_transform(
            pred_model_t,
            mode=alt_target_transform_mode,
            clip_value=alt_target_clip_value,
        )
        pred_latlon = restore_to_latlon(pred_model, seq_mask=seq_mask, ctx=coord_ctx)
        pred_aux_latlon = None
        pred_aux_model = None
        if out.get("pred_pos_aux") is not None:
            pred_aux_model_t = denormalize_coords(out["pred_pos_aux"], target_norm_stats)
            pred_aux_model = invert_alt_target_transform(
                pred_aux_model_t,
                mode=alt_target_transform_mode,
                clip_value=alt_target_clip_value,
            )
            pred_aux_latlon = restore_to_latlon(pred_aux_model, seq_mask=seq_mask, ctx=coord_ctx)
        pred_xy_latlon = None
        pred_xy_model = None
        if out.get("pred_xy_full") is not None:
            pred_xy_model_t = denormalize_coords(out["pred_xy_full"], target_norm_stats)
            pred_xy_model = invert_alt_target_transform(
                pred_xy_model_t,
                mode=alt_target_transform_mode,
                clip_value=alt_target_clip_value,
            )
            pred_xy_latlon = restore_to_latlon(pred_xy_model, seq_mask=seq_mask, ctx=coord_ctx)
        pred_z_latlon = None
        pred_z_model = None
        if out.get("pred_z_full") is not None:
            pred_z_model_t = denormalize_coords(out["pred_z_full"], target_norm_stats)
            pred_z_model = invert_alt_target_transform(
                pred_z_model_t,
                mode=alt_target_transform_mode,
                clip_value=alt_target_clip_value,
            )
            pred_z_latlon = restore_to_latlon(pred_z_model, seq_mask=seq_mask, ctx=coord_ctx)
        mu_f_model_t = denormalize_coords(out["mu_f"], target_norm_stats)
        mu_b_model_t = denormalize_coords(out["mu_b"], target_norm_stats)
        mu_f_model = invert_alt_target_transform(
            mu_f_model_t,
            mode=alt_target_transform_mode,
            clip_value=alt_target_clip_value,
        )
        mu_b_model = invert_alt_target_transform(
            mu_b_model_t,
            mode=alt_target_transform_mode,
            clip_value=alt_target_clip_value,
        )
        pred_latlon_f = restore_to_latlon(mu_f_model, seq_mask=seq_mask, ctx=coord_ctx)
        pred_latlon_b = restore_to_latlon(mu_b_model, seq_mask=seq_mask, ctx=coord_ctx)
        metrics = compute_metrics(
            pred_pos=pred_latlon,
            target_pos=target_pos,
            seq_mask=seq_mask,
            obs_mask=obs_mask,
            long_gap_threshold=long_gap_threshold,
        )

        totals["loss"] += float(loss_dict["loss"].detach().cpu())
        totals["loss_pos"] += float(loss_dict["loss_pos"].detach().cpu())
        totals["loss_smooth"] += float(loss_dict["loss_smooth"].detach().cpu())
        if "horizontal_loss" in loss_dict:
            totals["horizontal_loss"] += float(loss_dict["horizontal_loss"].detach().cpu())
        if "horizontal_increment_loss" in loss_dict:
            totals["horizontal_increment_loss"] += float(loss_dict["horizontal_increment_loss"].detach().cpu())
        if "step_increment_loss" in loss_dict:
            totals["step_increment_loss"] += float(loss_dict["step_increment_loss"].detach().cpu())
        if "planar_loss" in loss_dict:
            totals["planar_loss"] += float(loss_dict["planar_loss"].detach().cpu())
        if "vertical_loss" in loss_dict:
            totals["vertical_loss"] += float(loss_dict["vertical_loss"].detach().cpu())
        if "loss_xy" in loss_dict:
            totals["loss_xy"] += float(loss_dict["loss_xy"].detach().cpu())
        if "loss_z" in loss_dict:
            totals["loss_z"] += float(loss_dict["loss_z"].detach().cpu())
        for key in [
            "loss_anchor_raw",
            "loss_gap_raw",
            "loss_anchor_weighted",
            "loss_gap_weighted",
            "anchor_points",
            "gap_points",
            "anchor_raw_ratio",
            "gap_raw_ratio",
            "anchor_weighted_ratio",
            "gap_weighted_ratio",
            "cruise_gap_points",
            "cruise_weight_mean",
            "cruise_speed_smooth_loss",
            "cruise_heading_rate_loss",
            "cruise_vertical_rate_loss",
            "cruise_planar_accel_loss",
            "cruise_phys_loss",
            "multi_scale_planar_loss",
            "multi_scale_alt_loss",
            "multi_scale_points",
            "multi_scale_k5_loss",
            "multi_scale_k10_loss",
            "multi_scale_k20_loss",
            "multi_scale_k5_alt_loss",
            "multi_scale_k10_alt_loss",
            "multi_scale_k20_alt_loss",
            "multi_scale_k5_points",
            "multi_scale_k10_points",
            "multi_scale_k20_points",
            "fusion_reg_loss",
            "vertical_smooth_loss",
            "alt_bias_abs_mean",
            "alt_gate_supervision_loss",
            "alt_gate_risk_shrink_loss",
            "alt_gate_mean",
            "alt_gate_mean_risk",
            "alt_gate_mean_nonrisk",
            "alt_gate_mean_bucket_short",
            "alt_gate_mean_bucket_medium",
            "alt_gate_mean_bucket_long",
            "high_risk_gap_alt_rmse_proxy",
            "high_risk_edge_alt_jump_proxy",
            "local_spike_jump_loss",
            "local_spike_curve_loss",
            "target_rightstep2_jump_loss",
            "target_rightstep2_curve_loss",
            "var_reg_loss",
            "savca_alloc_loss",
            "savca_state_loss",
            "savca_smooth_loss",
            "savca_center_loss",
            "savca_supervised_segments",
            "savca_center_shift_mean",
            "savca_center_shift_long",
            "savca_p_entropy_mean",
            "savca_p_max_mean",
            "savca_state_conf_mean",
            "savca_shape_conf_mean",
            "savca_shape_loss",
            "savca_nonlinear_loss",
            "savca_shape_error_final",
            "savca_shape_error_linear",
            "savca_shape_error_savca",
            "savca_transition_concentration_pred",
            "savca_transition_concentration_adsb",
            "savca_transition_concentration_a1",
            "savca_shape_gain_vs_a1",
            "savca_d_nonlinear_mean",
            "savca_m_change_ratio",
            "savca_shape_error_final_change",
            "savca_d_nonlinear_change",
            "savca_fused_minus_a1_mean",
            "savca_fused_minus_a1_long",
            "savca_fused_minus_a1_change",
            "savca_change_score_loss",
            "savca_change_score_mean",
            "savca_change_score_pos_mean",
            "savca_change_score_neg_mean",
            "savca_change_score_acc",
            "savca_beta_floor_pred_mean",
            "savca_beta_mean_change",
            "savca_beta_mean_nonchange",
            "fltp_shape_loss",
            "fltp_center_loss",
            "fltp_supervised_segments",
            "fltp_center_shift_mean",
            "fltp_center_shift_long",
            "fltp_shape_error_final",
            "fltp_shape_error_linear",
            "fltp_shape_error_sig",
            "fltp_d_nonlinear_mean",
            "fltp_m_change_ratio",
            "fltp_transition_concentration_pred",
            "fltp_transition_concentration_adsb",
            "fltp_fused_minus_a1_mean",
            "fltp_fused_minus_a1_long",
            "fltp_fused_minus_a1_change",
            "ssvr_state_loss",
            "ssvr_smooth_loss",
            "ssvr_state_ce",
            "ssvr_state_acc",
            "ssvr_supervised_segments",
            "ssvr_pi_L_mean",
            "ssvr_pi_T_mean",
            "ssvr_pi_R_mean",
            "ssvr_rho_mean",
            "ssvr_state_entropy_mean",
            "ssvr_d_nonlinear_mean",
            "ssvr_shape_error_final",
            "ssvr_minus_a1_mean",
            "ssvr_m_change_ratio",
        ]:
            if key in loss_dict:
                totals[key] += float(loss_dict[key].detach().cpu())
        if "savca_beta" in out:
            beta = out["savca_beta"]
            beta_valid = out.get("savca_alloc_valid")
            beta_bucket = out.get("savca_beta_bucket_id")
            state_conf = out.get("savca_state_conf")
            p_entropy = out.get("savca_p_entropy")
            shape_conf = out.get("savca_shape_conf")
            beta_raw = out.get("savca_beta_raw")
            beta_min = out.get("savca_beta_min")
            beta_cap = out.get("savca_beta_cap")
            beta_floor_active = out.get("savca_beta_floor_active")
            state_gate = out.get("savca_state_gate")
            shape_gate = out.get("savca_shape_gate")
            confidence_gate = out.get("savca_confidence_gate")
            if beta_valid is not None:
                beta_mask = beta_valid > 0.5
                if bool(beta_mask.any()):
                    beta_vals = beta[beta_mask]
                    totals["savca_beta_mean"] += float(beta_vals.mean().detach().cpu())
                    totals["savca_beta_max"] += float(beta_vals.max().detach().cpu())
                    totals["savca_beta_p50"] += float(torch.quantile(beta_vals, 0.5).detach().cpu())
                    totals["savca_beta_p90"] += float(torch.quantile(beta_vals, 0.9).detach().cpu())
                    if beta_bucket is not None:
                        bucket_vals = beta_bucket[beta_mask]
                        for name, bucket_id in [("short", 0.0), ("medium", 1.0), ("long", 2.0)]:
                            mask = torch.isclose(bucket_vals, torch.tensor(bucket_id, device=bucket_vals.device, dtype=bucket_vals.dtype))
                            if bool(mask.any()):
                                totals[f"savca_beta_{name}"] += float(beta_vals[mask].mean().detach().cpu())
                    if state_conf is not None:
                        totals["savca_state_conf_mean_runtime"] += float(state_conf[beta_mask].mean().detach().cpu())
                    if p_entropy is not None:
                        totals["savca_p_entropy_mean_runtime"] += float(p_entropy[beta_mask].mean().detach().cpu())
                    if shape_conf is not None:
                        totals["savca_shape_conf_mean"] += float(shape_conf[beta_mask].mean().detach().cpu())
                    if beta_raw is not None:
                        totals["savca_beta_raw_mean"] += float(beta_raw[beta_mask].mean().detach().cpu())
                    if beta_min is not None:
                        totals["savca_beta_min_mean"] += float(beta_min[beta_mask].mean().detach().cpu())
                    if beta_cap is not None:
                        totals["savca_beta_cap_mean"] += float(beta_cap[beta_mask].mean().detach().cpu())
                    if beta_floor_active is not None:
                        totals["savca_beta_floor_active_ratio"] += float(
                            (beta_floor_active[beta_mask] > 0.5).float().mean().detach().cpu()
                        )
                    if state_gate is not None:
                        totals["savca_state_gate_mean"] += float(state_gate[beta_mask].mean().detach().cpu())
                    if shape_gate is not None:
                        totals["savca_shape_gate_mean"] += float(shape_gate[beta_mask].mean().detach().cpu())
                    if confidence_gate is not None:
                        totals["savca_confidence_gate_mean"] += float(confidence_gate[beta_mask].mean().detach().cpu())
        if "fltp_beta" in out:
            beta = out["fltp_beta"]
            beta_bucket = out.get("fltp_beta_bucket_id")
            c = out.get("fltp_c")
            w = out.get("fltp_w")
            beta_mask = (obs_mask <= 0.5) & (seq_mask > 0.5)
            if bool(beta_mask.any()):
                beta_vals = beta[beta_mask]
                totals["fltp_beta_mean"] += float(beta_vals.mean().detach().cpu())
                if beta_bucket is not None:
                    bucket_vals = beta_bucket[beta_mask]
                    for name, bucket_id in [("short", 0.0), ("medium", 1.0), ("long", 2.0)]:
                        mask = torch.isclose(bucket_vals, torch.tensor(bucket_id, device=bucket_vals.device, dtype=bucket_vals.dtype))
                        if bool(mask.any()):
                            totals[f"fltp_beta_{name}"] += float(beta_vals[mask].mean().detach().cpu())
                if c is not None:
                    c_vals = c[beta_mask]
                    totals["fltp_c_mean"] += float(c_vals.mean().detach().cpu())
                    totals["fltp_c_p25"] += float(torch.quantile(c_vals, 0.25).detach().cpu())
                    totals["fltp_c_p50"] += float(torch.quantile(c_vals, 0.50).detach().cpu())
                    totals["fltp_c_p75"] += float(torch.quantile(c_vals, 0.75).detach().cpu())
                if w is not None:
                    w_vals = w[beta_mask]
                    totals["fltp_w_mean"] += float(w_vals.mean().detach().cpu())
                    totals["fltp_w_p25"] += float(torch.quantile(w_vals, 0.25).detach().cpu())
                    totals["fltp_w_p50"] += float(torch.quantile(w_vals, 0.50).detach().cpu())
                    totals["fltp_w_p75"] += float(torch.quantile(w_vals, 0.75).detach().cpu())
        if "alt_bias" in out:
            totals["alt_bias_abs_mean"] += float(torch.mean(torch.abs(out["alt_bias"])).detach().cpu())
        d_dim = out["pred_pos"].shape[-1]
        for region in ["overall", "anchor", "gap", "long_gap"]:
            for d in range(d_dim):
                lk = f"{region}_dim{d}_loss"
                if lk in loss_dict:
                    totals[lk] += float(loss_dict[lk].detach().cpu())
        for d in range(d_dim):
            for lk in [f"pos_dim{d}_contrib", f"pos_dim{d}_contrib_ratio", f"total_dim{d}_contrib_ratio"]:
                if lk in loss_dict:
                    totals[lk] += float(loss_dict[lk].detach().cpu())
        valid_mask = seq_mask > 0.5
        anchor_mask = (obs_mask > 0.5) & valid_mask
        gap_mask = (obs_mask <= 0.5) & valid_mask
        long_gap_mask = _build_long_gap_mask(obs_mask=obs_mask, seq_mask=seq_mask, long_gap_threshold=long_gap_threshold)
        # Model-space dim2 audit: in u_relative_anchor mode this is alt_rel.
        altrel_err = pred_model[..., 2] - target_model_raw[..., 2]
        valid_f = valid_mask.float()
        anchor_f = anchor_mask.float()
        gap_f = gap_mask.float()
        totals["altrel_mae"] += float(((torch.abs(altrel_err) * valid_f).sum() / (valid_f.sum() + 1e-6)).detach().cpu())
        totals["altrel_rmse"] += float(
            torch.sqrt((((altrel_err**2) * valid_f).sum()) / (valid_f.sum() + 1e-6)).detach().cpu()
        )
        totals["anchor_altrel_mae"] += float(
            ((torch.abs(altrel_err) * anchor_f).sum() / (anchor_f.sum() + 1e-6)).detach().cpu()
        )
        totals["anchor_altrel_rmse"] += float(
            torch.sqrt((((altrel_err**2) * anchor_f).sum()) / (anchor_f.sum() + 1e-6)).detach().cpu()
        )
        totals["gap_altrel_mae"] += float(((torch.abs(altrel_err) * gap_f).sum() / (gap_f.sum() + 1e-6)).detach().cpu())
        totals["gap_altrel_rmse"] += float(
            torch.sqrt((((altrel_err**2) * gap_f).sum()) / (gap_f.sum() + 1e-6)).detach().cpu()
        )
        pred_altrel = pred_model[..., 2]
        true_altrel = target_model_raw[..., 2]
        pred_mean = (pred_altrel * gap_f).sum() / (gap_f.sum() + 1e-6)
        true_mean = (true_altrel * gap_f).sum() / (gap_f.sum() + 1e-6)
        pred_std = torch.sqrt((((pred_altrel - pred_mean) ** 2) * gap_f).sum() / (gap_f.sum() + 1e-6))
        true_std = torch.sqrt((((true_altrel - true_mean) ** 2) * gap_f).sum() / (gap_f.sum() + 1e-6))
        cov = (((pred_altrel - pred_mean) * (true_altrel - true_mean)) * gap_f).sum() / (gap_f.sum() + 1e-6)
        corr = cov / (pred_std * true_std + 1e-6)
        totals["altrel_pred_std"] += float(pred_std.detach().cpu())
        totals["altrel_true_std"] += float(true_std.detach().cpu())
        totals["altrel_corr"] += float(corr.detach().cpu())
        totals["altrel_bias_mean"] += float((((pred_altrel - true_altrel) * gap_f).sum() / (gap_f.sum() + 1e-6)).detach().cpu())
        main_metrics = compute_metrics(
            pred_pos=pred_latlon,
            target_pos=target_pos,
            seq_mask=seq_mask,
            obs_mask=obs_mask,
            long_gap_threshold=long_gap_threshold,
        )
        totals["pred_main_gap_lat_rmse"] += float(main_metrics["gap_dim0_rmse"])
        totals["pred_main_gap_lon_rmse"] += float(main_metrics["gap_dim1_rmse"])
        totals["pred_main_gap_alt_rmse"] += float(main_metrics["gap_dim2_rmse"])
        totals["final_gap_lat_rmse"] += float(main_metrics["gap_dim0_rmse"])
        totals["final_gap_lon_rmse"] += float(main_metrics["gap_dim1_rmse"])
        totals["final_gap_alt_rmse"] += float(main_metrics["gap_dim2_rmse"])
        if pred_xy_model is not None and pred_xy_latlon is not None:
            xy_metrics = compute_metrics(
                pred_pos=pred_xy_latlon,
                target_pos=target_pos,
                seq_mask=seq_mask,
                obs_mask=obs_mask,
                long_gap_threshold=long_gap_threshold,
            )
            totals["pred_aux_xy_gap_lat_rmse"] += float(xy_metrics["gap_dim0_rmse"])
            totals["pred_aux_xy_gap_lon_rmse"] += float(xy_metrics["gap_dim1_rmse"])
        if pred_z_model is not None and pred_z_latlon is not None:
            z_metrics = compute_metrics(
                pred_pos=pred_z_latlon,
                target_pos=target_pos,
                seq_mask=seq_mask,
                obs_mask=obs_mask,
                long_gap_threshold=long_gap_threshold,
            )
            z_altrel = pred_z_model[..., 2]
            z_mean = (z_altrel * gap_f).sum() / (gap_f.sum() + 1e-6)
            z_std = torch.sqrt((((z_altrel - z_mean) ** 2) * gap_f).sum() / (gap_f.sum() + 1e-6))
            totals["pred_zlinear_gap_alt_rmse"] += float(z_metrics["gap_dim2_rmse"])
            totals["pred_zlinear_gap_alt_mae"] += float(z_metrics["gap_dim2_mae"])
            totals["pred_z_std"] += float(z_std.detach().cpu())
        if pred_aux_model is not None and pred_aux_latlon is not None:
            aux_altrel = pred_aux_model[..., 2]
            aux_mean = (aux_altrel * gap_f).sum() / (gap_f.sum() + 1e-6)
            aux_std = torch.sqrt((((aux_altrel - aux_mean) ** 2) * gap_f).sum() / (gap_f.sum() + 1e-6))
            aux_cov = (((aux_altrel - aux_mean) * (true_altrel - true_mean)) * gap_f).sum() / (gap_f.sum() + 1e-6)
            aux_corr = aux_cov / (aux_std * true_std + 1e-6)
            aux_metrics = compute_metrics(
                pred_pos=pred_aux_latlon,
                target_pos=target_pos,
                seq_mask=seq_mask,
                obs_mask=obs_mask,
                long_gap_threshold=long_gap_threshold,
            )
            totals["aux_altrel_pred_std"] += float(aux_std.detach().cpu())
            totals["aux_altrel_corr"] += float(aux_corr.detach().cpu())
            totals["aux_gap_lat_rmse"] += float(aux_metrics["gap_dim0_rmse"])
            totals["aux_gap_lon_rmse"] += float(aux_metrics["gap_dim1_rmse"])
            totals["aux_gap_alt_rmse"] += float(aux_metrics["gap_dim2_rmse"])
            totals["aux_gap_alt_mae"] += float(aux_metrics["gap_dim2_mae"])

        if out.get("h_f") is not None and out.get("h_b") is not None:
            h_f_now = out["h_f"]
            h_b_now = out["h_b"]
            h_diff = torch.norm(h_f_now - h_b_now, dim=-1)
            totals["hbidir_diff_norm"] += float(((h_diff * valid_f).sum() / (valid_f.sum() + 1e-6)).detach().cpu())
        if out.get("gamma_z") is not None:
            totals["gamma_z"] += float(out["gamma_z"].detach().cpu())
        if out.get("beta_z_coarse") is not None:
            totals["beta_z_coarse"] += float(out["beta_z_coarse"].detach().cpu())
        if out.get("delta_h_z") is not None:
            dhz = torch.norm(out["delta_h_z"], dim=-1)
            totals["delta_h_z_norm"] += float(((dhz * valid_f).sum() / (valid_f.sum() + 1e-6)).detach().cpu())
        if out.get("delta_z_coarse") is not None:
            dzc = out["delta_z_coarse"].squeeze(-1)
            dzc_mean = (dzc * gap_f).sum() / (gap_f.sum() + 1e-6)
            dzc_std = torch.sqrt((((dzc - dzc_mean) ** 2) * gap_f).sum() / (gap_f.sum() + 1e-6))
            totals["delta_z_coarse_std"] += float(dzc_std.detach().cpu())
        if out.get("q_pred") is not None and q_true is not None and q_mask is not None:
            q_pred_now = out["q_pred"].squeeze(-1)
            q_true_now = q_true.squeeze(-1) if q_true.dim() == out["q_pred"].dim() else q_true
            q_mask_now = q_mask.squeeze(-1) if q_mask.dim() == out["q_pred"].dim() else q_mask
            q_mask_now = q_mask_now.to(device=q_pred_now.device, dtype=q_pred_now.dtype)
            q_denom = q_mask_now.sum() + 1e-6
            q_pred_mean = (q_pred_now * q_mask_now).sum() / q_denom
            q_true_mean = (q_true_now * q_mask_now).sum() / q_denom
            q_pred_std = torch.sqrt((((q_pred_now - q_pred_mean) ** 2) * q_mask_now).sum() / q_denom + 1e-6)
            q_true_std = torch.sqrt((((q_true_now - q_true_mean) ** 2) * q_mask_now).sum() / q_denom + 1e-6)
            q_cov = (((q_pred_now - q_pred_mean) * (q_true_now - q_true_mean)) * q_mask_now).sum() / q_denom
            q_corr = q_cov / (q_pred_std * q_true_std + 1e-6)
            totals["q_pred_mean"] += float(q_pred_mean.detach().cpu())
            totals["q_pred_std"] += float(q_pred_std.detach().cpu())
            totals["q_true_mean"] += float(q_true_mean.detach().cpu())
            totals["q_true_std"] += float(q_true_std.detach().cpu())
            totals["q_corr"] += float(q_corr.detach().cpu())
        if out.get("q_res_pred") is not None and q_res_true is not None and q_res_mask is not None:
            q_res_pred_now = out["q_res_pred"].squeeze(-1)
            q_res_true_now = q_res_true.squeeze(-1) if q_res_true.dim() == out["q_res_pred"].dim() else q_res_true
            q_res_mask_now = q_res_mask.squeeze(-1) if q_res_mask.dim() == out["q_res_pred"].dim() else q_res_mask
            q_res_mask_now = q_res_mask_now.to(device=q_res_pred_now.device, dtype=q_res_pred_now.dtype)
            q_res_denom = q_res_mask_now.sum() + 1e-6
            q_res_pred_mean = (q_res_pred_now * q_res_mask_now).sum() / q_res_denom
            q_res_true_mean = (q_res_true_now * q_res_mask_now).sum() / q_res_denom
            q_res_pred_std = torch.sqrt((((q_res_pred_now - q_res_pred_mean) ** 2) * q_res_mask_now).sum() / q_res_denom + 1e-6)
            q_res_true_std = torch.sqrt((((q_res_true_now - q_res_true_mean) ** 2) * q_res_mask_now).sum() / q_res_denom + 1e-6)
            q_res_cov = (((q_res_pred_now - q_res_pred_mean) * (q_res_true_now - q_res_true_mean)) * q_res_mask_now).sum() / q_res_denom
            q_res_corr = q_res_cov / (q_res_pred_std * q_res_true_std + 1e-6)
            totals["q_res_pred_mean"] += float(q_res_pred_mean.detach().cpu())
            totals["q_res_pred_std"] += float(q_res_pred_std.detach().cpu())
            totals["q_res_true_mean"] += float(q_res_true_mean.detach().cpu())
            totals["q_res_true_std"] += float(q_res_true_std.detach().cpu())
            totals["q_res_corr"] += float(q_res_corr.detach().cpu())
        if out.get("h_align") is not None:
            halign = torch.norm(out["h_align"], dim=-1)
            totals["h_align_norm"] += float(((halign * valid_f).sum() / (valid_f.sum() + 1e-6)).detach().cpu())
        if out.get("h_z") is not None:
            hz = torch.norm(out["h_z"], dim=-1)
            totals["h_z_norm"] += float(((hz * valid_f).sum() / (valid_f.sum() + 1e-6)).detach().cpu())
        if out.get("alpha_z") is not None:
            alpha_z = out["alpha_z"].squeeze(-1)
            totals["alpha_z_mean"] += float(((alpha_z * valid_f).sum() / (valid_f.sum() + 1e-6)).detach().cpu())
            alpha_mean = ((alpha_z * valid_f).sum() / (valid_f.sum() + 1e-6))
            alpha_var = ((((alpha_z - alpha_mean) ** 2) * valid_f).sum() / (valid_f.sum() + 1e-6))
            totals["alpha_z_std"] += float(torch.sqrt(alpha_var + 1e-6).detach().cpu())
            tau_now = out.get("alpha_tau")
            if tau_now is not None:
                tau_now = tau_now.squeeze(-1)
                left_mask = (tau_now < 0.25).float() * valid_f
                mid_mask = ((tau_now >= 0.25) & (tau_now <= 0.75)).float() * valid_f
                right_mask = (tau_now > 0.75).float() * valid_f
                for name, mask_now in [("left", left_mask), ("mid", mid_mask), ("right", right_mask)]:
                    denom = mask_now.sum() + 1e-6
                    totals[f"alpha_z_{name}_mean"] += float(((alpha_z * mask_now).sum() / denom).detach().cpu())
            gap_len_steps = out.get("alpha_gap_len_steps")
            if gap_len_steps is not None:
                gap_len_steps = gap_len_steps.squeeze(-1)
                short_mask = (gap_len_steps < float(long_gap_threshold)).float() * valid_f
                long_mask = (gap_len_steps >= float(long_gap_threshold)).float() * valid_f
                for name, mask_now in [("short_gap", short_mask), ("long_gap", long_mask)]:
                    denom = mask_now.sum() + 1e-6
                    totals[f"alpha_z_{name}_mean"] += float(((alpha_z * mask_now).sum() / denom).detach().cpu())

        for prefix, pred_branch_model, pred_branch_latlon in [
            ("fwd", mu_f_model, pred_latlon_f),
            ("bwd", mu_b_model, pred_latlon_b),
        ]:
            branch_altrel = pred_branch_model[..., 2]
            branch_mean = (branch_altrel * gap_f).sum() / (gap_f.sum() + 1e-6)
            branch_std = torch.sqrt((((branch_altrel - branch_mean) ** 2) * gap_f).sum() / (gap_f.sum() + 1e-6))
            branch_cov = (((branch_altrel - branch_mean) * (true_altrel - true_mean)) * gap_f).sum() / (gap_f.sum() + 1e-6)
            branch_corr = branch_cov / (branch_std * true_std + 1e-6)
            totals[f"{prefix}_altrel_pred_std"] += float(branch_std.detach().cpu())
            totals[f"{prefix}_altrel_corr"] += float(branch_corr.detach().cpu())
            branch_metrics = compute_metrics(
                pred_pos=pred_branch_latlon,
                target_pos=target_pos,
                seq_mask=seq_mask,
                obs_mask=obs_mask,
                long_gap_threshold=long_gap_threshold,
            )
            totals[f"{prefix}_gap_lat_rmse"] += float(branch_metrics["gap_dim0_rmse"])
            totals[f"{prefix}_gap_lon_rmse"] += float(branch_metrics["gap_dim1_rmse"])
            totals[f"{prefix}_gap_alt_rmse"] += float(branch_metrics["gap_dim2_rmse"])
            totals[f"{prefix}_gap_alt_mae"] += float(branch_metrics["gap_dim2_mae"])

        # Physical horizontal RMSE in restored trajectory space (meters), for checkpoint monitoring.
        pred_latlon_np = pred_latlon.detach().cpu().numpy()
        target_latlon_np = target_pos.detach().cpu().numpy()
        obs_latlon_np = obs_pos.detach().cpu().numpy()
        seq_np = seq_mask.detach().cpu().numpy()
        pred_enu_np = pred_latlon_np.copy()
        pred_enu_f_np = pred_latlon_np.copy()
        pred_enu_b_np = pred_latlon_np.copy()
        target_enu_np = target_latlon_np.copy()
        obs_enu_np = obs_latlon_np.copy()
        pred_latlon_f_np = pred_latlon_f.detach().cpu().numpy()
        pred_latlon_b_np = pred_latlon_b.detach().cpu().numpy()
        for i in range(pred_latlon_np.shape[0]):
            pred_enu_np[i] = _to_enu(pred_latlon_np[i], coord_ctx.refs[i], seq_np[i])
            pred_enu_f_np[i] = _to_enu(pred_latlon_f_np[i], coord_ctx.refs[i], seq_np[i])
            pred_enu_b_np[i] = _to_enu(pred_latlon_b_np[i], coord_ctx.refs[i], seq_np[i])
            target_enu_np[i] = _to_enu(target_latlon_np[i], coord_ctx.refs[i], seq_np[i])
            obs_enu_np[i] = _to_enu(obs_latlon_np[i], coord_ctx.refs[i], seq_np[i])
        pred_enu = torch.tensor(pred_enu_np, device=pred_model.device, dtype=pred_model.dtype)
        pred_enu_f = torch.tensor(pred_enu_f_np, device=pred_model.device, dtype=pred_model.dtype)
        pred_enu_b = torch.tensor(pred_enu_b_np, device=pred_model.device, dtype=pred_model.dtype)
        target_enu = torch.tensor(target_enu_np, device=pred_model.device, dtype=pred_model.dtype)
        obs_enu = torch.tensor(obs_enu_np, device=pred_model.device, dtype=pred_model.dtype)

        totals["overall_horizontal_rmse_m"] += _masked_horizontal_rmse_m(pred_enu, target_enu, valid_mask)
        totals["anchor_horizontal_rmse_m"] += _masked_horizontal_rmse_m(pred_enu, target_enu, anchor_mask)
        totals["gap_horizontal_rmse_m"] += _masked_horizontal_rmse_m(pred_enu, target_enu, gap_mask)
        totals["long_gap_horizontal_rmse_m"] += _masked_horizontal_rmse_m(pred_enu, target_enu, long_gap_mask)
        totals["fwd_gap_horizontal_rmse_m"] += _masked_horizontal_rmse_m(pred_enu_f, target_enu, gap_mask)
        totals["fwd_long_gap_horizontal_rmse_m"] += _masked_horizontal_rmse_m(pred_enu_f, target_enu, long_gap_mask)
        totals["bwd_gap_horizontal_rmse_m"] += _masked_horizontal_rmse_m(pred_enu_b, target_enu, gap_mask)
        totals["bwd_long_gap_horizontal_rmse_m"] += _masked_horizontal_rmse_m(pred_enu_b, target_enu, long_gap_mask)
        # Branch effect audit: does fused output materially differ from single branches?
        totals["fused_minus_fwd_mean_m"] += _masked_horizontal_mean_m(pred_enu, pred_enu_f, valid_mask)
        totals["fused_minus_bwd_mean_m"] += _masked_horizontal_mean_m(pred_enu, pred_enu_b, valid_mask)
        totals["fwd_minus_bwd_mean_m"] += _masked_horizontal_mean_m(pred_enu_f, pred_enu_b, valid_mask)
        totals["fused_minus_fwd_rmse_m"] += _masked_horizontal_rmse_m(pred_enu, pred_enu_f, valid_mask)
        totals["fused_minus_bwd_rmse_m"] += _masked_horizontal_rmse_m(pred_enu, pred_enu_b, valid_mask)
        totals["fwd_minus_bwd_rmse_m"] += _masked_horizontal_rmse_m(pred_enu_f, pred_enu_b, valid_mask)
        totals["gap_fused_minus_fwd_mean_m"] += _masked_horizontal_mean_m(pred_enu, pred_enu_f, gap_mask)
        totals["gap_fused_minus_bwd_mean_m"] += _masked_horizontal_mean_m(pred_enu, pred_enu_b, gap_mask)
        totals["gap_fwd_minus_bwd_mean_m"] += _masked_horizontal_mean_m(pred_enu_f, pred_enu_b, gap_mask)
        totals["gap_fused_minus_fwd_rmse_m"] += _masked_horizontal_rmse_m(pred_enu, pred_enu_f, gap_mask)
        totals["gap_fused_minus_bwd_rmse_m"] += _masked_horizontal_rmse_m(pred_enu, pred_enu_b, gap_mask)
        totals["gap_fwd_minus_bwd_rmse_m"] += _masked_horizontal_rmse_m(pred_enu_f, pred_enu_b, gap_mask)
        totals["anchor_obs_target_horizontal_rmse_m"] += _masked_horizontal_rmse_m(obs_enu, target_enu, anchor_mask)
        totals["anchor_pred_obs_horizontal_rmse_m"] += _masked_horizontal_rmse_m(pred_enu, obs_enu, anchor_mask)

        totals["horizontal_rmse"] += _masked_horizontal_rmse(pred_model, target_model, valid_mask)
        totals["anchor_horizontal_rmse"] += _masked_horizontal_rmse(pred_model, target_model, anchor_mask)
        totals["gap_horizontal_rmse"] += _masked_horizontal_rmse(pred_model, target_model, gap_mask)
        totals["long_gap_horizontal_rmse"] += _masked_horizontal_rmse(pred_model, target_model, long_gap_mask)
        for d in [0, 1]:
            pstd, tstd, ratio = _masked_std_ratio(pred_model[..., d], target_model[..., d], valid_mask)
            totals[f"en_dim{d}_pred_std"] += pstd
            totals[f"en_dim{d}_target_std"] += tstd
            totals[f"en_dim{d}_pred_over_target_std"] += ratio
            gpstd, gtstd, gratio = _masked_std_ratio(pred_model[..., d], target_model[..., d], gap_mask)
            totals[f"gap_en_dim{d}_pred_std"] += gpstd
            totals[f"gap_en_dim{d}_target_std"] += gtstd
            totals[f"gap_en_dim{d}_pred_over_target_std"] += gratio
        for key, value in metrics.items():
            totals[key] += float(value)

        # Gap-bucket audit on complete gap segments.
        for i in range(pred_model.shape[0]):
            vmask_i = valid_mask[i]
            amask_i = anchor_mask[i]
            if not bool(vmask_i.any()):
                continue
            gaps = _find_complete_gaps(amask_i, vmask_i)
            for (l, s, e, r) in gaps:
                gslice = slice(s, e + 1)
                glen = int(e - s + 1)
                bname = _gap_bucket_name(glen)
                key = f"gap_bucket_{bname}"
                err = torch.sqrt(
                    (pred_enu[i, gslice, 0] - target_enu[i, gslice, 0]) ** 2
                    + (pred_enu[i, gslice, 1] - target_enu[i, gslice, 1]) ** 2
                    + 1e-6
                )
                alt_err = pred_model[i, gslice, 2] - target_model_raw[i, gslice, 2]
                n_pts = float(err.numel())
                seg_acc[key]["num_segments"] += 1.0
                seg_acc[key]["num_points"] += n_pts
                seg_acc[key]["sum_err_sq"] += float((err**2).sum().detach().cpu())
                seg_acc[key]["sum_alt_err_sq"] += float((alt_err**2).sum().detach().cpu())
                seg_acc[key]["sum_mean_err"] += float(err.mean().detach().cpu())
                seg_acc[key]["sum_end_err"] += float(err[-1].detach().cpu())
                seg_err_points[key].extend([float(x) for x in err.detach().cpu().tolist()])
                seg_err_end[key].append(float(err[-1].detach().cpu()))

                pde = pred_model[i, gslice, 0]
                pdn = pred_model[i, gslice, 1]
                tde = target_model[i, gslice, 0]
                tdn = target_model[i, gslice, 1]
                seg_acc[key]["sum_pred_de"] += float(pde.sum().detach().cpu())
                seg_acc[key]["sum_pred_dn"] += float(pdn.sum().detach().cpu())
                seg_acc[key]["sum_true_de"] += float((target_enu[i, r, 0] - target_enu[i, l, 0]).detach().cpu())
                seg_acc[key]["sum_true_dn"] += float((target_enu[i, r, 1] - target_enu[i, l, 1]).detach().cpu())
                seg_acc[key]["sum_pred_de_step"] += float(pde.sum().detach().cpu())
                seg_acc[key]["sum_pred_dn_step"] += float(pdn.sum().detach().cpu())
                seg_acc[key]["sum_true_de_step"] += float(tde.sum().detach().cpu())
                seg_acc[key]["sum_true_dn_step"] += float(tdn.sum().detach().cpu())
                seg_acc[key]["sum_pred_de_step_sq"] += float((pde**2).sum().detach().cpu())
                seg_acc[key]["sum_pred_dn_step_sq"] += float((pdn**2).sum().detach().cpu())
                seg_acc[key]["sum_true_de_step_sq"] += float((tde**2).sum().detach().cpu())
                seg_acc[key]["sum_true_dn_step_sq"] += float((tdn**2).sum().detach().cpu())
                seg_acc[key]["num_steps"] += n_pts

                if out.get("fusion_weights") is not None:
                    wf_seg = out["fusion_weights"][i, gslice, 0]
                    wb_seg = out["fusion_weights"][i, gslice, 1]
                    seg_acc[key]["sum_wf_mean"] += float(wf_seg.mean().detach().cpu())
                    seg_acc[key]["sum_wb_mean"] += float(wb_seg.mean().detach().cpu())

        # Fusion behavior by rel_pos in gap points.
        if out.get("fusion_weights") is not None:
            rel_pos = dt_prev / (dt_prev + dt_next + 1e-6)
            wf_all = out["fusion_weights"][..., 0]
            wb_all = out["fusion_weights"][..., 1]
            d_fused = torch.sqrt((pred_enu[..., 0] - target_enu[..., 0]) ** 2 + (pred_enu[..., 1] - target_enu[..., 1]) ** 2 + 1e-6)
            d_fwd = torch.sqrt((pred_enu_f[..., 0] - target_enu[..., 0]) ** 2 + (pred_enu_f[..., 1] - target_enu[..., 1]) ** 2 + 1e-6)
            d_bwd = torch.sqrt((pred_enu_b[..., 0] - target_enu[..., 0]) ** 2 + (pred_enu_b[..., 1] - target_enu[..., 1]) ** 2 + 1e-6)
            for lo, hi in fusion_rel_bins:
                bkey = f"relpos_{int(lo*100):02d}_{int(hi*100):02d}"
                mask = gap_mask & (rel_pos >= lo) & (rel_pos < hi)
                n = float(mask.float().sum().detach().cpu())
                if n <= 0:
                    continue
                fusion_rel_acc[bkey]["count"] += n
                fusion_rel_acc[bkey]["sum_wf"] += float((wf_all * mask.float()).sum().detach().cpu())
                fusion_rel_acc[bkey]["sum_wb"] += float((wb_all * mask.float()).sum().detach().cpu())
                m = mask.float()
                relpos_err_acc[bkey]["count"] += n
                relpos_err_acc[bkey]["sum_sq_fused"] += float(((d_fused**2) * m).sum().detach().cpu())
                relpos_err_acc[bkey]["sum_sq_fwd"] += float(((d_fwd**2) * m).sum().detach().cpu())
                relpos_err_acc[bkey]["sum_sq_bwd"] += float(((d_bwd**2) * m).sum().detach().cpu())

        if (
            enable_verbose_diag
            and
            (not train)
            and u_relative_anchor
            and (en_relative_anchor or en_incremental)
            and coord_mode == "enu"
            and (not rel_sample_logged)
        ):
            n = int(min(10, int((seq_mask[0] > 0.5).sum().item())))
            if n > 0:
                e_anchor = (
                    torch.tensor(coord_ctx.en_anchor_tracks[0][:n, 0], device=pred_model.device)
                    if coord_ctx.en_anchor_tracks is not None
                    else torch.zeros((n,), device=pred_model.device)
                )
                n_anchor = (
                    torch.tensor(coord_ctx.en_anchor_tracks[0][:n, 1], device=pred_model.device)
                    if coord_ctx.en_anchor_tracks is not None
                    else torch.zeros((n,), device=pred_model.device)
                )
                e_rel_tgt = target_model[0, :n, 0]
                n_rel_tgt = target_model[0, :n, 1]
                e_rel_pred = pred_model[0, :n, 0]
                n_rel_pred = pred_model[0, :n, 1]
                if en_incremental:
                    e_abs_tgt = torch.zeros_like(e_rel_tgt)
                    n_abs_tgt = torch.zeros_like(n_rel_tgt)
                    for t in range(n):
                        if obs_mask[0, t] > 0.5:
                            e_abs_tgt[t] = e_anchor[t]
                            n_abs_tgt[t] = n_anchor[t]
                        elif t == 0:
                            e_abs_tgt[t] = e_anchor[t] + e_rel_tgt[t]
                            n_abs_tgt[t] = n_anchor[t] + n_rel_tgt[t]
                        else:
                            e_abs_tgt[t] = e_abs_tgt[t - 1] + e_rel_tgt[t]
                            n_abs_tgt[t] = n_abs_tgt[t - 1] + n_rel_tgt[t]
                    e_before = torch.zeros_like(e_rel_pred)
                    n_before = torch.zeros_like(n_rel_pred)
                    for t in range(n):
                        if obs_mask[0, t] > 0.5:
                            e_before[t] = e_anchor[t]
                            n_before[t] = n_anchor[t]
                        elif t == 0:
                            e_before[t] = e_anchor[t] + e_rel_pred[t]
                            n_before[t] = n_anchor[t] + n_rel_pred[t]
                        else:
                            e_before[t] = e_before[t - 1] + e_rel_pred[t]
                            n_before[t] = n_before[t - 1] + n_rel_pred[t]
                else:
                    e_abs_tgt = e_rel_tgt + e_anchor
                    n_abs_tgt = n_rel_tgt + n_anchor
                    e_before = e_rel_pred + e_anchor
                    n_before = n_rel_pred + n_anchor

                pred_latlon_np = pred_latlon[0, :n, :].detach().cpu().numpy()
                seq_np = seq_mask[0, :n].detach().cpu().numpy()
                en_after = _to_enu(pred_latlon_np, coord_ctx.refs[0], seq_np)

                en_tgt_lbl = "dE_tgt,dN_tgt" if en_incremental else "E_rel_tgt,N_rel_tgt"
                en_pred_lbl = "dE_pred,dN_pred" if en_incremental else "E_rel_pred,N_rel_pred"
                print(
                    "[rel_en_sample] t,obs_mask,"
                    f"E_abs_tgt,N_abs_tgt,E_anchor,N_anchor,{en_tgt_lbl},{en_pred_lbl},"
                    "E_before,N_before,E_after,N_after"
                )
                for t in range(n):
                    print(
                        f"[rel_en_sample] {t},"
                        f"{float(obs_mask[0,t].detach().cpu()):.1f},"
                        f"{float(e_abs_tgt[t].detach().cpu()):.3f},{float(n_abs_tgt[t].detach().cpu()):.3f},"
                        f"{float(e_anchor[t].detach().cpu()):.3f},{float(n_anchor[t].detach().cpu()):.3f},"
                        f"{float(e_rel_tgt[t].detach().cpu()):.3f},{float(n_rel_tgt[t].detach().cpu()):.3f},"
                        f"{float(e_rel_pred[t].detach().cpu()):.3f},{float(n_rel_pred[t].detach().cpu()):.3f},"
                        f"{float(e_before[t].detach().cpu()):.3f},{float(n_before[t].detach().cpu()):.3f},"
                        f"{float(en_after[t,0]):.3f},{float(en_after[t,1]):.3f}"
                    )
                anchor_i = (obs_mask[0, : int((seq_mask[0] > 0.5).sum().item())] > 0.5).nonzero(as_tuple=True)[0]
                if anchor_i.numel() > 0:
                    tgt_e = target_model[0, anchor_i, 0].abs().mean()
                    tgt_n = target_model[0, anchor_i, 1].abs().mean()
                    pred_e = pred_model[0, anchor_i, 0].abs().mean()
                    pred_n = pred_model[0, anchor_i, 1].abs().mean()
                    totals["anchor_rel_target_e_abs_mean"] += float(tgt_e.detach().cpu())
                    totals["anchor_rel_target_n_abs_mean"] += float(tgt_n.detach().cpu())
                    totals["anchor_rel_pred_e_abs_mean"] += float(pred_e.detach().cpu())
                    totals["anchor_rel_pred_n_abs_mean"] += float(pred_n.detach().cpu())
                    print(
                        "[rel_en_anchor_check] "
                        f"target_abs_mean=(E:{float(tgt_e.detach().cpu()):.6f},N:{float(tgt_n.detach().cpu()):.6f}) "
                        f"pred_abs_mean=(E:{float(pred_e.detach().cpu()):.6f},N:{float(pred_n.detach().cpu()):.6f})"
                    )
            rel_sample_logged = True
        if enable_verbose_diag and (not train) and (not gap_audit_logged):
            i = 0
            vmask_i = (seq_mask[i] > 0.5)
            amask_i = (obs_mask[i] > 0.5) & vmask_i
            gmask_i = (~amask_i) & vmask_i
            pred_e = pred_enu[i, :, 0]
            pred_n = pred_enu[i, :, 1]
            tgt_e = target_enu[i, :, 0]
            tgt_n = target_enu[i, :, 1]
            perr = torch.sqrt((pred_e - tgt_e) ** 2 + (pred_n - tgt_n) ** 2 + 1e-6)
            wf = out.get("fusion_weights")
            wf_i = wf[i, :, 0] if wf is not None else torch.zeros_like(perr)
            wb_i = wf[i, :, 1] if wf is not None else torch.zeros_like(perr)
            t_show = int(min(12, int(vmask_i.sum().item())))
            print("[gap_audit_point] sample_id=%s flight_id=%s" % (batch["sample_id"][0], batch["flight_id"][0]))
            print("[gap_audit_point] t,valid,anchor,gap,pred_E,pred_N,tgt_E,tgt_N,err_m,pred_dE,pred_dN,tgt_dE,tgt_dN,wf,wb")
            for t in range(t_show):
                print(
                    "[gap_audit_point] %d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f"
                    % (
                        t,
                        int(vmask_i[t].item()),
                        int(amask_i[t].item()),
                        int(gmask_i[t].item()),
                        float(pred_e[t].detach().cpu()),
                        float(pred_n[t].detach().cpu()),
                        float(tgt_e[t].detach().cpu()),
                        float(tgt_n[t].detach().cpu()),
                        float(perr[t].detach().cpu()),
                        float(pred_model[i, t, 0].detach().cpu()),
                        float(pred_model[i, t, 1].detach().cpu()),
                        float(target_model[i, t, 0].detach().cpu()),
                        float(target_model[i, t, 1].detach().cpu()),
                        float(wf_i[t].detach().cpu()),
                        float(wb_i[t].detach().cpu()),
                    )
                )
            gaps = _find_complete_gaps(amask_i, vmask_i)
            if gaps:
                print("[gap_audit_seg] left_anchor,gap_start,gap_end,right_anchor,gap_len,mean_err_m,end_err_m,pred_sum_dE,pred_sum_dN,true_sum_dE,true_sum_dN")
            for (l, s, e, r) in gaps[:8]:
                gslice = slice(s, e + 1)
                mean_err = float(perr[gslice].mean().detach().cpu())
                end_err = float(perr[e].detach().cpu())
                pred_sum_de = float(pred_model[i, gslice, 0].sum().detach().cpu())
                pred_sum_dn = float(pred_model[i, gslice, 1].sum().detach().cpu())
                true_sum_de = float((tgt_e[r] - tgt_e[l]).detach().cpu())
                true_sum_dn = float((tgt_n[r] - tgt_n[l]).detach().cpu())
                print(
                    "[gap_audit_seg] %d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f"
                    % (l, s, e, r, e - s + 1, mean_err, end_err, pred_sum_de, pred_sum_dn, true_sum_de, true_sum_dn)
                )
            gap_audit_logged = True
        w = out.get("fusion_weights")
        w_detail = out.get("fusion_weights_detail")
        if w is not None:
            valid = seq_mask.unsqueeze(-1)
            w_f = w[..., 0:1]
            w_b = w[..., 1:2]
            denom = valid.sum() + 1e-6
            wf_mean = float(((w_f * valid).sum() / denom).detach().cpu())
            wb_mean = float(((w_b * valid).sum() / denom).detach().cpu())
            wf2_mean = float((((w_f**2) * valid).sum() / denom).detach().cpu())
            wb2_mean = float((((w_b**2) * valid).sum() / denom).detach().cpu())
            wf_std = max(0.0, wf2_mean - wf_mean * wf_mean) ** 0.5
            wb_std = max(0.0, wb2_mean - wb_mean * wb_mean) ** 0.5
            valid_mask = seq_mask > 0.5
            if valid_mask.any():
                wf_vals = w[..., 0][valid_mask]
                wb_vals = w[..., 1][valid_mask]
                totals["wf_min"] += float(wf_vals.min().detach().cpu())
                totals["wf_max"] += float(wf_vals.max().detach().cpu())
                totals["wb_min"] += float(wb_vals.min().detach().cpu())
                totals["wb_max"] += float(wb_vals.max().detach().cpu())
            totals["wf_mean"] += wf_mean
            totals["wf_std"] += wf_std
            totals["wb_mean"] += wb_mean
            totals["wb_std"] += wb_std
            tau = dt_prev / (dt_prev + dt_next + 1e-6)
            gap_valid = gap_mask & valid_mask
            for name, m in [
                ("left", gap_valid & (tau < 0.25)),
                ("mid", gap_valid & (tau >= 0.25) & (tau <= 0.75)),
                ("right", gap_valid & (tau > 0.75)),
            ]:
                if bool(m.any()):
                    wf_m = w[..., 0][m]
                    totals[f"wf_{name}_mean"] += float(wf_m.mean().detach().cpu())
                    totals[f"wf_{name}_std"] += float(wf_m.std(unbiased=False).detach().cpu())
                else:
                    totals[f"wf_{name}_mean"] += 0.0
                    totals[f"wf_{name}_std"] += 0.0
            if w_detail is not None:
                for name, m in [
                    ("left", gap_valid & (tau < 0.25)),
                    ("mid", gap_valid & (tau >= 0.25) & (tau <= 0.75)),
                    ("right", gap_valid & (tau > 0.75)),
                ]:
                    if not bool(m.any()):
                        continue
                    if w_detail.shape[-2] == 2:
                        wf_xy = w_detail[..., 0, 0][m]
                        wf_z = w_detail[..., 1, 0][m]
                        totals[f"wf_xy_{name}_mean"] += float(wf_xy.mean().detach().cpu())
                        totals[f"wf_z_{name}_mean"] += float(wf_z.mean().detach().cpu())
                    elif w_detail.shape[-2] == 3:
                        wf_lat = w_detail[..., 0, 0][m]
                        wf_lon = w_detail[..., 1, 0][m]
                        wf_alt = w_detail[..., 2, 0][m]
                        totals[f"wf_lat_{name}_mean"] += float(wf_lat.mean().detach().cpu())
                        totals[f"wf_lon_{name}_mean"] += float(wf_lon.mean().detach().cpu())
                        totals[f"wf_alt_{name}_mean"] += float(wf_alt.mean().detach().cpu())
        _sync_if_cuda(device)
        t_metrics1 = time.perf_counter()
        metrics_time = t_metrics1 - t_metrics0
        batch_end = time.perf_counter()
        model_time = batch_end - model_t0
        step_time = batch_end - batch_start
        totals["data_time_sec"] += float(data_time)
        totals["model_time_sec"] += float(model_time)
        totals["forward_time_sec"] += float(forward_time)
        totals["loss_time_sec"] += float(loss_time)
        totals["metrics_time_sec"] += float(metrics_time)
        totals["step_time_sec"] += float(step_time)

        # Risk-aware altitude branch audit (validation-first, non-intrusive).
        if "dms_alt_delta_candidate" in out and "dms_alt_delta" in out:
            cand = out["dms_alt_delta_candidate"]
            used = out["dms_alt_delta"]
            gmask = gap_mask.float()
            totals["delta_candidate_mean_abs"] += float((torch.abs(cand) * gmask).sum().detach().cpu() / (gmask.sum().detach().cpu() + 1e-6))
            totals["delta_used_mean_abs"] += float((torch.abs(used) * gmask).sum().detach().cpu() / (gmask.sum().detach().cpu() + 1e-6))
            if bool((gmask > 0).any()):
                totals["bounded_residual_max_abs"] += float(torch.max(torch.abs(used[gmask > 0])).detach().cpu())
            else:
                totals["bounded_residual_max_abs"] += 0.0
            totals["bounded_residual_mean_abs"] += float((torch.abs(used) * gmask).sum().detach().cpu() / (gmask.sum().detach().cpu() + 1e-6))
            if out.get("alt_gate") is not None:
                gate_bt = out["alt_gate"]
                totals["gate_mean"] += float((gate_bt * gmask).sum().detach().cpu() / (gmask.sum().detach().cpu() + 1e-6))
            if out.get("left_edge_wrong_direction_mask") is not None:
                wrong_bt = out["left_edge_wrong_direction_mask"].to(used.dtype)
                totals["left_edge_wrong_direction_ratio"] += float((wrong_bt * gmask).sum().detach().cpu() / (gmask.sum().detach().cpu() + 1e-6))

            # By risk_level bucket from dataset metadata (string labels).
            risk_levels = batch.get("risk_level", None)
            if risk_levels is not None and isinstance(risk_levels, list):
                for rl in ("high", "medium", "low"):
                    idx = [ii for ii, v in enumerate(risk_levels) if str(v).lower() == rl]
                    if not idx:
                        continue
                    idx_t = torch.tensor(idx, device=used.device, dtype=torch.long)
                    g_rl = gmask.index_select(0, idx_t)
                    cand_rl = cand.index_select(0, idx_t)
                    used_rl = used.index_select(0, idx_t)
                    denom = g_rl.sum() + 1e-6
                    totals[f"gate_{rl}_count"] += float(len(idx))
                    totals[f"delta_candidate_mean_abs_{rl}"] += float(((torch.abs(cand_rl) * g_rl).sum() / denom).detach().cpu())
                    totals[f"delta_used_mean_abs_{rl}"] += float(((torch.abs(used_rl) * g_rl).sum() / denom).detach().cpu())
                    totals[f"bounded_residual_mean_abs_{rl}"] += float(((torch.abs(used_rl) * g_rl).sum() / denom).detach().cpu())
                    if bool((g_rl > 0).any()):
                        totals[f"bounded_residual_max_abs_{rl}"] += float(torch.max(torch.abs(used_rl[g_rl > 0])).detach().cpu())
                    else:
                        totals[f"bounded_residual_max_abs_{rl}"] += 0.0
                    if out.get("alt_gate") is not None:
                        gate_rl = out["alt_gate"].index_select(0, idx_t)
                        totals[f"gate_mean_{rl}"] += float(((gate_rl * g_rl).sum() / denom).detach().cpu())
                    if out.get("left_edge_wrong_direction_mask") is not None:
                        wrong_rl = out["left_edge_wrong_direction_mask"].index_select(0, idx_t).to(used.dtype)
                        totals[f"left_edge_wrong_direction_ratio_{rl}"] += float(((wrong_rl * g_rl).sum() / denom).detach().cpu())
        if heartbeat_enabled and heartbeat_interval > 0 and (step_idx % heartbeat_interval == 0):
            split_name = "train" if train else "val"
            print(
                f"[heartbeat][{split_name}] step={step_idx} "
                f"loss={float(loss_dict['loss'].detach().cpu()):.6f} "
                f"data_time={data_time:.4f}s forward={forward_time:.4f}s "
                f"loss_time={loss_time:.4f}s metrics={metrics_time:.4f}s step={step_time:.4f}s"
            )
        prev_batch_end = batch_end
        count += 1

    if count == 0:
        return {}
    out_stats = {k: v / count for k, v in totals.items()}
    # finalize gap-bucket audit stats
    for bname, acc in seg_acc.items():
        n_seg = acc.get("num_segments", 0.0)
        n_pts = acc.get("num_points", 0.0)
        n_steps = acc.get("num_steps", 0.0)
        if n_seg <= 0 or n_pts <= 0:
            continue
        out_stats[f"{bname}_num_segments"] = n_seg
        out_stats[f"{bname}_horizontal_rmse_m"] = (acc.get("sum_err_sq", 0.0) / max(n_pts, 1e-6)) ** 0.5
        out_stats[f"{bname}_altrel_rmse"] = (acc.get("sum_alt_err_sq", 0.0) / max(n_pts, 1e-6)) ** 0.5
        out_stats[f"{bname}_mean_err_m"] = acc.get("sum_mean_err", 0.0) / max(n_seg, 1e-6)
        out_stats[f"{bname}_end_err_m"] = acc.get("sum_end_err", 0.0) / max(n_seg, 1e-6)
        out_stats[f"{bname}_point_count"] = n_pts
        out_stats[f"{bname}_pred_sum_de"] = acc.get("sum_pred_de", 0.0) / max(n_seg, 1e-6)
        out_stats[f"{bname}_pred_sum_dn"] = acc.get("sum_pred_dn", 0.0) / max(n_seg, 1e-6)
        out_stats[f"{bname}_true_sum_de"] = acc.get("sum_true_de", 0.0) / max(n_seg, 1e-6)
        out_stats[f"{bname}_true_sum_dn"] = acc.get("sum_true_dn", 0.0) / max(n_seg, 1e-6)
        # step-level dE/dN moments inside bucket
        pred_de_mean = acc.get("sum_pred_de_step", 0.0) / max(n_steps, 1e-6)
        pred_dn_mean = acc.get("sum_pred_dn_step", 0.0) / max(n_steps, 1e-6)
        true_de_mean = acc.get("sum_true_de_step", 0.0) / max(n_steps, 1e-6)
        true_dn_mean = acc.get("sum_true_dn_step", 0.0) / max(n_steps, 1e-6)
        out_stats[f"{bname}_pred_de_mean"] = pred_de_mean
        out_stats[f"{bname}_pred_dn_mean"] = pred_dn_mean
        out_stats[f"{bname}_true_de_mean"] = true_de_mean
        out_stats[f"{bname}_true_dn_mean"] = true_dn_mean
        out_stats[f"{bname}_pred_de_std"] = max(
            0.0, acc.get("sum_pred_de_step_sq", 0.0) / max(n_steps, 1e-6) - pred_de_mean * pred_de_mean
        ) ** 0.5
        out_stats[f"{bname}_pred_dn_std"] = max(
            0.0, acc.get("sum_pred_dn_step_sq", 0.0) / max(n_steps, 1e-6) - pred_dn_mean * pred_dn_mean
        ) ** 0.5
        out_stats[f"{bname}_true_de_std"] = max(
            0.0, acc.get("sum_true_de_step_sq", 0.0) / max(n_steps, 1e-6) - true_de_mean * true_de_mean
        ) ** 0.5
        out_stats[f"{bname}_true_dn_std"] = max(
            0.0, acc.get("sum_true_dn_step_sq", 0.0) / max(n_steps, 1e-6) - true_dn_mean * true_dn_mean
        ) ** 0.5
        out_stats[f"{bname}_wf_mean"] = acc.get("sum_wf_mean", 0.0) / max(n_seg, 1e-6)
        out_stats[f"{bname}_wb_mean"] = acc.get("sum_wb_mean", 0.0) / max(n_seg, 1e-6)
        if seg_err_points.get(bname):
            pts = np.asarray(seg_err_points[bname], dtype=np.float64)
            out_stats[f"{bname}_point_err_median_m"] = float(np.median(pts))
            out_stats[f"{bname}_point_err_q90_m"] = float(np.quantile(pts, 0.9))
        else:
            out_stats[f"{bname}_point_err_median_m"] = 0.0
            out_stats[f"{bname}_point_err_q90_m"] = 0.0
        if seg_err_end.get(bname):
            ends = np.asarray(seg_err_end[bname], dtype=np.float64)
            out_stats[f"{bname}_end_err_median_m"] = float(np.median(ends))
            out_stats[f"{bname}_end_err_q90_m"] = float(np.quantile(ends, 0.9))
        else:
            out_stats[f"{bname}_end_err_median_m"] = 0.0
            out_stats[f"{bname}_end_err_q90_m"] = 0.0
    # finalize fusion-relpos stats
    for bkey, acc in fusion_rel_acc.items():
        n = acc.get("count", 0.0)
        if n <= 0:
            continue
        out_stats[f"{bkey}_count"] = n
        out_stats[f"{bkey}_wf_mean"] = acc.get("sum_wf", 0.0) / n
        out_stats[f"{bkey}_wb_mean"] = acc.get("sum_wb", 0.0) / n
    for bkey, acc in relpos_err_acc.items():
        n = acc.get("count", 0.0)
        if n <= 0:
            continue
        out_stats[f"{bkey}_fused_rmse_m"] = (acc.get("sum_sq_fused", 0.0) / n) ** 0.5
        out_stats[f"{bkey}_fwd_rmse_m"] = (acc.get("sum_sq_fwd", 0.0) / n) ** 0.5
        out_stats[f"{bkey}_bwd_rmse_m"] = (acc.get("sum_sq_bwd", 0.0) / n) ** 0.5
    fusion_reg_lambda = float(getattr(criterion, "fusion_reg_lambda", 0.0))
    cruise_lambda = float(getattr(criterion, "lambda_cruise_phys", 0.0))
    cruise_vertical_w = float(getattr(criterion, "cruise_vertical_rate_weight", 0.0))
    if out_stats.get("loss", 0.0) > 0.0:
        out_stats["fusion_reg_over_total"] = (
            fusion_reg_lambda * out_stats.get("fusion_reg_loss", 0.0)
        ) / (out_stats["loss"] + 1e-6)
        out_stats["cruise_phys_over_total"] = (
            cruise_lambda * out_stats.get("cruise_phys_loss", 0.0)
        ) / (out_stats["loss"] + 1e-6)
        out_stats["cruise_vertical_rate_over_total"] = (
            cruise_lambda * cruise_vertical_w * out_stats.get("cruise_vertical_rate_loss", 0.0)
        ) / (out_stats["loss"] + 1e-6)
    else:
        out_stats["fusion_reg_over_total"] = 0.0
        out_stats["cruise_phys_over_total"] = 0.0
        out_stats["cruise_vertical_rate_over_total"] = 0.0
    out_stats.update(grad_snapshot)
    return out_stats
