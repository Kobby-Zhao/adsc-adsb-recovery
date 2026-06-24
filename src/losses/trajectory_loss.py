from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


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
    return long_gap


def _build_gap_edge_mask(gap_mask: torch.Tensor, edge_steps: int) -> torch.Tensor:
    if edge_steps <= 0:
        return torch.zeros_like(gap_mask, dtype=torch.bool)
    edge_mask = torch.zeros_like(gap_mask, dtype=torch.bool)
    bsz, t_len = gap_mask.shape
    for i in range(bsz):
        t = 0
        while t < t_len:
            if not gap_mask[i, t]:
                t += 1
                continue
            start = t
            while t < t_len and gap_mask[i, t]:
                t += 1
            end = t
            left_end = min(start + edge_steps, end)
            right_start = max(start, end - edge_steps)
            edge_mask[i, start:left_end] = True
            edge_mask[i, right_start:end] = True
    return edge_mask


def _build_gap_edge_pair_mask(gap_mask: torch.Tensor, edge_steps: int) -> torch.Tensor:
    if edge_steps <= 0 or gap_mask.shape[1] <= 1:
        return torch.zeros_like(gap_mask[:, 1:], dtype=torch.bool)
    pair_mask = torch.zeros_like(gap_mask[:, 1:], dtype=torch.bool)
    bsz, t_len = gap_mask.shape
    for i in range(bsz):
        t = 0
        while t < t_len:
            if not gap_mask[i, t]:
                t += 1
                continue
            start = t
            while t < t_len and gap_mask[i, t]:
                t += 1
            end = t
            if end - start <= 1:
                continue
            left_pairs_end = min(start + edge_steps, end - 1)
            right_pairs_start = max(start, end - edge_steps - 1)
            pair_mask[i, start:left_pairs_end] = True
            pair_mask[i, right_pairs_start : end - 1] = True
    return pair_mask


def _build_gap_edge_second_mask(gap_mask: torch.Tensor, edge_steps: int) -> torch.Tensor:
    """Mask for second-difference centers near gap edges.

    Output shape is [B, T-2], corresponding to center index t in [1, T-2].
    """
    if edge_steps <= 0 or gap_mask.shape[1] <= 2:
        return torch.zeros_like(gap_mask[:, 1:-1], dtype=torch.bool)
    center_mask = torch.zeros_like(gap_mask[:, 1:-1], dtype=torch.bool)
    bsz, t_len = gap_mask.shape
    for i in range(bsz):
        t = 0
        while t < t_len:
            if not gap_mask[i, t]:
                t += 1
                continue
            start = t
            while t < t_len and gap_mask[i, t]:
                t += 1
            end = t - 1
            if end - start + 1 <= 2:
                continue
            l0 = start + 1
            l1 = min(start + edge_steps, end - 1)
            r0 = max(start + 1, end - edge_steps)
            r1 = end - 1
            if l0 <= l1:
                center_mask[i, l0 - 1 : l1] = True
            if r0 <= r1:
                center_mask[i, r0 - 1 : r1] = True
    return center_mask


def _build_anchor_boundary_mask(anchor_mask: torch.Tensor, gap_mask: torch.Tensor) -> torch.Tensor:
    """Anchor points adjacent to gap boundaries."""
    if anchor_mask.shape[1] <= 1:
        return anchor_mask
    prev_gap = torch.zeros_like(gap_mask, dtype=torch.bool)
    next_gap = torch.zeros_like(gap_mask, dtype=torch.bool)
    prev_gap[:, 1:] = gap_mask[:, :-1]
    next_gap[:, :-1] = gap_mask[:, 1:]
    return anchor_mask & (prev_gap | next_gap)


def _build_gap_first_second_masks(gap_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return masks for first and second steps inside each contiguous gap run."""
    first = torch.zeros_like(gap_mask, dtype=torch.bool)
    second = torch.zeros_like(gap_mask, dtype=torch.bool)
    bsz, t_len = gap_mask.shape
    for i in range(bsz):
        t = 0
        while t < t_len:
            if not bool(gap_mask[i, t]):
                t += 1
                continue
            s = t
            while t < t_len and bool(gap_mask[i, t]):
                t += 1
            e = t
            first[i, s] = True
            if s + 1 < e:
                second[i, s + 1] = True
    return first, second


def _build_gap_rightstep2_pair_mask(gap_mask: torch.Tensor) -> torch.Tensor:
    """Mask for pairwise diff at right-step2 location of each contiguous gap.

    Output shape [B, T-1], index j corresponds to |h[j+1]-h[j]|.
    For a gap run [s, e) (e exclusive), right-step2 jump is j = e-2.
    """
    pair = torch.zeros_like(gap_mask[:, 1:], dtype=torch.bool)
    bsz, t_len = gap_mask.shape
    for i in range(bsz):
        t = 0
        while t < t_len:
            if not bool(gap_mask[i, t]):
                t += 1
                continue
            s = t
            while t < t_len and bool(gap_mask[i, t]):
                t += 1
            e = t  # exclusive
            if (e - s) >= 2:
                j = e - 2
                if 0 <= j < (t_len - 1):
                    pair[i, j] = True
    return pair


def _build_gap_rightstep2_second_diff_mask(gap_mask: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
    """Mask for second-difference centered at right-step1 (uses right boundary point).

    For gap run [s, e), center is c=e-1; this uses points (e-2, e-1, e).
    Output shape [B, T-2], index k corresponds to center c=k+1.
    """
    mask = torch.zeros_like(gap_mask[:, 1:-1], dtype=torch.bool)
    bsz, t_len = gap_mask.shape
    valid = seq_mask > 0.5
    for i in range(bsz):
        t = 0
        while t < t_len:
            if not bool(gap_mask[i, t]):
                t += 1
                continue
            s = t
            while t < t_len and bool(gap_mask[i, t]):
                t += 1
            e = t  # exclusive
            if (e - s) >= 2 and e < t_len and bool(valid[i, e]):
                c = e - 1
                k = c - 1
                if 0 <= k < (t_len - 2):
                    mask[i, k] = True
    return mask


def _build_gap_rightstep2_point_mask(gap_mask: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
    """Mask on point index right_step2 (=t-2) of each contiguous gap. Shape [B, T]."""
    pair = _build_gap_rightstep2_pair_mask(gap_mask)
    point = torch.zeros_like(gap_mask, dtype=torch.bool)
    point[:, :-1] = pair
    return point & (seq_mask > 0.5)


def _bucket_name_to_id(name: str) -> int:
    m = {"short": 0, "medium": 1, "long": 2}
    return int(m.get(str(name).lower(), 1))


def _pattern_name_to_id(name: str) -> int:
    m = {"two_anchor": 0, "asymmetric": 1, "sparse_context": 2}
    return int(m.get(str(name).lower(), 0))


class TrajectoryLoss(nn.Module):
    def __init__(
        self,
        anchor_weight: float = 1.0,
        gap_weight: float = 1.0,
        lambda_pos: float = 1.0,
        lambda_smooth: float = 0.1,
        lambda_unc: float = 0.0,
        dim_weights: list[float] | tuple[float, ...] | None = None,
        alpha_vertical: float = 1.0,
        exo_feature_names: list[str] | None = None,
        lambda_cruise_phys: float = 0.0,
        cruise_speed_smooth_weight: float = 1.0,
        cruise_heading_rate_weight: float = 1.0,
        cruise_vertical_rate_weight: float = 1.0,
        cruise_planar_accel_weight: float = 1.0,
        cruise_max_abs_vertical_rate: float = 300.0,
        cruise_max_speed_delta: float = 30.0,
        cruise_max_heading_rate: float = 5.0,
        cruise_quality_weight_gain: float = 1.0,
        lambda_multi_scale: float = 0.0,
        multi_scale_scales: list[int] | tuple[int, ...] | None = None,
        multi_scale_include_alt: bool = False,
        fusion_reg_lambda: float = 0.0,
        fusion_reg_long_gap_weight: float = 1.0,
        gap_alt_weight: float = 1.0,
        lambda_vertical_smooth: float = 0.0,
        lambda_alt_residual: float = 0.0,
        lambda_alt_absolute_aux: float = 0.0,
        alt_edge_steps: int = 0,
        alt_edge_weight: float = 1.0,
        lambda_alt_edge_delta: float = 0.0,
        lambda_anchor_consistency: float = 0.0,
        anchor_boundary_weight: float = 2.0,
        lambda_alt_edge_first_diff: float = 0.0,
        lambda_alt_edge_second_diff: float = 0.0,
        lambda_alt_segment_bound: float = 0.0,
        lambda_alt_vertical_rate_penalty: float = 0.0,
        lambda_alt_boundary_anchor: float = 0.0,
        alt_vertical_rate_max: float = 300.0,
        segment_boundary_short_len: int = 15,
        segment_disturbed_alt_std_threshold: float = 120.0,
        alt_residual_cap_stable: float = 300.0,
        alt_residual_cap_disturbed: float = 180.0,
        alt_residual_cap_boundary: float = 120.0,
        alt_residual_cap_short: float = 100.0,
        lambda_alt_gate_supervision: float = 0.0,
        lambda_alt_gate_risk_shrink: float = 0.0,
        alt_gate_risk_target: float = 0.35,
        use_first_step_anchor_loss: bool = False,
        first_step_anchor_lambda: float = 0.0,
        use_second_step_anchor_loss: bool = False,
        second_step_anchor_lambda: float = 0.0,
        use_local_spike_loss: bool = False,
        local_spike_target_bucket: str = "medium",
        local_spike_target_pattern: str = "two_anchor",
        local_spike_use_rightstep2: bool = True,
        local_spike_use_second_diff: bool = True,
        local_spike_lambda_jump: float = 0.0,
        local_spike_lambda_curve: float = 0.0,
        use_targeted_rightstep2_loss: bool = False,
        target_bucket: str = "medium",
        target_anchor_pattern: str = "two_anchor",
        use_target_jump_loss: bool = True,
        use_target_curve_loss: bool = True,
        target_jump_lambda: float = 0.0,
        target_curve_lambda: float = 0.0,
        use_target_value_rightstep2_loss: bool = False,
        target_value_lambda: float = 0.0,
        target_interp_lambda: float = 0.5,
        use_target_rightstep2_boundary_pull: bool = False,
        target_rightstep2_boundary_pull_lambda: float = 0.0,
        # Variance regularization to prevent collapse (pred_std << true_std in gaps)
        lambda_var_reg: float = 0.0,
        var_reg_min_ratio: float = 0.3,
        # Auxiliary altitude supervision on backbone intermediate predictions.
        # Forces forward/backward LSTM to learn altitude features independently
        # of the E/N-dominated main loss.  Only active on gap points.
        lambda_alt_aux: float = 0.0,
        lambda_aux: float = 0.0,
        lambda_vprog: float = 0.0,
        vprog_enable_abs_dz_min: float = 100.0,
        lambda_vprog_res: float = 0.0,
        vprog_res_enable_abs_dz_min: float = 300.0,
        lambda_savca_alloc: float = 0.0,
        lambda_savca_state: float = 0.0,
        lambda_savca_smooth: float = 0.0,
        lambda_savca_center: float = 0.0,
        lambda_savca_final_shape: float = 0.0,
        lambda_savca_nonlinear: float = 0.0,
        lambda_savca_change_score: float = 0.0,
        lambda_ssvr_state: float = 0.0,
        lambda_ssvr_smooth: float = 0.0,
        ssvr_state_plateau_threshold: float = 0.15,
        lambda_fltp_shape: float = 0.0,
        lambda_fltp_center: float = 0.0,
        savca_alloc_min_anchor_delta_m: float = 30.0,
        savca_state_min_anchor_delta_m: float = 30.0,
        savca_active_min_anchor_delta_m: float = 30.0,
        savca_change_deadband_m: float = 3.0,
        savca_label_median_window: int = 5,
        savca_active_ratio_to_max: float = 0.25,
        savca_active_min_abs_change_m: float = 10.0,
        savca_active_expand_steps: int = 1,
        savca_center_min_anchor_delta_m: float = 100.0,
        savca_center_min_active_len: int = 1,
        savca_center_min_gap_len: int = 5,
        savca_beta_floor_min_anchor_delta_m: float = 100.0,
        savca_beta_floor_min_active_len: int = 1,
        savca_beta_floor_min_qmax: float = 0.20,
        savca_beta_floor_min_gap_len: int = 5,
        savca_shape_min_anchor_delta_m: float = 100.0,
        savca_shape_min_active_len: int = 1,
        savca_shape_min_qmax: float = 0.20,
        savca_shape_min_gap_len: int = 5,
        savca_change_score_min_anchor_delta_m: float = 100.0,
        savca_change_score_min_active_len: int = 1,
        savca_change_score_min_qmax: float = 0.20,
        savca_change_score_min_gap_len: int = 5,
        savca_nonlinear_margin: float = 0.05,
        savca_diag_long_gap_len: int = 45,
        fltp_shape_min_anchor_delta_m: float = 100.0,
        fltp_shape_min_active_len: int = 1,
        fltp_shape_min_qmax: float = 0.20,
        fltp_shape_min_gap_len: int = 5,
        # Which altitude series to use for edge-sensitive auxiliary losses.
        # "pred_pos": final model output (recommended, aligned with eval/replay)
        # "pred_pos_main": pre-refiner main branch (legacy behavior)
        aux_alt_loss_series: str = "pred_pos",
    ) -> None:
        super().__init__()
        self.anchor_weight = anchor_weight
        self.gap_weight = gap_weight
        self.lambda_pos = lambda_pos
        self.lambda_smooth = lambda_smooth
        self.lambda_unc = lambda_unc
        self.alpha_vertical = float(alpha_vertical)
        self.lambda_cruise_phys = float(lambda_cruise_phys)
        self.cruise_speed_smooth_weight = float(cruise_speed_smooth_weight)
        self.cruise_heading_rate_weight = float(cruise_heading_rate_weight)
        self.cruise_vertical_rate_weight = float(cruise_vertical_rate_weight)
        self.cruise_planar_accel_weight = float(cruise_planar_accel_weight)
        self.cruise_max_abs_vertical_rate = max(1e-6, float(cruise_max_abs_vertical_rate))
        self.cruise_max_speed_delta = max(1e-6, float(cruise_max_speed_delta))
        self.cruise_max_heading_rate = max(1e-6, float(cruise_max_heading_rate))
        self.cruise_quality_weight_gain = float(cruise_quality_weight_gain)
        self.lambda_multi_scale = float(lambda_multi_scale)
        self.multi_scale_include_alt = bool(multi_scale_include_alt)
        self.fusion_reg_lambda = float(fusion_reg_lambda)
        self.fusion_reg_long_gap_weight = max(1.0, float(fusion_reg_long_gap_weight))
        self.gap_alt_weight = max(1.0, float(gap_alt_weight))
        self.lambda_vertical_smooth = float(lambda_vertical_smooth)
        self.lambda_alt_residual = float(lambda_alt_residual)
        self.lambda_alt_absolute_aux = float(lambda_alt_absolute_aux)
        self.alt_edge_steps = max(0, int(alt_edge_steps))
        self.alt_edge_weight = max(1.0, float(alt_edge_weight))
        self.lambda_alt_edge_delta = float(lambda_alt_edge_delta)
        self.lambda_anchor_consistency = float(lambda_anchor_consistency)
        self.anchor_boundary_weight = max(1.0, float(anchor_boundary_weight))
        self.lambda_alt_edge_first_diff = float(lambda_alt_edge_first_diff)
        self.lambda_alt_edge_second_diff = float(lambda_alt_edge_second_diff)
        self.lambda_alt_segment_bound = float(lambda_alt_segment_bound)
        self.lambda_alt_vertical_rate_penalty = float(lambda_alt_vertical_rate_penalty)
        self.lambda_alt_boundary_anchor = float(lambda_alt_boundary_anchor)
        self.alt_vertical_rate_max = float(alt_vertical_rate_max)
        self.segment_boundary_short_len = max(1, int(segment_boundary_short_len))
        self.segment_disturbed_alt_std_threshold = float(segment_disturbed_alt_std_threshold)
        self.alt_residual_cap_stable = max(1e-6, float(alt_residual_cap_stable))
        self.alt_residual_cap_disturbed = max(1e-6, float(alt_residual_cap_disturbed))
        self.alt_residual_cap_boundary = max(1e-6, float(alt_residual_cap_boundary))
        self.alt_residual_cap_short = max(1e-6, float(alt_residual_cap_short))
        self.lambda_alt_aux = float(lambda_alt_aux)
        self.lambda_aux = float(lambda_aux)
        self.lambda_vprog = float(lambda_vprog)
        self.vprog_enable_abs_dz_min = float(vprog_enable_abs_dz_min)
        self.lambda_vprog_res = float(lambda_vprog_res)
        self.vprog_res_enable_abs_dz_min = float(vprog_res_enable_abs_dz_min)
        self.lambda_alt_gate_supervision = float(lambda_alt_gate_supervision)
        self.lambda_alt_gate_risk_shrink = float(lambda_alt_gate_risk_shrink)
        self.alt_gate_risk_target = float(alt_gate_risk_target)
        self.lambda_savca_alloc = float(lambda_savca_alloc)
        self.lambda_savca_state = float(lambda_savca_state)
        self.lambda_savca_smooth = float(lambda_savca_smooth)
        self.lambda_savca_center = float(lambda_savca_center)
        self.lambda_savca_final_shape = float(lambda_savca_final_shape)
        self.lambda_savca_nonlinear = float(lambda_savca_nonlinear)
        self.lambda_savca_change_score = float(lambda_savca_change_score)
        self.lambda_ssvr_state = float(lambda_ssvr_state)
        self.lambda_ssvr_smooth = float(lambda_ssvr_smooth)
        self.ssvr_state_plateau_threshold = float(ssvr_state_plateau_threshold)
        self.lambda_fltp_shape = float(lambda_fltp_shape)
        self.lambda_fltp_center = float(lambda_fltp_center)
        self.savca_alloc_min_anchor_delta_m = max(0.0, float(savca_alloc_min_anchor_delta_m))
        self.savca_state_min_anchor_delta_m = max(0.0, float(savca_state_min_anchor_delta_m))
        self.savca_active_min_anchor_delta_m = max(0.0, float(savca_active_min_anchor_delta_m))
        self.savca_change_deadband_m = max(0.0, float(savca_change_deadband_m))
        self.savca_label_median_window = max(1, int(savca_label_median_window))
        self.savca_active_ratio_to_max = float(max(0.0, min(1.0, savca_active_ratio_to_max)))
        self.savca_active_min_abs_change_m = max(0.0, float(savca_active_min_abs_change_m))
        self.savca_active_expand_steps = max(0, int(savca_active_expand_steps))
        self.savca_center_min_anchor_delta_m = max(0.0, float(savca_center_min_anchor_delta_m))
        self.savca_center_min_active_len = max(1, int(savca_center_min_active_len))
        self.savca_center_min_gap_len = max(1, int(savca_center_min_gap_len))
        self.savca_beta_floor_min_anchor_delta_m = max(0.0, float(savca_beta_floor_min_anchor_delta_m))
        self.savca_beta_floor_min_active_len = max(1, int(savca_beta_floor_min_active_len))
        self.savca_beta_floor_min_qmax = float(max(0.0, min(1.0, savca_beta_floor_min_qmax)))
        self.savca_beta_floor_min_gap_len = max(1, int(savca_beta_floor_min_gap_len))
        self.savca_shape_min_anchor_delta_m = max(0.0, float(savca_shape_min_anchor_delta_m))
        self.savca_shape_min_active_len = max(1, int(savca_shape_min_active_len))
        self.savca_shape_min_qmax = float(max(0.0, min(1.0, savca_shape_min_qmax)))
        self.savca_shape_min_gap_len = max(1, int(savca_shape_min_gap_len))
        self.savca_change_score_min_anchor_delta_m = max(0.0, float(savca_change_score_min_anchor_delta_m))
        self.savca_change_score_min_active_len = max(1, int(savca_change_score_min_active_len))
        self.savca_change_score_min_qmax = float(max(0.0, min(1.0, savca_change_score_min_qmax)))
        self.savca_change_score_min_gap_len = max(1, int(savca_change_score_min_gap_len))
        self.savca_nonlinear_margin = float(max(0.0, savca_nonlinear_margin))
        self.savca_diag_long_gap_len = max(1, int(savca_diag_long_gap_len))
        self.fltp_shape_min_anchor_delta_m = max(0.0, float(fltp_shape_min_anchor_delta_m))
        self.fltp_shape_min_active_len = max(1, int(fltp_shape_min_active_len))
        self.fltp_shape_min_qmax = float(max(0.0, min(1.0, fltp_shape_min_qmax)))
        self.fltp_shape_min_gap_len = max(1, int(fltp_shape_min_gap_len))
        self.use_first_step_anchor_loss = bool(use_first_step_anchor_loss)
        self.first_step_anchor_lambda = float(first_step_anchor_lambda)
        self.use_second_step_anchor_loss = bool(use_second_step_anchor_loss)
        self.second_step_anchor_lambda = float(second_step_anchor_lambda)
        self.use_local_spike_loss = bool(use_local_spike_loss)
        self.local_spike_target_bucket = str(local_spike_target_bucket)
        self.local_spike_target_pattern = str(local_spike_target_pattern)
        self.local_spike_use_rightstep2 = bool(local_spike_use_rightstep2)
        self.local_spike_use_second_diff = bool(local_spike_use_second_diff)
        self.local_spike_lambda_jump = float(local_spike_lambda_jump)
        self.local_spike_lambda_curve = float(local_spike_lambda_curve)
        self.local_spike_target_bucket_id = _bucket_name_to_id(local_spike_target_bucket)
        self.local_spike_target_pattern_id = _pattern_name_to_id(local_spike_target_pattern)
        self.use_targeted_rightstep2_loss = bool(use_targeted_rightstep2_loss)
        self.target_bucket = str(target_bucket)
        self.target_anchor_pattern = str(target_anchor_pattern)
        self.use_target_jump_loss = bool(use_target_jump_loss)
        self.use_target_curve_loss = bool(use_target_curve_loss)
        self.target_jump_lambda = float(target_jump_lambda)
        self.target_curve_lambda = float(target_curve_lambda)
        self.use_target_value_rightstep2_loss = bool(use_target_value_rightstep2_loss)
        self.target_value_lambda = float(target_value_lambda)
        self.target_interp_lambda = float(target_interp_lambda)
        self.use_target_rightstep2_boundary_pull = bool(use_target_rightstep2_boundary_pull)
        self.target_rightstep2_boundary_pull_lambda = float(target_rightstep2_boundary_pull_lambda)
        self.lambda_var_reg = float(lambda_var_reg)
        self.var_reg_min_ratio = float(max(0.0, min(1.0, var_reg_min_ratio)))
        self.target_bucket_id = _bucket_name_to_id(target_bucket)
        self.target_anchor_pattern_id = _pattern_name_to_id(target_anchor_pattern)
        self.aux_alt_loss_series = str(aux_alt_loss_series).strip().lower()
        if self.aux_alt_loss_series not in {"pred_pos", "final", "pred_pos_main", "main"}:
            self.aux_alt_loss_series = "pred_pos"
        scales = [int(s) for s in (multi_scale_scales or []) if int(s) > 1]
        self.multi_scale_scales = sorted(set(scales))
        names = exo_feature_names or []
        self.exo_index = {str(k): i for i, k in enumerate(names)}
        self.pos_loss = nn.SmoothL1Loss(reduction="none")
        if dim_weights is None:
            dim_weights = [1.0, 1.0, 1.0]
        if len(dim_weights) != 3:
            raise ValueError(f"dim_weights must have length 3, got {len(dim_weights)}")
        self.register_buffer("dim_weights", torch.tensor([float(x) for x in dim_weights], dtype=torch.float32))

    def _median_smooth_1d(self, x: torch.Tensor, window: int) -> torch.Tensor:
        if window <= 1 or x.numel() <= 2:
            return x
        half = window // 2
        values = []
        for i in range(int(x.numel())):
            start = max(0, i - half)
            end = min(int(x.numel()), i + half + 1)
            values.append(torch.median(x[start:end]))
        return torch.stack(values)

    def build_savca_change_targets(
        self,
        *,
        target_alt_abs: torch.Tensor,
        obs_mask: torch.Tensor,
        seq_mask: torch.Tensor,
        savca_alloc_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        device = target_alt_abs.device
        dtype = target_alt_abs.dtype
        q_out = torch.zeros_like(target_alt_abs, dtype=dtype, device=device)
        active_out = torch.zeros_like(target_alt_abs, dtype=dtype, device=device)
        active_len_out = torch.zeros_like(target_alt_abs, dtype=dtype, device=device)
        qmax_out = torch.zeros_like(target_alt_abs, dtype=dtype, device=device)
        anchor_delta_out = torch.zeros_like(target_alt_abs, dtype=dtype, device=device)
        gap_len_out = torch.zeros_like(target_alt_abs, dtype=dtype, device=device)
        change_mask_out = torch.zeros_like(target_alt_abs, dtype=dtype, device=device)
        change_score_mask_out = torch.zeros_like(target_alt_abs, dtype=dtype, device=device)
        floor_mask_out = torch.zeros_like(target_alt_abs, dtype=dtype, device=device)
        long_change_mask_out = torch.zeros_like(target_alt_abs, dtype=dtype, device=device)

        bsz = target_alt_abs.shape[0]
        for b in range(bsz):
            valid_b = seq_mask[b] > 0.5
            anchors_b = torch.where((obs_mask[b] > 0.5) & valid_b)[0]
            if anchors_b.numel() < 2:
                continue
            z = target_alt_abs[b]
            for left_t, right_t in zip(anchors_b[:-1], anchors_b[1:]):
                left = int(left_t.item())
                right = int(right_t.item())
                if right <= left:
                    continue
                interval = torch.arange(left + 1, right + 1, device=device)
                if interval.numel() < 2:
                    continue
                if not bool(torch.all(valid_b[left : right + 1])):
                    continue
                if savca_alloc_valid is not None and float(savca_alloc_valid[b, interval].min().detach().cpu()) <= 0.5:
                    continue

                z_seg = self._median_smooth_1d(z[left : right + 1], self.savca_label_median_window)
                diffs = torch.abs(z_seg[1:] - z_seg[:-1])
                if self.savca_change_deadband_m > 0.0:
                    diffs = torch.where(diffs >= self.savca_change_deadband_m, diffs, torch.zeros_like(diffs))
                anchor_delta = torch.abs(z[right] - z[left])
                active_diffs, active_mask, active_sum = self._savca_active_diffs(diffs, anchor_delta)
                if not bool(active_mask.any()):
                    continue

                q = active_diffs / (active_sum + 1e-6)
                gap_len = int(right - left - 1)
                active_len = int(active_mask.sum().item())
                q_max = float(q.max().detach().cpu())
                interval_change = (
                    float(anchor_delta.detach().cpu()) >= self.savca_shape_min_anchor_delta_m
                    and active_len >= self.savca_shape_min_active_len
                    and q_max >= self.savca_shape_min_qmax
                    and gap_len >= self.savca_shape_min_gap_len
                )
                interval_floor = (
                    float(anchor_delta.detach().cpu()) >= self.savca_beta_floor_min_anchor_delta_m
                    and active_len >= self.savca_beta_floor_min_active_len
                    and q_max >= self.savca_beta_floor_min_qmax
                    and gap_len >= self.savca_beta_floor_min_gap_len
                )
                interval_change_score = (
                    float(anchor_delta.detach().cpu()) >= self.savca_change_score_min_anchor_delta_m
                    and active_len >= self.savca_change_score_min_active_len
                    and q_max >= self.savca_change_score_min_qmax
                    and gap_len >= self.savca_change_score_min_gap_len
                )

                q_out[b, interval] = q
                active_out[b, interval] = active_mask.to(dtype=dtype)
                active_len_out[b, interval] = float(active_len)
                qmax_out[b, interval] = float(q_max)
                anchor_delta_out[b, interval] = anchor_delta
                gap_len_out[b, interval] = float(gap_len)
                if interval_change:
                    change_mask_out[b, interval] = 1.0
                    if gap_len >= self.savca_diag_long_gap_len and float(anchor_delta.detach().cpu()) > 300.0:
                        long_change_mask_out[b, interval] = 1.0
                if interval_floor:
                    floor_mask_out[b, interval] = 1.0
                if interval_change_score:
                    change_score_mask_out[b, interval] = 1.0

        return {
            "q": q_out,
            "active_mask": active_out,
            "active_len": active_len_out,
            "q_max": qmax_out,
            "anchor_delta_abs": anchor_delta_out,
            "gap_len": gap_len_out,
            "change_mask": change_mask_out,
            "change_score_mask": change_score_mask_out,
            "floor_mask": floor_mask_out,
            "long_large_change_mask": long_change_mask_out,
        }

    def build_savca_beta_floor_mask(
        self,
        *,
        target_alt_abs: torch.Tensor,
        obs_mask: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.build_savca_change_targets(
            target_alt_abs=target_alt_abs,
            obs_mask=obs_mask,
            seq_mask=seq_mask,
            savca_alloc_valid=None,
        )["floor_mask"]

    def _savca_supervision_losses(
        self,
        *,
        target_pos: torch.Tensor,
        target_alt_abs: torch.Tensor | None,
        obs_mask: torch.Tensor,
        seq_mask: torch.Tensor,
        savca_alloc_p: torch.Tensor | None,
        savca_state: torch.Tensor | None,
        savca_alloc_valid: torch.Tensor | None,
        savca_change_score: torch.Tensor | None = None,
        savca_beta: torch.Tensor | None = None,
        savca_beta_floor_pred: torch.Tensor | None = None,
        savca_g_linear: torch.Tensor | None = None,
        savca_g_savca: torch.Tensor | None = None,
        savca_g_final: torch.Tensor | None = None,
        savca_ref_linear_abs: torch.Tensor | None = None,
        savca_ref_savca_abs: torch.Tensor | None = None,
        savca_ref_final_abs: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        zero = torch.tensor(0.0, device=target_pos.device, dtype=target_pos.dtype)
        if (
            savca_alloc_p is None
            or savca_state is None
            or savca_alloc_valid is None
            or (
                self.lambda_savca_alloc <= 0.0
                and self.lambda_savca_state <= 0.0
                and self.lambda_savca_smooth <= 0.0
                and self.lambda_savca_center <= 0.0
                and self.lambda_savca_final_shape <= 0.0
                and self.lambda_savca_nonlinear <= 0.0
                and self.lambda_savca_change_score <= 0.0
            )
        ):
            return (zero,) * 33

        alloc_losses = []
        state_losses = []
        smooth_losses = []
        center_losses = []
        shape_losses = []
        nonlinear_losses = []
        center_shift_stats = []
        center_shift_long_stats = []
        p_entropy_stats = []
        p_max_stats = []
        state_conf_stats = []
        shape_error_final_stats = []
        shape_error_linear_stats = []
        shape_error_savca_stats = []
        q_max_shape_stats = []
        linear_concentration_stats = []
        nonlinear_stats = []
        change_mask_ratio_stats = []
        shape_error_final_change_stats = []
        nonlinear_change_stats = []
        fused_minus_a1_stats = []
        fused_minus_a1_long_stats = []
        fused_minus_a1_change_stats = []
        change_score_losses = []
        change_score_stats = []
        change_score_pos_stats = []
        change_score_neg_stats = []
        change_score_acc_stats = []
        beta_floor_pred_stats = []
        beta_change_stats = []
        beta_nonchange_stats = []
        eps = torch.tensor(1e-6, device=target_pos.device, dtype=target_pos.dtype)

        if target_alt_abs is None:
            target_alt_abs = target_pos[..., 2]
        bundle = self.build_savca_change_targets(
            target_alt_abs=target_alt_abs,
            obs_mask=obs_mask,
            seq_mask=seq_mask,
            savca_alloc_valid=savca_alloc_valid,
        )
        q_all = bundle["q"]
        active_all = bundle["active_mask"]
        change_all = bundle["change_mask"]
        change_score_all = bundle["change_score_mask"]
        long_change_all = bundle["long_large_change_mask"]

        bsz = target_pos.shape[0]
        for b in range(bsz):
            valid_b = seq_mask[b] > 0.5
            anchors_b = torch.where((obs_mask[b] > 0.5) & valid_b)[0]
            if anchors_b.numel() < 2:
                continue
            for left_t, right_t in zip(anchors_b[:-1], anchors_b[1:]):
                left = int(left_t.item())
                right = int(right_t.item())
                if right <= left:
                    continue
                interval = torch.arange(left + 1, right + 1, device=target_pos.device)
                if interval.numel() < 2:
                    continue
                if not bool(torch.all(valid_b[left : right + 1])):
                    continue
                if savca_alloc_valid is not None and float(savca_alloc_valid[b, interval].min().detach().cpu()) <= 0.5:
                    continue

                p_pred = torch.clamp(savca_alloc_p[b, interval], min=1e-6)
                p_pred = p_pred / (p_pred.sum() + eps)
                r_pred = torch.clamp(savca_state[b, interval], min=1e-4, max=1.0 - 1e-4)
                q = q_all[b, interval]
                active_mask = active_all[b, interval] > 0.5
                change_active = bool((change_all[b, interval] > 0.5).any().item())
                change_score_active = bool((change_score_all[b, interval] > 0.5).any().item())
                long_change_active = bool((long_change_all[b, interval] > 0.5).any().item())
                gap_len = int(right - left - 1)
                anchor_delta = torch.abs(target_alt_abs[b, right] - target_alt_abs[b, left])
                p_entropy_stats.append(
                    -torch.sum(p_pred * torch.log(torch.clamp(p_pred, min=1e-6)))
                    / torch.log(torch.tensor(float(max(2, interval.numel())), device=target_pos.device, dtype=target_pos.dtype))
                )
                p_max_stats.append(p_pred.max())
                state_conf_stats.append(torch.relu(r_pred.max() - r_pred.mean()))
                if savca_change_score is not None:
                    s_change = torch.clamp(savca_change_score[b, interval].mean(), min=1e-4, max=1.0 - 1e-4)
                    change_score_stats.append(s_change)
                    if change_score_active:
                        change_score_pos_stats.append(s_change)
                    else:
                        change_score_neg_stats.append(s_change)
                    change_score_acc_stats.append(((s_change >= 0.5) == change_score_active).to(dtype=target_pos.dtype))
                    if self.lambda_savca_change_score > 0.0:
                        target_change = torch.tensor(float(change_score_active), device=target_pos.device, dtype=target_pos.dtype)
                        change_score_losses.append(F.binary_cross_entropy(s_change, target_change, reduction="mean"))
                if savca_beta_floor_pred is not None:
                    beta_floor_pred_stats.append(savca_beta_floor_pred[b, interval].mean())
                if savca_beta is not None:
                    beta_seg = savca_beta[b, interval].mean()
                    if change_active:
                        beta_change_stats.append(beta_seg)
                    else:
                        beta_nonchange_stats.append(beta_seg)

                if self.lambda_savca_smooth > 0.0 and interval.numel() > 2:
                    smooth_losses.append(torch.mean(torch.abs(r_pred[1:] - r_pred[:-1])))

                if not bool(active_mask.any()):
                    continue

                y_state = active_mask.to(dtype=target_pos.dtype)
                tau = torch.arange(1, interval.numel() + 1, device=target_pos.device, dtype=target_pos.dtype) / float(interval.numel())
                center_shift = torch.abs(torch.sum(tau * p_pred) - torch.sum(tau * q))
                center_shift_stats.append(center_shift)
                if gap_len >= self.savca_diag_long_gap_len:
                    center_shift_long_stats.append(center_shift)
                if self.lambda_savca_alloc > 0.0:
                    if float(anchor_delta.detach().cpu()) >= self.savca_alloc_min_anchor_delta_m and int(active_mask.sum().item()) >= 2:
                        alloc_losses.append(torch.mean(torch.abs(torch.cumsum(p_pred, dim=0) - torch.cumsum(q, dim=0))))
                if self.lambda_savca_state > 0.0:
                    if float(anchor_delta.detach().cpu()) >= self.savca_state_min_anchor_delta_m:
                        state_losses.append(F.binary_cross_entropy(r_pred, y_state, reduction="mean"))
                if self.lambda_savca_center > 0.0:
                    if (
                        float(anchor_delta.detach().cpu()) >= self.savca_center_min_anchor_delta_m
                        and int(active_mask.sum().item()) >= self.savca_center_min_active_len
                        and gap_len >= self.savca_center_min_gap_len
                    ):
                        center_losses.append(center_shift)

                if savca_g_final is not None and change_active:
                    g_q = torch.cumsum(q, dim=0)
                    g_final = savca_g_final[b, interval]
                    g_linear = savca_g_linear[b, interval] if savca_g_linear is not None else tau
                    g_savca = savca_g_savca[b, interval] if savca_g_savca is not None else torch.cumsum(p_pred, dim=0)
                    shape_err_final = torch.mean(torch.abs(g_final - g_q))
                    shape_err_linear = torch.mean(torch.abs(g_linear - g_q))
                    shape_err_savca = torch.mean(torch.abs(g_savca - g_q))
                    shape_error_final_stats.append(shape_err_final)
                    shape_error_linear_stats.append(shape_err_linear)
                    shape_error_savca_stats.append(shape_err_savca)
                    q_max_shape_stats.append(q.max())
                    linear_concentration_stats.append(torch.tensor(1.0 / float(interval.numel()), device=target_pos.device, dtype=target_pos.dtype))
                    change_mask_ratio_stats.append(torch.tensor(1.0, device=target_pos.device, dtype=target_pos.dtype))
                    shape_error_final_change_stats.append(shape_err_final)
                    if self.lambda_savca_final_shape > 0.0:
                        shape_losses.append(shape_err_final)
                    d_nonlinear = torch.mean(torch.abs(g_final - tau))
                    nonlinear_stats.append(d_nonlinear)
                    nonlinear_change_stats.append(d_nonlinear)
                    if self.lambda_savca_nonlinear > 0.0:
                        nonlinear_losses.append(torch.relu(self.savca_nonlinear_margin - d_nonlinear))
                elif savca_g_final is not None:
                    change_mask_ratio_stats.append(torch.tensor(0.0, device=target_pos.device, dtype=target_pos.dtype))

                if savca_ref_linear_abs is not None and savca_ref_final_abs is not None:
                    tgt_abs = target_alt_abs[b, interval]
                    linear_err = torch.sqrt(torch.mean((savca_ref_linear_abs[b, interval] - tgt_abs) ** 2) + 1e-6)
                    final_err = torch.sqrt(torch.mean((savca_ref_final_abs[b, interval] - tgt_abs) ** 2) + 1e-6)
                    diff_err = final_err - linear_err
                    fused_minus_a1_stats.append(diff_err)
                    if gap_len >= self.savca_diag_long_gap_len:
                        fused_minus_a1_long_stats.append(diff_err)
                    if change_active:
                        fused_minus_a1_change_stats.append(diff_err)

        alloc_loss = torch.stack(alloc_losses).mean() if alloc_losses else zero
        state_loss = torch.stack(state_losses).mean() if state_losses else zero
        smooth_loss = torch.stack(smooth_losses).mean() if smooth_losses else zero
        center_loss = torch.stack(center_losses).mean() if center_losses else zero
        shape_loss = torch.stack(shape_losses).mean() if shape_losses else zero
        nonlinear_loss = torch.stack(nonlinear_losses).mean() if nonlinear_losses else zero
        supervised_segments = torch.tensor(float(len(alloc_losses) or len(state_losses) or len(shape_losses)), device=target_pos.device, dtype=target_pos.dtype)
        center_shift_mean = torch.stack(center_shift_stats).mean() if center_shift_stats else zero
        center_shift_long = torch.stack(center_shift_long_stats).mean() if center_shift_long_stats else zero
        p_entropy_mean = torch.stack(p_entropy_stats).mean() if p_entropy_stats else zero
        p_max_mean = torch.stack(p_max_stats).mean() if p_max_stats else zero
        state_conf_mean = torch.stack(state_conf_stats).mean() if state_conf_stats else zero
        shape_error_final = torch.stack(shape_error_final_stats).mean() if shape_error_final_stats else zero
        shape_error_linear = torch.stack(shape_error_linear_stats).mean() if shape_error_linear_stats else zero
        shape_error_savca = torch.stack(shape_error_savca_stats).mean() if shape_error_savca_stats else zero
        transition_concentration_adsb = torch.stack(q_max_shape_stats).mean() if q_max_shape_stats else zero
        transition_concentration_a1 = torch.stack(linear_concentration_stats).mean() if linear_concentration_stats else zero
        shape_gain_vs_a1 = shape_error_linear - shape_error_final if shape_error_linear_stats else zero
        nonlinear_mean = torch.stack(nonlinear_stats).mean() if nonlinear_stats else zero
        change_ratio = torch.stack(change_mask_ratio_stats).mean() if change_mask_ratio_stats else zero
        shape_error_final_change = torch.stack(shape_error_final_change_stats).mean() if shape_error_final_change_stats else zero
        nonlinear_change = torch.stack(nonlinear_change_stats).mean() if nonlinear_change_stats else zero
        fused_minus_a1_mean = torch.stack(fused_minus_a1_stats).mean() if fused_minus_a1_stats else zero
        fused_minus_a1_long = torch.stack(fused_minus_a1_long_stats).mean() if fused_minus_a1_long_stats else zero
        fused_minus_a1_change = torch.stack(fused_minus_a1_change_stats).mean() if fused_minus_a1_change_stats else zero
        change_score_loss = torch.stack(change_score_losses).mean() if change_score_losses else zero
        change_score_mean = torch.stack(change_score_stats).mean() if change_score_stats else zero
        change_score_pos_mean = torch.stack(change_score_pos_stats).mean() if change_score_pos_stats else zero
        change_score_neg_mean = torch.stack(change_score_neg_stats).mean() if change_score_neg_stats else zero
        change_score_acc = torch.stack(change_score_acc_stats).mean() if change_score_acc_stats else zero
        beta_floor_pred_mean = torch.stack(beta_floor_pred_stats).mean() if beta_floor_pred_stats else zero
        beta_change_mean = torch.stack(beta_change_stats).mean() if beta_change_stats else zero
        beta_nonchange_mean = torch.stack(beta_nonchange_stats).mean() if beta_nonchange_stats else zero
        return (
            alloc_loss,
            state_loss,
            smooth_loss,
            center_loss,
            shape_loss,
            nonlinear_loss,
            supervised_segments,
            center_shift_mean,
            center_shift_long,
            p_entropy_mean,
            p_max_mean,
            state_conf_mean,
            shape_error_final,
            shape_error_linear,
            shape_error_savca,
            transition_concentration_adsb,
            transition_concentration_a1,
            shape_gain_vs_a1,
            nonlinear_mean,
            change_ratio,
            shape_error_final_change,
            nonlinear_change,
            fused_minus_a1_mean,
            fused_minus_a1_long,
            fused_minus_a1_change,
            change_score_loss,
            change_score_mean,
            change_score_pos_mean,
            change_score_neg_mean,
            change_score_acc,
            beta_floor_pred_mean,
            beta_change_mean,
            beta_nonchange_mean,
        )

    def _savca_active_diffs(self, diffs: torch.Tensor, anchor_delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = torch.zeros_like(diffs)
        diff_sum = diffs.sum()
        diff_max = diffs.max()
        if (
            diffs.numel() == 0
            or float(anchor_delta.detach().cpu()) < self.savca_active_min_anchor_delta_m
            or float(diff_sum.detach().cpu()) <= 1e-6
            or float(diff_max.detach().cpu()) <= 1e-6
        ):
            return zero, zero.to(dtype=torch.bool), zero

        active_thr = max(
            self.savca_active_min_abs_change_m,
            self.savca_change_deadband_m,
            self.savca_active_ratio_to_max * float(diff_max.detach().cpu()),
        )
        active_mask = diffs >= active_thr
        if self.savca_active_expand_steps > 0 and bool(active_mask.any()):
            mask = active_mask
            for _ in range(self.savca_active_expand_steps):
                prev = torch.cat([mask[:1], mask[:-1]], dim=0)
                nxt = torch.cat([mask[1:], mask[-1:]], dim=0)
                mask = mask | prev | nxt
            active_mask = mask

        active_diffs = torch.where(active_mask, diffs, zero)
        active_sum = active_diffs.sum()
        return active_diffs, active_mask, active_sum

    def _fltp_supervision_losses(
        self,
        *,
        target_alt_abs: torch.Tensor,
        obs_mask: torch.Tensor,
        seq_mask: torch.Tensor,
        dt_prev: torch.Tensor,
        dt_next: torch.Tensor,
        fltp_beta: torch.Tensor | None,
        fltp_c: torch.Tensor | None,
        fltp_g_linear: torch.Tensor | None,
        fltp_g_sig: torch.Tensor | None,
        fltp_g_final: torch.Tensor | None,
        fltp_ref_linear_abs: torch.Tensor | None,
        fltp_ref_final_abs: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        zero = torch.tensor(0.0, device=target_alt_abs.device, dtype=target_alt_abs.dtype)
        if (
            fltp_g_final is None
            or fltp_g_linear is None
            or fltp_g_sig is None
            or fltp_c is None
            or (
                self.lambda_fltp_shape <= 0.0
                and self.lambda_fltp_center <= 0.0
            )
        ):
            return (zero,) * 15

        shape_losses = []
        center_losses = []
        center_shift_stats = []
        center_shift_long_stats = []
        shape_error_final_stats = []
        shape_error_linear_stats = []
        shape_error_sig_stats = []
        nonlinear_stats = []
        change_ratio_stats = []
        pred_concentration_stats = []
        adsb_concentration_stats = []
        fused_minus_a1_stats = []
        fused_minus_a1_long_stats = []
        fused_minus_a1_change_stats = []
        eps = torch.tensor(1e-6, device=target_alt_abs.device, dtype=target_alt_abs.dtype)

        bundle = self.build_savca_change_targets(
            target_alt_abs=target_alt_abs,
            obs_mask=obs_mask,
            seq_mask=seq_mask,
            savca_alloc_valid=None,
        )
        q_all = bundle["q"]
        active_all = bundle["active_mask"]
        change_all = bundle["change_mask"]
        long_change_all = bundle["long_large_change_mask"]

        bsz = target_alt_abs.shape[0]
        for b in range(bsz):
            valid_b = seq_mask[b] > 0.5
            anchors_b = torch.where((obs_mask[b] > 0.5) & valid_b)[0]
            if anchors_b.numel() < 2:
                continue
            for left_t, right_t in zip(anchors_b[:-1], anchors_b[1:]):
                left = int(left_t.item())
                right = int(right_t.item())
                if right <= left:
                    continue
                interval = torch.arange(left + 1, right, device=target_alt_abs.device)
                if interval.numel() < 1:
                    continue
                if not bool(torch.all(valid_b[left : right + 1])):
                    continue
                q = q_all[b, interval]
                active_mask = active_all[b, interval] > 0.5
                change_active = bool((change_all[b, interval] > 0.5).any().item())
                long_change_active = bool((long_change_all[b, interval] > 0.5).any().item())
                gap_len = int(interval.numel())
                if not bool(active_mask.any()):
                    continue
                tau_den = torch.clamp(dt_prev[b, interval] + dt_next[b, interval], min=1e-6)
                tau = torch.clamp(dt_prev[b, interval] / (tau_den + 1e-6), min=0.0, max=1.0)
                anchor_delta = torch.abs(target_alt_abs[b, right] - target_alt_abs[b, left])
                q_max = float(q.max().detach().cpu())
                fltp_active = (
                    float(anchor_delta.detach().cpu()) >= self.fltp_shape_min_anchor_delta_m
                    and int(active_mask.sum().item()) >= self.fltp_shape_min_active_len
                    and q_max >= self.fltp_shape_min_qmax
                    and gap_len >= self.fltp_shape_min_gap_len
                )
                change_ratio_stats.append(torch.tensor(1.0 if fltp_active else 0.0, device=target_alt_abs.device, dtype=target_alt_abs.dtype))
                if not fltp_active:
                    continue
                g_q = torch.cumsum(q, dim=0)
                g_linear = fltp_g_linear[b, interval]
                g_sig = fltp_g_sig[b, interval]
                g_final = fltp_g_final[b, interval]
                c_pred = fltp_c[b, interval][0]
                c_q = torch.sum(tau * q)
                center_shift = torch.abs(c_pred - c_q)
                center_shift_stats.append(center_shift)
                if gap_len >= self.savca_diag_long_gap_len:
                    center_shift_long_stats.append(center_shift)
                shape_err_final = torch.mean(torch.abs(g_final - g_q))
                shape_err_linear = torch.mean(torch.abs(g_linear - g_q))
                shape_err_sig = torch.mean(torch.abs(g_sig - g_q))
                shape_error_final_stats.append(shape_err_final)
                shape_error_linear_stats.append(shape_err_linear)
                shape_error_sig_stats.append(shape_err_sig)
                diff_final = torch.diff(torch.cat([torch.zeros(1, device=target_alt_abs.device, dtype=target_alt_abs.dtype), g_final]))
                pred_concentration_stats.append(diff_final.max())
                adsb_concentration_stats.append(q.max())
                nonlinear_stats.append(torch.mean(torch.abs(g_final - g_linear)))
                if self.lambda_fltp_shape > 0.0:
                    shape_losses.append(shape_err_final)
                if self.lambda_fltp_center > 0.0:
                    center_losses.append(center_shift)
                if fltp_ref_linear_abs is not None and fltp_ref_final_abs is not None:
                    tgt_abs = target_alt_abs[b, interval]
                    linear_err = torch.sqrt(torch.mean((fltp_ref_linear_abs[b, interval] - tgt_abs) ** 2) + 1e-6)
                    final_err = torch.sqrt(torch.mean((fltp_ref_final_abs[b, interval] - tgt_abs) ** 2) + 1e-6)
                    diff_err = final_err - linear_err
                    fused_minus_a1_stats.append(diff_err)
                    if gap_len >= self.savca_diag_long_gap_len:
                        fused_minus_a1_long_stats.append(diff_err)
                    if change_active:
                        fused_minus_a1_change_stats.append(diff_err)

        shape_loss = torch.stack(shape_losses).mean() if shape_losses else zero
        center_loss = torch.stack(center_losses).mean() if center_losses else zero
        center_shift_mean = torch.stack(center_shift_stats).mean() if center_shift_stats else zero
        center_shift_long = torch.stack(center_shift_long_stats).mean() if center_shift_long_stats else zero
        shape_error_final = torch.stack(shape_error_final_stats).mean() if shape_error_final_stats else zero
        shape_error_linear = torch.stack(shape_error_linear_stats).mean() if shape_error_linear_stats else zero
        shape_error_sig = torch.stack(shape_error_sig_stats).mean() if shape_error_sig_stats else zero
        nonlinear_mean = torch.stack(nonlinear_stats).mean() if nonlinear_stats else zero
        change_ratio = torch.stack(change_ratio_stats).mean() if change_ratio_stats else zero
        pred_concentration = torch.stack(pred_concentration_stats).mean() if pred_concentration_stats else zero
        adsb_concentration = torch.stack(adsb_concentration_stats).mean() if adsb_concentration_stats else zero
        fused_minus_a1_mean = torch.stack(fused_minus_a1_stats).mean() if fused_minus_a1_stats else zero
        fused_minus_a1_long = torch.stack(fused_minus_a1_long_stats).mean() if fused_minus_a1_long_stats else zero
        fused_minus_a1_change = torch.stack(fused_minus_a1_change_stats).mean() if fused_minus_a1_change_stats else zero
        supervised_segments = torch.tensor(float(len(shape_error_final_stats)), device=target_alt_abs.device, dtype=target_alt_abs.dtype)
        return (
            shape_loss,
            center_loss,
            supervised_segments,
            center_shift_mean,
            center_shift_long,
            shape_error_final,
            shape_error_linear,
            shape_error_sig,
            nonlinear_mean,
            change_ratio,
            pred_concentration,
            adsb_concentration,
            fused_minus_a1_mean,
            fused_minus_a1_long,
            fused_minus_a1_change,
        )

    def _ssvr_supervision_losses(
        self,
        *,
        target_alt_abs: torch.Tensor,
        obs_mask: torch.Tensor,
        seq_mask: torch.Tensor,
        alt_fwd: torch.Tensor,
        alt_bwd: torch.Tensor,
        ssvr_pi_L: torch.Tensor | None,
        ssvr_pi_T: torch.Tensor | None,
        ssvr_pi_R: torch.Tensor | None,
        ssvr_rho: torch.Tensor | None,
        ssvr_state_logits: torch.Tensor | None,
        ssvr_z_hat: torch.Tensor | None,
        ssvr_z_linear: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,  # state_loss
        torch.Tensor,  # smooth_loss
        torch.Tensor,  # state_ce_loss
        torch.Tensor,  # state_acc
        torch.Tensor,  # supervised_segments
        torch.Tensor,  # pi_L_mean
        torch.Tensor,  # pi_T_mean
        torch.Tensor,  # pi_R_mean
        torch.Tensor,  # rho_mean
        torch.Tensor,  # state_entropy_mean
        torch.Tensor,  # d_nonlinear
        torch.Tensor,  # shape_error_final
        torch.Tensor,  # ssvr_minus_a1_mean
        torch.Tensor,  # m_change_ratio
    ]:
        zero = torch.tensor(0.0, device=target_alt_abs.device, dtype=target_alt_abs.dtype)
        if (
            ssvr_pi_L is None
            or ssvr_pi_T is None
            or ssvr_pi_R is None
            or ssvr_state_logits is None
            or ssvr_z_hat is None
            or ssvr_z_linear is None
            or (
                self.lambda_ssvr_state <= 0.0
                and self.lambda_ssvr_smooth <= 0.0
            )
        ):
            return (zero,) * 14

        from src.models.alt_ssvr import _build_ssvr_state_active_mask, build_ssvr_state_labels

        valid = seq_mask > 0.5
        gap_mask = (obs_mask <= 0.5) & valid

        # Only supervise state on gaps with meaningful altitude change.
        state_active_mask = _build_ssvr_state_active_mask(
            z_L=alt_fwd,
            z_R=alt_bwd,
            gap_mask=gap_mask,
            min_anchor_delta_m=30.0,
        )

        # Build weak state labels.
        y_state = build_ssvr_state_labels(
            z_true_abs=target_alt_abs,
            z_L=alt_fwd,
            z_R=alt_bwd,
            gap_mask=gap_mask,
            plateau_threshold=self.ssvr_state_plateau_threshold,
        )

        state_losses = []
        smooth_losses = []
        state_ce_vals = []
        state_acc_vals = []
        pi_L_vals = []
        pi_T_vals = []
        pi_R_vals = []
        rho_vals = []
        entropy_vals = []
        d_nonlinear_vals = []
        shape_err_vals = []
        ssvr_minus_a1_vals = []
        change_ratio_vals = []

        bsz = target_alt_abs.shape[0]
        eps = torch.tensor(1e-6, device=target_alt_abs.device, dtype=target_alt_abs.dtype)
        for b in range(bsz):
            anchors_b = torch.where((obs_mask[b] > 0.5) & valid[b])[0]
            if anchors_b.numel() < 2:
                continue
            for left_t, right_t in zip(anchors_b[:-1], anchors_b[1:]):
                left = int(left_t.item())
                right = int(right_t.item())
                if right <= left:
                    continue
                interval = torch.arange(left + 1, right, device=target_alt_abs.device)
                if interval.numel() < 1:
                    continue
                if not bool(torch.all(valid[b, left : right + 1])):
                    continue

                gap_n = int(interval.numel())
                anchor_delta = torch.abs(alt_bwd[b, left] - alt_fwd[b, left])

                # --- state loss (cross-entropy), gated by |dz| >= 100 m ---
                y_seg = y_state[b, interval]
                active_seg = state_active_mask[b, interval]
                valid_label = (y_seg >= 0) & active_seg
                if valid_label.any() and self.lambda_ssvr_state > 0.0:
                    logits_seg = ssvr_state_logits[b, interval][valid_label]
                    y_seg_valid = y_seg[valid_label]
                    ce = F.cross_entropy(logits_seg, y_seg_valid, reduction="mean")
                    state_losses.append(ce)
                    state_ce_vals.append(ce.detach())
                    pred_label = logits_seg.argmax(dim=-1)
                    state_acc_vals.append((pred_label == y_seg_valid).float().mean())

                # --- state smooth loss ---
                if self.lambda_ssvr_smooth > 0.0 and gap_n >= 2:
                    pi_stack = torch.stack(
                        [ssvr_pi_L[b, interval], ssvr_pi_T[b, interval], ssvr_pi_R[b, interval]], dim=-1
                    )  # [n, 3]
                    smooth_losses.append(torch.mean(torch.abs(pi_stack[1:] - pi_stack[:-1])))

                # --- diagnostics ---
                pi_L_vals.append(ssvr_pi_L[b, interval].mean())
                pi_T_vals.append(ssvr_pi_T[b, interval].mean())
                pi_R_vals.append(ssvr_pi_R[b, interval].mean())
                if ssvr_rho is not None:
                    rho_vals.append(ssvr_rho[b, interval].mean())

                pi_seg = torch.stack(
                    [ssvr_pi_L[b, interval], ssvr_pi_T[b, interval], ssvr_pi_R[b, interval]], dim=-1
                )
                p_clamp = torch.clamp(pi_seg, min=1e-6)
                ent = -torch.sum(p_clamp * torch.log(p_clamp), dim=-1).mean() / torch.log(
                    torch.tensor(3.0, device=target_alt_abs.device, dtype=target_alt_abs.dtype)
                )
                entropy_vals.append(ent)

                # D_nonlinear: mean abs deviation from linear
                tau_gap = torch.arange(1, gap_n + 1, device=target_alt_abs.device, dtype=target_alt_abs.dtype) / float(gap_n)
                z_linear_seg = alt_fwd[b, interval] + tau_gap * (alt_bwd[b, interval] - alt_fwd[b, interval])
                z_hat_seg = ssvr_z_hat[b, interval]
                d_nonlinear_vals.append(torch.mean(torch.abs(z_hat_seg - z_linear_seg)))

                # Shape error vs ADS-B (gap interior only)
                z_true_seg = target_alt_abs[b, interval]
                z_left_ref = alt_fwd[b, left]
                z_right_ref = alt_bwd[b, right]
                dz_ref = z_right_ref - z_left_ref + eps
                g_q = (z_true_seg - z_left_ref) / dz_ref
                g_hat = (z_hat_seg - z_left_ref) / dz_ref
                shape_err_vals.append(torch.mean(torch.abs(g_hat - g_q)))

                # SSVR minus A1 linear RMSE
                a1_err = torch.sqrt(torch.mean((z_linear_seg - z_true_seg) ** 2) + 1e-6)
                ssvr_err = torch.sqrt(torch.mean((z_hat_seg - z_true_seg) ** 2) + 1e-6)
                ssvr_minus_a1_vals.append(ssvr_err - a1_err)

                # Whether this gap has meaningful altitude change
                has_change = float(anchor_delta.detach().cpu() > 100.0)
                change_ratio_vals.append(torch.tensor(has_change, device=target_alt_abs.device, dtype=target_alt_abs.dtype))

        state_loss = torch.stack(state_losses).mean() if state_losses else zero
        smooth_loss = torch.stack(smooth_losses).mean() if smooth_losses else zero
        state_ce = torch.stack(state_ce_vals).mean() if state_ce_vals else zero
        state_acc = torch.stack(state_acc_vals).mean() if state_acc_vals else zero
        supervised_segments = torch.tensor(
            float(len(state_losses)), device=target_alt_abs.device, dtype=target_alt_abs.dtype
        )
        pi_L_mean = torch.stack(pi_L_vals).mean() if pi_L_vals else zero
        pi_T_mean = torch.stack(pi_T_vals).mean() if pi_T_vals else zero
        pi_R_mean = torch.stack(pi_R_vals).mean() if pi_R_vals else zero
        rho_mean = torch.stack(rho_vals).mean() if rho_vals else zero
        entropy_mean = torch.stack(entropy_vals).mean() if entropy_vals else zero
        d_nonlinear = torch.stack(d_nonlinear_vals).mean() if d_nonlinear_vals else zero
        shape_error_final = torch.stack(shape_err_vals).mean() if shape_err_vals else zero
        ssvr_minus_a1_mean = torch.stack(ssvr_minus_a1_vals).mean() if ssvr_minus_a1_vals else zero
        m_change_ratio = torch.stack(change_ratio_vals).mean() if change_ratio_vals else zero

        return (
            state_loss,
            smooth_loss,
            state_ce,
            state_acc,
            supervised_segments,
            pi_L_mean,
            pi_T_mean,
            pi_R_mean,
            rho_mean,
            entropy_mean,
            d_nonlinear,
            shape_error_final,
            ssvr_minus_a1_mean,
            m_change_ratio,
        )

    def _build_segment_bound_targets(
        self,
        target_alt: torch.Tensor,
        obs_mask: torch.Tensor,
        seq_mask: torch.Tensor,
        gap_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct per-gap baseline and cap for segment-type bounded residual training."""
        bsz, t_len = target_alt.shape
        device = target_alt.device
        dtype = target_alt.dtype
        base = target_alt.detach().clone()
        cap = torch.zeros_like(target_alt, dtype=dtype, device=device)
        valid = seq_mask > 0.5
        anchor = (obs_mask > 0.5) & valid
        for i in range(bsz):
            t = 0
            while t < t_len:
                if not gap_mask[i, t]:
                    t += 1
                    continue
                s = t
                while t < t_len and gap_mask[i, t]:
                    t += 1
                e = t - 1
                seg_len = e - s + 1
                has_left = (s - 1 >= 0) and bool(anchor[i, s - 1])
                has_right = (e + 1 < t_len) and bool(anchor[i, e + 1])
                is_boundary = (seg_len <= self.segment_boundary_short_len) or (not (has_left and has_right))
                seg_alt = target_alt[i, s : e + 1]
                alt_std = float(torch.std(seg_alt, unbiased=False).detach().cpu()) if seg_len > 1 else 0.0
                is_disturbed = (not is_boundary) and (alt_std >= self.segment_disturbed_alt_std_threshold)
                if is_boundary:
                    c = self.alt_residual_cap_boundary
                elif is_disturbed:
                    c = self.alt_residual_cap_disturbed
                else:
                    c = self.alt_residual_cap_stable
                if seg_len <= self.segment_boundary_short_len:
                    c = min(c, self.alt_residual_cap_short)
                cap[i, s : e + 1] = float(c)

                # Segment baseline: anchor-anchor linear if possible, otherwise segment endpoint linear.
                if has_left and has_right:
                    left_alt = float(target_alt[i, s - 1].detach().cpu())
                    right_alt = float(target_alt[i, e + 1].detach().cpu())
                else:
                    left_alt = float(target_alt[i, s].detach().cpu())
                    right_alt = float(target_alt[i, e].detach().cpu())
                denom = max(1, seg_len + (1 if (has_left and has_right) else 0))
                vals = []
                for j in range(seg_len):
                    r = float((j + (1 if (has_left and has_right) else 0)) / denom)
                    vals.append(left_alt + r * (right_alt - left_alt))
                base[i, s : e + 1] = torch.tensor(vals, device=device, dtype=dtype)
        return base, cap

    def forward(
        self,
        pred_pos: torch.Tensor,
        target_pos: torch.Tensor,
        obs_mask: torch.Tensor,
        seq_mask: torch.Tensor,
        exo: torch.Tensor | None = None,
        quality: torch.Tensor | None = None,
        fusion_weights: torch.Tensor | None = None,
        dt_prev: torch.Tensor | None = None,
        dt_next: torch.Tensor | None = None,
        logvar: torch.Tensor | None = None,
        long_gap_threshold: int = 20,
        alt_base: torch.Tensor | None = None,
        residual_bound: torch.Tensor | None = None,
        delta_alt_pred_norm: torch.Tensor | None = None,
        alt_gate: torch.Tensor | None = None,
        teacher_scale: torch.Tensor | None = None,
        risk_flag: torch.Tensor | None = None,
        risk_flag_teacher: torch.Tensor | None = None,
        segment_bucket: torch.Tensor | None = None,
        anchor_pattern: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
        pred_pos_main: torch.Tensor | None = None,
        pred_pos_aux: torch.Tensor | None = None,
        pred_pos_aux_supervise_dims: torch.Tensor | None = None,
        left_boundary_alt: torch.Tensor | None = None,
        right_boundary_alt: torch.Tensor | None = None,
        mu_f: torch.Tensor | None = None,
        mu_b: torch.Tensor | None = None,
        savca_alloc_p: torch.Tensor | None = None,
        savca_state: torch.Tensor | None = None,
        savca_alloc_valid: torch.Tensor | None = None,
        savca_target_alt_abs: torch.Tensor | None = None,
        savca_change_score: torch.Tensor | None = None,
        savca_beta: torch.Tensor | None = None,
        savca_beta_floor_pred: torch.Tensor | None = None,
        savca_g_linear: torch.Tensor | None = None,
        savca_g_savca: torch.Tensor | None = None,
        savca_g_final: torch.Tensor | None = None,
        savca_ref_linear_abs: torch.Tensor | None = None,
        savca_ref_savca_abs: torch.Tensor | None = None,
        savca_ref_final_abs: torch.Tensor | None = None,
        fltp_beta: torch.Tensor | None = None,
        fltp_c: torch.Tensor | None = None,
        fltp_w: torch.Tensor | None = None,
        fltp_g_linear: torch.Tensor | None = None,
        fltp_g_sig: torch.Tensor | None = None,
        fltp_g_final: torch.Tensor | None = None,
        fltp_ref_linear_abs: torch.Tensor | None = None,
        fltp_ref_sig_abs: torch.Tensor | None = None,
        fltp_ref_final_abs: torch.Tensor | None = None,
        ssvr_pi_L: torch.Tensor | None = None,
        ssvr_pi_T: torch.Tensor | None = None,
        ssvr_pi_R: torch.Tensor | None = None,
        ssvr_rho: torch.Tensor | None = None,
        ssvr_state_logits: torch.Tensor | None = None,
        ssvr_z_hat: torch.Tensor | None = None,
        ssvr_z_linear: torch.Tensor | None = None,
        alt_fwd: torch.Tensor | None = None,
        alt_bwd: torch.Tensor | None = None,
        q_pred: torch.Tensor | None = None,
        q_true: torch.Tensor | None = None,
        q_mask: torch.Tensor | None = None,
        q_res_pred: torch.Tensor | None = None,
        q_res_true: torch.Tensor | None = None,
        q_res_mask: torch.Tensor | None = None,
    ) -> dict:
        pos_per_elem = self.pos_loss(pred_pos, target_pos)
        dim_w = self.dim_weights.to(device=pred_pos.device, dtype=pred_pos.dtype).view(1, 1, -1)
        pos_per_elem_weighted = pos_per_elem * dim_w
        diff = pred_pos - target_pos
        horizontal_per_t = torch.sqrt(diff[..., 0] ** 2 + diff[..., 1] ** 2 + 1e-6)
        vertical_per_t = pos_per_elem_weighted[..., 2]
        valid_mask = seq_mask > 0.5
        anchor_mask = (obs_mask > 0.5) & valid_mask
        gap_mask = (obs_mask <= 0.5) & valid_mask
        long_gap_mask = _build_long_gap_mask(
            obs_mask=obs_mask, seq_mask=seq_mask, long_gap_threshold=int(long_gap_threshold)
        )

        anchor_points = anchor_mask.float().sum()
        gap_points = gap_mask.float().sum()
        valid_points = valid_mask.float().sum()
        anchor_f = anchor_mask.float()
        gap_f = gap_mask.float()
        valid_f = valid_mask.float()
        point_weight = self.anchor_weight * anchor_f + self.gap_weight * gap_f
        weighted_points = point_weight.sum()

        anchor_h_sum = (horizontal_per_t * anchor_f).sum()
        gap_h_sum = (horizontal_per_t * gap_f).sum()
        vertical_sum_raw = (vertical_per_t * point_weight).sum()

        anchor_raw_mean = anchor_h_sum / (anchor_points + 1e-6)
        gap_raw_mean = gap_h_sum / (gap_points + 1e-6)

        anchor_weighted_sum = self.anchor_weight * anchor_h_sum
        gap_weighted_sum = self.gap_weight * gap_h_sum
        horizontal_weighted_mean = (anchor_weighted_sum + gap_weighted_sum) / (weighted_points + 1e-6)
        vertical_mean_raw = vertical_sum_raw / (weighted_points + 1e-6)
        vertical_mean_weighted = self.gap_alt_weight * vertical_mean_raw
        loss_xy = horizontal_weighted_mean
        loss_z = self.alpha_vertical * vertical_mean_weighted
        pos = loss_xy + loss_z
        aux_pos_loss = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        if self.lambda_aux > 0.0 and pred_pos_aux is not None:
            aux_per_elem = self.pos_loss(pred_pos_aux, target_pos)
            aux_dim_mask = None
            if pred_pos_aux_supervise_dims is not None:
                aux_dim_mask = pred_pos_aux_supervise_dims.to(device=pred_pos.device, dtype=pred_pos.dtype)
                aux_per_elem = aux_per_elem * aux_dim_mask
            aux_per_elem_weighted = aux_per_elem * dim_w
            aux_diff = pred_pos_aux - target_pos
            aux_horizontal_per_t = torch.sqrt(aux_diff[..., 0] ** 2 + aux_diff[..., 1] ** 2 + 1e-6)
            aux_vertical_per_t = aux_per_elem_weighted[..., 2]
            aux_anchor_h_sum = (aux_horizontal_per_t * anchor_f).sum()
            aux_gap_h_sum = (aux_horizontal_per_t * gap_f).sum()
            aux_anchor_weighted_sum = self.anchor_weight * aux_anchor_h_sum
            aux_gap_weighted_sum = self.gap_weight * aux_gap_h_sum
            aux_horizontal_weighted_mean = (aux_anchor_weighted_sum + aux_gap_weighted_sum) / (weighted_points + 1e-6)
            if aux_dim_mask is not None and float(aux_dim_mask[..., 2].max().detach().cpu()) <= 0.0:
                aux_vertical_mean_weighted = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
            else:
                aux_vertical_sum_raw = (aux_vertical_per_t * point_weight).sum()
                aux_vertical_mean_raw = aux_vertical_sum_raw / (weighted_points + 1e-6)
                aux_vertical_mean_weighted = self.gap_alt_weight * aux_vertical_mean_raw
            aux_pos_loss = aux_horizontal_weighted_mean + self.alpha_vertical * aux_vertical_mean_weighted

        vprog_loss = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        if self.lambda_vprog > 0.0 and q_pred is not None and q_true is not None and q_mask is not None:
            q_pred_use = q_pred.squeeze(-1)
            q_true_use = q_true.squeeze(-1) if q_true.dim() == q_pred.dim() else q_true
            q_mask_use = q_mask.squeeze(-1) if q_mask.dim() == q_pred.dim() else q_mask
            q_mask_use = q_mask_use.to(device=pred_pos.device, dtype=pred_pos.dtype)
            q_per_elem = F.smooth_l1_loss(q_pred_use, q_true_use, reduction="none")
            vprog_loss = (q_per_elem * q_mask_use).sum() / (q_mask_use.sum() + 1e-6)

        vprog_res_loss = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        if self.lambda_vprog_res > 0.0 and q_res_pred is not None and q_res_true is not None and q_res_mask is not None:
            q_res_pred_use = q_res_pred.squeeze(-1)
            q_res_true_use = q_res_true.squeeze(-1) if q_res_true.dim() == q_res_pred.dim() else q_res_true
            q_res_mask_use = q_res_mask.squeeze(-1) if q_res_mask.dim() == q_res_pred.dim() else q_res_mask
            q_res_mask_use = q_res_mask_use.to(device=pred_pos.device, dtype=pred_pos.dtype)
            q_res_per_elem = F.smooth_l1_loss(q_res_pred_use, q_res_true_use, reduction="none")
            vprog_res_loss = (q_res_per_elem * q_res_mask_use).sum() / (q_res_mask_use.sum() + 1e-6)

        raw_total = anchor_h_sum + gap_h_sum + 1e-6
        weighted_total = anchor_weighted_sum + gap_weighted_sum + 1e-6

        diff_pred = pred_pos[:, 1:, :] - pred_pos[:, :-1, :]
        diff_tgt = target_pos[:, 1:, :] - target_pos[:, :-1, :]
        smooth_mask = (seq_mask[:, 1:] * seq_mask[:, :-1]).unsqueeze(-1)
        smooth = torch.abs(diff_pred - diff_tgt) * smooth_mask
        smooth = smooth.mean()
        vertical_smooth_loss = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        if self.lambda_vertical_smooth > 0.0:
            # Weak vertical shape prior: only apply on consecutive GAP points in model target space.
            # This avoids forcing anchor points and avoids cross anchor-gap boundary smoothing.
            gap_pair_mask = (gap_mask[:, 1:] & gap_mask[:, :-1]).float()
            d_alt_pred = pred_pos[:, 1:, 2] - pred_pos[:, :-1, 2]
            d_alt_tgt = target_pos[:, 1:, 2] - target_pos[:, :-1, 2]
            d_alt_loss = F.smooth_l1_loss(d_alt_pred, d_alt_tgt, reduction="none")
            vertical_smooth_loss = (d_alt_loss * gap_pair_mask).sum() / (gap_pair_mask.sum() + 1e-6)

        # Optional altitude residual supervision (for AltBaseResidual variants).
        alt_residual_loss = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        alt_absolute_aux_loss = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        alt_edge_delta_loss = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        anchor_consistency_loss = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        alt_edge_first_diff_loss = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        alt_edge_second_diff_loss = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        alt_segment_bound_loss = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        var_reg_loss = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        alt_gate_supervision_loss = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        alt_gate_risk_shrink_loss = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        first_step_anchor_loss = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        second_step_anchor_loss = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        local_spike_jump_loss = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        local_spike_curve_loss = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        target_rightstep2_jump_loss = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        target_rightstep2_curve_loss = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        target_rightstep2_value_loss = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        target_rightstep2_boundary_pull_loss = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        high_risk_gap_alt_rmse_proxy = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        high_risk_edge_alt_jump_proxy = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        gate_mean_all = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        gate_mean_risk = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        gate_mean_nonrisk = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        gate_mean_bucket0 = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        gate_mean_bucket1 = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        gate_mean_bucket2 = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)

        if self.lambda_anchor_consistency > 0.0:
            anc_w = anchor_mask.float()
            if self.anchor_boundary_weight > 1.0:
                anchor_boundary = _build_anchor_boundary_mask(anchor_mask, gap_mask).float()
                anc_w = anc_w * (1.0 + (self.anchor_boundary_weight - 1.0) * anchor_boundary)
            l_anc = F.smooth_l1_loss(pred_pos[..., 2], target_pos[..., 2], reduction="none")
            anchor_consistency_loss = (l_anc * anc_w).sum() / (anc_w.sum() + 1e-6)

        if self.lambda_alt_edge_first_diff > 0.0 or self.lambda_alt_edge_second_diff > 0.0:
            edge_pair_mask = _build_gap_edge_pair_mask(gap_mask, self.alt_edge_steps).float()
            if self.lambda_alt_edge_first_diff > 0.0:
                d1_pred = pred_pos[:, 1:, 2] - pred_pos[:, :-1, 2]
                d1_tgt = target_pos[:, 1:, 2] - target_pos[:, :-1, 2]
                l_d1 = F.smooth_l1_loss(d1_pred, d1_tgt, reduction="none")
                alt_edge_first_diff_loss = (l_d1 * edge_pair_mask).sum() / (edge_pair_mask.sum() + 1e-6)
            if self.lambda_alt_edge_second_diff > 0.0:
                edge_second_mask = _build_gap_edge_second_mask(gap_mask, self.alt_edge_steps).float()
                d2_pred = pred_pos[:, 2:, 2] - 2.0 * pred_pos[:, 1:-1, 2] + pred_pos[:, :-2, 2]
                d2_tgt = target_pos[:, 2:, 2] - 2.0 * target_pos[:, 1:-1, 2] + target_pos[:, :-2, 2]
                l_d2 = F.smooth_l1_loss(d2_pred, d2_tgt, reduction="none")
                alt_edge_second_diff_loss = (l_d2 * edge_second_mask).sum() / (edge_second_mask.sum() + 1e-6)

        alt_vertical_rate_penalty = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        if self.lambda_alt_vertical_rate_penalty > 0.0:
            # Penalise vertical speed exceeding cruise limit (300 ft/min by default).
            # Only applied in gap regions — anchors are assumed correct.
            gap_f = gap_mask.float()
            v_rate = pred_pos[:, 1:, 2] - pred_pos[:, :-1, 2]   # Δh per minute
            excess = torch.relu(torch.abs(v_rate) - self.alt_vertical_rate_max)
            # Apply only on gap-to-gap transitions (both sides in gap)
            gap_pair = gap_f[:, 1:] * gap_f[:, :-1]
            alt_vertical_rate_penalty = (excess ** 2 * gap_pair).sum() / (gap_pair.sum() + 1e-6)

        alt_boundary_anchor_loss = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        if self.lambda_alt_boundary_anchor > 0.0 and left_boundary_alt is not None and right_boundary_alt is not None:
            # Penalise altitude deviation at anchor positions adjacent to gaps.
            # Build a mask for positions immediately adjacent to gap boundaries.
            bsz, tlen = obs_mask.shape[:2]
            anchor_mask = (obs_mask > 0.5).float()
            # First gap step after a left anchor
            gap_f = gap_mask.float()
            left_boundary = torch.zeros_like(gap_f)
            left_boundary[:, 1:] = anchor_mask[:, :-1] * gap_f[:, 1:]  # anchor just before gap
            right_boundary = torch.zeros_like(gap_f)
            right_boundary[:, :-1] = anchor_mask[:, 1:] * gap_f[:, :-1]  # anchor just after gap
            boundary_mask = torch.clamp(left_boundary + right_boundary, 0.0, 1.0)
            # L1 penalty on altitude deviation at boundary-adjacent gap positions
            alt_err = torch.abs(pred_pos[..., 2] - target_pos[..., 2])
            alt_boundary_anchor_loss = (alt_err * boundary_mask).sum() / (boundary_mask.sum() + 1e-6)

        if self.lambda_alt_segment_bound > 0.0:
            target_alt = target_pos[..., 2]
            pred_alt = pred_pos[..., 2]
            seg_base, seg_cap = self._build_segment_bound_targets(
                target_alt=target_alt,
                obs_mask=obs_mask,
                seq_mask=seq_mask,
                gap_mask=gap_mask,
            )
            pred_delta = torch.abs(pred_alt - seg_base)
            excess = torch.relu(pred_delta - seg_cap)
            alt_segment_bound_loss = (excess * gap_mask.float()).sum() / (gap_mask.float().sum() + 1e-6)
        if self.lambda_var_reg > 0.0:
            # Penalize per-sample variance collapse in gap regions:
            # if pred_std / true_std < var_reg_min_ratio, apply linear penalty.
            gap_f = gap_mask.float()
            pred_alt = pred_pos[..., 2]
            pred_mean = (pred_alt * gap_f).sum(dim=1, keepdim=True) / (gap_f.sum(dim=1, keepdim=True) + 1e-6)
            pred_var = (((pred_alt - pred_mean) ** 2) * gap_f).sum(dim=1) / (gap_f.sum(dim=1) + 1e-6)
            true_alt = target_pos[..., 2]
            true_mean = (true_alt * gap_f).sum(dim=1, keepdim=True) / (gap_f.sum(dim=1, keepdim=True) + 1e-6)
            true_var = (((true_alt - true_mean) ** 2) * gap_f).sum(dim=1) / (gap_f.sum(dim=1) + 1e-6)
            var_ratio = pred_var / (true_var + 1e-6)
            var_reg_loss = torch.relu(float(self.var_reg_min_ratio) - var_ratio).mean()
        if (
            self.lambda_alt_residual > 0.0
            and alt_base is not None
            and residual_bound is not None
            and delta_alt_pred_norm is not None
        ):
            gap_f = gap_mask.float()
            if self.alt_edge_steps > 0 and self.alt_edge_weight > 1.0:
                edge_mask = _build_gap_edge_mask(gap_mask, self.alt_edge_steps).float()
                gap_point_weight = gap_f * (1.0 + (self.alt_edge_weight - 1.0) * edge_mask)
            else:
                gap_point_weight = gap_f
            if edge_weight is not None:
                ew = edge_weight.to(device=pred_pos.device, dtype=pred_pos.dtype).view(-1, 1).expand_as(gap_point_weight)
                gap_point_weight = gap_point_weight * torch.clamp(ew, min=0.1)
            delta_gt = target_pos[..., 2] - alt_base
            delta_gt_norm = torch.clamp(delta_gt / (residual_bound + 1e-6), min=-1.0, max=1.0)
            l_res = F.smooth_l1_loss(delta_alt_pred_norm, delta_gt_norm, reduction="none")
            alt_residual_loss = (l_res * gap_point_weight).sum() / (gap_point_weight.sum() + 1e-6)

            l_abs = F.smooth_l1_loss(pred_pos[..., 2], target_pos[..., 2], reduction="none")
            alt_absolute_aux_loss = (l_abs * gap_point_weight).sum() / (gap_point_weight.sum() + 1e-6)

            if self.lambda_alt_edge_delta > 0.0:
                gap_pair_mask = (gap_mask[:, 1:] & gap_mask[:, :-1]).float()
                if self.alt_edge_steps > 0 and self.alt_edge_weight > 1.0:
                    edge_pair_mask = _build_gap_edge_pair_mask(gap_mask, self.alt_edge_steps).float()
                    gap_pair_weight = gap_pair_mask * (1.0 + (self.alt_edge_weight - 1.0) * edge_pair_mask)
                else:
                    gap_pair_weight = gap_pair_mask
                d_alt_pred = pred_pos[:, 1:, 2] - pred_pos[:, :-1, 2]
                d_alt_tgt = target_pos[:, 1:, 2] - target_pos[:, :-1, 2]
                d_alt_loss = F.smooth_l1_loss(d_alt_pred, d_alt_tgt, reduction="none")
                alt_edge_delta_loss = (d_alt_loss * gap_pair_weight).sum() / (gap_pair_weight.sum() + 1e-6)

        aux_alt = pred_pos[..., 2]
        if self.aux_alt_loss_series in {"pred_pos_main", "main"} and pred_pos_main is not None:
            aux_alt = pred_pos_main[..., 2]

        if left_boundary_alt is not None and (self.use_first_step_anchor_loss or self.use_second_step_anchor_loss):
            left_b = left_boundary_alt.to(device=pred_pos.device, dtype=pred_pos.dtype).view(-1, 1)
            first_mask, second_mask = _build_gap_first_second_masks(gap_mask)
            if self.use_first_step_anchor_loss and self.first_step_anchor_lambda > 0.0:
                m1 = first_mask.float()
                l1 = torch.abs(aux_alt - left_b)
                first_step_anchor_loss = (l1 * m1).sum() / (m1.sum() + 1e-6)
            if self.use_second_step_anchor_loss and self.second_step_anchor_lambda > 0.0:
                m2 = second_mask.float()
                l2 = torch.abs(aux_alt - left_b)
                second_step_anchor_loss = (l2 * m2).sum() / (m2.sum() + 1e-6)

        if (
            self.use_local_spike_loss
            and segment_bucket is not None
            and anchor_pattern is not None
            and (self.local_spike_lambda_jump > 0.0 or self.local_spike_lambda_curve > 0.0)
        ):
            sb = segment_bucket.to(device=pred_pos.device).view(-1)
            ap = anchor_pattern.to(device=pred_pos.device).view(-1)
            sample_sel = ((sb == int(self.local_spike_target_bucket_id)) & (ap == int(self.local_spike_target_pattern_id))).float()
            if sample_sel.sum() > 0:
                sample_mask_bt = sample_sel.view(-1, 1).expand_as(gap_mask.float())
                if self.local_spike_use_rightstep2 and self.local_spike_lambda_jump > 0.0:
                    pair_mask = _build_gap_rightstep2_pair_mask(gap_mask).float()
                    sample_mask_pair = sample_sel.view(-1, 1).expand_as(pair_mask)
                    m_jump = pair_mask * sample_mask_pair
                    d1 = torch.abs(aux_alt[:, 1:] - aux_alt[:, :-1])
                    local_spike_jump_loss = (d1 * m_jump).sum() / (m_jump.sum() + 1e-6)
                if self.local_spike_use_second_diff and self.local_spike_lambda_curve > 0.0:
                    d2 = torch.abs(aux_alt[:, 2:] - 2.0 * aux_alt[:, 1:-1] + aux_alt[:, :-2])
                    d2_mask = _build_gap_rightstep2_second_diff_mask(gap_mask, seq_mask).float()
                    sample_mask_d2 = sample_sel.view(-1, 1).expand_as(d2_mask)
                    m_curve = d2_mask * sample_mask_d2
                    local_spike_curve_loss = (d2 * m_curve).sum() / (m_curve.sum() + 1e-6)

        if (
            self.use_targeted_rightstep2_loss
            and segment_bucket is not None
            and anchor_pattern is not None
            and (
                self.target_jump_lambda > 0.0
                or self.target_curve_lambda > 0.0
                or (self.use_target_value_rightstep2_loss and self.target_value_lambda > 0.0)
                or (self.use_target_rightstep2_boundary_pull and self.target_rightstep2_boundary_pull_lambda > 0.0)
            )
        ):
            sb = segment_bucket.to(device=pred_pos.device).view(-1)
            ap = anchor_pattern.to(device=pred_pos.device).view(-1)
            sample_sel = ((sb == int(self.target_bucket_id)) & (ap == int(self.target_anchor_pattern_id))).float()
            if sample_sel.sum() > 0:
                if self.use_target_jump_loss and self.target_jump_lambda > 0.0:
                    pair_mask = _build_gap_rightstep2_pair_mask(gap_mask).float()
                    sample_mask_pair = sample_sel.view(-1, 1).expand_as(pair_mask)
                    m_jump = pair_mask * sample_mask_pair
                    d1 = torch.abs(aux_alt[:, 1:] - aux_alt[:, :-1])
                    target_rightstep2_jump_loss = (d1 * m_jump).sum() / (m_jump.sum() + 1e-6)
                if self.use_target_curve_loss and self.target_curve_lambda > 0.0:
                    d2 = torch.abs(aux_alt[:, 2:] - 2.0 * aux_alt[:, 1:-1] + aux_alt[:, :-2])
                    d2_mask = _build_gap_rightstep2_second_diff_mask(gap_mask, seq_mask).float()
                    sample_mask_d2 = sample_sel.view(-1, 1).expand_as(d2_mask)
                    m_curve = d2_mask * sample_mask_d2
                    target_rightstep2_curve_loss = (d2 * m_curve).sum() / (m_curve.sum() + 1e-6)
                if (
                    self.use_target_value_rightstep2_loss
                    and self.target_value_lambda > 0.0
                    and right_boundary_alt is not None
                ):
                    pt_mask = _build_gap_rightstep2_point_mask(gap_mask, seq_mask).float()
                    # Require a valid previous timestep so target h* = λ*h_{t-3} + (1-λ)*h_right is defined.
                    prev_mask = torch.zeros_like(pt_mask)
                    prev_mask[:, 1:] = pt_mask[:, :-1]
                    sample_mask_pt = sample_sel.view(-1, 1).expand_as(pt_mask)
                    m_target = pt_mask * prev_mask * sample_mask_pt

                    right_b = right_boundary_alt.to(device=pred_pos.device, dtype=pred_pos.dtype).view(-1, 1)
                    aux_prev = torch.zeros_like(aux_alt)
                    aux_prev[:, 1:] = aux_alt[:, :-1]
                    lam = float(self.target_interp_lambda)
                    target_step2 = lam * aux_prev + (1.0 - lam) * right_b
                    target_rightstep2_value_loss = (torch.abs(aux_alt - target_step2) * m_target).sum() / (m_target.sum() + 1e-6)
                if (
                    self.use_target_rightstep2_boundary_pull
                    and self.target_rightstep2_boundary_pull_lambda > 0.0
                    and right_boundary_alt is not None
                ):
                    pt_mask = _build_gap_rightstep2_point_mask(gap_mask, seq_mask).float()
                    sample_mask_pt = sample_sel.view(-1, 1).expand_as(pt_mask)
                    m_pull = pt_mask * sample_mask_pt
                    right_b = right_boundary_alt.to(device=pred_pos.device, dtype=pred_pos.dtype).view(-1, 1)
                    pull = torch.abs(aux_alt - right_b)
                    target_rightstep2_boundary_pull_loss = (pull * m_pull).sum() / (m_pull.sum() + 1e-6)

        # Optional gate supervision for DMS altitude increment branch.
        if alt_gate is not None:
            gap_f = gap_mask.float()
            gate_bt = alt_gate
            gate_mean_per_sample = (gate_bt * gap_f).sum(dim=1) / (gap_f.sum(dim=1) + 1e-6)
            gate_mean_all = gate_mean_per_sample.mean()
            if risk_flag is not None:
                r = (risk_flag > 0.5).float().view(-1)
                if risk_flag_teacher is not None:
                    r = torch.maximum(r, (risk_flag_teacher > 0.5).float().view(-1))
                gate_mean_risk = (gate_mean_per_sample * r).sum() / (r.sum() + 1e-6)
                gate_mean_nonrisk = (gate_mean_per_sample * (1.0 - r)).sum() / ((1.0 - r).sum() + 1e-6)
                if self.lambda_alt_gate_risk_shrink > 0.0:
                    tgt = torch.full_like(gate_mean_per_sample, float(self.alt_gate_risk_target))
                    if teacher_scale is not None:
                        tsc = teacher_scale.to(device=pred_pos.device, dtype=pred_pos.dtype).view(-1)
                        tgt = torch.minimum(tgt, tsc)
                    risk_excess = torch.relu(gate_mean_per_sample - tgt) * r
                    alt_gate_risk_shrink_loss = risk_excess.sum() / (r.sum() + 1e-6)

                # high-risk altitude error proxy (gap-only)
                alt_err = pred_pos[..., 2] - target_pos[..., 2]
                r_bt = r.view(-1, 1).expand_as(gap_f)
                m = gap_f * r_bt
                high_risk_gap_alt_rmse_proxy = torch.sqrt(((alt_err**2) * m).sum() / (m.sum() + 1e-6))

                # edge-jump proxy on high-risk gaps
                edge_pair = _build_gap_edge_pair_mask(gap_mask, max(1, self.alt_edge_steps)).float()
                d_alt_pred = pred_pos[:, 1:, 2] - pred_pos[:, :-1, 2]
                m_edge = edge_pair * r.view(-1, 1)
                high_risk_edge_alt_jump_proxy = (torch.abs(d_alt_pred) * m_edge).sum() / (m_edge.sum() + 1e-6)

            if segment_bucket is not None:
                sb = segment_bucket.to(device=pred_pos.device).view(-1)
                for bid, name in [(0, "bucket0"), (1, "bucket1"), (2, "bucket2")]:
                    bm = (sb == bid).float()
                    gm = (gate_mean_per_sample * bm).sum() / (bm.sum() + 1e-6)
                    if name == "bucket0":
                        gate_mean_bucket0 = gm
                    elif name == "bucket1":
                        gate_mean_bucket1 = gm
                    else:
                        gate_mean_bucket2 = gm

            if self.lambda_alt_gate_supervision > 0.0 and teacher_scale is not None:
                tsc = teacher_scale.to(device=pred_pos.device, dtype=pred_pos.dtype).view(-1)
                alt_gate_supervision_loss = F.smooth_l1_loss(gate_mean_per_sample, tsc, reduction="mean")

        unc = torch.tensor(0.0, device=pred_pos.device)
        if self.lambda_unc > 0 and logvar is not None:
            mse = ((pred_pos - target_pos) ** 2).mean(dim=-1, keepdim=True)
            nll = 0.5 * (torch.exp(-logvar) * mse + logvar)
            unc = (nll.squeeze(-1) * seq_mask).sum() / (seq_mask.sum() + 1e-6)

        cruise_phys_loss = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        cruise_speed_smooth = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        cruise_heading_rate = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        cruise_vertical_rate = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        cruise_planar_accel = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        cruise_weight_mean = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        cruise_gap_points = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        multi_scale_planar_loss = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        multi_scale_alt_loss = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        multi_scale_points = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        multi_scale_losses: dict[int, torch.Tensor] = {}
        multi_scale_alt_losses: dict[int, torch.Tensor] = {}
        multi_scale_counts: dict[int, torch.Tensor] = {}
        fusion_reg_loss = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        if self.lambda_cruise_phys > 0.0 and exo is not None:
            def _exo_col(name: str) -> torch.Tensor:
                idx = self.exo_index.get(name, -1)
                if idx < 0 or idx >= exo.shape[-1]:
                    return torch.zeros_like(obs_mask, dtype=pred_pos.dtype)
                return exo[..., idx].to(dtype=pred_pos.dtype)

            exo_vs = torch.abs(_exo_col("vertical_speed"))
            exo_sd = torch.abs(_exo_col("speed_delta"))
            exo_tr = torch.abs(_exo_col("turn_rate"))
            cruise_proxy = (
                (exo_vs <= self.cruise_max_abs_vertical_rate)
                & (exo_sd <= self.cruise_max_speed_delta)
                & (exo_tr <= self.cruise_max_heading_rate)
                & gap_mask
            )
            cruise_gap_points = cruise_proxy.float().sum()

            # Exogenous/quality-adaptive weights: higher-quality and more-stable points get stronger physics regularization.
            exo_stability = torch.exp(
                -(
                    exo_vs / self.cruise_max_abs_vertical_rate
                    + exo_sd / self.cruise_max_speed_delta
                    + exo_tr / self.cruise_max_heading_rate
                )
            )
            if quality is not None and quality.shape[-1] > 0:
                q_abs = torch.nan_to_num(quality, nan=0.0, posinf=0.0, neginf=0.0).abs().mean(dim=-1)
                q_weight = torch.sigmoid(q_abs)
            else:
                q_weight = torch.ones_like(obs_mask, dtype=pred_pos.dtype) * 0.5
            adaptive_w = torch.clamp(
                exo_stability * (1.0 + self.cruise_quality_weight_gain * q_weight),
                min=0.1,
                max=3.0,
            )

            # 1-step dynamics on cruise-gap points.
            pair_mask = cruise_proxy[:, 1:] & cruise_proxy[:, :-1]
            pair_w = torch.minimum(adaptive_w[:, 1:], adaptive_w[:, :-1]) * pair_mask.float()
            pair_denom = pair_w.sum() + 1e-6
            if pair_mask.any():
                speed = torch.sqrt(pred_pos[..., 0] ** 2 + pred_pos[..., 1] ** 2 + 1e-6)
                cruise_speed_smooth = (torch.abs(speed[:, 1:] - speed[:, :-1]) * pair_w).sum() / pair_denom

                hd = torch.atan2(pred_pos[..., 1], pred_pos[..., 0] + 1e-6)
                d_hd = hd[:, 1:] - hd[:, :-1]
                d_hd = (d_hd + torch.pi) % (2.0 * torch.pi) - torch.pi
                cruise_heading_rate = (torch.abs(d_hd) * pair_w).sum() / pair_denom

                d_alt = pred_pos[:, 1:, 2] - pred_pos[:, :-1, 2]
                cruise_vertical_rate = (torch.abs(d_alt) * pair_w).sum() / pair_denom

            # 2-step planar acceleration smoothness on cruise-gap points.
            tri_mask = cruise_proxy[:, 2:] & cruise_proxy[:, 1:-1] & cruise_proxy[:, :-2]
            tri_w = torch.minimum(torch.minimum(adaptive_w[:, 2:], adaptive_w[:, 1:-1]), adaptive_w[:, :-2]) * tri_mask.float()
            tri_denom = tri_w.sum() + 1e-6
            if tri_mask.any():
                dd_e = pred_pos[:, 2:, 0] - 2.0 * pred_pos[:, 1:-1, 0] + pred_pos[:, :-2, 0]
                dd_n = pred_pos[:, 2:, 1] - 2.0 * pred_pos[:, 1:-1, 1] + pred_pos[:, :-2, 1]
                cruise_planar_accel = (torch.sqrt(dd_e**2 + dd_n**2 + 1e-6) * tri_w).sum() / tri_denom

            cruise_phys_loss = (
                self.cruise_speed_smooth_weight * cruise_speed_smooth
                + self.cruise_heading_rate_weight * cruise_heading_rate
                + self.cruise_vertical_rate_weight * cruise_vertical_rate
                + self.cruise_planar_accel_weight * cruise_planar_accel
            )
            cruise_weight_mean = (adaptive_w * cruise_proxy.float()).sum() / (cruise_gap_points + 1e-6)

        if self.lambda_multi_scale > 0.0 and self.multi_scale_scales:
            de_pred = pred_pos[..., 0]
            dn_pred = pred_pos[..., 1]
            de_tgt = target_pos[..., 0]
            dn_tgt = target_pos[..., 1]
            t_len = pred_pos.shape[1]
            # Build relative absolute tracks by cumulative sum over dE/dN so that
            # k-step displacement is defined as E[t+k]-E[t] / N[t+k]-N[t].
            pred_abs_e = torch.cumsum(de_pred, dim=1)
            pred_abs_n = torch.cumsum(dn_pred, dim=1)
            tgt_abs_e = torch.cumsum(de_tgt, dim=1)
            tgt_abs_n = torch.cumsum(dn_tgt, dim=1)
            for k in self.multi_scale_scales:
                if k >= t_len:
                    continue
                # Valid starts t where [t, ..., t+k] are all gap points (continuous gap window),
                # so no anchor/pad/cross-boundary window contributes.
                win_len = t_len - k
                if win_len <= 0:
                    continue
                mask_k = gap_mask[:, :win_len] & gap_mask[:, k : k + win_len]
                for j in range(1, k):
                    mask_k = mask_k & gap_mask[:, j : j + win_len]
                n_k = mask_k.float().sum()
                if n_k <= 0:
                    multi_scale_losses[k] = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
                    multi_scale_counts[k] = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
                    continue
                pred_sum_de = pred_abs_e[:, k:] - pred_abs_e[:, :-k]
                pred_sum_dn = pred_abs_n[:, k:] - pred_abs_n[:, :-k]
                tgt_sum_de = tgt_abs_e[:, k:] - tgt_abs_e[:, :-k]
                tgt_sum_dn = tgt_abs_n[:, k:] - tgt_abs_n[:, :-k]
                l_de = F.smooth_l1_loss(pred_sum_de, tgt_sum_de, reduction="none")
                l_dn = F.smooth_l1_loss(pred_sum_dn, tgt_sum_dn, reduction="none")
                if self.multi_scale_include_alt:
                    # Optional altitude k-step trend term for height-focused runs.
                    pred_u = pred_pos[..., 2]
                    tgt_u = target_pos[..., 2]
                    pred_sum_u = pred_u[:, k:] - pred_u[:, :-k]
                    tgt_sum_u = tgt_u[:, k:] - tgt_u[:, :-k]
                    l_du = F.smooth_l1_loss(pred_sum_u, tgt_sum_u, reduction="none")
                    l_k = (((l_de + l_dn + l_du) / 3.0) * mask_k.float()).sum() / (n_k + 1e-6)
                    l_u_k = (l_du * mask_k.float()).sum() / (n_k + 1e-6)
                else:
                    l_k = ((l_de + l_dn) * mask_k.float()).sum() / (n_k + 1e-6)
                    l_u_k = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
                multi_scale_losses[k] = l_k
                multi_scale_alt_losses[k] = l_u_k
                multi_scale_counts[k] = n_k
                multi_scale_planar_loss = multi_scale_planar_loss + l_k
                multi_scale_alt_loss = multi_scale_alt_loss + l_u_k
                multi_scale_points = multi_scale_points + n_k
            if len(self.multi_scale_scales) > 0:
                multi_scale_planar_loss = multi_scale_planar_loss / max(float(len(self.multi_scale_scales)), 1.0)
                multi_scale_alt_loss = multi_scale_alt_loss / max(float(len(self.multi_scale_scales)), 1.0)

        if (
            self.fusion_reg_lambda > 0.0
            and fusion_weights is not None
            and dt_prev is not None
            and dt_next is not None
        ):
            rel_pos = dt_prev / (dt_prev + dt_next + 1e-6)
            wf_target = 1.0 - rel_pos
            wb_target = rel_pos
            wf = fusion_weights[..., 0]
            wb = fusion_weights[..., 1]
            # Fusion regularization applies only on gap points, and can emphasize long-gap points.
            m = gap_mask.float() * (
                1.0 + (self.fusion_reg_long_gap_weight - 1.0) * long_gap_mask.float()
            )
            denom = m.sum() + 1e-6
            fusion_reg_loss = (
                (F.smooth_l1_loss(wf, wf_target, reduction="none") * m).sum()
                + (F.smooth_l1_loss(wb, wb_target, reduction="none") * m).sum()
            ) / (2.0 * denom)

        alt_aux_loss = torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
        if self.lambda_alt_aux > 0.0 and mu_f is not None and mu_b is not None and dt_prev is not None and dt_next is not None:
            gap_f = gap_mask.float()
            # Supervise forward LSTM altitude near left boundary,
            # backward LSTM altitude near right boundary.
            gap_pos = dt_prev / (dt_prev + dt_next + 1e-6)
            w_fwd = (1.0 - gap_pos) * gap_f
            w_bwd = gap_pos * gap_f
            l_fwd = F.smooth_l1_loss(mu_f[..., 2], target_pos[..., 2], reduction="none")
            l_bwd = F.smooth_l1_loss(mu_b[..., 2], target_pos[..., 2], reduction="none")
            alt_aux_loss = (
                (l_fwd * w_fwd).sum() + (l_bwd * w_bwd).sum()
            ) / (w_fwd.sum() + w_bwd.sum() + 1e-6)

        (
            savca_alloc_loss,
            savca_state_loss,
            savca_smooth_loss,
            savca_center_loss,
            savca_shape_loss,
            savca_nonlinear_loss,
            savca_supervised_segments,
            savca_center_shift_mean,
            savca_center_shift_long,
            savca_p_entropy_mean,
            savca_p_max_mean,
            savca_state_conf_mean,
            savca_shape_error_final,
            savca_shape_error_linear,
            savca_shape_error_savca,
            savca_transition_concentration_adsb,
            savca_transition_concentration_a1,
            savca_shape_gain_vs_a1,
            savca_d_nonlinear_mean,
            savca_m_change_ratio,
            savca_shape_error_final_change,
            savca_d_nonlinear_change,
            savca_fused_minus_a1_mean,
            savca_fused_minus_a1_long,
            savca_fused_minus_a1_change,
            savca_change_score_loss,
            savca_change_score_mean,
            savca_change_score_pos_mean,
            savca_change_score_neg_mean,
            savca_change_score_acc,
            savca_beta_floor_pred_mean,
            savca_beta_change_mean,
            savca_beta_nonchange_mean,
        ) = self._savca_supervision_losses(
            target_pos=target_pos,
            target_alt_abs=savca_target_alt_abs,
            obs_mask=obs_mask,
            seq_mask=seq_mask,
            savca_alloc_p=savca_alloc_p,
            savca_state=savca_state,
            savca_alloc_valid=savca_alloc_valid,
            savca_change_score=savca_change_score,
            savca_beta=savca_beta,
            savca_beta_floor_pred=savca_beta_floor_pred,
            savca_g_linear=savca_g_linear,
            savca_g_savca=savca_g_savca,
            savca_g_final=savca_g_final,
            savca_ref_linear_abs=savca_ref_linear_abs,
            savca_ref_savca_abs=savca_ref_savca_abs,
            savca_ref_final_abs=savca_ref_final_abs,
        )
        (
            fltp_shape_loss,
            fltp_center_loss,
            fltp_supervised_segments,
            fltp_center_shift_mean,
            fltp_center_shift_long,
            fltp_shape_error_final,
            fltp_shape_error_linear,
            fltp_shape_error_sig,
            fltp_d_nonlinear_mean,
            fltp_m_change_ratio,
            fltp_transition_concentration_pred,
            fltp_transition_concentration_adsb,
            fltp_fused_minus_a1_mean,
            fltp_fused_minus_a1_long,
            fltp_fused_minus_a1_change,
        ) = self._fltp_supervision_losses(
            target_alt_abs=savca_target_alt_abs if savca_target_alt_abs is not None else target_pos[..., 2],
            obs_mask=obs_mask,
            seq_mask=seq_mask,
            dt_prev=dt_prev,
            dt_next=dt_next,
            fltp_beta=fltp_beta,
            fltp_c=fltp_c,
            fltp_g_linear=fltp_g_linear,
            fltp_g_sig=fltp_g_sig,
            fltp_g_final=fltp_g_final,
            fltp_ref_linear_abs=fltp_ref_linear_abs,
            fltp_ref_final_abs=fltp_ref_final_abs,
        )
        (
            ssvr_state_loss,
            ssvr_smooth_loss,
            ssvr_state_ce,
            ssvr_state_acc,
            ssvr_supervised_segments,
            ssvr_pi_L_mean,
            ssvr_pi_T_mean,
            ssvr_pi_R_mean,
            ssvr_rho_mean,
            ssvr_state_entropy_mean,
            ssvr_d_nonlinear_mean,
            ssvr_shape_error_final,
            ssvr_fused_minus_a1_mean,
            ssvr_m_change_ratio,
        ) = self._ssvr_supervision_losses(
            target_alt_abs=savca_target_alt_abs if savca_target_alt_abs is not None else target_pos[..., 2],
            obs_mask=obs_mask,
            seq_mask=seq_mask,
            alt_fwd=alt_fwd,
            alt_bwd=alt_bwd,
            ssvr_pi_L=ssvr_pi_L,
            ssvr_pi_T=ssvr_pi_T,
            ssvr_pi_R=ssvr_pi_R,
            ssvr_rho=ssvr_rho,
            ssvr_state_logits=ssvr_state_logits,
            ssvr_z_hat=ssvr_z_hat,
            ssvr_z_linear=ssvr_z_linear,
        )

        total = (
            self.lambda_pos * pos
            + self.lambda_smooth * smooth
            + self.lambda_unc * unc
            + self.lambda_cruise_phys * cruise_phys_loss
            + self.lambda_multi_scale * multi_scale_planar_loss
            + self.fusion_reg_lambda * fusion_reg_loss
            + self.lambda_vertical_smooth * vertical_smooth_loss
            + self.lambda_alt_residual * alt_residual_loss
            + self.lambda_alt_absolute_aux * alt_absolute_aux_loss
            + self.lambda_alt_edge_delta * alt_edge_delta_loss
            + self.lambda_anchor_consistency * anchor_consistency_loss
            + self.lambda_alt_edge_first_diff * alt_edge_first_diff_loss
            + self.lambda_alt_edge_second_diff * alt_edge_second_diff_loss
            + self.lambda_alt_segment_bound * alt_segment_bound_loss
            + self.lambda_alt_vertical_rate_penalty * alt_vertical_rate_penalty
            + self.lambda_alt_boundary_anchor * alt_boundary_anchor_loss
            + self.lambda_var_reg * var_reg_loss
            + self.lambda_alt_aux * alt_aux_loss
            + self.lambda_aux * aux_pos_loss
            + self.lambda_vprog * vprog_loss
            + self.lambda_vprog_res * vprog_res_loss
            + self.lambda_alt_gate_supervision * alt_gate_supervision_loss
            + self.lambda_alt_gate_risk_shrink * alt_gate_risk_shrink_loss
            + self.first_step_anchor_lambda * first_step_anchor_loss
            + self.second_step_anchor_lambda * second_step_anchor_loss
            + self.local_spike_lambda_jump * local_spike_jump_loss
            + self.local_spike_lambda_curve * local_spike_curve_loss
            + self.target_jump_lambda * target_rightstep2_jump_loss
            + self.target_curve_lambda * target_rightstep2_curve_loss
            + self.target_value_lambda * target_rightstep2_value_loss
            + self.target_rightstep2_boundary_pull_lambda * target_rightstep2_boundary_pull_loss
            + self.lambda_savca_alloc * savca_alloc_loss
            + self.lambda_savca_state * savca_state_loss
            + self.lambda_savca_smooth * savca_smooth_loss
            + self.lambda_savca_center * savca_center_loss
            + self.lambda_savca_final_shape * savca_shape_loss
            + self.lambda_savca_nonlinear * savca_nonlinear_loss
            + self.lambda_savca_change_score * savca_change_score_loss
            + self.lambda_fltp_shape * fltp_shape_loss
            + self.lambda_fltp_center * fltp_center_loss
            + self.lambda_ssvr_state * ssvr_state_loss
            + self.lambda_ssvr_smooth * ssvr_smooth_loss
        )

        out = {
            "loss": total,
            "loss_pos": pos.detach(),
            "loss_xy": loss_xy.detach(),
            "loss_z": loss_z.detach(),
            "loss_horizontal": loss_xy.detach(),
            "loss_vertical_raw": vertical_mean_raw.detach(),
            "loss_vertical_weighted": vertical_mean_weighted.detach(),
            "loss_aux_pos": aux_pos_loss.detach(),
            "vprog_loss": vprog_loss.detach(),
            "vprog_res_loss": vprog_res_loss.detach(),
            "loss_smooth": smooth.detach(),
            "loss_unc": unc.detach(),
            "step_increment_loss": horizontal_weighted_mean.detach(),
            "loss_anchor_raw": anchor_raw_mean.detach(),
            "loss_gap_raw": gap_raw_mean.detach(),
            "loss_anchor_weighted": (self.anchor_weight * anchor_raw_mean).detach(),
            "loss_gap_weighted": (self.gap_weight * gap_raw_mean).detach(),
            "anchor_points": anchor_points.detach(),
            "gap_points": gap_points.detach(),
            "anchor_raw_ratio": (anchor_h_sum / raw_total).detach(),
            "gap_raw_ratio": (gap_h_sum / raw_total).detach(),
            "anchor_weighted_ratio": (anchor_weighted_sum / weighted_total).detach(),
            "gap_weighted_ratio": (gap_weighted_sum / weighted_total).detach(),
            "cruise_gap_points": cruise_gap_points.detach(),
            "cruise_weight_mean": cruise_weight_mean.detach(),
            "cruise_speed_smooth_loss": cruise_speed_smooth.detach(),
            "cruise_heading_rate_loss": cruise_heading_rate.detach(),
            "cruise_vertical_rate_loss": cruise_vertical_rate.detach(),
            "cruise_planar_accel_loss": cruise_planar_accel.detach(),
            "cruise_phys_loss": cruise_phys_loss.detach(),
            "multi_scale_planar_loss": multi_scale_planar_loss.detach(),
            "multi_scale_alt_loss": multi_scale_alt_loss.detach(),
            "multi_scale_points": multi_scale_points.detach(),
            "fusion_reg_loss": fusion_reg_loss.detach(),
            "vertical_smooth_loss": vertical_smooth_loss.detach(),
            "alt_residual_loss": alt_residual_loss.detach(),
            "alt_absolute_aux_loss": alt_absolute_aux_loss.detach(),
            "alt_edge_delta_loss": alt_edge_delta_loss.detach(),
            "anchor_consistency_loss": anchor_consistency_loss.detach(),
            "alt_edge_first_diff_loss": alt_edge_first_diff_loss.detach(),
            "alt_edge_second_diff_loss": alt_edge_second_diff_loss.detach(),
            "alt_segment_bound_loss": alt_segment_bound_loss.detach(),
            "alt_vertical_rate_penalty": alt_vertical_rate_penalty.detach(),
            "alt_boundary_anchor_loss": alt_boundary_anchor_loss.detach(),
            "var_reg_loss": var_reg_loss.detach(),
            "alt_gate_supervision_loss": alt_gate_supervision_loss.detach(),
            "alt_gate_risk_shrink_loss": alt_gate_risk_shrink_loss.detach(),
            "first_step_anchor_loss": first_step_anchor_loss.detach(),
            "second_step_anchor_loss": second_step_anchor_loss.detach(),
            "local_spike_jump_loss": local_spike_jump_loss.detach(),
            "local_spike_curve_loss": local_spike_curve_loss.detach(),
            "target_rightstep2_jump_loss": target_rightstep2_jump_loss.detach(),
            "target_rightstep2_curve_loss": target_rightstep2_curve_loss.detach(),
            "target_rightstep2_value_loss": target_rightstep2_value_loss.detach(),
            "target_rightstep2_boundary_pull_loss": target_rightstep2_boundary_pull_loss.detach(),
            "savca_alloc_loss": savca_alloc_loss.detach(),
            "savca_state_loss": savca_state_loss.detach(),
            "savca_smooth_loss": savca_smooth_loss.detach(),
            "savca_center_loss": savca_center_loss.detach(),
            "savca_shape_loss": savca_shape_loss.detach(),
            "savca_nonlinear_loss": savca_nonlinear_loss.detach(),
            "savca_supervised_segments": savca_supervised_segments.detach(),
            "savca_center_shift_mean": savca_center_shift_mean.detach(),
            "savca_center_shift_long": savca_center_shift_long.detach(),
            "savca_p_entropy_mean": savca_p_entropy_mean.detach(),
            "savca_p_max_mean": savca_p_max_mean.detach(),
            "savca_state_conf_mean": savca_state_conf_mean.detach(),
            "savca_shape_error_final": savca_shape_error_final.detach(),
            "savca_shape_error_linear": savca_shape_error_linear.detach(),
            "savca_shape_error_savca": savca_shape_error_savca.detach(),
            "savca_transition_concentration_pred": savca_p_max_mean.detach(),
            "savca_transition_concentration_adsb": savca_transition_concentration_adsb.detach(),
            "savca_transition_concentration_a1": savca_transition_concentration_a1.detach(),
            "savca_shape_gain_vs_a1": savca_shape_gain_vs_a1.detach(),
            "savca_d_nonlinear_mean": savca_d_nonlinear_mean.detach(),
            "savca_m_change_ratio": savca_m_change_ratio.detach(),
            "savca_shape_error_final_change": savca_shape_error_final_change.detach(),
            "savca_d_nonlinear_change": savca_d_nonlinear_change.detach(),
            "savca_fused_minus_a1_mean": savca_fused_minus_a1_mean.detach(),
            "savca_fused_minus_a1_long": savca_fused_minus_a1_long.detach(),
            "savca_fused_minus_a1_change": savca_fused_minus_a1_change.detach(),
            "savca_change_score_loss": savca_change_score_loss.detach(),
            "savca_change_score_mean": savca_change_score_mean.detach(),
            "savca_change_score_pos_mean": savca_change_score_pos_mean.detach(),
            "savca_change_score_neg_mean": savca_change_score_neg_mean.detach(),
            "savca_change_score_acc": savca_change_score_acc.detach(),
            "savca_beta_floor_pred_mean": savca_beta_floor_pred_mean.detach(),
            "savca_beta_mean_change": savca_beta_change_mean.detach(),
            "savca_beta_mean_nonchange": savca_beta_nonchange_mean.detach(),
            "fltp_shape_loss": fltp_shape_loss.detach(),
            "fltp_center_loss": fltp_center_loss.detach(),
            "fltp_supervised_segments": fltp_supervised_segments.detach(),
            "fltp_center_shift_mean": fltp_center_shift_mean.detach(),
            "fltp_center_shift_long": fltp_center_shift_long.detach(),
            "fltp_shape_error_final": fltp_shape_error_final.detach(),
            "fltp_shape_error_linear": fltp_shape_error_linear.detach(),
            "fltp_shape_error_sig": fltp_shape_error_sig.detach(),
            "fltp_d_nonlinear_mean": fltp_d_nonlinear_mean.detach(),
            "fltp_m_change_ratio": fltp_m_change_ratio.detach(),
            "fltp_transition_concentration_pred": fltp_transition_concentration_pred.detach(),
            "fltp_transition_concentration_adsb": fltp_transition_concentration_adsb.detach(),
            "fltp_fused_minus_a1_mean": fltp_fused_minus_a1_mean.detach(),
            "fltp_fused_minus_a1_long": fltp_fused_minus_a1_long.detach(),
            "fltp_fused_minus_a1_change": fltp_fused_minus_a1_change.detach(),
            "ssvr_state_loss": ssvr_state_loss.detach(),
            "ssvr_smooth_loss": ssvr_smooth_loss.detach(),
            "ssvr_state_ce": ssvr_state_ce.detach(),
            "ssvr_state_acc": ssvr_state_acc.detach(),
            "ssvr_supervised_segments": ssvr_supervised_segments.detach(),
            "ssvr_pi_L_mean": ssvr_pi_L_mean.detach(),
            "ssvr_pi_T_mean": ssvr_pi_T_mean.detach(),
            "ssvr_pi_R_mean": ssvr_pi_R_mean.detach(),
            "ssvr_rho_mean": ssvr_rho_mean.detach(),
            "ssvr_state_entropy_mean": ssvr_state_entropy_mean.detach(),
            "ssvr_d_nonlinear_mean": ssvr_d_nonlinear_mean.detach(),
            "ssvr_shape_error_final": ssvr_shape_error_final.detach(),
            "ssvr_minus_a1_mean": ssvr_fused_minus_a1_mean.detach(),
            "ssvr_m_change_ratio": ssvr_m_change_ratio.detach(),
            "alt_gate_mean": gate_mean_all.detach(),
            "alt_gate_mean_risk": gate_mean_risk.detach(),
            "alt_gate_mean_nonrisk": gate_mean_nonrisk.detach(),
            "alt_gate_mean_bucket_short": gate_mean_bucket0.detach(),
            "alt_gate_mean_bucket_medium": gate_mean_bucket1.detach(),
            "alt_gate_mean_bucket_long": gate_mean_bucket2.detach(),
            "high_risk_gap_alt_rmse_proxy": high_risk_gap_alt_rmse_proxy.detach(),
            "high_risk_edge_alt_jump_proxy": high_risk_edge_alt_jump_proxy.detach(),
            "alt_aux_loss": alt_aux_loss.detach(),
        }
        for k in self.multi_scale_scales:
            out[f"multi_scale_k{k}_loss"] = multi_scale_losses.get(
                k, torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
            ).detach()
            out[f"multi_scale_k{k}_alt_loss"] = multi_scale_alt_losses.get(
                k, torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
            ).detach()
            out[f"multi_scale_k{k}_points"] = multi_scale_counts.get(
                k, torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)
            ).detach()
        d_dim = pred_pos.shape[-1]
        pos_denom = valid_points * float(d_dim) + 1e-6
        pos_total = pos.detach() + 1e-6
        total_detached = total.detach() + 1e-6
        long_gap_f = long_gap_mask.float()
        vertical_mean = vertical_mean_weighted
        out["horizontal_increment_loss"] = horizontal_weighted_mean.detach()
        out["planar_loss"] = horizontal_weighted_mean.detach()
        out["horizontal_loss"] = horizontal_weighted_mean.detach()
        out["vertical_loss"] = vertical_mean.detach()
        for d in range(d_dim):
            err_d = pos_per_elem_weighted[..., d]
            overall_sum_d = (err_d * valid_f).sum()
            anchor_sum_d = (err_d * anchor_f).sum()
            gap_sum_d = (err_d * gap_f).sum()
            long_gap_sum_d = (err_d * long_gap_f).sum()

            overall_mean_d = overall_sum_d / (valid_points + 1e-6)
            anchor_mean_d = anchor_sum_d / (anchor_points + 1e-6)
            gap_mean_d = gap_sum_d / (gap_points + 1e-6)
            long_gap_points = long_gap_f.sum()
            long_gap_mean_d = long_gap_sum_d / (long_gap_points + 1e-6)

            weighted_pos_component_d = (self.anchor_weight * anchor_sum_d + self.gap_weight * gap_sum_d) / (
                weighted_points + 1e-6
            )
            out[f"overall_dim{d}_loss"] = overall_mean_d.detach()
            out[f"anchor_dim{d}_loss"] = anchor_mean_d.detach()
            out[f"gap_dim{d}_loss"] = gap_mean_d.detach()
            out[f"long_gap_dim{d}_loss"] = long_gap_mean_d.detach()
            out[f"pos_dim{d}_contrib"] = weighted_pos_component_d.detach()
            out[f"pos_dim{d}_contrib_ratio"] = (weighted_pos_component_d / pos_total).detach()
            out[f"total_dim{d}_contrib_ratio"] = (
                (self.lambda_pos * weighted_pos_component_d) / total_detached
            ).detach()
        return out
