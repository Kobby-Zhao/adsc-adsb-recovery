from __future__ import annotations

import math

import torch
from torch import nn

from src.models.alt_base_residual import AltitudeBaselineBuilder, AltitudeResidualHead, ResidualRangeNormalizer
from src.models.alt_dms_refiner import AltDMSRefinerV1Head
from src.models.alt_ssvr import SSVRFeatureBuilder, SSVRHeightBranch
from src.models.bidirectional_predictor import BackwardPredictor, ForwardPredictor
from src.models.fusion import ConcatLinearFusion, FixedPositionPriorFusion, HiddenStateFusion, SimpleFusionHead
from src.models.sequence_baselines import (
    BiLSTMSequencePredictor,
    BiMambaProtoSequencePredictor,
    CNNLSTMSequencePredictor,
    KalmanFilterSequencePredictor,
    MambaEncoderSequencePredictor,
    MambaProtoSequencePredictor,
    MambaRecurrentSequencePredictor,
    MambaSequencePredictor,
    TransformerSequencePredictor,
    UniLSTMPredictor,
    build_anchor_condition_features,
)


class FLTPHeightHead(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int, beta_init_bias: float = -2.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_size)),
            nn.SiLU(),
            nn.Linear(int(hidden_size), 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            self.net[-1].bias.zero_()
            self.net[-1].bias[2] = float(beta_init_bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.net(x)
        return out[..., 0], out[..., 1], out[..., 2]


class TrajectoryRecoveryModel(nn.Module):
    def __init__(
        self,
        exo_dim: int,
        quality_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        fusion_hidden_size: int = 32,
        fusion_use_exo_quality: bool = False,
        fusion_position_prior_enabled: bool = True,
        fusion_position_prior_deviation: float = 0.30,
        fusion_weight_mode: str = "scalar",
        minimal_task_adapt_baseline: bool = False,
        vertical_exo_dim: int = 0,
        alt_bias_enabled: bool = False,
        alt_bias_hidden_size: int = 32,
        alt_bias_use_exo_quality: bool = True,
        vertical_projector_enabled: bool = False,
        vertical_projector_hidden_size: int = 32,
        vertical_projector_use_vertical_exo: bool = True,
        vertical_tune_enabled: bool = False,
        vertical_tune_hidden_size: int = 16,
        vertical_tune_temperature: float = 1.0,
        vertical_tune_mode: str = "combined",
        model_variant: str = "default",
        dms_refiner_hidden_size: int = 64,
        dms_refiner_latent_dim: int = 32,
        dms_refiner_num_heads: int = 2,
        dms_refiner_ff_multiplier: int = 2,
        dms_refiner_dropout: float = 0.0,
        alt_base_builder_type: str = "auto",
        alt_base_residual_hidden_size: int = 64,
        alt_base_residual_dropout: float = 0.0,
        alt_base_residual_bounds: list[float] | tuple[float, ...] | None = None,
        alt_base_residual_bound_enabled: bool = True,
        backbone_type: str = "bilstm",
        transformer_num_heads: int = 4,
        transformer_ff_multiplier: int = 4,
        v3_anchor_hard_consistency: bool = True,
        v3_edge_residual_damp_enabled: bool = True,
        v3_edge_residual_damp_strength: float = 0.7,
        v3_edge_residual_damp_steps: int = 2,
        alt_gate_enabled: bool = False,
        alt_gate_hidden_size: int = 32,
        alt_gate_mode: str = "learned",
        alt_gate_fixed_value: float = 1.0,
        alt_anchor_hard_consistency: bool = False,
        use_left_edge_directional_constraint: bool = False,
        left_edge_direction_mode: str = "anchor_based",
        left_edge_width: int = 2,
        left_edge_direction_strength: float = 1.0,
        left_edge_clip_mode: str = "hard",
        boundary_corrector_enabled: bool = False,
        boundary_corrector_hidden_size: int = 16,
        alt_main_mode: str = "absolute",
        alt_anchor_reference_mode: str = "local_linear",
        main_rmax_m: float | None = None,
        main_rmax_ft: float | None = None,
        main_rmax_min_m: float = 91.44,
        main_rmax_slope_m_per_min: float = 4.572,
        main_rmax_max_m: float = 365.76,
        alt_residual_anchor_delta_gate_enabled: bool = False,
        alt_residual_anchor_delta_gate_low_m: float = 60.0,
        alt_residual_anchor_delta_gate_high_m: float = 180.0,
        alt_residual_anchor_delta_gate_min_scale: float = 0.0,
        alt_residual_edge_taper_enabled: bool = False,
        alt_residual_edge_taper_steps: float = 3.0,
        alt_anchor_graph_min_step_gap_min: float = 8.0,
        alt_anchor_graph_step_center_ratio: float = 0.5,
        savca_hidden_size: int = 32,
        savca_min_uniform: float = 0.05,
        savca_state_eps: float = 0.05,
        savca_beta_enabled: bool = False,
        savca_beta_hidden_size: int = 32,
        savca_beta_init_bias: float = -2.0,
        savca_beta_default_max: float = 1.0,
        savca_beta_gap_cap_enabled: bool = False,
        savca_beta_medium_gap_thr: float = 15.0,
        savca_beta_long_gap_thr: float = 45.0,
        savca_beta_cap_short: float = 0.20,
        savca_beta_cap_medium: float = 0.12,
        savca_beta_cap_long: float = 0.05,
        savca_beta_conf_gate_enabled: bool = False,
        savca_beta_state_conf_threshold: float = 0.10,
        savca_beta_shape_conf_threshold: float = 0.18,
        savca_beta_gate_scale_state: float = 10.0,
        savca_beta_gate_scale_shape: float = 10.0,
        savca_beta_shape_conf_type: str = "pmax",
        savca_beta_floor_enabled: bool = False,
        savca_beta_floor_value: float = 0.03,
        savca_change_score_enabled: bool = False,
        savca_change_score_hidden_size: int = 32,
        savca_beta_floor_from_change_score: bool = False,
        fltp_hidden_size: int = 32,
        fltp_c_min: float = 0.05,
        fltp_c_max: float = 0.95,
        fltp_w_min: float = 0.05,
        fltp_w_max: float = 0.50,
        fltp_beta_init_bias: float = -2.0,
        fltp_gap_cap_enabled: bool = True,
        fltp_medium_gap_thr: float = 15.0,
        fltp_long_gap_thr: float = 45.0,
        fltp_beta_cap_short: float = 0.20,
        fltp_beta_cap_medium: float = 0.12,
        fltp_beta_cap_long: float = 0.05,
        ssvr_hidden_size: int = 64,
        ssvr_rho_max: float = 0.30,
        ssvr_dropout: float = 0.0,
        alt_transition_hidden_size: int = 32,
        alt_transition_logit_rmax: float = 6.0,
        alt_dms_route_mode: str = "none",
        alt_dms_route_gap_threshold_min: float = 9.0,
        alt_dms_route_low_risk_scale: float = 0.0,
        alt_dms_route_high_risk_scale: float = 1.0,
        alt_target_mode: str = "relative_to_left_anchor",
        proto_use_anchor_features: bool = True,
        proto_include_exo_quality: bool = False,
        bimamba_include_exo_quality: bool = False,
        use_z_adapter: bool = False,
        z_adapter_ratio: float = 0.25,
        z_adapter_gamma_init: float = 0.0,
        proto_gap_len_ref_min: float = 180.0,
        recurrent_anchor_init: str = "none",
        obs_anchor_feedback_update: bool = False,
    ) -> None:
        super().__init__()
        self.backbone_type = str(backbone_type).lower()
        self.minimal_task_adapt_baseline = bool(minimal_task_adapt_baseline)
        self.alt_target_mode = str(alt_target_mode).lower()
        self.proto_use_anchor_features = bool(proto_use_anchor_features)
        self.proto_include_exo_quality = bool(proto_include_exo_quality)
        self.bimamba_include_exo_quality = bool(bimamba_include_exo_quality)
        self.use_z_adapter = bool(use_z_adapter)
        self.z_adapter_ratio = float(z_adapter_ratio)
        self.z_adapter_gamma_init = float(z_adapter_gamma_init)
        self.proto_gap_len_ref_min = float(max(1.0, proto_gap_len_ref_min))
        self.recurrent_anchor_init = str(recurrent_anchor_init).lower()
        self.obs_anchor_feedback_update = bool(obs_anchor_feedback_update)
        self._minimal_baseline_backbones = {
            "unilstm",
            "bilstm",
            "cnnlstm",
            "transformer",
            "mamba_proto",
            "bimamba_proto",
            "unilstm_proto",
            "bilstm_proto",
            "cnnlstm_proto",
            "transformer_proto",
            "bimamba_context",
            "bimamba_context_xyzh",
            "bimamba_context_xyzh_zlinear",
            "bimamba_context_xyzh_sharedz",
            "bimamba_context_xyaux_zlinear",
            "bimamba_context_xyaux_zlinear_zadapter",
            "bimamba_context_xyaux_zlinear_zadapter_gapgate",
            "bimamba_context_xyaux_zlinear_zadapter_gapaware_small",
            "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_coarsetrend",
            "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_vprogaux",
            "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_vprog_resaux",
            "bimamba_direct",
        }
        self.is_minimal_task_baseline = self.minimal_task_adapt_baseline and self.backbone_type in self._minimal_baseline_backbones
        print(f"[model] backbone_type={self.backbone_type}")
        if self.alt_target_mode not in {"relative_to_left_anchor", "absolute"}:
            raise RuntimeError(
                f"Unsupported alt_target_mode={self.alt_target_mode!r}; expected 'relative_to_left_anchor' or 'absolute'."
            )
        print(f"[model] alt_target_mode={self.alt_target_mode}")
        proto_without_exo_quality = self.backbone_type.endswith("_proto") and (not self.proto_include_exo_quality)
        proto_use_anchor_features = self.backbone_type.endswith("_proto") and self.proto_use_anchor_features
        if self.backbone_type in {"unilstm", "unilstm_proto"}:
            self.forward_net = UniLSTMPredictor(
                exo_dim=exo_dim,
                quality_dim=quality_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                use_anchor_features=proto_use_anchor_features,
                include_exo_quality=not proto_without_exo_quality,
                recurrent_anchor_init=self.recurrent_anchor_init,
                obs_anchor_feedback_update=self.obs_anchor_feedback_update,
            )
            # UniLSTM is a single-direction baseline and should not be wrapped
            # into a pseudo bidirectional model with a duplicated branch.
            self.backward_net = None
        elif self.backbone_type in {"bilstm", "bilstm_proto"}:
            self.forward_net = BiLSTMSequencePredictor(
                exo_dim=exo_dim,
                quality_dim=quality_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                use_anchor_features=proto_use_anchor_features,
                include_exo_quality=not proto_without_exo_quality,
            )
            self.backward_net = None
        elif self.backbone_type in {"cnnlstm", "cnnlstm_proto"}:
            self.forward_net = CNNLSTMSequencePredictor(
                exo_dim=exo_dim,
                quality_dim=quality_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                reverse=False,
                conv_channels=hidden_size,
                conv_kernel_size=3,
                use_anchor_features=proto_use_anchor_features,
                include_exo_quality=not proto_without_exo_quality,
            )
            self.backward_net = None
        elif self.backbone_type in {"transformer", "transformer_proto"}:
            # Transformer uses self-attention over the full sequence — it is
            # already bidirectional.  A single encoder produces the prediction
            # directly; no backward_net / fusion are needed.
            self.forward_net = TransformerSequencePredictor(
                exo_dim=exo_dim,
                quality_dim=quality_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                reverse=False,
                num_heads=transformer_num_heads,
                ff_multiplier=transformer_ff_multiplier,
                use_anchor_features=proto_use_anchor_features,
                include_exo_quality=not proto_without_exo_quality,
            )
            self.backward_net = None
        elif self.backbone_type == "mamba_proto":
            self.forward_net = MambaProtoSequencePredictor(
                exo_dim=exo_dim,
                quality_dim=quality_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                reverse=False,
                use_anchor_features=proto_use_anchor_features,
                include_exo_quality=not proto_without_exo_quality,
            )
            self.backward_net = None
        elif self.backbone_type == "bimamba_proto":
            self.forward_net = BiMambaProtoSequencePredictor(
                exo_dim=exo_dim,
                quality_dim=quality_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                use_anchor_features=proto_use_anchor_features,
                include_exo_quality=not proto_without_exo_quality,
            )
            self.backward_net = None
        elif self.backbone_type == "bimamba_direct":
            self.forward_net = MambaEncoderSequencePredictor(
                exo_dim=exo_dim,
                quality_dim=quality_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                reverse=False,
                use_anchor_features=True,
                include_exo_quality=not proto_without_exo_quality,
            )
            self.backward_net = MambaEncoderSequencePredictor(
                exo_dim=exo_dim,
                quality_dim=quality_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                reverse=True,
                use_anchor_features=True,
                include_exo_quality=not proto_without_exo_quality,
            )
        elif self.backbone_type in {"bimamba_context", "bimamba_context_xyzh", "bimamba_context_xyzh_zlinear", "bimamba_context_xyzh_sharedz", "bimamba_context_xyaux_zlinear", "bimamba_context_xyaux_zlinear_zadapter", "bimamba_context_xyaux_zlinear_zadapter_gapgate", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_coarsetrend", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_vprogaux", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_vprog_resaux"}:
            self.forward_net = MambaEncoderSequencePredictor(
                exo_dim=exo_dim,
                quality_dim=quality_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                reverse=False,
                use_anchor_features=True,
                include_exo_quality=self.bimamba_include_exo_quality,
            )
            self.backward_net = MambaEncoderSequencePredictor(
                exo_dim=exo_dim,
                quality_dim=quality_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                reverse=True,
                use_anchor_features=True,
                include_exo_quality=self.bimamba_include_exo_quality,
            )
        elif self.backbone_type == "bimamba":
            self.forward_net = MambaSequencePredictor(
                exo_dim=exo_dim,
                quality_dim=quality_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                reverse=False,
            )
            self.backward_net = MambaSequencePredictor(
                exo_dim=exo_dim,
                quality_dim=quality_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                reverse=True,
            )
        elif self.backbone_type == "bimamba_recurrent":
            self.forward_net = MambaRecurrentSequencePredictor(
                exo_dim=exo_dim,
                quality_dim=quality_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                reverse=False,
            )
            self.backward_net = MambaRecurrentSequencePredictor(
                exo_dim=exo_dim,
                quality_dim=quality_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                reverse=True,
            )
        elif self.backbone_type == "kalman_filter":
            raise RuntimeError(
                "backbone_type=kalman_filter has been removed from the shared neural recovery framework. "
                "Use the standalone RTS Kalman smoother baseline script instead."
            )
        elif self.backbone_type == "lstm_attention":
            raise RuntimeError(
                "backbone_type=lstm_attention has been removed from the codebase and is no longer supported."
            )
        else:
            self.forward_net = ForwardPredictor(
                exo_dim=exo_dim,
                quality_dim=quality_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
            self.backward_net = BackwardPredictor(
                exo_dim=exo_dim,
                quality_dim=quality_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
        if self.backbone_type in {
            "transformer",
            "unilstm",
            "bilstm",
            "cnnlstm",
            "transformer_proto",
            "unilstm_proto",
            "bilstm_proto",
            "cnnlstm_proto",
            "bimamba_proto",
        }:
            # Transformer, standard BiLSTM, and prototype CNN+LSTM are encoder-style backbones.
            # UniLSTM is intentionally single-directional.
            # These variants should emit the forward branch directly without an
            # extra bidirectional fusion wrapper.
            self.fusion = None
        elif self.backbone_type in {"bimamba", "bimamba_recurrent", "legacy_bidirectional"}:
            # Paper Mamba path: the only extra mechanism beyond the original
            # single-direction Mamba backbone is bidirectional prediction plus
            # gap-aware fusion on branch outputs.  Do not route through
            # hidden-state fusion or other structured post-head innovations.
            self.fusion = SimpleFusionHead(
                exo_dim=exo_dim,
                quality_dim=quality_dim,
                global_quality_dim=quality_dim,
                hidden_size=fusion_hidden_size,
                use_exo_quality=fusion_use_exo_quality,
                position_prior_enabled=bool(fusion_position_prior_enabled),
                position_prior_deviation=float(fusion_position_prior_deviation),
                weight_mode=str(fusion_weight_mode),
            )
        elif self.backbone_type in {"bimamba_direct", "bimamba_context", "bimamba_context_xyzh", "bimamba_context_xyzh_zlinear", "bimamba_context_xyzh_sharedz", "bimamba_context_xyaux_zlinear", "bimamba_context_xyaux_zlinear_zadapter", "bimamba_context_xyaux_zlinear_zadapter_gapgate", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_coarsetrend", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_vprogaux", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_vprog_resaux"}:
            self.fusion = None
        else:
            # Baseline models use deterministic position-prior fusion only
            # (no learnable parameters). This ensures fair comparison: the
            # learned fusion is part of the bilstm method's contribution.
            self.fusion = FixedPositionPriorFusion()
        self.hidden_fusion = None
        self.hidden_mu_horiz_head = None
        self.hidden_mu_alt_head = None
        self.hidden_logvar_horiz_head = None
        self.hidden_logvar_alt_head = None
        self.bimamba_direct_mu_head = None
        self.bimamba_direct_logvar_head = None
        self.bimamba_context_align = None
        self.bimamba_context_mu_head = None
        self.bimamba_context_logvar_head = None
        self.bimamba_context_mu_xy_head = None
        self.bimamba_context_mu_z_head = None
        self.bimamba_context_logvar_xy_head = None
        self.bimamba_context_logvar_z_head = None
        self.bimamba_context_mu_f_head = None
        self.bimamba_context_mu_b_head = None
        self.bimamba_context_logvar_f_head = None
        self.bimamba_context_logvar_b_head = None
        self.bimamba_context_aux_fusion = None
        self.bimamba_context_z_adapter = None
        self.bimamba_context_z_gamma = None
        self.bimamba_context_z_gate = None
        self.bimamba_context_coarse_trend_head = None
        self.bimamba_context_coarse_beta = None
        self.bimamba_context_vprog_head = None
        self.bimamba_context_vprog_res_head = None
        if self.backbone_type == "bimamba_direct":
            self.bimamba_direct_mu_head = nn.Linear(hidden_size * 2, 3)
            self.bimamba_direct_logvar_head = nn.Linear(hidden_size * 2, 3)
        elif self.backbone_type in {"bimamba_context", "bimamba_context_xyzh", "bimamba_context_xyzh_zlinear", "bimamba_context_xyzh_sharedz", "bimamba_context_xyaux_zlinear", "bimamba_context_xyaux_zlinear_zadapter", "bimamba_context_xyaux_zlinear_zadapter_gapgate", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_coarsetrend", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_vprogaux", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_vprog_resaux"}:
            self.bimamba_context_align = nn.Sequential(
                nn.Linear(hidden_size * 4 + 1, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden_size, hidden_size),
            )
            if self.backbone_type == "bimamba_context":
                self.bimamba_context_mu_head = nn.Linear(hidden_size, 3)
                self.bimamba_context_logvar_head = nn.Linear(hidden_size, 3)
            else:
                self.bimamba_context_mu_xy_head = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.SiLU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(hidden_size, 2),
                )
                self.bimamba_context_logvar_xy_head = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.SiLU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(hidden_size, 2),
                )
                if self.backbone_type in {"bimamba_context_xyzh_zlinear", "bimamba_context_xyaux_zlinear", "bimamba_context_xyaux_zlinear_zadapter", "bimamba_context_xyaux_zlinear_zadapter_gapgate", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_coarsetrend", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_vprogaux", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_vprog_resaux"}:
                    self.bimamba_context_mu_z_head = nn.Linear(hidden_size, 1)
                    self.bimamba_context_logvar_z_head = nn.Linear(hidden_size, 1)
                    if self.backbone_type in {"bimamba_context_xyaux_zlinear_zadapter", "bimamba_context_xyaux_zlinear_zadapter_gapgate", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_coarsetrend", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_vprogaux", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_vprog_resaux"}:
                        if self.backbone_type in {"bimamba_context_xyaux_zlinear_zadapter_gapaware_small", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_coarsetrend", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_vprogaux", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_vprog_resaux"}:
                            z_hidden = max(1, int(hidden_size // 8))
                            z_in_dim = hidden_size + 6
                        else:
                            z_hidden = max(1, int(round(hidden_size * self.z_adapter_ratio)))
                            z_in_dim = hidden_size
                        self.bimamba_context_z_adapter = nn.Sequential(
                            nn.Linear(z_in_dim, z_hidden),
                            nn.SiLU(),
                            nn.Linear(z_hidden, hidden_size),
                        )
                        self.bimamba_context_z_gamma = nn.Parameter(torch.tensor(float(self.z_adapter_gamma_init)))
                        if self.backbone_type == "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_coarsetrend":
                            trend_hidden = max(1, int(hidden_size // 8))
                            trend_in_dim = hidden_size + 4
                            self.bimamba_context_coarse_trend_head = nn.Sequential(
                                nn.Linear(trend_in_dim, trend_hidden),
                                nn.SiLU(),
                                nn.Linear(trend_hidden, 1),
                            )
                            self.bimamba_context_coarse_beta = nn.Parameter(torch.tensor(0.0))
                        if self.backbone_type == "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_vprogaux":
                            self.bimamba_context_vprog_head = nn.Linear(hidden_size, 1)
                        if self.backbone_type == "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_vprog_resaux":
                            self.bimamba_context_vprog_res_head = nn.Linear(hidden_size, 1)
                    if self.backbone_type == "bimamba_context_xyaux_zlinear_zadapter_gapgate":
                        self.bimamba_context_z_gate = nn.Sequential(
                            nn.Linear(4, 16),
                            nn.SiLU(),
                            nn.Linear(16, 1),
                        )
                elif self.backbone_type == "bimamba_context_xyzh_sharedz":
                    self.bimamba_context_mu_head = nn.Linear(hidden_size, 3)
                    self.bimamba_context_logvar_head = nn.Linear(hidden_size, 3)
                else:
                    self.bimamba_context_mu_z_head = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.LayerNorm(hidden_size),
                        nn.SiLU(),
                        nn.Dropout(float(dropout)),
                        nn.Linear(hidden_size, 1),
                    )
                    self.bimamba_context_logvar_z_head = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.LayerNorm(hidden_size),
                        nn.SiLU(),
                        nn.Dropout(float(dropout)),
                        nn.Linear(hidden_size, 1),
                    )
            self.bimamba_context_mu_f_head = nn.Linear(hidden_size, 3)
            self.bimamba_context_mu_b_head = nn.Linear(hidden_size, 3)
            self.bimamba_context_logvar_f_head = nn.Linear(hidden_size, 3)
            self.bimamba_context_logvar_b_head = nn.Linear(hidden_size, 3)
            self.bimamba_context_aux_fusion = SimpleFusionHead(
                exo_dim=0,
                quality_dim=0,
                global_quality_dim=quality_dim,
                hidden_size=fusion_hidden_size,
                use_exo_quality=False,
                position_prior_enabled=bool(fusion_position_prior_enabled),
                position_prior_deviation=float(fusion_position_prior_deviation),
                weight_mode="dimension",
            )
        self.alt_bias_enabled = bool(alt_bias_enabled)
        self.alt_bias_use_exo_quality = bool(alt_bias_use_exo_quality)
        self.vertical_projector_enabled = bool(vertical_projector_enabled)
        self.vertical_projector_use_vertical_exo = bool(vertical_projector_use_vertical_exo)
        self.vertical_tune_enabled = bool(vertical_tune_enabled) and self.vertical_projector_enabled
        self.vertical_tune_temperature = max(1e-3, float(vertical_tune_temperature))
        self.vertical_tune_mode = str(vertical_tune_mode).lower()
        self.vertical_tune_num_groups = 3  # [left_anchor, right_anchor, position]
        self.vertical_tune_last_weights: torch.Tensor | None = None
        self.model_variant = str(model_variant).lower()
        self.alt_dms_refiner_enabled = self.model_variant in {
            "bilstm_alt_dms_refiner_v1",
            "bilstm_alt_dms_refiner_v1_1_5",
            "bilstm_alt_dms_refiner_v2_1",
            "bilstm_alt_dms_refiner_v2",
            "bilstm_alt_dms_refiner_v3",
        }
        self.v3_anchor_hard_consistency = bool(self.model_variant == "bilstm_alt_dms_refiner_v3") and bool(
            v3_anchor_hard_consistency
        )
        self.v3_edge_residual_damp_enabled = bool(self.model_variant == "bilstm_alt_dms_refiner_v3") and bool(
            v3_edge_residual_damp_enabled
        )
        self.v3_edge_residual_damp_strength = float(v3_edge_residual_damp_strength)
        self.v3_edge_residual_damp_steps = max(1, int(v3_edge_residual_damp_steps))
        self.alt_base_residual_enabled = self.model_variant == "bilstm_alt_base_residual_v1"
        self.alt_base_residual_bound_enabled = bool(alt_base_residual_bound_enabled)
        self.alt_gate_enabled = bool(alt_gate_enabled)
        self.alt_gate_mode = str(alt_gate_mode).lower()
        self.alt_gate_fixed_value = float(alt_gate_fixed_value)
        self.alt_anchor_hard_consistency = bool(alt_anchor_hard_consistency)
        self.use_left_edge_directional_constraint = bool(use_left_edge_directional_constraint)
        self.left_edge_direction_mode = str(left_edge_direction_mode).lower()
        self.left_edge_width = max(1, int(left_edge_width))
        self.left_edge_direction_strength = float(max(0.0, min(1.0, left_edge_direction_strength)))
        self.left_edge_clip_mode = str(left_edge_clip_mode).lower()
        self.alt_main_mode = str(alt_main_mode).lower()
        self.alt_anchor_reference_mode = str(alt_anchor_reference_mode).lower()
        if self.is_minimal_task_baseline:
            self.alt_main_mode = "absolute"
            self.alt_anchor_reference_mode = "local_linear"
            self.alt_dms_refiner_enabled = False
            self.v3_anchor_hard_consistency = False
            self.v3_edge_residual_damp_enabled = False
            self.alt_base_residual_enabled = False
            self.alt_base_residual_bound_enabled = False
            self.alt_gate_enabled = False
            self.alt_anchor_hard_consistency = False
            self.use_left_edge_directional_constraint = False
            self.boundary_corrector_enabled = False
            self.vertical_projector_enabled = False
            self.vertical_tune_enabled = False
            self.alt_bias_enabled = False
            print(
                "[baseline_minimal] enabled=1 "
                f"backbone={self.backbone_type} "
                f"alt_target_mode={self.alt_target_mode} "
                "structured altitude/gating modules disabled, alt_main_mode forced to absolute"
            )
        if main_rmax_m is None:
            main_rmax_m = 152.4 if main_rmax_ft is None else float(main_rmax_ft) * 0.3048
        self.main_rmax_m = float(max(0.0, main_rmax_m))
        self.main_rmax_min_m = float(max(0.0, main_rmax_min_m))
        self.main_rmax_slope_m_per_min = float(max(0.0, main_rmax_slope_m_per_min))
        self.main_rmax_max_m = float(max(self.main_rmax_min_m, main_rmax_max_m))
        self.alt_residual_anchor_delta_gate_enabled = bool(alt_residual_anchor_delta_gate_enabled)
        self.alt_residual_anchor_delta_gate_low_m = float(max(0.0, alt_residual_anchor_delta_gate_low_m))
        self.alt_residual_anchor_delta_gate_high_m = float(max(self.alt_residual_anchor_delta_gate_low_m + 1e-6, alt_residual_anchor_delta_gate_high_m))
        self.alt_residual_anchor_delta_gate_min_scale = float(max(0.0, min(1.0, alt_residual_anchor_delta_gate_min_scale)))
        self.alt_residual_edge_taper_enabled = bool(alt_residual_edge_taper_enabled)
        self.alt_residual_edge_taper_steps = float(max(0.0, alt_residual_edge_taper_steps))
        self.alt_anchor_graph_min_step_gap_min = float(max(1.0, alt_anchor_graph_min_step_gap_min))
        self.alt_anchor_graph_step_center_ratio = float(max(0.1, min(0.9, alt_anchor_graph_step_center_ratio)))
        self.savca_min_uniform = float(max(0.0, min(0.5, savca_min_uniform)))
        self.savca_state_eps = float(max(1e-4, min(1.0, savca_state_eps)))
        self.savca_beta_enabled = bool(savca_beta_enabled)
        self.savca_beta_default_max = float(max(0.0, min(1.0, savca_beta_default_max)))
        self.savca_beta_gap_cap_enabled = bool(savca_beta_gap_cap_enabled)
        self.savca_beta_medium_gap_thr = float(max(1.0, savca_beta_medium_gap_thr))
        self.savca_beta_long_gap_thr = float(max(self.savca_beta_medium_gap_thr + 1.0, savca_beta_long_gap_thr))
        self.savca_beta_cap_short = float(max(0.0, min(1.0, savca_beta_cap_short)))
        self.savca_beta_cap_medium = float(max(0.0, min(1.0, savca_beta_cap_medium)))
        self.savca_beta_cap_long = float(max(0.0, min(1.0, savca_beta_cap_long)))
        self.savca_beta_conf_gate_enabled = bool(savca_beta_conf_gate_enabled)
        self.savca_beta_state_conf_threshold = float(savca_beta_state_conf_threshold)
        self.savca_beta_shape_conf_threshold = float(savca_beta_shape_conf_threshold)
        self.savca_beta_gate_scale_state = float(max(1e-3, savca_beta_gate_scale_state))
        self.savca_beta_gate_scale_shape = float(max(1e-3, savca_beta_gate_scale_shape))
        self.savca_beta_floor_enabled = bool(savca_beta_floor_enabled)
        self.savca_beta_floor_value = float(max(0.0, min(1.0, savca_beta_floor_value)))
        self.savca_change_score_enabled = bool(savca_change_score_enabled)
        self.savca_beta_floor_from_change_score = bool(savca_beta_floor_from_change_score)
        self.fltp_c_min = float(max(0.0, min(1.0, fltp_c_min)))
        self.fltp_c_max = float(max(self.fltp_c_min + 1e-3, min(1.0, fltp_c_max)))
        self.fltp_w_min = float(max(1e-3, fltp_w_min))
        self.fltp_w_max = float(max(self.fltp_w_min + 1e-3, fltp_w_max))
        self.fltp_gap_cap_enabled = bool(fltp_gap_cap_enabled)
        self.fltp_medium_gap_thr = float(max(1.0, fltp_medium_gap_thr))
        self.fltp_long_gap_thr = float(max(self.fltp_medium_gap_thr + 1.0, fltp_long_gap_thr))
        self.fltp_beta_cap_short = float(max(0.0, min(1.0, fltp_beta_cap_short)))
        self.fltp_beta_cap_medium = float(max(0.0, min(1.0, fltp_beta_cap_medium)))
        self.fltp_beta_cap_long = float(max(0.0, min(1.0, fltp_beta_cap_long)))
        shape_conf_type = str(savca_beta_shape_conf_type).strip().lower()
        if shape_conf_type not in {"pmax", "pmax_minus_entropy"}:
            shape_conf_type = "pmax"
        self.savca_beta_shape_conf_type = shape_conf_type
        self._runtime_savca_beta_max: float | None = None
        self.alt_transition_logit_rmax = float(max(0.0, alt_transition_logit_rmax))
        self.alt_dms_route_mode = str(alt_dms_route_mode).lower()
        self.alt_dms_route_gap_threshold_min = float(max(0.0, alt_dms_route_gap_threshold_min))
        self.alt_dms_route_low_risk_scale = float(max(0.0, min(1.0, alt_dms_route_low_risk_scale)))
        self.alt_dms_route_high_risk_scale = float(max(0.0, min(1.0, alt_dms_route_high_risk_scale)))
        if self.alt_dms_refiner_enabled and self.backbone_type != "bilstm":
            raise RuntimeError(
                "FATAL: model_variant=bilstm_alt_dms_refiner_v1 requires backbone_type=bilstm "
                f"but got backbone_type={self.backbone_type}."
            )
        if self.alt_base_residual_enabled and self.backbone_type != "bilstm":
            raise RuntimeError(
                "FATAL: model_variant=bilstm_alt_base_residual_v1 requires backbone_type=bilstm "
                f"but got backbone_type={self.backbone_type}."
            )
        if self.alt_main_mode not in {"absolute", "anchor_relative", "anchor_transition"}:
            raise RuntimeError(
                "FATAL: unsupported alt_main_mode. "
                f"Got alt_main_mode={self.alt_main_mode}, "
                "expected one of ['absolute', 'anchor_relative', 'anchor_transition']."
            )
        if self.alt_anchor_reference_mode not in {"local_linear", "anchor_graph", "savca", "fltp", "ssvr"}:
            raise RuntimeError(
                "FATAL: unsupported alt_anchor_reference_mode. "
                f"Got alt_anchor_reference_mode={self.alt_anchor_reference_mode}, "
                "expected one of ['local_linear', 'anchor_graph', 'savca', 'fltp', 'ssvr']."
            )
        if self.backbone_type in {"bimamba", "bimamba_recurrent"} and self.alt_main_mode == "absolute":
            print(
                "[warn] anchor-conditioned BiMamba is running with alt_main_mode=absolute. "
                "This can leave the main altitude head disconnected from anchor-relative "
                "height structure even when the backbone input is fixed."
            )
        if self.alt_dms_route_mode not in {"none", "gap_threshold"}:
            raise RuntimeError(
                "FATAL: unsupported alt_dms_route_mode. "
                f"Got alt_dms_route_mode={self.alt_dms_route_mode}, expected one of ['none', 'gap_threshold']."
            )
        if self.alt_bias_enabled:
            bias_in_dim = 5 + 1 + (exo_dim + quality_dim if self.alt_bias_use_exo_quality else 0)
            self.alt_bias_head = nn.Sequential(
                nn.Linear(bias_in_dim, int(alt_bias_hidden_size)),
                nn.SiLU(),
                nn.Linear(int(alt_bias_hidden_size), 1),
            )
            nn.init.normal_(self.alt_bias_head[-1].weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.alt_bias_head[-1].bias)
        else:
            self.alt_bias_head = None
        if self.alt_main_mode == "anchor_transition":
            # Learn a bounded logit correction over the linear gap position alpha.
            # This keeps the altitude baseline between two anchors while allowing
            # either smooth interpolation or a short altitude-level transition.
            trans_in_dim = 10 + int(exo_dim) + int(quality_dim)
            self.alt_transition_head = nn.Sequential(
                nn.Linear(trans_in_dim, int(alt_transition_hidden_size)),
                nn.SiLU(),
                nn.Linear(int(alt_transition_hidden_size), 1),
            )
            # Start exactly from the old linear baseline: logit_delta = 0,
            # so g_t = alpha_t before learning any transition correction.
            nn.init.zeros_(self.alt_transition_head[-1].weight)
            nn.init.zeros_(self.alt_transition_head[-1].bias)
        else:
            self.alt_transition_head = None
        if self.alt_anchor_reference_mode == "savca":
            # SAVCA predicts where the net anchor-to-anchor altitude change
            # should be allocated.  The cumulative allocation forms an
            # anchor-consistent altitude main trend; A2/A3 residuals are still
            # available for local correction.
            savca_in_dim = 10 + int(exo_dim) + int(quality_dim)
            self.savca_state_head = nn.Sequential(
                nn.Linear(savca_in_dim, int(savca_hidden_size)),
                nn.SiLU(),
                nn.Linear(int(savca_hidden_size), 1),
            )
            self.savca_alloc_head = nn.Sequential(
                nn.Linear(savca_in_dim, int(savca_hidden_size)),
                nn.SiLU(),
                nn.Linear(int(savca_hidden_size), 1),
            )
            nn.init.zeros_(self.savca_state_head[-1].weight)
            nn.init.constant_(self.savca_state_head[-1].bias, -1.0)
            nn.init.zeros_(self.savca_alloc_head[-1].weight)
            nn.init.zeros_(self.savca_alloc_head[-1].bias)
            if self.savca_beta_enabled:
                # Gap-level confidence gate deciding how far the final altitude
                # main trend can deviate from conservative A1 linear interpolation.
                beta_in_dim = 14
                self.savca_beta_head = nn.Sequential(
                    nn.Linear(beta_in_dim, int(savca_beta_hidden_size)),
                    nn.SiLU(),
                    nn.Linear(int(savca_beta_hidden_size), 1),
                )
                nn.init.zeros_(self.savca_beta_head[-1].weight)
                nn.init.constant_(self.savca_beta_head[-1].bias, float(savca_beta_init_bias))
            else:
                self.savca_beta_head = None
            if self.savca_change_score_enabled:
                change_in_dim = 15
                self.savca_change_score_head = nn.Sequential(
                    nn.Linear(change_in_dim, int(savca_change_score_hidden_size)),
                    nn.SiLU(),
                    nn.Linear(int(savca_change_score_hidden_size), 1),
                )
                nn.init.zeros_(self.savca_change_score_head[-1].weight)
                nn.init.constant_(self.savca_change_score_head[-1].bias, -1.5)
            else:
                self.savca_change_score_head = None
            print("[alt_reference] enabled=SAVCA")
        else:
            self.savca_state_head = None
            self.savca_alloc_head = None
            self.savca_beta_head = None
            self.savca_change_score_head = None
        if self.alt_anchor_reference_mode == "fltp":
            fltp_in_dim = 8
            self.fltp_head = FLTPHeightHead(
                in_dim=fltp_in_dim,
                hidden_size=int(fltp_hidden_size),
                beta_init_bias=float(fltp_beta_init_bias),
            )
            print("[alt_reference] enabled=FLTP")
        else:
            self.fltp_head = None
        if self.alt_anchor_reference_mode == "ssvr":
            ssvr_in_dim = 10
            self.ssvr_head = SSVRHeightBranch(
                in_dim=ssvr_in_dim,
                hidden_size=int(ssvr_hidden_size),
                rho_max=float(ssvr_rho_max),
                dropout=float(ssvr_dropout),
            )
            print(f"[alt_reference] enabled=SSVR rho_max={float(ssvr_rho_max)} hidden={int(ssvr_hidden_size)}")
        else:
            self.ssvr_head = None
        if self.vertical_projector_enabled:
            vp_in_dim = 5 + (int(vertical_exo_dim) if self.vertical_projector_use_vertical_exo else 0)
            self.vertical_projector = nn.Sequential(
                nn.Linear(vp_in_dim, int(vertical_projector_hidden_size)),
                nn.SiLU(),
                nn.Linear(int(vertical_projector_hidden_size), 1),
            )
            nn.init.normal_(self.vertical_projector[-1].weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.vertical_projector[-1].bias)
            if self.vertical_tune_enabled:
                # Gate input is purely structural and anchor-related scalar context.
                # No future target leakage: all terms are available at inference.
                gate_in_dim = 9  # dt_prev, dt_next, gap_len, r, obs, anchor_prev,next,delta,interp
                self.vertical_tune_gate = nn.Sequential(
                    nn.Linear(gate_in_dim, int(vertical_tune_hidden_size)),
                    nn.SiLU(),
                    nn.Linear(int(vertical_tune_hidden_size), self.vertical_tune_num_groups),
                )
            else:
                self.vertical_tune_gate = None
        else:
            self.vertical_projector = None
            self.vertical_tune_gate = None
        self.boundary_corrector_enabled = bool(boundary_corrector_enabled)
        if self.boundary_corrector_enabled:
            # Input: geometric features near gap boundary
            # alt_base_main, delta_main, dms_alt_delta, dist_to_boundary, obs_mask, anchor_diff
            bc_in_dim = 6
            self.boundary_corrector = nn.Sequential(
                nn.Linear(bc_in_dim, int(boundary_corrector_hidden_size)),
                nn.SiLU(),
                nn.Linear(int(boundary_corrector_hidden_size), 1),
            )
            nn.init.normal_(self.boundary_corrector[-1].weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.boundary_corrector[-1].bias)
        else:
            self.boundary_corrector = None
        if self.alt_dms_refiner_enabled:
            # Altitude-only features: mu_f[alt], mu_b[alt], baseline, residual = 4.
            # Horizontal dimensions are excluded so DMS focuses purely on altitude.
            self.alt_dms_refiner = AltDMSRefinerV1Head(
                backbone_feature_dim=4,
                exo_dim=exo_dim,
                quality_dim=quality_dim,
                hidden_size=int(dms_refiner_hidden_size),
                latent_dim=int(dms_refiner_latent_dim),
                refiner_num_heads=int(dms_refiner_num_heads),
                refiner_ff_multiplier=int(dms_refiner_ff_multiplier),
                dropout=float(dms_refiner_dropout),
            )
            print(f"[model_variant] enabled={self.model_variant}")
            if self.alt_gate_enabled:
                # Minimal training linkage gate:
                # gate_t = sigmoid(MLP([dms_hidden, obs, dt_prev, dt_next, gap_pos, risk, teacher, bucket]))
                # dms_hidden(4) + obs/dt_prev/dt_next/gap_len/gap_pos(5) + risk/teacher/bucket(3)
                gate_in_dim = 12
                self.alt_gate_head = nn.Sequential(
                    nn.Linear(gate_in_dim, int(alt_gate_hidden_size)),
                    nn.SiLU(),
                    nn.Linear(int(alt_gate_hidden_size), 1),
                    nn.Sigmoid(),
                )
                # Conservative prior: initialise gate ≈ 0.35 (sigmoid(-0.6))
                # so the model starts cautious and learns when to trust residuals.
                nn.init.constant_(self.alt_gate_head[-2].bias, -0.6)
            else:
                self.alt_gate_head = None
        else:
            self.alt_dms_refiner = None
            self.alt_gate_head = None
        if self.alt_base_residual_enabled:
            # Input: [mu_f(3), mu_b(3), pred_main(3), obs_mask, dt_prev, dt_next, gap_pos, alt_base, exo, quality]
            abr_in_dim = 3 + 3 + 3 + 1 + 1 + 1 + 1 + 1 + exo_dim + quality_dim
            self.alt_base_builder = AltitudeBaselineBuilder(baseline_type=str(alt_base_builder_type))
            self.alt_residual_norm = ResidualRangeNormalizer(bounds=alt_base_residual_bounds)
            self.alt_residual_head = AltitudeResidualHead(
                in_dim=int(abr_in_dim),
                hidden_size=int(alt_base_residual_hidden_size),
                dropout=float(alt_base_residual_dropout),
                use_tanh=self.alt_base_residual_bound_enabled,
            )
            print("[model_variant] enabled=bilstm_alt_base_residual_v1")
        else:
            self.alt_base_builder = None
            self.alt_residual_norm = None
            self.alt_residual_head = None
        self._shape_logged = False

    def set_runtime_savca_beta_max(self, beta_max: float | None) -> None:
        if beta_max is None:
            self._runtime_savca_beta_max = None
            return
        self._runtime_savca_beta_max = float(max(0.0, min(1.0, beta_max)))

    def _savca_gap_cap_and_bucket(self, gap_len_points: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.savca_beta_gap_cap_enabled:
            cap = torch.tensor(self.savca_beta_default_max, device=device, dtype=dtype)
            return cap, torch.tensor(1.0, device=device, dtype=dtype)
        if gap_len_points <= self.savca_beta_medium_gap_thr:
            return torch.tensor(self.savca_beta_cap_short, device=device, dtype=dtype), torch.tensor(0.0, device=device, dtype=dtype)
        if gap_len_points <= self.savca_beta_long_gap_thr:
            return torch.tensor(self.savca_beta_cap_medium, device=device, dtype=dtype), torch.tensor(1.0, device=device, dtype=dtype)
        return torch.tensor(self.savca_beta_cap_long, device=device, dtype=dtype), torch.tensor(2.0, device=device, dtype=dtype)

    def _fltp_gap_cap_and_bucket(self, gap_len_points: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.fltp_gap_cap_enabled:
            return torch.tensor(1.0, device=device, dtype=dtype), torch.tensor(1.0, device=device, dtype=dtype)
        if gap_len_points <= self.fltp_medium_gap_thr:
            return torch.tensor(self.fltp_beta_cap_short, device=device, dtype=dtype), torch.tensor(0.0, device=device, dtype=dtype)
        if gap_len_points <= self.fltp_long_gap_thr:
            return torch.tensor(self.fltp_beta_cap_medium, device=device, dtype=dtype), torch.tensor(1.0, device=device, dtype=dtype)
        return torch.tensor(self.fltp_beta_cap_long, device=device, dtype=dtype), torch.tensor(2.0, device=device, dtype=dtype)

    def _build_fltp_alt_ref_rel(
        self,
        *,
        mu_f_alt: torch.Tensor,
        mu_b_alt: torch.Tensor,
        pred_alt: torch.Tensor,
        alt_fwd_abs: torch.Tensor,
        alt_bwd_abs: torch.Tensor,
        obs_mask: torch.Tensor,
        seq_mask: torch.Tensor,
        alpha: torch.Tensor,
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
    ]:
        device = alpha.device
        dtype = alpha.dtype
        rel_out = torch.zeros_like(alpha)
        beta_out = torch.zeros_like(alpha)
        beta_cap_out = torch.zeros_like(alpha)
        beta_bucket_out = torch.zeros_like(alpha)
        c_out = torch.zeros_like(alpha)
        w_out = torch.zeros_like(alpha)
        linear_ref_abs_out = alt_fwd_abs.clone()
        sig_ref_abs_out = alt_fwd_abs.clone()
        g_linear_out = torch.zeros_like(alpha)
        g_sig_out = torch.zeros_like(alpha)
        g_final_out = torch.zeros_like(alpha)

        bsz = alpha.shape[0]
        eps = torch.tensor(1e-6, device=device, dtype=dtype)
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
                interval = torch.arange(left + 1, right, device=device)
                if interval.numel() < 1:
                    continue
                if not bool(torch.all(valid_b[left : right + 1])):
                    continue
                z_left_track = alt_fwd_abs[b, interval]
                z_right_track = alt_bwd_abs[b, interval]
                dz_track = z_right_track - z_left_track
                signed_dz = dz_track.mean()
                abs_dz = torch.abs(signed_dz)
                gap_len_points = int(interval.numel())
                gap_len_val = torch.tensor(float(max(gap_len_points, 1)), device=device, dtype=dtype)
                pred_mean = pred_alt[b, interval].mean()
                pred_std = pred_alt[b, interval].std(unbiased=False) if interval.numel() > 1 else torch.zeros((), device=device, dtype=dtype)
                mu_f_mean = mu_f_alt[b, interval].mean()
                mu_b_mean = mu_b_alt[b, interval].mean()
                feat = torch.stack(
                    [
                        abs_dz / 1000.0,
                        signed_dz / 1000.0,
                        gap_len_val / 120.0,
                        abs_dz / (gap_len_val + eps) / 100.0,
                        pred_mean / 1000.0,
                        pred_std / 100.0,
                        mu_f_mean / 1000.0,
                        mu_b_mean / 1000.0,
                    ]
                )
                c_logit, w_logit, beta_logit = self.fltp_head(feat.unsqueeze(0))
                c_logit = c_logit.squeeze(0)
                w_logit = w_logit.squeeze(0)
                beta_logit = beta_logit.squeeze(0)
                c = self.fltp_c_min + (self.fltp_c_max - self.fltp_c_min) * torch.sigmoid(c_logit)
                w = self.fltp_w_min + (self.fltp_w_max - self.fltp_w_min) * torch.sigmoid(w_logit)
                beta_cap, bucket_id = self._fltp_gap_cap_and_bucket(gap_len_points, device=device, dtype=dtype)
                beta_raw = torch.sigmoid(beta_logit)
                beta = beta_cap * beta_raw

                tau = alpha[b, interval]
                s0 = torch.sigmoid((torch.zeros((), device=device, dtype=dtype) - c) / (w + eps))
                s1 = torch.sigmoid((torch.ones((), device=device, dtype=dtype) - c) / (w + eps))
                st = torch.sigmoid((tau - c) / (w + eps))
                g_sig = torch.clamp((st - s0) / (s1 - s0 + eps), min=0.0, max=1.0)
                g_linear = tau
                g_final = (1.0 - beta) * g_linear + beta * g_sig
                ref_final_abs = z_left_track + g_final * dz_track
                ref_sig_abs = z_left_track + g_sig * dz_track

                rel_out[b, interval] = ref_final_abs - alt_fwd_abs[b, interval]
                beta_out[b, interval] = beta
                beta_cap_out[b, interval] = beta_cap
                beta_bucket_out[b, interval] = bucket_id
                c_out[b, interval] = c
                w_out[b, interval] = w
                linear_ref_abs_out[b, interval] = z_left_track + g_linear * dz_track
                sig_ref_abs_out[b, interval] = ref_sig_abs
                g_linear_out[b, interval] = g_linear
                g_sig_out[b, interval] = g_sig
                g_final_out[b, interval] = g_final

        rel_out = torch.where(obs_mask > 0.5, torch.zeros_like(rel_out), rel_out)
        beta_out = torch.where(obs_mask > 0.5, torch.zeros_like(beta_out), beta_out)
        beta_cap_out = torch.where(obs_mask > 0.5, torch.zeros_like(beta_cap_out), beta_cap_out)
        beta_bucket_out = torch.where(obs_mask > 0.5, torch.zeros_like(beta_bucket_out), beta_bucket_out)
        c_out = torch.where(obs_mask > 0.5, torch.zeros_like(c_out), c_out)
        w_out = torch.where(obs_mask > 0.5, torch.zeros_like(w_out), w_out)
        sig_ref_abs_out = torch.where(obs_mask > 0.5, alt_fwd_abs, sig_ref_abs_out)
        g_linear_out = torch.where(obs_mask > 0.5, torch.zeros_like(g_linear_out), g_linear_out)
        g_sig_out = torch.where(obs_mask > 0.5, torch.zeros_like(g_sig_out), g_sig_out)
        g_final_out = torch.where(obs_mask > 0.5, torch.zeros_like(g_final_out), g_final_out)
        return (
            rel_out,
            beta_out,
            beta_cap_out,
            beta_bucket_out,
            c_out,
            w_out,
            linear_ref_abs_out,
            sig_ref_abs_out,
            g_linear_out,
            g_sig_out,
            g_final_out,
        )

    def _anchor_feature_extract(
        self,
        v_exo: torch.Tensor | None,
        dt_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Default zero if vertical exogenous features are absent.
        z = torch.zeros_like(dt_prev)
        if v_exo is None or v_exo.ndim != 3:
            return z, z, z, z
        # Current vertical_exo layout in this project:
        # [is_anchor, dt_prev, dt_next, gap_len, gap_pos_ratio,
        #  vertical_speed, speed_delta, turn_rate,
        #  anchor_alt_prev, anchor_alt_next, anchor_alt_delta, anchor_alt_interp]
        if v_exo.shape[-1] < 12:
            # Compact layout supported by the refactored Exp4 path:
            # [..., anchor_alt_prev, anchor_alt_next, anchor_alt_delta, anchor_alt_interp]
            if v_exo.shape[-1] >= 4:
                return v_exo[..., -4], v_exo[..., -3], v_exo[..., -2], v_exo[..., -1]
            return z, z, z, z
        return v_exo[..., 8], v_exo[..., 9], v_exo[..., 10], v_exo[..., 11]

    def _build_anchor_graph_alt_ref_abs(
        self,
        alt_fwd_abs: torch.Tensor,
        alt_bwd_abs: torch.Tensor,
        obs_mask: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Flight-level anchor-graph reference in absolute altitude space.

        This is a deterministic, reversible replacement for the old local-linear
        A1 baseline. It only uses observed anchor altitudes and gap position.
        """
        ref = alt_fwd_abs + alpha * (alt_bwd_abs - alt_fwd_abs)
        bsz, tlen = ref.shape
        small_delta = 60.0
        step_delta = 120.0
        stable_std = 45.0
        context_tol = 90.0
        context_radius = 2
        transition_m_per_min = 150.0
        min_transition = 2
        max_transition = 8
        min_step_gap = int(round(self.alt_anchor_graph_min_step_gap_min))
        for b in range(bsz):
            anchors = torch.where(obs_mask[b] > 0.5)[0]
            if anchors.numel() < 2:
                continue
            anchor_alt = alt_fwd_abs[b, anchors]
            for edge_i in range(int(anchors.numel()) - 1):
                left = int(anchors[edge_i].item())
                right = int(anchors[edge_i + 1].item())
                n = right - left + 1
                if n <= 1:
                    continue
                z_left = alt_fwd_abs[b, left]
                z_right = alt_fwd_abs[b, right]
                dz = z_right - z_left
                abs_dz = torch.abs(dz)
                if abs_dz <= small_delta:
                    continue
                lo = max(0, edge_i - context_radius)
                hi = min(int(anchors.numel()), edge_i + 2 + context_radius)
                left_ctx = anchor_alt[lo : edge_i + 1]
                right_ctx = anchor_alt[edge_i + 1 : hi]
                left_med = torch.median(left_ctx) if left_ctx.numel() > 0 else z_left
                right_med = torch.median(right_ctx) if right_ctx.numel() > 0 else z_right
                left_std = torch.std(left_ctx, unbiased=False) if left_ctx.numel() > 1 else torch.zeros_like(z_left)
                right_std = torch.std(right_ctx, unbiased=False) if right_ctx.numel() > 1 else torch.zeros_like(z_right)
                left_stable = bool((left_std <= stable_std).item() and (torch.abs(z_left - left_med) <= context_tol).item())
                right_stable = bool((right_std <= stable_std).item() and (torch.abs(z_right - right_med) <= context_tol).item())
                gap_len_i = right - left - 1
                if not (abs_dz >= step_delta and gap_len_i >= min_step_gap and left_stable and right_stable):
                    continue
                duration = int(max(min_transition, min(max_transition, round(float(abs_dz.item()) / transition_m_per_min))))
                # In sparse ADS-C cruise gaps, a detected level change should
                # preserve the previous flight level for most of the interval
                # unless there is stronger evidence.  A configurable late
                # transition center is more conservative than the original
                # midpoint switch and reduces large errors from premature
                # level changes.
                center_raw = int(round((n - 1) * self.alt_anchor_graph_step_center_ratio))
                center = max(1, min(n - 2, center_raw))
                start = max(1, min(n - duration - 1, center - duration // 2))
                end = max(start + 1, min(n - 1, start + duration))
                seg = torch.full((n,), z_left, device=ref.device, dtype=ref.dtype)
                u = torch.linspace(0.0, 1.0, end - start + 1, device=ref.device, dtype=ref.dtype)
                s = u * u * (3.0 - 2.0 * u)
                seg[start : end + 1] = z_left + dz * s
                seg[end + 1 :] = z_right
                seg[0] = z_left
                seg[-1] = z_right
                ref[b, left : right + 1] = seg
        return torch.where(obs_mask > 0.5, alt_fwd_abs, ref)

    def _build_savca_alt_ref_rel(
        self,
        *,
        savca_state: torch.Tensor,
        savca_alloc_raw: torch.Tensor,
        mu_f_alt: torch.Tensor,
        mu_b_alt: torch.Tensor,
        pred_alt: torch.Tensor,
        alt_fwd_abs: torch.Tensor,
        alt_bwd_abs: torch.Tensor,
        obs_mask: torch.Tensor,
        seq_mask: torch.Tensor,
        alpha: torch.Tensor,
        savca_beta_floor_mask: torch.Tensor | None = None,
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
    ]:
        """Build SAVCA anchor-relative altitude reference.

        p is stored on interval end-points. For anchor pair [L, R], p[L+1]
        belongs to interval L->L+1 and p[R] belongs to R-1->R.  Interior
        altitude at t in (L, R) uses cumulative p[L+1:t].
        """
        ref_abs = alt_fwd_abs + alpha * (alt_bwd_abs - alt_fwd_abs)
        p_out = torch.zeros_like(alpha)
        valid_out = torch.zeros_like(alpha)
        beta_out = torch.zeros_like(alpha)
        beta_cap_out = torch.zeros_like(alpha)
        beta_bucket_out = torch.zeros_like(alpha)
        state_conf_out = torch.zeros_like(alpha)
        p_entropy_out = torch.zeros_like(alpha)
        shape_conf_out = torch.zeros_like(alpha)
        beta_raw_out = torch.zeros_like(alpha)
        state_gate_out = torch.zeros_like(alpha)
        shape_gate_out = torch.zeros_like(alpha)
        confidence_gate_out = torch.zeros_like(alpha)
        beta_min_out = torch.zeros_like(alpha)
        beta_floor_active_out = torch.zeros_like(alpha)
        change_score_out = torch.zeros_like(alpha)
        beta_floor_pred_out = torch.zeros_like(alpha)
        linear_ref_abs_out = alt_fwd_abs + alpha * (alt_bwd_abs - alt_fwd_abs)
        savca_ref_abs_out = linear_ref_abs_out.clone()
        g_linear_out = torch.zeros_like(alpha)
        g_savca_out = torch.zeros_like(alpha)
        g_final_out = torch.zeros_like(alpha)
        bsz, _ = alpha.shape
        eps = 1e-6
        runtime_beta_cap = self.savca_beta_default_max if self._runtime_savca_beta_max is None else self._runtime_savca_beta_max
        for b in range(bsz):
            anchors_b = torch.where(obs_mask[b] > 0.5)[0]
            if anchors_b.numel() < 2:
                continue
            for left_t, right_t in zip(anchors_b[:-1], anchors_b[1:]):
                left = int(left_t.item())
                right = int(right_t.item())
                if right <= left:
                    continue
                interval_all = torch.arange(left + 1, right + 1, device=alpha.device)
                valid_interval_mask = seq_mask[b, interval_all] > 0.5
                interval = interval_all[valid_interval_mask]
                n = int(interval.numel())
                if n <= 0:
                    continue
                z_left = alt_fwd_abs[b, left]
                z_right = alt_bwd_abs[b, right]
                dz = z_right - z_left
                raw = torch.nn.functional.softplus(savca_alloc_raw[b, interval])
                state = savca_state[b, interval]
                score = raw * (state + self.savca_state_eps) + eps
                p = score / (score.sum() + eps)
                if self.savca_min_uniform > 0.0:
                    uniform = torch.full_like(p, 1.0 / float(n))
                    p = (1.0 - self.savca_min_uniform) * p + self.savca_min_uniform * uniform
                    p = p / (p.sum() + eps)
                p_out[b, interval] = p
                valid_out[b, interval] = 1.0
                g_linear = alpha[b, interval]
                g_savca = torch.cumsum(p, dim=0)
                gap_len_points = max(0, right - left - 1)
                gap_beta_cap, bucket_id = self._savca_gap_cap_and_bucket(gap_len_points, device=p.device, dtype=p.dtype)
                gap_beta_cap = torch.minimum(gap_beta_cap, torch.tensor(runtime_beta_cap, device=p.device, dtype=p.dtype))
                if self.savca_beta_enabled and self.savca_beta_head is not None:
                    p_entropy = -(p * torch.log(torch.clamp(p, min=eps))).sum() / max(float(torch.log(torch.tensor(float(n), device=p.device, dtype=p.dtype)).item()), eps)
                    tau = torch.arange(1, n + 1, device=p.device, dtype=p.dtype) / float(n)
                    c_p = torch.sum(tau * p)
                    r_mean = state.mean()
                    r_max = state.max()
                    r_std = state.std(unbiased=False) if n > 1 else torch.zeros((), device=p.device, dtype=p.dtype)
                    state_conf = torch.relu(r_max - r_mean)
                    pred_alt_mean = pred_alt[b, interval].mean()
                    pred_alt_std = pred_alt[b, interval].std(unbiased=False) if n > 1 else torch.zeros((), device=p.device, dtype=p.dtype)
                    center_conf_proxy = torch.clamp(1.0 - 2.0 * torch.abs(c_p - 0.5), min=0.0, max=1.0)
                    shape_conf = p.max() if self.savca_beta_shape_conf_type == "pmax" else torch.relu(p.max() - p_entropy)
                    delta_over_gap = torch.abs(dz) / torch.tensor(float(max(1, gap_len_points)), device=p.device, dtype=p.dtype)
                    feat = torch.stack(
                        [
                            torch.abs(dz) / 1000.0,
                            torch.tensor(float(gap_len_points), device=p.device, dtype=p.dtype) / 120.0,
                            mu_f_alt[b, interval].mean(),
                            mu_b_alt[b, interval].mean(),
                            pred_alt_mean,
                            pred_alt_std,
                            r_mean,
                            r_max,
                            r_std,
                            state_conf,
                            p_entropy,
                            p.max(),
                            c_p,
                            center_conf_proxy,
                        ],
                        dim=0,
                    )
                    change_feat = torch.stack(
                        [
                            torch.abs(dz) / 1000.0,
                            torch.tensor(float(gap_len_points), device=p.device, dtype=p.dtype) / 120.0,
                            delta_over_gap / 100.0,
                            p_entropy,
                            p.max(),
                            r_mean,
                            r_max,
                            r_std,
                            state_conf,
                            pred_alt_mean,
                            pred_alt_std,
                            mu_f_alt[b, interval].mean(),
                            mu_b_alt[b, interval].mean(),
                            c_p,
                            center_conf_proxy,
                        ],
                        dim=0,
                    )
                    beta_raw = torch.sigmoid(self.savca_beta_head(feat.unsqueeze(0)).squeeze(0).squeeze(-1))
                    change_score = (
                        torch.sigmoid(self.savca_change_score_head(change_feat.unsqueeze(0)).squeeze(0).squeeze(-1))
                        if self.savca_change_score_enabled and self.savca_change_score_head is not None
                        else torch.zeros((), device=p.device, dtype=p.dtype)
                    )
                    state_gate = torch.ones((), device=p.device, dtype=p.dtype)
                    shape_gate = torch.ones((), device=p.device, dtype=p.dtype)
                    if self.savca_beta_conf_gate_enabled:
                        state_gate = torch.sigmoid(
                            self.savca_beta_gate_scale_state * (state_conf - self.savca_beta_state_conf_threshold)
                        )
                        shape_gate = torch.sigmoid(
                            self.savca_beta_gate_scale_shape * (shape_conf - self.savca_beta_shape_conf_threshold)
                        )
                    confidence_gate = state_gate * shape_gate
                else:
                    state_conf = torch.relu(state.max() - state.mean())
                    p_entropy = -(p * torch.log(torch.clamp(p, min=eps))).sum() / max(float(torch.log(torch.tensor(float(n), device=p.device, dtype=p.dtype)).item()), eps)
                    shape_conf = p.max() if self.savca_beta_shape_conf_type == "pmax" else torch.relu(p.max() - p_entropy)
                    beta_raw = torch.ones((), device=p.device, dtype=p.dtype)
                    state_gate = torch.ones((), device=p.device, dtype=p.dtype)
                    shape_gate = torch.ones((), device=p.device, dtype=p.dtype)
                    confidence_gate = torch.ones((), device=p.device, dtype=p.dtype)
                    change_score = torch.zeros((), device=p.device, dtype=p.dtype)
                beta_min = torch.zeros((), device=p.device, dtype=p.dtype)
                floor_active = torch.zeros((), device=p.device, dtype=p.dtype)
                beta_floor_pred = torch.zeros((), device=p.device, dtype=p.dtype)
                if self.savca_beta_floor_from_change_score and self.savca_change_score_enabled:
                    beta_floor_pred = torch.minimum(change_score * self.savca_beta_floor_value, gap_beta_cap)
                    beta_min = beta_floor_pred
                    floor_active = (beta_floor_pred > 0.0).to(dtype=p.dtype)
                elif self.savca_beta_floor_enabled and savca_beta_floor_mask is not None:
                    floor_mask = savca_beta_floor_mask[b, interval]
                    floor_active = (floor_mask > 0.5).any().to(dtype=p.dtype)
                    beta_min = floor_active * self.savca_beta_floor_value
                    beta_min = torch.minimum(beta_min, gap_beta_cap)
                    beta_floor_pred = beta_min
                beta = beta_min + (gap_beta_cap - beta_min) * beta_raw * confidence_gate
                beta_out[b, interval] = beta
                beta_cap_out[b, interval] = gap_beta_cap
                beta_bucket_out[b, interval] = bucket_id
                state_conf_out[b, interval] = state_conf
                p_entropy_out[b, interval] = p_entropy
                shape_conf_out[b, interval] = shape_conf
                beta_raw_out[b, interval] = beta_raw
                state_gate_out[b, interval] = state_gate
                shape_gate_out[b, interval] = shape_gate
                confidence_gate_out[b, interval] = confidence_gate
                beta_min_out[b, interval] = beta_min
                beta_floor_active_out[b, interval] = floor_active
                change_score_out[b, interval] = change_score
                beta_floor_pred_out[b, interval] = beta_floor_pred
                g_linear_out[b, interval] = g_linear
                g_savca_out[b, interval] = g_savca
                g_final = (1.0 - beta) * g_linear + beta * g_savca
                g_final_out[b, interval] = g_final
                linear_ref_abs_out[b, interval] = z_left + dz * g_linear
                savca_ref_abs_out[b, interval] = z_left + dz * g_savca
                ref_abs[b, interval] = z_left + dz * g_final
        ref_abs = torch.where(obs_mask > 0.5, alt_fwd_abs, ref_abs)
        linear_ref_abs_out = torch.where(obs_mask > 0.5, alt_fwd_abs, linear_ref_abs_out)
        savca_ref_abs_out = torch.where(obs_mask > 0.5, alt_fwd_abs, savca_ref_abs_out)
        return (
            ref_abs - alt_fwd_abs,
            ref_abs,
            p_out,
            valid_out,
            beta_out,
            beta_cap_out,
            beta_bucket_out,
            state_conf_out,
            p_entropy_out,
            shape_conf_out,
            beta_raw_out,
            state_gate_out,
            shape_gate_out,
            confidence_gate_out,
            beta_min_out,
            beta_floor_active_out,
            change_score_out,
            beta_floor_pred_out,
            linear_ref_abs_out,
            savca_ref_abs_out,
            g_linear_out,
            g_savca_out,
            g_final_out,
        )

    def _build_vertical_tune_scales(
        self,
        v_in: torch.Tensor,
        dt_prev: torch.Tensor,
        dt_next: torch.Tensor,
        gap_len: torch.Tensor,
        gap_pos_ratio: torch.Tensor,
        obs_mask: torch.Tensor,
        v_exo: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, tlen, d = v_in.shape
        device = v_in.device
        dtype = v_in.dtype
        anchor_prev, anchor_next, anchor_delta, anchor_interp = self._anchor_feature_extract(v_exo=v_exo, dt_prev=dt_prev)
        gate_in = torch.cat(
            [
                dt_prev.unsqueeze(-1),
                dt_next.unsqueeze(-1),
                gap_len.unsqueeze(-1),
                gap_pos_ratio.unsqueeze(-1),
                obs_mask.unsqueeze(-1),
                anchor_prev.unsqueeze(-1),
                anchor_next.unsqueeze(-1),
                anchor_delta.unsqueeze(-1),
                anchor_interp.unsqueeze(-1),
            ],
            dim=-1,
        )
        logits = self.vertical_tune_gate(gate_in) if self.vertical_tune_gate is not None else torch.zeros(
            (bsz, tlen, self.vertical_tune_num_groups), device=device, dtype=dtype
        )
        mode = self.vertical_tune_mode
        if mode == "left_only":
            w = torch.zeros_like(logits)
            w[..., 0] = 1.0
        elif mode == "right_only":
            w = torch.zeros_like(logits)
            w[..., 1] = 1.0
        elif mode == "position_only":
            w = torch.zeros_like(logits)
            w[..., 2] = 1.0
        else:
            w = torch.softmax(logits / self.vertical_tune_temperature, dim=-1)

        # Build per-dim modulation masks on v_in.
        left_mask = torch.zeros((d,), device=device, dtype=dtype)
        right_mask = torch.zeros((d,), device=device, dtype=dtype)
        pos_mask = torch.zeros((d,), device=device, dtype=dtype)
        # base 5 structural fields: [dt_prev, dt_next, gap_len, gap_pos_ratio, obs_mask]
        if d >= 5:
            left_mask[0] = 1.0
            right_mask[1] = 1.0
            pos_mask[2] = 1.0
            pos_mask[3] = 1.0
            pos_mask[4] = 1.0
        # If full vertical_exo fields exist, add anchor-specific indices.
        if d >= 17:
            # v_in[5 + 8/9/10/11] = anchor_alt_prev/next/delta/interp
            left_mask[13] = 1.0
            left_mask[15] = 1.0
            right_mask[14] = 1.0
            right_mask[15] = 1.0
            pos_mask[16] = 1.0
        scale = (
            1.0
            + w[..., 0:1] * left_mask.view(1, 1, -1)
            + w[..., 1:2] * right_mask.view(1, 1, -1)
            + w[..., 2:3] * pos_mask.view(1, 1, -1)
        )
        return scale, w

    def _build_alt_residual_scale(
        self,
        anchor_delta: torch.Tensor,
        alpha: torch.Tensor,
        gap_len: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Deterministic safety gate for A2/A3 altitude residuals."""
        scale = torch.ones_like(alpha)
        if self.alt_residual_anchor_delta_gate_enabled:
            denom = max(self.alt_residual_anchor_delta_gate_high_m - self.alt_residual_anchor_delta_gate_low_m, 1e-6)
            g = (torch.abs(anchor_delta) - self.alt_residual_anchor_delta_gate_low_m) / denom
            g = torch.clamp(g, min=0.0, max=1.0)
            if self.alt_residual_anchor_delta_gate_min_scale > 0.0:
                g = self.alt_residual_anchor_delta_gate_min_scale + (1.0 - self.alt_residual_anchor_delta_gate_min_scale) * g
            scale = scale * g
        if self.alt_residual_edge_taper_enabled and self.alt_residual_edge_taper_steps > 0.0:
            edge_dist = torch.minimum(alpha, 1.0 - alpha)
            edge_band = torch.clamp(self.alt_residual_edge_taper_steps / (gap_len + 1e-6), min=1e-6, max=0.5)
            taper = torch.clamp(edge_dist / edge_band, min=0.0, max=1.0)
            scale = scale * taper
        return torch.where(obs_mask > 0.5, torch.zeros_like(scale), scale)

    def forward(
        self,
        obs_pos: torch.Tensor,
        obs_mask: torch.Tensor,
        dt_prev: torch.Tensor,
        dt_next: torch.Tensor,
        exo: torch.Tensor,
        quality: torch.Tensor,
        global_quality: torch.Tensor,
        seq_mask: torch.Tensor | None = None,
        vertical_exo: torch.Tensor | None = None,
        anchor_alt: torch.Tensor | None = None,
        risk_flag: torch.Tensor | None = None,
        teacher_scale: torch.Tensor | None = None,
        risk_flag_teacher: torch.Tensor | None = None,
        segment_bucket: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,  # passthrough for future heads
        residual_rmax_m: torch.Tensor | None = None,
        residual_rmax_ft: torch.Tensor | None = None,
        gate_bias: torch.Tensor | None = None,
        left_boundary_alt: torch.Tensor | None = None,
        right_boundary_alt: torch.Tensor | None = None,
        anchor_left: torch.Tensor | None = None,
        anchor_right: torch.Tensor | None = None,
        target_pos: torch.Tensor | None = None,
        savca_beta_floor_mask: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
        return_vertical_tune_weights: bool = False,
    ) -> dict:
        def _unpack_branch(out: tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
            if isinstance(out, dict):
                return out["mu"], out["logvar"], out.get("h")
            mu, logvar = out
            return mu, logvar, None

        if seq_mask is None:
            seq_mask = torch.ones(obs_pos.shape[:2], device=obs_pos.device, dtype=obs_pos.dtype)
        is_anchor = (obs_mask > 0.5).float()
        anchor_alt_src = anchor_alt if anchor_alt is not None else obs_pos[..., 2]
        alt_fwd = torch.zeros_like(anchor_alt_src)
        alt_bwd = torch.zeros_like(anchor_alt_src)
        bsz, tlen = anchor_alt_src.shape
        for b in range(bsz):
            anchors_b = torch.where(is_anchor[b] > 0.5)[0]
            if len(anchors_b) == 0:
                continue
            first = int(anchors_b[0].item())
            last_a = int(anchors_b[-1].item())
            alt_fwd[b, : first + 1] = anchor_alt_src[b, first]
            alt_bwd[b, : first + 1] = anchor_alt_src[b, first]
            for left_t, right_t in zip(anchors_b[:-1], anchors_b[1:]):
                left = int(left_t.item())
                right = int(right_t.item())
                alt_fwd[b, left : right + 1] = anchor_alt_src[b, left]
                alt_bwd[b, left : right + 1] = anchor_alt_src[b, right]
            if last_a + 1 < tlen:
                alt_fwd[b, last_a + 1 :] = anchor_alt_src[b, last_a]
                alt_bwd[b, last_a + 1 :] = anchor_alt_src[b, last_a]

        anchor_features_f = build_anchor_condition_features(
            obs_pos=obs_pos,
            obs_mask=obs_mask,
            seq_mask=seq_mask,
            dt_prev=dt_prev,
            dt_next=dt_next,
            anchor_left=anchor_left,
            anchor_right=anchor_right,
            gap_len_ref=self.proto_gap_len_ref_min,
            reverse=False,
        )
        anchor_features_b = build_anchor_condition_features(
            obs_pos=obs_pos,
            obs_mask=obs_mask,
            seq_mask=seq_mask,
            dt_prev=dt_prev,
            dt_next=dt_next,
            anchor_left=anchor_left,
            anchor_right=anchor_right,
            gap_len_ref=self.proto_gap_len_ref_min,
            reverse=True,
        )
        out_f = self.forward_net(
            obs_pos=obs_pos,
            obs_mask=obs_mask,
            dt=dt_prev,
            exo=exo,
            quality=quality,
            target_pos=target_pos,
            teacher_forcing_ratio=teacher_forcing_ratio,
            seq_mask=seq_mask,
            anchor_features=anchor_features_f,
        )
        mu_f, logvar_f, h_f = _unpack_branch(out_f)
        if self.backward_net is not None:
            out_b = self.backward_net(
                obs_pos=obs_pos,
                obs_mask=obs_mask,
                dt=dt_next,
                exo=exo,
                quality=quality,
                target_pos=target_pos,
                teacher_forcing_ratio=teacher_forcing_ratio,
                seq_mask=seq_mask,
                anchor_features=anchor_features_b,
            )
            mu_b, logvar_b, h_b = _unpack_branch(out_b)
        else:
            # Transformer: single encoder, already bidirectional.
            mu_b = mu_f
            logvar_b = logvar_f
            h_b = None
        fusion_detail_weights = None
        pred_aux = None
        pred_xy = None
        pred_z = None
        pred_xy_full = None
        pred_z_full = None
        pred_aux_supervise_dims = None
        h_bi = None
        if self.hidden_fusion is not None and h_f is not None and h_b is not None:
            struct_feat = torch.cat(
                [
                    anchor_features_f[..., 10:13],
                    dt_prev.unsqueeze(-1),
                    dt_next.unsqueeze(-1),
                    anchor_features_f[..., 15:16],
                    anchor_features_f[..., 16:17],
                ],
                dim=-1,
            )
            h_bi, weights = self.hidden_fusion(h_f=h_f, h_b=h_b, struct_feat=struct_feat)
            pred = weights[..., :1] * mu_f + weights[..., 1:] * mu_b
            logvar = weights[..., :1] * logvar_f + weights[..., 1:] * logvar_b
            pred = pred * seq_mask.unsqueeze(-1)
            logvar = logvar * seq_mask.unsqueeze(-1)
        elif self.backbone_type == "bimamba_direct" and h_f is not None and h_b is not None:
            h_bi = torch.cat([h_f, h_b], dim=-1)
            pred = self.bimamba_direct_mu_head(h_bi)
            logvar = self.bimamba_direct_logvar_head(h_bi)
            pred = pred * seq_mask.unsqueeze(-1)
            logvar = logvar * seq_mask.unsqueeze(-1)
            bsz, tlen = pred.shape[:2]
            weights = torch.full((bsz, tlen, 2), 0.5, device=pred.device, dtype=pred.dtype)
        elif self.backbone_type in {"bimamba_context", "bimamba_context_xyzh", "bimamba_context_xyzh_zlinear", "bimamba_context_xyzh_sharedz", "bimamba_context_xyaux_zlinear", "bimamba_context_xyaux_zlinear_zadapter", "bimamba_context_xyaux_zlinear_zadapter_gapgate", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_coarsetrend", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_vprogaux", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_vprog_resaux"} and h_f is not None and h_b is not None:
            tau = anchor_features_f[..., 15:16]
            d_left_norm = anchor_features_f[..., 13:14]
            d_right_norm = anchor_features_f[..., 14:15]
            gap_len_norm = anchor_features_f[..., 16:17]
            h_align_input = torch.cat([h_f, h_b, h_f - h_b, h_f * h_b, tau], dim=-1)
            h_bi = self.bimamba_context_align(h_align_input)
            h_align = h_bi
            h_z = h_bi
            delta_h_z = None
            alpha_z = None
            delta_z_coarse = None
            q_pred = None
            q_res_pred = None
            gap_len_steps = torch.expm1(gap_len_norm * math.log1p(self.proto_gap_len_ref_min))
            if self.backbone_type == "bimamba_context":
                pred = self.bimamba_context_mu_head(h_bi)
                logvar = self.bimamba_context_logvar_head(h_bi)
            else:
                pred_xy = self.bimamba_context_mu_xy_head(h_bi)
                logvar_xy = self.bimamba_context_logvar_xy_head(h_bi)
                if self.backbone_type == "bimamba_context_xyzh_sharedz":
                    shared_pred = self.bimamba_context_mu_head(h_bi)
                    shared_logvar = self.bimamba_context_logvar_head(h_bi)
                    pred_z = shared_pred[..., 2:3]
                    logvar_z = shared_logvar[..., 2:3]
                else:
                    if self.backbone_type in {"bimamba_context_xyaux_zlinear_zadapter", "bimamba_context_xyaux_zlinear_zadapter_gapgate", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_coarsetrend", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_vprogaux", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_vprog_resaux"}:
                        if self.backbone_type in {"bimamba_context_xyaux_zlinear_zadapter_gapaware_small", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_coarsetrend", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_vprogaux", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_vprog_resaux"}:
                            delta_z = anchor_features_f[..., 12:13]
                            rate_z = delta_z / (gap_len_norm + 1e-6)
                            z_cond = torch.cat(
                                [h_bi, tau, d_left_norm, d_right_norm, gap_len_norm, delta_z, rate_z],
                                dim=-1,
                            )
                            delta_h_z = self.bimamba_context_z_adapter(z_cond)
                        else:
                            delta_h_z = self.bimamba_context_z_adapter(h_bi)
                        if self.backbone_type == "bimamba_context_xyaux_zlinear_zadapter_gapgate":
                            alpha_input = torch.cat([tau, d_left_norm, d_right_norm, gap_len_norm], dim=-1)
                            alpha_z = torch.sigmoid(self.bimamba_context_z_gate(alpha_input))
                        else:
                            alpha_z = torch.ones_like(tau)
                        h_z = h_bi + self.bimamba_context_z_gamma * alpha_z * delta_h_z
                    pred_z = self.bimamba_context_mu_z_head(h_z)
                    if self.backbone_type == "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_coarsetrend":
                        delta_z = anchor_features_f[..., 12:13]
                        trend_rate = delta_z / (gap_len_norm + 1e-6)
                        trend_input = torch.cat([h_bi, tau, gap_len_norm, delta_z, trend_rate], dim=-1)
                        delta_z_coarse = self.bimamba_context_coarse_trend_head(trend_input)
                        pred_z = pred_z + self.bimamba_context_coarse_beta * delta_z_coarse
                    if self.backbone_type == "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_vprogaux":
                        q_pred = torch.sigmoid(self.bimamba_context_vprog_head(h_z))
                    if self.backbone_type == "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_vprog_resaux":
                        q_res_pred = self.bimamba_context_vprog_res_head(h_z)
                    logvar_z = self.bimamba_context_logvar_z_head(h_z)
                pred = torch.cat([pred_xy, pred_z], dim=-1)
                logvar = torch.cat([logvar_xy, logvar_z], dim=-1)
            mu_f = self.bimamba_context_mu_f_head(h_f)
            mu_b = self.bimamba_context_mu_b_head(h_b)
            logvar_f = self.bimamba_context_logvar_f_head(h_f)
            logvar_b = self.bimamba_context_logvar_b_head(h_b)
            pred = pred * seq_mask.unsqueeze(-1)
            logvar = logvar * seq_mask.unsqueeze(-1)
            pred_xy = pred[..., :2].clone()
            pred_z = pred[..., 2:3].clone()
            fusion_out = self.bimamba_context_aux_fusion(
                mu_f=mu_f,
                mu_b=mu_b,
                dt_prev=dt_prev,
                dt_next=dt_next,
                obs_mask=obs_mask,
                exo=exo[..., :0],
                quality=quality[..., :0],
                global_quality=global_quality,
            )
            pred_aux, weights, fusion_detail_weights = fusion_out
            pred_aux = pred_aux * seq_mask.unsqueeze(-1)
            if self.backbone_type in {"bimamba_context_xyaux_zlinear", "bimamba_context_xyaux_zlinear_zadapter", "bimamba_context_xyaux_zlinear_zadapter_gapgate", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_coarsetrend", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_vprogaux", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_vprog_resaux"}:
                pred_xy = pred_aux[..., :2]
                pred = torch.cat([pred_xy, pred_z], dim=-1)
                pred_aux_supervise_dims = torch.tensor([1.0, 1.0, 0.0], device=pred.device, dtype=pred.dtype).view(1, 1, 3)
        if self.fusion is not None:
            if self.hidden_fusion is None or h_f is None or h_b is None:
                fusion_out = self.fusion(
                    mu_f=mu_f,
                    mu_b=mu_b,
                    dt_prev=dt_prev,
                    dt_next=dt_next,
                    obs_mask=obs_mask,
                    exo=exo,
                    quality=quality,
                    global_quality=global_quality,
                )
                if isinstance(fusion_out, tuple) and len(fusion_out) == 3:
                    pred, weights, fusion_detail_weights = fusion_out
                else:
                    pred, weights = fusion_out
                logvar = logvar_f
        elif self.backbone_type not in {"bimamba_context", "bimamba_context_xyzh", "bimamba_context_xyzh_zlinear", "bimamba_context_xyzh_sharedz", "bimamba_context_xyaux_zlinear", "bimamba_context_xyaux_zlinear_zadapter", "bimamba_context_xyaux_zlinear_zadapter_gapgate", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_coarsetrend", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_vprogaux", "bimamba_context_xyaux_zlinear_zadapter_gapaware_small_vprog_resaux"}:
            # Transformer: no fusion, output is the encoder prediction directly.
            pred = mu_f
            logvar = logvar_f
            bsz, tlen = pred.shape[:2]
            weights = torch.full((bsz, tlen, 2), 0.5, device=pred.device, dtype=pred.dtype)
        pred_main = pred
        # -- Shared per-timestep anchor altitude features --
        # Compute per-gap linear baseline in ENU U = alpha * (h_R - h_L),
        # which after restore_to_latlon becomes h_L + alpha*(h_R-h_L).
        gap_len = torch.clamp(dt_prev + dt_next, min=1e-6)
        alpha = torch.clamp(dt_prev / (gap_len + 1e-6), min=0.0, max=1.0)
        anchor_delta_abs = torch.zeros_like(alpha)
        alt_residual_scale = torch.ones_like(alpha)
        # ENU U = alpha * (h_R - h_L); after restore: U + h_L = linear baseline.
        alt_base_main = alpha * (alt_bwd - alt_fwd)
        alt_transition_g = alpha
        savca_alloc_p = torch.zeros_like(alpha)
        savca_state = torch.zeros_like(alpha)
        savca_alloc_valid = torch.zeros_like(alpha)
        savca_beta = torch.zeros_like(alpha)
        savca_beta_cap = torch.zeros_like(alpha)
        savca_beta_bucket_id = torch.zeros_like(alpha)
        savca_state_conf = torch.zeros_like(alpha)
        savca_p_entropy = torch.zeros_like(alpha)
        savca_shape_conf = torch.zeros_like(alpha)
        savca_beta_raw = torch.zeros_like(alpha)
        savca_state_gate = torch.zeros_like(alpha)
        savca_shape_gate = torch.zeros_like(alpha)
        savca_confidence_gate = torch.zeros_like(alpha)
        savca_beta_min = torch.zeros_like(alpha)
        savca_beta_floor_active = torch.zeros_like(alpha)
        savca_change_score = torch.zeros_like(alpha)
        savca_beta_floor_pred = torch.zeros_like(alpha)
        savca_ref_linear_abs = alt_fwd + alpha * (alt_bwd - alt_fwd)
        savca_ref_savca_abs = savca_ref_linear_abs.clone()
        savca_g_linear = torch.zeros_like(alpha)
        savca_g_savca = torch.zeros_like(alpha)
        savca_g_final = torch.zeros_like(alpha)
        fltp_beta = torch.zeros_like(alpha)
        fltp_beta_cap = torch.zeros_like(alpha)
        fltp_beta_bucket_id = torch.zeros_like(alpha)
        fltp_c = torch.zeros_like(alpha)
        fltp_w = torch.zeros_like(alpha)
        fltp_ref_linear_abs = alt_fwd + alpha * (alt_bwd - alt_fwd)
        fltp_ref_sig_abs = fltp_ref_linear_abs.clone()
        fltp_g_linear = torch.zeros_like(alpha)
        fltp_g_sig = torch.zeros_like(alpha)
        fltp_g_final = torch.zeros_like(alpha)
        ssvr_pi_L = torch.zeros_like(alpha)
        ssvr_pi_T = torch.zeros_like(alpha)
        ssvr_pi_R = torch.zeros_like(alpha)
        ssvr_rho = torch.zeros_like(alpha)
        ssvr_z_linear = torch.zeros_like(alpha)
        ssvr_z_T = torch.zeros_like(alpha)
        ssvr_state_logits = torch.zeros_like(alpha)
        if self.alt_anchor_reference_mode == "anchor_graph":
            alt_graph_ref_abs = self._build_anchor_graph_alt_ref_abs(
                alt_fwd_abs=alt_fwd,
                alt_bwd_abs=alt_bwd,
                obs_mask=obs_mask,
                alpha=alpha,
            )
            # Model altitude is anchor-relative U in the current training setup.
            alt_base_main = alt_graph_ref_abs - alt_fwd
        elif self.alt_anchor_reference_mode == "savca":
            anchor_delta = alt_bwd - alt_fwd
            savca_feat = torch.cat(
                [
                    mu_f[..., 2:3],
                    mu_b[..., 2:3],
                    pred[..., 2:3],
                    obs_mask.unsqueeze(-1),
                    dt_prev.unsqueeze(-1),
                    dt_next.unsqueeze(-1),
                    gap_len.unsqueeze(-1),
                    alpha.unsqueeze(-1),
                    anchor_delta.unsqueeze(-1),
                    alt_base_main.unsqueeze(-1),
                    exo,
                    quality,
                ],
                dim=-1,
            )
            if self.savca_state_head is None or self.savca_alloc_head is None:
                raise RuntimeError("FATAL: SAVCA reference mode requires SAVCA heads.")
            savca_state = torch.sigmoid(self.savca_state_head(savca_feat).squeeze(-1))
            savca_alloc_raw = self.savca_alloc_head(savca_feat).squeeze(-1)
            (
                alt_base_main,
                _,
                savca_alloc_p,
                savca_alloc_valid,
                savca_beta,
                savca_beta_cap,
                savca_beta_bucket_id,
                savca_state_conf,
                savca_p_entropy,
                savca_shape_conf,
                savca_beta_raw,
                savca_state_gate,
                savca_shape_gate,
                savca_confidence_gate,
                savca_beta_min,
                savca_beta_floor_active,
                savca_change_score,
                savca_beta_floor_pred,
                savca_ref_linear_abs,
                savca_ref_savca_abs,
                savca_g_linear,
                savca_g_savca,
                savca_g_final,
            ) = self._build_savca_alt_ref_rel(
                savca_state=savca_state,
                savca_alloc_raw=savca_alloc_raw,
                mu_f_alt=mu_f[..., 2],
                mu_b_alt=mu_b[..., 2],
                pred_alt=pred[..., 2],
                alt_fwd_abs=alt_fwd,
                alt_bwd_abs=alt_bwd,
                obs_mask=obs_mask,
                seq_mask=seq_mask,
                alpha=alpha,
                savca_beta_floor_mask=savca_beta_floor_mask,
            )
        elif self.alt_anchor_reference_mode == "fltp":
            if self.fltp_head is None:
                raise RuntimeError("FATAL: FLTP reference mode requires FLTP head.")
            (
                alt_base_main,
                fltp_beta,
                fltp_beta_cap,
                fltp_beta_bucket_id,
                fltp_c,
                fltp_w,
                fltp_ref_linear_abs,
                fltp_ref_sig_abs,
                fltp_g_linear,
                fltp_g_sig,
                fltp_g_final,
            ) = self._build_fltp_alt_ref_rel(
                mu_f_alt=mu_f[..., 2],
                mu_b_alt=mu_b[..., 2],
                pred_alt=pred[..., 2],
                alt_fwd_abs=alt_fwd,
                alt_bwd_abs=alt_bwd,
                obs_mask=obs_mask,
                seq_mask=seq_mask,
                alpha=alpha,
            )
        elif self.alt_anchor_reference_mode == "ssvr":
            if self.ssvr_head is None:
                raise RuntimeError("FATAL: SSVR reference mode requires SSVR head.")

            # Compute backbone altitude candidate in absolute space.
            # pred[..., 2] is the raw backbone Z head output (anchor-relative
            # when alt_target_mode == "relative_to_left_anchor").
            if self.alt_target_mode == "relative_to_left_anchor":
                z_main_abs = alt_fwd + pred[..., 2]
            else:
                z_main_abs = pred[..., 2]

            # Build per-timestep features using z_main_abs (not raw pred_z).
            ssvr_feat = SSVRFeatureBuilder.build(
                z_main_abs=z_main_abs,
                tau=alpha,
                z_L=alt_fwd,
                z_R=alt_bwd,
                dt_prev=dt_prev,
                dt_next=dt_next,
                gap_len=gap_len,
            )

            # TEMPORARY: force linear mode to verify A1 equivalence
            _force_mode = "linear"  # FIXME: remove after sanity check
            ssvr_out = self.ssvr_head(
                features=ssvr_feat,
                z_L=alt_fwd,
                z_R=alt_bwd,
                z_main_abs=z_main_abs,
                tau_gap=alpha,
                force_mode=_force_mode,
            )

            # Express SSVR result as anchor-relative for consistency with the
            # existing alt_main_mode machinery.
            alt_base_main = ssvr_out["z_hat"] - alt_fwd

            # --- one-time debug print (removed after first batch) ---
            if not hasattr(self, "_ssvr_debug_done"):
                self._ssvr_debug_done = True
                gap_mask = (obs_mask <= 0.5) & (seq_mask > 0.5)
                if gap_mask.any():
                    print("[SSVR DEBUG]")
                    print(f"  alt_fwd range:      [{alt_fwd[gap_mask].min():.2f}, {alt_fwd[gap_mask].max():.2f}]")
                    print(f"  alt_bwd range:      [{alt_bwd[gap_mask].min():.2f}, {alt_bwd[gap_mask].max():.2f}]")
                    print(f"  pred_z(raw) range:  [{pred[..., 2][gap_mask].min():.2f}, {pred[..., 2][gap_mask].max():.2f}]")
                    print(f"  z_hat range:        [{ssvr_out['z_hat'][gap_mask].min():.2f}, {ssvr_out['z_hat'][gap_mask].max():.2f}]")
                    print(f"  z_linear range:     [{ssvr_out['z_linear'][gap_mask].min():.2f}, {ssvr_out['z_linear'][gap_mask].max():.2f}]")
                    print(f"  alt_base_main range:[{alt_base_main[gap_mask].min():.2f}, {alt_base_main[gap_mask].max():.2f}]")
                    print(f"  target_z range:     [{target_pos[..., 2][gap_mask].min():.2f}, {target_pos[..., 2][gap_mask].max():.2f}]")
                    print(f"  pi_T mean: {ssvr_out['pi_T'][gap_mask].mean():.4f}")
                    print(f"  rho mean:  {ssvr_out['rho'][gap_mask].mean():.4f}")
                    print(f"  |pred-a1| mean: {(alt_base_main[gap_mask] - (alpha*(alt_bwd-alt_fwd))[gap_mask]).abs().mean():.4f}")
            # --- end debug ---

            # Store SSVR outputs for diagnostics.
            ssvr_pi_L = ssvr_out["pi_L"]
            ssvr_pi_T = ssvr_out["pi_T"]
            ssvr_pi_R = ssvr_out["pi_R"]
            ssvr_rho = ssvr_out["rho"]
            ssvr_z_linear = ssvr_out["z_linear"]
            ssvr_z_T = ssvr_out["z_T"]
            ssvr_state_logits = ssvr_out["state_logits"]

            # SSVR directly sets the altitude channel regardless of alt_main_mode.
            # alt_base_main is anchor-relative (matches alt_target_mode format).
            pred = pred.clone()
            pred[..., 2] = alt_base_main
            pred_main = pred.clone()

        if self.alt_main_mode in {"anchor_relative", "anchor_transition"}:
            if self.alt_main_mode == "anchor_transition" and self.alt_transition_head is not None:
                anchor_delta = alt_bwd - alt_fwd
                trans_feat = torch.cat(
                    [
                        mu_f[..., 2:3],
                        mu_b[..., 2:3],
                        pred[..., 2:3],
                        obs_mask.unsqueeze(-1),
                        dt_prev.unsqueeze(-1),
                        dt_next.unsqueeze(-1),
                        gap_len.unsqueeze(-1),
                        alpha.unsqueeze(-1),
                        anchor_delta.unsqueeze(-1),
                        alt_base_main.unsqueeze(-1),
                        exo,
                        quality,
                    ],
                    dim=-1,
                )
                alpha_safe = torch.clamp(alpha, min=1e-4, max=1.0 - 1e-4)
                alpha_logit = torch.logit(alpha_safe)
                logit_delta = torch.tanh(self.alt_transition_head(trans_feat).squeeze(-1)) * self.alt_transition_logit_rmax
                alt_transition_g = torch.sigmoid(alpha_logit + logit_delta)
                # Observed anchors are hard-replaced later; keeping g at alpha on
                # anchors makes diagnostics and auxiliary terms easier to read.
                alt_transition_g = torch.where(is_anchor > 0.5, alpha, alt_transition_g)
                alt_base_main = alt_transition_g * anchor_delta

            if float(self.main_rmax_m) > 0:
                gap_len_rmax = torch.clamp(gap_len, min=1.0, max=60.0)
                dynamic_rmax = torch.clamp(
                    self.main_rmax_min_m + self.main_rmax_slope_m_per_min * gap_len_rmax,
                    min=self.main_rmax_min_m,
                    max=self.main_rmax_max_m,
                )
                delta_main = torch.tanh(pred[..., 2]) * dynamic_rmax
            else:
                delta_main = torch.zeros_like(pred[..., 2])
            anchor_delta_abs = torch.abs(alt_bwd - alt_fwd)
            alt_residual_scale = self._build_alt_residual_scale(
                anchor_delta=alt_bwd - alt_fwd,
                alpha=alpha,
                gap_len=gap_len,
                obs_mask=obs_mask,
            )
            delta_main = delta_main * alt_residual_scale

            main_alt = alt_base_main + delta_main
            if self.alt_anchor_reference_mode == "ssvr":
                # SSVR directly produces the final height through its three-state
                # mixture.  The backbone altitude already participates via rho
                # inside z_T, so we skip the per-timestep delta_main for altitude.
                main_alt = alt_base_main
            pred_main = pred.clone()
            pred_main[..., 2] = main_alt
            pred = pred.clone()
            pred[..., 2] = main_alt
        alt_bias = torch.zeros_like(pred[..., 2])
        vertical_delta = torch.zeros_like(pred[..., 2])
        dms_alt_delta = torch.zeros_like(pred[..., 2])
        dms_alt_delta_candidate = torch.zeros_like(pred[..., 2])
        dms_route_scale = torch.ones_like(pred[..., 2])
        alt_base = torch.zeros_like(pred[..., 2])
        residual_bound = torch.ones_like(pred[..., 2])
        delta_alt_pred_norm = torch.zeros_like(pred[..., 2])
        dms_attn_weights = None
        alt_gate = None
        left_edge_wrong_direction_mask = None
        left_edge_target_direction = None
        vertical_tune_weights = None
        if self.alt_base_residual_enabled and self.alt_base_builder is not None:
            obs_alt = obs_pos[..., 2]
            alt_base = self.alt_base_builder(
                obs_alt=obs_alt,
                obs_mask=obs_mask,
                dt_prev=dt_prev,
                dt_next=dt_next,
            )
            gap_len = dt_prev + dt_next
            residual_bound = (
                self.alt_residual_norm(gap_len)
                if (self.alt_base_residual_bound_enabled and self.alt_residual_norm is not None)
                else torch.ones_like(gap_len)
            )
            gap_pos = dt_prev / (gap_len + 1e-6)
            abr_feat = torch.cat(
                [
                    mu_f,
                    mu_b,
                    pred_main,
                    obs_mask.unsqueeze(-1),
                    dt_prev.unsqueeze(-1),
                    dt_next.unsqueeze(-1),
                    gap_pos.unsqueeze(-1),
                    alt_base.unsqueeze(-1),
                    exo,
                    quality,
                ],
                dim=-1,
            )
            delta_alt_pred_norm = self.alt_residual_head(abr_feat)
            pred = pred.clone()
            pred[..., 2] = alt_base + delta_alt_pred_norm * residual_bound
        if self.alt_dms_refiner_enabled and self.alt_dms_refiner is not None:
            anchor_prev = alt_fwd
            anchor_next = alt_bwd
            anchor_delta = alt_bwd - alt_fwd
            anchor_interp = alt_base_main
            dms_hidden = torch.cat([
                mu_f[..., 2:3],              # forward LSTM altitude
                mu_b[..., 2:3],              # backward LSTM altitude
                alt_base_main.unsqueeze(-1), # geometric linear baseline
                delta_main.unsqueeze(-1),    # Step 1 bounded residual
            ], dim=-1)
            dms_alt_delta, dms_aux = self.alt_dms_refiner(
                hidden_seq=dms_hidden,
                obs_mask=obs_mask,
                dt_prev=dt_prev,
                dt_next=dt_next,
                exo=exo,
                quality=quality,
                anchor_prev=anchor_prev,
                anchor_next=anchor_next,
                anchor_delta=anchor_delta,
                anchor_interp=anchor_interp,
                valid_mask=None,
            )
            dms_alt_delta_candidate = dms_alt_delta.clone()
            # Left-edge directional residual constraint (optional):
            # enforce candidate residual direction near left gap edge before bounded residual.
            if self.use_left_edge_directional_constraint:
                bsz, tlen, _ = dms_hidden.shape
                dtype = dms_alt_delta.dtype
                device = dms_alt_delta.device
                gap_len = torch.clamp(dt_prev + dt_next, min=1.0)
                # left-edge zone: missing positions close to left boundary.
                left_zone = (obs_mask <= 0.5) & (dt_prev <= float(self.left_edge_width) + 1e-6)
                if self.left_edge_direction_mode == "anchor_based":
                    ref_alt = anchor_prev
                else:
                    # fallback for currently unsupported mode: keep anchor-based semantics.
                    ref_alt = anchor_prev
                baseline_alt = pred_main[..., 2]
                target_direction = torch.sign(baseline_alt - ref_alt)
                cand = dms_alt_delta
                cand_sign = torch.sign(cand)
                wrong = left_zone & (torch.abs(target_direction) > 0.0) & (cand_sign != target_direction)
                # project candidate to target direction
                projected = torch.where(
                    target_direction > 0.0,
                    torch.clamp(cand, min=0.0),
                    torch.where(target_direction < 0.0, torch.clamp(cand, max=0.0), torch.zeros_like(cand)),
                )
                if self.left_edge_clip_mode == "hard":
                    if self.left_edge_direction_strength >= 1.0:
                        cand = torch.where(wrong, projected, cand)
                    else:
                        a = self.left_edge_direction_strength
                        cand = torch.where(wrong, (1.0 - a) * cand + a * projected, cand)
                else:
                    # soft shrink on wrong-direction residual.
                    a = self.left_edge_direction_strength
                    cand = torch.where(wrong, (1.0 - a) * cand + a * projected, cand)
                dms_alt_delta = cand
                left_edge_wrong_direction_mask = wrong.to(dtype=dtype)
                left_edge_target_direction = target_direction
            if self.alt_gate_enabled and self.alt_gate_head is not None:
                bsz, tlen, _ = dms_hidden.shape
                dtype = dms_hidden.dtype
                device = dms_hidden.device
                gap_len = dt_prev + dt_next
                gap_pos = dt_prev / (gap_len + 1e-6)
                # Broadcast sample-level meta to [B,T] with safe defaults.
                if risk_flag is None:
                    risk_bt = torch.zeros((bsz, tlen), dtype=dtype, device=device)
                else:
                    risk_bt = risk_flag.to(device=device, dtype=dtype).view(bsz, 1).expand(bsz, tlen)
                if teacher_scale is None:
                    teacher_bt = torch.ones((bsz, tlen), dtype=dtype, device=device)
                else:
                    teacher_bt = teacher_scale.to(device=device, dtype=dtype).view(bsz, 1).expand(bsz, tlen)
                if segment_bucket is None:
                    bucket_bt = torch.zeros((bsz, tlen), dtype=dtype, device=device)
                else:
                    bucket_bt = segment_bucket.to(device=device, dtype=dtype).view(bsz, 1).expand(bsz, tlen) / 2.0
                if gate_bias is None:
                    gate_bias_bt = torch.zeros((bsz, tlen), dtype=dtype, device=device)
                else:
                    gate_bias_bt = gate_bias.to(device=device, dtype=dtype).view(bsz, 1).expand(bsz, tlen)
                gate_in = torch.cat(
                    [
                        dms_hidden,
                        obs_mask.unsqueeze(-1),
                        dt_prev.unsqueeze(-1),
                        dt_next.unsqueeze(-1),
                        gap_len.unsqueeze(-1),
                        gap_pos.unsqueeze(-1),
                        risk_bt.unsqueeze(-1),
                        teacher_bt.unsqueeze(-1),
                        bucket_bt.unsqueeze(-1),
                    ],
                    dim=-1,
                )
                if self.alt_gate_mode == "teacher_fixed":
                    # Legacy: gate is a fixed per-sample constant from risk rules.
                    if teacher_scale is None:
                        alt_gate = torch.full_like(teacher_bt, float(self.alt_gate_fixed_value))
                    else:
                        alt_gate = torch.clamp(teacher_bt, 0.0, 1.0)
                else:
                    # Learned gate: model decides per-timestep how much residual to admit.
                    # teacher_scale still guides training via supervision loss, but the
                    # gate value itself is produced by a lightweight MLP over gap features.
                    alt_gate = self.alt_gate_head(gate_in).squeeze(-1)
                    if gate_bias is not None:
                        alt_gate = alt_gate + gate_bias_bt
                    alt_gate = torch.sigmoid(alt_gate)  # [B, T] in (0, 1)
                dms_alt_delta = alt_gate * dms_alt_delta
            if residual_rmax_m is None and residual_rmax_ft is not None:
                residual_rmax_m = residual_rmax_ft * 0.3048
            if residual_rmax_m is not None:
                # Segment-aware bounded residual to avoid extreme altitude jumps.
                bsz, tlen, _ = dms_hidden.shape
                rmax_bt = residual_rmax_m.to(device=dms_alt_delta.device, dtype=dms_alt_delta.dtype).view(bsz, 1).expand(bsz, tlen)
                rmax_bt = torch.clamp(rmax_bt, min=1.0)
                dms_alt_delta = torch.tanh(dms_alt_delta / (rmax_bt + 1e-6)) * rmax_bt
            if self.alt_dms_route_mode == "gap_threshold":
                gap_mask = obs_mask <= 0.5
                high_risk_gap = (dt_prev + dt_next) >= self.alt_dms_route_gap_threshold_min
                high = torch.full_like(dms_alt_delta, self.alt_dms_route_high_risk_scale)
                low = torch.full_like(dms_alt_delta, self.alt_dms_route_low_risk_scale)
                dms_route_scale = torch.where(gap_mask & high_risk_gap, high, low)
                dms_alt_delta = dms_alt_delta * dms_route_scale
            dms_alt_delta = dms_alt_delta * alt_residual_scale
            if self.model_variant == "bilstm_alt_dms_refiner_v3" and self.v3_edge_residual_damp_enabled:
                # V3: damp residual near gap boundaries to reduce first/last-step spikes.
                gap_len = dt_prev + dt_next
                gap_pos = dt_prev / (gap_len + 1e-6)
                edge_dist = torch.minimum(gap_pos, 1.0 - gap_pos)
                edge_steps = float(self.v3_edge_residual_damp_steps)
                # Approximate edge band in relative coordinate by steps / segment length.
                edge_band = torch.clamp(edge_steps / (gap_len + 1e-6), min=0.0, max=0.5)
                near_edge = (edge_dist <= edge_band).to(dms_alt_delta.dtype)
                damp = 1.0 - near_edge * float(self.v3_edge_residual_damp_strength)
                damp = torch.clamp(damp, min=0.0, max=1.0)
                dms_alt_delta = dms_alt_delta * damp
            pred = pred.clone()
            pred[..., 2] = pred[..., 2] + dms_alt_delta
            dms_attn_weights = dms_aux.get("attn_weights")
        if self.boundary_corrector_enabled and self.boundary_corrector is not None:
            # Build boundary mask: first 2 and last 2 steps of each contiguous gap.
            gap_mask = (obs_mask <= 0.5).float()
            # Shift left/right to find gap boundaries
            gap_enter = gap_mask - torch.cat([torch.zeros_like(gap_mask[:, :1]), gap_mask[:, :-1]], dim=1)
            gap_exit = torch.cat([gap_mask[:, 1:], torch.zeros_like(gap_mask[:, :1])], dim=1) - gap_mask
            # First step of a gap: gap_enter == 1
            first_step = (gap_enter > 0.5).float()
            second_step = torch.cat([first_step[:, 1:], torch.zeros_like(first_step[:, :1])], dim=1)
            # Last step of a gap: gap_exit == 1
            last_step = (gap_exit > 0.5).float()
            second_last = torch.cat([torch.zeros_like(last_step[:, :1]), last_step[:, :-1]], dim=1)
            boundary_mask = torch.clamp(first_step + second_step + second_last + last_step, 0.0, 1.0)

            bc_in = torch.cat([
                alt_base_main.unsqueeze(-1),
                delta_main.unsqueeze(-1),
                dms_alt_delta.unsqueeze(-1),
                torch.minimum(dt_prev, dt_next).unsqueeze(-1),  # distance to nearest anchor
                obs_mask.unsqueeze(-1),
                (alt_bwd - alt_fwd).unsqueeze(-1),  # anchor altitude difference
            ], dim=-1)
            bc_delta = self.boundary_corrector(bc_in).squeeze(-1)
            pred = pred.clone()
            pred[..., 2] = pred[..., 2] + bc_delta * boundary_mask
        if self.vertical_projector_enabled and self.vertical_projector is not None:
            gap_len = dt_prev + dt_next
            gap_pos_ratio = dt_prev / (gap_len + 1e-6)
            v_exo = vertical_exo if vertical_exo is not None else exo
            if self.vertical_tune_enabled and self.vertical_projector_use_vertical_exo:
                if v_exo is None or v_exo.ndim != 3 or v_exo.shape[-1] < 12:
                    got_shape = None if v_exo is None else tuple(v_exo.shape)
                    raise RuntimeError(
                        "FATAL: vertical_tune_enabled requires vertical_exo with shape [B,T,C>=12]. "
                        f"Got shape={got_shape}. Check vertical_exo_cols/feature pipeline."
                    )
            v_chunks = [
                dt_prev.unsqueeze(-1),
                dt_next.unsqueeze(-1),
                gap_len.unsqueeze(-1),
                gap_pos_ratio.unsqueeze(-1),
                obs_mask.unsqueeze(-1),
            ]
            if self.vertical_projector_use_vertical_exo:
                v_chunks.append(v_exo)
            v_in = torch.cat(v_chunks, dim=-1)
            if self.vertical_tune_enabled:
                v_scale, vertical_tune_weights = self._build_vertical_tune_scales(
                    v_in=v_in,
                    dt_prev=dt_prev,
                    dt_next=dt_next,
                    gap_len=gap_len,
                    gap_pos_ratio=gap_pos_ratio,
                    obs_mask=obs_mask,
                    v_exo=v_exo,
                )
                v_in = v_in * v_scale
            vertical_delta = self.vertical_projector(v_in).squeeze(-1)
            pred = pred.clone()
            pred[..., 2] = pred[..., 2] + vertical_delta
        if self.alt_bias_enabled and self.alt_bias_head is not None:
            gap_len = dt_prev + dt_next
            gap_pos_ratio = dt_prev / (gap_len + 1e-6)
            if anchor_alt is None:
                anchor_alt = torch.zeros_like(dt_prev)
            anchor_alt_scaled = anchor_alt / 10000.0
            chunks = [
                dt_prev.unsqueeze(-1),
                dt_next.unsqueeze(-1),
                gap_len.unsqueeze(-1),
                gap_pos_ratio.unsqueeze(-1),
                obs_mask.unsqueeze(-1),
                anchor_alt_scaled.unsqueeze(-1),
            ]
            if self.alt_bias_use_exo_quality:
                chunks.extend([exo, quality])
            bias_in = torch.cat(chunks, dim=-1)
            alt_bias = self.alt_bias_head(bias_in).squeeze(-1)
            pred = pred.clone()
            pred[..., 2] = pred[..., 2] + alt_bias
        # Observation-conditioned recovery: anchors are known inputs, not targets to be re-predicted.
        # Apply to every model variant and every coordinate dimension before loss/evaluation.
        anchor_bt = (obs_mask > 0.5).unsqueeze(-1)
        pred = torch.where(anchor_bt, obs_pos, pred)
        if pred_aux is not None:
            pred_aux = torch.where(anchor_bt, obs_pos, pred_aux)
        if pred_xy is not None:
            pred_xy = torch.where(anchor_bt, obs_pos[..., :2], pred_xy)
        if pred_z is not None:
            pred_z = torch.where(anchor_bt, obs_pos[..., 2:3], pred_z)
        if pred_xy is not None:
            pred_xy_full = obs_pos.clone()
            pred_xy_full[..., :2] = pred_xy
            pred_xy_full[..., 2:3] = pred[..., 2:3]
        if pred_z is not None:
            pred_z_full = obs_pos.clone()
            pred_z_full[..., :2] = pred[..., :2]
            pred_z_full[..., 2:3] = pred_z
        if not self._shape_logged:
            print(
                "[shape] "
                f"mu_f={tuple(mu_f.shape)} "
                f"mu_b={tuple(mu_b.shape)} "
                f"logvar_f={tuple(logvar_f.shape)} "
                f"logvar_b={tuple(logvar_b.shape)} "
                f"pred={tuple(pred.shape)} "
                f"fusion_weights={tuple(weights.shape)}"
            )
            self._shape_logged = True
        return {
            "pred_pos": pred,
            "pred": pred,
            "mu_f": mu_f,
            "mu_b": mu_b,
            "logvar": logvar,
            "logvar_f": logvar_f,
            "logvar_b": logvar_b,
            "h_f": h_f,
            "h_b": h_b,
            "h": h_bi,
            "h_bi": h_bi,
            "h_forward": h_f,
            "h_backward": h_b,
            "pred_pos_main": pred_main,
            "pred_main": pred_main,
            "pred_pos_aux": pred_aux,
            "pred_aux": pred_aux,
            "pred_pos_aux_supervise_dims": pred_aux_supervise_dims,
            "pred_xy": pred_xy,
            "pred_z": pred_z,
            "pred_xy_full": pred_xy_full,
            "pred_z_full": pred_z_full,
            "alt_bias": alt_bias,
            "vertical_delta": vertical_delta,
            "dms_alt_delta": dms_alt_delta,
            "dms_alt_delta_candidate": dms_alt_delta_candidate,
            "dms_route_scale": dms_route_scale,
            "alt_transition_g": alt_transition_g,
            "savca_alloc_p": savca_alloc_p,
            "savca_state": savca_state,
            "savca_alloc_valid": savca_alloc_valid,
            "savca_beta": savca_beta,
            "savca_beta_cap": savca_beta_cap,
            "savca_beta_bucket_id": savca_beta_bucket_id,
            "savca_state_conf": savca_state_conf,
            "savca_p_entropy": savca_p_entropy,
            "savca_shape_conf": savca_shape_conf,
            "savca_beta_raw": savca_beta_raw,
            "savca_state_gate": savca_state_gate,
            "savca_shape_gate": savca_shape_gate,
            "savca_confidence_gate": savca_confidence_gate,
            "savca_beta_min": savca_beta_min,
            "savca_beta_floor_active": savca_beta_floor_active,
            "savca_change_score": savca_change_score,
            "savca_beta_floor_pred": savca_beta_floor_pred,
            "savca_ref_linear_abs": savca_ref_linear_abs,
            "savca_ref_savca_abs": savca_ref_savca_abs,
            "savca_ref_final_abs": alt_fwd + alt_base_main,
            "savca_g_linear": savca_g_linear,
            "savca_g_savca": savca_g_savca,
            "savca_g_final": savca_g_final,
            "fltp_beta": fltp_beta,
            "fltp_beta_cap": fltp_beta_cap,
            "fltp_beta_bucket_id": fltp_beta_bucket_id,
            "fltp_c": fltp_c,
            "fltp_w": fltp_w,
            "fltp_ref_linear_abs": fltp_ref_linear_abs,
            "fltp_ref_sig_abs": fltp_ref_sig_abs,
            "fltp_ref_final_abs": alt_fwd + alt_base_main,
            "fltp_g_linear": fltp_g_linear,
            "fltp_g_sig": fltp_g_sig,
            "fltp_g_final": fltp_g_final,
            "ssvr_pi_L": ssvr_pi_L,
            "ssvr_pi_T": ssvr_pi_T,
            "ssvr_pi_R": ssvr_pi_R,
            "ssvr_rho": ssvr_rho,
            "ssvr_z_linear": ssvr_z_linear,
            "ssvr_z_T": ssvr_z_T,
            "ssvr_state_logits": ssvr_state_logits,
            "ssvr_z_hat": alt_fwd + alt_base_main if self.alt_anchor_reference_mode == "ssvr" else torch.zeros_like(alpha),
            "alt_base_main": alt_base_main,
            "alt_residual_scale": alt_residual_scale,
            "anchor_delta_abs": anchor_delta_abs,
            "left_edge_wrong_direction_mask": left_edge_wrong_direction_mask,
            "left_edge_target_direction": left_edge_target_direction,
            "dms_attn_weights": dms_attn_weights if return_vertical_tune_weights else None,
            "alt_gate": alt_gate,
            "alt_base": alt_base,
            "residual_bound": residual_bound,
            "delta_alt_pred_norm": delta_alt_pred_norm,
            "vertical_tune_weights": vertical_tune_weights if return_vertical_tune_weights else None,
            "fusion_weights": weights,
            "fusion_weights_detail": fusion_detail_weights,
            "pred_z": pred_z,
            "h_align": h_bi if h_bi is not None else None,
            "h_z": h_z if 'h_z' in locals() else None,
            "delta_h_z": delta_h_z if 'delta_h_z' in locals() else None,
            "delta_z_coarse": delta_z_coarse if 'delta_z_coarse' in locals() else None,
            "q_pred": q_pred if 'q_pred' in locals() else None,
            "q_res_pred": q_res_pred if 'q_res_pred' in locals() else None,
            "alpha_z": alpha_z if 'alpha_z' in locals() else None,
            "alpha_tau": tau if 'tau' in locals() else None,
            "alpha_gap_len_steps": gap_len_steps if 'gap_len_steps' in locals() else None,
            "gamma_z": self.bimamba_context_z_gamma if self.bimamba_context_z_gamma is not None else None,
            "beta_z_coarse": self.bimamba_context_coarse_beta if self.bimamba_context_coarse_beta is not None else None,
            "seq_mask": seq_mask,
            "alt_fwd": alt_fwd,
            "alt_bwd": alt_bwd,
        }
