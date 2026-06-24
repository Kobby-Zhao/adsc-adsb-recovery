from __future__ import annotations

import torch
from torch import nn

from src.models.bidirectional_predictor import ForwardPredictor
from src.models.gap_aware_lstm import GapAwareLSTMCell


def _ensure_seq_mask(seq_mask: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor:
    if seq_mask is None:
        return torch.ones(ref.shape[:2], device=ref.device, dtype=ref.dtype)
    return seq_mask.to(device=ref.device, dtype=ref.dtype)


def _masked_seq(x: torch.Tensor, seq_mask: torch.Tensor | None) -> torch.Tensor:
    mask = _ensure_seq_mask(seq_mask, x).unsqueeze(-1)
    return x * mask


def reverse_by_seq_len(x: torch.Tensor, seq_mask: torch.Tensor | None) -> torch.Tensor:
    mask = _ensure_seq_mask(seq_mask, x)
    out = x.clone()
    for b in range(x.shape[0]):
        valid_len = int((mask[b] > 0.5).sum().item())
        if valid_len > 0:
            out[b, :valid_len] = x[b, :valid_len].flip(0)
    return out


def restore_by_seq_len(x_rev: torch.Tensor, seq_mask: torch.Tensor | None) -> torch.Tensor:
    return reverse_by_seq_len(x_rev, seq_mask)


def _expand_dt(dt: torch.Tensor) -> torch.Tensor:
    return dt if dt.ndim == 3 else dt.unsqueeze(-1)


def build_anchor_condition_features(
    obs_pos: torch.Tensor,
    obs_mask: torch.Tensor,
    seq_mask: torch.Tensor | None,
    dt_prev: torch.Tensor,
    dt_next: torch.Tensor,
    anchor_left: torch.Tensor | None = None,
    anchor_right: torch.Tensor | None = None,
    gap_len_ref: float = 180.0,
    reverse: bool = False,
) -> torch.Tensor:
    mask = _ensure_seq_mask(seq_mask, obs_pos)
    if anchor_left is None or anchor_right is None:
        raise RuntimeError("build_anchor_condition_features requires explicit anchor_left and anchor_right tensors.")

    gap_len = torch.clamp(dt_prev + dt_next, min=1e-6)
    tau_fwd = torch.clamp(dt_prev / gap_len, min=0.0, max=1.0)
    d_left_norm_fwd = tau_fwd
    d_right_norm_fwd = 1.0 - tau_fwd
    gap_len_ref = float(max(1.0, gap_len_ref))
    gap_len_norm = torch.clamp(
        torch.log1p(gap_len) / torch.log1p(torch.tensor(gap_len_ref, device=gap_len.device, dtype=gap_len.dtype)),
        min=0.0,
        max=1.0,
    ).unsqueeze(-1)

    if reverse:
        y_left = anchor_right
        y_right = anchor_left
        delta_y = y_right - y_left
        d_left_norm = d_right_norm_fwd.unsqueeze(-1)
        d_right_norm = d_left_norm_fwd.unsqueeze(-1)
        tau = d_right_norm_fwd.unsqueeze(-1)
    else:
        y_left = anchor_left
        y_right = anchor_right
        delta_y = y_right - y_left
        d_left_norm = d_left_norm_fwd.unsqueeze(-1)
        d_right_norm = d_right_norm_fwd.unsqueeze(-1)
        tau = tau_fwd.unsqueeze(-1)

    x = torch.cat(
        [
            obs_pos,
            obs_mask.unsqueeze(-1),
            y_left,
            y_right,
            delta_y,
            d_left_norm,
            d_right_norm,
            tau,
            gap_len_norm,
        ],
        dim=-1,
    )
    return _masked_seq(x, mask)


def _load_mamba_class():
    try:
        from mamba_ssm.modules.mamba_simple import Mamba
    except Exception as exc:  # pragma: no cover - optional CUDA extension
        raise RuntimeError(
            "backbone_type=bimamba requires a working mamba-ssm installation. "
            "Verify `from mamba_ssm.modules.mamba_simple import Mamba` in the active environment."
        ) from exc
    return Mamba


def _directional_inputs(
    obs_pos: torch.Tensor,
    obs_mask: torch.Tensor,
    dt: torch.Tensor,
    exo: torch.Tensor,
    quality: torch.Tensor,
    seq_mask: torch.Tensor | None,
    reverse: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if reverse:
        obs_pos = reverse_by_seq_len(obs_pos, seq_mask)
        obs_mask = restore_by_seq_len(obs_mask.unsqueeze(-1), seq_mask).squeeze(-1)
        dt = restore_by_seq_len(_expand_dt(dt), seq_mask).squeeze(-1)
        exo = reverse_by_seq_len(exo, seq_mask)
        quality = reverse_by_seq_len(quality, seq_mask)
    return obs_pos, obs_mask, dt, exo, quality


def _previous_observation_feedback(obs_pos: torch.Tensor) -> torch.Tensor:
    bsz, t_len, _ = obs_pos.shape
    feedback = torch.zeros((bsz, t_len, 3), device=obs_pos.device, dtype=obs_pos.dtype)
    if t_len > 1:
        feedback[:, 1:, :] = obs_pos[:, :-1, :]
    return feedback


class UniLSTMPredictor(nn.Module):
    """Single-direction LSTM baseline with the same predictor interface."""

    def __init__(
        self,
        exo_dim: int,
        quality_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        use_anchor_features: bool = False,
        include_exo_quality: bool = True,
        recurrent_anchor_init: str = "none",
        obs_anchor_feedback_update: bool = False,
    ) -> None:
        super().__init__()
        self.use_anchor_features = bool(use_anchor_features)
        self.include_exo_quality = bool(include_exo_quality)
        self.net = ForwardPredictor(
            exo_dim=exo_dim,
            quality_dim=quality_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            use_anchor_features=use_anchor_features,
            include_exo_quality=include_exo_quality,
            recurrent_anchor_init=recurrent_anchor_init,
            obs_anchor_feedback_update=obs_anchor_feedback_update,
        )
        # Expose heads for existing grad diagnostics code path.
        self.mu_horiz_head = self.net.mu_horiz_head
        self.mu_alt_head = self.net.mu_alt_head
        self.logvar_head = self.net.logvar_horiz_head

    def forward(
        self,
        obs_pos: torch.Tensor,
        obs_mask: torch.Tensor,
        dt: torch.Tensor,
        exo: torch.Tensor,
        quality: torch.Tensor,
        target_pos: torch.Tensor | None,
        teacher_forcing_ratio: float,
        seq_mask: torch.Tensor | None = None,
        anchor_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net(
            obs_pos=obs_pos,
            obs_mask=obs_mask,
            dt=dt,
            exo=exo,
            quality=quality,
            target_pos=target_pos,
            teacher_forcing_ratio=teacher_forcing_ratio,
            seq_mask=seq_mask,
            anchor_features=anchor_features,
        )


class BiLSTMSequencePredictor(nn.Module):
    """Standard bidirectional LSTM encoder baseline."""

    def __init__(
        self,
        exo_dim: int,
        quality_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        use_anchor_features: bool = False,
        include_exo_quality: bool = True,
    ) -> None:
        super().__init__()
        self.use_anchor_features = bool(use_anchor_features)
        self.include_exo_quality = bool(include_exo_quality)
        extra_dim = (exo_dim + quality_dim) if self.include_exo_quality else 0
        input_size = (17 if self.use_anchor_features else (3 + 3 + 1 + 1)) + extra_dim
        hs = int(hidden_size)
        self.input_encoder = nn.Sequential(
            nn.Linear(input_size, hs),
            nn.LayerNorm(hs),
            nn.SiLU(),
        )
        self.lstm = nn.LSTM(
            input_size=hs,
            hidden_size=hs,
            num_layers=int(num_layers),
            batch_first=True,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            bidirectional=True,
        )
        out_dim = hs * 2
        self.mu_head = nn.Linear(out_dim, 3)
        self.logvar_head = nn.Linear(out_dim, 3)

    def forward(
        self,
        obs_pos: torch.Tensor,
        obs_mask: torch.Tensor,
        dt: torch.Tensor,
        exo: torch.Tensor,
        quality: torch.Tensor,
        target_pos: torch.Tensor | None,
        teacher_forcing_ratio: float,
        seq_mask: torch.Tensor | None = None,
        anchor_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del target_pos, teacher_forcing_ratio
        if self.use_anchor_features:
            if anchor_features is None:
                raise RuntimeError("BiLSTMSequencePredictor configured with use_anchor_features=True but anchor_features is missing.")
            parts = [anchor_features]
            if self.include_exo_quality:
                parts.extend([exo, quality])
            x = torch.cat(parts, dim=-1)
        else:
            parts = [torch.zeros_like(obs_pos), obs_pos, obs_mask.unsqueeze(-1), dt.unsqueeze(-1)]
            if self.include_exo_quality:
                parts.extend([exo, quality])
            x = torch.cat(parts, dim=-1)
        h = self.input_encoder(x)
        h = _masked_seq(h, seq_mask)
        h, _ = self.lstm(h)
        h = _masked_seq(h, seq_mask)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar


class TransformerSequencePredictor(nn.Module):
    """
    Standard Transformer encoder baseline with directional sequence handling.

    This is a non-autoregressive encoder predictor used for backbone comparison.
    """

    def __init__(
        self,
        exo_dim: int,
        quality_dim: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        reverse: bool = False,
        num_heads: int = 4,
        ff_multiplier: int = 4,
        use_anchor_features: bool = False,
        include_exo_quality: bool = True,
    ) -> None:
        super().__init__()
        self.reverse = reverse
        self.use_anchor_features = bool(use_anchor_features)
        self.include_exo_quality = bool(include_exo_quality)
        extra_dim = (exo_dim + quality_dim) if self.include_exo_quality else 0
        input_size = (17 if self.use_anchor_features else (3 + 3 + 1 + 1)) + extra_dim
        self.input_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=int(num_heads),
            dim_feedforward=int(hidden_size * ff_multiplier),
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.mu_head = nn.Linear(hidden_size, 3)
        self.logvar_head = nn.Linear(hidden_size, 3)

    def forward(
        self,
        obs_pos: torch.Tensor,
        obs_mask: torch.Tensor,
        dt: torch.Tensor,
        exo: torch.Tensor,
        quality: torch.Tensor,
        target_pos: torch.Tensor | None,
        teacher_forcing_ratio: float,
        seq_mask: torch.Tensor | None = None,
        anchor_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del target_pos, teacher_forcing_ratio
        if self.reverse:
            obs_pos, obs_mask, dt, exo, quality = _directional_inputs(
                obs_pos=obs_pos,
                obs_mask=obs_mask,
                dt=dt,
                exo=exo,
                quality=quality,
                seq_mask=seq_mask,
                reverse=True,
            )
            if anchor_features is not None:
                anchor_features = reverse_by_seq_len(anchor_features, seq_mask)

        if self.use_anchor_features:
            if anchor_features is None:
                raise RuntimeError("TransformerSequencePredictor configured with use_anchor_features=True but anchor_features is missing.")
            parts = [anchor_features]
            if self.include_exo_quality:
                parts.extend([exo, quality])
            x = torch.cat(parts, dim=-1)
        else:
            parts = [torch.zeros_like(obs_pos), obs_pos, obs_mask.unsqueeze(-1), dt.unsqueeze(-1)]
            if self.include_exo_quality:
                parts.extend([exo, quality])
            x = torch.cat(parts, dim=-1)
        h = self.input_encoder(x)
        h = _masked_seq(h, seq_mask)
        pad_mask = None
        if seq_mask is not None:
            pad_mask = (_ensure_seq_mask(seq_mask, obs_pos) <= 0.5)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        h = _masked_seq(h, seq_mask)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)

        if self.reverse:
            mu = restore_by_seq_len(mu, seq_mask)
            logvar = restore_by_seq_len(logvar, seq_mask)
        return mu, logvar


class MambaProtoSequencePredictor(nn.Module):
    """
    Single-direction encoder-only Mamba baseline.

    This keeps the same minimal-task-adaptation contract as the other proto
    baselines: anchor-conditioned inputs in model space, no bidirectional
    wrapper, no recurrent feedback decoder, and direct coordinate heads.
    """

    def __init__(
        self,
        exo_dim: int,
        quality_dim: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        reverse: bool = False,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_anchor_features: bool = False,
        include_exo_quality: bool = True,
    ) -> None:
        super().__init__()
        self.reverse = bool(reverse)
        self.use_anchor_features = bool(use_anchor_features)
        self.include_exo_quality = bool(include_exo_quality)
        extra_dim = (exo_dim + quality_dim) if self.include_exo_quality else 0
        input_size = (17 if self.use_anchor_features else (3 + 3 + 1 + 1)) + extra_dim
        hs = int(hidden_size)
        Mamba = _load_mamba_class()

        self.input_encoder = nn.Sequential(
            nn.Linear(input_size, hs),
            nn.LayerNorm(hs),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
        )
        self.blocks = nn.ModuleList(
            [
                Mamba(
                    d_model=hs,
                    d_state=int(d_state),
                    d_conv=int(d_conv),
                    expand=int(expand),
                    use_fast_path=False,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hs) for _ in range(int(num_layers))])
        self.dropout = nn.Dropout(float(dropout))
        self.mu_head = nn.Linear(hs, 3)
        self.logvar_head = nn.Linear(hs, 3)

    def forward(
        self,
        obs_pos: torch.Tensor,
        obs_mask: torch.Tensor,
        dt: torch.Tensor,
        exo: torch.Tensor,
        quality: torch.Tensor,
        target_pos: torch.Tensor | None,
        teacher_forcing_ratio: float,
        seq_mask: torch.Tensor | None = None,
        anchor_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del target_pos, teacher_forcing_ratio
        if self.reverse:
            obs_pos, obs_mask, dt, exo, quality = _directional_inputs(
                obs_pos=obs_pos,
                obs_mask=obs_mask,
                dt=dt,
                exo=exo,
                quality=quality,
                seq_mask=seq_mask,
                reverse=True,
            )
            if anchor_features is not None:
                anchor_features = reverse_by_seq_len(anchor_features, seq_mask)

        if self.use_anchor_features:
            if anchor_features is None:
                raise RuntimeError("MambaProtoSequencePredictor configured with use_anchor_features=True but anchor_features is missing.")
            parts = [anchor_features]
            if self.include_exo_quality:
                parts.extend([exo, quality])
            x = torch.cat(parts, dim=-1)
        else:
            parts = [torch.zeros_like(obs_pos), obs_pos, obs_mask.unsqueeze(-1), dt.unsqueeze(-1)]
            if self.include_exo_quality:
                parts.extend([exo, quality])
            x = torch.cat(parts, dim=-1)

        h = self.input_encoder(x)
        h = _masked_seq(h, seq_mask)
        for block, norm in zip(self.blocks, self.norms):
            h = norm(h + self.dropout(block(h)))
            h = _masked_seq(h, seq_mask)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)

        if self.reverse:
            mu = restore_by_seq_len(mu, seq_mask)
            logvar = restore_by_seq_len(logvar, seq_mask)
        return mu, logvar


class MambaEncoderSequencePredictor(nn.Module):
    """Directional encoder-only Mamba that also exposes hidden states."""

    def __init__(
        self,
        exo_dim: int,
        quality_dim: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        reverse: bool = False,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_anchor_features: bool = False,
        include_exo_quality: bool = True,
    ) -> None:
        super().__init__()
        self.reverse = bool(reverse)
        self.use_anchor_features = bool(use_anchor_features)
        self.include_exo_quality = bool(include_exo_quality)
        extra_dim = (exo_dim + quality_dim) if self.include_exo_quality else 0
        input_size = (17 if self.use_anchor_features else (3 + 3 + 1 + 1)) + extra_dim
        hs = int(hidden_size)
        Mamba = _load_mamba_class()

        self.input_encoder = nn.Sequential(
            nn.Linear(input_size, hs),
            nn.LayerNorm(hs),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
        )
        self.blocks = nn.ModuleList(
            [
                Mamba(
                    d_model=hs,
                    d_state=int(d_state),
                    d_conv=int(d_conv),
                    expand=int(expand),
                    use_fast_path=False,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hs) for _ in range(int(num_layers))])
        self.dropout = nn.Dropout(float(dropout))
        self.mu_head = nn.Linear(hs, 3)
        self.logvar_head = nn.Linear(hs, 3)

    def forward(
        self,
        obs_pos: torch.Tensor,
        obs_mask: torch.Tensor,
        dt: torch.Tensor,
        exo: torch.Tensor,
        quality: torch.Tensor,
        target_pos: torch.Tensor | None,
        teacher_forcing_ratio: float,
        seq_mask: torch.Tensor | None = None,
        anchor_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del target_pos, teacher_forcing_ratio
        if self.reverse:
            obs_pos, obs_mask, dt, exo, quality = _directional_inputs(
                obs_pos=obs_pos,
                obs_mask=obs_mask,
                dt=dt,
                exo=exo,
                quality=quality,
                seq_mask=seq_mask,
                reverse=True,
            )
            if anchor_features is not None:
                anchor_features = reverse_by_seq_len(anchor_features, seq_mask)

        if self.use_anchor_features:
            if anchor_features is None:
                raise RuntimeError("MambaEncoderSequencePredictor configured with use_anchor_features=True but anchor_features is missing.")
            parts = [anchor_features]
            if self.include_exo_quality:
                parts.extend([exo, quality])
            x = torch.cat(parts, dim=-1)
        else:
            parts = [torch.zeros_like(obs_pos), obs_pos, obs_mask.unsqueeze(-1), dt.unsqueeze(-1)]
            if self.include_exo_quality:
                parts.extend([exo, quality])
            x = torch.cat(parts, dim=-1)

        h = self.input_encoder(x)
        h = _masked_seq(h, seq_mask)
        for block, norm in zip(self.blocks, self.norms):
            h = norm(h + self.dropout(block(h)))
            h = _masked_seq(h, seq_mask)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)

        if self.reverse:
            mu = restore_by_seq_len(mu, seq_mask)
            logvar = restore_by_seq_len(logvar, seq_mask)
            h = restore_by_seq_len(h, seq_mask)
        return {"mu": mu, "logvar": logvar, "h": h}


class BiMambaProtoSequencePredictor(nn.Module):
    """
    Fair bidirectional Mamba baseline.

    This variant mirrors the other proto baselines: it only adds bidirectional
    context encoding, then uses a single shared direct head for the final
    3D output. It intentionally does not include hidden alignment MLPs,
    auxiliary xy/z decoupling, output-level fusion, or any method-specific
    routing logic.
    """

    def __init__(
        self,
        exo_dim: int,
        quality_dim: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_anchor_features: bool = False,
        include_exo_quality: bool = True,
    ) -> None:
        super().__init__()
        hs = int(hidden_size)
        self.forward_encoder = MambaEncoderSequencePredictor(
            exo_dim=exo_dim,
            quality_dim=quality_dim,
            hidden_size=hs,
            num_layers=num_layers,
            dropout=dropout,
            reverse=False,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_anchor_features=use_anchor_features,
            include_exo_quality=include_exo_quality,
        )
        self.backward_encoder = MambaEncoderSequencePredictor(
            exo_dim=exo_dim,
            quality_dim=quality_dim,
            hidden_size=hs,
            num_layers=num_layers,
            dropout=dropout,
            reverse=True,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_anchor_features=use_anchor_features,
            include_exo_quality=include_exo_quality,
        )
        self.mu_head = nn.Linear(hs * 2, 3)
        self.logvar_head = nn.Linear(hs * 2, 3)

    def forward(
        self,
        obs_pos: torch.Tensor,
        obs_mask: torch.Tensor,
        dt: torch.Tensor,
        exo: torch.Tensor,
        quality: torch.Tensor,
        target_pos: torch.Tensor | None,
        teacher_forcing_ratio: float,
        seq_mask: torch.Tensor | None = None,
        anchor_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out_f = self.forward_encoder(
            obs_pos=obs_pos,
            obs_mask=obs_mask,
            dt=dt,
            exo=exo,
            quality=quality,
            target_pos=target_pos,
            teacher_forcing_ratio=teacher_forcing_ratio,
            seq_mask=seq_mask,
            anchor_features=anchor_features,
        )
        out_b = self.backward_encoder(
            obs_pos=obs_pos,
            obs_mask=obs_mask,
            dt=dt,
            exo=exo,
            quality=quality,
            target_pos=target_pos,
            teacher_forcing_ratio=teacher_forcing_ratio,
            seq_mask=seq_mask,
            anchor_features=anchor_features,
        )
        h_bi = torch.cat([out_f["h"], out_b["h"]], dim=-1)
        h_bi = _masked_seq(h_bi, seq_mask)
        mu = self.mu_head(h_bi)
        logvar = self.logvar_head(h_bi)
        return mu, logvar


class MambaSequencePredictor(nn.Module):
    """
    Directional Mamba encoder predictor for clean backbone comparison.

    This module keeps the same input contract and output heads as the existing
    clean sequence baselines. It does not include altitude branching, SAVCA, or
    rule-based post-processing.
    """

    def __init__(
        self,
        exo_dim: int,
        quality_dim: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        reverse: bool = False,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.reverse = bool(reverse)
        hs = int(hidden_size)
        context_input_size = 17
        Mamba = _load_mamba_class()

        self.context_encoder = nn.Sequential(
            nn.Linear(context_input_size, hs),
            nn.LayerNorm(hs),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
        )
        self.blocks = nn.ModuleList(
            [
                Mamba(
                    d_model=hs,
                    d_state=int(d_state),
                    d_conv=int(d_conv),
                    expand=int(expand),
                    use_fast_path=False,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hs) for _ in range(int(num_layers))])
        self.dropout = nn.Dropout(float(dropout))
        self.step_decoder = nn.Sequential(
            nn.Linear(hs + 3, hs),
            nn.LayerNorm(hs),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
        )
        self.mu_horiz_head = nn.Linear(hs, 2)
        self.mu_alt_head = nn.Linear(hs, 1)
        self.logvar_horiz_head = nn.Linear(hs, 2)
        self.logvar_alt_head = nn.Linear(hs, 1)
        self.init_feedback = nn.Sequential(
            nn.Linear(7, hs),
            nn.SiLU(),
            nn.Linear(hs, 3),
        )

    def forward(
        self,
        obs_pos: torch.Tensor,
        obs_mask: torch.Tensor,
        dt: torch.Tensor,
        exo: torch.Tensor,
        quality: torch.Tensor,
        target_pos: torch.Tensor | None,
        teacher_forcing_ratio: float,
        seq_mask: torch.Tensor | None = None,
        anchor_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        bsz, t_len, _ = obs_pos.shape
        device = obs_pos.device
        mask = _ensure_seq_mask(seq_mask, obs_pos)
        if anchor_features is None:
            raise RuntimeError("MambaSequencePredictor requires anchor-conditioned features.")
        obs_pos_dir = obs_pos
        obs_mask_dir = obs_mask
        if self.reverse:
            anchor_features = reverse_by_seq_len(anchor_features, mask)
            obs_pos_dir = reverse_by_seq_len(obs_pos, mask)
            obs_mask_dir = reverse_by_seq_len(obs_mask.unsqueeze(-1), mask).squeeze(-1)
            target_pos = None if target_pos is None else reverse_by_seq_len(target_pos, mask)
        context_input = anchor_features
        h = self.context_encoder(context_input)
        h = _masked_seq(h, mask)
        for block, norm in zip(self.blocks, self.norms):
            h = norm(h + self.dropout(block(h)))
            h = _masked_seq(h, mask)

        init_feat = torch.cat([context_input[:, 0, 4:7], context_input[:, 0, 10:13], context_input[:, 0, 16:17]], dim=-1)
        feedback_for_current_step = self.init_feedback(init_feat) * mask[:, 0:1]
        mu_steps: list[torch.Tensor] = []
        logvar_steps: list[torch.Tensor] = []
        hidden_steps: list[torch.Tensor] = []
        for step in range(t_len):
            anchor_t = ((obs_mask_dir[:, step] > 0.5).to(dtype=obs_pos.dtype) * mask[:, step]).unsqueeze(-1)
            anchor_ref_t = context_input[:, step, 4:7]
            step_h = self.step_decoder(torch.cat([feedback_for_current_step, h[:, step, :]], dim=-1))
            mu_t = torch.cat([self.mu_horiz_head(step_h), self.mu_alt_head(step_h)], dim=-1)
            logvar_t = torch.cat(
                [self.logvar_horiz_head(step_h), self.logvar_alt_head(step_h)],
                dim=-1,
            )
            valid_t = mask[:, step : step + 1]
            mu_t = torch.where(anchor_t > 0.5, anchor_ref_t, mu_t)
            mu_t = mu_t * valid_t
            logvar_t = logvar_t * valid_t
            if self.training and target_pos is not None and teacher_forcing_ratio > 0:
                use_teacher = torch.rand((bsz, 1), device=device) < float(teacher_forcing_ratio)
                next_feedback = torch.where(use_teacher, target_pos[:, step, :], mu_t)
            else:
                next_feedback = mu_t
            next_feedback = torch.where(anchor_t > 0.5, anchor_ref_t, next_feedback)
            feedback_for_current_step = valid_t * next_feedback + (1.0 - valid_t) * feedback_for_current_step
            mu_steps.append(mu_t)
            logvar_steps.append(logvar_t)
            hidden_steps.append(step_h * valid_t)

        mu = torch.stack(mu_steps, dim=1)
        logvar = torch.stack(logvar_steps, dim=1)
        hidden = torch.stack(hidden_steps, dim=1)

        if self.reverse:
            mu = restore_by_seq_len(mu, mask)
            logvar = restore_by_seq_len(logvar, mask)
            hidden = restore_by_seq_len(hidden, mask)
        return {"mu": mu, "logvar": logvar, "h": hidden}


class MambaRecurrentSequencePredictor(nn.Module):
    """
    Mamba-context + gap-aware recurrent decoder predictor.

    Mamba encodes directional context over the whole segment, while the decoder
    preserves the original anchor-to-gap stepwise recovery bias through feedback
    and teacher forcing. This is still a clean backbone replacement: no altitude
    branch or post-processing is used.
    """

    def __init__(
        self,
        exo_dim: int,
        quality_dim: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        reverse: bool = False,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.reverse = bool(reverse)
        self.num_layers = int(num_layers)
        hs = int(hidden_size)
        context_input_size = 17
        step_input_size = 3 + 17 + hs
        Mamba = _load_mamba_class()

        self.context_encoder = nn.Sequential(
            nn.Linear(context_input_size, hs),
            nn.LayerNorm(hs),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
        )
        self.context_blocks = nn.ModuleList(
            [
                Mamba(
                    d_model=hs,
                    d_state=int(d_state),
                    d_conv=int(d_conv),
                    expand=int(expand),
                    use_fast_path=False,
                )
                for _ in range(max(1, int(num_layers)))
            ]
        )
        self.context_norms = nn.ModuleList([nn.LayerNorm(hs) for _ in range(max(1, int(num_layers)))])
        self.step_encoder = nn.Sequential(
            nn.Linear(step_input_size, hs),
            nn.LayerNorm(hs),
            nn.SiLU(),
        )
        self.cells = nn.ModuleList([GapAwareLSTMCell(input_size=hs, hidden_size=hs) for _ in range(self.num_layers)])
        self.dropout = nn.Dropout(float(dropout))
        self.mu_horiz_head = nn.Linear(hs, 2)
        self.mu_alt_head = nn.Linear(hs, 1)
        self.logvar_horiz_head = nn.Linear(hs, 2)
        self.logvar_alt_head = nn.Linear(hs, 1)
        self.init_state = nn.Sequential(
            nn.Linear(7, hs),
            nn.SiLU(),
            nn.Linear(hs, hs * self.num_layers * 2),
        )
        self.init_feedback = nn.Sequential(
            nn.Linear(7, hs),
            nn.SiLU(),
            nn.Linear(hs, 3),
        )

    def forward(
        self,
        obs_pos: torch.Tensor,
        obs_mask: torch.Tensor,
        dt: torch.Tensor,
        exo: torch.Tensor,
        quality: torch.Tensor,
        target_pos: torch.Tensor | None,
        teacher_forcing_ratio: float,
        seq_mask: torch.Tensor | None = None,
        anchor_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        bsz, t_len, _ = obs_pos.shape
        device = obs_pos.device
        del dt, exo, quality
        mask = _ensure_seq_mask(seq_mask, obs_pos)
        if anchor_features is None:
            raise RuntimeError("MambaRecurrentSequencePredictor requires anchor-conditioned features.")
        obs_pos_dir = obs_pos
        obs_mask_dir = obs_mask
        if self.reverse:
            anchor_features = reverse_by_seq_len(anchor_features, mask)
            obs_pos_dir = reverse_by_seq_len(obs_pos, mask)
            obs_mask_dir = reverse_by_seq_len(obs_mask.unsqueeze(-1), mask).squeeze(-1)
            target_pos = None if target_pos is None else reverse_by_seq_len(target_pos, mask)
        context_input = anchor_features
        context = self.context_encoder(context_input)
        context = _masked_seq(context, mask)
        for block, norm in zip(self.context_blocks, self.context_norms):
            context = norm(context + self.dropout(block(context)))
            context = _masked_seq(context, mask)

        init_feat = torch.cat([context_input[:, 0, 4:7], context_input[:, 0, 10:13], context_input[:, 0, 16:17]], dim=-1)
        init_state = self.init_state(init_feat).view(bsz, self.num_layers, 2, self.cells[0].hidden_size)
        h_states = [init_state[:, i, 0, :] * mask[:, 0:1] for i in range(self.num_layers)]
        c_states = [init_state[:, i, 1, :] * mask[:, 0:1] for i in range(self.num_layers)]
        feedback_for_current_step = self.init_feedback(init_feat) * mask[:, 0:1]
        mu_steps: list[torch.Tensor] = []
        logvar_steps: list[torch.Tensor] = []
        hidden_steps: list[torch.Tensor] = []

        for step in range(t_len):
            valid_t = mask[:, step : step + 1]
            anchor_t = ((obs_mask_dir[:, step] > 0.5).to(dtype=obs_pos.dtype) * mask[:, step]).unsqueeze(-1)
            anchor_ref_t = context_input[:, step, 4:7]
            reset_feat = torch.cat(
                [context_input[:, step, 4:7], context_input[:, step, 10:13], context_input[:, step, 16:17]],
                dim=-1,
            )
            reset_state = self.init_state(reset_feat).view(bsz, self.num_layers, 2, self.cells[0].hidden_size)
            x_t = torch.cat(
                [
                    feedback_for_current_step,
                    context_input[:, step, :],
                    context[:, step, :],
                ],
                dim=-1,
            )
            x = self.step_encoder(x_t)
            x = x * valid_t
            for layer_idx, cell in enumerate(self.cells):
                dt_t = context_input[:, step, 13]
                h_old = h_states[layer_idx]
                c_old = c_states[layer_idx]
                h_new, c_new = cell(x, dt_t, h_old, c_old)
                h_candidate = valid_t * h_new + (1.0 - valid_t) * h_old
                c_candidate = valid_t * c_new + (1.0 - valid_t) * c_old
                h_states[layer_idx] = torch.where(anchor_t > 0.5, reset_state[:, layer_idx, 0, :], h_candidate)
                c_states[layer_idx] = torch.where(anchor_t > 0.5, reset_state[:, layer_idx, 1, :], c_candidate)
                x = self.dropout(h_states[layer_idx]) if layer_idx < self.num_layers - 1 else h_states[layer_idx]
                x = valid_t * x

            mu_t = torch.cat([self.mu_horiz_head(x), self.mu_alt_head(x)], dim=-1)
            logvar_t = torch.cat([self.logvar_horiz_head(x), self.logvar_alt_head(x)], dim=-1)
            mu_t = torch.where(anchor_t > 0.5, anchor_ref_t, mu_t)
            mu_t = mu_t * valid_t
            logvar_t = logvar_t * valid_t
            if self.training and target_pos is not None and teacher_forcing_ratio > 0:
                use_teacher = torch.rand((bsz, 1), device=device) < float(teacher_forcing_ratio)
                next_feedback = torch.where(use_teacher, target_pos[:, step, :], mu_t)
            else:
                next_feedback = mu_t
            next_feedback = torch.where(anchor_t > 0.5, anchor_ref_t, next_feedback)
            feedback_for_current_step = valid_t * next_feedback + (1.0 - valid_t) * feedback_for_current_step

            mu_steps.append(mu_t)
            logvar_steps.append(logvar_t)
            hidden_steps.append(x * valid_t)

        mu = torch.stack(mu_steps, dim=1)
        logvar = torch.stack(logvar_steps, dim=1)
        hidden = torch.stack(hidden_steps, dim=1)
        if self.reverse:
            mu = restore_by_seq_len(mu, mask)
            logvar = restore_by_seq_len(logvar, mask)
            hidden = restore_by_seq_len(hidden, mask)
        return {"mu": mu, "logvar": logvar, "h": hidden}


class CNNLSTMSequencePredictor(nn.Module):
    """
    Lightweight CNN + LSTM baseline.

    Temporal Conv1D first captures short-range local dynamics, then LSTM models
    longer dependencies. This is a baseline variant (not replacing existing backbones).
    """

    def __init__(
        self,
        exo_dim: int,
        quality_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        reverse: bool = False,
        conv_channels: int = 128,
        conv_kernel_size: int = 3,
        use_anchor_features: bool = False,
        include_exo_quality: bool = True,
    ) -> None:
        super().__init__()
        self.reverse = bool(reverse)
        self.use_anchor_features = bool(use_anchor_features)
        self.include_exo_quality = bool(include_exo_quality)
        extra_dim = (exo_dim + quality_dim) if self.include_exo_quality else 0
        in_dim = (17 if self.use_anchor_features else (3 + 3 + 1 + 1)) + extra_dim
        cc = int(conv_channels)
        ks = int(conv_kernel_size)
        pad = ks // 2

        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, cc, kernel_size=ks, padding=pad),
            nn.SiLU(),
            nn.Conv1d(cc, cc, kernel_size=ks, padding=pad),
            nn.SiLU(),
        )
        self.lstm = nn.LSTM(
            input_size=cc,
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            bidirectional=False,
        )
        self.mu_head = nn.Linear(int(hidden_size), 3)
        self.logvar_head = nn.Linear(int(hidden_size), 3)

    def forward(
        self,
        obs_pos: torch.Tensor,
        obs_mask: torch.Tensor,
        dt: torch.Tensor,
        exo: torch.Tensor,
        quality: torch.Tensor,
        target_pos: torch.Tensor | None,
        teacher_forcing_ratio: float,
        seq_mask: torch.Tensor | None = None,
        anchor_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del target_pos, teacher_forcing_ratio
        if self.reverse:
            obs_pos, obs_mask, dt, exo, quality = _directional_inputs(
                obs_pos=obs_pos,
                obs_mask=obs_mask,
                dt=dt,
                exo=exo,
                quality=quality,
                seq_mask=seq_mask,
                reverse=True,
            )
            if anchor_features is not None:
                anchor_features = reverse_by_seq_len(anchor_features, seq_mask)

        if self.use_anchor_features:
            if anchor_features is None:
                raise RuntimeError("CNNLSTMSequencePredictor configured with use_anchor_features=True but anchor_features is missing.")
            parts = [anchor_features]
            if self.include_exo_quality:
                parts.extend([exo, quality])
            x = torch.cat(parts, dim=-1)
        else:
            parts = [torch.zeros_like(obs_pos), obs_pos, obs_mask.unsqueeze(-1), dt.unsqueeze(-1)]
            if self.include_exo_quality:
                parts.extend([exo, quality])
            x = torch.cat(parts, dim=-1)  # [B,T,C]

        # Conv1d expects [B,C,T]
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)  # [B,T,cc]
        h = _masked_seq(h, seq_mask)
        h, _ = self.lstm(h)
        h = _masked_seq(h, seq_mask)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)

        if self.reverse:
            mu = restore_by_seq_len(mu, seq_mask)
            logvar = restore_by_seq_len(logvar, seq_mask)
        return mu, logvar


class KalmanFilterSequencePredictor(nn.Module):
    """
    Constant-velocity Kalman filter baseline.

    This predictor is non-neural and deterministic. It keeps a 6D state
    [x, y, z, vx, vy, vz] and updates with observed positions when available.
    """

    def __init__(
        self,
        exo_dim: int,
        quality_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        reverse: bool = False,
        process_var: float = 1.0,
        measure_var: float = 0.25,
    ) -> None:
        del exo_dim, quality_dim, hidden_size, num_layers, dropout
        super().__init__()
        self.reverse = bool(reverse)
        self.process_var = float(max(1e-5, process_var))
        self.measure_var = float(max(1e-5, measure_var))
        # Kept for compatibility with diagnostics paths that probe these attrs.
        self.mu_head = nn.Identity()
        self.logvar_head = nn.Identity()

    def forward(
        self,
        obs_pos: torch.Tensor,
        obs_mask: torch.Tensor,
        dt: torch.Tensor,
        exo: torch.Tensor,
        quality: torch.Tensor,
        target_pos: torch.Tensor | None,
        teacher_forcing_ratio: float,
        seq_mask: torch.Tensor | None = None,
        anchor_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del exo, quality, target_pos, teacher_forcing_ratio, anchor_features
        if self.reverse:
            obs_pos = reverse_by_seq_len(obs_pos, seq_mask)
            obs_mask = restore_by_seq_len(obs_mask.unsqueeze(-1), seq_mask).squeeze(-1)
            dt = restore_by_seq_len(dt.unsqueeze(-1), seq_mask).squeeze(-1)

        bsz, tlen, _ = obs_pos.shape
        device = obs_pos.device
        dtype = obs_pos.dtype
        mask = _ensure_seq_mask(seq_mask, obs_pos)

        # State x=[pos(3), vel(3)].
        x = torch.zeros((bsz, 6), device=device, dtype=dtype)
        x[:, :3] = obs_pos[:, 0, :]
        p_diag = torch.ones((bsz, 6), device=device, dtype=dtype)

        out_mu = torch.zeros((bsz, tlen, 3), device=device, dtype=dtype)
        out_logvar = torch.zeros((bsz, tlen, 3), device=device, dtype=dtype)

        h = torch.zeros((bsz, 3, 6), device=device, dtype=dtype)
        h[:, :, :3] = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(bsz, 3, 3)

        i3 = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(bsz, 3, 3)
        i6_diag = torch.ones((bsz, 6), device=device, dtype=dtype)

        for t in range(tlen):
            dt_t = torch.clamp(dt[:, t], min=1e-3)
            valid_t = mask[:, t : t + 1]
            # Predict: pos += vel * dt.
            x[:, :3] = valid_t * (x[:, :3] + x[:, 3:] * dt_t.unsqueeze(-1)) + (1.0 - valid_t) * x[:, :3]
            # Diagonal covariance growth.
            q_pos = self.process_var * (dt_t**2)
            q_vel = self.process_var * dt_t
            p_diag[:, :3] = valid_t * (p_diag[:, :3] + q_pos.unsqueeze(-1)) + (1.0 - valid_t) * p_diag[:, :3]
            p_diag[:, 3:] = valid_t * (p_diag[:, 3:] + q_vel.unsqueeze(-1)) + (1.0 - valid_t) * p_diag[:, 3:]

            # Update when observed.
            m = (obs_mask[:, t] > 0.5).to(dtype=dtype).unsqueeze(-1)  # [B,1]
            z = obs_pos[:, t, :]  # [B,3]
            # S = HPH^T + R -> diagonal approx on position dims.
            s_diag = p_diag[:, :3] + self.measure_var
            k_diag = p_diag[:, :3] / torch.clamp(s_diag, min=1e-6)
            innov = z - x[:, :3]
            x[:, :3] = x[:, :3] + m * (k_diag * innov)
            # Joseph-lite diagonal form for pos block.
            p_diag[:, :3] = (1.0 - m * k_diag) * p_diag[:, :3] + (1.0 - m) * 0.0
            # Keep velocity loosely coupled by damping uncertainty after measurement.
            p_diag[:, 3:] = (1.0 - 0.2 * m) * p_diag[:, 3:]

            out_mu[:, t, :] = x[:, :3] * valid_t
            pos_var = torch.clamp(p_diag[:, :3], min=1e-6)
            out_logvar[:, t, :] = torch.log(pos_var) * valid_t

        if self.reverse:
            out_mu = restore_by_seq_len(out_mu, seq_mask)
            out_logvar = restore_by_seq_len(out_logvar, seq_mask)
        return out_mu, out_logvar
