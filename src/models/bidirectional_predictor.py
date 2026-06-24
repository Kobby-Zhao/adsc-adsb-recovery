from __future__ import annotations

import torch
from torch import nn

from src.models.gap_aware_lstm import GapAwareLSTMCell


class _DirectionalPredictor(nn.Module):
    def __init__(
        self,
        exo_dim: int,
        quality_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        reverse: bool = False,
        use_anchor_features: bool = False,
        include_exo_quality: bool = True,
        recurrent_anchor_init: str = "none",
        obs_anchor_feedback_update: bool = False,
    ) -> None:
        super().__init__()
        self.reverse = reverse
        self.num_layers = num_layers
        self.use_anchor_features = bool(use_anchor_features)
        self.include_exo_quality = bool(include_exo_quality)
        self.recurrent_anchor_init = str(recurrent_anchor_init).lower()
        self.obs_anchor_feedback_update = bool(obs_anchor_feedback_update)
        self.dropout = nn.Dropout(dropout)
        extra_dim = (exo_dim + quality_dim) if self.include_exo_quality else 0
        if self.use_anchor_features:
            input_size = 3 + 17 + extra_dim
        else:
            input_size = 3 + 3 + 1 + 1 + extra_dim
        self.input_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        self.cells = nn.ModuleList(
            [
                GapAwareLSTMCell(input_size=hidden_size, hidden_size=hidden_size)
                for _ in range(num_layers)
            ]
        )
        # Separate heads: horizontal (E,N) and altitude (U) have different
        # physical dynamics and should not share the same output projection.
        self.mu_horiz_head = nn.Linear(hidden_size, 2)
        self.mu_alt_head = nn.Linear(hidden_size, 1)
        self.logvar_horiz_head = nn.Linear(hidden_size, 2)
        self.logvar_alt_head = nn.Linear(hidden_size, 1)

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
        bsz, t_len, _ = obs_pos.shape
        device = obs_pos.device
        if seq_mask is None:
            seq_mask = torch.ones((bsz, t_len), device=device, dtype=obs_pos.dtype)
        else:
            seq_mask = seq_mask.to(device=device, dtype=obs_pos.dtype)

        if self.use_anchor_features and anchor_features is None:
            raise RuntimeError("Directional predictor configured with use_anchor_features=True but anchor_features is missing.")
        if self.recurrent_anchor_init not in {"none", "feedback", "hidden"}:
            raise RuntimeError(
                f"Unsupported recurrent_anchor_init={self.recurrent_anchor_init!r}; expected one of ['none','feedback','hidden']."
            )
        if self.recurrent_anchor_init == "hidden":
            raise RuntimeError(
                "recurrent_anchor_init='hidden' is not implemented yet; use 'none' or 'feedback'."
            )

        h_states = [torch.zeros((bsz, self.cells[0].hidden_size), device=device) for _ in range(self.num_layers)]
        c_states = [torch.zeros((bsz, self.cells[0].hidden_size), device=device) for _ in range(self.num_layers)]
        # Feedback used by current step input.
        # Forward mode: this is x_{t-1} feedback for step t.
        # Reverse mode: this is x_{t+1} feedback for step t.
        feedback_for_current_step = torch.zeros((bsz, 3), device=device)
        if self.use_anchor_features and self.recurrent_anchor_init == "feedback":
            feedback_for_current_step = anchor_features[:, 0, 4:7] * seq_mask[:, 0:1]

        mu_steps = []
        logvar_steps = []

        time_iter = range(t_len - 1, -1, -1) if self.reverse else range(t_len)

        for step in time_iter:
            obs_t = obs_pos[:, step, :]
            mask_t = obs_mask[:, step].unsqueeze(-1)
            dt_t = dt[:, step]
            exo_t = exo[:, step, :]
            quality_t = quality[:, step, :]
            valid_t = seq_mask[:, step].unsqueeze(-1)

            if self.use_anchor_features:
                parts = [feedback_for_current_step, anchor_features[:, step, :]]
                if self.include_exo_quality:
                    parts.extend([exo_t, quality_t])
                x_t = torch.cat(parts, dim=-1)
            else:
                parts = [feedback_for_current_step, obs_t, mask_t, dt_t.unsqueeze(-1)]
                if self.include_exo_quality:
                    parts.extend([exo_t, quality_t])
                x_t = torch.cat(parts, dim=-1)
            x = self.input_encoder(x_t)
            x = x * valid_t

            for layer_idx, cell in enumerate(self.cells):
                h_old = h_states[layer_idx]
                c_old = c_states[layer_idx]
                h_new, c_new = cell(x, dt_t, h_old, c_old)
                h_states[layer_idx] = valid_t * h_new + (1.0 - valid_t) * h_old
                c_states[layer_idx] = valid_t * c_new + (1.0 - valid_t) * c_old
                x = self.dropout(h_states[layer_idx]) if layer_idx < self.num_layers - 1 else h_states[layer_idx]

            mu_horiz = self.mu_horiz_head(x)
            mu_alt = self.mu_alt_head(x)
            mu_t = torch.cat([mu_horiz, mu_alt], dim=-1) * valid_t
            logvar_horiz = self.logvar_horiz_head(x)
            logvar_alt = self.logvar_alt_head(x)
            logvar_t = torch.cat([logvar_horiz, logvar_alt], dim=-1) * valid_t

            # IMPORTANT:
            # target_pos[:, step, :] is assigned only as feedback for the NEXT loop iteration.
            # So forward step t never consumes target_t directly in its own input;
            # reverse step t never consumes target_t directly either.
            if self.training and target_pos is not None and teacher_forcing_ratio > 0:
                use_teacher = torch.rand((bsz, 1), device=device) < teacher_forcing_ratio
                next_feedback = torch.where(use_teacher, target_pos[:, step, :], mu_t)
            else:
                next_feedback = mu_t
            if self.obs_anchor_feedback_update:
                next_feedback = torch.where(mask_t > 0.5, obs_t, next_feedback)
            feedback_for_current_step = valid_t * next_feedback + (1.0 - valid_t) * feedback_for_current_step

            mu_steps.append(mu_t)
            logvar_steps.append(logvar_t)

        if self.reverse:
            mu_steps.reverse()
            logvar_steps.reverse()

        mu = torch.stack(mu_steps, dim=1)
        logvar = torch.stack(logvar_steps, dim=1)
        return mu, logvar


class ForwardPredictor(_DirectionalPredictor):
    def __init__(
        self,
        exo_dim: int,
        quality_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        use_anchor_features: bool = False,
        include_exo_quality: bool = True,
        recurrent_anchor_init: str = "none",
        obs_anchor_feedback_update: bool = False,
    ) -> None:
        super().__init__(
            exo_dim=exo_dim,
            quality_dim=quality_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            reverse=False,
            use_anchor_features=use_anchor_features,
            include_exo_quality=include_exo_quality,
            recurrent_anchor_init=recurrent_anchor_init,
            obs_anchor_feedback_update=obs_anchor_feedback_update,
        )


class BackwardPredictor(_DirectionalPredictor):
    def __init__(
        self,
        exo_dim: int,
        quality_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        use_anchor_features: bool = False,
        include_exo_quality: bool = True,
        recurrent_anchor_init: str = "none",
        obs_anchor_feedback_update: bool = False,
    ) -> None:
        super().__init__(
            exo_dim=exo_dim,
            quality_dim=quality_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            reverse=True,
            use_anchor_features=use_anchor_features,
            include_exo_quality=include_exo_quality,
            recurrent_anchor_init=recurrent_anchor_init,
            obs_anchor_feedback_update=obs_anchor_feedback_update,
        )
