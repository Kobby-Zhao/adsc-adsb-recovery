from __future__ import annotations

import torch
from torch import nn


class SimpleFusionHead(nn.Module):
    """Position-aware adaptive fusion of forward/backward branch outputs.

    The fusion produces per-timestep weights wf, wb where:
      pred = wf * mu_f + wb * mu_b

    A structural position prior sets the centre weight (wf_centre = 1 - gap_pos_ratio).
    The MLP learns a bounded deviation from this prior, so the position signal
    can never be fully overridden:

      wf = clip(wf_centre + tanh(mlp_output) * max_deviation, eps, 1-eps)

    This guarantees that near the left boundary (gap_pos -> 0) the forward branch
    dominates, and near the right boundary (gap_pos -> 1) the backward branch
    dominates — regardless of how the MLP weights evolve during training.
    """

    def __init__(
        self,
        exo_dim: int,
        quality_dim: int,
        global_quality_dim: int,
        hidden_size: int = 32,
        use_exo_quality: bool = False,
        position_prior_enabled: bool = True,
        position_prior_deviation: float = 0.30,
        weight_mode: str = "scalar",
    ) -> None:
        super().__init__()
        self.use_exo_quality = bool(use_exo_quality)
        self.position_prior_enabled = bool(position_prior_enabled)
        self.weight_mode = str(weight_mode).lower()
        if self.weight_mode not in {"scalar", "group", "dimension"}:
            raise ValueError(f"Unsupported fusion weight_mode={self.weight_mode!r}")
        # How far the MLP can deviate from the position prior (0 = no deviation, 0.5 = full override)
        self.max_deviation = float(max(0.0, min(0.49, position_prior_deviation)))
        self.num_groups = 1 if self.weight_mode == "scalar" else (2 if self.weight_mode == "group" else 3)
        # Input: mu_f(3) + mu_b(3) + |mu_f-mu_b|(3) + dt_prev + dt_next + gap_len + gap_pos_ratio + obs_mask + gq
        in_dim = 14 + global_quality_dim + (exo_dim + quality_dim if self.use_exo_quality else 0)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.num_groups),
        )

        self.register_buffer("_eps", torch.tensor(1e-8))
        self.last_detail_weights: torch.Tensor | None = None

        print(
            f"[fusion] input_dim={in_dim} use_exo_quality={int(self.use_exo_quality)} "
            f"position_prior_enabled={int(self.position_prior_enabled)} "
            f"max_deviation={self.max_deviation:.2f} "
            f"weight_mode={self.weight_mode}"
        )

    def forward(
        self,
        mu_f: torch.Tensor,
        mu_b: torch.Tensor,
        dt_prev: torch.Tensor,
        dt_next: torch.Tensor,
        obs_mask: torch.Tensor,
        exo: torch.Tensor,
        quality: torch.Tensor,
        global_quality: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, t_len, _ = mu_f.shape
        gq = global_quality.unsqueeze(1).expand(bsz, t_len, -1)
        gap_len = dt_prev + dt_next
        gap_pos_ratio = dt_prev / (gap_len + 1e-6)

        chunks = [
            mu_f,
            mu_b,
            torch.abs(mu_f - mu_b),
            dt_prev.unsqueeze(-1),
            dt_next.unsqueeze(-1),
            gap_len.unsqueeze(-1),
            gap_pos_ratio.unsqueeze(-1),
            obs_mask.unsqueeze(-1),
            gq,
        ]
        if self.use_exo_quality:
            chunks.extend([exo, quality])
        x = torch.cat(chunks, dim=-1)  # [B, T, in_dim]

        mlp_out = self.mlp(x)  # [B, T, G]

        if self.position_prior_enabled:
            # Position prior: wf_centre = 1 - gap_pos_ratio
            wf_centre = (1.0 - gap_pos_ratio).unsqueeze(-1)  # [B, T, 1]
            if self.max_deviation > 0:
                deviation = torch.tanh(mlp_out) * self.max_deviation  # [B, T, G]
            else:
                deviation = torch.zeros_like(mlp_out)

            wf = torch.clamp(wf_centre + deviation, min=self._eps, max=1.0 - self._eps)  # [B,T,G]
            wb = 1.0 - wf
        else:
            logits = torch.stack([mlp_out, -mlp_out], dim=-1)  # [B,T,G,2]
            w = torch.softmax(logits, dim=-1)
            wf = w[..., 0]
            wb = w[..., 1]

        detail_w = torch.stack([wf, wb], dim=-1)  # [B,T,G,2]
        if self.weight_mode == "scalar":
            pred = detail_w[..., 0, 0:1] * mu_f + detail_w[..., 0, 1:2] * mu_b
        elif self.weight_mode == "group":
            wf_xy = detail_w[..., 0, 0:1]
            wb_xy = detail_w[..., 0, 1:2]
            wf_z = detail_w[..., 1, 0:1]
            wb_z = detail_w[..., 1, 1:2]
            pred_xy = wf_xy * mu_f[..., 0:2] + wb_xy * mu_b[..., 0:2]
            pred_z = wf_z * mu_f[..., 2:3] + wb_z * mu_b[..., 2:3]
            pred = torch.cat([pred_xy, pred_z], dim=-1)
        else:
            wf_dim = detail_w[..., 0]
            wb_dim = detail_w[..., 1]
            pred = wf_dim * mu_f + wb_dim * mu_b

        scalar_w = detail_w.mean(dim=-2)  # [B,T,2], for backward-compatible aggregate diagnostics
        self.last_detail_weights = detail_w
        return pred, scalar_w, detail_w


class ConcatLinearFusion(nn.Module):
    """Concatenate forward/backward features and project back to 3-d output.

    Instead of learning explicit scalar weights wf, wb, the fusion
    concatenates mu_f and mu_b (3 → 6 dims) and lets a single Linear
    layer learn the optimal combination per output dimension.  This
    allows lat, lon, alt to receive different fusion coefficients —
    something a scalar-weighted sum cannot express.

    Used for the BiLSTM baseline to keep its fusion simple and
    contrast with OurMethod's structured position-aware fusion.
    """

    def __init__(self, **__kwargs: object) -> None:
        super().__init__()
        self.proj = nn.Linear(6, 3)

    def forward(
        self,
        mu_f: torch.Tensor,
        mu_b: torch.Tensor,
        dt_prev: torch.Tensor,
        dt_next: torch.Tensor,
        obs_mask: torch.Tensor,
        exo: torch.Tensor,
        quality: torch.Tensor,
        global_quality: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([mu_f, mu_b], dim=-1)  # [B, T, 6]
        pred = self.proj(x)                   # [B, T, 3]
        bsz, t_len = pred.shape[:2]
        w = torch.full((bsz, t_len, 2), 0.5, device=pred.device, dtype=pred.dtype)
        return pred, w


class FixedPositionPriorFusion(nn.Module):
    """Deterministic position-prior fusion: wf = 1 - gap_pos_ratio.

    No learnable parameters. Used for non-bilstm baseline models.
    """

    def __init__(self, **__kwargs: object) -> None:
        super().__init__()

    def forward(
        self,
        mu_f: torch.Tensor,
        mu_b: torch.Tensor,
        dt_prev: torch.Tensor,
        dt_next: torch.Tensor,
        obs_mask: torch.Tensor,
        exo: torch.Tensor,
        quality: torch.Tensor,
        global_quality: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gap_len = dt_prev + dt_next
        gap_pos_ratio = dt_prev / (gap_len + 1e-6)  # [B, T]
        wf = (1.0 - gap_pos_ratio).unsqueeze(-1)     # [B, T, 1]
        wb = 1.0 - wf
        w = torch.cat([wf, wb], dim=-1)              # [B, T, 2]
        pred = w[..., :1] * mu_f + w[..., 1:] * mu_b
        return pred, w


class HiddenStateFusion(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        extra_dim: int = 7,
        position_prior_deviation: float = 0.30,
    ) -> None:
        super().__init__()
        hs = int(hidden_size)
        self.max_deviation = float(max(0.0, min(0.49, position_prior_deviation)))
        self.register_buffer("_eps", torch.tensor(1e-8))
        self.gate = nn.Sequential(
            nn.Linear(hs * 2 + int(extra_dim), hs),
            nn.SiLU(),
            nn.Linear(hs, 1),
        )
        self.post = nn.Sequential(
            nn.Linear(hs + int(extra_dim), hs),
            nn.LayerNorm(hs),
            nn.SiLU(),
        )

    def forward(
        self,
        h_f: torch.Tensor,
        h_b: torch.Tensor,
        struct_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate_delta = self.gate(torch.cat([h_f, h_b, struct_feat], dim=-1))
        # struct_feat = [delta_y(3), d_left, d_right, tau, gap_len]
        tau = struct_feat[..., 5:6]
        wf_centre = 1.0 - tau
        gate = torch.clamp(
            wf_centre + torch.tanh(gate_delta) * self.max_deviation,
            min=self._eps,
            max=1.0 - self._eps,
        )
        h = gate * h_f + (1.0 - gate) * h_b
        h = self.post(torch.cat([h, struct_feat], dim=-1))
        weights = torch.cat([gate, 1.0 - gate], dim=-1)
        return h, weights
