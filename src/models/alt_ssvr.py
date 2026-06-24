"""SSVR: State-Switching Vertical Recovery branch for gap-only altitude recovery.

This is a v1 module that models gap-interior altitude as a soft mixture of three states:
  L (left-plateau): height stays near z_L
  T (transition):   height follows a controlled blend of linear interpolation and
                     backbone height candidate
  R (right-plateau): height stays near z_R

The design avoids full-segment linearity (A1) and uncontrolled backbone drift.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def build_ssvr_state_labels(
    *,
    z_true_abs: torch.Tensor,
    z_L: torch.Tensor,
    z_R: torch.Tensor,
    gap_mask: torch.Tensor,
    plateau_threshold: float = 0.15,
) -> torch.Tensor:
    """Build weak state labels from ADS-B ground truth.

    Labels:
      0 = L (left-plateau)
      1 = T (transition)
      2 = R (right-plateau)

    A timestep is labelled L if |z_true - z_L| < thr * |z_R - z_L|,
    R if |z_true - z_R| < thr * |z_R - z_L|, and T otherwise.
    Anchor / padding positions are filled with -1 (ignore index).
    """
    device = z_true_abs.device
    delta_z = torch.abs(z_R - z_L)
    thr = plateau_threshold * torch.clamp(delta_z, min=1.0)

    dist_L = torch.abs(z_true_abs - z_L)
    dist_R = torch.abs(z_true_abs - z_R)

    y_state = torch.full_like(z_true_abs, -1, dtype=torch.long)

    is_L = (dist_L < thr) & gap_mask
    is_R = (dist_R < thr) & gap_mask
    is_T = gap_mask & (~is_L) & (~is_R)

    y_state = torch.where(is_L, torch.tensor(0, device=device, dtype=torch.long), y_state)
    y_state = torch.where(is_T, torch.tensor(1, device=device, dtype=torch.long), y_state)
    y_state = torch.where(is_R, torch.tensor(2, device=device, dtype=torch.long), y_state)

    return y_state


def _build_ssvr_state_active_mask(
    *,
    z_L: torch.Tensor,
    z_R: torch.Tensor,
    gap_mask: torch.Tensor,
    min_anchor_delta_m: float = 30.0,
) -> torch.Tensor:
    """Return a per-timestep bool mask indicating where state CE loss is active.

    State supervision is only applied inside gaps where |z_R - z_L| >= min_anchor_delta_m.
    Small-delta plateaus use L_rec and L_smooth only.
    """
    delta_z_abs = torch.abs(z_R - z_L)
    return gap_mask & (delta_z_abs >= min_anchor_delta_m)


class SSVRHeightBranch(nn.Module):
    """State-Switching Vertical Recovery branch v1.

    For each gap-interior timestep t:
      - Predicts three soft state probabilities [pi_L, pi_T, pi_R] via state_head.
      - Predicts a rho_t in [0, rho_max] controlling backbone height contribution
        in the T state.
      - Transition height: z_T(t) = (1-rho_t)*z_linear(t) + rho_t*z_main(t)
      - Final height: z_hat(t) = pi_L*z_L + pi_T*z_T(t) + pi_R*z_R

    Supports a *force_mode* for sanity-checking:
      - "linear": pi_T=1, rho=0  → z_hat = z_linear  (≈ A1)
      - "left":   pi_L=1         → z_hat = z_L
      - "right":  pi_R=1         → z_hat = z_R
      - "main":   pi_T=1, rho=1  → z_hat = z_main    (backbone only)
      - None: normal operation
    """

    def __init__(
        self,
        in_dim: int,
        hidden_size: int = 64,
        rho_max: float = 0.30,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.rho_max = float(rho_max)

        self.state_head = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_size)),
            nn.LayerNorm(int(hidden_size)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size), 3),
        )

        self.rho_head = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_size)),
            nn.LayerNorm(int(hidden_size)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size), 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        # State head: initialise towards high pi_T so that at initialisation
        # z_hat ≈ z_linear (A1), since rho also starts small.
        nn.init.zeros_(self.state_head[-1].weight)
        with torch.no_grad():
            self.state_head[-1].bias.copy_(
                torch.tensor([-2.0, 2.0, -2.0], dtype=self.state_head[-1].bias.dtype)
            )

        # Rho head: start with low rho (bias < 0 → sigmoid < 0.5 → rho small).
        nn.init.zeros_(self.rho_head[-1].weight)
        nn.init.constant_(self.rho_head[-1].bias, -2.0)

    def forward(
        self,
        features: torch.Tensor,
        z_L: torch.Tensor,
        z_R: torch.Tensor,
        z_main_abs: torch.Tensor,
        tau_gap: torch.Tensor,
        force_mode: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute SSVR height.

        Args:
            features: [B, T, in_dim] per-timestep feature vector.
            z_L: [B, T] left-anchor altitude broadcast within each gap.
            z_R: [B, T] right-anchor altitude broadcast within each gap.
            z_main_abs: [B, T] backbone altitude candidate (same scale as z_L/z_R).
            tau_gap: [B, T] normalised gap position in (0, 1).
            force_mode: optional sanity-check override.

        Returns dict with keys:
            z_hat, pi_L, pi_T, pi_R, rho, z_linear, z_T, state_logits
        """
        # -- linear baseline -----------------------------------------------------
        z_linear = z_L + tau_gap * (z_R - z_L)          # [B, T]

        if force_mode is not None:
            return self._forced_forward(
                force_mode=force_mode,
                z_L=z_L, z_R=z_R, z_main_abs=z_main_abs,
                z_linear=z_linear,
            )

        # -- state probabilities -------------------------------------------------
        state_logits = self.state_head(features)          # [B, T, 3]
        pi = F.softmax(state_logits, dim=-1)              # [B, T, 3]
        pi_L = pi[..., 0]
        pi_T = pi[..., 1]
        pi_R = pi[..., 2]

        # -- controlled backbone contribution ------------------------------------
        rho_raw = torch.sigmoid(self.rho_head(features).squeeze(-1))  # [B, T]
        rho = rho_raw * self.rho_max                                   # [B, T]

        # -- transition height ---------------------------------------------------
        z_T = (1.0 - rho) * z_linear + rho * z_main_abs  # [B, T]

        # -- final mixture -------------------------------------------------------
        z_hat = pi_L * z_L + pi_T * z_T + pi_R * z_R    # [B, T]

        return {
            "z_hat": z_hat,
            "pi_L": pi_L,
            "pi_T": pi_T,
            "pi_R": pi_R,
            "rho": rho,
            "z_linear": z_linear,
            "z_T": z_T,
            "state_logits": state_logits,
        }

    @staticmethod
    def _forced_forward(
        force_mode: str,
        z_L: torch.Tensor,
        z_R: torch.Tensor,
        z_main_abs: torch.Tensor,
        z_linear: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return the output that would result from a hard-coded state mixture."""
        B, T = z_L.shape
        device = z_L.device
        dtype = z_L.dtype

        if force_mode == "linear":
            # pi_T = 1, rho = 0  →  z_hat = z_linear
            pi_L = torch.zeros(B, T, device=device, dtype=dtype)
            pi_T = torch.ones(B, T, device=device, dtype=dtype)
            pi_R = torch.zeros(B, T, device=device, dtype=dtype)
            rho = torch.zeros(B, T, device=device, dtype=dtype)
            z_hat = z_linear
        elif force_mode == "left":
            # pi_L = 1  →  z_hat = z_L
            pi_L = torch.ones(B, T, device=device, dtype=dtype)
            pi_T = torch.zeros(B, T, device=device, dtype=dtype)
            pi_R = torch.zeros(B, T, device=device, dtype=dtype)
            rho = torch.zeros(B, T, device=device, dtype=dtype)
            z_hat = z_L
        elif force_mode == "right":
            # pi_R = 1  →  z_hat = z_R
            pi_L = torch.zeros(B, T, device=device, dtype=dtype)
            pi_T = torch.zeros(B, T, device=device, dtype=dtype)
            pi_R = torch.ones(B, T, device=device, dtype=dtype)
            rho = torch.zeros(B, T, device=device, dtype=dtype)
            z_hat = z_R
        elif force_mode == "main":
            # pi_T = 1, rho = 1  →  z_hat = z_main
            pi_L = torch.zeros(B, T, device=device, dtype=dtype)
            pi_T = torch.ones(B, T, device=device, dtype=dtype)
            pi_R = torch.zeros(B, T, device=device, dtype=dtype)
            rho = torch.ones(B, T, device=device, dtype=dtype)
            z_hat = z_main_abs
        else:
            raise ValueError(f"Unknown force_mode: {force_mode}")

        return {
            "z_hat": z_hat,
            "pi_L": pi_L,
            "pi_T": pi_T,
            "pi_R": pi_R,
            "rho": rho,
            "z_linear": z_linear,
            "z_T": z_hat,  # not meaningful in forced mode
            "state_logits": torch.zeros(B, T, 3, device=device, dtype=dtype),
        }


class SSVRFeatureBuilder:
    """Build per-timestep features for the SSVR state and rho heads.

    All features are scaled to roughly unit range.  z_main_abs, z_L, z_R must
    share the same height scale (metres or consistent normalised units).
    """

    @staticmethod
    def build(
        *,
        z_main_abs: torch.Tensor,
        tau: torch.Tensor,
        z_L: torch.Tensor,
        z_R: torch.Tensor,
        dt_prev: torch.Tensor,
        dt_next: torch.Tensor,
        gap_len: torch.Tensor,
    ) -> torch.Tensor:
        """Construct per-timestep feature tensor [B, T, 10].

        Args:
            z_main_abs: [B, T] backbone altitude candidate (same scale as z_L/z_R).
            tau: [B, T] normalised gap position.
            z_L: [B, T] left-anchor altitude.
            z_R: [B, T] right-anchor altitude.
            dt_prev: [B, T] distance to left anchor (minutes).
            dt_next: [B, T] distance to right anchor (minutes).
            gap_len: [B, T] gap length (minutes).
        """
        delta_z = z_R - z_L
        z_linear = z_L + tau * delta_z
        feat_list = [
            z_main_abs / 1000.0,
            tau,
            delta_z / 1000.0,
            dt_prev / 120.0,
            dt_next / 120.0,
            gap_len / 120.0,
            torch.abs(delta_z) / 1000.0,
            (z_main_abs - z_L) / 1000.0,
            (z_main_abs - z_R) / 1000.0,
            (z_main_abs - z_linear) / 1000.0,
        ]
        return torch.stack(feat_list, dim=-1)
