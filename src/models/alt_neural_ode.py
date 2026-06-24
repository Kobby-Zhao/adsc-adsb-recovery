"""
Neural ODE altitude refiner — experimental module.

Models altitude recovery as a boundary value problem:
  dh/dτ = f_θ(h, τ, context)
  h(0) = h_L,  target: h(1) ≈ h_R

The ODE is integrated from the left anchor (τ=0) to the right anchor (τ=1)
using a fixed-step RK4 solver. The derivative function f_θ is a lightweight
MLP that takes the current altitude, normalised time, and gap-context features.

This module is an ALTERNATIVE to the DMS refiner — it can be plugged into
the height branch in place of (or in addition to) the existing residual head.

Reference: Jarry et al. (2025) — NODE-FDM, J. Open Aviation Science.
"""

from __future__ import annotations

import torch
from torch import nn


class NeuralODEFunc(nn.Module):
    """Derivative function dh/dτ = f(h, τ, context) for altitude ODE.

    Parameters
    ----------
    context_dim : int
        Dimensionality of the (time-invariant) gap-context vector.
    hidden_size : int
        Hidden dimension of the internal MLP.
    """

    def __init__(self, context_dim: int, hidden_size: int = 64) -> None:
        super().__init__()
        # Input: h (1), τ (1), context (context_dim)
        in_dim = 2 + context_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )
        # Initialise final layer near zero so the ODE starts near-identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self, h: torch.Tensor, tau: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """Compute dh/dτ.

        Parameters
        ----------
        h :  [B]               current altitude per sample
        tau : [B]              normalised time in [0, 1]
        context : [B, C]       per-sample gap-context (constant across τ)

        Returns
        -------
        dh/dτ : [B]
        """
        x = torch.cat([h.unsqueeze(-1), tau.unsqueeze(-1), context], dim=-1)
        return self.net(x).squeeze(-1)


def _rk4_step(
    func: NeuralODEFunc,
    h: torch.Tensor,
    tau: torch.Tensor,
    dtau: float,
    context: torch.Tensor,
) -> torch.Tensor:
    """Single RK4 integration step."""
    k1 = func(h, tau, context)
    k2 = func(h + 0.5 * dtau * k1, tau + 0.5 * dtau, context)
    k3 = func(h + 0.5 * dtau * k2, tau + 0.5 * dtau, context)
    k4 = func(h + dtau * k3, tau + dtau, context)
    return h + (dtau / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class NeuralODEAltitudeRefiner(nn.Module):
    """Neural ODE altitude refinement head.

    Integrates dh/dτ = f_θ(h, τ, context) from τ=0 (left anchor)
    to τ=1 (right anchor) using fixed-step RK4, producing a smooth
    altitude trajectory that satisfies boundary conditions by construction.

    The boundary condition h(0)=h_L is enforced as the integration
    initial value.  The right-anchor loss is applied externally by the
    caller (e.g. MSE at τ=1 vs h_R).

    Parameters
    ----------
    context_dim : int
        Dimensionality of the gap-context feature vector.
    hidden_size : int
        Hidden size of the ODE function MLP.
    n_steps : int
        Number of RK4 integration steps.  More steps = smoother but slower.
    """

    def __init__(
        self, context_dim: int, hidden_size: int = 64, n_steps: int = 20
    ) -> None:
        super().__init__()
        self.n_steps = int(n_steps)
        self.ode_func = NeuralODEFunc(context_dim=context_dim, hidden_size=hidden_size)
        self.dtau = 1.0 / float(max(1, self.n_steps))

    def forward(
        self,
        h_left: torch.Tensor,
        h_right: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce an altitude trajectory via ODE integration.

        Parameters
        ----------
        h_left :  [B]           left-anchor altitude (ENU-relative)
        h_right : [B]           right-anchor altitude (not used internally;
                                caller applies boundary loss)
        context : [B, C]        per-sample gap-context vector

        Returns
        -------
        h_traj : [B, n_steps+1]  altitude at τ = 0, 1/n_steps, …, 1
        tau_grid : [B, n_steps+1] normalised time grid
        """
        B = h_left.shape[0]
        device = h_left.device
        dtype = h_left.dtype

        # Initial state
        h = h_left.clone()  # [B]
        # Storage for trajectory at each step
        h_traj = [h.clone()]
        tau_vals = [torch.zeros(B, device=device, dtype=dtype)]

        tau = torch.zeros(B, device=device, dtype=dtype)
        for _ in range(self.n_steps):
            h = _rk4_step(self.ode_func, h, tau, self.dtau, context)
            tau = tau + self.dtau
            h_traj.append(h.clone())
            tau_vals.append(tau.clone())

        h_stacked = torch.stack(h_traj, dim=1)     # [B, n_steps+1]
        tau_stacked = torch.stack(tau_vals, dim=1)  # [B, n_steps+1]

        return h_stacked, tau_stacked


def build_ode_context(
    h_left: torch.Tensor,
    h_right: torch.Tensor,
    gap_len: torch.Tensor,
    mu_f: torch.Tensor | None = None,
    mu_b: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a per-sample context vector for the Neural ODE.

    Parameters
    ----------
    h_left : [B]        left-anchor altitude
    h_right : [B]       right-anchor altitude
    gap_len : [B]       total gap length in minutes
    mu_f : [B, T, D] or None   forward hidden states (mean-pooled)
    mu_b : [B, T, D] or None   backward hidden states (mean-pooled)

    Returns
    -------
    context : [B, C]
    """
    chunks = [
        h_left.unsqueeze(-1),    # 1
        h_right.unsqueeze(-1),   # 1
        (h_right - h_left).unsqueeze(-1),  # altitude span
        gap_len.unsqueeze(-1),   # gap length
    ]
    if mu_f is not None:
        chunks.append(mu_f.mean(dim=1))   # [B, D]
    if mu_b is not None:
        chunks.append(mu_b.mean(dim=1))   # [B, D]
    return torch.cat(chunks, dim=-1)
