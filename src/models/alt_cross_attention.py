"""
Cross-attention anchor refiner — replacement for AnchorAwareTemporalAttention.

Instead of per-timestep element-wise scaling, this module implements
proper cross-attention:

  Query  ← each gap timestep's features  (what am I?)
  Key    ← boundary anchor features      (what constrains me?)
  Value  ← boundary anchor features      (what information do I need?)

Each timestep in the gap attends to the boundary anchors that constrain
it, explicitly aggregating information from the known points on either side.

Reference: DAMOT (2025) boundary prediction layer concept.
"""

from __future__ import annotations

import torch
from torch import nn


class CrossAttentionAnchorRefiner(nn.Module):
    """Cross-attention: gap timesteps attend to boundary anchor features.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of the backbone hidden sequence [B, T, hidden_dim].
    sparse_feat_dim : int
        Dimensionality of per-timestep sparse features.
    attn_hidden_size : int
        Hidden size of the Q/K projection MLPs.
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int,
        sparse_feat_dim: int,
        attn_hidden_size: int = 64,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = max(1, int(attn_hidden_size) // self.num_heads)
        self.scale = self.head_dim ** -0.5

        # Project hidden+sparse features to query/key/value space
        feat_dim = hidden_dim + sparse_feat_dim
        self.q_proj = nn.Linear(feat_dim, attn_hidden_size)
        self.k_proj = nn.Linear(feat_dim, attn_hidden_size)
        self.v_proj = nn.Linear(feat_dim, attn_hidden_size)
        self.out_proj = nn.Linear(attn_hidden_size, hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def _reshape_for_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, N, D] → [B*num_heads, N, head_dim]"""
        B, N, D = x.shape
        x = x.view(B, N, self.num_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3)  # [B, n_heads, N, head_dim]
        return x.reshape(B * self.num_heads, N, self.head_dim)

    def _extract_boundary_features(
        self,
        hidden_seq: torch.Tensor,
        sparse_feat_seq: torch.Tensor,
        obs_mask: torch.Tensor,
        dt_prev: torch.Tensor,
        dt_next: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build per-timestep left/right anchor Key-Value features.

        For each timestep, we forward-fill and backward-fill the features
        from the nearest anchor points.  This gives per-timestep (not global)
        boundary context that encodes *which* anchors constrain this position.

        Returns
        -------
        K_left :  [B, T, feat_dim]  left-anchor Key
        V_left :  [B, T, feat_dim]  left-anchor Value
        K_right : [B, T, feat_dim]  right-anchor Key
        V_right : [B, T, feat_dim]  right-anchor Value
        """
        B, T, _ = hidden_seq.shape
        feat = torch.cat([hidden_seq, sparse_feat_seq], dim=-1)  # [B, T, F]
        is_anchor = (obs_mask > 0.5).float()  # [B, T]

        # Forward-fill: carry anchor features left→right
        # At anchor: feat; at gap: nearest left anchor's feat
        left_feat = feat.clone()
        for t in range(1, T):
            mask = (is_anchor[:, t] < 0.5)  # not an anchor → copy from left
            left_feat[:, t] = torch.where(mask.unsqueeze(-1), left_feat[:, t - 1], left_feat[:, t])

        # Backward-fill: carry anchor features right→left
        right_feat = feat.clone()
        for t in range(T - 2, -1, -1):
            mask = (is_anchor[:, t] < 0.5)
            right_feat[:, t] = torch.where(mask.unsqueeze(-1), right_feat[:, t + 1], right_feat[:, t])

        # K/V share the same features (simplest form)
        return left_feat, left_feat, right_feat, right_feat

    def forward(
        self,
        hidden_seq: torch.Tensor,
        sparse_feat_seq: torch.Tensor,
        obs_mask: torch.Tensor,
        dt_prev: torch.Tensor,
        dt_next: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cross-attention: each timestep attends to its nearest left & right anchors.

        Query  ← current timestep features   [B, T, F]
        Key    ← left/right anchor features  [B, T, 2, F]  (per-timestep)
        Value  ← left/right anchor features  [B, T, 2, F]

        Each timestep computes attention over its two bounding anchors
        (the nearest anchor to the left and right).  Attention weights
        are learned and can depend on distance, features, context, etc.

        Returns
        -------
        attended_seq : [B, T, hidden_dim]
        context : [B, hidden_dim]
        attn_weights : [B, T, 2]   left/right attention per timestep
        """
        B, T, _ = hidden_seq.shape

        # Per-timestep boundary features via ffill/bfill
        K_left, V_left, K_right, V_right = self._extract_boundary_features(
            hidden_seq, sparse_feat_seq, obs_mask, dt_prev, dt_next,
        )  # each [B, T, F]

        # Stack anchors → [B, T, 2, F]
        K_anchors = torch.stack([K_left, K_right], dim=2)
        V_anchors = torch.stack([V_left, V_right], dim=2)

        # Query from each timestep
        feat = torch.cat([hidden_seq, sparse_feat_seq], dim=-1)  # [B, T, F]
        Q = self.q_proj(feat)        # [B, T, D]

        # Key/Value from per-timestep anchors → flatten [B*T, 2, F]
        K_flat = K_anchors.reshape(B * T, 2, -1)
        V_flat = V_anchors.reshape(B * T, 2, -1)
        K = self.k_proj(K_flat)      # [B*T, 2, D]
        V = self.v_proj(V_flat)      # [B*T, 2, D]

        # Per-timestep attention: each position attends to its 2 anchors
        Q_2d = Q.view(B * T, 1, -1)   # [B*T, 1, D]
        K_2d = K.view(B * T, 2, -1)   # [B*T, 2, D]
        V_2d = V.view(B * T, 2, -1)   # [B*T, 2, D]

        # Scaled dot-product: [B*T, 1, D] @ [B*T, D, 2] → [B*T, 1, 2]
        scores = torch.bmm(Q_2d, K_2d.transpose(1, 2)) / (K_2d.shape[-1] ** 0.5)
        attn_w = torch.softmax(scores, dim=-1)           # [B*T, 1, 2]
        attn_out = torch.bmm(attn_w, V_2d).squeeze(1)    # [B*T, D]

        # Reshape back
        attn_out = attn_out.view(B, T, -1)               # [B, T, D]
        attn_w_2d = attn_w.squeeze(1).view(B, T, 2)      # [B, T, 2]

        # Output projection + residual
        attended_seq = self.out_proj(attn_out) + hidden_seq

        # Global context
        if valid_mask is not None:
            am = attended_seq * valid_mask.unsqueeze(-1).float()
            context = am.sum(dim=1) / valid_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        else:
            context = attended_seq.mean(dim=1)

        return attended_seq, context, attn_w_2d
