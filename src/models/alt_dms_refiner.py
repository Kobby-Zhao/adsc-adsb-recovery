from __future__ import annotations

import torch
from torch import nn


class AnchorAwareTemporalAttention(nn.Module):
    """Compute per-step attention scores conditioned on sparse-observation context.

    This module explicitly mixes temporal hidden features with anchor/visibility/freshness
    features to avoid uniform averaging across time under sparse observations.
    """

    def __init__(self, hidden_dim: int, sparse_feat_dim: int, attn_hidden_size: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        in_dim = int(hidden_dim + sparse_feat_dim)
        self.score_mlp = nn.Sequential(
            nn.Linear(in_dim, int(attn_hidden_size)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(attn_hidden_size), 1),
        )

    def forward(
        self,
        hidden_seq: torch.Tensor,
        sparse_feat_seq: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([hidden_seq, sparse_feat_seq], dim=-1)
        scores = self.score_mlp(x).squeeze(-1)
        if valid_mask is not None:
            scores = scores.masked_fill(valid_mask <= 0, -1e9)
        attn_weights = torch.softmax(scores, dim=1)
        attended_seq = hidden_seq * attn_weights.unsqueeze(-1)
        context = attended_seq.sum(dim=1)
        return attended_seq, context, attn_weights


class DMSAltitudeDecoder(nn.Module):
    """Decode sequence-level altitude latent features from attended states + context.

    DMS here means sequence-level decoder for altitude branch (not per-step isolated head).
    """

    def __init__(
        self,
        attended_dim: int,
        context_dim: int,
        sparse_feat_dim: int,
        latent_dim: int = 32,
        hidden_size: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        in_dim = int(attended_dim + context_dim + sparse_feat_dim)
        self.decode = nn.Sequential(
            nn.Linear(in_dim, int(hidden_size)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size), int(latent_dim)),
        )

    def forward(self, attended_seq: torch.Tensor, context: torch.Tensor, sparse_feat_seq: torch.Tensor) -> torch.Tensor:
        bsz, tlen, _ = attended_seq.shape
        ctx = context.unsqueeze(1).expand(bsz, tlen, -1)
        x = torch.cat([attended_seq, ctx, sparse_feat_seq], dim=-1)
        return self.decode(x)


class LightweightAltitudeRefiner(nn.Module):
    """Lightweight global consistency refinement for altitude latent sequence."""

    def __init__(
        self,
        latent_dim: int = 32,
        num_heads: int = 2,
        ff_multiplier: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d_model = int(latent_dim)
        n_heads = max(1, min(int(num_heads), d_model))
        if d_model % n_heads != 0:
            n_heads = 1
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=max(d_model * int(ff_multiplier), d_model),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.out = nn.Linear(d_model, 1)
        nn.init.normal_(self.out.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.out.bias)

    def forward(self, latent_seq: torch.Tensor, valid_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        key_padding_mask = None
        if valid_mask is not None:
            key_padding_mask = valid_mask <= 0
        refined = self.encoder(latent_seq, src_key_padding_mask=key_padding_mask)
        delta_alt = self.out(refined).squeeze(-1)
        return delta_alt, refined


class AltDMSRefinerV1Head(nn.Module):
    """Anchor-aware temporal attention + DMS decoder + lightweight global refiner.

    This head produces altitude residual sequence and is designed as an add-on branch
    over existing BiLSTM gap-aware backbone outputs.
    """

    def __init__(
        self,
        backbone_feature_dim: int,
        exo_dim: int,
        quality_dim: int,
        hidden_size: int = 64,
        latent_dim: int = 32,
        refiner_num_heads: int = 2,
        refiner_ff_multiplier: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.exo_dim = int(exo_dim)
        self.quality_dim = int(quality_dim)
        # obs + dt_prev + dt_next + gap_len + gap_pos + left/right flags + anchor_prev/next/delta/interp
        self.sparse_feat_dim = 1 + 1 + 1 + 1 + 1 + 1 + 1 + 2 + 4 + self.exo_dim + self.quality_dim
        self.attn = AnchorAwareTemporalAttention(
            hidden_dim=int(backbone_feature_dim),
            sparse_feat_dim=int(self.sparse_feat_dim),
            attn_hidden_size=int(hidden_size),
            dropout=float(dropout),
        )
        self.decoder = DMSAltitudeDecoder(
            attended_dim=int(backbone_feature_dim),
            context_dim=int(backbone_feature_dim),
            sparse_feat_dim=int(self.sparse_feat_dim),
            latent_dim=int(latent_dim),
            hidden_size=int(hidden_size),
            dropout=float(dropout),
        )
        self.refiner = LightweightAltitudeRefiner(
            latent_dim=int(latent_dim),
            num_heads=int(refiner_num_heads),
            ff_multiplier=int(refiner_ff_multiplier),
            dropout=float(dropout),
        )

    def build_sparse_features(
        self,
        obs_mask: torch.Tensor,
        dt_prev: torch.Tensor,
        dt_next: torch.Tensor,
        exo: torch.Tensor,
        quality: torch.Tensor,
        anchor_prev: torch.Tensor,
        anchor_next: torch.Tensor,
        anchor_delta: torch.Tensor,
        anchor_interp: torch.Tensor,
    ) -> torch.Tensor:
        gap_len = dt_prev + dt_next
        gap_pos = dt_prev / (gap_len + 1e-6)
        has_left_anchor = (dt_prev > 0).to(dt_prev.dtype)
        has_right_anchor = (dt_next > 0).to(dt_next.dtype)
        anchor_proximity_prev = 1.0 / (1.0 + dt_prev)
        anchor_proximity_next = 1.0 / (1.0 + dt_next)
        chunks = [
            obs_mask.unsqueeze(-1),
            dt_prev.unsqueeze(-1),
            dt_next.unsqueeze(-1),
            gap_len.unsqueeze(-1),
            gap_pos.unsqueeze(-1),
            has_left_anchor.unsqueeze(-1),
            has_right_anchor.unsqueeze(-1),
            anchor_proximity_prev.unsqueeze(-1),
            anchor_proximity_next.unsqueeze(-1),
            anchor_prev.unsqueeze(-1),
            anchor_next.unsqueeze(-1),
            anchor_delta.unsqueeze(-1),
            anchor_interp.unsqueeze(-1),
            exo,
            quality,
        ]
        return torch.cat(chunks, dim=-1)

    def forward(
        self,
        hidden_seq: torch.Tensor,
        obs_mask: torch.Tensor,
        dt_prev: torch.Tensor,
        dt_next: torch.Tensor,
        exo: torch.Tensor,
        quality: torch.Tensor,
        anchor_prev: torch.Tensor,
        anchor_next: torch.Tensor,
        anchor_delta: torch.Tensor,
        anchor_interp: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sparse_feat = self.build_sparse_features(
            obs_mask=obs_mask,
            dt_prev=dt_prev,
            dt_next=dt_next,
            exo=exo,
            quality=quality,
            anchor_prev=anchor_prev,
            anchor_next=anchor_next,
            anchor_delta=anchor_delta,
            anchor_interp=anchor_interp,
        )
        attended_seq, context, attn_weights = self.attn(
            hidden_seq=hidden_seq,
            sparse_feat_seq=sparse_feat,
            valid_mask=valid_mask,
        )
        latent_seq = self.decoder(attended_seq=attended_seq, context=context, sparse_feat_seq=sparse_feat)
        delta_alt, refined_seq = self.refiner(latent_seq=latent_seq, valid_mask=valid_mask)
        return delta_alt, {
            "attn_weights": attn_weights,
            "sparse_feat": sparse_feat,
            "latent_seq": latent_seq,
            "refined_seq": refined_seq,
        }
