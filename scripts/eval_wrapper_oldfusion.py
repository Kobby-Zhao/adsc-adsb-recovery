#!/usr/bin/env python3
"""
Wrapper: monkey-patches old SimpleFusionHead, then delegates to evaluate.py.
Usage: same as evaluate.py
"""
import sys
from pathlib import Path

# Ensure ROOT is on sys.path (needed when called via subprocess)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---- Monkey-patch MUST happen before evaluate.py imports TrajectoryRecoveryModel ----
import torch
from torch import nn

# Force-import the fusion module and replace its SimpleFusionHead
import src.models.fusion as fusion_mod

class _OldSimpleFusionHead(nn.Module):
    def __init__(
        self, exo_dim, quality_dim, global_quality_dim,
        hidden_size=32, use_exo_quality=False,
        position_prior_enabled=False, position_prior_deviation=0.20,
    ):
        super().__init__()
        self.use_exo_quality = bool(use_exo_quality)
        self.position_prior_enabled = False
        self.max_deviation = 0.0
        in_dim = 11 + global_quality_dim + (exo_dim + quality_dim if self.use_exo_quality else 0)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),  # OLD: output 2 logits
        )
        self.register_buffer("_eps", torch.tensor(1e-8))

    def forward(self, mu_f, mu_b, dt_prev, dt_next, obs_mask, exo, quality, global_quality):
        bsz, t_len, _ = mu_f.shape
        gq = global_quality.unsqueeze(1).expand(bsz, t_len, -1)
        gap_len = dt_prev + dt_next
        gap_pos_ratio = dt_prev / (gap_len + 1e-6)
        chunks = [mu_f, mu_b, dt_prev.unsqueeze(-1), dt_next.unsqueeze(-1),
                  gap_len.unsqueeze(-1), gap_pos_ratio.unsqueeze(-1),
                  obs_mask.unsqueeze(-1), gq]
        if self.use_exo_quality:
            chunks.extend([exo, quality])
        x = torch.cat(chunks, dim=-1)
        mlp_out = self.mlp(x)  # [B, T, 2]
        w = torch.softmax(mlp_out, dim=-1)
        pred = w[..., :1] * mu_f + w[..., 1:] * mu_b
        return pred, w

# Patch both the fusion module AND full_model (which cached the import)
fusion_mod.SimpleFusionHead = _OldSimpleFusionHead
import src.models.full_model as full_model_mod
full_model_mod.SimpleFusionHead = _OldSimpleFusionHead
print(f"[wrapper] patched SimpleFusionHead in fusion + full_model — output dim=2", flush=True)

# ---- Patch evaluate.py's load_state_dict to use strict=False ----
import scripts.evaluate as _eval_mod
_original_main = _eval_mod.main
def _patched_main():
    # Monkey-patch torch load_state_dict to be strict=False
    import torch.nn as _nn
    _orig_load = _nn.Module.load_state_dict
    def _load_strict_false(self, state_dict, strict=None):
        return _orig_load(self, state_dict, strict=False)
    _nn.Module.load_state_dict = _load_strict_false
    return _original_main()
_eval_mod.main = _patched_main

# ---- Now delegate to the real evaluate.py main ----
# Remove ourselves from argv[0] so argparse works correctly
sys.argv[0] = str(Path(__file__).resolve().parents[0] / "evaluate.py")
_eval_mod.main()
