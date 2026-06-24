from __future__ import annotations

import torch
from torch import nn


class GapAwareLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.decay_w = nn.Parameter(torch.ones(hidden_size))
        self.decay_b = nn.Parameter(torch.zeros(hidden_size))

    def forward(
        self,
        x_t: torch.Tensor,
        dt_t: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dt_t = dt_t.unsqueeze(-1)
        decay = torch.exp(-torch.relu(self.decay_w * dt_t + self.decay_b))
        h_prev = h_prev * decay
        c_prev = c_prev * decay
        h_t, c_t = self.cell(x_t, (h_prev, c_prev))
        return h_t, c_t
