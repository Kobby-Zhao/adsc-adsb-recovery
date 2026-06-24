from __future__ import annotations

import numpy as np
import torch

from src.inference.postprocess import linear_interpolate_track


class TrajectoryInferencer:
    def __init__(self, model, device: torch.device) -> None:
        self.model = model
        self.device = device

    @torch.no_grad()
    def predict_batch(self, batch: dict, interpolate: bool = True) -> dict:
        out = self.model(
            obs_pos=batch["obs_pos"].to(self.device),
            obs_mask=batch["obs_mask"].to(self.device),
            seq_mask=batch["seq_mask"].to(self.device),
            dt_prev=batch["dt_prev"].to(self.device),
            dt_next=batch["dt_next"].to(self.device),
            exo=batch["exo"].to(self.device),
            vertical_exo=batch["vertical_exo"].to(self.device) if "vertical_exo" in batch else None,
            quality=batch["quality"].to(self.device),
            global_quality=batch["global_quality"].to(self.device),
            anchor_alt=batch.get("anchor_alt", None).to(self.device) if batch.get("anchor_alt", None) is not None else None,
            anchor_left=batch.get("anchor_left", None).to(self.device) if batch.get("anchor_left", None) is not None else None,
            anchor_right=batch.get("anchor_right", None).to(self.device) if batch.get("anchor_right", None) is not None else None,
            target_pos=None,
            teacher_forcing_ratio=0.0,
        )
        pred = out["pred_pos"].detach().cpu().numpy()

        if interpolate:
            for i in range(pred.shape[0]):
                pred[i] = linear_interpolate_track(pred[i])

        return {"pred_pos": pred}
