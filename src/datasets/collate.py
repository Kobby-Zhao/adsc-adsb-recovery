from __future__ import annotations

import torch


def trajectory_collate_fn(batch: list[dict]) -> dict:
    max_len = max(item["target_pos"].shape[0] for item in batch)
    bsz = len(batch)

    padded = {}
    keys_2d = [
        "target_pos",
        "obs_pos",
        "exo",
        "vertical_exo",
        "quality",
    ]
    for x_name in keys_2d:
        sample = batch[0][x_name]
        out = torch.zeros((bsz, max_len, sample.shape[-1]), dtype=sample.dtype)
        for i, item in enumerate(batch):
            n = item[x_name].shape[0]
            out[i, :n] = item[x_name]
        padded[x_name] = out

    keys_1d = ["obs_mask", "dt_prev", "dt_next"]
    for name in keys_1d:
        sample = batch[0][name]
        out = torch.zeros((bsz, max_len), dtype=sample.dtype)
        for i, item in enumerate(batch):
            n = item[name].shape[0]
            out[i, :n] = item[name]
        padded[name] = out

    lengths = torch.tensor([item["target_pos"].shape[0] for item in batch], dtype=torch.long)
    seq_mask = torch.zeros((bsz, max_len), dtype=torch.float32)
    for i, n in enumerate(lengths.tolist()):
        seq_mask[i, :n] = 1.0

    out = {
        "sample_id": [item["sample_id"] for item in batch],
        "flight_id": [item["flight_id"] for item in batch],
        "times": [item["times"] for item in batch],
        "lengths": lengths,
        "seq_mask": seq_mask,
        "global_quality": torch.stack([item["global_quality"] for item in batch], dim=0),
        **padded,
    }

    # Optional segment-level metadata (new fields). Keep fully backward-compatible:
    # if old dataset does not provide these keys, collate output remains unchanged.
    numeric_meta_keys = [
        "segment_len",          # float32 [B]
        "segment_bucket",       # int64 [B]
        "anchor_pattern",       # int64 [B]
        "risk_flag",            # float32 [B]
        "risk_flag_teacher",    # float32 [B]
        "teacher_scale",        # float32 [B]
        "edge_weight",          # float32 [B]
        "residual_rmax_m",      # float32 [B]
        "residual_rmax_ft",     # deprecated compatibility alias [B]
        "gate_bias",            # float32 [B]
        "teacher_mode",         # int64 [B]
        "is_edge_sensitive",    # float32 [B]
        "last_two_step_geom",   # float32 [B]
        "last_two_step_geom_abs",   # float32 [B]
        "is_medium_two_anchor_high_last_two_step_geom",  # float32 [B]
        "sample_weight",        # float32 [B]
        "left_boundary_alt",    # float32 [B]
        "right_boundary_alt",   # float32 [B]
    ]
    for k in numeric_meta_keys:
        if k in batch[0]:
            out[k] = torch.stack([item[k] for item in batch], dim=0)

    string_meta_keys = [
        "segment_bucket_name",
        "anchor_pattern_name",
        "risk_level",
        "matched_risk_rule",
        "teacher_mode_name",
    ]
    for k in string_meta_keys:
        if k in batch[0]:
            out[k] = [str(item[k]) for item in batch]

    return out
