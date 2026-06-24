from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.utils.geo import ecef_to_enu, ecef_to_wgs84, enu_to_ecef, wgs84_to_ecef


@dataclass
class CoordTransformContext:
    mode: str
    refs: list[tuple[float, float, float]]
    en_anchor_tracks: list[np.ndarray] | None = None
    anchor_masks: list[np.ndarray] | None = None
    obs_latlon_tracks: list[np.ndarray] | None = None
    u_anchor_tracks: list[np.ndarray] | None = None
    u_relative_anchor: bool = False
    en_relative_anchor: bool = False
    en_incremental: bool = False


def _pick_reference(
    target_i: np.ndarray,
    obs_i: np.ndarray,
    obs_mask_i: np.ndarray,
    seq_mask_i: np.ndarray,
    allow_target_fallback: bool = True,
) -> tuple[float, float, float]:
    valid_idx = np.where(seq_mask_i > 0.5)[0]
    if len(valid_idx) == 0:
        return 0.0, 0.0, 0.0

    anchor_idx = np.where((seq_mask_i > 0.5) & (obs_mask_i > 0.5))[0]
    if len(anchor_idx):
        ref_idx = int(anchor_idx[0])
        ref = obs_i[ref_idx]
    elif allow_target_fallback:
        ref_idx = int(valid_idx[0])
        ref = target_i[ref_idx]
    else:
        raise ValueError("No observed anchor is available for ENU reference in inference mode.")
    return float(ref[0]), float(ref[1]), float(ref[2])


def _to_enu(track: np.ndarray, ref: tuple[float, float, float], seq_mask_i: np.ndarray) -> np.ndarray:
    out = np.zeros_like(track, dtype=np.float32)
    ref_ecef = wgs84_to_ecef(ref[0], ref[1], ref[2])
    for t in range(track.shape[0]):
        if seq_mask_i[t] <= 0.5:
            continue
        lat, lon, alt = float(track[t, 0]), float(track[t, 1]), float(track[t, 2])
        x, y, z = wgs84_to_ecef(lat, lon, alt)
        e, n, u = ecef_to_enu(x, y, z, ref[0], ref[1], ref_ecef)
        out[t, 0] = e
        out[t, 1] = n
        out[t, 2] = u
    return out


def _from_enu(track: np.ndarray, ref: tuple[float, float, float], seq_mask_i: np.ndarray) -> np.ndarray:
    out = np.zeros_like(track, dtype=np.float32)
    ref_ecef = wgs84_to_ecef(ref[0], ref[1], ref[2])
    for t in range(track.shape[0]):
        if seq_mask_i[t] <= 0.5:
            continue
        e, n, u = float(track[t, 0]), float(track[t, 1]), float(track[t, 2])
        x, y, z = enu_to_ecef(e, n, u, ref[0], ref[1], ref_ecef)
        lat, lon, alt = ecef_to_wgs84(x, y, z)
        out[t, 0] = lat
        out[t, 1] = lon
        out[t, 2] = alt
    return out


def _build_u_anchor_track(obs_u: np.ndarray, obs_mask_i: np.ndarray, seq_mask_i: np.ndarray) -> np.ndarray:
    t_len = obs_u.shape[0]
    out = np.zeros((t_len,), dtype=np.float32)
    valid = seq_mask_i > 0.5
    anchor_idx = np.where(valid & (obs_mask_i > 0.5))[0]
    if anchor_idx.size == 0:
        return out

    # Forward-fill with the most recent anchor altitude.
    cur = float(obs_u[int(anchor_idx[0])])
    for t in range(t_len):
        if not valid[t]:
            continue
        if obs_mask_i[t] > 0.5:
            cur = float(obs_u[t])
        out[t] = cur

    # Back-fill leading valid steps before the first anchor with first anchor altitude.
    first = int(anchor_idx[0])
    first_u = float(obs_u[first])
    for t in range(first - 1, -1, -1):
        if valid[t]:
            out[t] = first_u
    return out


def build_anchor_alt_tracks(obs_pos: torch.Tensor, obs_mask: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
    obs_alt = obs_pos[..., 2]
    out = torch.zeros_like(obs_alt)
    bsz, t_len = obs_alt.shape
    for i in range(bsz):
        valid = seq_mask[i] > 0.5
        anchor = (obs_mask[i] > 0.5) & valid
        if not bool(anchor.any()):
            continue
        first = int(torch.nonzero(anchor, as_tuple=False)[0, 0].item())
        cur = float(obs_alt[i, first].item())
        for t in range(t_len):
            if not bool(valid[t]):
                continue
            if bool(anchor[t]):
                cur = float(obs_alt[i, t].item())
            out[i, t] = cur
        for t in range(first - 1, -1, -1):
            if bool(valid[t]):
                out[i, t] = float(obs_alt[i, first].item())
    return out


def build_anchor_pair_tracks(
    obs_pos: torch.Tensor,
    obs_mask: torch.Tensor,
    seq_mask: torch.Tensor,
    ctx: CoordTransformContext,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build per-timestep left/right anchor tracks in the same model coordinate
    space used by `prepare_model_coordinates()`.

    For `u_relative_anchor=True`, the altitude channel follows:
      z_left_rel = 0
      z_right_rel = z_R - z_L

    For `en_relative_anchor=True`, the planar channels follow:
      y_left_en = 0
      y_right_en = y_R - y_L
    """
    obs_np = obs_pos.detach().cpu().numpy()
    obs_mask_np = obs_mask.detach().cpu().numpy()
    seq_mask_np = seq_mask.detach().cpu().numpy()

    left_out = np.zeros_like(obs_np, dtype=np.float32)
    right_out = np.zeros_like(obs_np, dtype=np.float32)

    for i in range(obs_np.shape[0]):
        valid = seq_mask_np[i] > 0.5
        anchor_idx = np.where(valid & (obs_mask_np[i] > 0.5))[0]
        if anchor_idx.size == 0:
            continue

        if ctx.mode == "latlon":
            obs_abs = obs_np[i].astype(np.float32).copy()
        else:
            obs_abs = _to_enu(obs_np[i], ctx.refs[i], seq_mask_np[i])

        left_abs = np.zeros_like(obs_abs, dtype=np.float32)
        right_abs = np.zeros_like(obs_abs, dtype=np.float32)

        first = int(anchor_idx[0])
        last = int(anchor_idx[-1])

        cur_left = obs_abs[first].copy()
        for t in range(obs_abs.shape[0]):
            if not valid[t]:
                continue
            if obs_mask_np[i, t] > 0.5:
                cur_left = obs_abs[t].copy()
            left_abs[t] = cur_left

        cur_right = obs_abs[last].copy()
        for t in range(obs_abs.shape[0] - 1, -1, -1):
            if not valid[t]:
                continue
            if obs_mask_np[i, t] > 0.5:
                cur_right = obs_abs[t].copy()
            right_abs[t] = cur_right

        if ctx.en_relative_anchor or ctx.en_incremental:
            left_out[i, :, 0:2] = 0.0
            right_out[i, :, 0:2] = right_abs[:, 0:2] - left_abs[:, 0:2]
        else:
            left_out[i, :, 0:2] = left_abs[:, 0:2]
            right_out[i, :, 0:2] = right_abs[:, 0:2]

        if ctx.u_relative_anchor:
            left_alt = np.zeros((obs_abs.shape[0],), dtype=np.float32)
            right_alt = np.zeros((obs_abs.shape[0],), dtype=np.float32)
            cur_left_alt = float(obs_np[i, first, 2])
            for t in range(obs_abs.shape[0]):
                if not valid[t]:
                    continue
                if obs_mask_np[i, t] > 0.5:
                    cur_left_alt = float(obs_np[i, t, 2])
                left_alt[t] = cur_left_alt
            cur_right_alt = float(obs_np[i, last, 2])
            for t in range(obs_abs.shape[0] - 1, -1, -1):
                if not valid[t]:
                    continue
                if obs_mask_np[i, t] > 0.5:
                    cur_right_alt = float(obs_np[i, t, 2])
                right_alt[t] = cur_right_alt
            left_out[i, :, 2] = 0.0
            right_out[i, :, 2] = right_alt - left_alt
        else:
            left_out[i, :, 2] = left_abs[:, 2]
            right_out[i, :, 2] = right_abs[:, 2]

        left_out[i, ~valid] = 0.0
        right_out[i, ~valid] = 0.0

    return (
        torch.tensor(left_out, device=obs_pos.device, dtype=obs_pos.dtype),
        torch.tensor(right_out, device=obs_pos.device, dtype=obs_pos.dtype),
    )


def _build_en_increment_track(en_abs: np.ndarray, obs_mask_i: np.ndarray, seq_mask_i: np.ndarray) -> np.ndarray:
    out = np.zeros_like(en_abs, dtype=np.float32)
    valid = seq_mask_i > 0.5
    for t in range(en_abs.shape[0]):
        if not valid[t]:
            continue
        if t == 0 or seq_mask_i[t - 1] <= 0.5 or obs_mask_i[t] > 0.5:
            out[t] = 0.0
        else:
            out[t] = float(en_abs[t] - en_abs[t - 1])
    return out


def prepare_model_coordinates(
    target_pos: torch.Tensor,
    obs_pos: torch.Tensor,
    obs_mask: torch.Tensor,
    seq_mask: torch.Tensor,
    mode: str,
    allow_target_fallback: bool = True,
    u_relative_anchor: bool = False,
    en_relative_anchor: bool = True,
    en_incremental: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, CoordTransformContext]:
    if mode == "latlon":
        bsz = target_pos.shape[0]
        refs = [(0.0, 0.0, 0.0) for _ in range(bsz)]
        return target_pos, obs_pos, CoordTransformContext(mode=mode, refs=refs)

    if mode != "enu":
        raise ValueError(f"Unsupported coord mode: {mode}")

    target_np = target_pos.detach().cpu().numpy()
    obs_np = obs_pos.detach().cpu().numpy()
    obs_mask_np = obs_mask.detach().cpu().numpy()
    seq_mask_np = seq_mask.detach().cpu().numpy()

    target_out = np.zeros_like(target_np, dtype=np.float32)
    obs_out = np.zeros_like(obs_np, dtype=np.float32)
    refs: list[tuple[float, float, float]] = []
    en_anchor_tracks: list[np.ndarray] | None = [] if u_relative_anchor else None
    anchor_masks: list[np.ndarray] | None = [] if u_relative_anchor else None
    obs_latlon_tracks: list[np.ndarray] | None = [] if u_relative_anchor else None
    u_anchor_tracks: list[np.ndarray] | None = [] if u_relative_anchor else None

    for i in range(target_np.shape[0]):
        ref = _pick_reference(
            target_i=target_np[i],
            obs_i=obs_np[i],
            obs_mask_i=obs_mask_np[i],
            seq_mask_i=seq_mask_np[i],
            allow_target_fallback=allow_target_fallback,
        )
        refs.append(ref)
        target_i = _to_enu(target_np[i], ref, seq_mask_np[i])
        obs_i = _to_enu(obs_np[i], ref, seq_mask_np[i])
        if u_relative_anchor:
            en_anchor_e = _build_u_anchor_track(obs_i[:, 0], obs_mask_np[i], seq_mask_np[i])
            en_anchor_n = _build_u_anchor_track(obs_i[:, 1], obs_mask_np[i], seq_mask_np[i])
            en_anchor_i = np.stack([en_anchor_e, en_anchor_n], axis=-1)
            # Build anchor baseline in physical altitude space to avoid single-reference ENU U drift.
            u_anchor_i = _build_u_anchor_track(obs_np[i, :, 2], obs_mask_np[i], seq_mask_np[i])
            if en_incremental:
                target_i[:, 0] = _build_en_increment_track(target_i[:, 0], obs_mask_np[i], seq_mask_np[i])
                target_i[:, 1] = _build_en_increment_track(target_i[:, 1], obs_mask_np[i], seq_mask_np[i])
                # Keep EN anchor observations in input space.
                # Target stays in dE/dN space, but observed anchor E/N should remain available
                # so the network is not forced to infer planar context from mask/time features only.
            elif en_relative_anchor:
                target_i[:, 0:2] = target_i[:, 0:2] - en_anchor_i
                obs_i[:, 0:2] = obs_i[:, 0:2] - en_anchor_i
            target_i[:, 2] = target_np[i, :, 2] - u_anchor_i
            obs_i[:, 2] = obs_np[i, :, 2] - u_anchor_i
            assert en_anchor_tracks is not None
            en_anchor_tracks.append(en_anchor_i)
            assert anchor_masks is not None
            anchor_masks.append(obs_mask_np[i].astype(np.float32).copy())
            assert obs_latlon_tracks is not None
            obs_latlon_tracks.append(obs_np[i].astype(np.float32).copy())
            assert u_anchor_tracks is not None
            u_anchor_tracks.append(u_anchor_i)
        obs_i[obs_mask_np[i] <= 0.5] = 0.0
        target_out[i] = target_i
        obs_out[i] = obs_i

    device = target_pos.device
    return (
        torch.tensor(target_out, device=device),
        torch.tensor(obs_out, device=device),
        CoordTransformContext(
            mode=mode,
            refs=refs,
            en_anchor_tracks=en_anchor_tracks,
            anchor_masks=anchor_masks,
            obs_latlon_tracks=obs_latlon_tracks,
            u_anchor_tracks=u_anchor_tracks,
            u_relative_anchor=bool(u_relative_anchor),
            en_relative_anchor=bool(en_relative_anchor),
            en_incremental=bool(en_incremental),
        ),
    )


def restore_to_latlon(pred_pos: torch.Tensor, seq_mask: torch.Tensor, ctx: CoordTransformContext) -> torch.Tensor:
    if ctx.mode == "latlon":
        return pred_pos

    pred_np = pred_pos.detach().cpu().numpy()
    seq_mask_np = seq_mask.detach().cpu().numpy()
    out = np.zeros_like(pred_np, dtype=np.float32)

    for i in range(pred_np.shape[0]):
        pred_i = pred_np[i].copy()
        anchor_i = None
        if ctx.anchor_masks is not None:
            anchor_i = (ctx.anchor_masks[i] > 0.5) & (seq_mask_np[i] > 0.5)
        if ctx.en_anchor_tracks is not None and ctx.u_relative_anchor:
            if ctx.en_incremental:
                e_abs = np.zeros_like(pred_i[:, 0], dtype=np.float32)
                n_abs = np.zeros_like(pred_i[:, 1], dtype=np.float32)
                anchors = ctx.anchor_masks[i] if ctx.anchor_masks is not None else np.zeros_like(pred_i[:, 0], dtype=np.float32)
                seq_i = seq_mask_np[i]
                for t in range(pred_i.shape[0]):
                    if seq_i[t] <= 0.5:
                        continue
                    if anchors[t] > 0.5:
                        e_abs[t] = float(ctx.en_anchor_tracks[i][t, 0])
                        n_abs[t] = float(ctx.en_anchor_tracks[i][t, 1])
                    elif t == 0 or seq_i[t - 1] <= 0.5:
                        e_abs[t] = float(ctx.en_anchor_tracks[i][t, 0] + pred_i[t, 0])
                        n_abs[t] = float(ctx.en_anchor_tracks[i][t, 1] + pred_i[t, 1])
                    else:
                        e_abs[t] = e_abs[t - 1] + float(pred_i[t, 0])
                        n_abs[t] = n_abs[t - 1] + float(pred_i[t, 1])
                pred_i[:, 0] = e_abs
                pred_i[:, 1] = n_abs
            elif ctx.en_relative_anchor:
                pred_i[:, 0:2] = pred_i[:, 0:2] + ctx.en_anchor_tracks[i]
        if ctx.u_anchor_tracks is not None:
            alt_abs = pred_i[:, 2] + ctx.u_anchor_tracks[i]
            # For U-relative mode, dim2 is altitude delta (not ENU absolute U).
            # Reconstruct a consistent ENU U component for geo inversion using ref-alt displacement.
            # This avoids inconsistent (lat, lon, alt) tuples that can corrupt anchor-space metrics.
            if ctx.u_relative_anchor:
                ref_alt = float(ctx.refs[i][2])
                pred_i[:, 2] = alt_abs - ref_alt
                out_i = _from_enu(pred_i, ctx.refs[i], seq_mask_np[i])
                # Force anchor timestamps to exactly match observed positions.
                # This prevents drift crossing anchors from contaminating anchor-space metrics.
                if anchor_i is not None and ctx.obs_latlon_tracks is not None:
                    out_i[anchor_i, 0:2] = ctx.obs_latlon_tracks[i][anchor_i, 0:2]
                    out_i[anchor_i, 2] = ctx.obs_latlon_tracks[i][anchor_i, 2]
                out_i[:, 2] = alt_abs
                if anchor_i is not None and ctx.obs_latlon_tracks is not None:
                    out_i[anchor_i, 2] = ctx.obs_latlon_tracks[i][anchor_i, 2]
                out[i] = out_i
                continue
            pred_i[:, 2] = alt_abs
        out_i = _from_enu(pred_i, ctx.refs[i], seq_mask_np[i])
        if anchor_i is not None and ctx.obs_latlon_tracks is not None:
            out_i[anchor_i, 0:2] = ctx.obs_latlon_tracks[i][anchor_i, 0:2]
            out_i[anchor_i, 2] = ctx.obs_latlon_tracks[i][anchor_i, 2]
        out[i] = out_i

    return torch.tensor(out, device=pred_pos.device)
