from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from scripts.analyze_bidirectional_prediction_mechanism import (  # noqa: E402
    _build_model,
    _prepare_dataset,
    _restore_series,
)
from scripts.evaluate import _boundary_alt_from_model_obs, _max_gap_len  # noqa: E402
from src.datasets import trajectory_collate_fn  # noqa: E402
from src.training.alt_target import apply_alt_target_transform, invert_alt_target_transform  # noqa: E402
from src.training.coords import build_anchor_alt_tracks, build_anchor_pair_tracks, prepare_model_coordinates  # noqa: E402
from src.training.target_norm import denormalize_coords, load_target_stats, normalize_coords  # noqa: E402
from src.training.utils import load_config, set_seed  # noqa: E402


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _find_longest_gap_segment(obs_mask: np.ndarray) -> tuple[int, int]:
    best_s, best_e, best_len = -1, -1, -1
    t = 0
    n = len(obs_mask)
    while t < n:
        if obs_mask[t] > 0.5:
            t += 1
            continue
        s = t
        while t < n and obs_mask[t] <= 0.5:
            t += 1
        e = t
        if (e - s) > best_len:
            best_s, best_e, best_len = s, e, e - s
    if best_len <= 0:
        raise RuntimeError("No gap segment found.")
    return best_s, best_e


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) <= 1 or len(y) <= 1:
        return float("nan")
    if np.std(x) <= 1e-8 or np.std(y) <= 1e-8:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _trend_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) <= 2 or len(y) <= 2:
        return float("nan")
    dx = np.diff(x)
    dy = np.diff(y)
    return _safe_corr(dx, dy)


def _abs_to_right(arr: np.ndarray, right_anchor_alt: float, idxs: np.ndarray) -> np.ndarray:
    return np.abs(arr[idxs] - right_anchor_alt)


def _bucket_class(row: dict) -> str:
    jump = float(row["final_step_jump_m"])
    rmse = float(row["gap_alt_rmse"])
    if jump > 300.0:
        return "severe_boundary_jump_failure"
    if jump > 100.0:
        return "boundary_jump_failure"
    if rmse > 150.0 and jump <= 100.0:
        return "shape_failure"
    return "normal_or_mild"


def _boundary_subtype(row: dict) -> str:
    if row["failure_class"] not in {"boundary_jump_failure", "severe_boundary_jump_failure"}:
        return ""
    close_thr = 100.0
    mu_b_last1 = float(row["mu_b_last1_to_right_anchor"])
    pred_last1 = float(row["pred_last1_to_right_anchor"])
    z_before_last1 = float(row["z_before_last1_to_right_anchor"])
    z_after_last1 = float(row["z_after_last1_to_right_anchor"])
    if z_before_last1 <= close_thr and z_after_last1 > close_thr:
        return "C_zadapter_overcorrection"
    if mu_b_last1 <= close_thr and pred_last1 > close_thr:
        return "A_backward_ok_fusion_or_readout_bad"
    if z_before_last1 > close_thr and z_after_last1 > close_thr:
        return "D_trend_modeling_failure"
    return "B_backward_also_far"


def _plot_failure_case(out_path: Path, title: str, x: np.ndarray, truth_alt: np.ndarray, pred_alt: np.ndarray, obs_mask: np.ndarray, last_gap_idx: int, right_anchor_idx: int) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.plot(x, truth_alt, color="#111111", lw=2.2, label="ADS-B truth")
    ax.plot(x, pred_alt, color="#d62728", lw=2.0, label="Prediction")
    anchor = obs_mask > 0.5
    ax.scatter(x[anchor], truth_alt[anchor], s=22, color="#1b9e77", zorder=5, label="ADS-C anchors")
    ax.scatter([x[last_gap_idx]], [pred_alt[last_gap_idx]], s=40, color="#ff7f0e", zorder=6, label="Last gap point")
    ax.scatter([x[right_anchor_idx]], [truth_alt[right_anchor_idx]], s=46, color="#2ca02c", zorder=6, label="Right anchor")
    gap = ~anchor
    ax.fill_between(x, truth_alt.min(), truth_alt.max(), where=gap, color="#f1f1f1", alpha=0.45)
    ax.set_title(title)
    ax.set_xlabel("Minute index")
    ax.set_ylabel("Altitude (m)")
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _prepare_split_long_ids(per_sample_csv: Path, long_gap_threshold: int) -> set[str]:
    df = pd.read_csv(per_sample_csv)
    if "is_long_gap" in df.columns:
        long_df = df[df["is_long_gap"] == True].copy()  # noqa: E712
    else:
        long_df = df[df["max_gap_minutes"] >= long_gap_threshold].copy()
    return set(long_df["sample_id"].astype(str))


def _run_split(cfg: dict, checkpoint: Path, split: str, device: torch.device, out_dir: Path, long_gap_threshold: int = 20, few_anchor_threshold: int = 6) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_dir = _resolve(cfg["outputs"]["run_dir"])
    long_ids = _prepare_split_long_ids(run_dir / f"main_task_metrics_{split}_per_sample.csv", long_gap_threshold)

    ds = _prepare_dataset(cfg, split_name=split)
    ds.samples = [s for s in ds.samples if str(s["sample_id"]) in long_ids]
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=trajectory_collate_fn)
    model = _build_model(cfg, checkpoint, device)

    target_norm_cfg = cfg["training"].get("target_norm", {})
    target_norm_stats = None
    if bool(target_norm_cfg.get("enabled", False)):
        target_norm_stats = load_target_stats(_resolve(cfg["outputs"]["run_dir"]) / "target_model_scaler.json")
    alt_target_mode = str(cfg["training"].get("alt_target_robust", {}).get("mode", "none")).lower()
    alt_target_clip = float(cfg["training"].get("alt_target_robust", {}).get("clip_value", 3000.0))
    use_segment_teacher = bool(cfg.get("training", {}).get("risk_aware", {}).get("use_segment_teacher", True))
    use_alt_baseline_residual = bool(cfg.get("training", {}).get("risk_aware", {}).get("use_alt_baseline_residual", True))

    failure_rows: list[dict] = []
    branch_rows: list[dict] = []
    zadapter_rows: list[dict] = []
    feature_rows: list[dict] = []
    curve_cache: list[dict] = []

    with torch.no_grad():
        for batch in loader:
            obs_pos = batch["obs_pos"].to(device)
            obs_mask = batch["obs_mask"].to(device)
            dt_prev = batch["dt_prev"].to(device)
            dt_next = batch["dt_next"].to(device)
            exo = batch["exo"].to(device)
            quality = batch["quality"].to(device)
            global_quality = batch["global_quality"].to(device)
            target_pos = batch["target_pos"].to(device)
            seq_mask = batch["seq_mask"].to(device)
            target_model, obs_model, coord_ctx = prepare_model_coordinates(
                target_pos=target_pos,
                obs_pos=obs_pos,
                obs_mask=obs_mask,
                seq_mask=seq_mask,
                mode=str(cfg["model"].get("coord_mode", "latlon")),
                u_relative_anchor=bool(cfg["model"].get("u_relative_anchor", False)),
                en_relative_anchor=bool(cfg["model"].get("en_relative_anchor", True)),
                en_incremental=bool(cfg["model"].get("en_incremental", False)),
            )
            target_model_raw = target_model
            target_model = apply_alt_target_transform(target_model, mode=alt_target_mode, clip_value=alt_target_clip)
            obs_model = apply_alt_target_transform(obs_model, mode=alt_target_mode, clip_value=alt_target_clip)
            target_for_model = normalize_coords(target_model, target_norm_stats)
            obs_for_model = normalize_coords(obs_model, target_norm_stats)
            anchor_left_raw, anchor_right_raw = build_anchor_pair_tracks(
                obs_pos=obs_pos, obs_mask=obs_mask, seq_mask=seq_mask, ctx=coord_ctx
            )
            anchor_left_model = normalize_coords(
                apply_alt_target_transform(anchor_left_raw, mode=alt_target_mode, clip_value=alt_target_clip),
                target_norm_stats,
            )
            anchor_right_model = normalize_coords(
                apply_alt_target_transform(anchor_right_raw, mode=alt_target_mode, clip_value=alt_target_clip),
                target_norm_stats,
            )
            left_boundary_alt_model, right_boundary_alt_model = _boundary_alt_from_model_obs(
                obs_for_model=obs_for_model, obs_mask=obs_mask, seq_mask=seq_mask
            )
            anchor_alt = build_anchor_alt_tracks(obs_pos=obs_pos, obs_mask=obs_mask, seq_mask=seq_mask)

            out = model(
                obs_pos=obs_for_model,
                obs_mask=obs_mask,
                seq_mask=seq_mask,
                dt_prev=dt_prev,
                dt_next=dt_next,
                exo=exo,
                vertical_exo=batch["vertical_exo"].to(device) if "vertical_exo" in batch else None,
                quality=quality,
                global_quality=global_quality,
                anchor_alt=anchor_alt,
                risk_flag=batch["risk_flag"].to(device) if "risk_flag" in batch else None,
                teacher_scale=batch["teacher_scale"].to(device) if ("teacher_scale" in batch and use_segment_teacher) else None,
                risk_flag_teacher=batch["risk_flag_teacher"].to(device) if ("risk_flag_teacher" in batch and use_segment_teacher) else None,
                segment_bucket=batch["segment_bucket"].to(device) if "segment_bucket" in batch else None,
                edge_weight=batch["edge_weight"].to(device) if ("edge_weight" in batch and use_segment_teacher) else None,
                residual_rmax_m=batch["residual_rmax_m"].to(device) if ("residual_rmax_m" in batch and use_alt_baseline_residual) else None,
                residual_rmax_ft=batch["residual_rmax_ft"].to(device) if ("residual_rmax_ft" in batch and use_alt_baseline_residual) else None,
                gate_bias=batch["gate_bias"].to(device) if ("gate_bias" in batch and use_segment_teacher) else None,
                left_boundary_alt=left_boundary_alt_model,
                right_boundary_alt=right_boundary_alt_model,
                anchor_left=anchor_left_model,
                anchor_right=anchor_right_model,
                target_pos=None,
                teacher_forcing_ratio=0.0,
            )

            pred_latlon = _restore_series(
                out["pred_pos"],
                seq_mask=seq_mask,
                coord_ctx=coord_ctx,
                target_norm_stats=target_norm_stats,
                alt_target_mode=alt_target_mode,
                alt_target_clip=alt_target_clip,
            )

            mu_f_latlon = _restore_series(
                out["mu_f"],
                seq_mask=seq_mask,
                coord_ctx=coord_ctx,
                target_norm_stats=target_norm_stats,
                alt_target_mode=alt_target_mode,
                alt_target_clip=alt_target_clip,
            )
            mu_b_latlon = _restore_series(
                out["mu_b"],
                seq_mask=seq_mask,
                coord_ctx=coord_ctx,
                target_norm_stats=target_norm_stats,
                alt_target_mode=alt_target_mode,
                alt_target_clip=alt_target_clip,
            )

            h_align = out.get("h_align")
            h_z = out.get("h_z")
            z_before_latlon = None
            z_after_latlon = None
            if h_align is not None:
                z_before_raw = model.bimamba_context_mu_z_head(h_align)
                z_before_full = out["pred_pos"].clone()
                z_before_full[..., 2:3] = z_before_raw
                z_before_latlon = _restore_series(
                    z_before_full,
                    seq_mask=seq_mask,
                    coord_ctx=coord_ctx,
                    target_norm_stats=target_norm_stats,
                    alt_target_mode=alt_target_mode,
                    alt_target_clip=alt_target_clip,
                )
            if h_z is not None:
                z_after_raw = model.bimamba_context_mu_z_head(h_z)
                z_after_full = out["pred_pos"].clone()
                z_after_full[..., 2:3] = z_after_raw
                z_after_latlon = _restore_series(
                    z_after_full,
                    seq_mask=seq_mask,
                    coord_ctx=coord_ctx,
                    target_norm_stats=target_norm_stats,
                    alt_target_mode=alt_target_mode,
                    alt_target_clip=alt_target_clip,
                )

            sid = str(batch["sample_id"][0])
            fid = str(batch["flight_id"][0])
            valid = batch["seq_mask"][0].detach().cpu().numpy() > 0.5
            obs_i = batch["obs_mask"][0].detach().cpu().numpy()[valid]
            tgt_i = batch["target_pos"][0].detach().cpu().numpy()[valid]
            pred_i = pred_latlon[0][valid]
            mu_f_i = mu_f_latlon[0][valid]
            mu_b_i = mu_b_latlon[0][valid]
            z_before_i = z_before_latlon[0][valid] if z_before_latlon is not None else pred_i.copy()
            z_after_i = z_after_latlon[0][valid] if z_after_latlon is not None else pred_i.copy()
            dt_prev_i = batch["dt_prev"][0].detach().cpu().numpy()[valid]
            dt_next_i = batch["dt_next"][0].detach().cpu().numpy()[valid]

            s, e = _find_longest_gap_segment(obs_i)
            left_idx = s - 1
            right_idx = e
            if left_idx < 0 or right_idx >= len(obs_i):
                continue
            gap_idx = np.arange(s, e)
            last_gap_idx = int(gap_idx[-1])
            last5_idx = gap_idx[-min(5, len(gap_idx)) :]
            last10_n = max(1, int(math.ceil(0.1 * len(gap_idx))))
            last10_idx = gap_idx[-last10_n:]

            left_anchor_alt = float(tgt_i[left_idx, 2])
            right_anchor_alt = float(tgt_i[right_idx, 2])
            delta_z_abs = float(abs(right_anchor_alt - left_anchor_alt))

            pred_gap_alt = pred_i[gap_idx, 2]
            true_gap_alt = tgt_i[gap_idx, 2]
            mu_f_gap_alt = mu_f_i[gap_idx, 2]
            mu_b_gap_alt = mu_b_i[gap_idx, 2]
            z_before_gap_alt = z_before_i[gap_idx, 2]
            z_after_gap_alt = z_after_i[gap_idx, 2]

            row = {
                "split": split,
                "sample_id": sid,
                "flight_id": fid,
                "segment_id": sid,
                "anchor_count": int((obs_i > 0.5).sum()),
                "missing_ratio": float((obs_i <= 0.5).mean()),
                "max_gap_len": int(len(gap_idx)),
                "delta_z_abs": delta_z_abs,
                "gap_alt_rmse": float(np.sqrt(np.mean((pred_gap_alt - true_gap_alt) ** 2))),
                "gap_alt_mae": float(np.mean(np.abs(pred_gap_alt - true_gap_alt))),
                "final_step_jump_m": float(abs(pred_i[right_idx, 2] - pred_i[last_gap_idx, 2])),
                "pre_anchor_gap_to_right_anchor_m": float(abs(pred_i[last_gap_idx, 2] - right_anchor_alt)),
                "last_5_gap_to_right_anchor_mean": float(np.mean(np.abs(pred_i[last5_idx, 2] - right_anchor_alt))),
                "last_10pct_gap_to_right_anchor_mean": float(np.mean(np.abs(pred_i[last10_idx, 2] - right_anchor_alt))),
                "gap_alt_bias_mean": float(np.mean(pred_gap_alt - true_gap_alt)),
                "gap_alt_trend_corr": _trend_corr(pred_gap_alt, true_gap_alt),
                "delta_z_sign_correct": bool(np.sign(pred_i[last_gap_idx, 2] - left_anchor_alt) == np.sign(right_anchor_alt - left_anchor_alt)),
                "long_gap_flag": True,
                "few_anchor_flag": bool(int((obs_i > 0.5).sum()) <= few_anchor_threshold),
                "delta_z_gt300_flag": bool(delta_z_abs > 300.0),
            }
            row["long_gap_delta_z_gt300_flag"] = bool(row["long_gap_flag"] and row["delta_z_gt300_flag"])
            row["failure_class"] = _bucket_class(row)

            # branch/tail diagnostics
            bdiag = {
                "split": split,
                "sample_id": sid,
                "flight_id": fid,
                "segment_id": sid,
                "mu_f_last1_to_right_anchor": float(abs(mu_f_i[last_gap_idx, 2] - right_anchor_alt)),
                "mu_b_last1_to_right_anchor": float(abs(mu_b_i[last_gap_idx, 2] - right_anchor_alt)),
                "pred_last1_to_right_anchor": float(abs(pred_i[last_gap_idx, 2] - right_anchor_alt)),
                "z_before_last1_to_right_anchor": float(abs(float(z_before_i[last_gap_idx, 2]) - right_anchor_alt)),
                "z_after_last1_to_right_anchor": float(abs(float(z_after_i[last_gap_idx, 2]) - right_anchor_alt)),
                "mu_f_last5_to_right_anchor_mean": float(np.mean(np.abs(mu_f_i[last5_idx, 2] - right_anchor_alt))),
                "mu_b_last5_to_right_anchor_mean": float(np.mean(np.abs(mu_b_i[last5_idx, 2] - right_anchor_alt))),
                "pred_last5_to_right_anchor_mean": float(np.mean(np.abs(pred_i[last5_idx, 2] - right_anchor_alt))),
                "z_before_last5_to_right_anchor_mean": float(np.mean(np.abs(z_before_i[last5_idx, 2].astype(np.float64) - right_anchor_alt))),
                "z_after_last5_to_right_anchor_mean": float(np.mean(np.abs(z_after_i[last5_idx, 2].astype(np.float64) - right_anchor_alt))),
                "mu_f_last10pct_to_right_anchor_mean": float(np.mean(np.abs(mu_f_i[last10_idx, 2] - right_anchor_alt))),
                "mu_b_last10pct_to_right_anchor_mean": float(np.mean(np.abs(mu_b_i[last10_idx, 2] - right_anchor_alt))),
                "pred_last10pct_to_right_anchor_mean": float(np.mean(np.abs(pred_i[last10_idx, 2] - right_anchor_alt))),
                "z_before_last10pct_to_right_anchor_mean": float(np.mean(np.abs(z_before_i[last10_idx, 2].astype(np.float64) - right_anchor_alt))),
                "z_after_last10pct_to_right_anchor_mean": float(np.mean(np.abs(z_after_i[last10_idx, 2].astype(np.float64) - right_anchor_alt))),
                "mu_f_gap_alt_rmse": float(np.sqrt(np.mean((mu_f_gap_alt - true_gap_alt) ** 2))),
                "mu_b_gap_alt_rmse": float(np.sqrt(np.mean((mu_b_gap_alt - true_gap_alt) ** 2))),
                "pred_gap_alt_rmse": float(np.sqrt(np.mean((pred_gap_alt - true_gap_alt) ** 2))),
            }
            row["boundary_subtype"] = _boundary_subtype({**row, **bdiag})

            zdiag = {
                "split": split,
                "sample_id": sid,
                "flight_id": fid,
                "segment_id": sid,
                "gamma_z": float(out["gamma_z"].detach().cpu()) if out.get("gamma_z") is not None else float("nan"),
                "delta_h_z_norm": float(torch.norm(out["delta_h_z"][0][valid], dim=-1).mean().detach().cpu()) if out.get("delta_h_z") is not None else float("nan"),
                "h_align_norm": float(torch.norm(out["h_align"][0][valid], dim=-1).mean().detach().cpu()) if out.get("h_align") is not None else float("nan"),
                "h_z_norm": float(torch.norm(out["h_z"][0][valid], dim=-1).mean().detach().cpu()) if out.get("h_z") is not None else float("nan"),
                "z_before_adapter_std": float(np.std(z_before_gap_alt)),
                "z_after_adapter_std": float(np.std(z_after_gap_alt)),
            }
            zdiag["delta_h_over_halign"] = float(zdiag["delta_h_z_norm"] / (zdiag["h_align_norm"] + 1e-6)) if np.isfinite(zdiag["delta_h_z_norm"]) and np.isfinite(zdiag["h_align_norm"]) else float("nan")

            gap_len = float(dt_prev_i[last_gap_idx] + dt_next_i[last_gap_idx])
            feat = {
                "split": split,
                "sample_id": sid,
                "flight_id": fid,
                "segment_id": sid,
                "d_left": float(dt_prev_i[last_gap_idx]),
                "d_right": float(dt_next_i[last_gap_idx]),
                "gap_pos_ratio": float(dt_prev_i[last_gap_idx] / (gap_len + 1e-6)),
                "gap_len": gap_len,
                "gap_len_norm": float(np.log1p(gap_len) / np.log1p(180.0)),
                "tau": float(dt_prev_i[last_gap_idx] / (gap_len + 1e-6)),
                "obs_mask": float(obs_i[last_gap_idx]),
                "right_anchor_alt": right_anchor_alt,
                "last_gap_pred_alt": float(pred_i[last_gap_idx, 2]),
                "last_gap_true_alt": float(tgt_i[last_gap_idx, 2]),
            }

            failure_rows.append(row)
            branch_rows.append(bdiag)
            zadapter_rows.append(zdiag)
            feature_rows.append(feat)
            curve_cache.append(
                {
                    "split": split,
                    "sample_id": sid,
                    "flight_id": fid,
                    "failure_class": row["failure_class"],
                    "title": f"{split}:{sid} | rmse={row['gap_alt_rmse']:.1f} jump={row['final_step_jump_m']:.1f}",
                    "x": np.arange(len(obs_i)),
                    "truth_alt": tgt_i[:, 2].copy(),
                    "pred_alt": pred_i[:, 2].copy(),
                    "obs_mask": obs_i.copy(),
                    "last_gap_idx": last_gap_idx,
                    "right_anchor_idx": right_idx,
                    "severity": row["final_step_jump_m"] if row["failure_class"] in {"boundary_jump_failure", "severe_boundary_jump_failure"} else row["gap_alt_rmse"],
                }
            )

    fdf = pd.DataFrame(failure_rows)
    bdf = pd.DataFrame(branch_rows)
    zdf = pd.DataFrame(zadapter_rows)
    xdf = pd.DataFrame(feature_rows)

    plots_dir = out_dir / f"plots_{split}"
    plots_dir.mkdir(parents=True, exist_ok=True)
    if not fdf.empty:
        merged = fdf.merge(pd.DataFrame(curve_cache), on=["split", "sample_id", "flight_id", "failure_class"], how="left")
        for cls in ["severe_boundary_jump_failure", "boundary_jump_failure", "shape_failure", "normal_or_mild"]:
            sub = merged[merged["failure_class"].eq(cls)].sort_values("severity", ascending=False).head(3)
            class_dir = plots_dir / cls
            class_dir.mkdir(parents=True, exist_ok=True)
            for _, r in sub.iterrows():
                _plot_failure_case(
                    class_dir / f"{r['sample_id']}.png",
                    str(r["title"]),
                    r["x"],
                    r["truth_alt"],
                    r["pred_alt"],
                    r["obs_mask"],
                    int(r["last_gap_idx"]),
                    int(r["right_anchor_idx"]),
                )
    return fdf, bdf, zdf, xdf


def _write_summary(out_dir: Path, fdf: pd.DataFrame, *, filename: str = "failure_mode_summary.md") -> None:
    def _df_to_text(df: pd.DataFrame) -> str:
        if df.empty:
            return "(empty)"
        return df.to_string(index=False)

    lines = ["# Failure Mode Summary", ""]
    if fdf.empty:
        lines.append("No long-gap samples found.")
        (out_dir / filename).write_text("\n".join(lines), encoding="utf-8")
        return
    lines.append("## Counts")
    counts = fdf.groupby(["split", "failure_class"]).size().rename("count").reset_index()
    lines.append(_df_to_text(counts))
    lines.append("")
    lines.append("## Boundary Subtypes")
    bsub = fdf[fdf["boundary_subtype"].astype(str) != ""].groupby(["split", "boundary_subtype"]).size().rename("count").reset_index()
    if not bsub.empty:
        lines.append(_df_to_text(bsub))
        lines.append("")
    lines.append("## Aggregate Means")
    cols = ["gap_alt_rmse", "gap_alt_mae", "final_step_jump_m", "pre_anchor_gap_to_right_anchor_m", "last_5_gap_to_right_anchor_mean", "last_10pct_gap_to_right_anchor_mean", "gap_alt_bias_mean", "gap_alt_trend_corr"]
    agg = fdf.groupby("split")[cols].mean(numeric_only=True).reset_index()
    lines.append(_df_to_text(agg))
    (out_dir / filename).write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/ab24_xyaux_zlinear_zadapter.yaml")
    ap.add_argument("--checkpoint", default="outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/ab24_xyaux_zlinear_zadapter/best.pt")
    ap.add_argument("--out-dir", default="outputs/experiments/obs_conditioned_gaponly/longgap_failure_diagnosis_20260531")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--long-gap-threshold", type=int, default=20)
    ap.add_argument("--few-anchor-threshold", type=int, default=6)
    return_parser = ap
    args = return_parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for BiMamba diagnosis, but torch.cuda.is_available() is false.")
    device = torch.device(args.device)
    set_seed(42)

    cfg = load_config(str(_resolve(args.config)))
    ckpt = _resolve(args.checkpoint)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    branch_frames = []
    z_frames = []
    feat_frames = []
    for split in ["val", "test"]:
        fdf, bdf, zdf, xdf = _run_split(
            cfg=cfg,
            checkpoint=ckpt,
            split=split,
            device=device,
            out_dir=out_dir,
            long_gap_threshold=args.long_gap_threshold,
            few_anchor_threshold=args.few_anchor_threshold,
        )
        frames.append(fdf)
        branch_frames.append(bdf)
        z_frames.append(zdf)
        feat_frames.append(xdf)

    failure_df = pd.concat(frames, ignore_index=True)
    branch_df = pd.concat(branch_frames, ignore_index=True)
    zadapter_df = pd.concat(z_frames, ignore_index=True)
    feature_df = pd.concat(feat_frames, ignore_index=True)

    branch_with_failure = branch_df.merge(
        failure_df[["split", "sample_id", "flight_id", "segment_id", "failure_class", "boundary_subtype"]],
        on=["split", "sample_id", "flight_id", "segment_id"],
        how="left",
    )

    failure_df.to_csv(out_dir / "failure_mode_diagnosis.csv", index=False)
    failure_df.to_csv(out_dir / "failure_mode_summary.csv", index=False)
    branch_df.to_csv(out_dir / "branch_tail_diagnosis.csv", index=False)
    branch_with_failure.to_csv(out_dir / "branch_level_failure_diagnosis.csv", index=False)
    branch_with_failure.to_csv(out_dir / "branch_level_failure_diagnosis_fixed.csv", index=False)
    zadapter_df.to_csv(out_dir / "zadapter_tail_diagnosis.csv", index=False)
    zadapter_df.to_csv(out_dir / "zadapter_tail_diagnosis_fixed.csv", index=False)
    feature_df.to_csv(out_dir / "feature_tail_diagnosis.csv", index=False)
    failure_df[
        failure_df["failure_class"].isin(["boundary_jump_failure", "severe_boundary_jump_failure"])
    ].sort_values(["final_step_jump_m", "gap_alt_rmse"], ascending=[False, False]).head(10).to_csv(
        out_dir / "top10_boundary_jump_failure.csv", index=False
    )
    failure_df[
        failure_df["failure_class"].eq("shape_failure")
    ].sort_values(["gap_alt_rmse", "final_step_jump_m"], ascending=[False, True]).head(10).to_csv(
        out_dir / "top10_shape_failure.csv", index=False
    )
    _write_summary(out_dir, failure_df)
    _write_summary(out_dir, failure_df, filename="failure_mode_summary_fixed.md")

    print(out_dir / "failure_mode_diagnosis.csv")
    print(out_dir / "failure_mode_summary.csv")
    print(out_dir / "branch_tail_diagnosis.csv")
    print(out_dir / "branch_level_failure_diagnosis.csv")
    print(out_dir / "branch_level_failure_diagnosis_fixed.csv")
    print(out_dir / "zadapter_tail_diagnosis.csv")
    print(out_dir / "zadapter_tail_diagnosis_fixed.csv")
    print(out_dir / "feature_tail_diagnosis.csv")
    print(out_dir / "top10_boundary_jump_failure.csv")
    print(out_dir / "top10_shape_failure.csv")
    print(out_dir / "failure_mode_summary.md")
    print(out_dir / "failure_mode_summary_fixed.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
