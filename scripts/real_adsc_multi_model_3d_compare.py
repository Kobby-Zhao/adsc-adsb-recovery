from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train import load_config
from scripts.real_adsc_replay_eval import (
    _apply_conditional_rightstep2_fuse,
    _apply_right_edge_smoothing,
    _quality_flags_from_altitude,
    _build_fill_frame,
    _build_fill_intervals,
    _build_known_blocks,
    _extract_block_boundary_state,
    _parse_flight_meta,
    _predict_on_frame,
    _resample_minute,
)

def _postprocess_altitude_segment(
    alt_pred: pd.Series,
    left_anchor_alt: float,
    right_anchor_alt: float,
    max_vertical_rate_per_min: float = 180.0,
    spike_threshold: float = 260.0,
    baseline_blend: float = 0.35,
) -> pd.Series:
    x = alt_pred.astype(float).copy().reset_index(drop=True)
    n = len(x)
    if n <= 2:
        if n >= 1:
            x.iloc[0] = float(left_anchor_alt)
        if n >= 2:
            x.iloc[-1] = float(right_anchor_alt)
        return x
    x.iloc[0] = float(left_anchor_alt)
    x.iloc[-1] = float(right_anchor_alt)
    for _ in range(2):
        for i in range(1, n - 1):
            nbr = 0.5 * (x.iloc[i - 1] + x.iloc[i + 1])
            if abs(x.iloc[i] - nbr) > float(spike_threshold):
                x.iloc[i] = nbr
    m = float(max_vertical_rate_per_min)
    for i in range(1, n):
        lo = x.iloc[i - 1] - m
        hi = x.iloc[i - 1] + m
        x.iloc[i] = min(max(x.iloc[i], lo), hi)
    for i in range(n - 2, -1, -1):
        lo = x.iloc[i + 1] - m
        hi = x.iloc[i + 1] + m
        x.iloc[i] = min(max(x.iloc[i], lo), hi)
    lin = pd.Series(
        [left_anchor_alt + (right_anchor_alt - left_anchor_alt) * (k / (n - 1)) for k in range(n)],
        dtype=float,
    )
    b = float(baseline_blend)
    x = (1.0 - b) * x + b * lin
    y = x.copy()
    for i in range(1, n - 1):
        y.iloc[i] = 0.25 * x.iloc[i - 1] + 0.5 * x.iloc[i] + 0.25 * x.iloc[i + 1]
    x = y
    x.iloc[0] = float(left_anchor_alt)
    x.iloc[-1] = float(right_anchor_alt)
    return x


@dataclass
class ModelSpec:
    name: str
    config: str
    checkpoint: str


def _default_models() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="UniLSTM",
            config="configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_unilstm_baseline_24e.yaml",
            checkpoint="outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_unilstm_baseline_24e/best.pt",
        ),
        ModelSpec(
            name="BiLSTM_Baseline",
            config="configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_bilstm_baseline_24e.yaml",
            checkpoint="outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_bilstm_baseline_24e/best.pt",
        ),
        ModelSpec(
            name="Transformer",
            config="configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_transformer_baseline_24e.yaml",
            checkpoint="outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_transformer_baseline_24e/best.pt",
        ),
        ModelSpec(
            name="CNN+LSTM",
            config="configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_cnnlstm_baseline_24e.yaml",
            checkpoint="outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_cnnlstm_baseline_24e/best.pt",
        ),
        ModelSpec(
            name="Kalman-Filter",
            config="configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_kalman_filter_baseline_24e.yaml",
            checkpoint="outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_kalman_filter_baseline_24e/best.pt",
        ),
        ModelSpec(
            name="OurMethod",
            config="configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_proposed_24e.yaml",
            checkpoint="outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_proposed_24e/best.pt",
        ),
    ]


def _segment_adsb(adsb_all: pd.DataFrame, gap_break_min: float) -> list[pd.DataFrame]:
    if adsb_all.empty:
        return []
    x = adsb_all.sort_values("minute_ts").copy()
    x["dt_min"] = x["minute_ts"].diff().dt.total_seconds().div(60.0).fillna(0.0)
    x["seg_id"] = (x["dt_min"] >= gap_break_min).astype("int64").cumsum()
    return [g.copy() for _, g in x.groupby("seg_id")]


def _build_sample_fill_frames(
    fp: Path,
    min_adsc_anchors: int,
    min_gap_minutes: int,
    max_gap_minutes: int,
    context_minutes: int,
    gap_break_min: float,
    middle_anchor_only: bool,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, list[dict], pd.DataFrame] | None:
    fdf = pd.read_csv(fp)
    if "source" not in fdf.columns:
        return None
    fdf["source"] = fdf["source"].astype(str).str.lower()
    adsb_raw = fdf[fdf["source"].eq("adsb")].copy()
    adsc_raw = fdf[fdf["source"].eq("adsc")].copy()
    if len(adsc_raw) < 2:
        return None
    adsb_min = _resample_minute(adsb_raw)
    adsc_min = _resample_minute(adsc_raw)
    if len(adsc_min) < max(2, min_adsc_anchors):
        return None
    adsc_min = adsc_min.sort_values("minute_ts").drop_duplicates("minute_ts")
    # pick one representative gap per flight: largest valid consecutive ADS-C gap
    best = None
    for i in range(len(adsc_min) - 1):
        s = adsc_min.iloc[i]
        e = adsc_min.iloc[i + 1]
        st = pd.to_datetime(s["minute_ts"], utc=True)
        et = pd.to_datetime(e["minute_ts"], utc=True)
        gap_m = int((et - st).total_seconds() // 60)
        if gap_m < min_gap_minutes or gap_m > max_gap_minutes:
            continue
        if (best is None) or (gap_m > best["gap_minutes"]):
            best = {
                "i": i,
                "start_t": st,
                "end_t": et,
                "gap_minutes": gap_m,
                "start_anchor": (float(s["lat"]), float(s["lon"]), float(s["alt"])),
                "end_anchor": (float(e["lat"]), float(e["lon"]), float(e["alt"])),
            }
    if best is None:
        return None

    if len(adsb_min):
        adsb_min["minute_ts"] = pd.to_datetime(adsb_min["minute_ts"], utc=True)
    adsc_raw = adsc_raw.copy()
    adsc_raw["timestamp"] = pd.to_datetime(adsc_raw["timestamp"], utc=True)
    w_start = min(
        best["start_t"] - pd.Timedelta(minutes=240),
        adsb_min["minute_ts"].min() if len(adsb_min) else best["start_t"],
        adsc_raw["timestamp"].min(),
    )
    w_end = max(
        best["end_t"] + pd.Timedelta(minutes=240),
        adsb_min["minute_ts"].max() if len(adsb_min) else best["end_t"],
        adsc_raw["timestamp"].max(),
    )
    blocks = _build_known_blocks(adsb_min, adsc_raw, w_start, w_end, gap_break_min=gap_break_min)
    fills = _build_fill_intervals(blocks, min_gap_min=2.0)
    if middle_anchor_only:
        fills = [f for f in fills if f["left_block_type"] == "adsc_anchor" and f["right_block_type"] == "adsc_anchor"]
    if not fills:
        return None

    fill_frames: list[pd.DataFrame] = []
    fill_meta: list[dict] = []
    sample_id = f"{fp.stem}_a{best['i']:03d}"
    flight_id, flight_date = _parse_flight_meta(fp)
    for fi, f in enumerate(fills, start=1):
        left_block = blocks[f["left_block_idx"]]
        right_block = blocks[f["right_block_idx"]]
        left_state = _extract_block_boundary_state(left_block, "end")
        right_state = _extract_block_boundary_state(right_block, "start")
        fill_id = f"f{fi:02d}"
        fill_sid = f"{sample_id}__{fill_id}"
        frame, _, _ = _build_fill_frame(
            minute_all_adsb=adsb_min,
            fill_start=f["fill_start_time"],
            fill_end=f["fill_end_time"],
            left_state=left_state,
            right_state=right_state,
            task_type="pure_adsc",
            context_minutes=context_minutes,
        )
        if frame.empty:
            continue
        frame["sample_id"] = fill_sid
        frame["flight_id"] = flight_id
        fill_frames.append(frame)
        fill_meta.append(
            {
                "fill_id": fill_id,
                "fill_sid": fill_sid,
                "fill_start_time": f["fill_start_time"],
                "fill_end_time": f["fill_end_time"],
                "fill_minutes": f["fill_minutes"],
                "left_block_type": f["left_block_type"],
                "right_block_type": f["right_block_type"],
                "left_boundary_alt": float(left_state["alt"]),
                "right_boundary_alt": float(right_state["alt"]),
            }
        )
    if not fill_frames:
        return None

    meta = {
        "sample_id": sample_id,
        "flight_id": flight_id,
        "flight_date": flight_date,
        "gap_minutes": best["gap_minutes"],
    }
    return meta, adsb_min, adsc_raw, fill_meta, pd.concat(fill_frames, ignore_index=True)


def _plot_one_3d(
    out_png: Path,
    meta: dict,
    adsb_min: pd.DataFrame,
    adsc_raw: pd.DataFrame,
    fill_meta: list[dict],
    pred_by_model: dict[str, pd.DataFrame],
    frame_all: pd.DataFrame,
    gap_break_min: float,
    hide_adsb_known: bool = False,
    ourmethod_postprocess: bool = False,
    global_postprocess: bool = False,
    max_vertical_rate: float = 180.0,
    spike_threshold: float = 260.0,
    baseline_blend: float = 0.35,
) -> None:
    colors = {
        "UniLSTM": "#1f77b4",
        "BiLSTM_Baseline": "#2ca02c",
        "Transformer": "#ff7f0e",
        "OurMethod_BiLSTM": "#d62728",
        "Exp11_fixlogic": "#8c564b",
        "AltBaseResidualV1_full": "#9467bd",
        "BiLSTM_Alt_DMS_Refiner_V2": "#8c564b",
        "BiLSTM_Alt_DMS_Refiner_V2_1": "#17becf",
        "BiLSTM_Alt_DMS_Refiner_V3": "#bcbd22",
        "Exp4_fix2_raw": "#7f7f7f",
        "Exp8B_right_local_band": "#e377c2",
        "Exp9A_conditional_fuse": "#111111",
        "Exp9B_conditional_fuse_2cond": "#8c2d04",
    }
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Known ADS-B segments (gray, broken by real time gaps)
    if not hide_adsb_known:
        for i, seg in enumerate(_segment_adsb(adsb_min, gap_break_min)):
            ax.plot(seg["lon"], seg["lat"], seg["alt"], color="#7f7f7f", lw=1.0, alpha=0.8, label="ADS-B known" if i == 0 else None)

    # ADS-C anchors (purple)
    if len(adsc_raw):
        x = adsc_raw.copy()
        x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
        ax.scatter(x["lon"], x["lat"], x["baroaltitude"], color="#7b3294", s=18, alpha=0.9, label="ADS-C anchors")

    # Recovered fill segments by model
    for model_name, pred in pred_by_model.items():
        c = colors.get(model_name, None)
        first = True
        sort_col = "time"
        if "minute_ts" in pred.columns:
            sort_col = "minute_ts"
        elif "timestamp" in pred.columns:
            sort_col = "timestamp"
        fref = frame_all[["sample_id", "minute_ts", "obs_mask", "obs_lat", "obs_lon", "obs_alt"]].copy()
        fref["minute_ts"] = pd.to_datetime(fref["minute_ts"], utc=True)
        for f in fill_meta:
            seg = pred[pred["sample_id"].astype(str).eq(f["fill_sid"])].sort_values(sort_col)
            if seg.empty:
                continue
            seg = seg.copy()
            seg[sort_col] = pd.to_datetime(seg[sort_col], utc=True)
            # Display-only hard anchor consistency to avoid perspective/rounding misread.
            seg = seg.merge(
                fref[fref["sample_id"].astype(str).eq(f["fill_sid"])],
                how="left",
                left_on=["sample_id", sort_col],
                right_on=["sample_id", "minute_ts"],
                suffixes=("", "_ref"),
            )
            if "obs_mask_ref" in seg.columns:
                m = seg["obs_mask_ref"].fillna(0).astype(float) > 0.5
                seg.loc[m, "pred_lat"] = seg.loc[m, "obs_lat"]
                seg.loc[m, "pred_lon"] = seg.loc[m, "obs_lon"]
                seg.loc[m, "pred_alt"] = seg.loc[m, "obs_alt"]
            if model_name == "Exp8B_right_local_band":
                right_alt = float(f.get("right_boundary_alt", seg["pred_alt"].iloc[-1]))
                sm, _ = _apply_right_edge_smoothing(
                    alt_final=seg["pred_alt"].to_numpy(dtype=float),
                    right_boundary_alt=right_alt,
                    enabled=True,
                    mode="right_local_band",
                    steps=2,
                    blend_betas=[0.2, 0.5],
                    right_local_band=200.0,
                )
                seg["pred_alt"] = sm
            if model_name in {"Exp9A_conditional_fuse", "Exp9B_conditional_fuse_2cond"}:
                right_alt = float(f.get("right_boundary_alt", seg["pred_alt"].iloc[-1]))
                seg_bucket = "short" if float(f.get("fill_minutes", 0.0)) <= 15.0 else ("medium" if float(f.get("fill_minutes", 0.0)) <= 60.0 else "long")
                seg_pattern = (
                    "two_anchor"
                    if (str(f.get("left_block_type", "")) == "adsc_anchor" and str(f.get("right_block_type", "")) == "adsc_anchor")
                    else ("asymmetric" if ("adsc_anchor" in {str(f.get("left_block_type", "")), str(f.get("right_block_type", ""))}) else "sparse_context")
                )
                sm, _, _, _, _ = _apply_conditional_rightstep2_fuse(
                    seg["pred_alt"].to_numpy(dtype=float),
                    segment_bucket=seg_bucket,
                    anchor_pattern=seg_pattern,
                    right_boundary_alt=right_alt,
                    enabled=True,
                    target_bucket="medium",
                    target_pattern="two_anchor",
                    tau_jump=200.0,
                    mode="local_interp",
                    fuse_lambda=0.5,
                    use_second_condition=(model_name == "Exp9B_conditional_fuse_2cond"),
                    tau_curve=200.0,
                    right_local_band=200.0,
                )
                seg["pred_alt"] = sm
            apply_pp = bool(global_postprocess) or (bool(ourmethod_postprocess) and model_name == "OurMethod_BiLSTM")
            if apply_pp:
                left_obs = float(seg.iloc[0]["obs_alt"]) if "obs_alt" in seg.columns else float(seg.iloc[0]["pred_alt"])
                right_obs = float(seg.iloc[-1]["obs_alt"]) if "obs_alt" in seg.columns else float(seg.iloc[-1]["pred_alt"])
                seg["pred_alt"] = _postprocess_altitude_segment(
                    alt_pred=seg["pred_alt"],
                    left_anchor_alt=left_obs,
                    right_anchor_alt=right_obs,
                    max_vertical_rate_per_min=float(max_vertical_rate),
                    spike_threshold=float(spike_threshold),
                    baseline_blend=float(baseline_blend),
                ).values
            ax.plot(
                seg["pred_lon"],
                seg["pred_lat"],
                seg["pred_alt"],
                lw=2.0,
                alpha=0.95,
                color=c,
                label=model_name if first else None,
            )
            first = False

    ax.set_xlabel("Lon")
    ax.set_ylabel("Lat")
    ax.set_zlabel("Altitude")
    ax.set_title(
        f"{meta['sample_id']} | flight={meta['flight_id']} | gap={meta['gap_minutes']}m | models_compare_3d",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _plot_one_alt_2d(
    out_png: Path,
    meta: dict,
    adsc_raw: pd.DataFrame,
    fill_meta: list[dict],
    pred_by_model: dict[str, pd.DataFrame],
    frame_all: pd.DataFrame,
    ourmethod_postprocess: bool = False,
    global_postprocess: bool = False,
    max_vertical_rate: float = 180.0,
    spike_threshold: float = 260.0,
    baseline_blend: float = 0.35,
) -> None:
    colors = {
        "UniLSTM": "#1f77b4",
        "BiLSTM_Baseline": "#2ca02c",
        "Transformer": "#ff7f0e",
        "OurMethod_BiLSTM": "#d62728",
        "Exp11_fixlogic": "#8c564b",
        "AltBaseResidualV1_full": "#9467bd",
        "BiLSTM_Alt_DMS_Refiner_V2": "#8c564b",
        "BiLSTM_Alt_DMS_Refiner_V2_1": "#17becf",
        "BiLSTM_Alt_DMS_Refiner_V3": "#bcbd22",
        "Exp4_fix2_raw": "#7f7f7f",
        "Exp8B_right_local_band": "#e377c2",
        "Exp9A_conditional_fuse": "#111111",
        "Exp9B_conditional_fuse_2cond": "#8c2d04",
    }
    fig, ax = plt.subplots(figsize=(11, 4))

    # ADS-C anchors
    if len(adsc_raw):
        x = adsc_raw.copy()
        x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
        x = x.sort_values("timestamp")
        ax.scatter(
            x["timestamp"],
            x["baroaltitude"],
            color="#7b3294",
            s=24,
            alpha=0.9,
            label="ADS-C anchors",
            zorder=5,
        )

    # Model recovered altitude (plot by fill segment to preserve discontinuities)
    for model_name, pred in pred_by_model.items():
        c = colors.get(model_name, None)
        first = True
        sort_col = "time"
        if "minute_ts" in pred.columns:
            sort_col = "minute_ts"
        elif "timestamp" in pred.columns:
            sort_col = "timestamp"
        fref = frame_all[["sample_id", "minute_ts", "obs_mask", "obs_alt"]].copy()
        fref["minute_ts"] = pd.to_datetime(fref["minute_ts"], utc=True)
        for f in fill_meta:
            seg = pred[pred["sample_id"].astype(str).eq(f["fill_sid"])].copy()
            if seg.empty:
                continue
            seg[sort_col] = pd.to_datetime(seg[sort_col], utc=True)
            seg = seg.sort_values(sort_col)
            seg = seg.merge(
                fref[fref["sample_id"].astype(str).eq(f["fill_sid"])],
                how="left",
                left_on=["sample_id", sort_col],
                right_on=["sample_id", "minute_ts"],
                suffixes=("", "_ref"),
            )
            if "obs_mask_ref" in seg.columns:
                m = seg["obs_mask_ref"].fillna(0).astype(float) > 0.5
                seg.loc[m, "pred_alt"] = seg.loc[m, "obs_alt"]
            if model_name == "Exp8B_right_local_band":
                right_alt = float(f.get("right_boundary_alt", seg["pred_alt"].iloc[-1]))
                sm, _ = _apply_right_edge_smoothing(
                    alt_final=seg["pred_alt"].to_numpy(dtype=float),
                    right_boundary_alt=right_alt,
                    enabled=True,
                    mode="right_local_band",
                    steps=2,
                    blend_betas=[0.2, 0.5],
                    right_local_band=200.0,
                )
                seg["pred_alt"] = sm
            if model_name in {"Exp9A_conditional_fuse", "Exp9B_conditional_fuse_2cond"}:
                right_alt = float(f.get("right_boundary_alt", seg["pred_alt"].iloc[-1]))
                seg_bucket = "short" if float(f.get("fill_minutes", 0.0)) <= 15.0 else ("medium" if float(f.get("fill_minutes", 0.0)) <= 60.0 else "long")
                seg_pattern = (
                    "two_anchor"
                    if (str(f.get("left_block_type", "")) == "adsc_anchor" and str(f.get("right_block_type", "")) == "adsc_anchor")
                    else ("asymmetric" if ("adsc_anchor" in {str(f.get("left_block_type", "")), str(f.get("right_block_type", ""))}) else "sparse_context")
                )
                sm, _, _, _, _ = _apply_conditional_rightstep2_fuse(
                    seg["pred_alt"].to_numpy(dtype=float),
                    segment_bucket=seg_bucket,
                    anchor_pattern=seg_pattern,
                    right_boundary_alt=right_alt,
                    enabled=True,
                    target_bucket="medium",
                    target_pattern="two_anchor",
                    tau_jump=200.0,
                    mode="local_interp",
                    fuse_lambda=0.5,
                    use_second_condition=(model_name == "Exp9B_conditional_fuse_2cond"),
                    tau_curve=200.0,
                    right_local_band=200.0,
                )
                seg["pred_alt"] = sm
            apply_pp = bool(global_postprocess) or (bool(ourmethod_postprocess) and model_name == "OurMethod_BiLSTM")
            if apply_pp:
                left_obs = float(seg.iloc[0]["obs_alt"]) if "obs_alt" in seg.columns else float(seg.iloc[0]["pred_alt"])
                right_obs = float(seg.iloc[-1]["obs_alt"]) if "obs_alt" in seg.columns else float(seg.iloc[-1]["pred_alt"])
                seg["pred_alt"] = _postprocess_altitude_segment(
                    alt_pred=seg["pred_alt"],
                    left_anchor_alt=left_obs,
                    right_anchor_alt=right_obs,
                    max_vertical_rate_per_min=float(max_vertical_rate),
                    spike_threshold=float(spike_threshold),
                    baseline_blend=float(baseline_blend),
                ).values
            ax.plot(
                seg[sort_col],
                seg["pred_alt"],
                lw=2.0,
                alpha=0.95,
                color=c,
                label=model_name if first else None,
            )
            first = False

    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Altitude")
    ax.set_title(
        f"{meta['sample_id']} | flight={meta['flight_id']} | gap={meta['gap_minutes']}m | altitude_2d_compare",
        fontsize=9,
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _plot_one_alt_2d_detail(
    out_png: Path,
    meta: dict,
    adsc_raw: pd.DataFrame,
    fill_meta: list[dict],
    pred_by_model: dict[str, pd.DataFrame],
    frame_all: pd.DataFrame,
    ourmethod_postprocess: bool = False,
    global_postprocess: bool = False,
    max_vertical_rate: float = 180.0,
    spike_threshold: float = 260.0,
    baseline_blend: float = 0.35,
    detail_minutes: int = 8,
    use_nonuniform_y: bool = True,
    y_warp_scale: float = 180.0,
) -> None:
    colors = {
        "UniLSTM": "#1f77b4",
        "BiLSTM_Baseline": "#2ca02c",
        "Transformer": "#ff7f0e",
        "OurMethod_BiLSTM": "#d62728",
        "Exp11_fixlogic": "#8c564b",
        "AltBaseResidualV1_full": "#9467bd",
        "BiLSTM_Alt_DMS_Refiner_V2": "#8c564b",
        "BiLSTM_Alt_DMS_Refiner_V2_1": "#17becf",
        "BiLSTM_Alt_DMS_Refiner_V3": "#bcbd22",
        "Exp4_fix2_raw": "#7f7f7f",
        "Exp8B_right_local_band": "#e377c2",
        "Exp9A_conditional_fuse": "#111111",
        "Exp9B_conditional_fuse_2cond": "#8c2d04",
    }
    fref = frame_all[["sample_id", "minute_ts", "obs_mask", "obs_alt"]].copy()
    fref["minute_ts"] = pd.to_datetime(fref["minute_ts"], utc=True)

    first_fill_start = pd.to_datetime(min(f["fill_start_time"] for f in fill_meta), utc=True)
    last_fill_end = pd.to_datetime(max(f["fill_end_time"] for f in fill_meta), utc=True)
    left_lo = first_fill_start - pd.Timedelta(minutes=int(detail_minutes))
    left_hi = first_fill_start + pd.Timedelta(minutes=int(detail_minutes))
    right_lo = last_fill_end - pd.Timedelta(minutes=int(detail_minutes))
    right_hi = last_fill_end + pd.Timedelta(minutes=int(detail_minutes))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 4.2), sharey=True)
    windows = [(ax_l, left_lo, left_hi, "Left edge zoom"), (ax_r, right_lo, right_hi, "Right edge zoom")]

    # Non-uniform Y metric: dense around local anchor center, coarse for far values.
    if use_nonuniform_y:
        left_center = float(np.mean([f.get("left_boundary_alt", 0.0) for f in fill_meta]))
        right_center = float(np.mean([f.get("right_boundary_alt", 0.0) for f in fill_meta]))
        centers = [left_center, right_center]
        k = max(float(y_warp_scale), 1e-6)
        for (ax, _, _, _), c in zip(windows, centers):
            def _forward(y, cc=c, kk=k):
                arr = np.asarray(y, dtype=float)
                return np.arcsinh((arr - cc) / kk)
            def _inverse(z, cc=c, kk=k):
                arr = np.asarray(z, dtype=float)
                return cc + kk * np.sinh(arr)
            ax.set_yscale("function", functions=(_forward, _inverse))

    if len(adsc_raw):
        x = adsc_raw.copy()
        x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
        x = x.sort_values("timestamp")
        for ax, w_lo, w_hi, _ in windows:
            xs = x[x["timestamp"].between(w_lo, w_hi)]
            if len(xs):
                ax.scatter(xs["timestamp"], xs["baroaltitude"], color="#7b3294", s=32, alpha=0.95, label="ADS-C anchors", zorder=6)

    for model_name, pred in pred_by_model.items():
        c = colors.get(model_name, None)
        sort_col = "time"
        if "minute_ts" in pred.columns:
            sort_col = "minute_ts"
        elif "timestamp" in pred.columns:
            sort_col = "timestamp"
        first = True
        for f in fill_meta:
            seg = pred[pred["sample_id"].astype(str).eq(f["fill_sid"])].copy()
            if seg.empty:
                continue
            seg[sort_col] = pd.to_datetime(seg[sort_col], utc=True)
            seg = seg.sort_values(sort_col)
            seg = seg.merge(
                fref[fref["sample_id"].astype(str).eq(f["fill_sid"])],
                how="left",
                left_on=["sample_id", sort_col],
                right_on=["sample_id", "minute_ts"],
                suffixes=("", "_ref"),
            )
            if "obs_mask_ref" in seg.columns:
                m = seg["obs_mask_ref"].fillna(0).astype(float) > 0.5
                seg.loc[m, "pred_alt"] = seg.loc[m, "obs_alt"]
            if model_name in {"Exp9A_conditional_fuse", "Exp9B_conditional_fuse_2cond"}:
                right_alt = float(f.get("right_boundary_alt", seg["pred_alt"].iloc[-1]))
                seg_bucket = "short" if float(f.get("fill_minutes", 0.0)) <= 15.0 else ("medium" if float(f.get("fill_minutes", 0.0)) <= 60.0 else "long")
                seg_pattern = (
                    "two_anchor"
                    if (str(f.get("left_block_type", "")) == "adsc_anchor" and str(f.get("right_block_type", "")) == "adsc_anchor")
                    else ("asymmetric" if ("adsc_anchor" in {str(f.get("left_block_type", "")), str(f.get("right_block_type", ""))}) else "sparse_context")
                )
                sm, _, _, _, _ = _apply_conditional_rightstep2_fuse(
                    seg["pred_alt"].to_numpy(dtype=float),
                    segment_bucket=seg_bucket,
                    anchor_pattern=seg_pattern,
                    right_boundary_alt=right_alt,
                    enabled=True,
                    target_bucket="medium",
                    target_pattern="two_anchor",
                    tau_jump=200.0,
                    mode="local_interp",
                    fuse_lambda=0.5,
                    use_second_condition=(model_name == "Exp9B_conditional_fuse_2cond"),
                    tau_curve=200.0,
                    right_local_band=200.0,
                )
                seg["pred_alt"] = sm
            apply_pp = bool(global_postprocess) or (bool(ourmethod_postprocess) and model_name == "OurMethod_BiLSTM")
            if apply_pp:
                left_obs = float(seg.iloc[0]["obs_alt"]) if "obs_alt" in seg.columns else float(seg.iloc[0]["pred_alt"])
                right_obs = float(seg.iloc[-1]["obs_alt"]) if "obs_alt" in seg.columns else float(seg.iloc[-1]["pred_alt"])
                seg["pred_alt"] = _postprocess_altitude_segment(
                    alt_pred=seg["pred_alt"],
                    left_anchor_alt=left_obs,
                    right_anchor_alt=right_obs,
                    max_vertical_rate_per_min=float(max_vertical_rate),
                    spike_threshold=float(spike_threshold),
                    baseline_blend=float(baseline_blend),
                ).values
            for ax, w_lo, w_hi, title in windows:
                s = seg[seg[sort_col].between(w_lo, w_hi)]
                if s.empty:
                    continue
                ax.plot(
                    s[sort_col],
                    s["pred_alt"],
                    lw=2.4,
                    marker="o",
                    markersize=3.2,
                    alpha=0.98,
                    color=c,
                    label=model_name if first else None,
                )
                ax.set_title(title, fontsize=9)
            first = False

    for ax, w_lo, w_hi, _ in windows:
        ax.set_xlim(w_lo, w_hi)
        ax.grid(alpha=0.25)
        ax.set_xlabel("Time (UTC)")
    ax_l.set_ylabel("Altitude")

    handles, labels = ax_l.get_legend_handles_labels()
    seen = set()
    hh, ll = [], []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        hh.append(h)
        ll.append(l)
    ax_l.legend(hh, ll, fontsize=7, loc="best")

    fig.suptitle(
        f"{meta['sample_id']} | flight={meta['flight_id']} | gap={meta['gap_minutes']}m | altitude_local_detail",
        fontsize=10,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def _plot_one_alt_error_points(
    out_png: Path,
    meta: dict,
    fill_meta: list[dict],
    pred_by_model: dict[str, pd.DataFrame],
    frame_all: pd.DataFrame,
    ourmethod_postprocess: bool = False,
    global_postprocess: bool = False,
    max_vertical_rate: float = 180.0,
    spike_threshold: float = 260.0,
    baseline_blend: float = 0.35,
) -> None:
    colors = {
        "UniLSTM": "#1f77b4",
        "BiLSTM_Baseline": "#2ca02c",
        "Transformer": "#ff7f0e",
        "OurMethod_BiLSTM": "#d62728",
        "Exp11_fixlogic": "#8c564b",
        "AltBaseResidualV1_full": "#9467bd",
        "BiLSTM_Alt_DMS_Refiner_V2": "#8c564b",
        "BiLSTM_Alt_DMS_Refiner_V2_1": "#17becf",
        "BiLSTM_Alt_DMS_Refiner_V3": "#bcbd22",
        "Exp4_fix2_raw": "#7f7f7f",
        "Exp8B_right_local_band": "#e377c2",
        "Exp9A_conditional_fuse": "#111111",
        "Exp9B_conditional_fuse_2cond": "#8c2d04",
    }
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor("#f2f2f2")
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.0)

    fref = frame_all[["sample_id", "minute_ts", "obs_mask", "obs_alt"]].copy()
    fref["minute_ts"] = pd.to_datetime(fref["minute_ts"], utc=True)

    for model_name, pred in pred_by_model.items():
        c = colors.get(model_name, None)
        sort_col = "time"
        if "minute_ts" in pred.columns:
            sort_col = "minute_ts"
        elif "timestamp" in pred.columns:
            sort_col = "timestamp"

        seg_errors = []
        for f in fill_meta:
            seg = pred[pred["sample_id"].astype(str).eq(f["fill_sid"])].copy()
            if seg.empty:
                continue
            seg[sort_col] = pd.to_datetime(seg[sort_col], utc=True)
            seg = seg.sort_values(sort_col)
            seg = seg.merge(
                fref[fref["sample_id"].astype(str).eq(f["fill_sid"])],
                how="left",
                left_on=["sample_id", sort_col],
                right_on=["sample_id", "minute_ts"],
                suffixes=("", "_ref"),
            )
            if "obs_mask_ref" in seg.columns:
                m = seg["obs_mask_ref"].fillna(0).astype(float) > 0.5
                seg.loc[m, "pred_alt"] = seg.loc[m, "obs_alt"]
            if model_name == "Exp8B_right_local_band":
                right_alt = float(f.get("right_boundary_alt", seg["pred_alt"].iloc[-1]))
                sm, _ = _apply_right_edge_smoothing(
                    alt_final=seg["pred_alt"].to_numpy(dtype=float),
                    right_boundary_alt=right_alt,
                    enabled=True,
                    mode="right_local_band",
                    steps=2,
                    blend_betas=[0.2, 0.5],
                    right_local_band=200.0,
                )
                seg["pred_alt"] = sm
            if model_name in {"Exp9A_conditional_fuse", "Exp9B_conditional_fuse_2cond"}:
                right_alt = float(f.get("right_boundary_alt", seg["pred_alt"].iloc[-1]))
                seg_bucket = "short" if float(f.get("fill_minutes", 0.0)) <= 15.0 else ("medium" if float(f.get("fill_minutes", 0.0)) <= 60.0 else "long")
                seg_pattern = (
                    "two_anchor"
                    if (str(f.get("left_block_type", "")) == "adsc_anchor" and str(f.get("right_block_type", "")) == "adsc_anchor")
                    else ("asymmetric" if ("adsc_anchor" in {str(f.get("left_block_type", "")), str(f.get("right_block_type", ""))}) else "sparse_context")
                )
                sm, _, _, _, _ = _apply_conditional_rightstep2_fuse(
                    seg["pred_alt"].to_numpy(dtype=float),
                    segment_bucket=seg_bucket,
                    anchor_pattern=seg_pattern,
                    right_boundary_alt=right_alt,
                    enabled=True,
                    target_bucket="medium",
                    target_pattern="two_anchor",
                    tau_jump=200.0,
                    mode="local_interp",
                    fuse_lambda=0.5,
                    use_second_condition=(model_name == "Exp9B_conditional_fuse_2cond"),
                    tau_curve=200.0,
                    right_local_band=200.0,
                )
                seg["pred_alt"] = sm
            apply_pp = bool(global_postprocess) or (bool(ourmethod_postprocess) and model_name == "OurMethod_BiLSTM")
            if apply_pp:
                left_obs = float(seg.iloc[0]["obs_alt"]) if "obs_alt" in seg.columns else float(seg.iloc[0]["pred_alt"])
                right_obs = float(seg.iloc[-1]["obs_alt"]) if "obs_alt" in seg.columns else float(seg.iloc[-1]["pred_alt"])
                seg["pred_alt"] = _postprocess_altitude_segment(
                    alt_pred=seg["pred_alt"],
                    left_anchor_alt=left_obs,
                    right_anchor_alt=right_obs,
                    max_vertical_rate_per_min=float(max_vertical_rate),
                    spike_threshold=float(spike_threshold),
                    baseline_blend=float(baseline_blend),
                ).values

            mm = (seg["obs_mask_ref"].fillna(0).astype(float) > 0.5) & seg["obs_alt"].notna()
            ss = seg.loc[mm, ["pred_alt", "obs_alt"]].copy()
            if not ss.empty:
                seg_errors.extend((ss["pred_alt"] - ss["obs_alt"]).astype(float).tolist())

        if seg_errors:
            xs = np.arange(len(seg_errors))
            ax.scatter(xs, seg_errors, s=14, alpha=0.9, color=c, label=model_name, edgecolors="none")

    ax.axhline(0.0, color="#444444", lw=1.1, ls="--", alpha=0.9)
    ax.set_xlabel("Prediction points")
    ax.set_ylabel("Altitude Error")
    ax.set_title(
        f"{meta['sample_id']} | flight={meta['flight_id']} | gap={meta['gap_minutes']}m | altitude_error_points",
        fontsize=9,
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="lower left", frameon=True, framealpha=0.95)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Real ADS-C replay multi-model 3D compare")
    p.add_argument("--adsc-dir", default="outputs/adsc/adsc_flight_point/2026-01-13-2110/all")
    p.add_argument("--out-dir", default="outputs/runs/real_adsc_model_compare_3d_20260402")
    p.add_argument("--max-flights", type=int, default=10)
    p.add_argument("--min-adsc-anchors", type=int, default=6)
    p.add_argument("--min-gap-minutes", type=int, default=5)
    p.add_argument("--max-gap-minutes", type=int, default=120)
    p.add_argument("--context-minutes", type=int, default=5)
    p.add_argument("--gap-break-min", type=float, default=10.0)
    p.add_argument(
        "--middle-anchor-only",
        type=int,
        default=1,
        help="1: only recover ADS-C anchor->anchor middle gaps (basic validation mode), 0: recover all fill intervals.",
    )
    p.add_argument(
        "--hide-adsb-known",
        type=int,
        default=0,
        help="1: hide ADS-B known track in plot (show only ADS-C anchors + recovered model curves).",
    )
    p.add_argument("--ourmethod-postprocess", type=int, default=0, help="1: apply spike-suppression postprocess only to OurMethod_BiLSTM")
    p.add_argument("--global-postprocess", type=int, default=0, help="1: apply the same spike-suppression postprocess to all models")
    p.add_argument("--max-vertical-rate", type=float, default=180.0)
    p.add_argument("--spike-threshold", type=float, default=260.0)
    p.add_argument("--baseline-blend", type=float, default=0.35)
    p.add_argument(
        "--exclude-models",
        type=str,
        default="",
        help="Comma-separated model names to exclude from plotting/evaluation.",
    )
    p.add_argument("--detail-minutes", type=int, default=8, help="Half-window minutes for local detail altitude plots.")
    p.add_argument("--detail-nonuniform-y", type=int, default=1, help="1: use non-uniform Y metric (dense near anchors, coarse far away).")
    p.add_argument("--detail-y-warp-scale", type=float, default=180.0, help="Y warp scale (in altitude unit). Smaller=more local detail.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    adsc_dir = Path(args.adsc_dir)
    out_dir = Path(args.out_dir)
    plot_dir = out_dir / "plots_3d_per_flight"
    plot_alt_dir = out_dir / "plots_alt_2d_per_flight"
    plot_alt_error_dir = out_dir / "plots_alt_error_points_per_flight"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_alt_dir.mkdir(parents=True, exist_ok=True)
    plot_alt_error_dir.mkdir(parents=True, exist_ok=True)

    model_specs = _default_models()
    exclude_names = {x.strip() for x in str(args.exclude_models).split(",") if x.strip()}
    if exclude_names:
        model_specs = [m for m in model_specs if m.name not in exclude_names]
        if not model_specs:
            raise RuntimeError(f"All models excluded by --exclude-models={sorted(exclude_names)}")
    for ms in model_specs:
        if not Path(ms.config).exists():
            raise RuntimeError(f"config not found: {ms.config}")
        if not Path(ms.checkpoint).exists():
            raise RuntimeError(f"checkpoint not found: {ms.checkpoint}")

    files = sorted(adsc_dir.glob("*.csv"))
    selected_rows = []
    metric_rows = []
    n_done = 0
    for fp in files:
        if n_done >= int(args.max_flights):
            break
        built = _build_sample_fill_frames(
            fp=fp,
            min_adsc_anchors=int(args.min_adsc_anchors),
            min_gap_minutes=int(args.min_gap_minutes),
            max_gap_minutes=int(args.max_gap_minutes),
            context_minutes=int(args.context_minutes),
            gap_break_min=float(args.gap_break_min),
            middle_anchor_only=bool(int(args.middle_anchor_only)),
        )
        if built is None:
            continue
        meta, adsb_min, adsc_raw, fill_meta, frame_all = built
        pred_by_model: dict[str, pd.DataFrame] = {}
        ok = True
        for ms in model_specs:
            cfg = load_config(ms.config)
            pred = _predict_on_frame(cfg=cfg, checkpoint=Path(ms.checkpoint), frame=frame_all, pred_key="pred_pos")
            if pred.empty:
                ok = False
                break
            pred_by_model[ms.name] = pred
        if "Exp4_fix2_raw" in pred_by_model:
            if "Exp8B_right_local_band" not in exclude_names:
                pred_by_model["Exp8B_right_local_band"] = pred_by_model["Exp4_fix2_raw"].copy()
            if "Exp9A_conditional_fuse" not in exclude_names:
                pred_by_model["Exp9A_conditional_fuse"] = pred_by_model["Exp4_fix2_raw"].copy()
            if "Exp9B_conditional_fuse_2cond" not in exclude_names:
                pred_by_model["Exp9B_conditional_fuse_2cond"] = pred_by_model["Exp4_fix2_raw"].copy()
        if not ok:
            continue

        # per-model metrics on reconstructed segments (proxy metrics for real ADS-C replay)
        fref = frame_all[["sample_id", "minute_ts", "obs_mask", "obs_alt"]].copy()
        fref["minute_ts"] = pd.to_datetime(fref["minute_ts"], utc=True)
        for model_name, pred in pred_by_model.items():
            sort_col = "time"
            if "minute_ts" in pred.columns:
                sort_col = "minute_ts"
            elif "timestamp" in pred.columns:
                sort_col = "timestamp"
            for f in fill_meta:
                seg = pred[pred["sample_id"].astype(str).eq(f["fill_sid"])].copy()
                if seg.empty:
                    continue
                seg[sort_col] = pd.to_datetime(seg[sort_col], utc=True)
                seg = seg.sort_values(sort_col)
                seg = seg.merge(
                    fref[fref["sample_id"].astype(str).eq(f["fill_sid"])],
                    how="left",
                    left_on=["sample_id", sort_col],
                    right_on=["sample_id", "minute_ts"],
                    suffixes=("", "_ref"),
                )
                if "obs_mask_ref" in seg.columns:
                    m = seg["obs_mask_ref"].fillna(0).astype(float) > 0.5
                    seg.loc[m, "pred_alt"] = seg.loc[m, "obs_alt"]
                if model_name in {"Exp9A_conditional_fuse", "Exp9B_conditional_fuse_2cond"}:
                    right_alt = float(f.get("right_boundary_alt", seg["pred_alt"].iloc[-1]))
                    seg_bucket = "short" if float(f.get("fill_minutes", 0.0)) <= 15.0 else ("medium" if float(f.get("fill_minutes", 0.0)) <= 60.0 else "long")
                    seg_pattern = (
                        "two_anchor"
                        if (str(f.get("left_block_type", "")) == "adsc_anchor" and str(f.get("right_block_type", "")) == "adsc_anchor")
                        else ("asymmetric" if ("adsc_anchor" in {str(f.get("left_block_type", "")), str(f.get("right_block_type", ""))}) else "sparse_context")
                    )
                    sm, _, _, _, _ = _apply_conditional_rightstep2_fuse(
                        seg["pred_alt"].to_numpy(dtype=float),
                        segment_bucket=seg_bucket,
                        anchor_pattern=seg_pattern,
                        right_boundary_alt=right_alt,
                        enabled=True,
                        target_bucket="medium",
                        target_pattern="two_anchor",
                        tau_jump=200.0,
                        mode="local_interp",
                        fuse_lambda=0.5,
                        use_second_condition=(model_name == "Exp9B_conditional_fuse_2cond"),
                        tau_curve=200.0,
                        right_local_band=200.0,
                    )
                    seg["pred_alt"] = sm
                apply_pp = bool(int(args.global_postprocess)) or (bool(int(args.ourmethod_postprocess)) and model_name == "OurMethod_BiLSTM")
                if apply_pp:
                    left_obs = float(seg.iloc[0]["obs_alt"]) if "obs_alt" in seg.columns else float(seg.iloc[0]["pred_alt"])
                    right_obs = float(seg.iloc[-1]["obs_alt"]) if "obs_alt" in seg.columns else float(seg.iloc[-1]["pred_alt"])
                    seg["pred_alt"] = _postprocess_altitude_segment(
                        alt_pred=seg["pred_alt"],
                        left_anchor_alt=left_obs,
                        right_anchor_alt=right_obs,
                        max_vertical_rate_per_min=float(args.max_vertical_rate),
                        spike_threshold=float(args.spike_threshold),
                        baseline_blend=float(args.baseline_blend),
                    ).values

                q = _quality_flags_from_altitude(
                    alt=seg["pred_alt"].to_numpy(dtype=float),
                    left_boundary_alt=float(f.get("left_boundary_alt", seg["pred_alt"].iloc[0])),
                    right_boundary_alt=float(f.get("right_boundary_alt", seg["pred_alt"].iloc[-1])),
                )
                anchor_mae = float("nan")
                anchor_rmse = float("nan")
                if "obs_mask_ref" in seg.columns and "obs_alt" in seg.columns:
                    am = seg["obs_mask_ref"].fillna(0).astype(float) > 0.5
                    if am.any():
                        err = (seg.loc[am, "pred_alt"].to_numpy(dtype=float) - seg.loc[am, "obs_alt"].to_numpy(dtype=float))
                        anchor_mae = float(np.mean(np.abs(err)))
                        anchor_rmse = float(np.sqrt(np.mean(err ** 2)))
                metric_rows.append(
                    {
                        "model_name": model_name,
                        "sample_id": meta["sample_id"],
                        "fill_id": f["fill_id"],
                        "fill_minutes": float(f.get("fill_minutes", 0.0)),
                        "postprocess_applied": float(1.0 if apply_pp else 0.0),
                        "overshoot_flag": float(1.0 if q.get("overshoot_flag", False) else 0.0),
                        "edge_spike_flag": float(1.0 if q.get("edge_spike_flag", False) else 0.0),
                        "abnormal_flag": float(1.0 if q.get("abnormal_flag", False) else 0.0),
                        "warn_flag": float(1.0 if q.get("warn_flag", False) else 0.0),
                        "keep_flag": float(1.0 if q.get("keep_flag", False) else 0.0),
                        "max_vertical_rate_inside": float(q.get("max_vertical_rate_inside", float("nan"))),
                        "anchor_alt_mae": anchor_mae,
                        "anchor_alt_rmse": anchor_rmse,
                    }
                )

        png = plot_dir / f"{n_done+1:02d}_{meta['sample_id']}_{meta['flight_id']}_models_3d.png"
        _plot_one_3d(
            out_png=png,
            meta=meta,
            adsb_min=adsb_min,
            adsc_raw=adsc_raw,
            fill_meta=fill_meta,
            pred_by_model=pred_by_model,
            frame_all=frame_all,
            gap_break_min=float(args.gap_break_min),
            hide_adsb_known=bool(int(args.hide_adsb_known)),
            ourmethod_postprocess=bool(int(args.ourmethod_postprocess)),
            global_postprocess=bool(int(args.global_postprocess)),
            max_vertical_rate=float(args.max_vertical_rate),
            spike_threshold=float(args.spike_threshold),
            baseline_blend=float(args.baseline_blend),
        )
        png_alt = plot_alt_dir / f"{n_done+1:02d}_{meta['sample_id']}_{meta['flight_id']}_alt_2d.png"
        _plot_one_alt_2d(
            out_png=png_alt,
            meta=meta,
            adsc_raw=adsc_raw,
            fill_meta=fill_meta,
            pred_by_model=pred_by_model,
            frame_all=frame_all,
            ourmethod_postprocess=bool(int(args.ourmethod_postprocess)),
            global_postprocess=bool(int(args.global_postprocess)),
            max_vertical_rate=float(args.max_vertical_rate),
            spike_threshold=float(args.spike_threshold),
            baseline_blend=float(args.baseline_blend),
        )
        png_alt_error = plot_alt_error_dir / f"{n_done+1:02d}_{meta['sample_id']}_{meta['flight_id']}_alt_error_points.png"
        _plot_one_alt_error_points(
            out_png=png_alt_error,
            meta=meta,
            fill_meta=fill_meta,
            pred_by_model=pred_by_model,
            frame_all=frame_all,
            ourmethod_postprocess=bool(int(args.ourmethod_postprocess)),
            global_postprocess=bool(int(args.global_postprocess)),
            max_vertical_rate=float(args.max_vertical_rate),
            spike_threshold=float(args.spike_threshold),
            baseline_blend=float(args.baseline_blend),
        )
        selected_rows.append(
            {
                "idx": n_done + 1,
                "sample_id": meta["sample_id"],
                "flight_id": meta["flight_id"],
                "flight_date": meta["flight_date"],
                "gap_minutes": meta["gap_minutes"],
                "fills_count": len(fill_meta),
                "plot_path": str(png),
                "plot_alt_2d_path": str(png_alt),
                "plot_alt_error_points_path": str(png_alt_error),
            }
        )
        n_done += 1

    pd.DataFrame(selected_rows).to_csv(out_dir / "selected_flights_for_3d_compare.csv", index=False)
    metric_df = pd.DataFrame(metric_rows)
    metric_df.to_csv(out_dir / "model_segment_proxy_metrics.csv", index=False)
    if len(metric_df):
        summary = (
            metric_df.groupby("model_name", as_index=False)
            .agg(
                segment_count=("fill_id", "count"),
                overshoot_rate=("overshoot_flag", "mean"),
                edge_spike_rate=("edge_spike_flag", "mean"),
                abnormal_ratio=("abnormal_flag", "mean"),
                warn_ratio=("warn_flag", "mean"),
                keep_ratio=("keep_flag", "mean"),
                mean_max_vertical_rate_inside=("max_vertical_rate_inside", "mean"),
                anchor_alt_mae=("anchor_alt_mae", "mean"),
                anchor_alt_rmse=("anchor_alt_rmse", "mean"),
                postprocess_applied=("postprocess_applied", "mean"),
            )
            .sort_values(["abnormal_ratio", "edge_spike_rate", "overshoot_rate"], ascending=[True, True, True])
        )
    else:
        summary = pd.DataFrame(
            columns=[
                "model_name",
                "segment_count",
                "overshoot_rate",
                "edge_spike_rate",
                "abnormal_ratio",
                "warn_ratio",
                "keep_ratio",
                "mean_max_vertical_rate_inside",
                "anchor_alt_mae",
                "anchor_alt_rmse",
                "postprocess_applied",
            ]
        )
    summary.to_csv(out_dir / "model_proxy_metrics_summary.csv", index=False)
    print(f"[ok] plots={n_done} dir={plot_dir}")
    print(f"[ok] selected_csv={out_dir / 'selected_flights_for_3d_compare.csv'}")
    print(f"[ok] model_metrics={out_dir / 'model_proxy_metrics_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
