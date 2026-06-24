from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.real_adsc_replay_eval import (
    _build_fill_frame,
    _build_fill_intervals,
    _build_known_blocks,
    _extract_block_boundary_state,
    _predict_on_frame,
    _resample_minute,
)
from src.training.utils import load_config


def _safe_ratio(num: float, denom: float, eps: float = 1e-3) -> float:
    return float(num) / float(max(abs(denom), eps))


def _bucket_len(m: float) -> str:
    if m <= 15:
        return "<=15"
    if m <= 60:
        return "15-60"
    if m <= 180:
        return "60-180"
    return ">180"


def _quality_class(row: pd.Series) -> tuple[str, str]:
    if row["shape_abnormal_flag"]:
        return "abnormal", row["abnormal_reason"]
    if row["overshoot_flag"] or row["edge_spike_flag"]:
        return "warn", "overshoot_or_edge_spike"
    return "keep", "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--adsc-dir", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-samples", type=int, default=200)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ckpt = Path(args.checkpoint)
    adsc_dir = Path(args.adsc_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audit = pd.read_csv(args.audit)
    if "sample_id" not in audit.columns:
        raise RuntimeError("audit file missing sample_id")

    csv_files = sorted(adsc_dir.glob("*.csv"))
    flight_adsb_cache: dict[str, pd.DataFrame] = {}
    flight_adsc_cache: dict[str, pd.DataFrame] = {}
    for fp in csv_files:
        fdf = pd.read_csv(fp)
        if "source" not in fdf.columns:
            continue
        fdf["source"] = fdf["source"].astype(str).str.lower()
        adsb_min = _resample_minute(fdf[fdf["source"].eq("adsb")].copy())
        adsc_raw = fdf[fdf["source"].eq("adsc")].copy()
        if len(adsb_min):
            flight_adsb_cache[fp.stem] = adsb_min
        if len(adsc_raw):
            flight_adsc_cache[fp.stem] = adsc_raw

    fill_meta = []
    fill_frames = []

    max_samples = min(args.max_samples, len(audit))
    for _, row in audit.head(max_samples).iterrows():
        sample_id = str(row["sample_id"])
        base = sample_id.split("_a")[0]
        adsb_all = flight_adsb_cache.get(base, pd.DataFrame(columns=["minute_ts", "lat", "lon", "alt"])).copy()
        adsc_raw = flight_adsc_cache.get(base, pd.DataFrame(columns=["timestamp", "lat", "lon", "baroaltitude"])).copy()
        if len(adsb_all):
            adsb_all["minute_ts"] = pd.to_datetime(adsb_all["minute_ts"], utc=True)
        if len(adsc_raw):
            adsc_raw["timestamp"] = pd.to_datetime(adsc_raw["timestamp"], utc=True)

        if len(adsb_all):
            w_start = adsb_all["minute_ts"].min()
            w_end = adsb_all["minute_ts"].max()
        else:
            w_start = pd.to_datetime(row["adsc_anchor_start_time"], utc=True)
            w_end = pd.to_datetime(row["adsc_anchor_end_time"], utc=True)
        if len(adsc_raw):
            w_start = min(w_start, adsc_raw["timestamp"].min())
            w_end = max(w_end, adsc_raw["timestamp"].max())

        blocks = _build_known_blocks(adsb_all, adsc_raw, w_start, w_end, gap_break_min=10.0)
        fills = _build_fill_intervals(blocks, min_gap_min=2.0)
        for fi, f in enumerate(fills, start=1):
            left_block = blocks[f["left_block_idx"]]
            right_block = blocks[f["right_block_idx"]]
            left_state = _extract_block_boundary_state(left_block, "end")
            right_state = _extract_block_boundary_state(right_block, "start")
            fill_id = f"f{fi:02d}"
            fill_sid = f"{sample_id}__{fill_id}"
            frame, _, _ = _build_fill_frame(
                minute_all_adsb=adsb_all,
                fill_start=f["fill_start_time"],
                fill_end=f["fill_end_time"],
                left_state=left_state,
                right_state=right_state,
                task_type="adsc_plus_local_adsb",
                context_minutes=int(cfg.get("data", {}).get("context_minutes", 5)),
            )
            if frame.empty:
                continue
            frame["sample_id"] = fill_sid
            frame["flight_id"] = row.get("flight_id", "")
            fill_frames.append(frame)
            fill_meta.append(
                {
                    "sample_id": sample_id,
                    "fill_id": fill_id,
                    "fill_sid": fill_sid,
                    "left_block_type": f["left_block_type"],
                    "right_block_type": f["right_block_type"],
                    "fill_minutes": f["fill_minutes"],
                    "fill_start_time": f["fill_start_time"],
                    "fill_end_time": f["fill_end_time"],
                    "left_boundary_alt": float(left_state["alt"]),
                    "right_boundary_alt": float(right_state["alt"]),
                }
            )

    if not fill_frames:
        raise RuntimeError("no fill frames built for audit")

    infer_frame = pd.concat(fill_frames, ignore_index=True)
    pred_on = _predict_on_frame(cfg, ckpt, infer_frame, pred_key="pred_pos")
    pred_off = _predict_on_frame(cfg, ckpt, infer_frame, pred_key="pred_pos_main")
    pred_on["minute_ts"] = pd.to_datetime(pred_on["minute_ts"], utc=True)
    pred_off["minute_ts"] = pd.to_datetime(pred_off["minute_ts"], utc=True)

    rows = []
    for meta in fill_meta:
        seg = pred_on[pred_on["sample_id"].astype(str).eq(meta["fill_sid"])].copy()
        if len(seg) < 2:
            continue
        seg = seg.sort_values("minute_ts").reset_index(drop=True)
        alt = seg["pred_alt"].to_numpy(dtype=float)
        left_alt = float(meta["left_boundary_alt"])
        right_alt = float(meta["right_boundary_alt"])
        gap = right_alt - left_alt
        alt_min = float(np.min(alt))
        alt_max = float(np.max(alt))
        peak_idx = int(np.argmax(alt))
        trough_idx = int(np.argmin(alt))
        peak_offset = peak_idx
        trough_offset = trough_idx
        overshoot_up = alt_max - max(left_alt, right_alt)
        undershoot_down = min(left_alt, right_alt) - alt_min
        overshoot_ratio = _safe_ratio(overshoot_up, gap)
        undershoot_ratio = _safe_ratio(undershoot_down, gap)

        diffs = np.diff(alt)
        first_jump = float(diffs[0]) if len(diffs) > 0 else 0.0
        second_jump = float(diffs[1]) if len(diffs) > 1 else 0.0
        last_jump = float(diffs[-1]) if len(diffs) > 0 else 0.0
        second_last_jump = float(diffs[-2]) if len(diffs) > 1 else 0.0
        first_two_peak = peak_idx <= 2 or trough_idx <= 2
        last_two_peak = peak_idx >= len(alt) - 3 or trough_idx >= len(alt) - 3

        max_vr = float(np.max(np.abs(diffs))) if len(diffs) else 0.0
        mean_vr = float(np.mean(np.abs(diffs))) if len(diffs) else 0.0
        smoothness = float(np.mean(np.abs(np.diff(alt, n=2)))) if len(alt) > 2 else 0.0
        sign_changes = int(np.sum(np.diff(np.sign(diffs)) != 0)) if len(diffs) > 1 else 0

        overshoot_flag = overshoot_up > 200.0
        undershoot_flag = undershoot_down > 200.0
        edge_spike_flag = (first_two_peak or last_two_peak) and (max_vr > 300.0)
        peak_then_return_flag = first_two_peak or last_two_peak
        shape_abnormal_flag = (overshoot_flag or undershoot_flag) and (edge_spike_flag or peak_then_return_flag)
        abnormal_reason = "overshoot_edge" if shape_abnormal_flag else ""

        if first_two_peak:
            anomaly_position = "first_two_steps"
        elif last_two_peak:
            anomaly_position = "last_two_steps"
        elif overshoot_flag or undershoot_flag:
            anomaly_position = "middle"
        else:
            anomaly_position = "none"

        rows.append(
            {
                "sample_id": meta["sample_id"],
                "fill_id": meta["fill_id"],
                "left_block_type": meta["left_block_type"],
                "right_block_type": meta["right_block_type"],
                "fill_minutes": meta["fill_minutes"],
                "point_count": int(len(seg)),
                "recovery_mode": "fill_intervals",
                "residual_enabled": bool(
                    cfg.get("model", {}).get("vertical_projector_enabled", False)
                    or cfg.get("model", {}).get("vertical_tune_enabled", False)
                ),
                "left_boundary_alt": left_alt,
                "right_boundary_alt": right_alt,
                "boundary_alt_gap": gap,
                "boundary_alt_mean": (left_alt + right_alt) / 2.0,
                "segment_alt_min": alt_min,
                "segment_alt_max": alt_max,
                "segment_alt_range": alt_max - alt_min,
                "peak_index": peak_idx,
                "trough_index": trough_idx,
                "peak_time_offset_min": peak_offset,
                "trough_time_offset_min": trough_offset,
                "overshoot_up": overshoot_up,
                "undershoot_down": undershoot_down,
                "overshoot_ratio": overshoot_ratio,
                "undershoot_ratio": undershoot_ratio,
                "first_step_alt_jump": first_jump,
                "second_step_alt_jump": second_jump,
                "last_step_alt_jump": last_jump,
                "second_last_step_alt_jump": second_last_jump,
                "first_two_step_peak_flag": first_two_peak,
                "last_two_step_peak_flag": last_two_peak,
                "max_vertical_rate_inside": max_vr,
                "mean_vertical_rate_inside": mean_vr,
                "altitude_smoothness_score": smoothness,
                "sign_change_count_in_alt_diff": sign_changes,
                "overshoot_flag": overshoot_flag,
                "undershoot_flag": undershoot_flag,
                "peak_then_return_flag": peak_then_return_flag,
                "edge_spike_flag": edge_spike_flag,
                "shape_abnormal_flag": shape_abnormal_flag,
                "abnormal_reason": abnormal_reason,
                "anomaly_position": anomaly_position,
            }
        )

    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(out_dir / "fill_segment_shape_audit.csv", index=False)

    audit_df["fill_type"] = audit_df["left_block_type"] + "->" + audit_df["right_block_type"]
    audit_df["length_bucket"] = audit_df["fill_minutes"].apply(_bucket_len)
    group_rows = []
    def _append_group(g: pd.DataFrame, group_by: str, group_key: str) -> None:
        group_rows.append(
            {
                "group_by": group_by,
                "group_key": group_key,
                "count": int(len(g)),
                "overshoot_flag_ratio": float(g["overshoot_flag"].mean()),
                "edge_spike_flag_ratio": float(g["edge_spike_flag"].mean()),
                "shape_abnormal_flag_ratio": float(g["shape_abnormal_flag"].mean()),
                "mean_overshoot_up": float(g["overshoot_up"].mean()),
                "p90_overshoot_up": float(g["overshoot_up"].quantile(0.9)),
                "mean_max_vertical_rate_inside": float(g["max_vertical_rate_inside"].mean()),
            }
        )

    for k, g in audit_df.groupby("fill_type"):
        _append_group(g, "fill_type", str(k))
    for k, g in audit_df.groupby("length_bucket"):
        _append_group(g, "length_bucket", str(k))
    for k, g in audit_df.groupby("residual_enabled"):
        _append_group(g, "residual_enabled", str(k))
    for k, g in audit_df.groupby("anomaly_position"):
        _append_group(g, "anomaly_position", str(k))
    pd.DataFrame(group_rows).to_csv(out_dir / "fill_segment_group_stats.csv", index=False)

    qc_rows = []
    for _, r in audit_df.iterrows():
        qc, reason = _quality_class(r)
        qc_rows.append(
            {
                "sample_id": r["sample_id"],
                "fill_id": r["fill_id"],
                "fill_type": r["left_block_type"] + "->" + r["right_block_type"],
                "fill_minutes": r["fill_minutes"],
                "quality_class": qc,
                "main_reason": reason,
                "residual_enabled": r["residual_enabled"],
                "notes": "",
            }
        )
    pd.DataFrame(qc_rows).to_csv(out_dir / "fill_segment_quality_classification.csv", index=False)

    ablation_rows = []
    top = audit_df.sort_values("overshoot_up", ascending=False).head(100)
    for _, r in top.iterrows():
        sid = f"{r['sample_id']}__{r['fill_id']}"
        seg_on = pred_on[pred_on["sample_id"].astype(str).eq(sid)].copy().sort_values("minute_ts")
        seg_off = pred_off[pred_off["sample_id"].astype(str).eq(sid)].copy().sort_values("minute_ts")
        if len(seg_on) < 2 or len(seg_off) < 2:
            continue
        for mode, seg in [("on", seg_on), ("off", seg_off)]:
            alt = seg["pred_alt"].to_numpy(dtype=float)
            left_alt = float(r["left_boundary_alt"])
            right_alt = float(r["right_boundary_alt"])
            overshoot_up = float(np.max(alt) - max(left_alt, right_alt))
            diffs = np.diff(alt)
            max_vr = float(np.max(np.abs(diffs))) if len(diffs) else 0.0
            peak_idx = int(np.argmax(alt))
            trough_idx = int(np.argmin(alt))
            first_two_peak = peak_idx <= 2 or trough_idx <= 2
            last_two_peak = peak_idx >= len(alt) - 3 or trough_idx >= len(alt) - 3
            overshoot_flag = overshoot_up > 200.0
            edge_spike_flag = (first_two_peak or last_two_peak) and (max_vr > 300.0)
            shape_abnormal_flag = overshoot_flag and edge_spike_flag
            ablation_rows.append(
                {
                    "sample_id": r["sample_id"],
                    "fill_id": r["fill_id"],
                    "mode": mode,
                    "overshoot_up": overshoot_up,
                    "edge_spike_flag": edge_spike_flag,
                    "shape_abnormal_flag": shape_abnormal_flag,
                    "max_vertical_rate_inside": max_vr,
                    "residual_enabled": mode == "on",
                }
            )
    pd.DataFrame(ablation_rows).to_csv(out_dir / "fill_segment_residual_ablation.csv", index=False)

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    abnormal = audit_df.sort_values("overshoot_up", ascending=False).head(12)
    for _, r in abnormal.iterrows():
        seg = pred_on[pred_on["sample_id"].astype(str).eq(f"{r['sample_id']}__{r['fill_id']}")].copy()
        if seg.empty:
            continue
        seg = seg.sort_values("minute_ts")
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        ax.plot(seg["minute_ts"], seg["pred_alt"], color="#e7298a", lw=1.6)
        ax.scatter(
            [seg["minute_ts"].iloc[0], seg["minute_ts"].iloc[-1]],
            [r["left_boundary_alt"], r["right_boundary_alt"]],
            c="#1f78b4",
            s=30,
        )
        ax.scatter(
            [seg["minute_ts"].iloc[int(r["peak_index"])], seg["minute_ts"].iloc[int(r["trough_index"])]],
            [seg["pred_alt"].iloc[int(r["peak_index"])], seg["pred_alt"].iloc[int(r["trough_index"])]],
            c=["#d7301f", "#0570b0"],
            s=28,
            label="peak/trough",
        )
        ax.set_title(
            f"{r['sample_id']} {r['fill_id']} {r['left_block_type']}->{r['right_block_type']} "
            f"{int(r['fill_minutes'])}m | overshoot={r['overshoot_up']:.1f} | residual=on"
        )
        fig.tight_layout()
        fig.savefig(plots_dir / f"abnormal_{r['sample_id']}_{r['fill_id']}.png", dpi=140)
        plt.close(fig)

    grp = pd.read_csv(out_dir / "fill_segment_group_stats.csv")
    if len(grp):
        g_type = grp[grp["group_by"].eq("fill_type")]
        g_len = grp[grp["group_by"].eq("length_bucket")]
        g_pos = grp[grp["group_by"].eq("anomaly_position")]
        pivot = g_type.set_index("group_key")["shape_abnormal_flag_ratio"]
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        pivot.plot(kind="bar", ax=ax, legend=False)
        ax.set_ylabel("abnormal_ratio")
        ax.set_title("Abnormal Ratio by Fill Type")
        fig.tight_layout()
        fig.savefig(plots_dir / "abnormal_ratio_by_type.png", dpi=140)
        plt.close(fig)

        pivot2 = g_len.set_index("group_key")["shape_abnormal_flag_ratio"]
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        pivot2.plot(kind="bar", ax=ax, legend=False)
        ax.set_ylabel("abnormal_ratio")
        ax.set_title("Abnormal Ratio by Length Bucket")
        fig.tight_layout()
        fig.savefig(plots_dir / "abnormal_ratio_by_length.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(7, 3))
        audit_df["overshoot_up"].plot(kind="hist", bins=40, ax=ax, color="#756bb1", alpha=0.85)
        ax.set_xlabel("overshoot_up")
        ax.set_ylabel("count")
        ax.set_title("Overshoot Distribution")
        fig.tight_layout()
        fig.savefig(plots_dir / "overshoot_up_distribution.png", dpi=140)
        plt.close(fig)

        if len(g_pos):
            fig, ax = plt.subplots(1, 1, figsize=(6, 3))
            g_pos.set_index("group_key")["shape_abnormal_flag_ratio"].plot(kind="bar", ax=ax, legend=False, color="#2ca25f")
            ax.set_ylabel("abnormal_ratio")
            ax.set_title("Abnormal Ratio by Anomaly Position")
            fig.tight_layout()
            fig.savefig(plots_dir / "abnormal_ratio_by_anomaly_position.png", dpi=140)
            plt.close(fig)

    summary = [
        "# Fill Segment Shape Anomaly Summary",
        "",
        f"- samples: {len(audit_df)}",
        f"- abnormal_ratio: {audit_df['shape_abnormal_flag'].mean():.4f}",
        f"- overshoot_ratio: {audit_df['overshoot_flag'].mean():.4f}",
        f"- edge_spike_ratio: {audit_df['edge_spike_flag'].mean():.4f}",
        "",
        "## Notes",
        "- residual ablation included: mode=on uses pred_pos, mode=off uses pred_pos_main.",
    ]
    (out_dir / "fill_segment_shape_anomaly_summary.md").write_text("\n".join(summary), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
