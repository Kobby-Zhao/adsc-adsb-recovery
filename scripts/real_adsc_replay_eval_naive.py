from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train import load_config
from scripts.real_adsc_replay_eval import (
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
    """Lightweight inference-time fixer for unrealistic altitude spikes.

    This is a display/inference postprocess only; it does not alter training.
    """
    x = alt_pred.astype(float).copy().reset_index(drop=True)
    n = len(x)
    if n <= 2:
        if n >= 1:
            x.iloc[0] = float(left_anchor_alt)
        if n >= 2:
            x.iloc[-1] = float(right_anchor_alt)
        return x

    # 1) hard anchor consistency
    x.iloc[0] = float(left_anchor_alt)
    x.iloc[-1] = float(right_anchor_alt)

    # 2) edge/interior single-point spike suppression
    for _ in range(2):
        for i in range(1, n - 1):
            nbr = 0.5 * (x.iloc[i - 1] + x.iloc[i + 1])
            if abs(x.iloc[i] - nbr) > float(spike_threshold):
                x.iloc[i] = nbr

    # 3) forward rate clamp
    m = float(max_vertical_rate_per_min)
    for i in range(1, n):
        lo = x.iloc[i - 1] - m
        hi = x.iloc[i - 1] + m
        x.iloc[i] = min(max(x.iloc[i], lo), hi)

    # 4) backward rate clamp (to respect right boundary too)
    for i in range(n - 2, -1, -1):
        lo = x.iloc[i + 1] - m
        hi = x.iloc[i + 1] + m
        x.iloc[i] = min(max(x.iloc[i], lo), hi)

    # 5) blend toward anchor-to-anchor baseline to avoid unrealistic zig-zag
    lin = pd.Series(
        [left_anchor_alt + (right_anchor_alt - left_anchor_alt) * (k / (n - 1)) for k in range(n)],
        dtype=float,
    )
    b = float(baseline_blend)
    x = (1.0 - b) * x + b * lin

    # 6) light interior smoothing
    y = x.copy()
    for i in range(1, n - 1):
        y.iloc[i] = 0.25 * x.iloc[i - 1] + 0.5 * x.iloc[i] + 0.25 * x.iloc[i + 1]
    x = y

    # final hard anchors
    x.iloc[0] = float(left_anchor_alt)
    x.iloc[-1] = float(right_anchor_alt)
    return x


def _segment_adsb(adsb_all: pd.DataFrame, gap_break_min: float) -> list[pd.DataFrame]:
    if adsb_all.empty:
        return []
    x = adsb_all.sort_values("minute_ts").copy()
    x["dt_min"] = x["minute_ts"].diff().dt.total_seconds().div(60.0).fillna(0.0)
    x["seg_id"] = (x["dt_min"] >= gap_break_min).astype("int64").cumsum()
    return [g.copy() for _, g in x.groupby("seg_id")]


def _plot_2d(
    out_png: Path,
    flight_meta: dict,
    adsb_min: pd.DataFrame,
    adsc_raw: pd.DataFrame,
    pred: pd.DataFrame,
    gap_break_min: float,
    mode_label: str = "naive",
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Lat/Lon
    for i, seg in enumerate(_segment_adsb(adsb_min, gap_break_min)):
        ax1.plot(seg["lon"], seg["lat"], color="#7f7f7f", lw=1.2, alpha=0.85, label="ADS-B known" if i == 0 else None)
    if len(adsc_raw):
        x = adsc_raw.copy()
        x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
        ax1.scatter(x["lon"], x["lat"], s=18, c="#7b3294", label="ADS-C anchors", zorder=4)

    # Plot recovered fill segments separately to keep real gaps visible.
    sort_col = "minute_ts" if "minute_ts" in pred.columns else ("time" if "time" in pred.columns else "timestamp")
    for i, (sid, g) in enumerate(pred.groupby("sample_id")):
        gs = g.copy()
        gs[sort_col] = pd.to_datetime(gs[sort_col], utc=True)
        gs = gs.sort_values(sort_col)
        ax1.plot(gs["pred_lon"], gs["pred_lat"], color="#d62728", lw=2.0, alpha=0.95, label=f"OurMethod_BiLSTM ({mode_label})" if i == 0 else None)

    ax1.set_title("Lat/Lon Replay (naive)")
    ax1.set_xlabel("Lon")
    ax1.set_ylabel("Lat")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25)

    # Alt vs Time
    for i, seg in enumerate(_segment_adsb(adsb_min, gap_break_min)):
        ax2.plot(seg["minute_ts"], seg["alt"], color="#7f7f7f", lw=1.2, alpha=0.85, label="ADS-B known" if i == 0 else None)
    if len(adsc_raw):
        x = adsc_raw.copy()
        x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
        ax2.scatter(x["timestamp"], x["baroaltitude"], s=18, c="#7b3294", label="ADS-C anchors", zorder=4)

    for i, (sid, g) in enumerate(pred.groupby("sample_id")):
        gs = g.copy()
        gs[sort_col] = pd.to_datetime(gs[sort_col], utc=True)
        gs = gs.sort_values(sort_col)
        ax2.plot(gs[sort_col], gs["pred_alt"], color="#d62728", lw=2.0, alpha=0.95, label=f"OurMethod_BiLSTM ({mode_label})" if i == 0 else None)

    ax2.set_title("Altitude Replay (naive)")
    ax2.set_xlabel("Time (UTC)")
    ax2.set_ylabel("Altitude")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)

    fig.suptitle(
        f"{flight_meta['sample_id']} | flight={flight_meta['flight_id']} | task=real_adsc_{mode_label}",
        fontsize=10,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("Real ADS-C replay eval (naive, no strategy optimization)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--adsc-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-flights", type=int, default=20)
    ap.add_argument("--context-minutes", type=int, default=5)
    ap.add_argument("--gap-break-min", type=float, default=10.0)
    ap.add_argument("--middle-anchor-only", type=int, default=1, help="1=only adsc->adsc gaps, 0=all fill intervals")
    ap.add_argument("--alt-postprocess", type=int, default=0, help="1=enable lightweight altitude postprocess to suppress spikes")
    ap.add_argument("--max-vertical-rate", type=float, default=180.0)
    ap.add_argument("--spike-threshold", type=float, default=260.0)
    ap.add_argument("--baseline-blend", type=float, default=0.35)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    ckpt = Path(args.checkpoint)
    adsc_dir = Path(args.adsc_dir)
    out_dir = Path(args.out_dir)
    plot_dir = out_dir / "plots"
    series_dir = out_dir / "pred_series"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    series_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    done = 0
    for fp in sorted(adsc_dir.glob("*.csv")):
        if done >= int(args.max_flights):
            break
        raw = pd.read_csv(fp)
        if "source" not in raw.columns:
            continue
        raw["source"] = raw["source"].astype(str).str.lower()
        adsb_raw = raw[raw["source"].eq("adsb")].copy()
        adsc_raw = raw[raw["source"].eq("adsc")].copy()
        if adsc_raw.empty:
            continue

        adsb_min = _resample_minute(adsb_raw)
        adsc_min = _resample_minute(adsc_raw)
        if adsc_min.empty:
            continue

        if len(adsb_min):
            adsb_min["minute_ts"] = pd.to_datetime(adsb_min["minute_ts"], utc=True)
        adsc_raw["timestamp"] = pd.to_datetime(adsc_raw["timestamp"], utc=True)
        w_start = min(
            adsc_raw["timestamp"].min(),
            adsb_min["minute_ts"].min() if len(adsb_min) else adsc_raw["timestamp"].min(),
        )
        w_end = max(
            adsc_raw["timestamp"].max(),
            adsb_min["minute_ts"].max() if len(adsb_min) else adsc_raw["timestamp"].max(),
        )

        blocks = _build_known_blocks(adsb_min, adsc_raw, w_start, w_end, gap_break_min=float(args.gap_break_min))
        fills = _build_fill_intervals(blocks, min_gap_min=2.0)
        if bool(int(args.middle_anchor_only)):
            fills = [f for f in fills if f["left_block_type"] == "adsc_anchor" and f["right_block_type"] == "adsc_anchor"]
        if not fills:
            continue

        sample_id_base = f"{fp.stem}_naive"
        flight_id, flight_date = _parse_flight_meta(fp)
        frames = []
        fmeta = []
        for i, f in enumerate(fills, start=1):
            left = _extract_block_boundary_state(blocks[f["left_block_idx"]], "end")
            right = _extract_block_boundary_state(blocks[f["right_block_idx"]], "start")
            fill_sid = f"{sample_id_base}__f{i:02d}"
            frame, _, _ = _build_fill_frame(
                minute_all_adsb=adsb_min,
                fill_start=f["fill_start_time"],
                fill_end=f["fill_end_time"],
                left_state=left,
                right_state=right,
                task_type="pure_adsc",
                context_minutes=int(args.context_minutes),
            )
            if frame.empty:
                continue
            frame["sample_id"] = fill_sid
            frame["flight_id"] = flight_id
            frames.append(frame)
            fmeta.append({"fill_sid": fill_sid, "fill_minutes": float(f["fill_minutes"])})
        if not frames:
            continue

        frame_all = pd.concat(frames, ignore_index=True)
        pred = _predict_on_frame(cfg=cfg, checkpoint=ckpt, frame=frame_all, pred_key="pred_pos")
        if pred.empty:
            continue

        if bool(int(args.alt_postprocess)):
            # merge obs anchors, then fix each fill segment independently
            ref = frame_all[["sample_id", "minute_ts", "obs_mask", "obs_alt"]].copy()
            ref["minute_ts"] = pd.to_datetime(ref["minute_ts"], utc=True)
            pred["minute_ts"] = pd.to_datetime(pred["minute_ts"], utc=True)
            pred = pred.merge(ref, on=["sample_id", "minute_ts"], how="left")
            fixed_parts = []
            for sid, g in pred.groupby("sample_id", sort=False):
                gs = g.sort_values("minute_ts").copy()
                left_obs = gs.iloc[0]["obs_alt"] if len(gs) else 0.0
                right_obs = gs.iloc[-1]["obs_alt"] if len(gs) else 0.0
                gs["pred_alt_raw"] = gs["pred_alt"].astype(float)
                gs["pred_alt"] = _postprocess_altitude_segment(
                    alt_pred=gs["pred_alt"].astype(float),
                    left_anchor_alt=float(left_obs),
                    right_anchor_alt=float(right_obs),
                    max_vertical_rate_per_min=float(args.max_vertical_rate),
                    spike_threshold=float(args.spike_threshold),
                    baseline_blend=float(args.baseline_blend),
                ).values
                fixed_parts.append(gs)
            pred = pd.concat(fixed_parts, ignore_index=True)

        pred_csv = series_dir / f"{sample_id_base}_pred_series.csv"
        pred.to_csv(pred_csv, index=False)

        meta = {
            "sample_id": sample_id_base,
            "flight_id": flight_id,
            "flight_date": flight_date,
        }
        png = plot_dir / f"{done+1:02d}_{sample_id_base}_{flight_id}_naive.png"
        _plot_2d(
            out_png=png,
            flight_meta=meta,
            adsb_min=adsb_min,
            adsc_raw=adsc_raw,
            pred=pred,
            gap_break_min=float(args.gap_break_min),
            mode_label="naive_fix" if bool(int(args.alt_postprocess)) else "naive",
        )
        rows.append(
            {
                "idx": done + 1,
                "sample_id": sample_id_base,
                "flight_id": flight_id,
                "flight_date": flight_date,
                "fills_count": len(fmeta),
                "pred_csv": str(pred_csv),
                "plot_path": str(png),
            }
        )
        done += 1

    pd.DataFrame(rows).to_csv(out_dir / "naive_selected_flights.csv", index=False)
    print(f"[ok] naive_replay_done={done} out={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
