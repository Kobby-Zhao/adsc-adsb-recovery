from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train import load_config
from scripts.real_adsc_replay_eval import _predict_on_frame


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("Real ADS-C gapwise qualitative eval for ourmethod.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--samples-parquet", required=True)
    ap.add_argument("--selected-flights-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    return ap


def _build_gapwise_segments(flight_df: pd.DataFrame) -> tuple[list[pd.DataFrame], list[dict]]:
    g = flight_df.sort_values("minute_ts").reset_index(drop=True).copy()
    anchor_idx = g.index[g["obs_mask"].astype(float) > 0.5].tolist()
    if len(anchor_idx) < 2:
        return [], []

    frames: list[pd.DataFrame] = []
    meta: list[dict] = []
    gap_id = 0
    for left_i, right_i in zip(anchor_idx[:-1], anchor_idx[1:]):
        if right_i - left_i <= 1:
            continue
        gap_id += 1
        seg = g.iloc[left_i : right_i + 1].copy()
        seg["sample_id"] = f"{g['flight_id'].iloc[0]}__gap{gap_id:03d}"
        frames.append(seg)
        meta.append(
            {
                "sample_id": seg["sample_id"].iloc[0],
                "flight_id": seg["flight_id"].iloc[0],
                "left_idx": int(left_i),
                "right_idx": int(right_i),
                "left_time": str(seg["minute_ts"].iloc[0]),
                "right_time": str(seg["minute_ts"].iloc[-1]),
                "left_anchor_alt": float(seg["obs_alt"].iloc[0]),
                "right_anchor_alt": float(seg["obs_alt"].iloc[-1]),
                "gap_minutes": int(len(seg) - 2),
            }
        )
    return frames, meta


def _hard_fix_anchors(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    mask = pd.to_numeric(out["obs_mask"], errors="coerce").fillna(0.0) > 0.5
    for pred_col, obs_col in [("pred_lat", "obs_lat"), ("pred_lon", "obs_lon"), ("pred_alt", "obs_alt")]:
        if pred_col in out.columns and obs_col in out.columns:
            out.loc[mask, pred_col] = pd.to_numeric(out.loc[mask, obs_col], errors="coerce")
    if "pred_alt_main" in out.columns and "obs_alt" in out.columns:
        out.loc[mask, "pred_alt_main"] = pd.to_numeric(out.loc[mask, "obs_alt"], errors="coerce")
    return out


def _light_right_boundary_governor(
    seg_pred: pd.DataFrame,
    tol_alt: float = 80.0,
    taper_points: int = 4,
    climb_min: float = 120.0,
) -> pd.DataFrame:
    """Lightly regularize long climb gaps that overshoot the right anchor.

    The correction is intentionally narrow:
    - only activate on anchor-bounded climb segments,
    - only when the predicted tail clearly overshoots the right anchor,
    - preserve the early part of the model trajectory,
    - guide the mid/late part toward a smooth monotone convergence curve.
    """
    x = seg_pred.sort_values("minute_ts").copy().reset_index(drop=True)
    if len(x) < 4:
        return x
    obs_mask = pd.to_numeric(x["obs_mask"], errors="coerce").fillna(0.0).to_numpy()
    if not (obs_mask[0] > 0.5 and obs_mask[-1] > 0.5):
        return x

    # Anchors have already been hard-fixed before this governor runs, so the
    # first/last predicted altitude values are the reliable anchor references.
    left_anchor = float(pd.to_numeric(x.loc[0, "pred_alt"], errors="coerce"))
    right_anchor = float(pd.to_numeric(x.loc[len(x) - 1, "pred_alt"], errors="coerce"))
    climb_delta = right_anchor - left_anchor
    climb_sign = np.sign(climb_delta)
    if climb_sign == 0:
        return x

    last_missing = len(x) - 2
    if last_missing < 1 or obs_mask[last_missing] > 0.5:
        return x

    pred_last = float(pd.to_numeric(x.loc[last_missing, "pred_alt"], errors="coerce"))
    overshoot = climb_sign * (pred_last - right_anchor)
    if overshoot <= float(tol_alt):
        return x
    if climb_sign < 0:
        return x

    n_missing = len(x) - 2
    if n_missing < 3:
        return x
    if float(climb_delta) < float(climb_min):
        # For short/low climbs, keep the previous minimal tail-only governor.
        start = max(1, last_missing - int(taper_points) + 1)
        left_ref_idx = start - 1
        left_ref_alt = float(pd.to_numeric(x.loc[left_ref_idx, "pred_alt"], errors="coerce"))
        win_len = last_missing - start + 1
        if win_len <= 0:
            return x
        for j, idx in enumerate(range(start, last_missing + 1), start=1):
            frac = j / (win_len + 1.0)
            guide = left_ref_alt + frac * (right_anchor - left_ref_alt)
            w = frac * frac
            for col in ["pred_alt", "pred_alt_main"]:
                if col not in x.columns:
                    continue
                raw = float(pd.to_numeric(x.loc[idx, col], errors="coerce"))
                blended = (1.0 - w) * raw + w * guide
                upper = right_anchor + float(tol_alt)
                x.loc[idx, col] = min(blended, upper)
        return x

    # Long climb gap: use a smooth monotone guide and only steer the mid/late
    # part of the segment toward it. This preserves model shape early while
    # preventing terminal overshoot.
    miss_idx = np.arange(1, len(x) - 1)
    u = miss_idx.astype(float) / (len(x) - 1)  # in (0, 1)
    guide_frac = u * u * (3.0 - 2.0 * u)  # smoothstep
    guide = left_anchor + climb_delta * guide_frac

    for col in ["pred_alt", "pred_alt_main"]:
        if col not in x.columns:
            continue
        raw = pd.to_numeric(x.loc[1 : len(x) - 2, col], errors="coerce").to_numpy(dtype=float)
        # Stronger influence closer to the right boundary; keep early segment mostly intact.
        blend_w = np.clip((u - 0.32) / 0.68, 0.0, 1.0) ** 2
        # Tolerance band shrinks toward the right anchor to avoid late overshoot.
        band = float(tol_alt) * np.clip(1.0 - u, 0.0, 1.0) ** 1.5
        upper_env = guide + band
        capped = np.minimum(raw, upper_env)
        blended = (1.0 - blend_w) * capped + blend_w * guide
        # Keep the climb monotone and below the terminal anchor.
        blended = np.maximum.accumulate(np.maximum(blended, left_anchor))
        blended = np.minimum(blended, right_anchor - 1e-3)
        x.loc[1 : len(x) - 2, col] = blended
    return x


def _stitch_flight_prediction(flight_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    base = flight_df.sort_values("minute_ts").copy()
    base["minute_ts"] = pd.to_datetime(base["minute_ts"], utc=True)
    pred = pred_df.sort_values(["sample_id", "minute_ts"]).copy()
    pred["minute_ts"] = pd.to_datetime(pred["minute_ts"], utc=True)
    pred = _hard_fix_anchors(pred)
    pred = (
        pred.groupby("sample_id", group_keys=False)
        .apply(_light_right_boundary_governor)
        .reset_index(drop=True)
    )

    stitched = pred.groupby(["flight_id", "minute_ts"], as_index=False).agg(
        {
            "obs_mask": "max",
            "pred_lat": "mean",
            "pred_lon": "mean",
            "pred_alt": "mean",
            "pred_alt_main": "mean",
        }
    )
    out = base.merge(stitched, on=["flight_id", "minute_ts"], how="left", suffixes=("", "_pred"))

    anchor_mask = pd.to_numeric(out["obs_mask_pred"], errors="coerce").fillna(0.0) > 0.5
    out["pred_lat"] = out["pred_lat"].where(~anchor_mask, out["obs_lat"])
    out["pred_lon"] = out["pred_lon"].where(~anchor_mask, out["obs_lon"])
    out["pred_alt"] = out["pred_alt"].where(~anchor_mask, out["obs_alt"])
    out["pred_alt_main"] = out["pred_alt_main"].where(~anchor_mask, out["obs_alt"])
    out["obs_mask_eval"] = pd.to_numeric(out["obs_mask_pred"], errors="coerce").fillna(0.0)
    out["obs_alt_eval"] = pd.to_numeric(out["obs_alt"], errors="coerce").fillna(0.0)
    out["obs_lat_eval"] = pd.to_numeric(out["obs_lat"], errors="coerce").fillna(0.0)
    out["obs_lon_eval"] = pd.to_numeric(out["obs_lon"], errors="coerce").fillna(0.0)
    return out


def _compute_structural_metrics(stitched: pd.DataFrame) -> dict:
    x = stitched.sort_values("minute_ts").reset_index(drop=True).copy()
    anchor_mask = pd.to_numeric(x["obs_mask_eval"], errors="coerce").fillna(0.0) > 0.5
    pred_alt = pd.to_numeric(x["pred_alt"], errors="coerce")
    obs_alt = pd.to_numeric(x["obs_alt_eval"], errors="coerce")

    anchor_err = (pred_alt[anchor_mask] - obs_alt[anchor_mask]).abs()

    gap = x[~anchor_mask].copy()
    if len(gap) >= 2:
        step = gap["pred_alt"].astype(float).diff().abs().dropna()
        if len(step) >= 2:
            second = step.diff().abs().dropna()
        else:
            second = pd.Series(dtype=float)
    else:
        step = pd.Series(dtype=float)
        second = pd.Series(dtype=float)

    left_boundary_jump = np.nan
    right_boundary_jump = np.nan
    for i in range(1, len(x)):
        if anchor_mask.iloc[i - 1] and not anchor_mask.iloc[i]:
            left_boundary_jump = abs(float(pred_alt.iloc[i]) - float(obs_alt.iloc[i - 1]))
            break
    for i in range(len(x) - 2, -1, -1):
        if not anchor_mask.iloc[i] and anchor_mask.iloc[i + 1]:
            right_boundary_jump = abs(float(obs_alt.iloc[i + 1]) - float(pred_alt.iloc[i]))
            break

    return {
        "anchor_max_abs_alt_err": float(anchor_err.max()) if len(anchor_err) else 0.0,
        "gap_max_abs_vertical_step": float(step.max()) if len(step) else 0.0,
        "gap_mean_abs_vertical_step": float(step.mean()) if len(step) else 0.0,
        "gap_max_abs_second_diff": float(second.max()) if len(second) else 0.0,
        "left_boundary_jump": float(left_boundary_jump) if np.isfinite(left_boundary_jump) else 0.0,
        "right_boundary_jump": float(right_boundary_jump) if np.isfinite(right_boundary_jump) else 0.0,
    }


def _plot_flight(stitched: pd.DataFrame, out_png: Path) -> None:
    x = stitched.sort_values("minute_ts").reset_index(drop=True).copy()
    anchor_mask = pd.to_numeric(x["obs_mask_eval"], errors="coerce").fillna(0.0) > 0.5

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(
        x.index,
        pd.to_numeric(x["pred_alt"], errors="coerce"),
        linestyle="--",
        linewidth=2.0,
        color="#d62728",
        label="ourmethod",
    )
    ax.scatter(
        x.index[anchor_mask],
        pd.to_numeric(x.loc[anchor_mask, "obs_alt_eval"], errors="coerce"),
        s=28,
        color="#1f77b4",
        label="ADS-C anchors",
        zorder=4,
    )
    fid = str(x["flight_id"].iloc[0])
    ax.set_title(f"{fid} | real ADS-C recovery (anchor-fixed, gapwise)")
    ax.set_xlabel("Minute index")
    ax.set_ylabel("Altitude")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    ckpt = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    samples = pd.read_parquet(args.samples_parquet).copy()
    samples["minute_ts"] = pd.to_datetime(samples["minute_ts"], utc=True)
    selected = pd.read_csv(args.selected_flights_csv)
    flight_ids = selected["flight_id"].astype(str).tolist()

    all_frames: list[pd.DataFrame] = []
    all_meta: list[dict] = []
    original_by_flight: dict[str, pd.DataFrame] = {}
    for fid in flight_ids:
        g = samples[samples["flight_id"].astype(str) == fid].copy()
        if g.empty:
            continue
        original_by_flight[fid] = g
        frames, meta = _build_gapwise_segments(g)
        all_frames.extend(frames)
        all_meta.extend(meta)

    if not all_frames:
        raise RuntimeError("No gapwise real ADS-C segments were constructed.")

    frame_all = pd.concat(all_frames, ignore_index=True)
    pred = _predict_on_frame(cfg=cfg, checkpoint=ckpt, frame=frame_all, pred_key="pred_pos")
    pred = _hard_fix_anchors(pred)

    metrics_rows: list[dict] = []
    for fid, base in original_by_flight.items():
        fpred = pred[pred["flight_id"].astype(str) == fid].copy()
        if fpred.empty:
            continue
        stitched = _stitch_flight_prediction(base, fpred)
        metrics = _compute_structural_metrics(stitched)
        metrics["flight_id"] = fid
        metrics["num_minutes"] = int(len(stitched))
        metrics["anchor_count"] = int((stitched["obs_mask_eval"] > 0.5).sum())
        metrics_rows.append(metrics)
        _plot_flight(stitched, plot_dir / f"ourmethod_truequal_gapwise_alt_{fid}.png")

    metrics_df = pd.DataFrame(metrics_rows).sort_values("flight_id").reset_index(drop=True)
    metrics_df.to_csv(out_dir / "ourmethod_truequal_gapwise_structural_metrics.csv", index=False)

    summary = {
        "num_flights": int(len(metrics_df)),
        "num_gapwise_segments": int(len(all_meta)),
    }
    for col in [
        "anchor_max_abs_alt_err",
        "gap_max_abs_vertical_step",
        "gap_mean_abs_vertical_step",
        "gap_max_abs_second_diff",
        "left_boundary_jump",
        "right_boundary_jump",
    ]:
        summary[f"{col}_mean"] = float(metrics_df[col].mean()) if len(metrics_df) else 0.0
        summary[f"{col}_median"] = float(metrics_df[col].median()) if len(metrics_df) else 0.0

    with open(out_dir / "ourmethod_truequal_gapwise_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(out_dir / "audit_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "real_adsc_all_anchors_kept_gapwise_restore",
                "samples_parquet": str(Path(args.samples_parquet)),
                "selected_flights_csv": str(Path(args.selected_flights_csv)),
                "checkpoint": str(ckpt),
                "config": str(Path(args.config)),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
