from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "outputs/runs/paper_showcase_cross_ocean_all_model_compare_20260518"
OUT_DIR = ROOT / "outputs/runs/shape_aware_altitude_postprocess_20260519_v2_conservative"

WINDOWS = {
    "39d2a8_0013": (150, 560),
    "407fcd_0019": (150, 530),
    "4076e8_0021": (150, 540),
    "a9c5c2_0001": (70, 350),
    "407943_0020": (50, 500),
}


@dataclass
class SegmentDecision:
    case_id: str
    start_idx: int
    end_idx: int
    start_time: str
    end_time: str
    gap_minutes: int
    left_alt_m: float
    right_alt_m: float
    delta_alt_m: float
    shape_type: str
    transition_idx: int | None
    transition_rel: float | None
    reason: str


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _infer_step_center_from_bilstm(seg: pd.DataFrame) -> int:
    vals = pd.to_numeric(seg["BiLSTM-clean_pred_alt"], errors="coerce").to_numpy(dtype=float)
    if len(vals) < 3 or np.all(~np.isfinite(vals)):
        return len(seg) // 2
    vals = pd.Series(vals).interpolate(limit_direction="both").to_numpy(dtype=float)
    grad = np.abs(np.diff(vals))
    if len(grad) == 0 or not np.isfinite(grad).any():
        return len(seg) // 2
    return int(np.nanargmax(grad) + 1)


def _segment_shape_aware_alt(seg: pd.DataFrame, case_id: str) -> tuple[np.ndarray, SegmentDecision]:
    old = pd.to_numeric(seg["Ours-A3_pred_alt"], errors="coerce").to_numpy(dtype=float)
    bilstm = pd.to_numeric(seg["BiLSTM-clean_pred_alt"], errors="coerce").to_numpy(dtype=float)
    n = len(seg)
    left_alt = float(seg["alt"].iloc[0])
    right_alt = float(seg["alt"].iloc[-1])
    delta = right_alt - left_alt
    abs_delta = abs(delta)
    gap_minutes = max(0, n - 1)
    alpha = np.linspace(0.0, 1.0, n)

    if n <= 2:
        shape = "too_short"
        out = old.copy()
        trans_idx = None
        reason = "gap too short; keep old A3"
    elif abs_delta < 45.0:
        shape = "stable"
        out = old.copy()
        trans_idx = None
        reason = "|anchor delta| < 45 m; stable segment, keep old A3"
    elif abs_delta >= 180.0 and gap_minutes >= 20:
        trans_idx = _infer_step_center_from_bilstm(seg)
        trans_rel = trans_idx / max(1, n - 1)
        if trans_rel <= 0.12 or trans_rel >= 0.88:
            shape = "gradual"
            out = old.copy()
            reason = "large anchor delta, but inferred transition is too close to boundary; keep old A3"
            trans_idx = None
            decision = SegmentDecision(
                case_id=case_id,
                start_idx=int(seg.index[0]),
                end_idx=int(seg.index[-1]),
                start_time=str(seg["minute_ts"].iloc[0]),
                end_time=str(seg["minute_ts"].iloc[-1]),
                gap_minutes=int(gap_minutes),
                left_alt_m=left_alt,
                right_alt_m=right_alt,
                delta_alt_m=float(delta),
                shape_type=shape,
                transition_idx=None,
                transition_rel=None,
                reason=reason,
            )
            return np.asarray(out, dtype=float), decision
        shape = "step"
        # ADS-B references show 1-2 min transitions; use a slightly wider 5-min
        # smooth band to avoid nonphysical discontinuity in minute-level recovery.
        half_width = max(2, min(4, gap_minutes // 12))
        left = max(0, trans_idx - half_width)
        right = min(n - 1, trans_idx + half_width)
        s = np.zeros(n, dtype=float)
        if right > left:
            s[left : right + 1] = _smoothstep(np.linspace(0.0, 1.0, right - left + 1))
            s[right + 1 :] = 1.0
        else:
            s[trans_idx:] = 1.0
        step_base = left_alt + s * delta
        # Add only a small old-A3 residual so the template does not become jagged.
        lin = left_alt + alpha * delta
        residual = np.nan_to_num(old - lin, nan=0.0)
        out = step_base + 0.15 * residual
        out[0] = left_alt
        out[-1] = right_alt
        reason = "|anchor delta| >= 180 m and long enough; transition center inferred from BiLSTM max gradient"
    else:
        shape = "gradual"
        out = old.copy()
        trans_idx = None
        reason = "moderate anchor delta; keep old A3 gradual trend"

    out = np.asarray(out, dtype=float)
    decision = SegmentDecision(
        case_id=case_id,
        start_idx=int(seg.index[0]),
        end_idx=int(seg.index[-1]),
        start_time=str(seg["minute_ts"].iloc[0]),
        end_time=str(seg["minute_ts"].iloc[-1]),
        gap_minutes=int(gap_minutes),
        left_alt_m=left_alt,
        right_alt_m=right_alt,
        delta_alt_m=float(delta),
        shape_type=shape,
        transition_idx=None if trans_idx is None else int(seg.index[min(max(trans_idx, 0), n - 1)]),
        transition_rel=None if trans_idx is None or n <= 1 else float(trans_idx / (n - 1)),
        reason=reason,
    )
    return out, decision


def apply_shape_aware(df: pd.DataFrame, case_id: str) -> tuple[pd.DataFrame, list[SegmentDecision]]:
    x = df.sort_values("minute_ts").reset_index(drop=True).copy()
    x["A3-shape-aware_pred_alt"] = pd.to_numeric(x["Ours-A3_pred_alt"], errors="coerce")
    decisions: list[SegmentDecision] = []
    anchors = np.where(pd.to_numeric(x["is_adsc_anchor"], errors="coerce").fillna(0).to_numpy() > 0.5)[0]
    if len(anchors) < 2:
        return x, decisions
    for a, b in zip(anchors[:-1], anchors[1:]):
        if b <= a:
            continue
        seg = x.iloc[a : b + 1].copy()
        if len(seg) < 2:
            continue
        alt, decision = _segment_shape_aware_alt(seg, case_id)
        x.loc[a:b, "A3-shape-aware_pred_alt"] = alt
        decisions.append(decision)
    return x, decisions


def _metrics_for_case(df: pd.DataFrame) -> dict[str, float]:
    gap = pd.to_numeric(df["is_adsc_anchor"], errors="coerce").fillna(0).eq(0)
    has_truth = pd.to_numeric(df["alt"], errors="coerce").notna()
    mask = gap & has_truth
    out: dict[str, float] = {}
    for col, prefix in [("Ours-A3_pred_alt", "old_a3"), ("A3-shape-aware_pred_alt", "shape_aware")]:
        err = pd.to_numeric(df.loc[mask, col], errors="coerce") - pd.to_numeric(df.loc[mask, "alt"], errors="coerce")
        out[f"{prefix}_gap_alt_rmse_m"] = float(np.sqrt(np.nanmean(np.square(err)))) if len(err) else np.nan
        out[f"{prefix}_gap_alt_mae_m"] = float(np.nanmean(np.abs(err))) if len(err) else np.nan
        pred = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        d = np.diff(pred[np.isfinite(pred)])
        out[f"{prefix}_vertical_roughness_m"] = float(np.nanmean(np.abs(np.diff(d)))) if len(d) > 1 else np.nan
        out[f"{prefix}_max_step_m_per_min"] = float(np.nanmax(np.abs(d))) if len(d) else np.nan
    return out


def _plot_case(case_id: str, df: pd.DataFrame, out_png: Path, window: tuple[int, int] | None = None) -> None:
    x = df.copy()
    x["minute_ts"] = pd.to_datetime(x["minute_ts"], utc=True)
    t0 = x["minute_ts"].min()
    x["rel_min"] = (x["minute_ts"] - t0).dt.total_seconds() / 60.0
    if window is not None:
        x = x[(x["rel_min"] >= window[0]) & (x["rel_min"] <= window[1])].copy()

    fig, ax = plt.subplots(figsize=(12.5, 5.8), facecolor="white")
    rel = x["rel_min"].to_numpy(dtype=float)
    ax.plot(rel, pd.to_numeric(x["alt"], errors="coerce"), color="black", lw=1.6, label="ADS-B truth/reference")
    ax.scatter(
        rel[x["is_adsc_anchor"].astype(int).eq(1)],
        pd.to_numeric(x.loc[x["is_adsc_anchor"].astype(int).eq(1), "alt"], errors="coerce"),
        color="black",
        marker="*",
        s=120,
        zorder=5,
        label="ADS-C anchors",
    )
    ax.plot(rel, pd.to_numeric(x["Ours-A3_pred_alt"], errors="coerce"), color="#d62728", lw=2.0, label="Old A3")
    ax.plot(
        rel,
        pd.to_numeric(x["A3-shape-aware_pred_alt"], errors="coerce"),
        color="#0072b2",
        lw=2.2,
        label="A3 shape-aware postprocess",
    )
    ax.plot(rel, pd.to_numeric(x["BiLSTM-clean_pred_alt"], errors="coerce"), color="#f0a202", lw=1.4, alpha=0.75, label="BiLSTM reference")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title(case_id)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_decisions: list[dict] = []
    metric_rows: list[dict] = []
    case_dirs = [p for p in BASE_DIR.iterdir() if p.is_dir() and (p / "recovered_minute_compare.csv").exists()]
    for case_dir in sorted(case_dirs):
        case_id = case_dir.name
        recovered = pd.read_csv(case_dir / "recovered_minute_compare.csv")
        if "Ours-A3_pred_alt" not in recovered or "BiLSTM-clean_pred_alt" not in recovered:
            continue
        out_df, decisions = apply_shape_aware(recovered, case_id)
        out_case = OUT_DIR / case_id
        out_case.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_case / "recovered_minute_compare_shape_aware.csv", index=False, encoding="utf-8-sig")
        _plot_case(case_id, out_df, out_case / "altitude_shape_aware_full.png")
        if case_id in WINDOWS:
            _plot_case(case_id, out_df, out_case / f"altitude_shape_aware_window_{WINDOWS[case_id][0]}_{WINDOWS[case_id][1]}.png", WINDOWS[case_id])
        for d in decisions:
            all_decisions.append(d.__dict__)
        row = {"case_id": case_id}
        row.update(_metrics_for_case(out_df))
        metric_rows.append(row)

    pd.DataFrame(all_decisions).to_csv(OUT_DIR / "shape_aware_segment_decisions.csv", index=False, encoding="utf-8-sig")
    metrics = pd.DataFrame(metric_rows)
    if not metrics.empty:
        metrics["delta_gap_alt_rmse_m"] = metrics["shape_aware_gap_alt_rmse_m"] - metrics["old_a3_gap_alt_rmse_m"]
        metrics["delta_gap_alt_mae_m"] = metrics["shape_aware_gap_alt_mae_m"] - metrics["old_a3_gap_alt_mae_m"]
        metrics["delta_vertical_roughness_m"] = metrics["shape_aware_vertical_roughness_m"] - metrics["old_a3_vertical_roughness_m"]
    metrics.to_csv(OUT_DIR / "shape_aware_case_metrics.csv", index=False, encoding="utf-8-sig")
    print(f"[done] out_dir={OUT_DIR}")
    print(f"[done] cases={len(metric_rows)} decisions={len(all_decisions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
