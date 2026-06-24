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
OUT_DIR = ROOT / "outputs/runs/oracle_step_altitude_upper_bound_20260519"

WINDOWS = {
    "39d2a8_0013": (150, 560),
    "407fcd_0019": (150, 530),
    "4076e8_0021": (150, 540),
    "a9c5c2_0001": (70, 350),
    "407943_0020": (50, 500),
}


@dataclass
class GapOracleResult:
    case_id: str
    gap_id: int
    start_idx: int
    end_idx: int
    start_time: str
    end_time: str
    gap_minutes: int
    truth_count: int
    left_alt_m: float
    right_alt_m: float
    delta_alt_m: float
    best_center_rel: float
    best_width_min: int
    linear_rmse_m: float
    oracle_step_rmse_m: float
    oracle_best_template: str
    oracle_best_rmse_m: float
    old_a3_rmse_m: float
    a3_plus_oracle_residual_rmse_m: float
    linear_mae_m: float
    oracle_step_mae_m: float
    oracle_best_mae_m: float
    old_a3_mae_m: float
    a3_plus_oracle_residual_mae_m: float
    oracle_gain_vs_linear_rmse_m: float
    oracle_gain_vs_old_a3_rmse_m: float
    a3_oracle_gain_vs_old_a3_rmse_m: float
    subset_long_gap: int
    subset_large_alt_delta: int
    subset_core: int


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def make_step_curve(n: int, left_alt: float, right_alt: float, center: int, width: int) -> np.ndarray:
    """Build a smooth platform-to-platform step curve with hard endpoint consistency."""
    if n <= 1:
        return np.array([left_alt], dtype=float)
    center = int(np.clip(center, 0, n - 1))
    width = max(1, int(width))
    half = max(1, width // 2)
    start = max(0, center - half)
    end = min(n - 1, center + half)
    s = np.zeros(n, dtype=float)
    if end > start:
        s[start : end + 1] = smoothstep(np.linspace(0.0, 1.0, end - start + 1))
        s[end + 1 :] = 1.0
    else:
        s[center:] = 1.0
    out = left_alt + s * (right_alt - left_alt)
    out[0] = left_alt
    out[-1] = right_alt
    return out


def rmse(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    e = pred[mask] - truth[mask]
    e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e * e))) if e.size else float("nan")


def mae(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    e = pred[mask] - truth[mask]
    e = e[np.isfinite(e)]
    return float(np.mean(np.abs(e))) if e.size else float("nan")


def best_oracle_step(truth: np.ndarray, mask: np.ndarray, left_alt: float, right_alt: float) -> tuple[np.ndarray, int, int, float]:
    n = len(truth)
    best_curve = None
    best_center = 0
    best_width = 1
    best_score = float("inf")
    # Minute-level ADS-B references show 1-2 min transitions, but allow wider
    # smooth bands to test whether shape is useful without imposing a hard jump.
    candidate_widths = sorted({1, 2, 3, 5, 7, max(3, min(15, n // 10))})
    for width in candidate_widths:
        for center in range(1, max(2, n - 1)):
            curve = make_step_curve(n, left_alt, right_alt, center=center, width=width)
            score = rmse(curve, truth, mask)
            if np.isfinite(score) and score < best_score:
                best_score = score
                best_curve = curve
                best_center = center
                best_width = width
    if best_curve is None:
        best_curve = np.linspace(left_alt, right_alt, n)
        best_score = rmse(best_curve, truth, mask)
    return best_curve, best_center, best_width, best_score


def evaluate_gap(case_id: str, gap_id: int, seg: pd.DataFrame) -> tuple[pd.DataFrame, GapOracleResult] | None:
    n = len(seg)
    if n < 2:
        return None
    truth = pd.to_numeric(seg["alt"], errors="coerce").to_numpy(dtype=float)
    old_a3 = pd.to_numeric(seg["Ours-A3_pred_alt"], errors="coerce").to_numpy(dtype=float)
    left_alt = float(truth[0])
    right_alt = float(truth[-1])
    linear = np.linspace(left_alt, right_alt, n)
    mask = np.isfinite(truth)
    # Evaluate recovery only inside the gap, not on hard anchors.
    if n > 2:
        mask[[0, -1]] = False
    if mask.sum() < 2:
        return None

    oracle_step, center, width, oracle_rmse = best_oracle_step(truth, mask, left_alt, right_alt)
    linear_rmse = rmse(linear, truth, mask)
    linear_mae = mae(linear, truth, mask)
    oracle_mae = mae(oracle_step, truth, mask)
    if oracle_rmse + 1e-9 < linear_rmse:
        oracle_best = oracle_step
        oracle_best_template = "step"
        oracle_best_rmse = oracle_rmse
        oracle_best_mae = oracle_mae
    else:
        oracle_best = linear
        oracle_best_template = "linear"
        oracle_best_rmse = linear_rmse
        oracle_best_mae = linear_mae
    # Upper bound for "A3 + shape residual": keep A3 residual, but replace the
    # geometric main trend from linear to the oracle step template.
    a3_plus_oracle = old_a3 + (oracle_step - linear)
    a3_plus_oracle[0] = left_alt
    a3_plus_oracle[-1] = right_alt

    out = seg.copy()
    out["linear_alt_m"] = linear
    out["oracle_step_alt_m"] = oracle_step
    out["oracle_best_alt_m"] = oracle_best
    out["a3_plus_oracle_residual_alt_m"] = a3_plus_oracle
    out["gap_local_idx"] = np.arange(n)
    out["oracle_step_center_idx"] = center
    out["oracle_step_width_min"] = width

    old_a3_rmse = rmse(old_a3, truth, mask)
    a3_oracle_rmse = rmse(a3_plus_oracle, truth, mask)
    old_a3_mae = mae(old_a3, truth, mask)
    a3_oracle_mae = mae(a3_plus_oracle, truth, mask)
    gap_minutes = int(n - 1)
    abs_delta = abs(right_alt - left_alt)
    result = GapOracleResult(
        case_id=case_id,
        gap_id=gap_id,
        start_idx=int(seg.index[0]),
        end_idx=int(seg.index[-1]),
        start_time=str(seg["minute_ts"].iloc[0]),
        end_time=str(seg["minute_ts"].iloc[-1]),
        gap_minutes=gap_minutes,
        truth_count=int(mask.sum()),
        left_alt_m=left_alt,
        right_alt_m=right_alt,
        delta_alt_m=float(right_alt - left_alt),
        best_center_rel=float(center / max(1, n - 1)),
        best_width_min=int(width),
        linear_rmse_m=linear_rmse,
        oracle_step_rmse_m=oracle_rmse,
        oracle_best_template=oracle_best_template,
        oracle_best_rmse_m=oracle_best_rmse,
        old_a3_rmse_m=old_a3_rmse,
        a3_plus_oracle_residual_rmse_m=a3_oracle_rmse,
        linear_mae_m=linear_mae,
        oracle_step_mae_m=oracle_mae,
        oracle_best_mae_m=oracle_best_mae,
        old_a3_mae_m=old_a3_mae,
        a3_plus_oracle_residual_mae_m=a3_oracle_mae,
        oracle_gain_vs_linear_rmse_m=linear_rmse - oracle_rmse,
        oracle_gain_vs_old_a3_rmse_m=old_a3_rmse - oracle_rmse,
        a3_oracle_gain_vs_old_a3_rmse_m=old_a3_rmse - a3_oracle_rmse,
        subset_long_gap=int(gap_minutes >= 30),
        subset_large_alt_delta=int(abs_delta >= 180),
        subset_core=int(gap_minutes >= 30 and abs_delta >= 180),
    )
    return out, result


def plot_case(case_id: str, df: pd.DataFrame, out_png: Path, window: tuple[int, int] | None = None) -> None:
    x = df.sort_values("minute_ts").copy()
    x["minute_ts"] = pd.to_datetime(x["minute_ts"], utc=True)
    t0 = x["minute_ts"].min()
    x["rel_min"] = (x["minute_ts"] - t0).dt.total_seconds() / 60.0
    if window is not None:
        x = x[(x["rel_min"] >= window[0]) & (x["rel_min"] <= window[1])].copy()
    fig, ax = plt.subplots(figsize=(13.0, 6.0), facecolor="white")
    rel = x["rel_min"].to_numpy(dtype=float)
    ax.plot(rel, pd.to_numeric(x["alt"], errors="coerce"), color="black", lw=1.8, label="ADS-B truth")
    anchors = x["is_adsc_anchor"].astype(int).eq(1)
    ax.scatter(
        rel[anchors],
        pd.to_numeric(x.loc[anchors, "alt"], errors="coerce"),
        color="black",
        marker="*",
        s=125,
        zorder=5,
        label="ADS-C anchors",
    )
    ax.plot(rel, pd.to_numeric(x["linear_alt_m"], errors="coerce"), color="#777777", lw=1.4, ls="--", label="Linear")
    ax.plot(rel, pd.to_numeric(x["Ours-A3_pred_alt"], errors="coerce"), color="#d62728", lw=2.1, label="Old A3")
    ax.plot(rel, pd.to_numeric(x["oracle_step_alt_m"], errors="coerce"), color="#0072b2", lw=2.0, label="Oracle-step")
    ax.plot(rel, pd.to_numeric(x["oracle_best_alt_m"], errors="coerce"), color="#cc79a7", lw=1.7, ls="-.", label="Oracle-best")
    ax.plot(
        rel,
        pd.to_numeric(x["a3_plus_oracle_residual_alt_m"], errors="coerce"),
        color="#009e73",
        lw=2.0,
        label="A3 + oracle-step residual",
    )
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title(case_id)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def aggregate(gaps: pd.DataFrame, subset_col: str | None = None) -> dict[str, float | int | str]:
    x = gaps if subset_col is None else gaps[gaps[subset_col].astype(int).eq(1)]
    label = "all" if subset_col is None else subset_col
    out: dict[str, float | int | str] = {"subset": label, "gap_count": int(len(x))}
    for col in [
        "linear_rmse_m",
        "oracle_step_rmse_m",
        "oracle_best_rmse_m",
        "old_a3_rmse_m",
        "a3_plus_oracle_residual_rmse_m",
        "linear_mae_m",
        "oracle_step_mae_m",
        "oracle_best_mae_m",
        "old_a3_mae_m",
        "a3_plus_oracle_residual_mae_m",
        "oracle_gain_vs_linear_rmse_m",
        "oracle_gain_vs_old_a3_rmse_m",
        "a3_oracle_gain_vs_old_a3_rmse_m",
    ]:
        out[f"{col}_mean"] = float(x[col].mean()) if len(x) else float("nan")
        out[f"{col}_median"] = float(x[col].median()) if len(x) else float("nan")
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gap_rows: list[dict] = []
    case_rows: list[dict] = []
    for case_csv in sorted(BASE_DIR.glob("*/recovered_minute_compare.csv")):
        case_id = case_csv.parent.name
        df = pd.read_csv(case_csv)
        required = {"minute_ts", "alt", "is_adsc_anchor", "Ours-A3_pred_alt"}
        if not required.issubset(df.columns):
            continue
        df = df.sort_values("minute_ts").reset_index(drop=True)
        df["linear_alt_m"] = np.nan
        df["oracle_step_alt_m"] = np.nan
        df["oracle_best_alt_m"] = np.nan
        df["a3_plus_oracle_residual_alt_m"] = np.nan
        anchors = np.where(pd.to_numeric(df["is_adsc_anchor"], errors="coerce").fillna(0).to_numpy() > 0.5)[0]
        gap_id = 0
        for left, right in zip(anchors[:-1], anchors[1:]):
            if right <= left:
                continue
            evaluated = evaluate_gap(case_id, gap_id, df.iloc[left : right + 1].copy())
            if evaluated is None:
                continue
            gap_df, res = evaluated
            for col in ["linear_alt_m", "oracle_step_alt_m", "oracle_best_alt_m", "a3_plus_oracle_residual_alt_m"]:
                df.loc[left:right, col] = gap_df[col].to_numpy()
            gap_rows.append(res.__dict__)
            gap_id += 1
        out_case = OUT_DIR / case_id
        out_case.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_case / "oracle_step_compare_full.csv", index=False, encoding="utf-8-sig")
        plot_case(case_id, df, out_case / "oracle_step_compare_full.png")
        if case_id in WINDOWS:
            plot_case(case_id, df, out_case / f"oracle_step_compare_window_{WINDOWS[case_id][0]}_{WINDOWS[case_id][1]}.png", WINDOWS[case_id])
        c = pd.DataFrame([r for r in gap_rows if r["case_id"] == case_id])
        if len(c):
            row = {"case_id": case_id, "gap_count": int(len(c))}
            for col in [
                "linear_rmse_m",
                "oracle_step_rmse_m",
                "oracle_best_rmse_m",
                "old_a3_rmse_m",
                "a3_plus_oracle_residual_rmse_m",
            ]:
                row[col] = float(c[col].mean())
            case_rows.append(row)

    gaps = pd.DataFrame(gap_rows)
    gaps.to_csv(OUT_DIR / "oracle_step_gap_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(case_rows).to_csv(OUT_DIR / "oracle_step_case_metrics.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(
        [
            aggregate(gaps, None),
            aggregate(gaps, "subset_long_gap"),
            aggregate(gaps, "subset_large_alt_delta"),
            aggregate(gaps, "subset_core"),
        ]
    )
    if len(gaps):
        template_counts = (
            gaps.groupby(["subset_core", "oracle_best_template"]).size().reset_index(name="count")
        )
        template_counts.to_csv(OUT_DIR / "oracle_best_template_counts.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "oracle_step_summary_by_subset.csv", index=False, encoding="utf-8-sig")
    print(f"[done] out_dir={OUT_DIR}")
    print(f"[done] cases={len(case_rows)} gaps={len(gaps)}")
    print(summary[[
        "subset",
        "gap_count",
        "linear_rmse_m_mean",
        "oracle_step_rmse_m_mean",
        "oracle_best_rmse_m_mean",
        "old_a3_rmse_m_mean",
        "a3_plus_oracle_residual_rmse_m_mean",
        "oracle_gain_vs_linear_rmse_m_mean",
        "a3_oracle_gain_vs_old_a3_rmse_m_mean",
    ]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
