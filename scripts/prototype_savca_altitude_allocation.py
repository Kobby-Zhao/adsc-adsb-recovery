from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _smooth(x: np.ndarray, window: int = 5) -> np.ndarray:
    if len(x) < 3 or window <= 1:
        return x.astype(float, copy=True)
    window = int(max(3, window))
    if window % 2 == 0:
        window += 1
    pad = window // 2
    xp = np.pad(x.astype(float), (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(xp, kernel, mode="valid")


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _edge_taper(n: int, edge_steps: int = 4) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=float)
    if edge_steps <= 0:
        return np.ones(n, dtype=float)
    idx = np.arange(1, n + 1, dtype=float)
    left = np.clip(idx / float(edge_steps + 1), 0.0, 1.0)
    right = np.clip((n + 1 - idx) / float(edge_steps + 1), 0.0, 1.0)
    return np.minimum(left, right)


def _allocation_from_backbone(
    main_alt: np.ndarray,
    dz: float,
    n_steps: int,
    *,
    temp: float = 2.0,
    min_uniform: float = 0.08,
    smooth_window: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized vertical-change allocation and state proxy.

    Allocation lives on intervals from the left anchor to each next minute. For a
    gap with n interior minutes, there are n+1 intervals; the final interval
    lands exactly on the right anchor.
    """
    if n_steps <= 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    diff = np.diff(_smooth(main_alt, smooth_window))
    if len(diff) != n_steps:
        diff = np.resize(diff, n_steps)
    if dz >= 0:
        signed = np.maximum(diff, 0.0)
    else:
        signed = np.maximum(-diff, 0.0)
    mag = np.abs(diff)
    raw = 0.70 * signed + 0.30 * mag
    if np.nanmax(raw) > 1e-6:
        state = raw / (np.nanmax(raw) + 1e-6)
        score = np.exp(temp * state)
    else:
        state = np.zeros_like(raw)
        score = np.ones_like(raw)
    score = np.nan_to_num(score, nan=1.0, posinf=1.0, neginf=1.0)
    uniform = np.ones(n_steps, dtype=float) / float(n_steps)
    p = score / max(float(score.sum()), 1e-6)
    p = (1.0 - min_uniform) * p + min_uniform * uniform
    p = p / max(float(p.sum()), 1e-6)
    return p, state


def _savca_segment(
    z_left: float,
    z_right: float,
    main_alt_full: np.ndarray,
    linear_full: np.ndarray | None,
    a3_full: np.ndarray | None,
    left: int,
    right: int,
    *,
    small_delta_m: float = 30.0,
    short_gap_min: int = 8,
    residual_cap_m: float = 35.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = right - left - 1
    if n <= 0:
        return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
    dz = float(z_right - z_left)
    main_seg = main_alt_full[left : right + 1].astype(float)
    if not np.isfinite(main_seg).all():
        main_seg = np.linspace(z_left, z_right, n + 2)

    if abs(dz) <= small_delta_m:
        base_inner = np.full(n, z_left + dz * 0.5, dtype=float)
        p = np.ones(n + 1, dtype=float) / float(n + 1)
        state = np.zeros(n + 1, dtype=float)
    elif n + 1 <= short_gap_min:
        p = np.ones(n + 1, dtype=float) / float(n + 1)
        state = np.ones(n + 1, dtype=float) * 0.5
        cum = np.cumsum(p)[:-1]
        base_inner = z_left + dz * cum
    else:
        p, state = _allocation_from_backbone(main_seg, dz, n + 1)
        cum = np.cumsum(p)[:-1]
        base_inner = z_left + dz * cum

    residual = np.zeros(n, dtype=float)
    if linear_full is not None and a3_full is not None:
        raw = a3_full[left + 1 : right] - linear_full[left + 1 : right]
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        residual = np.clip(raw, -residual_cap_m, residual_cap_m) * _edge_taper(n)
    final = base_inner + residual
    return base_inner, final, p, state


def _oracle_segment(
    truth_full: np.ndarray,
    left: int,
    right: int,
    *,
    residual_cap_m: float = 35.0,
) -> tuple[np.ndarray, np.ndarray]:
    n = right - left - 1
    if n <= 0:
        return np.zeros(0), np.zeros(0)
    z_left = float(truth_full[left])
    z_right = float(truth_full[right])
    dz = z_right - z_left
    truth_gap = truth_full[left + 1 : right]
    diffs = np.diff(truth_full[left : right + 1])
    if not np.isfinite(diffs).all() or abs(dz) < 1e-6:
        base = np.full(n, z_left, dtype=float)
    else:
        signed = np.maximum(diffs, 0.0) if dz > 0 else np.maximum(-diffs, 0.0)
        mass = signed if float(np.nansum(signed)) > 1e-6 else np.abs(diffs)
        if float(np.nansum(mass)) < 1e-6:
            p = np.ones(n + 1, dtype=float) / float(n + 1)
        else:
            p = np.nan_to_num(mass, nan=0.0, posinf=0.0, neginf=0.0)
            p = p / max(float(p.sum()), 1e-6)
        base = z_left + dz * np.cumsum(p)[:-1]
    residual = np.clip(truth_gap - base, -residual_cap_m, residual_cap_m) * _edge_taper(n)
    return base, base + residual


def _apply_savca(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    n = len(out)
    if "rel_min" not in out.columns and "minute_index" in out.columns:
        out["rel_min"] = out["minute_index"]
    if "is_adsc_anchor" not in out.columns and "obs_mask" in out.columns:
        out["is_adsc_anchor"] = out["obs_mask"]
    if "A0-backbone_pred_alt_m" not in out.columns and "A0-backbone_alt_m" in out.columns:
        out["A0-backbone_pred_alt_m"] = out["A0-backbone_alt_m"]
    if "A1-anchor-main_pred_alt_m" not in out.columns and "A1-anchor-main_alt_m" in out.columns:
        out["A1-anchor-main_pred_alt_m"] = out["A1-anchor-main_alt_m"]
    if "A3-gated-routed_pred_alt_m" not in out.columns and "A3-gated-routed_alt_m" in out.columns:
        out["A3-gated-routed_pred_alt_m"] = out["A3-gated-routed_alt_m"]
    out["SAVCA_base_alt_m"] = np.nan
    out["SAVCA_proto_alt_m"] = np.nan
    out["SAVCA_oracle_base_alt_m"] = np.nan
    out["SAVCA_oracle_res35_alt_m"] = np.nan
    out["SAVCA_alloc_p"] = np.nan
    out["SAVCA_state_proxy"] = np.nan
    anchors = np.flatnonzero(pd.to_numeric(out["is_adsc_anchor"], errors="coerce").fillna(0).to_numpy(dtype=int) == 1)
    truth = pd.to_numeric(out["adsb_alt_m"], errors="coerce").to_numpy(dtype=float)
    anchor_alt = pd.to_numeric(out["adsc_anchor_alt_m"], errors="coerce").to_numpy(dtype=float)
    a0 = pd.to_numeric(out.get("A0-backbone_pred_alt_m", pd.Series(np.nan, index=out.index)), errors="coerce").to_numpy(dtype=float)
    a1 = pd.to_numeric(out.get("A1-anchor-main_pred_alt_m", pd.Series(np.nan, index=out.index)), errors="coerce").to_numpy(dtype=float)
    a3 = pd.to_numeric(out.get("A3-gated-routed_pred_alt_m", pd.Series(np.nan, index=out.index)), errors="coerce").to_numpy(dtype=float)
    rows: list[dict] = []
    for li, ri in zip(anchors[:-1], anchors[1:]):
        if ri <= li + 1:
            continue
        z_left = anchor_alt[li] if np.isfinite(anchor_alt[li]) else truth[li]
        z_right = anchor_alt[ri] if np.isfinite(anchor_alt[ri]) else truth[ri]
        base, final, p, state = _savca_segment(
            z_left,
            z_right,
            a0,
            a1 if np.isfinite(a1).any() else None,
            a3 if np.isfinite(a3).any() else None,
            li,
            ri,
        )
        idx = np.arange(li + 1, ri)
        out.loc[out.index[idx], "SAVCA_base_alt_m"] = base
        out.loc[out.index[idx], "SAVCA_proto_alt_m"] = final
        if np.isfinite(truth[[li, ri]]).all() and np.isfinite(truth[idx]).all():
            oracle_base, oracle_res = _oracle_segment(truth, li, ri)
            out.loc[out.index[idx], "SAVCA_oracle_base_alt_m"] = oracle_base
            out.loc[out.index[idx], "SAVCA_oracle_res35_alt_m"] = oracle_res
        # Store interval allocation on the minute reached by that interval.
        alloc_idx = np.arange(li + 1, ri + 1)
        out.loc[out.index[alloc_idx], "SAVCA_alloc_p"] = p
        out.loc[out.index[alloc_idx], "SAVCA_state_proxy"] = state
        gap_truth = truth[idx]
        for name, pred in [
            ("A0-backbone", a0[idx]),
            ("A1-anchor-main", a1[idx]),
            ("A3-gated-routed", a3[idx]),
            ("SAVCA-base", base),
            ("SAVCA-proto", final),
            ("SAVCA-oracle-base", out["SAVCA_oracle_base_alt_m"].to_numpy(dtype=float)[idx]),
            ("SAVCA-oracle-res35", out["SAVCA_oracle_res35_alt_m"].to_numpy(dtype=float)[idx]),
        ]:
            ok = np.isfinite(gap_truth) & np.isfinite(pred)
            if not ok.any():
                continue
            err = pred[ok] - gap_truth[ok]
            rows.append(
                {
                    "case_id": str(out.get("case_id", pd.Series([""])).iloc[0]) if "case_id" in out else "",
                    "gap_left_rel_min": int(out["rel_min"].iloc[li]),
                    "gap_right_rel_min": int(out["rel_min"].iloc[ri]),
                    "gap_len_min": int(ri - li - 1),
                    "anchor_delta_alt_m": float(z_right - z_left),
                    "model": name,
                    "gap_alt_RMSE_m": float(np.sqrt(np.mean(err * err))),
                    "gap_alt_MAE_m": float(np.mean(np.abs(err))),
                    "gap_alt_MaxAE_m": float(np.max(np.abs(err))),
                }
            )
    out.loc[anchors, "SAVCA_base_alt_m"] = anchor_alt[anchors]
    out.loc[anchors, "SAVCA_proto_alt_m"] = anchor_alt[anchors]
    out.loc[anchors, "SAVCA_oracle_base_alt_m"] = truth[anchors]
    out.loc[anchors, "SAVCA_oracle_res35_alt_m"] = truth[anchors]
    return out, pd.DataFrame(rows)


def _plot_case(df: pd.DataFrame, path: Path, title: str) -> None:
    x = pd.to_numeric(df["rel_min"], errors="coerce").to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    ax.plot(x, df["adsb_alt_m"], color="black", lw=1.6, label="ADS-B truth")
    anchors = pd.to_numeric(df["is_adsc_anchor"], errors="coerce").fillna(0).to_numpy(dtype=int) == 1
    ax.scatter(x[anchors], df.loc[anchors, "adsc_anchor_alt_m"], marker="*", s=120, color="black", label="ADS-C anchors", zorder=8)
    for col, label, color, ls in [
        ("A1-anchor-main_pred_alt_m", "A1 linear", "#888888", "--"),
        ("A3-gated-routed_pred_alt_m", "Old A3", "#f28e2b", "-"),
        ("SAVCA_base_alt_m", "SAVCA base", "#4e79a7", "-"),
        ("SAVCA_proto_alt_m", "SAVCA + bounded residual", "#d62728", "-"),
        ("SAVCA_oracle_base_alt_m", "SAVCA oracle base", "#9467bd", ":"),
    ]:
        if col in df:
            ax.plot(x, df[col], lw=1.4, color=color, ls=ls, label=label)
    ax.set_ylabel("Altitude (m)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=9)
    ax2 = axes[1]
    ax2.plot(x, df["SAVCA_alloc_p"], color="#4e79a7", lw=1.2, label="allocation p")
    ax2.plot(x, df["SAVCA_state_proxy"], color="#59a14f", lw=1.2, label="state proxy")
    ax2.set_xlabel("Relative minute")
    ax2.set_ylabel("SAVCA")
    ax2.grid(alpha=0.25)
    ax2.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    return (
        metrics.groupby("model", as_index=False)
        .agg(
            gap_count=("gap_alt_RMSE_m", "count"),
            gap_alt_RMSE_mean_m=("gap_alt_RMSE_m", "mean"),
            gap_alt_MAE_mean_m=("gap_alt_MAE_m", "mean"),
            gap_alt_MaxAE_mean_m=("gap_alt_MaxAE_m", "mean"),
        )
        .sort_values("gap_alt_RMSE_mean_m")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Prototype SAVCA altitude allocation on existing recovery CSVs.")
    parser.add_argument("--input-dir", default="outputs/runs/paper_showcase_cross_ocean_gated_height_ablation_20260520")
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--out-dir", default="outputs/runs/savca_prototype_from_existing_recovery_20260520")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: list[pd.DataFrame] = []
    if args.input_csv:
        csv_paths = [Path(args.input_csv)]
    else:
        input_dir = Path(args.input_dir)
        csv_paths = sorted(input_dir.glob("*/recovered_minute_compare.csv"))
    for csv_path in csv_paths:
        df_all = pd.read_csv(csv_path)
        if "sample_id" in df_all.columns:
            groups = [(str(sid), g.reset_index(drop=True)) for sid, g in df_all.groupby("sample_id", sort=False)]
        else:
            groups = [(csv_path.parent.name, df_all)]
        for case_id, df in groups:
            df = df.copy()
            df["case_id"] = case_id
            out_df, metrics = _apply_savca(df)
            case_dir = out_dir / case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            out_df.to_csv(case_dir / "recovered_minute_compare_with_savca_proto.csv", index=False, encoding="utf-8-sig")
            if not metrics.empty:
                metrics["case_id"] = case_id
                metrics.to_csv(case_dir / "savca_proto_gap_metrics.csv", index=False, encoding="utf-8-sig")
                all_metrics.append(metrics)
            _plot_case(out_df, case_dir / "savca_proto_altitude_compare.png", case_id)
    if all_metrics:
        metrics_all = pd.concat(all_metrics, ignore_index=True)
        metrics_all.to_csv(out_dir / "savca_proto_gap_metrics_all.csv", index=False, encoding="utf-8-sig")
        summary = _summarize(metrics_all)
        summary.to_csv(out_dir / "savca_proto_gap_metrics_summary.csv", index=False, encoding="utf-8-sig")
        print(summary.to_string(index=False))
    else:
        print("No metrics generated.")
    print(f"[out] {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
