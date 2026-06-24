from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Plot clean cross-ocean ADS-C/ADS-B cases with frozen and missing ADS-B spans annotated.")
    p.add_argument("--case-csv", default="outputs/runs/clean_cross_ocean_adsc_adsb_cases_20260517/selected_clean_cross_ocean_cases.csv")
    p.add_argument("--out-dir", default="outputs/runs/clean_cross_ocean_adsc_adsb_cases_20260517/plots")
    p.add_argument("--freeze-min-len", type=int, default=2)
    p.add_argument("--gap-threshold-min", type=float, default=1.5)
    return p


def _same_latlon_runs(adsb: pd.DataFrame, min_len: int) -> list[tuple[int, int]]:
    if adsb.empty:
        return []
    same = adsb[["lat", "lon"]].round(6).eq(adsb[["lat", "lon"]].round(6).shift()).all(axis=1).to_numpy()
    runs: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(adsb)):
        if not same[i]:
            if i - start >= min_len:
                runs.append((start, i - 1))
            start = i
    if len(adsb) - start >= min_len:
        runs.append((start, len(adsb) - 1))
    return runs


def _gap_edges(adsb: pd.DataFrame, threshold_min: float) -> list[tuple[int, int, float]]:
    if len(adsb) < 2:
        return []
    t = pd.to_datetime(adsb["minute_ts"], utc=True, errors="coerce")
    dt = t.diff().dt.total_seconds().div(60.0)
    gaps: list[tuple[int, int, float]] = []
    for i, minutes in enumerate(dt):
        if i == 0 or not np.isfinite(minutes):
            continue
        if float(minutes) > float(threshold_min):
            gaps.append((i - 1, i, float(minutes)))
    return gaps


def _plot_line_segments(ax: plt.Axes, x: np.ndarray, y: np.ndarray, *, color: str, lw: float, label: str | None = None) -> None:
    if len(x) < 2:
        return
    ax.plot(x, y, color=color, lw=lw, label=label)


def plot_case(row: pd.Series, out_dir: Path, freeze_min_len: int, gap_threshold_min: float) -> dict[str, object]:
    pair_id = str(row["pair_id"])
    adsb = pd.read_csv(ROOT / str(row["adsb_minute_csv"]), parse_dates=["minute_ts"]).sort_values("minute_ts").reset_index(drop=True)
    adsc = pd.read_csv(ROOT / str(row["adsc_anchor_csv"]), parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if adsb.empty or adsc.empty:
        return {"pair_id": pair_id, "status": "empty"}

    adsb["minute_ts"] = pd.to_datetime(adsb["minute_ts"], utc=True, errors="coerce")
    adsc["timestamp"] = pd.to_datetime(adsc["timestamp"], utc=True, errors="coerce")
    freeze_runs = _same_latlon_runs(adsb, min_len=int(freeze_min_len))
    gaps = _gap_edges(adsb, threshold_min=float(gap_threshold_min))

    t0 = adsb["minute_ts"].min()
    x_adsb = (adsb["minute_ts"] - t0).dt.total_seconds().div(60.0).to_numpy(dtype=float)
    x_adsc = (adsc["timestamp"] - t0).dt.total_seconds().div(60.0).to_numpy(dtype=float)
    lon = pd.to_numeric(adsb["lon"], errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(adsb["lat"], errors="coerce").to_numpy(dtype=float)
    alt = pd.to_numeric(adsb["alt"], errors="coerce").to_numpy(dtype=float)

    fig, (ax_map, ax_alt) = plt.subplots(2, 1, figsize=(13.2, 7.8), facecolor="white")
    ax_map.set_facecolor("white")
    ax_alt.set_facecolor("white")

    gap_after_idx = {a for a, _, _ in gaps}
    start = 0
    normal_labeled = False
    for i in range(len(adsb) - 1):
        if i in gap_after_idx:
            _plot_line_segments(
                ax_map,
                lon[start : i + 1],
                lat[start : i + 1],
                color="#1f77b4",
                lw=1.45,
                label="ADS-B normal" if not normal_labeled else None,
            )
            _plot_line_segments(
                ax_alt,
                x_adsb[start : i + 1],
                alt[start : i + 1],
                color="#1f77b4",
                lw=1.45,
                label="ADS-B normal" if not normal_labeled else None,
            )
            normal_labeled = True
            start = i + 1
    _plot_line_segments(
        ax_map,
        lon[start:],
        lat[start:],
        color="#1f77b4",
        lw=1.45,
        label="ADS-B normal" if not normal_labeled else None,
    )
    _plot_line_segments(
        ax_alt,
        x_adsb[start:],
        alt[start:],
        color="#1f77b4",
        lw=1.45,
        label="ADS-B normal" if not normal_labeled else None,
    )

    freeze_labeled = False
    for a, b in freeze_runs:
        ax_map.plot(
            lon[a : b + 1],
            lat[a : b + 1],
            color="#d62728",
            lw=3.2,
            solid_capstyle="round",
            label="ADS-B frozen lat/lon" if not freeze_labeled else None,
            zorder=6,
        )
        ax_map.scatter(lon[a : b + 1], lat[a : b + 1], s=13, color="#d62728", alpha=0.85, zorder=7)
        ax_alt.plot(
            x_adsb[a : b + 1],
            alt[a : b + 1],
            color="#d62728",
            lw=3.2,
            solid_capstyle="round",
            label="ADS-B frozen lat/lon" if not freeze_labeled else None,
            zorder=6,
        )
        freeze_labeled = True

    gap_labeled = False
    for left, right, minutes in gaps:
        ax_map.plot(
            [lon[left], lon[right]],
            [lat[left], lat[right]],
            color="#ff9f1c",
            lw=2.4,
            linestyle=(0, (4, 3)),
            label="ADS-B missing interval" if not gap_labeled else None,
            zorder=5,
        )
        ax_map.scatter([lon[left], lon[right]], [lat[left], lat[right]], s=28, color="#ff9f1c", zorder=8)
        ax_alt.axvspan(x_adsb[left], x_adsb[right], color="#ff9f1c", alpha=0.22, linewidth=0)
        ax_alt.plot(
            [x_adsb[left], x_adsb[right]],
            [alt[left], alt[right]],
            color="#ff9f1c",
            lw=2.0,
            linestyle=(0, (4, 3)),
            label="ADS-B missing interval" if not gap_labeled else None,
            zorder=5,
        )
        x_mid = (x_adsb[left] + x_adsb[right]) / 2.0
        y_mid = np.nanmax(alt) if np.isfinite(np.nanmax(alt)) else 0.0
        if minutes >= 30:
            ax_alt.text(x_mid, y_mid, f"{minutes:.0f} min gap", color="#9a5a00", fontsize=7, ha="center", va="bottom")
        gap_labeled = True

    ax_map.scatter(
        pd.to_numeric(adsc["longitude"], errors="coerce"),
        pd.to_numeric(adsc["latitude"], errors="coerce"),
        color="#2ca02c",
        edgecolor="#111111",
        linewidth=0.5,
        s=42,
        zorder=9,
        label="ADS-C anchors",
    )
    ax_alt.scatter(
        x_adsc,
        pd.to_numeric(adsc["altitude_m"], errors="coerce"),
        color="#2ca02c",
        edgecolor="#111111",
        linewidth=0.5,
        s=42,
        zorder=9,
        label="ADS-C anchors",
    )

    ax_map.set_title(f"{pair_id} | ADS-B quality annotated: frozen segments and missing intervals")
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.grid(alpha=0.25)
    ax_map.legend(fontsize=8, ncol=2, loc="best")

    ax_alt.set_xlabel("Minutes from ADS-B flight start")
    ax_alt.set_ylabel("Altitude (m)")
    ax_alt.grid(alpha=0.25)
    ax_alt.legend(fontsize=8, ncol=2, loc="best")

    fig.tight_layout()
    out_png = out_dir / f"{pair_id}_overlay_annotated.png"
    fig.savefig(out_png, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "pair_id": pair_id,
        "status": "ok",
        "annotated_plot_png": str(out_png.relative_to(ROOT)),
        "freeze_run_count": len(freeze_runs),
        "freeze_minutes": int(sum(b - a + 1 for a, b in freeze_runs)),
        "missing_gap_count": len(gaps),
        "missing_gap_minutes_total": float(sum(minutes - 1.0 for _, _, minutes in gaps)),
        "max_missing_gap_min": float(max([g[2] for g in gaps], default=0.0)),
    }


def main() -> int:
    args = build_parser().parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(ROOT / args.case_csv)
    rows = []
    for _, row in cases.iterrows():
        rows.append(plot_case(row, out_dir, int(args.freeze_min_len), float(args.gap_threshold_min)))
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "annotated_plot_summary.csv", index=False)
    print(f"[done] plots={len(summary)} out={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
