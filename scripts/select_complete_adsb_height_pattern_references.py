from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Select complete ADS-B reference flights by cruise altitude pattern.")
    p.add_argument("--adsb-minute", default="outputs/mvp_global_202410_202503_full1000_20260414/adsb_minute.parquet")
    p.add_argument("--out-dir", default="outputs/runs/complete_adsb_height_pattern_references_20260519")
    p.add_argument("--min-flight-minutes", type=int, default=120)
    p.add_argument("--min-cruise-minutes", type=int, default=45)
    p.add_argument("--cruise-alt-threshold-m", type=float, default=9000.0)
    p.add_argument("--max-time-gap-min", type=float, default=1.5)
    p.add_argument("--max-frozen-run-min", type=int, default=1)
    return p


def _max_true_run(mask: np.ndarray) -> int:
    best = cur = 0
    for x in mask:
        if bool(x):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _longest_true_segment(mask: np.ndarray) -> tuple[int, int]:
    best_s = best_e = -1
    cur_s = None
    for i, x in enumerate(mask):
        if bool(x) and cur_s is None:
            cur_s = i
        if cur_s is not None and ((not bool(x)) or i == len(mask) - 1):
            e = i if bool(x) and i == len(mask) - 1 else i - 1
            if e - cur_s > best_e - best_s:
                best_s, best_e = cur_s, e
            cur_s = None
    return int(best_s), int(best_e)


def _find_stable_window(g: pd.DataFrame, min_len: int = 45, max_range_m: float = 90.0) -> tuple[int, int, dict[str, float]] | None:
    alt = pd.to_numeric(g["alt"], errors="coerce").to_numpy(dtype=float)
    high = alt >= 9000.0
    best: tuple[int, int] | None = None
    left = 0
    for right in range(len(alt)):
        if not high[right]:
            left = right + 1
            continue
        while left <= right and (not high[left]):
            left += 1
        while left <= right and np.nanmax(alt[left : right + 1]) - np.nanmin(alt[left : right + 1]) > max_range_m:
            left += 1
        if right - left + 1 >= min_len:
            if best is None or (right - left) > (best[1] - best[0]):
                best = (left, right)
    if best is None:
        return None
    s, e = best
    seg = alt[s : e + 1]
    return s, e, {
        "cruise_minutes": float(e - s + 1),
        "alt_range_m": float(np.nanmax(seg) - np.nanmin(seg)),
        "alt_std_m": float(np.nanstd(seg)),
        "max_step_m": float(np.nanmax(np.abs(np.diff(seg)))) if len(seg) > 1 else 0.0,
        "big_step_count_ge120m": 0.0,
        "direction_changes_ge15m": 0.0,
        "step_left_std_m": float("nan"),
        "step_right_std_m": float("nan"),
    }


def _find_single_step_segment(g: pd.DataFrame, side_len: int = 25, min_step_m: float = 120.0) -> tuple[int, int, dict[str, float]] | None:
    alt = pd.to_numeric(g["alt"], errors="coerce").to_numpy(dtype=float)
    if len(alt) < side_len * 2 + 2:
        return None
    diff = np.diff(alt)
    candidates = np.argsort(np.abs(diff))[::-1]
    for idx in candidates[:20]:
        step = float(abs(diff[idx]))
        if step < min_step_m:
            break
        ls = max(0, int(idx) - side_len + 1)
        le = int(idx)
        rs = int(idx) + 1
        re = min(len(alt) - 1, int(idx) + side_len)
        left = alt[ls : le + 1]
        right = alt[rs : re + 1]
        if len(left) < side_len or len(right) < side_len:
            continue
        if np.nanmin(left) < 9000.0 or np.nanmin(right) < 9000.0:
            continue
        left_range = float(np.nanmax(left) - np.nanmin(left))
        right_range = float(np.nanmax(right) - np.nanmin(right))
        left_std = float(np.nanstd(left))
        right_std = float(np.nanstd(right))
        median_delta = float(abs(np.nanmedian(right) - np.nanmedian(left)))
        if left_range <= 160.0 and right_range <= 160.0 and median_delta >= min_step_m:
            seg = alt[ls : re + 1]
            return ls, re, {
                "cruise_minutes": float(re - ls + 1),
                "alt_range_m": float(np.nanmax(seg) - np.nanmin(seg)),
                "alt_std_m": float(np.nanstd(seg)),
                "max_step_m": step,
                "big_step_count_ge120m": float(np.sum(np.abs(np.diff(seg)) >= 120.0)),
                "direction_changes_ge15m": 0.0,
                "step_left_std_m": left_std,
                "step_right_std_m": right_std,
            }
    return None


def _classify_cruise(seg: pd.DataFrame) -> tuple[str | None, dict[str, float]]:
    alt = pd.to_numeric(seg["alt"], errors="coerce").to_numpy(dtype=float)
    if len(alt) < 3:
        return None, {}
    diff = np.diff(alt)
    alt_range = float(np.nanmax(alt) - np.nanmin(alt))
    alt_std = float(np.nanstd(alt))
    max_step = float(np.nanmax(np.abs(diff))) if len(diff) else 0.0
    big_steps = int(np.sum(np.abs(diff) >= 120.0))
    # Direction changes after suppressing tiny quantization jitter.
    trend = np.sign(np.where(np.abs(diff) >= 15.0, diff, 0.0))
    trend = trend[trend != 0]
    direction_changes = int(np.sum(trend[1:] * trend[:-1] < 0)) if len(trend) >= 2 else 0

    # Step pattern: exactly one dominant step, and both sides relatively stable.
    step_idx = int(np.argmax(np.abs(diff))) if len(diff) else -1
    left = alt[: step_idx + 1]
    right = alt[step_idx + 1 :]
    left_std = float(np.nanstd(left)) if len(left) >= 8 else float("inf")
    right_std = float(np.nanstd(right)) if len(right) >= 8 else float("inf")

    metrics = {
        "cruise_minutes": float(len(seg)),
        "alt_range_m": alt_range,
        "alt_std_m": alt_std,
        "max_step_m": max_step,
        "big_step_count_ge120m": float(big_steps),
        "direction_changes_ge15m": float(direction_changes),
        "step_left_std_m": left_std,
        "step_right_std_m": right_std,
    }

    if alt_range <= 80.0 and alt_std <= 20.0 and max_step <= 35.0:
        return "stable_cruise", metrics
    if max_step >= 120.0 and big_steps <= 2 and left_std <= 45.0 and right_std <= 45.0:
        return "single_step_cruise", metrics
    if alt_range >= 160.0 and direction_changes >= 2 and big_steps <= 8:
        return "oscillating_cruise", metrics
    return None, metrics


def _plot_altitude(df: pd.DataFrame, cruise_s: int, cruise_e: int, title: str, out_png: Path) -> None:
    x = np.arange(len(df), dtype=float)
    alt = pd.to_numeric(df["alt"], errors="coerce")
    fig, ax = plt.subplots(figsize=(11.8, 4.8), facecolor="white")
    ax.set_facecolor("white")
    ax.axvspan(cruise_s, cruise_e, color="#bdeff2", alpha=0.45, label="selected cruise segment")
    ax.plot(x, alt, color="#111111", lw=1.6, label="ADS-B altitude")
    ax.set_title(title)
    ax.set_xlabel("Minutes from flight start")
    ax.set_ylabel("Altitude (m)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_3d(df: pd.DataFrame, cruise_s: int, cruise_e: int, title: str, out_png: Path) -> None:
    fig = plt.figure(figsize=(10.5, 7.2), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    lon = pd.to_numeric(df["lon"], errors="coerce")
    lat = pd.to_numeric(df["lat"], errors="coerce")
    alt = pd.to_numeric(df["alt"], errors="coerce")
    ax.plot(lon, lat, alt, color="#7f7f7f", lw=1.1, alpha=0.65, label="full ADS-B")
    c = df.iloc[cruise_s : cruise_e + 1]
    ax.plot(
        pd.to_numeric(c["lon"], errors="coerce"),
        pd.to_numeric(c["lat"], errors="coerce"),
        pd.to_numeric(c["alt"], errors="coerce"),
        color="#d62728",
        lw=2.0,
        label="selected cruise segment",
    )
    ax.scatter(lon.iloc[0], lat.iloc[0], alt.iloc[0], color="#2ca02c", s=32, label="start")
    ax.scatter(lon.iloc[-1], lat.iloc[-1], alt.iloc[-1], color="#1f77b4", s=32, label="end")
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Altitude (m)")
    ax.view_init(elev=18, azim=-58)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.adsb_minute)
    df["minute_ts"] = pd.to_datetime(df["minute_ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["flight_id", "minute_ts", "lat", "lon", "alt"]).copy()

    quality_rows: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    for fid, g in df.groupby("flight_id", sort=False):
        g = g.sort_values("minute_ts").reset_index(drop=True).copy()
        if len(g) < int(args.min_flight_minutes):
            continue
        dt = g["minute_ts"].diff().dt.total_seconds().div(60.0)
        max_gap = float(dt.iloc[1:].max()) if len(dt) > 1 else 0.0
        if max_gap > float(args.max_time_gap_min):
            continue
        same_pos = g[["lat", "lon"]].round(6).eq(g[["lat", "lon"]].round(6).shift()).all(axis=1).fillna(False).to_numpy()
        max_frozen_run = _max_true_run(same_pos)
        if max_frozen_run > int(args.max_frozen_run_min):
            continue
        alt_all = pd.to_numeric(g["alt"], errors="coerce").to_numpy(dtype=float)
        cruise_mask = alt_all >= float(args.cruise_alt_threshold_m)
        s, e = _longest_true_segment(cruise_mask)
        if s < 0 or (e - s + 1) < int(args.min_cruise_minutes):
            continue

        quality_base = {
            "flight_id": str(fid),
            "adsb_icao": str(g["adsb_icao"].iloc[0]) if "adsb_icao" in g.columns else "",
            "flight_minutes": int(len(g)),
            "max_time_gap_min": max_gap,
            "max_frozen_run_min": max_frozen_run,
            "start_ts": g["minute_ts"].iloc[0],
            "end_ts": g["minute_ts"].iloc[-1],
        }
        quality_rows.append(
            {
                **quality_base,
                "high_alt_start_idx": s,
                "high_alt_end_idx": e,
                "high_alt_minutes": int(e - s + 1),
                "high_alt_range_m": float(np.nanmax(alt_all[s : e + 1]) - np.nanmin(alt_all[s : e + 1])),
            }
        )

        pattern_segments: list[tuple[str, tuple[int, int, dict[str, float]] | None]] = [
            ("single_step_cruise", _find_single_step_segment(g, side_len=25, min_step_m=120.0)),
            ("stable_cruise", _find_stable_window(g, min_len=int(args.min_cruise_minutes), max_range_m=90.0)),
        ]
        high_seg = g.iloc[s : e + 1].copy()
        label, metrics = _classify_cruise(high_seg)
        if label == "oscillating_cruise":
            pattern_segments.append(("oscillating_cruise", (s, e, metrics)))

        for pattern, found in pattern_segments:
            if found is None:
                continue
            ps, pe, pmetrics = found
            candidates.append(
                {
                    **quality_base,
                    "pattern": pattern,
                    "cruise_start_idx": ps,
                    "cruise_end_idx": pe,
                    **pmetrics,
                }
            )

    cand = pd.DataFrame(candidates)
    if cand.empty:
        raise RuntimeError("No candidate flights found. Loosen thresholds.")

    # Prefer balanced categories: 4 stable, 3 single-step, 3 oscillating.
    picks: list[pd.DataFrame] = []
    quotas = {"single_step_cruise": 3, "oscillating_cruise": 3, "stable_cruise": 4}
    sort_cols = {
        "stable_cruise": ["alt_std_m", "alt_range_m", "flight_minutes"],
        "single_step_cruise": ["step_left_std_m", "step_right_std_m", "flight_minutes"],
        "oscillating_cruise": ["direction_changes_ge15m", "alt_range_m", "flight_minutes"],
    }
    ascending = {
        "stable_cruise": [True, True, False],
        "single_step_cruise": [True, True, False],
        "oscillating_cruise": [False, False, False],
    }
    used_flights: set[str] = set()
    for pattern, quota in quotas.items():
        sub = cand[cand["pattern"].eq(pattern)].copy()
        sub = sub[~sub["flight_id"].isin(used_flights)]
        if sub.empty:
            continue
        sub = sub.sort_values(sort_cols[pattern], ascending=ascending[pattern]).head(quota)
        used_flights.update(sub["flight_id"].astype(str))
        picks.append(sub)
    selected = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame()
    if len(selected) < 10:
        remaining = cand[~cand["flight_id"].isin(set(selected["flight_id"]))].copy()
        remaining = remaining.sort_values(["cruise_minutes", "flight_minutes"], ascending=[False, False]).head(10 - len(selected))
        selected = pd.concat([selected, remaining], ignore_index=True)
    selected = selected.head(10).copy()

    rows = []
    for i, row in selected.iterrows():
        fid = str(row["flight_id"])
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in fid)[:120]
        case_id = f"{i+1:02d}_{row['pattern']}_{safe}"
        case_dir = out_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        g = df[df["flight_id"].astype(str).eq(fid)].sort_values("minute_ts").reset_index(drop=True).copy()
        csv_path = case_dir / "adsb_complete_minute.csv"
        alt_png = case_dir / "altitude_2d.png"
        plot3d_png = case_dir / "trajectory_3d.png"
        g.to_csv(csv_path, index=False)
        title = f"{case_id} | {row['pattern']} | complete ADS-B reference"
        _plot_altitude(g, int(row["cruise_start_idx"]), int(row["cruise_end_idx"]), title, alt_png)
        _plot_3d(g, int(row["cruise_start_idx"]), int(row["cruise_end_idx"]), title, plot3d_png)
        out_row = dict(row)
        out_row.update(
            {
                "case_id": case_id,
                "adsb_csv": str(csv_path),
                "altitude_2d_plot": str(alt_png),
                "trajectory_3d_plot": str(plot3d_png),
            }
        )
        rows.append(out_row)

    selected_out = pd.DataFrame(rows)
    selected_out.to_csv(out_dir / "selected_complete_adsb_reference_flights.csv", index=False)
    cand.to_csv(out_dir / "candidate_complete_adsb_reference_flights.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(out_dir / "quality_pass_complete_adsb_flights.csv", index=False)
    print(f"[done] selected={out_dir / 'selected_complete_adsb_reference_flights.csv'}")
    print(selected_out[["case_id", "pattern", "flight_minutes", "cruise_minutes", "alt_range_m", "alt_std_m", "max_step_m", "direction_changes_ge15m"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
