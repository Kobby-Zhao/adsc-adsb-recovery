from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.refetch_cross_ocean_adsb_raw import _fetch_history_chunked, _normalize_history, minute_agg_adsb
from src.io_utils import load_settings


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Select clean cross-ocean ADS-C anchor cases and fetch corresponding ADS-B flights.")
    p.add_argument(
        "--cross-ocean-routes-csv",
        default="outputs/runs/cross_ocean_adsc_anchor_routes_20260517/cross_ocean_adsc_anchor_routes_selected.csv",
    )
    p.add_argument(
        "--adsc-points-csv",
        default="outputs/runs/adsc_flight_segmentation_4h_20260421_fix_epoch_v2/adsc_points_with_flight_id_4h_min2_latest1200_minute_agg.csv",
    )
    p.add_argument("--out-dir", default="outputs/runs/clean_cross_ocean_adsc_adsb_cases_20260517")
    p.add_argument("--settings", default="config/settings.yaml")
    p.add_argument("--target", type=int, default=10)
    p.add_argument("--max-candidates", type=int, default=80)
    p.add_argument("--min-anchors", type=int, default=2)
    p.add_argument("--min-duration-min", type=float, default=60.0)
    p.add_argument("--min-ocean-ratio", type=float, default=0.8)
    p.add_argument("--max-frozen-run-min", type=int, default=3)
    p.add_argument("--min-adsb-minute-rows", type=int, default=80)
    p.add_argument("--flight-pad-hours", type=float, default=8.0)
    p.add_argument("--fetch-pad-min", type=float, default=10.0)
    p.add_argument("--chunk-min", type=float, default=90.0)
    p.add_argument("--sleep-sec", type=float, default=0.5)
    p.add_argument("--cached", action="store_true", default=True)
    p.add_argument("--overwrite", action="store_true")
    return p


def _to_utc(value) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True, errors="coerce")


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value))


def _set_opensky_env(settings_path: str) -> None:
    settings = load_settings(settings_path)
    username = settings.get("opensky", {}).get("username") or os.getenv("OPENSKY_USERNAME")
    password = settings.get("opensky", {}).get("password") or os.getenv("OPENSKY_PASSWORD")
    if username:
        os.environ["OPENSKY_USERNAME"] = str(username).lower()
        os.environ["OPENSKY_TRINO_USERNAME"] = str(username).lower()
    if password:
        os.environ["OPENSKY_PASSWORD"] = str(password)
        os.environ["OPENSKY_TRINO_PASSWORD"] = str(password)
    os.environ.setdefault("TQDM_DISABLE", "1")


def _normalize_flightlist(flights: pd.DataFrame | None) -> pd.DataFrame:
    if flights is None or len(flights) == 0:
        return pd.DataFrame()
    d = flights.copy()
    cols = {c.lower(): c for c in d.columns}
    first_col = cols.get("firstseen")
    last_col = cols.get("lastseen")
    if first_col is None or last_col is None:
        return pd.DataFrame()
    if pd.api.types.is_numeric_dtype(d[first_col]):
        d["firstSeen_ts"] = pd.to_datetime(d[first_col], unit="s", utc=True, errors="coerce")
    else:
        d["firstSeen_ts"] = pd.to_datetime(d[first_col], utc=True, errors="coerce")
    if pd.api.types.is_numeric_dtype(d[last_col]):
        d["lastSeen_ts"] = pd.to_datetime(d[last_col], unit="s", utc=True, errors="coerce")
    else:
        d["lastSeen_ts"] = pd.to_datetime(d[last_col], utc=True, errors="coerce")
    for name in ["callsign", "estDepartureAirport", "estArrivalAirport"]:
        col = cols.get(name.lower())
        d[name] = d[col] if col is not None else ""
    return d.dropna(subset=["firstSeen_ts", "lastSeen_ts"]).copy()


def _max_same_latlon_run(minute: pd.DataFrame) -> dict[str, object]:
    if minute.empty:
        return {"max_frozen_run_min": 0, "frozen_start": "", "frozen_end": ""}
    d = minute.sort_values("minute_ts").reset_index(drop=True).copy()
    same = d[["lat", "lon"]].round(6).eq(d[["lat", "lon"]].round(6).shift()).all(axis=1).to_numpy()
    best_len, best_start = 1, 0
    cur_len, cur_start = 1, 0
    for i in range(1, len(d)):
        if same[i]:
            cur_len += 1
        else:
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
            cur_len, cur_start = 1, i
    if cur_len > best_len:
        best_len, best_start = cur_len, cur_start
    seg = d.iloc[best_start : best_start + best_len]
    return {
        "max_frozen_run_min": int(best_len),
        "frozen_start": str(seg["minute_ts"].iloc[0]),
        "frozen_end": str(seg["minute_ts"].iloc[-1]),
    }


def _gap_stats(minute: pd.DataFrame) -> dict[str, object]:
    if minute.empty or len(minute) < 2:
        return {"max_adsb_gap_min": np.nan, "adsb_gap_count_gt_1min": 0}
    dt = pd.to_datetime(minute["minute_ts"], utc=True).sort_values().diff().dt.total_seconds().div(60.0)
    return {"max_adsb_gap_min": float(dt.max()), "adsb_gap_count_gt_1min": int((dt > 1.5).sum())}


def _plot_overlay(pair_id: str, adsc: pd.DataFrame, minute: pd.DataFrame, out_png: Path) -> None:
    if minute.empty or adsc.empty:
        return
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12.5, 7.2), facecolor="white")
    minute = minute.sort_values("minute_ts")
    adsc = adsc.sort_values("timestamp")
    t0 = pd.to_datetime(minute["minute_ts"], utc=True).min()
    x_adsb = (pd.to_datetime(minute["minute_ts"], utc=True) - t0).dt.total_seconds().div(60.0)
    x_adsc = (pd.to_datetime(adsc["timestamp"], utc=True) - t0).dt.total_seconds().div(60.0)
    ax0.plot(minute["lon"], minute["lat"], color="#1f77b4", lw=1.4, label="ADS-B minute")
    ax0.scatter(adsc["longitude"], adsc["latitude"], color="#2ca02c", edgecolor="#111111", s=34, zorder=5, label="ADS-C anchors")
    ax0.set_title(f"{pair_id} | ADS-C anchors on matched cross-ocean ADS-B flight")
    ax0.set_xlabel("Longitude")
    ax0.set_ylabel("Latitude")
    ax0.grid(alpha=0.25)
    ax0.legend(fontsize=8)

    ax1.plot(x_adsb, minute["alt"], color="#1f77b4", lw=1.4, label="ADS-B minute altitude")
    ax1.scatter(x_adsc, adsc["altitude_m"], color="#2ca02c", edgecolor="#111111", s=34, zorder=5, label="ADS-C anchors")
    ax1.set_xlabel("Minutes from ADS-B flight start")
    ax1.set_ylabel("Altitude (m)")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    out_dir = ROOT / args.out_dir
    raw_dir = out_dir / "raw_opensky"
    minute_dir = out_dir / "adsb_minute"
    adsc_dir = out_dir / "adsc_anchors"
    overlay_dir = out_dir / "overlay_csv"
    plot_dir = out_dir / "plots"
    for d in [raw_dir, minute_dir, adsc_dir, overlay_dir, plot_dir]:
        d.mkdir(parents=True, exist_ok=True)

    _set_opensky_env(args.settings)
    from pyopensky.trino import Trino

    routes = pd.read_csv(ROOT / args.cross_ocean_routes_csv)
    routes["start_ts"] = pd.to_datetime(routes["start_ts"], utc=True, errors="coerce")
    routes["end_ts"] = pd.to_datetime(routes["end_ts"], utc=True, errors="coerce")
    routes = routes.dropna(subset=["flight_id", "icao24", "start_ts", "end_ts"]).copy()
    routes = routes[
        (routes["anchor_count"] >= int(args.min_anchors))
        & (routes["duration_min"] >= float(args.min_duration_min))
        & (routes["ocean_ratio_gc"] >= float(args.min_ocean_ratio))
    ].copy()
    routes = routes.sort_values(
        ["ocean_ratio_gc", "endpoint_distance_km", "anchor_count"], ascending=[False, False, False]
    ).head(int(args.max_candidates))

    adsc_all = pd.read_csv(ROOT / args.adsc_points_csv)
    adsc_all["timestamp"] = pd.to_datetime(adsc_all["timestamp"], utc=True, errors="coerce")
    adsc_all["flight_id"] = adsc_all["flight_id"].astype(str)
    adsc_groups = {fid: g.sort_values("timestamp").copy() for fid, g in adsc_all.groupby("flight_id")}

    trino = Trino()
    selected = []
    audit_rows = []
    query_log = []
    for idx, r in routes.iterrows():
        if len(selected) >= int(args.target):
            break
        adsc_flight_id = str(r["flight_id"])
        icao24 = str(r["icao24"]).lower()
        adsc_start = _to_utc(r["start_ts"])
        adsc_end = _to_utc(r["end_ts"])
        pair_id = _safe(adsc_flight_id)
        adsc = adsc_groups.get(adsc_flight_id, pd.DataFrame()).copy()
        if adsc.empty:
            continue

        q_start = adsc_start - pd.Timedelta(hours=float(args.flight_pad_hours))
        q_end = adsc_end + pd.Timedelta(hours=float(args.flight_pad_hours))
        print(f"[candidate] {pair_id} icao24={icao24} adsc={adsc_start}~{adsc_end}", flush=True)
        try:
            flights = trino.flightlist(
                start=q_start.to_pydatetime(),
                stop=q_end.to_pydatetime(),
                icao24=icao24,
                cached=bool(args.cached),
            )
            fl = _normalize_flightlist(flights)
            q_status, q_error = "ok", ""
        except Exception as exc:
            fl = pd.DataFrame()
            q_status, q_error = "failed", f"{type(exc).__name__}: {exc}"
        query_log.append(
            {
                "adsc_flight_id": adsc_flight_id,
                "icao24": icao24,
                "q_start": q_start,
                "q_end": q_end,
                "status": q_status,
                "error": q_error,
                "flightlist_rows": int(len(fl)),
            }
        )
        if fl.empty:
            continue
        cand = fl[(fl["firstSeen_ts"] <= adsc_start) & (fl["lastSeen_ts"] >= adsc_end)].copy()
        if cand.empty:
            audit_rows.append({"adsc_flight_id": adsc_flight_id, "status": "reject_no_containing_adsb_flight"})
            continue
        cand["contain_span_sec"] = (cand["lastSeen_ts"] - cand["firstSeen_ts"]).dt.total_seconds()
        best = cand.sort_values("contain_span_sec").iloc[0]
        callsign = str(best.get("callsign") or "").strip()
        adsb_flight_id = f"{best['firstSeen_ts'].strftime('%Y-%m-%d')}-{callsign}-{best['firstSeen_ts'].strftime('%Y%m%dT%H%M%SZ')}-{icao24}"

        raw_path = raw_dir / f"{pair_id}_raw_opensky.csv"
        start = best["firstSeen_ts"] - pd.Timedelta(minutes=float(args.fetch_pad_min))
        stop = best["lastSeen_ts"] + pd.Timedelta(minutes=float(args.fetch_pad_min))
        try:
            if raw_path.exists() and not bool(args.overwrite):
                raw = pd.read_csv(raw_path)
                raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
                raw = _normalize_history(raw)
                print(f"[reuse] {pair_id} raw_rows={len(raw)}", flush=True)
            else:
                raw = _fetch_history_chunked(
                    trino=trino,
                    icao24=icao24,
                    start=start,
                    stop=stop,
                    chunk_min=float(args.chunk_min),
                    cached=bool(args.cached),
                )
            fetch_status, fetch_error = "ok", ""
        except Exception as exc:
            raw = pd.DataFrame()
            fetch_status, fetch_error = "failed", f"{type(exc).__name__}: {exc}"
        raw.to_csv(raw_path, index=False)
        minute = minute_agg_adsb(raw)
        if not minute.empty:
            minute.insert(0, "adsc_flight_id", adsc_flight_id)
            minute.insert(1, "adsb_flight_id", adsb_flight_id)
            minute.insert(2, "icao24", icao24)
        minute_path = minute_dir / f"{pair_id}_adsb_minute.csv"
        minute.to_csv(minute_path, index=False)
        adsc_path = adsc_dir / f"{pair_id}_adsc_anchors.csv"
        adsc.to_csv(adsc_path, index=False)

        freeze = _max_same_latlon_run(minute)
        gaps = _gap_stats(minute)
        inside = 0
        if not minute.empty:
            mt = pd.to_datetime(minute["minute_ts"], utc=True, errors="coerce")
            inside = int(((mt >= adsc_start.floor("min")) & (mt <= adsc_end.ceil("min"))).sum())
        passed = (
            fetch_status == "ok"
            and len(minute) >= int(args.min_adsb_minute_rows)
            and int(freeze["max_frozen_run_min"]) <= int(args.max_frozen_run_min)
        )
        audit = {
            "adsc_flight_id": adsc_flight_id,
            "pair_id": pair_id,
            "icao24": icao24,
            "passed": int(passed),
            "reject_reason": "" if passed else "adsb_quality_failed",
            "fetch_status": fetch_status,
            "fetch_error": fetch_error,
            "adsb_flight_id": adsb_flight_id,
            "callsign": callsign,
            "adsb_firstSeen": best["firstSeen_ts"],
            "adsb_lastSeen": best["lastSeen_ts"],
            "dep_airport": best.get("estDepartureAirport", ""),
            "arr_airport": best.get("estArrivalAirport", ""),
            "adsc_start": adsc_start,
            "adsc_end": adsc_end,
            "adsc_anchor_count": int(r["anchor_count"]),
            "adsc_ocean_ratio_gc": float(r["ocean_ratio_gc"]),
            "adsc_endpoint_distance_km": float(r["endpoint_distance_km"]),
            "raw_rows": int(len(raw)),
            "adsb_minute_rows": int(len(minute)),
            "adsb_rows_inside_adsc_window": inside,
            "raw_csv": str(raw_path.relative_to(ROOT)),
            "adsb_minute_csv": str(minute_path.relative_to(ROOT)),
            "adsc_anchor_csv": str(adsc_path.relative_to(ROOT)),
        }
        audit.update(freeze)
        audit.update(gaps)
        audit_rows.append(audit)

        if passed:
            overlay = pd.concat(
                [
                    minute.assign(source="adsb_minute").rename(columns={"minute_ts": "timestamp"}),
                    adsc.rename(columns={"latitude": "lat", "longitude": "lon", "altitude_m": "alt"}).assign(
                        source="adsc_anchor", adsb_flight_id=adsb_flight_id
                    ),
                ],
                ignore_index=True,
                sort=False,
            )
            overlay_path = overlay_dir / f"{pair_id}_overlay.csv"
            overlay.to_csv(overlay_path, index=False)
            _plot_overlay(pair_id, adsc, minute, plot_dir / f"{pair_id}_overlay.png")
            audit["overlay_csv"] = str(overlay_path.relative_to(ROOT))
            audit["plot_png"] = str((plot_dir / f"{pair_id}_overlay.png").relative_to(ROOT))
            selected.append(audit)
            print(f"[select] {pair_id} selected={len(selected)}/{args.target}", flush=True)
        else:
            print(
                f"[reject] {pair_id} rows={len(minute)} frozen={freeze['max_frozen_run_min']} inside={inside}",
                flush=True,
            )
        pd.DataFrame(query_log).to_csv(out_dir / "flightlist_query_log.csv", index=False)
        pd.DataFrame(audit_rows).to_csv(out_dir / "candidate_quality_audit.csv", index=False)
        pd.DataFrame(selected).to_csv(out_dir / "selected_clean_cross_ocean_cases.csv", index=False)
        if float(args.sleep_sec) > 0:
            time.sleep(float(args.sleep_sec))

    pd.DataFrame(query_log).to_csv(out_dir / "flightlist_query_log.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(out_dir / "candidate_quality_audit.csv", index=False)
    pd.DataFrame(selected).to_csv(out_dir / "selected_clean_cross_ocean_cases.csv", index=False)
    print(f"[done] selected={len(selected)} out={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
