from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import timezone
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.io_utils import load_settings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live match ADS-C 20 flights to OpenSky flightlist with ±window hours")
    p.add_argument("--seed-csv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--window-hours", type=float, default=4.0)
    p.add_argument("--rate-limit-seconds", type=float, default=10.0)
    return p.parse_args()


def _setup_opensky_env() -> None:
    settings = load_settings("config/settings.yaml")
    username = settings["opensky"].get("username") or os.getenv("OPENSKY_USERNAME") or os.getenv("OPENSKY_TRINO_USERNAME")
    password = settings["opensky"].get("password") or os.getenv("OPENSKY_PASSWORD") or os.getenv("OPENSKY_TRINO_PASSWORD")
    if username:
        os.environ["OPENSKY_USERNAME"] = str(username).lower()
        os.environ["OPENSKY_TRINO_USERNAME"] = str(username).lower()
    if password:
        os.environ["OPENSKY_PASSWORD"] = str(password)
        os.environ["OPENSKY_TRINO_PASSWORD"] = str(password)
    if (not os.getenv("OPENSKY_TRINO_USERNAME")) or (not os.getenv("OPENSKY_TRINO_PASSWORD")):
        raise RuntimeError("Missing OPENSKY_TRINO_USERNAME/OPENSKY_TRINO_PASSWORD; refusing OAuth popup fallback.")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = pd.read_csv(args.seed_csv, parse_dates=["start_ts", "end_ts"])
    _setup_opensky_env()

    # import after env setup to avoid OAuth popup fallback
    from pyopensky.trino import Trino

    trino = Trino()

    match_csv = out_dir / "adsc_adsb_live_match20.csv"
    cand_csv = out_dir / "adsc_adsb_live_match20_candidates_top5.csv"
    log_csv = out_dir / "adsc_adsb_live_match20_query_log.csv"

    rows = []
    cand_rows = []
    logs = []

    if match_csv.exists():
        rows = pd.read_csv(match_csv).to_dict("records")
    if cand_csv.exists():
        cand_rows = pd.read_csv(cand_csv).to_dict("records")
    if log_csv.exists():
        logs = pd.read_csv(log_csv).to_dict("records")

    processed = {str(x.get("adsc_flight_id")) for x in logs if x.get("adsc_flight_id") is not None}

    def _flush_checkpoint() -> None:
        pd.DataFrame(rows).to_csv(match_csv, index=False)
        pd.DataFrame(cand_rows).to_csv(cand_csv, index=False)
        pd.DataFrame(logs).to_csv(log_csv, index=False)

    for i, r in enumerate(seed.itertuples(index=False), start=1):
        if str(r.flight_id) in processed:
            print(f"[{i}/{len(seed)}] skip adsc_flight_id={r.flight_id} (already processed)", flush=True)
            continue

        icao = str(r.icao24).lower()
        st = pd.Timestamp(r.start_ts)
        et = pd.Timestamp(r.end_ts)
        q_st = (st - pd.Timedelta(hours=float(args.window_hours))).to_pydatetime()
        q_et = (et + pd.Timedelta(hours=float(args.window_hours))).to_pydatetime()
        if q_st.tzinfo is None:
            q_st = q_st.replace(tzinfo=timezone.utc)
        if q_et.tzinfo is None:
            q_et = q_et.replace(tzinfo=timezone.utc)

        print(f"[{i}/{len(seed)}] query icao={icao} window={q_st} -> {q_et}", flush=True)
        try:
            flights = trino.flightlist(start=q_st, stop=q_et, icao24=icao, cached=True)
        except Exception as e:
            logs.append(
                {
                    "adsc_flight_id": r.flight_id,
                    "icao24": icao,
                    "status": "error",
                    "rows": 0,
                    "error": str(e)[:500],
                }
            )
            _flush_checkpoint()
            print(f"  -> error: {str(e)[:180]}", flush=True)
            time.sleep(float(args.rate_limit_seconds))
            continue

        if flights is None or len(flights) == 0:
            logs.append({"adsc_flight_id": r.flight_id, "icao24": icao, "status": "empty", "rows": 0})
            _flush_checkpoint()
            print("  -> empty", flush=True)
            time.sleep(float(args.rate_limit_seconds))
            continue

        f = flights.copy()
        cols = {c.lower(): c for c in f.columns}
        first_col = cols.get("firstseen")
        last_col = cols.get("lastseen")
        if first_col is None or last_col is None:
            logs.append({"adsc_flight_id": r.flight_id, "icao24": icao, "status": "missing_first_last", "rows": len(f)})
            _flush_checkpoint()
            print("  -> missing firstSeen/lastSeen", flush=True)
            time.sleep(float(args.rate_limit_seconds))
            continue

        f["firstSeen"] = pd.to_datetime(f[first_col], unit="s", utc=True, errors="coerce")
        f["lastSeen"] = pd.to_datetime(f[last_col], unit="s", utc=True, errors="coerce")
        f = f.dropna(subset=["firstSeen", "lastSeen"]).copy()

        span_sec = max((et - st).total_seconds(), 1.0)
        scored = []
        for c in f.itertuples(index=False):
            ov_start = max(st, c.firstSeen)
            ov_end = min(et, c.lastSeen)
            ov_sec = max((ov_end - ov_start).total_seconds(), 0.0)
            ov_ratio = min(ov_sec / span_sec, 1.0)
            contain = int((c.firstSeen <= st) and (c.lastSeen >= et))
            win_span = (c.lastSeen - c.firstSeen).total_seconds()
            score = 0.7 * ov_ratio + 0.3 * contain - 1e-7 * win_span
            call_col = cols.get("callsign")
            dep_col = cols.get("estdepartureairport")
            arr_col = cols.get("estarrivalairport")
            callsign = (str(getattr(c, call_col, "")).strip() if call_col else "")
            dep = getattr(c, dep_col, None) if dep_col else None
            arr = getattr(c, arr_col, None) if arr_col else None
            scored.append((score, ov_ratio, contain, win_span, callsign, c.firstSeen, c.lastSeen, dep, arr))

        if not scored:
            logs.append({"adsc_flight_id": r.flight_id, "icao24": icao, "status": "no_valid_rows", "rows": 0})
            _flush_checkpoint()
            print("  -> no valid rows", flush=True)
            time.sleep(float(args.rate_limit_seconds))
            continue

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[0]
        sec = scored[1] if len(scored) > 1 else None
        margin = (top[0] - sec[0]) if sec else 1.0
        is_unique = int((margin >= 0.05) or (sec is None))

        rows.append(
            {
                "adsc_flight_id": r.flight_id,
                "icao24": icao,
                "adsc_start_ts": st,
                "adsc_end_ts": et,
                "adsc_anchor_count": int(r.n_anchor),
                "adsb_callsign": top[4],
                "adsb_firstSeen": top[5],
                "adsb_lastSeen": top[6],
                "dep_airport_icao": top[7],
                "arr_airport_icao": top[8],
                "match_score": top[0],
                "overlap_ratio": top[1],
                "contain_flag": top[2],
                "adsb_win_span_sec": top[3],
                "score_margin_vs_2nd": margin,
                "is_unique_match": is_unique,
            }
        )

        for rank, x in enumerate(scored[:5], start=1):
            cand_rows.append(
                {
                    "adsc_flight_id": r.flight_id,
                    "icao24": icao,
                    "candidate_rank": rank,
                    "adsb_callsign": x[4],
                    "adsb_firstSeen": x[5],
                    "adsb_lastSeen": x[6],
                    "dep_airport_icao": x[7],
                    "arr_airport_icao": x[8],
                    "match_score": x[0],
                    "overlap_ratio": x[1],
                    "contain_flag": x[2],
                    "adsb_win_span_sec": x[3],
                }
            )

        logs.append({"adsc_flight_id": r.flight_id, "icao24": icao, "status": "ok", "rows": len(f)})
        _flush_checkpoint()
        print(
            f"  -> ok rows={len(f)} top={top[4]} overlap={top[1]:.3f} contain={top[2]} unique={is_unique} margin={margin:.4f}",
            flush=True,
        )
        time.sleep(float(args.rate_limit_seconds))

    m = pd.DataFrame(rows)
    c = pd.DataFrame(cand_rows)
    l = pd.DataFrame(logs)

    m.to_csv(match_csv, index=False)
    c.to_csv(cand_csv, index=False)
    l.to_csv(log_csv, index=False)

    summary = {
        "input": int(len(seed)),
        "matched": int(len(m)),
        "unique": int(m["is_unique_match"].sum()) if not m.empty else 0,
        "ambiguous": int((m["is_unique_match"] == 0).sum()) if not m.empty else 0,
        "query_ok": int((l["status"] == "ok").sum()) if not l.empty else 0,
        "query_empty": int((l["status"] == "empty").sum()) if not l.empty else 0,
        "query_error": int((l["status"] == "error").sum()) if not l.empty else 0,
    }
    (out_dir / "match_summary_live20.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[done]", summary, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
