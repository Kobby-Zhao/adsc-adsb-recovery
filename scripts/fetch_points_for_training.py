from __future__ import annotations

import argparse
import math
import os
import re
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.io_utils import ensure_dir, load_settings, write_csv


ICAO24_RE = re.compile(r"^[0-9a-fA-F]{6}$")


def _parse_dt(value) -> datetime:
    value_str = str(value).strip()
    # Support unix timestamp passed as int-like/float-like string, e.g. "1727767890" or "1727767890.0".
    try:
        ts = float(value_str)
        if math.isfinite(ts):
            dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
            return dt
    except Exception:
        pass
    if value_str.isdigit():
        dt = datetime.fromtimestamp(int(value_str), tz=timezone.utc)
    else:
        if value_str.endswith("Z"):
            value_str = value_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(value_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _safe_name(value: str) -> str:
    value = (value or "").strip().upper()
    if not value:
        return "UNKNOWN"
    return re.sub(r"[^A-Z0-9_-]+", "", value)


def _existing_keys(raw_dir: Path, recursive: bool = False) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    iterator = raw_dir.rglob("*.csv") if recursive else raw_dir.glob("*.csv")
    for p in iterator:
        stem = p.stem
        parts = stem.split("-")
        if len(parts) < 4:
            continue
        icao24 = parts[-1].lower()
        firstseen = parts[-2]
        day = "-".join(parts[:3]) if len(parts) > 4 else parts[0]
        keys.add((icao24, firstseen, day))
    return keys


def _log(msg: str) -> None:
    print(msg, flush=True)


def _enable_line_buffering() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)


def _run_with_timeout(fn, timeout_seconds: int):
    if timeout_seconds <= 0 or os.name != "posix":
        return fn()

    def _handler(signum, frame):
        raise TimeoutError(f"query timeout after {timeout_seconds}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_seconds)
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch ADS-B points for many distinct flights (training set expansion).")
    parser.add_argument(
        "--flights-csv",
        default="outputs/flights/processed/complete_RJTT_2024-05-01_to_2024-05-30.csv",
        help="Flight list CSV with icao24/callsign/firstseen/lastseen/day",
    )
    parser.add_argument(
        "--raw-dir",
        default="train_fetch_resume",
        help="Output dir for raw points. default outputs/points/raw/train_fetch_resume (supports resume)",
    )
    parser.add_argument("--max-flights", type=int, default=1000)
    parser.add_argument("--day-start", default=None)
    parser.add_argument("--day-end", default=None)
    parser.add_argument("--cached", action="store_true", default=True)
    parser.add_argument("--query-timeout-seconds", type=int, default=120, help="Timeout for one Trino query")
    parser.add_argument("--progress-every", type=int, default=20, help="Print summary every N flights")
    parser.add_argument("--warn-slow-seconds", type=float, default=30.0, help="Warn when single query is slow")
    parser.add_argument("--disable-tqdm", action="store_true", default=True, help="Disable third-party tqdm output")
    parser.add_argument(
        "--dedup-scope",
        choices=["current_run", "all_runs"],
        default="all_runs",
        help="Skip already fetched flights in current run dir or in all outputs/points/raw runs",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    _enable_line_buffering()
    args = build_parser().parse_args()
    settings = load_settings("config/settings.yaml")

    flights_csv = Path(args.flights_csv)
    if not flights_csv.exists():
        raise FileNotFoundError(f"Missing flights csv: {flights_csv}")

    raw_root = Path(settings["output"].get("points_raw_dir") or settings["output"]["raw_dir"])
    run_dir = args.raw_dir
    out_dir = Path(run_dir) if os.path.isabs(run_dir) else raw_root / run_dir
    ensure_dir(str(out_dir))

    df = pd.read_csv(flights_csv, low_memory=False)
    required = ["icao24", "callsign", "firstseen", "lastseen", "day"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    df["icao24"] = df["icao24"].astype(str).str.lower()
    df["callsign"] = df["callsign"].astype(str).str.strip()
    df = df[df["icao24"].apply(lambda x: bool(ICAO24_RE.match(x)))]
    df = df[(df["callsign"] != "") & (df["callsign"].str.lower() != "nan")]

    if args.day_start:
        df = df[df["day"] >= args.day_start]
    if args.day_end:
        df = df[df["day"] <= args.day_end]

    df = df.drop_duplicates(subset=["icao24", "firstseen", "lastseen", "day"]).reset_index(drop=True)

    dedup_root = raw_root if args.dedup_scope == "all_runs" else out_dir
    existing = _existing_keys(dedup_root, recursive=(args.dedup_scope == "all_runs"))
    existing_before = len(existing)
    keep = []
    for _, row in df.iterrows():
        day = str(row["day"])
        fs = str(row["firstseen"])
        key = (str(row["icao24"]).lower(), fs, day)
        if key in existing:
            continue
        keep.append(row)
    df = pd.DataFrame(keep)

    if df.empty:
        _log("[info] no_new_flights_to_fetch")
        return 0

    if args.max_flights > 0:
        df = df.head(args.max_flights)

    _log(
        f"[info] candidate_flights={len(df)} out_dir={out_dir} "
        f"dedup_scope={args.dedup_scope} existing_keys={existing_before}"
    )
    if args.dry_run:
        _log(df[["icao24", "callsign", "day", "firstseen", "lastseen"]].head(10).to_string(index=False))
        return 0

    username = settings["opensky"].get("username") or os.getenv("OPENSKY_USERNAME")
    password = settings["opensky"].get("password") or os.getenv("OPENSKY_PASSWORD")
    if username:
        os.environ["OPENSKY_TRINO_USERNAME"] = username.lower()
        os.environ["OPENSKY_USERNAME"] = username.lower()
    if password:
        os.environ["OPENSKY_TRINO_PASSWORD"] = password
        os.environ["OPENSKY_PASSWORD"] = password
    if args.disable_tqdm:
        os.environ["TQDM_DISABLE"] = "1"

    from pyopensky.trino import Trino

    trino = Trino()
    _log("[trino] client_initialized")
    _log(
        f"[info] start_fetch requested={len(df)} timeout={args.query_timeout_seconds}s "
        f"progress_every={args.progress_every}"
    )

    ok = 0
    fail = 0
    skip = 0
    total_points_rows = 0
    total_start = time.time()

    try:
        for i, row in df.iterrows():
            idx = i + 1
            icao24 = str(row["icao24"]).lower()
            callsign = str(row["callsign"]).strip()
            day = str(row["day"])
            start_dt = _parse_dt(row["firstseen"])
            stop_dt = _parse_dt(row["lastseen"])

            _log(
                f"[flight] {idx}/{len(df)} start icao24={icao24} callsign={callsign} "
                f"day={day} window={start_dt.isoformat()}~{stop_dt.isoformat()}"
            )

            start_ts = time.time()
            try:
                points = _run_with_timeout(
                    lambda: trino.history(
                        start=start_dt,
                        stop=stop_dt,
                        icao24=icao24,
                        selected_columns=("time", "icao24", "lat", "lon", "baroaltitude", "velocity", "heading"),
                        cached=bool(args.cached),
                    ),
                    timeout_seconds=int(args.query_timeout_seconds),
                )
            except Exception as exc:
                fail += 1
                elapsed = time.time() - start_ts
                _log(
                    f"[fail] {idx}/{len(df)} icao24={icao24} callsign={callsign} "
                    f"elapsed={elapsed:.1f}s err={type(exc).__name__}: {exc}"
                )
                continue

            rows = 0 if points is None else len(points)
            elapsed = time.time() - start_ts
            if rows == 0:
                skip += 1
                _log(f"[skip] {idx}/{len(df)} no_points icao24={icao24} callsign={callsign} elapsed={elapsed:.1f}s")
                continue

            safe_day = _safe_name(day)
            safe_callsign = _safe_name(callsign)
            safe_fs = _safe_name(str(row["firstseen"]))
            name = f"{safe_day}-{safe_callsign}-{safe_fs}-{icao24}.csv"
            out_path = out_dir / name
            write_csv(points, str(out_path))
            ok += 1
            total_points_rows += int(rows)
            slow_tag = " [slow]" if elapsed >= float(args.warn_slow_seconds) else ""
            _log(f"[ok] {idx}/{len(df)} rows={rows} elapsed={elapsed:.1f}s file={out_path.name}{slow_tag}")

            if int(args.progress_every) > 0 and idx % int(args.progress_every) == 0:
                spent = time.time() - total_start
                _log(
                    f"[progress] {idx}/{len(df)} fetched={ok} skipped={skip} failed={fail} "
                    f"points_rows={total_points_rows} elapsed_total={spent/60.0:.1f}min "
                    f"save_dir={out_dir}"
                )
    except KeyboardInterrupt:
        _log("[stop] interrupted_by_user")

    total_elapsed = time.time() - total_start
    _log(
        f"[done] fetched={ok} skipped={skip} failed={fail} requested={len(df)} "
        f"out_dir={out_dir} elapsed_total={total_elapsed/60.0:.1f}min"
    )
    _log(f"[result] successful_flights={ok}")
    _log(f"[result] total_points_rows={total_points_rows}")
    _log(f"[result] saved_dir={out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
