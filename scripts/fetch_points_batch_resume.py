from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


ICAO24_RE = re.compile(r"^[0-9a-fA-F]{6}$")


def _safe_name(value: str) -> str:
    if value is None:
        value = ""
    value = str(value).strip().upper()
    if not value:
        return "UNKNOWN"
    return re.sub(r"[^A-Z0-9_-]+", "", value)


def _expected_filename(day: str, callsign: str, firstseen: str, icao24: str) -> str:
    return f"{_safe_name(day)}-{_safe_name(callsign)}-{_safe_name(firstseen)}-{str(icao24).lower()}.csv"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Resume ADS-B fetching in fixed-size batches with strict filename dedup.")
    p.add_argument(
        "--flights-csv",
        default="outputs/flights/processed/complete_RJTT_2024-05-01_to_2024-05-30.csv",
    )
    p.add_argument("--raw-dir", default="outputs/points/raw/train_fetch_5000_20260321")
    p.add_argument("--target-total", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--max-rounds", type=int, default=20)
    p.add_argument("--query-timeout-seconds", type=int, default=120)
    p.add_argument("--cached", action="store_true", default=True)
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--dry-run", action="store_true")
    return p


def _load_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    req = ["icao24", "callsign", "firstseen", "lastseen", "day"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise RuntimeError(f"Missing columns in flights csv: {miss}")
    df = df.copy()
    df["icao24"] = df["icao24"].astype(str).str.lower()
    df["callsign"] = df["callsign"].astype(str).str.strip()
    df = df[df["icao24"].apply(lambda x: bool(ICAO24_RE.match(x)))]
    df = df[(df["callsign"] != "") & (df["callsign"].str.lower() != "nan")]
    df = df.drop_duplicates(subset=["icao24", "firstseen", "lastseen", "day"]).reset_index(drop=True)
    df["expected_file"] = [
        _expected_filename(d, c, f, i)
        for d, c, f, i in zip(df["day"], df["callsign"], df["firstseen"], df["icao24"])
    ]
    return df


def _existing_files(raw_dir: Path) -> set[str]:
    return {p.name for p in raw_dir.glob("*.csv")}


def main() -> int:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    flights_csv = Path(args.flights_csv)
    raw_dir = Path(args.raw_dir)
    if not flights_csv.is_absolute():
        flights_csv = root / flights_csv
    if not raw_dir.is_absolute():
        raw_dir = root / raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    batch_dir = raw_dir / "_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)

    df = _load_candidates(flights_csv)
    total_target = min(int(args.target_total), len(df))
    py = sys.executable
    fetch_script = root / "scripts" / "fetch_points_for_training.py"

    print(f"[info] candidates={len(df)} target_total={total_target} raw_dir={raw_dir}", flush=True)
    rounds = 0
    while rounds < int(args.max_rounds):
        existing = _existing_files(raw_dir)
        done = int((df["expected_file"].isin(existing)).sum())
        remaining = df[~df["expected_file"].isin(existing)].copy()
        print(
            f"[status] round={rounds} done={done}/{total_target} remaining={len(remaining)}",
            flush=True,
        )
        if done >= total_target:
            print("[done] target reached", flush=True)
            return 0
        if remaining.empty:
            print("[done] no remaining candidates", flush=True)
            return 0

        need = min(int(args.batch_size), int(total_target - done), len(remaining))
        batch = remaining.head(need).copy()
        batch_csv = batch_dir / f"batch_round_{rounds:03d}_{need}.csv"
        batch[["icao24", "callsign", "firstseen", "lastseen", "day"]].to_csv(batch_csv, index=False)
        print(f"[batch] file={batch_csv} size={need}", flush=True)

        if args.dry_run:
            rounds += 1
            continue

        cmd = [
            py,
            str(fetch_script),
            "--flights-csv",
            str(batch_csv),
            "--raw-dir",
            str(raw_dir),
            "--max-flights",
            str(need),
            "--dedup-scope",
            "current_run",
            "--query-timeout-seconds",
            str(int(args.query_timeout_seconds)),
            "--progress-every",
            str(int(args.progress_every)),
        ]
        if args.cached:
            cmd.append("--cached")

        print(f"[run] {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, cwd=str(root))
        if proc.returncode != 0:
            print(f"[warn] fetch script exited with code={proc.returncode}", flush=True)

        existing2 = _existing_files(raw_dir)
        done2 = int((df["expected_file"].isin(existing2)).sum())
        gained = done2 - done
        print(f"[round_result] round={rounds} gained={gained} done={done2}/{total_target}", flush=True)
        if gained <= 0:
            print("[warn] no progress in this round, stopping early", flush=True)
            return 2
        rounds += 1

    existing = _existing_files(raw_dir)
    done = int((df["expected_file"].isin(existing)).sum())
    print(f"[done] max_rounds reached done={done}/{total_target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
