from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.io_utils import ensure_dir, load_settings


CHINA_ICAO_PREFIX = "Z"


@dataclass
class MonthResult:
    month: str
    queried_days: int
    raw_rows: int
    unique_rows: int
    selected_rows: int
    selected_domestic_ratio: float
    selected_long_ratio: float
    note: str


def _month_iter(start_month: str, end_month: str) -> list[str]:
    s = datetime.strptime(start_month, "%Y-%m").replace(day=1)
    e = datetime.strptime(end_month, "%Y-%m").replace(day=1)
    out = []
    cur = s
    while cur <= e:
        out.append(cur.strftime("%Y-%m"))
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)
    return out


def _month_days(month: str) -> list[tuple[datetime, datetime]]:
    start = datetime.strptime(month, "%Y-%m").replace(day=1, tzinfo=timezone.utc)
    if start.month == 12:
        nxt = start.replace(year=start.year + 1, month=1)
    else:
        nxt = start.replace(month=start.month + 1)
    out = []
    cur = start
    while cur < nxt:
        out.append((cur, cur + timedelta(days=1)))
        cur += timedelta(days=1)
    return out


def _normalize_flightlist_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "icao24",
                "callsign",
                "firstseen",
                "lastseen",
                "estdepartureairport",
                "estarrivalairport",
            ]
        )
    x = df.copy()
    lower = {c.lower(): c for c in x.columns}
    def pick(*names: str) -> str | None:
        for n in names:
            if n.lower() in lower:
                return lower[n.lower()]
        return None

    c_icao = pick("icao24")
    c_call = pick("callsign")
    c_first = pick("firstseen")
    c_last = pick("lastseen")
    c_dep = pick("estdepartureairport", "departureairport", "departure")
    c_arr = pick("estarrivalairport", "arrivalairport", "arrival")
    if not all([c_icao, c_call, c_first, c_last]):
        return pd.DataFrame(
            columns=[
                "icao24",
                "callsign",
                "firstseen",
                "lastseen",
                "estdepartureairport",
                "estarrivalairport",
            ]
        )

    first_dt = pd.to_datetime(x[c_first], utc=True, errors="coerce")
    last_dt = pd.to_datetime(x[c_last], utc=True, errors="coerce")
    valid_time = first_dt.notna() & last_dt.notna()
    x = x.loc[valid_time].copy()
    first_dt = first_dt.loc[valid_time]
    last_dt = last_dt.loc[valid_time]
    y = pd.DataFrame(
        {
            "icao24": x[c_icao].astype(str).str.lower(),
            "callsign": x[c_call].astype(str).str.strip(),
            "firstseen": (first_dt.astype("int64") // 10**9).astype("float64"),
            "lastseen": (last_dt.astype("int64") // 10**9).astype("float64"),
            "estdepartureairport": x[c_dep].astype(str).str.upper() if c_dep else "",
            "estarrivalairport": x[c_arr].astype(str).str.upper() if c_arr else "",
        }
    )
    y = y.dropna(subset=["firstseen", "lastseen"])
    y = y[y["lastseen"] > y["firstseen"]]
    y = y[(y["callsign"] != "") & (y["callsign"].str.lower() != "nan")]
    return y


def _is_domestic_cn(dep: str, arr: str) -> bool:
    if not dep or not arr:
        return False
    return dep.startswith(CHINA_ICAO_PREFIX) and arr.startswith(CHINA_ICAO_PREFIX)


def _select_monthly(
    df: pd.DataFrame,
    target_total: int,
    domestic_ratio: float,
    min_long_ratio: float,
    min_duration_sec: int,
) -> tuple[pd.DataFrame, str]:
    if df.empty:
        return df, "empty_month_pool"

    x = df.copy()
    x["duration_seconds"] = x["lastseen"] - x["firstseen"]
    x["is_long"] = x["duration_seconds"] >= int(min_duration_sec)
    x["is_domestic"] = [
        _is_domestic_cn(d, a) for d, a in zip(x["estdepartureairport"], x["estarrivalairport"])
    ]
    x = x.drop_duplicates(subset=["icao24", "callsign", "firstseen", "lastseen"]).reset_index(drop=True)

    n_total = min(int(target_total), len(x))
    if n_total <= 0:
        return x.head(0), "no_candidates"

    n_dom = int(round(n_total * float(domestic_ratio)))
    n_non = n_total - n_dom

    dom = x[x["is_domestic"]].copy()
    non = x[~x["is_domestic"]].copy()
    dom_long = dom[dom["is_long"]]
    dom_short = dom[~dom["is_long"]]
    non_long = non[non["is_long"]]
    non_short = non[~non["is_long"]]

    pick_dom = pd.concat([dom_long, dom_short], ignore_index=False).head(n_dom)
    pick_non = pd.concat([non_long, non_short], ignore_index=False).head(n_non)
    sel = pd.concat([pick_dom, pick_non], ignore_index=False)

    if len(sel) < n_total:
        rest = x[~x.index.isin(sel.index)]
        sel = pd.concat([sel, rest.head(n_total - len(sel))], ignore_index=False)

    min_long = int(np.ceil(n_total * float(min_long_ratio)))
    cur_long = int(sel["is_long"].sum())
    if cur_long < min_long:
        need = min_long - cur_long
        add_pool = x[(~x.index.isin(sel.index)) & (x["is_long"])]
        drop_pool = sel[~sel["is_long"]]
        add_idx = add_pool.index.tolist()[:need]
        drop_idx = drop_pool.index.tolist()[: min(need, len(drop_pool))]
        if drop_idx and add_idx:
            sel = sel.drop(index=drop_idx)
            sel = pd.concat([sel, x.loc[add_idx]], ignore_index=False)

    sel = sel.head(n_total).copy()
    return sel.reset_index(drop=True), "ok"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Collect monthly global flightlists from OpenSky (single-thread).")
    p.add_argument("--start-month", default="2024-06")
    p.add_argument("--end-month", default="2025-03")
    p.add_argument("--target-per-month", type=int, default=500)
    p.add_argument("--domestic-ratio", type=float, default=0.10, help="Domestic (China domestic) ratio.")
    p.add_argument("--min-long-ratio", type=float, default=0.70, help="At least this ratio with duration>=4h.")
    p.add_argument("--min-duration-hours", type=float, default=4.0)
    p.add_argument("--day-limit", type=int, default=12000, help="Flightlist LIMIT per day query.")
    p.add_argument("--domestic-day-limit", type=int, default=6000, help="Dedicated China-airport pool LIMIT per day.")
    p.add_argument("--use-domestic-pool", type=int, default=1, help="1: query extra airport pool for China domestic coverage.")
    p.add_argument("--rate-limit-seconds", type=float, default=8.0)
    p.add_argument("--output-dir", default="outputs/flights/global_monthly_202406_to_202503")
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    ensure_dir(str(out_dir))
    ensure_dir(str(out_dir / "monthly_raw"))
    ensure_dir(str(out_dir / "monthly_selected"))

    settings = load_settings("config/settings.yaml")
    username = settings["opensky"].get("username")
    password = settings["opensky"].get("password")
    if username:
        os.environ["OPENSKY_USERNAME"] = username.lower()
        os.environ["OPENSKY_TRINO_USERNAME"] = username.lower()
    if password:
        os.environ["OPENSKY_PASSWORD"] = password
        os.environ["OPENSKY_TRINO_PASSWORD"] = password

    months = _month_iter(args.start_month, args.end_month)
    print(f"[plan] months={months} target_per_month={args.target_per_month} domestic_ratio={args.domestic_ratio} min_long_ratio={args.min_long_ratio}")
    if args.dry_run:
        return 0

    from pyopensky.trino import Trino

    trino = Trino()
    monthly_stats: list[MonthResult] = []
    all_selected = []
    min_duration_sec = int(round(float(args.min_duration_hours) * 3600))
    china_airports = []
    try:
        ap = pd.read_csv(Path("outputs/flights/raw/airports.csv"), usecols=["icao"])
        ap["icao"] = ap["icao"].astype(str).str.upper()
        # China ICAO identifiers are mostly Z***.
        china_airports = sorted(ap[ap["icao"].str.match(r"^Z[A-Z0-9]{3}$", na=False)]["icao"].unique().tolist())
    except Exception:
        china_airports = []

    for month in months:
        print(f"[month] {month} start")
        day_windows = _month_days(month)
        rows = []
        q_days = 0
        for ds, de in day_windows:
            q_days += 1
            try:
                q = trino.flightlist(
                    start=ds,
                    stop=de,
                    cached=True,
                    limit=int(args.day_limit),
                )
                z = _normalize_flightlist_cols(q)
                if int(args.use_domestic_pool) == 1 and china_airports:
                    q_cn = trino.flightlist(
                        start=ds,
                        stop=de,
                        airport=china_airports,
                        cached=True,
                        limit=int(args.domestic_day_limit),
                    )
                    z_cn = _normalize_flightlist_cols(q_cn)
                    if not z_cn.empty:
                        z = pd.concat([z, z_cn], ignore_index=True)
                        z = z.drop_duplicates(subset=["icao24", "callsign", "firstseen", "lastseen"])
                if not z.empty:
                    z["day"] = ds.strftime("%Y-%m-%d")
                    rows.append(z)
                print(f"[day] {ds.strftime('%Y-%m-%d')} rows={0 if z is None else len(z)}")
            except Exception as e:
                print(f"[day][fail] {ds.strftime('%Y-%m-%d')} err={type(e).__name__}: {e}")
            time.sleep(float(args.rate_limit_seconds))

        month_raw = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
            columns=["icao24", "callsign", "firstseen", "lastseen", "estdepartureairport", "estarrivalairport", "day"]
        )
        month_raw.to_csv(out_dir / "monthly_raw" / f"flightlist_raw_{month}.csv", index=False)
        selected, note = _select_monthly(
            month_raw,
            target_total=int(args.target_per_month),
            domestic_ratio=float(args.domestic_ratio),
            min_long_ratio=float(args.min_long_ratio),
            min_duration_sec=min_duration_sec,
        )
        if not selected.empty:
            selected["month"] = month
            selected["duration_seconds"] = selected["lastseen"] - selected["firstseen"]
            selected["day"] = pd.to_datetime(selected["firstseen"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
            selected["is_domestic_cn"] = [
                _is_domestic_cn(d, a) for d, a in zip(selected["estdepartureairport"], selected["estarrivalairport"])
            ]
            selected["is_long"] = selected["duration_seconds"] >= int(min_duration_sec)
        selected.to_csv(out_dir / "monthly_selected" / f"flightlist_selected_{month}.csv", index=False)

        dom_ratio = float(selected["is_domestic_cn"].mean()) if len(selected) else 0.0
        long_ratio = float(selected["is_long"].mean()) if len(selected) else 0.0
        st = MonthResult(
            month=month,
            queried_days=q_days,
            raw_rows=int(len(month_raw)),
            unique_rows=int(len(month_raw.drop_duplicates(subset=["icao24", "callsign", "firstseen", "lastseen"]))),
            selected_rows=int(len(selected)),
            selected_domestic_ratio=dom_ratio,
            selected_long_ratio=long_ratio,
            note=note,
        )
        monthly_stats.append(st)
        print(
            f"[month][done] {month} selected={st.selected_rows} "
            f"dom_ratio={st.selected_domestic_ratio:.3f} long_ratio={st.selected_long_ratio:.3f}"
        )
        if not selected.empty:
            all_selected.append(selected)

    summary = pd.DataFrame([s.__dict__ for s in monthly_stats])
    summary.to_csv(out_dir / "monthly_collection_summary.csv", index=False)
    if all_selected:
        merged = pd.concat(all_selected, ignore_index=True)
        merged = merged.drop_duplicates(subset=["icao24", "callsign", "firstseen", "lastseen"])
        merged.to_csv(out_dir / "flightlist_selected_all_months.csv", index=False)
        print(f"[done] total_selected={len(merged)} file={out_dir/'flightlist_selected_all_months.csv'}")
    else:
        print("[done] no selected flights")
    print(f"[done] summary={out_dir/'monthly_collection_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
