#!/usr/bin/env python3
"""
Streaming training monitor — one-line per epoch + health check every N epochs.

Usage:
    tail -f /path/to/training.log | python3 scripts/monitor_training.py --stdin -n 5
    python3 scripts/monitor_training.py --logfile /path/to/training.log -n 5
    python3 scripts/monitor_training.py --history outputs/.../history.json -n 5
"""
import argparse, json, os, re, sys, time
from datetime import datetime

# ── helpers ──────────────────────────────────────────────────────────────

def _parse_epoch_line(line: str) -> dict | None:
    m = re.search(r'\[epoch_end\s+(\d+)/(\d+)\]\s+(.*)', line)
    if not m:
        return None
    ep = int(m.group(1))
    total = int(m.group(2))
    rest = m.group(2) + " " + m.group(3)  # keep "024 train_loss=..." for parsing
    metrics = {"epoch": ep, "total_epochs": total}
    # Re-parse from the full match to get all key=value pairs
    full = m.group(0)
    for kv in re.finditer(r'(\w+)=([0-9.e+\-]+)', full):
        metrics[kv.group(1)] = float(kv.group(2))
    return metrics

def _load_history(path: str) -> list[dict]:
    d = json.load(open(path))
    out = []
    for i, e in enumerate(d.get("val", d if isinstance(d, list) else [])):
        e_copy = dict(e)
        e_copy["epoch"] = i + 1
        out.append(e_copy)
    return out

# ── health rules ────────────────────────────────────────────────────────

RULES = {
    "val_gap_alt_rmse":   ("↓", "gap_alt_rmse",   "ft"),
    "val_alt_rmse":       ("↓", "alt_rmse",       "ft"),
    "val_altrel_corr":    ("↑", "altrel_corr",    ""),
    "val_pred_std":       ("≠0","pred_std",       ""),
    "val_gap_horiz_rmse_m":("↓","horiz_rmse",     "m"),
}

def _check(epochs: list[dict]) -> dict:
    """Return health dict with warnings and verdict."""
    if len(epochs) < 3:
        return {"verdict": "⌛ waiting for more epochs", "warn": [], "ok": []}

    first = epochs[0]
    last = epochs[-1]
    recent = epochs[-5:] if len(epochs) >= 5 else epochs
    n = len(epochs)

    warn, ok = [], []

    for key, (direction, short, unit) in RULES.items():
        if key not in last:
            continue
        v0, v1 = first[key], last[key]
        delta = v1 - v0

        if direction == "↓":
            if delta < 0:
                ok.append(f"{short}: {v0:.1f}→{v1:.1f} ({delta:+.1f}{unit})")
            elif delta == 0 and n >= 8:
                warn.append(f"{short}: STUCK at {v1:.1f}{unit} ({n} epochs, no change)")
            else:
                ok.append(f"{short}: {v0:.1f}→{v1:.1f} ({delta:+.1f}{unit})")

        elif direction == "↑":
            if delta > 0:
                ok.append(f"{short}: {v0:.3f}→{v1:.3f} ({delta:+.3f})")
            elif n >= 8:
                warn.append(f"{short}: stuck {v0:.3f}→{v1:.3f}")
            else:
                ok.append(f"{short}: {v0:.3f}→{v1:.3f}")

        elif direction == "≠0":
            if v1 < 1.0:
                warn.append(f"{short}: DEAD (={v1:.4f}) — model NOT learning altitude!")
            elif v1 < 50:
                warn.append(f"{short}: LOW ({v1:.1f}) — expect >50 for anchor_relative")
            else:
                ok.append(f"{short}: {v1:.1f} ✓")

    if not warn:
        verdict = "✓ HEALTHY"
    elif len(warn) <= 2:
        verdict = "⚠ WATCH"
    else:
        verdict = "✗ STOP — check config/logic"

    return {"verdict": verdict, "warn": warn, "ok": ok}

# ── output ───────────────────────────────────────────────────────────────

def _epoch_line(e: dict) -> str:
    gap = e.get("val_gap_alt_rmse", 0)
    alt = e.get("val_alt_rmse", 0)
    corr = e.get("val_altrel_corr", 0)
    pstd = e.get("val_pred_std", 0)
    return f"  ep {e['epoch']:>3d} │ gap_alt={gap:>7.1f}  alt={alt:>7.1f}  corr={corr:.3f}  pred_std={pstd:.1f}"

def _health_block(epochs: list[dict], n: int) -> str:
    h = _check(epochs)
    lines = [
        "─" * 55,
        f"  HEALTH CHECK  (epochs {epochs[0]['epoch']}-{epochs[-1]['epoch']}, {len(epochs)} total)  {h['verdict']}",
    ]
    for w in h["warn"]:
        lines.append(f"    ⚠ {w}")
    for o in h["ok"]:
        lines.append(f"    {o}")
    lines.append("─" * 55)
    return "\n".join(lines)

# ── main modes ───────────────────────────────────────────────────────────

def stream_stdin(interval: int):
    """Read epoch lines from stdin, print one-liner each, health every N."""
    epochs = []
    for line in sys.stdin:
        e = _parse_epoch_line(line)
        if e is None:
            continue
        epochs.append(e)
        print(_epoch_line(e), flush=True)
        if e["epoch"] % interval == 0 or e["epoch"] == e.get("total_epochs", 0):
            print(_health_block(epochs, interval), flush=True)
            print()

def follow_logfile(logfile: str, interval: int):
    """Tail a log file, printing epoch lines and periodic health checks."""
    # Wait for file to appear
    waited = False
    while not os.path.exists(logfile):
        if not waited:
            print(f"Waiting for {logfile} ...", flush=True)
            waited = True
        time.sleep(5)
    if waited:
        print(f"File found, monitoring...", flush=True)

    with open(logfile, "r") as f:
        f.seek(0, os.SEEK_END)
        epochs = []
        while True:
            line = f.readline()
            if line:
                e = _parse_epoch_line(line)
                if e is not None:
                    epochs.append(e)
                    print(_epoch_line(e), flush=True)
                    if e["epoch"] % interval == 0 or e["epoch"] == e.get("total_epochs", 0):
                        print(_health_block(epochs, interval), flush=True)
                        print()
            else:
                time.sleep(5)

def snapshot(logfile: str, interval: int):
    """One-shot: read entire log and print health report."""
    with open(logfile, "r") as f:
        text = f.read()
    epochs = []
    for line in text.splitlines():
        e = _parse_epoch_line(line)
        if e is not None:
            epochs.append(e)
    if not epochs:
        print("No epoch data in log.")
        return
    # Print last 5 epoch lines
    for e in epochs[-5:]:
        print(_epoch_line(e))
    print()
    print(_health_block(epochs, interval))

# ── cli ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Streaming training monitor")
    p.add_argument("--logfile", type=str, help="Path to training log")
    p.add_argument("--history", type=str, help="Path to history.json")
    p.add_argument("--stdin", action="store_true", help="Read from stdin (pipe tail -f)")
    p.add_argument("-n", "--interval", type=int, default=5, help="Health check interval (epochs)")
    p.add_argument("--once", action="store_true", help="One-shot snapshot (with --logfile)")
    args = p.parse_args()

    if args.stdin:
        stream_stdin(args.interval)
    elif args.history:
        epochs = _load_history(args.history)
        for e in epochs:
            print(_epoch_line(e))
        print()
        print(_health_block(epochs, args.interval))
    elif args.logfile:
        if args.once:
            snapshot(args.logfile, args.interval)
        else:
            follow_logfile(args.logfile, args.interval)
    else:
        p.print_help()

if __name__ == "__main__":
    main()
