from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select checkpoint by posthoc rule from one training history.")
    parser.add_argument("--run-dir", required=True, help="Training run dir containing history.json and checkpoints/")
    parser.add_argument(
        "--rule",
        default="combo_alt_top3_then_30plus",
        choices=["default_gap_horizontal", "alt_gap_alt", "combo_alt_top3_then_30plus"],
    )
    parser.add_argument("--topk", type=int, default=3, help="Top-k epochs by val_gap_alt_rmse for combo rule.")
    parser.add_argument("--default-metric", default="val_gap_horizontal_rmse_m")
    parser.add_argument("--alt-metric", default="val_gap_alt_rmse")
    parser.add_argument("--secondary-metric", default="val_gap_bucket_30_plus_altrel_rmse")
    parser.add_argument("--out-name", default="best_posthoc_combo.pt")
    return parser


def _argmin_idx(values: list[dict], key: str) -> int:
    return min(range(len(values)), key=lambda i: float(values[i].get(key, float("inf"))))


def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    hist_path = run_dir / "history.json"
    ckpt_dir = run_dir / "checkpoints"
    if not hist_path.exists():
        raise FileNotFoundError(f"history.json not found: {hist_path}")
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"checkpoints dir not found: {ckpt_dir}")

    history = json.loads(hist_path.read_text(encoding="utf-8"))
    if "val" not in history or not isinstance(history["val"], list) or not history["val"]:
        raise RuntimeError("history.json has no non-empty 'val' list.")
    vals = history["val"]

    if args.rule == "default_gap_horizontal":
        idx = _argmin_idx(vals, args.default_metric)
        rule_info = {
            "rule": args.rule,
            "metric": args.default_metric,
            "metric_val": vals[idx].get(args.default_metric),
        }
    elif args.rule == "alt_gap_alt":
        idx = _argmin_idx(vals, args.alt_metric)
        rule_info = {
            "rule": args.rule,
            "metric": args.alt_metric,
            "metric_val": vals[idx].get(args.alt_metric),
        }
    else:
        order = sorted(range(len(vals)), key=lambda i: float(vals[i].get(args.alt_metric, float("inf"))))
        topk = max(1, int(args.topk))
        top_idx = order[:topk]
        idx = min(top_idx, key=lambda i: float(vals[i].get(args.secondary_metric, float("inf"))))
        rule_info = {
            "rule": args.rule,
            "primary_metric": args.alt_metric,
            "topk": topk,
            "topk_epochs": [int(i + 1) for i in top_idx],
            "topk_primary_vals": [vals[i].get(args.alt_metric) for i in top_idx],
            "secondary_metric": args.secondary_metric,
            "secondary_metric_val": vals[idx].get(args.secondary_metric),
        }

    epoch = int(idx + 1)
    src_ckpt = ckpt_dir / f"epoch_{epoch:03d}.pt"
    if not src_ckpt.exists():
        raise FileNotFoundError(f"Selected checkpoint not found: {src_ckpt}")

    dst_ckpt = run_dir / args.out_name
    shutil.copy2(src_ckpt, dst_ckpt)

    summary = {
        "run_dir": str(run_dir),
        "selected_epoch": epoch,
        "selected_checkpoint": str(src_ckpt),
        "exported_checkpoint": str(dst_ckpt),
        "rule_info": rule_info,
        "selected_val_metrics": {
            args.default_metric: vals[idx].get(args.default_metric),
            args.alt_metric: vals[idx].get(args.alt_metric),
            args.secondary_metric: vals[idx].get(args.secondary_metric),
        },
    }
    out_json = run_dir / "posthoc_checkpoint_selection.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[ok] posthoc_selection={out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
