from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.preprocessing.standardize import (
    apply_standardizer,
    build_standardization_report,
    fit_standardizer,
    select_continuous_feature_cols,
)
from src.training import load_config, split_by_flight_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report feature standardization before/after statistics.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--out-csv", default="outputs/reports/feature_standardization_stats.csv")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_config(args.config)

    df = pd.read_parquet(cfg["data"]["samples_path"])
    splits = split_by_flight_id(
        df=df,
        flight_id_col=cfg["data"]["flight_id_col"],
        train_ratio=float(cfg["data"]["split"]["train_ratio"]),
        val_ratio=float(cfg["data"]["split"]["val_ratio"]),
        seed=int(cfg.get("seed", 42)),
    )

    candidate_cols = list(dict.fromkeys(["dt_prev", "dt_next"] + cfg["data"]["exo_cols"] + cfg["data"]["quality_cols"]))
    continuous_cols = select_continuous_feature_cols(
        splits["train"],
        candidate_cols=candidate_cols,
        exclude_cols={cfg["data"]["obs_mask_col"]},
    )
    stats = fit_standardizer(splits["train"], feature_cols=continuous_cols)
    train_scaled = apply_standardizer(splits["train"], stats)
    report = build_standardization_report(
        before_train=splits["train"][list(stats.keys())],
        after_train=train_scaled[list(stats.keys())],
        stats=stats,
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_csv, index=False)
    print(f"[ok] report={out_csv} features={len(report)}")
    print(report.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
