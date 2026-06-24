from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.datasets import DatasetConfig, TrajectoryDataset
from src.training import load_config, set_seed, split_by_flight_id
from src.training.altitude_governance import add_anchor_alt_features, add_vertical_v2_features


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Audit segment teacher rule matching on train split")
    p.add_argument("--config", required=True)
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--out-dir", default="")
    return p


def _to_df(ds: TrajectoryDataset) -> pd.DataFrame:
    rows = []
    for s in ds.samples:
        rows.append(
            {
                "sample_id": str(s["sample_id"]),
                "flight_id": str(s["flight_id"]),
                "segment_len": float(s["segment_len"].item()),
                "segment_bucket": str(s["segment_bucket_name"]),
                "anchor_pattern": str(s["anchor_pattern_name"]),
                "risk_flag": int(s["risk_flag"].item()),
                "risk_level": str(s["risk_level"]),
                "risk_flag_teacher": int(float(s["risk_flag_teacher"].item()) > 0.5),
                "teacher_scale": float(s["teacher_scale"].item()),
                "edge_weight": float(s["edge_weight"].item()),
                "residual_rmax_ft": float(s["residual_rmax_ft"].item()),
                "gate_bias": float(s["gate_bias"].item()),
                "matched_risk_rule": str(s["matched_risk_rule"]),
                "fallback_flag": int(str(s["matched_risk_rule"]).lower() in {"legacy_meta", "no_gap_default", "default"}),
            }
        )
    return pd.DataFrame(rows)


def _value_stats(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["risk_level", f"{col}_mean", f"{col}_min", f"{col}_max"])
    out = (
        df.groupby("risk_level", as_index=False)
        .agg(
            **{
                f"{col}_mean": (col, "mean"),
                f"{col}_min": (col, "min"),
                f"{col}_max": (col, "max"),
            }
        )
    )
    return out


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    samples_path = Path(cfg["data"]["samples_path"])
    if not samples_path.exists():
        raise RuntimeError(f"samples_path not found: {samples_path}")
    df = pd.read_parquet(samples_path)
    split = split_by_flight_id(
        df,
        flight_id_col=str(cfg["data"]["flight_id_col"]),
        train_ratio=float(cfg["data"]["split"]["train_ratio"]),
        val_ratio=float(cfg["data"]["split"]["val_ratio"]),
        seed=int(cfg.get("seed", 42)),
    )
    sdf = split[str(args.split)]
    sdf = add_anchor_alt_features(sdf)
    sdf = add_vertical_v2_features(sdf)
    dcfg = DatasetConfig(
        sample_id_col=cfg["data"]["sample_id_col"],
        flight_id_col=cfg["data"]["flight_id_col"],
        time_col=cfg["data"]["time_col"],
        target_cols=cfg["data"]["target_cols"],
        obs_cols=cfg["data"]["obs_cols"],
        obs_mask_col=cfg["data"]["obs_mask_col"],
        exo_cols=cfg["data"]["exo_cols"],
        vertical_exo_cols=cfg["data"].get("vertical_exo_cols", []),
        quality_cols=cfg["data"].get("quality_cols", []),
        split_on_time_gap=bool(cfg["data"].get("split_on_time_gap", True)),
        max_time_gap_minutes=float(cfg["data"].get("max_time_gap_minutes", 5.0)),
        short_segment_max_minutes=int(cfg["data"].get("short_segment_max_minutes", 15)),
        medium_segment_max_minutes=int(cfg["data"].get("medium_segment_max_minutes", 60)),
        segment_risk_rules_path=cfg["data"].get("segment_risk_rules_path"),
    )
    ds = TrajectoryDataset(sdf, dcfg)
    meta = _to_df(ds)

    run_name = Path(cfg["outputs"]["run_dir"]).name
    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs/segment_teacher_audit") / run_name / str(args.split)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out_dir / "segment_teacher_sample_meta.csv", index=False)

    overall = pd.DataFrame(
        [
            {"group": "risk_level", "key": k, "count": int(v)}
            for k, v in meta["risk_level"].value_counts(dropna=False).items()
        ]
        + [
            {"group": "matched_risk_rule", "key": k, "count": int(v)}
            for k, v in meta["matched_risk_rule"].value_counts(dropna=False).head(200).items()
        ]
        + [
            {"group": "risk_flag_teacher", "key": str(k), "count": int(v)}
            for k, v in meta["risk_flag_teacher"].value_counts(dropna=False).items()
        ]
    )
    overall["ratio"] = overall["count"] / max(1, len(meta))
    overall.to_csv(out_dir / "overall_distribution.csv", index=False)

    cross_rule = (
        meta.groupby(["segment_bucket", "anchor_pattern", "matched_risk_rule"], as_index=False)
        .size()
        .rename(columns={"size": "total"})
        .sort_values("total", ascending=False)
    )
    cross_rule.to_csv(out_dir / "cross_bucket_anchor_rule.csv", index=False)
    cross_level = (
        meta.groupby(["segment_bucket", "anchor_pattern", "risk_level"], as_index=False)
        .size()
        .rename(columns={"size": "total"})
        .sort_values("total", ascending=False)
    )
    cross_level.to_csv(out_dir / "cross_bucket_anchor_risklevel.csv", index=False)

    v1 = _value_stats(meta, "teacher_scale")
    v2 = _value_stats(meta, "edge_weight")
    v3 = _value_stats(meta, "residual_rmax_ft")
    v4 = _value_stats(meta, "gate_bias")
    param_by_risk = v1.merge(v2, on="risk_level", how="outer").merge(v3, on="risk_level", how="outer").merge(v4, on="risk_level", how="outer")
    param_by_risk.to_csv(out_dir / "teacher_params_by_risk_level.csv", index=False)

    fallback_ratio = float((meta["fallback_flag"] > 0).mean()) if len(meta) else 0.0
    summary = {
        "samples": int(len(meta)),
        "split": str(args.split),
        "run_name": run_name,
        "fallback_ratio": fallback_ratio,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    def _hit(seg_b: str, anc: str) -> pd.DataFrame:
        x = meta[meta["segment_bucket"].eq(seg_b) & meta["anchor_pattern"].eq(anc)].copy()
        if x.empty:
            return pd.DataFrame(columns=["segment_bucket", "anchor_pattern", "matched_risk_rule", "count", "ratio"])
        g = (
            x.groupby("matched_risk_rule", as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values("count", ascending=False)
        )
        g["ratio"] = g["count"] / max(1, len(x))
        g.insert(0, "anchor_pattern", anc)
        g.insert(0, "segment_bucket", seg_b)
        return g

    key_hits = pd.concat(
        [
            _hit("short", "two_anchor"),
            _hit("medium", "asymmetric"),
            _hit("long", "asymmetric"),
        ],
        ignore_index=True,
    )
    key_hits.to_csv(out_dir / "key_pattern_rule_hits.csv", index=False)

    print(f"[audit] out_dir={out_dir}")
    print(f"[audit] samples={len(meta)} fallback_ratio={fallback_ratio:.4f}")
    for seg_b, anc in [("short", "two_anchor"), ("medium", "asymmetric"), ("long", "asymmetric")]:
        m = key_hits[key_hits["segment_bucket"].eq(seg_b) & key_hits["anchor_pattern"].eq(anc)]
        top = m.head(3).to_dict(orient="records")
        print(f"[audit] key_hit {seg_b}+{anc}: {top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
