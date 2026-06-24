from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Summarize Exp0/1/2 risk-aware round1")
    p.add_argument("--exp0-dir", required=True)
    p.add_argument("--exp1-dir", required=True)
    p.add_argument("--exp2-dir", required=True)
    p.add_argument("--out-dir", required=True)
    return p


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_map = {
        "exp0_baseline_recheck": Path(args.exp0_dir),
        "exp1_bounded_residual_uniform": Path(args.exp1_dir),
        "exp2_risk_aware_teacher_fixed": Path(args.exp2_dir),
    }

    rows = []
    for name, p in exp_map.items():
        eval_json = _read_json(p / "main_task_metrics_test.json")
        replay_summary = _safe_read_csv(p / "replay" / "production_chain_v1_keep_warn_abnormal_summary.csv")
        replay_seg = _safe_read_csv(p / "replay" / "production_chain_v1_segment_audit.csv")
        keep_ratio = warn_ratio = abnormal_ratio = float("nan")
        if not replay_summary.empty:
            mp = {str(r["label"]): float(r["ratio"]) for _, r in replay_summary.iterrows()}
            keep_ratio = mp.get("keep", float("nan"))
            warn_ratio = mp.get("warn", float("nan"))
            abnormal_ratio = mp.get("abnormal", float("nan"))
        overshoot_rate = float(replay_seg["overshoot_flag"].mean()) if ("overshoot_flag" in replay_seg.columns and len(replay_seg)) else float("nan")
        edge_spike_rate = float(replay_seg["edge_spike_flag"].mean()) if ("edge_spike_flag" in replay_seg.columns and len(replay_seg)) else float("nan")
        rows.append(
            {
                "experiment_name": name,
                "run_dir": str(p),
                "alt_rmse_test": float(eval_json.get("alt_rmse", float("nan"))),
                "gap_alt_rmse_test": float(eval_json.get("gap_alt_rmse", float("nan"))),
                "lat_rmse_test": float(eval_json.get("lat_rmse", float("nan"))),
                "lon_rmse_test": float(eval_json.get("lon_rmse", float("nan"))),
                "keep_ratio_replay": keep_ratio,
                "warn_ratio_replay": warn_ratio,
                "abnormal_ratio_replay": abnormal_ratio,
                "overshoot_rate_replay": overshoot_rate,
                "edge_spike_rate_replay": edge_spike_rate,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "round1_compare_summary.csv", index=False)

    # Q1~Q6 quick answers scaffold from exported stats.
    ans = []
    ans.append("Q1 risk_rules命中预期高风险结构: 查看 segment_teacher_audit/*/key_pattern_rule_hits.csv。")
    if len(summary):
        s0 = summary[summary["experiment_name"].eq("exp0_baseline_recheck")].head(1)
        s1 = summary[summary["experiment_name"].eq("exp1_bounded_residual_uniform")].head(1)
        s2 = summary[summary["experiment_name"].eq("exp2_risk_aware_teacher_fixed")].head(1)
        if len(s0) and len(s1):
            ans.append(
                f"Q4 Exp-1 vs Exp-0 overshoot/edge_spike: {float(s1['overshoot_rate_replay'].iloc[0]):.4f}/{float(s1['edge_spike_rate_replay'].iloc[0]):.4f} vs "
                f"{float(s0['overshoot_rate_replay'].iloc[0]):.4f}/{float(s0['edge_spike_rate_replay'].iloc[0]):.4f}"
            )
        if len(s1) and len(s2):
            ans.append(
                f"Q5 Exp-2 vs Exp-1 abnormal/overshoot/edge_spike: {float(s2['abnormal_ratio_replay'].iloc[0]):.4f}/"
                f"{float(s2['overshoot_rate_replay'].iloc[0]):.4f}/{float(s2['edge_spike_rate_replay'].iloc[0]):.4f} vs "
                f"{float(s1['abnormal_ratio_replay'].iloc[0]):.4f}/{float(s1['overshoot_rate_replay'].iloc[0]):.4f}/{float(s1['edge_spike_rate_replay'].iloc[0]):.4f}"
            )
        if len(s0) and len(s2):
            ans.append(
                f"Q6 Exp-2 ADS-B test alt_rmse/gap_alt_rmse vs Exp-0: {float(s2['alt_rmse_test'].iloc[0]):.4f}/{float(s2['gap_alt_rmse_test'].iloc[0]):.4f} vs "
                f"{float(s0['alt_rmse_test'].iloc[0]):.4f}/{float(s0['gap_alt_rmse_test'].iloc[0]):.4f}"
            )
    (out_dir / "round1_six_questions_summary.txt").write_text("\n".join(ans), encoding="utf-8")
    print(f"[ok] {out_dir / 'round1_compare_summary.csv'}")
    print(f"[ok] {out_dir / 'round1_six_questions_summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

