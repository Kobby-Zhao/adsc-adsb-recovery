from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "outputs/analysis/stagewise_eval_proto_compare_24e"
DEFAULT_OUT = DEFAULT_IN / "stagewise_proto_compare_summary.csv"

DISPLAY = {
    "proposed": "本文方案",
    "kalman_filter": "Kalman Filter",
    "lstm_proto": "LSTM",
    "bilstm_proto": "BiLSTM",
    "cnnlstm_proto": "CNN+LSTM",
    "transformer_proto": "Transformer",
    "bimamba": "bimamba",
    "mamba_proto": "mamba",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", default=str(DEFAULT_IN))
    ap.add_argument("--out-csv", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    rows = []
    for p in sorted(Path(args.in_root).glob("*/main_task_metrics_test_summary_dim.csv")):
        model_key, stage = p.parent.name.rsplit("_", 1)
        df = pd.read_csv(p)
        row = df.iloc[0].to_dict()
        rows.append({
            "model_key": model_key,
            "model": DISPLAY.get(model_key, model_key),
            "dataset": stage,
            "rmse_lon": row["gap_rmse_dim1"],
            "rmse_lat": row["gap_rmse_dim0"],
            "rmse_alt": row["gap_rmse_dim2"],
            "mae_lon": row["gap_mae_dim1"],
            "mae_lat": row["gap_mae_dim0"],
            "mae_alt": row["gap_mae_dim2"],
        })

    out = pd.DataFrame(rows)
    if out.empty:
        print("[warn] no stagewise evaluation results found")
        return 1

    out["dataset"] = pd.Categorical(out["dataset"], ["S1", "S2", "S3"], ordered=True)
    out = out.sort_values(["model", "dataset"]).reset_index(drop=True)
    out.to_csv(args.out_csv, index=False)
    print(f"[ok] {args.out_csv}")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
