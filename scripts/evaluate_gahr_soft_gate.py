from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.anchor_graph_height_reference import (  # noqa: E402
    AnchorGraphParams,
    anchor_graph_gap_features,
    build_anchor_graph_reference,
    build_reference_candidates,
)


FEATURE_COLS = [
    "anchor_count",
    "gap_len",
    "delta_z",
    "abs_delta_z",
    "left_context_std",
    "right_context_std",
    "left_context_median_delta",
    "right_context_median_delta",
    "prev_delta_z",
    "next_delta_z",
    "gap_index_ratio",
]
MODES = ["hold", "switch", "trend"]


class SoftGate(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _extract_gap_examples(df: pd.DataFrame, params: AnchorGraphParams) -> tuple[pd.DataFrame, dict[str, dict]]:
    rows = []
    gap_payload: dict[str, dict] = {}
    for sid, g in df.groupby("sample_id", sort=False):
        g = g.sort_values("minute_index").reset_index()
        obs = g["obs_mask"].to_numpy(dtype=int) == 1
        anchors = np.where(obs)[0]
        if len(anchors) < 2:
            continue
        anchor_alt = g.loc[anchors, "adsb_alt_m"].to_numpy(dtype=float)
        for edge_i, (left, right) in enumerate(zip(anchors[:-1], anchors[1:])):
            n = int(right - left + 1)
            if n <= 2:
                continue
            z_left = float(g["adsb_alt_m"].iloc[left])
            z_right = float(g["adsb_alt_m"].iloc[right])
            raw = g["本文方案_alt_m"].iloc[left : right + 1].to_numpy(dtype=float)
            truth = g["adsb_alt_m"].iloc[left : right + 1].to_numpy(dtype=float)
            candidates = build_reference_candidates(z_left, z_right, n, raw, params)
            inner = np.arange(1, n - 1)
            errs = {}
            for mode, curve in candidates.items():
                e = curve[inner] - truth[inner]
                errs[mode] = float(np.sqrt(np.mean(np.square(e))))
            label_mode = min(errs, key=errs.get)
            label = MODES.index(label_mode)
            feat = anchor_graph_gap_features(anchor_alt, edge_i, right - left - 1, z_left, z_right, params)
            gap_id = f"{sid}__gap{edge_i:03d}"
            row = {
                "gap_id": gap_id,
                "sample_id": sid,
                "source_case": g["source_case"].iloc[0],
                "anchor_count": int(g["anchor_count"].iloc[0]),
                "edge_i": int(edge_i),
                "label": int(label),
                "label_mode": label_mode,
                "rmse_hold": errs["hold"],
                "rmse_switch": errs["switch"],
                "rmse_trend": errs["trend"],
            }
            row.update(feat)
            rows.append(row)
            gap_payload[gap_id] = {
                "sample_id": sid,
                "source_case": g["source_case"].iloc[0],
                "row_indices": g["index"].iloc[left : right + 1].to_numpy(dtype=int),
                "inner_local_indices": inner,
                "candidates": candidates,
            }
    return pd.DataFrame(rows), gap_payload


def _fit_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(x, axis=0)
    sigma = np.nanstd(x, axis=0)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    return mu.astype(np.float32), sigma.astype(np.float32)


def _train_gate(x: np.ndarray, y: np.ndarray, epochs: int = 500, seed: int = 42) -> SoftGate:
    torch.manual_seed(seed)
    model = SoftGate(x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    counts = np.bincount(y, minlength=3).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    tx = torch.tensor(x, dtype=torch.float32)
    ty = torch.tensor(y, dtype=torch.long)
    model.train()
    for _ in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(tx), ty)
        loss.backward()
        opt.step()
    return model.eval()


def _predict_probs(model: SoftGate, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        p = torch.softmax(model(torch.tensor(x, dtype=torch.float32)), dim=-1).cpu().numpy()
    return p


def _build_oof_soft_predictions(df: pd.DataFrame, examples: pd.DataFrame, payload: dict[str, dict], epochs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out["GAHR_soft_alt_m"] = np.nan
    out["GAHR_oracle_alt_m"] = np.nan
    pred_rows = []
    cases = sorted(examples["source_case"].unique())
    for case in cases:
        train = examples[examples["source_case"] != case].reset_index(drop=True)
        test = examples[examples["source_case"] == case].reset_index(drop=True)
        if len(train) == 0 or len(test) == 0:
            continue
        x_train_raw = train[FEATURE_COLS].to_numpy(dtype=np.float32)
        y_train = train["label"].to_numpy(dtype=np.int64)
        mu, sigma = _fit_scaler(x_train_raw)
        x_train = (x_train_raw - mu) / sigma
        x_test = (test[FEATURE_COLS].to_numpy(dtype=np.float32) - mu) / sigma
        model = _train_gate(x_train, y_train, epochs=epochs)
        probs = _predict_probs(model, x_test)
        for i, row in test.iterrows():
            gap_id = row["gap_id"]
            p = probs[i]
            curves = payload[gap_id]["candidates"]
            soft_curve = p[0] * curves["hold"] + p[1] * curves["switch"] + p[2] * curves["trend"]
            oracle_curve = curves[str(row["label_mode"])]
            idx = payload[gap_id]["row_indices"]
            out.loc[idx, "GAHR_soft_alt_m"] = soft_curve
            out.loc[idx, "GAHR_oracle_alt_m"] = oracle_curve
            pred_rows.append(
                {
                    "gap_id": gap_id,
                    "source_case": row["source_case"],
                    "sample_id": row["sample_id"],
                    "anchor_count": int(row["anchor_count"]),
                    "label_mode": row["label_mode"],
                    "pi_hold": float(p[0]),
                    "pi_switch": float(p[1]),
                    "pi_trend": float(p[2]),
                    "pred_mode": MODES[int(np.argmax(p))],
                }
            )
    anchor_mask = out["obs_mask"].to_numpy(dtype=int) == 1
    out.loc[anchor_mask, "GAHR_soft_alt_m"] = out.loc[anchor_mask, "adsb_alt_m"]
    out.loc[anchor_mask, "GAHR_oracle_alt_m"] = out.loc[anchor_mask, "adsb_alt_m"]
    return out, pd.DataFrame(pred_rows)


def _metrics_by_sample(df: pd.DataFrame, pred_col: str, model: str) -> pd.DataFrame:
    rows = []
    for sid, g in df.groupby("sample_id", sort=False):
        gap = g["obs_mask"].to_numpy(dtype=int) == 0
        truth = g.loc[gap, "adsb_alt_m"].to_numpy(dtype=float)
        pred = g.loc[gap, pred_col].to_numpy(dtype=float)
        ok = np.isfinite(truth) & np.isfinite(pred)
        if not ok.any():
            continue
        e = pred[ok] - truth[ok]
        rows.append(
            {
                "model": model,
                "sample_id": sid,
                "source_case": g["source_case"].iloc[0],
                "anchor_count": int(g["anchor_count"].iloc[0]),
                "alt_RMSE_m": float(np.sqrt(np.mean(np.square(e)))),
                "alt_MAE_m": float(np.mean(np.abs(e))),
                "alt_MaxAE_m": float(np.max(np.abs(e))),
            }
        )
    return pd.DataFrame(rows)


def _plot(df: pd.DataFrame, out_dir: Path, anchor_counts: set[int]) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for (source_case, anchor_count), g in df.groupby(["source_case", "anchor_count"], sort=False):
        if int(anchor_count) not in anchor_counts:
            continue
        g = g.sort_values("minute_index")
        x = g["minute_index"].to_numpy(dtype=float)
        obs = g["obs_mask"].to_numpy(dtype=int) == 1
        fig, ax = plt.subplots(figsize=(12.5, 5.2), facecolor="white")
        ax.plot(x, g["adsb_alt_m"], color="black", lw=2.2, label="ADS-B truth")
        ax.scatter(x[obs], g.loc[obs, "adsb_alt_m"], color="black", marker="*", s=130, zorder=7, label="anchors")
        ax.plot(x, g["本文方案_alt_m"], color="#aaaaaa", lw=1.2, alpha=0.75, label="Ours-A3 raw")
        ax.plot(x, g["分段线性插值_alt_m"], color="#2a9d8f", lw=1.2, alpha=0.75, label="Linear")
        ax.plot(x, g["AnchorGraph_rule_alt_m"], color="#457b9d", lw=1.6, alpha=0.9, label="GAHR-rule")
        ax.plot(x, g["GAHR_soft_alt_m"], color="#d62828", lw=2.4, label="GAHR-soft")
        ax.set_title(f"{source_case} | anchors={anchor_count}")
        ax.set_xlabel("Cruise window minute")
        ax.set_ylabel("Altitude (m)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=4)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{source_case}_anchor{anchor_count}_gahr_soft.png", dpi=180)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs/runs/complete_adsb_sparse_cruise_eval_20260519")
    parser.add_argument("--out-dir", default="outputs/runs/gahr_soft_gate_trial_20260520")
    parser.add_argument("--epochs", type=int, default=600)
    args = parser.parse_args()

    input_dir = _resolve(args.input_dir)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = AnchorGraphParams()
    df = pd.read_csv(input_dir / "sparse_cruise_model_predictions.csv")
    examples, payload = _extract_gap_examples(df, params)
    examples.to_csv(out_dir / "gahr_soft_gap_training_examples.csv", index=False, encoding="utf-8-sig")
    out, pred_rows = _build_oof_soft_predictions(df, examples, payload, epochs=args.epochs)

    # Add deterministic rule reference for direct comparison.
    refs = []
    for _, g in out.groupby("sample_id", sort=False):
        g = g.sort_values("minute_index")
        ref, _ = build_anchor_graph_reference(
            time_index=g["minute_index"].to_numpy(dtype=float),
            anchor_mask=g["obs_mask"].to_numpy(dtype=int) == 1,
            anchor_alt_or_truth=np.where(g["obs_mask"].to_numpy(dtype=int) == 1, g["adsb_alt_m"].to_numpy(dtype=float), np.nan),
            raw_alt=g["本文方案_alt_m"].to_numpy(dtype=float),
            params=params,
        )
        refs.append(pd.Series(ref, index=g.index))
    out["AnchorGraph_rule_alt_m"] = pd.concat(refs).sort_index()
    out.to_csv(out_dir / "gahr_soft_predictions.csv", index=False, encoding="utf-8-sig")
    pred_rows.to_csv(out_dir / "gahr_soft_oof_gate_predictions.csv", index=False, encoding="utf-8-sig")

    metric_parts = [
        _metrics_by_sample(out, "本文方案_alt_m", "Ours-A3 raw"),
        _metrics_by_sample(out, "分段线性插值_alt_m", "Linear"),
        _metrics_by_sample(out, "AnchorGraph_rule_alt_m", "GAHR-rule"),
        _metrics_by_sample(out, "GAHR_soft_alt_m", "GAHR-soft"),
        _metrics_by_sample(out, "GAHR_oracle_alt_m", "GAHR-oracle-upper"),
    ]
    metrics = pd.concat(metric_parts, ignore_index=True)
    metrics.to_csv(out_dir / "gahr_soft_metrics_by_sample.csv", index=False, encoding="utf-8-sig")
    by_model = metrics.groupby("model", as_index=False).agg(
        sample_count=("sample_id", "nunique"),
        alt_RMSE_m=("alt_RMSE_m", "mean"),
        alt_MAE_m=("alt_MAE_m", "mean"),
        alt_MaxAE_m=("alt_MaxAE_m", "mean"),
    )
    by_anchor = metrics.groupby(["model", "anchor_count"], as_index=False).agg(
        sample_count=("sample_id", "nunique"),
        alt_RMSE_m=("alt_RMSE_m", "mean"),
        alt_MAE_m=("alt_MAE_m", "mean"),
        alt_MaxAE_m=("alt_MaxAE_m", "mean"),
    )
    by_model.to_csv(out_dir / "gahr_soft_metrics_by_model.csv", index=False, encoding="utf-8-sig")
    by_anchor.to_csv(out_dir / "gahr_soft_metrics_by_anchor_count.csv", index=False, encoding="utf-8-sig")
    _plot(out, out_dir, anchor_counts={3, 8})

    print(f"[done] out_dir={out_dir}")
    print("\n[label distribution]")
    print(examples["label_mode"].value_counts().to_string())
    print("\n[OOF gate confusion]")
    print(pd.crosstab(pred_rows["label_mode"], pred_rows["pred_mode"]).to_string())
    print("\n[by model]")
    print(by_model.round(3).to_string(index=False))
    print("\n[by anchor count]")
    print(by_anchor.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
