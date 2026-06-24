from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
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


CANDIDATE_MODES = ["trend", "hold_left", "switch_early", "switch_mid", "switch_late", "switch_right"]

FEATURE_COLS = [
    "anchor_count",
    "gap_len_min",
    "edge_index_ratio",
    "delta_alt_m",
    "abs_delta_alt_m",
    "prev_delta_alt_m",
    "next_delta_alt_m",
    "left_context_std_m",
    "right_context_std_m",
    "left_context_median_delta_m",
    "right_context_median_delta_m",
    "anchor_horizontal_dist_deg",
    "anchor_lat_delta_deg",
    "anchor_lon_delta_deg",
    "anchor_abs_lat_mean_deg",
]


@dataclass(frozen=True)
class LocatorParams:
    transition_m_per_min: float = 150.0
    min_transition_min: int = 2
    max_transition_min: int = 8
    context_radius: int = 2


class TransitionLocator(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 48),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(48, 24),
            nn.SiLU(),
            nn.Linear(24, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _smoothstep(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _transition_curve(z_left: float, z_right: float, n: int, center_ratio: float, params: LocatorParams) -> np.ndarray:
    if n <= 1:
        return np.asarray([z_left], dtype=float)
    dz = float(z_right - z_left)
    duration = int(np.clip(round(abs(dz) / params.transition_m_per_min), params.min_transition_min, params.max_transition_min))
    center = int(round((n - 1) * float(center_ratio)))
    center = int(np.clip(center, 1, max(1, n - 2)))
    start = int(np.clip(center - duration // 2, 1, max(1, n - duration - 1)))
    end = int(np.clip(start + duration, start + 1, n - 1))
    out = np.full(n, float(z_left), dtype=float)
    u = np.linspace(0.0, 1.0, end - start + 1)
    out[start : end + 1] = float(z_left) + dz * _smoothstep(u)
    out[end + 1 :] = float(z_right)
    out[0] = float(z_left)
    out[-1] = float(z_right)
    return out


def _candidate_curves(z_left: float, z_right: float, n: int, params: LocatorParams) -> dict[str, np.ndarray]:
    trend = np.linspace(float(z_left), float(z_right), int(n))
    return {
        "trend": trend,
        "hold_left": _transition_curve(z_left, z_right, n, 0.95, params),
        "switch_early": _transition_curve(z_left, z_right, n, 0.25, params),
        "switch_mid": _transition_curve(z_left, z_right, n, 0.50, params),
        "switch_late": _transition_curve(z_left, z_right, n, 0.75, params),
        "switch_right": _transition_curve(z_left, z_right, n, 0.90, params),
    }


def _std(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) <= 1:
        return 0.0
    return float(np.std(values))


def _median_or(values: np.ndarray, fallback: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float(fallback)
    return float(np.median(values))


def _build_gap_examples(df: pd.DataFrame, params: LocatorParams) -> tuple[pd.DataFrame, dict[str, dict]]:
    rows: list[dict] = []
    payload: dict[str, dict] = {}
    for sample_id, g0 in df.groupby("sample_id", sort=False):
        g = g0.sort_values("minute_index").reset_index()
        obs = g["obs_mask"].to_numpy(dtype=int) == 1
        anchors = np.where(obs)[0]
        if len(anchors) < 2:
            continue
        anchor_alt = g.loc[anchors, "adsb_alt_m"].to_numpy(dtype=float)
        anchor_lat = g.loc[anchors, "adsb_lat"].to_numpy(dtype=float)
        anchor_lon = g.loc[anchors, "adsb_lon"].to_numpy(dtype=float)
        source_case = str(g["source_case"].iloc[0])
        for edge_i, (left, right) in enumerate(zip(anchors[:-1], anchors[1:])):
            n = int(right - left + 1)
            if n <= 2:
                continue
            z_left = float(g["adsb_alt_m"].iloc[left])
            z_right = float(g["adsb_alt_m"].iloc[right])
            truth = g["adsb_alt_m"].iloc[left : right + 1].to_numpy(dtype=float)
            curves = _candidate_curves(z_left, z_right, n, params)
            inner = np.arange(1, n - 1)
            rmse = {
                mode: float(np.sqrt(np.mean(np.square(curve[inner] - truth[inner]))))
                for mode, curve in curves.items()
            }
            label_mode = min(rmse, key=rmse.get)
            label = CANDIDATE_MODES.index(label_mode)

            lo = max(0, edge_i - params.context_radius)
            hi = min(len(anchor_alt), edge_i + 2 + params.context_radius)
            left_ctx = anchor_alt[lo : edge_i + 1]
            right_ctx = anchor_alt[edge_i + 1 : hi]
            prev_delta = float(anchor_alt[edge_i] - anchor_alt[edge_i - 1]) if edge_i >= 1 else 0.0
            next_delta = float(anchor_alt[edge_i + 2] - anchor_alt[edge_i + 1]) if edge_i + 2 < len(anchor_alt) else 0.0
            lat_delta = float(anchor_lat[edge_i + 1] - anchor_lat[edge_i])
            lon_delta = float(anchor_lon[edge_i + 1] - anchor_lon[edge_i])
            # Use direct angular distance as an auditable proxy. It is visible from anchors only.
            horizontal_dist = float(np.sqrt(lat_delta * lat_delta + lon_delta * lon_delta))
            gap_id = f"{sample_id}__gap{edge_i:03d}"
            row = {
                "gap_id": gap_id,
                "sample_id": sample_id,
                "source_case": source_case,
                "anchor_count": int(g["anchor_count"].iloc[0]),
                "edge_i": int(edge_i),
                "label": int(label),
                "label_mode": label_mode,
                "oracle_best_rmse_m": float(rmse[label_mode]),
                "linear_rmse_m": float(rmse["trend"]),
                "gap_len_min": float(right - left - 1),
                "edge_index_ratio": float(edge_i / max(len(anchors) - 2, 1)),
                "delta_alt_m": float(z_right - z_left),
                "abs_delta_alt_m": float(abs(z_right - z_left)),
                "prev_delta_alt_m": prev_delta,
                "next_delta_alt_m": next_delta,
                "left_context_std_m": _std(left_ctx),
                "right_context_std_m": _std(right_ctx),
                "left_context_median_delta_m": float(z_left - _median_or(left_ctx, z_left)),
                "right_context_median_delta_m": float(z_right - _median_or(right_ctx, z_right)),
                "anchor_horizontal_dist_deg": horizontal_dist,
                "anchor_lat_delta_deg": lat_delta,
                "anchor_lon_delta_deg": lon_delta,
                "anchor_abs_lat_mean_deg": float((abs(anchor_lat[edge_i]) + abs(anchor_lat[edge_i + 1])) / 2.0),
            }
            for mode, value in rmse.items():
                row[f"rmse_{mode}_m"] = float(value)
            rows.append(row)
            payload[gap_id] = {
                "row_indices": g["index"].iloc[left : right + 1].to_numpy(dtype=int),
                "curves": curves,
                "label_mode": label_mode,
            }
    return pd.DataFrame(rows), payload


def _fit_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(x, axis=0)
    sigma = np.nanstd(x, axis=0)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    return mu.astype(np.float32), sigma.astype(np.float32)


def _train_model(x: np.ndarray, y: np.ndarray, epochs: int, seed: int) -> TransitionLocator:
    torch.manual_seed(seed)
    model = TransitionLocator(x.shape[1], len(CANDIDATE_MODES))
    counts = np.bincount(y, minlength=len(CANDIDATE_MODES)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / max(float(weights.mean()), 1e-6)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    tx = torch.tensor(x, dtype=torch.float32)
    ty = torch.tensor(y, dtype=torch.long)
    model.train()
    for _ in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(tx), ty)
        loss.backward()
        opt.step()
    return model.eval()


def _predict(model: TransitionLocator, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32))
        return torch.softmax(logits, dim=-1).cpu().numpy()


def _apply_leave_one_case_locator(
    df: pd.DataFrame,
    examples: pd.DataFrame,
    payload: dict[str, dict],
    epochs: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out["ATL_oracle_alt_m"] = np.nan
    out["ATL_learned_hard_alt_m"] = np.nan
    out["ATL_learned_soft_alt_m"] = np.nan
    pred_rows: list[dict] = []
    for case in sorted(examples["source_case"].unique()):
        train = examples[examples["source_case"] != case].reset_index(drop=True)
        test = examples[examples["source_case"] == case].reset_index(drop=True)
        if train.empty or test.empty:
            continue
        x_train_raw = train[FEATURE_COLS].to_numpy(dtype=np.float32)
        y_train = train["label"].to_numpy(dtype=np.int64)
        mu, sigma = _fit_scaler(x_train_raw)
        x_train = (x_train_raw - mu) / sigma
        x_test = (test[FEATURE_COLS].to_numpy(dtype=np.float32) - mu) / sigma
        model = _train_model(x_train, y_train, epochs=epochs, seed=seed)
        probs = _predict(model, x_test)
        for i, row in test.iterrows():
            gap_id = str(row["gap_id"])
            curves = payload[gap_id]["curves"]
            oracle_curve = curves[str(row["label_mode"])]
            hard_mode = CANDIDATE_MODES[int(np.argmax(probs[i]))]
            hard_curve = curves[hard_mode]
            soft_curve = np.zeros_like(hard_curve, dtype=float)
            for j, mode in enumerate(CANDIDATE_MODES):
                soft_curve += float(probs[i, j]) * curves[mode]
            idx = payload[gap_id]["row_indices"]
            out.loc[idx, "ATL_oracle_alt_m"] = oracle_curve
            out.loc[idx, "ATL_learned_hard_alt_m"] = hard_curve
            out.loc[idx, "ATL_learned_soft_alt_m"] = soft_curve
            pred_row = {
                "gap_id": gap_id,
                "sample_id": row["sample_id"],
                "source_case": row["source_case"],
                "anchor_count": int(row["anchor_count"]),
                "edge_i": int(row["edge_i"]),
                "label_mode": row["label_mode"],
                "pred_mode": hard_mode,
            }
            for j, mode in enumerate(CANDIDATE_MODES):
                pred_row[f"prob_{mode}"] = float(probs[i, j])
            pred_rows.append(pred_row)
    anchor_mask = out["obs_mask"].to_numpy(dtype=int) == 1
    for col in ["ATL_oracle_alt_m", "ATL_learned_hard_alt_m", "ATL_learned_soft_alt_m"]:
        out.loc[anchor_mask, col] = out.loc[anchor_mask, "adsb_alt_m"]
    return out, pd.DataFrame(pred_rows)


def _metrics_by_sample(df: pd.DataFrame, pred_col: str, model: str) -> pd.DataFrame:
    rows: list[dict] = []
    for sample_id, g in df.groupby("sample_id", sort=False):
        gap = g["obs_mask"].to_numpy(dtype=int) == 0
        truth = g.loc[gap, "adsb_alt_m"].to_numpy(dtype=float)
        pred = g.loc[gap, pred_col].to_numpy(dtype=float)
        ok = np.isfinite(truth) & np.isfinite(pred)
        if not ok.any():
            continue
        err = pred[ok] - truth[ok]
        rows.append(
            {
                "model": model,
                "sample_id": sample_id,
                "source_case": g["source_case"].iloc[0],
                "anchor_count": int(g["anchor_count"].iloc[0]),
                "point_count": int(ok.sum()),
                "alt_RMSE_m": float(np.sqrt(np.mean(np.square(err)))),
                "alt_MAE_m": float(np.mean(np.abs(err))),
                "alt_MaxAE_m": float(np.max(np.abs(err))),
            }
        )
    return pd.DataFrame(rows)


def _summarize(metrics: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    by_model.to_csv(out_dir / "atl_metrics_by_model.csv", index=False, encoding="utf-8-sig")
    by_anchor.to_csv(out_dir / "atl_metrics_by_anchor_count.csv", index=False, encoding="utf-8-sig")
    return by_model, by_anchor


def _plot_examples(df: pd.DataFrame, out_dir: Path, max_plots: int) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    # Prefer a mix of sparse and denser anchors, and larger visible differences between linear and oracle.
    score_rows = []
    for sample_id, g in df.groupby("sample_id", sort=False):
        gap = g["obs_mask"].to_numpy(dtype=int) == 0
        if not gap.any():
            continue
        truth = g.loc[gap, "adsb_alt_m"].to_numpy(dtype=float)
        linear = g.loc[gap, "分段线性插值_alt_m"].to_numpy(dtype=float)
        oracle = g.loc[gap, "ATL_oracle_alt_m"].to_numpy(dtype=float)
        if not np.isfinite(oracle).all():
            continue
        score = float(np.sqrt(np.mean(np.square(linear - truth))) - np.sqrt(np.mean(np.square(oracle - truth))))
        score_rows.append((score, sample_id))
    chosen = [sid for _, sid in sorted(score_rows, reverse=True)[: int(max_plots)]]
    for sample_id in chosen:
        g = df[df["sample_id"] == sample_id].sort_values("minute_index")
        x = g["minute_index"].to_numpy(dtype=float)
        obs = g["obs_mask"].to_numpy(dtype=int) == 1
        fig, ax = plt.subplots(figsize=(12.8, 5.4), facecolor="white")
        ax.plot(x, g["adsb_alt_m"], color="black", lw=2.2, label="ADS-B truth")
        ax.scatter(x[obs], g.loc[obs, "adsb_alt_m"], color="black", marker="*", s=130, zorder=8, label="sparse anchors")
        ax.plot(x, g["分段线性插值_alt_m"], color="#2a9d8f", lw=1.5, alpha=0.85, label="Linear")
        ax.plot(x, g["本文方案_alt_m"], color="#9a9a9a", lw=1.2, alpha=0.75, label="Ours-A3 raw")
        ax.plot(x, g["ATL_oracle_alt_m"], color="#d00000", lw=2.3, label="ATL oracle")
        ax.plot(x, g["ATL_learned_hard_alt_m"], color="#1d4ed8", lw=1.8, label="ATL learned hard")
        ax.plot(x, g["ATL_learned_soft_alt_m"], color="#f97316", lw=1.6, alpha=0.9, label="ATL learned soft")
        ax.set_title(f"{g['source_case'].iloc[0]} | anchors={int(g['anchor_count'].iloc[0])}")
        ax.set_xlabel("Cruise window minute")
        ax.set_ylabel("Altitude (m)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=4)
        fig.tight_layout()
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in sample_id)[:180]
        fig.savefig(plot_dir / f"{safe}_atl_compare.png", dpi=180)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/runs/complete_adsb_sparse_cruise_eval_20260519/sparse_cruise_model_predictions.csv")
    parser.add_argument("--out-dir", default="outputs/runs/altitude_transition_locator_20260520")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-plots", type=int, default=12)
    args = parser.parse_args()

    input_path = _resolve(args.input)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = LocatorParams()
    df = pd.read_csv(input_path)
    required = {
        "sample_id",
        "source_case",
        "anchor_count",
        "minute_index",
        "obs_mask",
        "adsb_lat",
        "adsb_lon",
        "adsb_alt_m",
        "分段线性插值_alt_m",
        "本文方案_alt_m",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    examples, payload = _build_gap_examples(df, params)
    if examples.empty:
        raise ValueError("no bounded gap examples were extracted")
    examples.to_csv(out_dir / "atl_gap_examples_with_oracle_labels.csv", index=False, encoding="utf-8-sig")
    out, pred_rows = _apply_leave_one_case_locator(df, examples, payload, epochs=args.epochs, seed=args.seed)
    out.to_csv(out_dir / "atl_predictions.csv", index=False, encoding="utf-8-sig")
    pred_rows.to_csv(out_dir / "atl_leave_one_case_predictions.csv", index=False, encoding="utf-8-sig")

    metric_parts = [
        _metrics_by_sample(out, "分段线性插值_alt_m", "Linear"),
        _metrics_by_sample(out, "本文方案_alt_m", "Ours-A3 raw"),
        _metrics_by_sample(out, "ATL_oracle_alt_m", "ATL-oracle"),
        _metrics_by_sample(out, "ATL_learned_hard_alt_m", "ATL-learned-hard"),
        _metrics_by_sample(out, "ATL_learned_soft_alt_m", "ATL-learned-soft"),
    ]
    metrics = pd.concat(metric_parts, ignore_index=True)
    metrics.to_csv(out_dir / "atl_metrics_by_sample.csv", index=False, encoding="utf-8-sig")
    by_model, by_anchor = _summarize(metrics, out_dir)
    label_dist = examples["label_mode"].value_counts().rename_axis("label_mode").reset_index(name="gap_count")
    label_dist.to_csv(out_dir / "atl_oracle_label_distribution.csv", index=False, encoding="utf-8-sig")
    confusion = pd.crosstab(pred_rows["label_mode"], pred_rows["pred_mode"])
    confusion.to_csv(out_dir / "atl_leave_one_case_confusion.csv", encoding="utf-8-sig")
    _plot_examples(out, out_dir, max_plots=args.max_plots)

    manifest = {
        "input": str(input_path),
        "out_dir": str(out_dir),
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "candidate_modes": CANDIDATE_MODES,
        "feature_cols": FEATURE_COLS,
        "leakage_guard": "Oracle labels use hidden ADS-B truth only for offline supervision; learned predictions use anchor-visible features only.",
    }
    (out_dir / "atl_experiment_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {out_dir}")
    print("\n[label distribution]")
    print(label_dist.to_string(index=False))
    print("\n[confusion]")
    print(confusion.to_string())
    print("\n[by model]")
    print(by_model.round(3).to_string(index=False))
    print("\n[by anchor_count]")
    print(by_anchor.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
