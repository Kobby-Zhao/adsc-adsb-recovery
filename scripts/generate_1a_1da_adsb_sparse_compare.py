from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.models.fusion as fusion_mod
import src.models.full_model as full_model_mod
from scripts.analyze_bidirectional_prediction_mechanism import _run_model_for_samples
from scripts.evaluate_complete_adsb_sparse_cruise import _build_sparse_cruise_dataset, _collect_predictions
from src.training.utils import load_config


class _OldSimpleFusionHead(nn.Module):
    def __init__(
        self,
        exo_dim,
        quality_dim,
        global_quality_dim,
        hidden_size=32,
        use_exo_quality=False,
        position_prior_enabled=False,
        position_prior_deviation=0.20,
        weight_mode="scalar",
    ):
        super().__init__()
        self.use_exo_quality = bool(use_exo_quality)
        in_dim = 11 + global_quality_dim + (exo_dim + quality_dim if self.use_exo_quality else 0)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )
        self.register_buffer("_eps", torch.tensor(1e-8))

    def forward(self, mu_f, mu_b, dt_prev, dt_next, obs_mask, exo, quality, global_quality):
        bsz, t_len, _ = mu_f.shape
        gq = global_quality.unsqueeze(1).expand(bsz, t_len, -1)
        gap_len = dt_prev + dt_next
        gap_pos_ratio = dt_prev / (gap_len + 1e-6)
        chunks = [
            mu_f,
            mu_b,
            dt_prev.unsqueeze(-1),
            dt_next.unsqueeze(-1),
            gap_len.unsqueeze(-1),
            gap_pos_ratio.unsqueeze(-1),
            obs_mask.unsqueeze(-1),
            gq,
        ]
        if self.use_exo_quality:
            chunks.extend([exo, quality])
        x = torch.cat(chunks, dim=-1)
        w = torch.softmax(self.mlp(x), dim=-1)
        pred = w[..., :1] * mu_f + w[..., 1:] * mu_b
        return pred, w


fusion_mod.SimpleFusionHead = _OldSimpleFusionHead
full_model_mod.SimpleFusionHead = _OldSimpleFusionHead


MODEL_SPECS = {
    "1A-gapcap": {
        "config": "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_savca_beta_v2/configs/savca_beta_1a_gapcap_evalcpu.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_savca_beta_v2/savca_beta_1a_gapcap/best.pt",
    },
    "1D-a-shape": {
        "config": "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_savca_beta_v2/configs/savca_beta_1d_a_change_score_shape_evalcpu.yaml",
        "checkpoint": "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_savca_beta_v2/savca_beta_1d_a_change_score_shape_sanity5/best.pt",
    },
}


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Generate 1A vs 1D-a ADS-B sparse cruise comparison plots.")
    p.add_argument("--input-dir", default="outputs/runs/complete_adsb_height_pattern_references_20260519_final")
    p.add_argument("--out-dir", default="outputs/runs/0525/compare_1a_1da_adsb_sparse")
    p.add_argument("--anchor-counts", default="3,8")
    p.add_argument("--max-window", type=int, default=180)
    p.add_argument("--min-alt", type=float, default=8000.0)
    p.add_argument("--device", default="cpu")
    return p


def _write_eval_configs(samples_path: Path, out_dir: Path, device: str) -> dict[str, dict[str, str]]:
    cfg_dir = out_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    specs: dict[str, dict[str, str]] = {}
    for model_name, spec in MODEL_SPECS.items():
        cfg = load_config(str(_resolve(spec["config"])))
        if model_name == "1A-gapcap":
            # This checkpoint uses the older shared bidirectional backbone
            # plus the legacy 2-logit fusion head.
            cfg.setdefault("model", {})["backbone_type"] = "legacy_bidirectional"
        cfg["data"]["samples_path"] = str(samples_path)
        cfg["data"]["split"] = {"train_ratio": 0.0, "val_ratio": 0.0, "test_ratio": 1.0}
        cfg["training"]["batch_size"] = 16
        cfg["training"]["device"] = str(device)
        out_cfg = cfg_dir / f"{model_name.replace(' ', '_').replace('/', '_')}.yaml"
        with out_cfg.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        specs[model_name] = {"config": str(out_cfg), "checkpoint": spec["checkpoint"]}
    return specs


def _plot_cases(pred_df: pd.DataFrame, out_dir: Path, anchor_counts_to_plot: set[int]) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for (source_case, anchor_count), g in pred_df.groupby(["source_case", "anchor_count"], sort=False):
        if int(anchor_count) not in anchor_counts_to_plot:
            continue
        g = g.sort_values("minute_index")
        x = g["minute_index"].to_numpy()
        obs = g["obs_mask"].to_numpy(dtype=bool)
        fig, ax = plt.subplots(figsize=(12.5, 5.2), facecolor="white")
        ax.set_facecolor("white")
        ax.plot(x, g["adsb_alt_m"], color="black", lw=2.3, label="ADS-B truth")
        ax.scatter(x[obs], g.loc[obs, "adsb_alt_m"], color="black", marker="*", s=130, zorder=6, label="ADS-C-like anchors")
        if "1A-gapcap_alt_m" in g.columns:
            ax.plot(x, g["1A-gapcap_alt_m"], lw=2.0, color="#1f77b4", linestyle="--", alpha=0.95, label="1A")
        if "1D-a-shape_alt_m" in g.columns:
            ax.plot(x, g["1D-a-shape_alt_m"], lw=2.1, color="#d62728", linestyle="-", alpha=0.92, label="1D-a")
        ax.set_title(f"{source_case} | anchor_count={anchor_count} | 1A vs 1D-a")
        ax.set_xlabel("Cruise window minute")
        ax.set_ylabel("Altitude (m)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9, ncol=3)
        fig.tight_layout()
        safe = str(source_case).replace("/", "_")
        fig.savefig(plot_dir / f"{safe}_anchor{anchor_count}_1a_vs_1da_compare.png", dpi=180)
        plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    out_dir = _resolve(args.out_dir)
    anchor_counts = [int(x) for x in args.anchor_counts.split(",") if x.strip()]
    frame = _build_sparse_cruise_dataset(_resolve(args.input_dir), out_dir, anchor_counts, args.max_window, args.min_alt)
    samples_path = out_dir / "sparse_cruise_samples.parquet"
    specs = _write_eval_configs(samples_path, out_dir, args.device)
    selected_ids = set(frame["sample_id"].astype(str).unique())

    model_results: dict[str, dict] = {}
    for model_name, spec in specs.items():
        model_results[model_name] = _run_model_for_samples(
            model_name,
            model_specs={model_name: spec},
            selected_ids=selected_ids,
            split_name="test",
            device=torch.device(args.device),
        )

    metrics, predictions = _collect_predictions(frame, model_results)
    metrics.to_csv(out_dir / "compare_1a_1da_metrics_by_sample.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(out_dir / "compare_1a_1da_predictions.csv", index=False, encoding="utf-8-sig")
    by_model = (
        metrics.groupby(["model", "anchor_count"], as_index=False)
        .agg(
            sample_count=("sample_id", "nunique"),
            alt_RMSE_m=("alt_RMSE_m", "mean"),
            alt_MAE_m=("alt_MAE_m", "mean"),
        )
        .sort_values(["anchor_count", "alt_RMSE_m", "model"])
    )
    by_model.to_csv(out_dir / "compare_1a_1da_metrics_by_anchor_count.csv", index=False, encoding="utf-8-sig")
    _plot_cases(predictions, out_dir, anchor_counts_to_plot=set(anchor_counts))
    print(f"[done] out_dir={out_dir}", flush=True)
    print(by_model.round(3).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
