#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.benchmark_efficiency_24e import (
    FINAL_CFG_ROOT,
    MODEL_SPECS,
    _build_model,
    _history_stats,
    _interp_linear_gapwise,
    _make_dataset,
    _prepare_forward_inputs,
    _run_neural_once,
    _sample_to_frame,
    _smooth_sample,
)
from src.datasets import trajectory_collate_fn
from src.training.utils import load_config, set_seed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified end-to-end single-sample latency benchmark for final 24-epoch models."
    )
    p.add_argument(
        "--samples",
        default="outputs/mvp_merged_250_20260514_clean/stage3_clean/samples.parquet",
        help="Benchmark split source parquet; defaults to final S3 clean split.",
    )
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--max-samples", type=int, default=64)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--out-csv",
        default="outputs/analysis/efficiency_benchmark_24e_unified_e2e.csv",
    )
    return p


def _selected_samples(cfg: dict, split: str, max_samples: int) -> tuple[list[dict], dict | None]:
    ds, target_norm_stats = _make_dataset(cfg, split)
    samples = [s for s in ds.samples if float(s["obs_mask"].sum().item()) > 0.5][:max_samples]
    if not samples:
        raise RuntimeError("No anchor-valid samples found for benchmark.")
    return samples, target_norm_stats


def _benchmark_neural_e2e(spec: dict, args: argparse.Namespace, raw_cfg: dict) -> dict:
    device = torch.device(args.device)
    cfg = dict(raw_cfg)
    cfg["data"]["samples_path"] = str(Path(args.samples))
    samples, target_norm_stats = _selected_samples(cfg, args.split, args.max_samples)

    model = _build_model(cfg, Path(spec["checkpoint"]), device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    warmup = min(args.warmup, len(samples))
    with torch.no_grad():
        for i in range(warmup):
            batch = trajectory_collate_fn([samples[i]])
            prepared = _prepare_forward_inputs(batch, cfg, device, target_norm_stats)
            _ = _run_neural_once(model, prepared)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        elapsed = 0.0
        measured = 0
        for sample in samples:
            for _ in range(args.repeat):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                batch = trajectory_collate_fn([sample])
                prepared = _prepare_forward_inputs(batch, cfg, device, target_norm_stats)
                _ = _run_neural_once(model, prepared)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed += time.perf_counter() - t0
                measured += 1

    peak_mb = float("nan")
    if device.type == "cuda":
        peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))

    row = {
        "model": spec["model"],
        "params": total_params,
        "trainable_params": trainable_params,
        "params_m": total_params / 1e6,
        "unified_e2e_ms_per_sample": (elapsed / max(measured, 1)) * 1000.0,
        "peak_gpu_mem_mb": peak_mb,
        "epochs": np.nan,
        "train_total_sec": np.nan,
        "train_total_min": np.nan,
        "train_avg_epoch_sec": np.nan,
        "device": str(device),
        "benchmark_scope": "single_sample_end_to_end",
        "max_samples": int(args.max_samples),
        "warmup": int(args.warmup),
        "repeat": int(args.repeat),
        "batch_size": 1,
        "notes": "Neural timing includes collate, tensor transfer/preparation, forward, denorm and restore.",
    }
    row.update(_history_stats(Path(spec["history"])))
    return row


def _benchmark_traditional_e2e(kind: str, args: argparse.Namespace, cfg: dict) -> dict:
    cfg = dict(cfg)
    cfg["data"]["samples_path"] = str(Path(args.samples))
    samples, _ = _selected_samples(cfg, args.split, args.max_samples)
    fn = _interp_linear_gapwise if kind == "piecewise_linear" else _smooth_sample
    name = "分段线性插值" if kind == "piecewise_linear" else "Kalman Filter"

    warmup = min(args.warmup, len(samples))
    for i in range(warmup):
        _ = fn(_sample_to_frame(samples[i]))

    elapsed = 0.0
    measured = 0
    for sample in samples:
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            sdf = _sample_to_frame(sample)
            _ = fn(sdf)
            elapsed += time.perf_counter() - t0
            measured += 1

    return {
        "model": name,
        "params": np.nan,
        "trainable_params": np.nan,
        "params_m": np.nan,
        "unified_e2e_ms_per_sample": (elapsed / max(measured, 1)) * 1000.0,
        "peak_gpu_mem_mb": np.nan,
        "epochs": 0,
        "train_total_sec": np.nan,
        "train_total_min": np.nan,
        "train_avg_epoch_sec": np.nan,
        "device": "cpu",
        "benchmark_scope": "single_sample_end_to_end",
        "max_samples": int(args.max_samples),
        "warmup": int(args.warmup),
        "repeat": int(args.repeat),
        "batch_size": 1,
        "notes": "Traditional timing includes sample DataFrame construction and algorithm execution.",
    }


def main() -> int:
    args = build_parser().parse_args()
    set_seed(42)
    base_cfg = load_config(str(FINAL_CFG_ROOT / "formal_24ep_gapaware_small.yaml"))

    rows = []
    for spec in MODEL_SPECS:
        print(f"[benchmark-e2e] {spec['model']}")
        if spec["kind"] == "piecewise_linear":
            rows.append(_benchmark_traditional_e2e("piecewise_linear", args, base_cfg))
            continue
        if spec["kind"] == "kalman":
            rows.append(_benchmark_traditional_e2e("kalman", args, base_cfg))
            continue
        cfg = load_config(str(spec["config"]))
        rows.append(_benchmark_neural_e2e(spec, args, cfg))

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[done] saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
