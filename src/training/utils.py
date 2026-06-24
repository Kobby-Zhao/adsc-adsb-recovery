from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.io_utils import deep_update


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(config_path: str) -> dict:
    cfg_path = Path(config_path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_ref = cfg.get("base_config")
    if not base_ref:
        return cfg

    base_path = Path(base_ref)
    if not base_path.is_absolute():
        base_path = (cfg_path.parent / base_path).resolve()
    # Recursively resolve base_config so multi-level chains work.
    base = load_config(str(base_path))

    cfg = dict(cfg)
    cfg.pop("base_config", None)
    return deep_update(base, cfg)


def split_by_flight_id(df: pd.DataFrame, flight_id_col: str, train_ratio: float, val_ratio: float, seed: int) -> dict:
    flights = sorted(df[flight_id_col].astype(str).unique().tolist())
    rng = np.random.default_rng(seed)
    rng.shuffle(flights)

    n = len(flights)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = set(flights[:n_train])
    val_ids = set(flights[n_train : n_train + n_val])
    test_ids = set(flights[n_train + n_val :])

    return {
        "train": df[df[flight_id_col].astype(str).isin(train_ids)].copy(),
        "val": df[df[flight_id_col].astype(str).isin(val_ids)].copy(),
        "test": df[df[flight_id_col].astype(str).isin(test_ids)].copy(),
    }


def validate_inference_frame(frame: pd.DataFrame, cfg: dict) -> None:
    data_cfg = cfg["data"]
    required = (
        [data_cfg["sample_id_col"], data_cfg["flight_id_col"], data_cfg["time_col"], data_cfg["obs_mask_col"]]
        + data_cfg["obs_cols"]
        + ["dt_prev", "dt_next"]
        + data_cfg["exo_cols"]
        + data_cfg.get("vertical_exo_cols", [])
        + data_cfg["quality_cols"]
    )
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise RuntimeError(f"Inference input missing required columns: {missing}")

    overlap = set(data_cfg["target_cols"]) & (
        set(data_cfg["obs_cols"])
        | set(data_cfg["exo_cols"])
        | set(data_cfg.get("vertical_exo_cols", []))
        | set(data_cfg["quality_cols"])
    )
    if overlap:
        raise RuntimeError(f"Target columns leak into model inputs: {sorted(overlap)}")
