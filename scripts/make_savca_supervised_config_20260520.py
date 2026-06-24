from __future__ import annotations

import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_savca_only_v1/configs/savca_only.yaml"
RUN_TAG = os.environ.get("RUN_TAG", "obscons_gaponly_savca_supervised_v1")
SMOKE = os.environ.get("SMOKE", "0") == "1"
PROFILE = os.environ.get("SAVCA_PROFILE", "default").strip().lower()
OUT_ROOT = ROOT / "outputs/experiments/obs_conditioned_gaponly" / RUN_TAG
OUT_CFG = OUT_ROOT / ("configs/savca_supervised_smoke.yaml" if SMOKE else "configs/savca_supervised.yaml")


def main() -> int:
    with SRC.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["loss"]["lambda_savca_alloc"] = 0.05
    cfg["loss"]["lambda_savca_state"] = 0.02
    cfg["loss"]["lambda_savca_smooth"] = 0.005
    cfg["loss"]["savca_state_min_anchor_delta_m"] = 30.0
    cfg["loss"]["savca_alloc_min_anchor_delta_m"] = 100.0
    cfg["loss"]["savca_active_min_anchor_delta_m"] = 30.0
    cfg["loss"]["savca_change_deadband_m"] = 3.0
    cfg["loss"]["savca_label_median_window"] = 5
    cfg["loss"]["savca_active_ratio_to_max"] = 0.25
    cfg["loss"]["savca_active_min_abs_change_m"] = 10.0
    cfg["loss"]["savca_active_expand_steps"] = 1
    cfg["training"]["curriculum"].setdefault("savca_sampling", {})
    cfg["training"]["curriculum"]["savca_sampling"]["enabled"] = True
    cfg["training"]["curriculum"]["savca_sampling"]["state_boost"] = 1.5
    cfg["training"]["curriculum"]["savca_sampling"]["alloc_boost"] = 2.5
    cfg["training"]["curriculum"]["savca_sampling"]["long_gap_boost"] = 1.5
    cfg["training"]["curriculum"]["savca_sampling"]["long_gap_min_minutes"] = 30.0

    if PROFILE in {"strong", "strong_stable"}:
        # The default weights are intentionally conservative but their explicit
        # contribution is tiny compared with the trajectory loss.  This profile
        # makes allocation supervision comparable to the vertical reconstruction
        # term while lowering LR to reduce epoch-to-epoch allocation instability.
        cfg["loss"]["lambda_savca_alloc"] = 3.0
        cfg["loss"]["lambda_savca_state"] = 0.5
        cfg["loss"]["lambda_savca_smooth"] = 0.05
        cfg["loss"]["lambda_smooth"] = 0.2
        cfg["training"]["lr"] = 0.0003
        cfg["training"]["teacher_forcing_decay"] = 0.98
        cfg["training"]["curriculum"]["train_samples_per_epoch"] = 1600
        cfg["experiment_note"] = (
            "SAVCA supervised strong-stable: absolute-altitude allocation/state labels, "
            "stronger allocation supervision, lower LR, and larger per-epoch curriculum sample."
        )
    else:
        cfg["experiment_note"] = (
            "SAVCA supervised: SAVCA-only altitude reference with ADS-B-derived "
            "allocation/state supervision; A2/A3 disabled for isolation."
        )

    # Keep this as a SAVCA-only experiment: no A2 main offset and no A3 residual refiner.
    cfg["model"]["alt_anchor_reference_mode"] = "savca"
    cfg["model"]["model_variant"] = "default"
    cfg["model"]["main_rmax_m"] = 0.0
    cfg["model"]["main_rmax_min_m"] = 0.0
    cfg["model"]["main_rmax_slope_m_per_min"] = 0.0
    cfg["model"]["main_rmax_max_m"] = 0.0
    cfg["model"]["alt_dms_route_mode"] = "none"
    cfg["model"]["alt_gate_enabled"] = False
    cfg["training"]["risk_aware"]["use_alt_baseline_residual"] = False
    cfg["training"]["risk_aware"]["use_segment_teacher"] = False
    cfg["outputs"]["run_dir"] = str(
        Path("outputs/experiments/obs_conditioned_gaponly") / RUN_TAG / "savca_supervised"
    )
    if SMOKE:
        cfg["training"]["epochs"] = 1
        cfg["training"]["device"] = "cpu"
        cfg["training"]["curriculum"]["train_samples_per_epoch"] = 64
        cfg["training"]["step_heartbeat"]["interval"] = 1
        cfg["outputs"]["run_dir"] = str(
            Path("outputs/experiments/obs_conditioned_gaponly") / RUN_TAG / "savca_supervised_smoke"
        )
        cfg["experiment_note"] += " Smoke-test config with one epoch and 64 training samples."

    OUT_CFG.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CFG.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(OUT_CFG.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
