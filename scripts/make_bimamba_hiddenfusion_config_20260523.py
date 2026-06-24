from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_CFG = (
    ROOT
    / "outputs/experiments/obs_conditioned_gaponly/"
    / "obscons_gaponly_physical_time_ablation_v1/configs/bilstm_clean_absolute.yaml"
)
RUN_TAG = "obscons_gaponly_bimamba_hiddenfusion_anchorrelative_v3"
OUT_DIR = ROOT / "outputs/experiments/obs_conditioned_gaponly" / RUN_TAG
OUT_CFG = OUT_DIR / "configs/bimamba_hiddenfusion_clean_absolute.yaml"


def main() -> int:
    with BASE_CFG.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["model"]["backbone_type"] = "bimamba_recurrent"
    cfg["model"]["model_variant"] = "default"
    cfg["model"]["alt_main_mode"] = "anchor_relative"
    cfg["model"]["main_rmax_m"] = 152.4
    cfg["training"]["device"] = "cuda"
    cfg["training"]["epochs"] = 5
    if cfg["training"].get("curriculum", {}).get("enabled"):
        cfg["training"]["curriculum"]["schedule"] = [
            {"end_epoch": 2, "weights": {"stage1": 0.7, "stage2": 0.25, "stage3": 0.05}},
            {"end_epoch": 3, "weights": {"stage1": 0.2, "stage2": 0.6, "stage3": 0.2}},
            {"end_epoch": 4, "weights": {"stage1": 0.15, "stage2": 0.35, "stage3": 0.6}},
            {"end_epoch": 5, "weights": {"stage1": 0.05, "stage2": 0.2, "stage3": 0.75}},
        ]
    cfg["loss"]["alpha_vertical"] = 100.0
    cfg["loss"]["lambda_smooth"] = 0.0
    cfg["loss"]["lambda_alt_aux"] = 0.2
    cfg["loss"]["gap_alt_weight"] = 2.0
    cfg["outputs"]["run_dir"] = str(
        Path("outputs/experiments/obs_conditioned_gaponly") / RUN_TAG / "bimamba_hiddenfusion_anchorrelative"
    )
    cfg["experiment_note"] = (
        "BiMamba recurrent 5-epoch smoke run with seq_mask-aware processing, "
        "length-aware reverse, anchor-conditioned input, anchor-initialized directional states, "
        "hidden-state fusion before output heads, altitude-focused loss weighting, "
        "and anchor-relative main altitude prediction."
    )

    OUT_CFG.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CFG.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    print(OUT_CFG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
