from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_CFG = (
    ROOT
    / "outputs/experiments/obs_conditioned_gaponly/"
    / "obscons_gaponly_physical_time_ablation_v1/configs/bilstm_clean_absolute.yaml"
)
RUN_TAG = "obscons_gaponly_bimamba_recurrent_anchorfix_v4"
OUT_DIR = ROOT / "outputs/experiments/obs_conditioned_gaponly" / RUN_TAG
OUT_CFG = OUT_DIR / "configs/bimamba_recurrent_clean_absolute.yaml"


def main() -> int:
    with BASE_CFG.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["model"]["backbone_type"] = "bimamba_recurrent"
    cfg["model"]["model_variant"] = "default"
    cfg["training"]["device"] = "cuda"
    cfg["training"]["epochs"] = 5
    if cfg["training"].get("curriculum", {}).get("enabled"):
        cfg["training"]["curriculum"]["schedule"] = [
            {"end_epoch": 2, "weights": {"stage1": 0.7, "stage2": 0.25, "stage3": 0.05}},
            {"end_epoch": 3, "weights": {"stage1": 0.2, "stage2": 0.6, "stage3": 0.2}},
            {"end_epoch": 4, "weights": {"stage1": 0.15, "stage2": 0.35, "stage3": 0.6}},
            {"end_epoch": 5, "weights": {"stage1": 0.05, "stage2": 0.2, "stage3": 0.75}},
        ]
    cfg["outputs"]["run_dir"] = str(
        Path("outputs/experiments/obs_conditioned_gaponly") / RUN_TAG / "bimamba_clean_absolute"
    )
    cfg["experiment_note"] = (
        "Anchor-conditioned BiMamba recurrent smoke run for 5 epochs. "
        "Uses seq_mask-aware directional processing, length-aware reverse, and "
        "anchor-conditioned positional inputs under mamba_ssm."
    )

    OUT_CFG.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CFG.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    print(OUT_CFG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
