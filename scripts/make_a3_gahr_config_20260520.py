from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_physical_time_ablation_v1/configs/a3_risk_routed.yaml"
OUT_DIR = ROOT / "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_a3_gahr_v1/configs"
OUT = OUT_DIR / "a3_gahr_routed.yaml"
SMOKE_OUT = OUT_DIR / "a3_gahr_routed_smoke_e1.yaml"


def main() -> int:
    with SRC.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["model"]["alt_anchor_reference_mode"] = "anchor_graph"
    cfg["outputs"]["run_dir"] = "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_a3_gahr_v1/a3_gahr_routed"
    cfg["experiment_note"] = (
        "A3-GAHR: reversible trial replacing old local-linear A1 with "
        "global anchor-graph height reference; set alt_anchor_reference_mode=local_linear to rollback."
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    smoke = yaml.safe_load(yaml.safe_dump(cfg))
    smoke["training"]["epochs"] = 1
    smoke["training"]["curriculum"]["train_samples_per_epoch"] = 64
    smoke["training"]["device"] = "cpu"
    smoke["outputs"]["run_dir"] = "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_a3_gahr_v1/a3_gahr_routed_smoke_e1"
    smoke["experiment_note"] = "Smoke test for A3-GAHR runtime validation."
    with SMOKE_OUT.open("w", encoding="utf-8") as f:
        yaml.safe_dump(smoke, f, allow_unicode=True, sort_keys=False)
    print(OUT)
    print(SMOKE_OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
