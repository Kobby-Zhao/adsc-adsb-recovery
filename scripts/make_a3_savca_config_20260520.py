from __future__ import annotations

from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.utils import load_config


def main() -> int:
    src = ROOT / "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_height_ablation_gated_cpu_v1/configs/a3_gated_routed.yaml"
    out_root = ROOT / "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_a3_savca_v1"
    cfg_dir = out_root / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(src))
    cfg["experiment_note"] = (
        "A3-SAVCA: replace local-linear A1 with state-guided anchor-normalized "
        "vertical change allocation reference; keep existing gated A2/A3 residual path."
    )
    cfg["outputs"]["run_dir"] = "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_a3_savca_v1/a3_savca_routed"
    cfg["training"]["device"] = "cuda"
    cfg["model"]["alt_anchor_reference_mode"] = "savca"
    cfg["model"]["savca_hidden_size"] = 48
    cfg["model"]["savca_min_uniform"] = 0.03
    cfg["model"]["savca_state_eps"] = 0.05
    # Keep residuals conservative in the first training run so the effect of
    # the learned allocation reference remains visible.
    cfg["model"]["alt_residual_anchor_delta_gate_enabled"] = True
    cfg["model"]["alt_residual_anchor_delta_gate_low_m"] = 60.0
    cfg["model"]["alt_residual_anchor_delta_gate_high_m"] = 180.0
    cfg["model"]["alt_residual_anchor_delta_gate_min_scale"] = 0.0
    cfg["model"]["alt_residual_edge_taper_enabled"] = True
    cfg["model"]["alt_residual_edge_taper_steps"] = 3.0
    out_path = cfg_dir / "a3_savca_routed.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
