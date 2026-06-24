from __future__ import annotations

from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.utils import load_config


def main() -> int:
    src = ROOT / "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_height_ablation_gated_cpu_v1/configs/a1_linear_alt_baseline.yaml"
    out_root = ROOT / "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_savca_only_v1"
    cfg_dir = out_root / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(src))
    cfg["experiment_note"] = (
        "SAVCA-only: replace A1 local-linear baseline with SAVCA allocation reference; "
        "disable A2 main offset and A3 DMS residual."
    )
    cfg["outputs"]["run_dir"] = "outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_savca_only_v1/savca_only"
    cfg["training"]["device"] = "cuda"
    cfg["model"]["alt_main_mode"] = "anchor_relative"
    cfg["model"]["alt_anchor_reference_mode"] = "savca"
    cfg["model"]["main_rmax_m"] = 0.0
    cfg["model"]["main_rmax_min_m"] = 0.0
    cfg["model"]["main_rmax_slope_m_per_min"] = 0.0
    cfg["model"]["main_rmax_max_m"] = 0.0
    cfg["model"]["model_variant"] = "default"
    cfg["model"]["alt_dms_route_mode"] = "none"
    cfg["model"]["alt_residual_anchor_delta_gate_enabled"] = False
    cfg["model"]["alt_residual_edge_taper_enabled"] = False
    cfg["model"]["savca_hidden_size"] = 48
    cfg["model"]["savca_min_uniform"] = 0.03
    cfg["model"]["savca_state_eps"] = 0.05
    cfg["loss"]["lambda_vertical_smooth"] = 0.0
    cfg["loss"]["lambda_alt_edge_first_diff"] = 0.0
    cfg["loss"]["lambda_alt_edge_second_diff"] = 0.0
    cfg["loss"]["lambda_alt_segment_bound"] = 0.0
    cfg["loss"]["lambda_alt_vertical_rate_penalty"] = 0.0
    cfg["loss"]["lambda_alt_boundary_anchor"] = 0.0
    out_path = cfg_dir / "savca_only.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
