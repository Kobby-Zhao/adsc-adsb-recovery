"""SSVR final config — matches formal_24ep_bimamba baseline exactly, + SSVR head."""
from __future__ import annotations
from pathlib import Path
import sys, yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.training.utils import load_config

BASE = "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_bimamba.yaml"
OUT_DIR = ROOT / "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum"
CFG_DIR = OUT_DIR / "configs"
CFG_DIR.mkdir(parents=True, exist_ok=True)

cfg = load_config(str(ROOT / BASE))

# ── SSVR overrides ──
cfg["model"]["alt_main_mode"] = "absolute"
cfg["model"]["alt_anchor_reference_mode"] = "ssvr"
cfg["model"]["ssvr_hidden_size"] = 64
cfg["model"]["ssvr_rho_max"] = 0.0
cfg["model"]["ssvr_dropout"] = 0.1
cfg["model"]["main_rmax_m"] = 0.0
cfg["model"]["main_rmax_min_m"] = 0.0
cfg["model"]["main_rmax_slope_m_per_min"] = 0.0
cfg["model"]["main_rmax_max_m"] = 0.0

cfg["loss"]["lambda_ssvr_state"] = 0.3
cfg["loss"]["lambda_ssvr_smooth"] = 0.01
cfg["loss"]["ssvr_state_plateau_threshold"] = 0.15

cfg["outputs"]["run_dir"] = str(OUT_DIR / "ssvr_final")
cfg["experiment_note"] = (
    "SSVR final: bimamba_context_xyaux_zlinear + S1/S2_medium/S3 curriculum. "
    "rho_max=0, lambda_state=0.3, target_norm enabled, alpha_vertical=12."
)

out_path = CFG_DIR / "ssvr_final.yaml"
with out_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
print(f"[OK] {out_path}")
print(f"\nTrain:")
print(f"  python scripts/train.py --config {out_path}")
