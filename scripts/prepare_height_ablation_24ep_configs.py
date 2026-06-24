#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_CFG = ROOT / "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_gapaware_small.yaml"
OUT_DIR = ROOT / "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/ablation_height_24ep"


ABLATIONS = {
    "A0_shared3d_24ep": {
        "backbone_type": "bimamba_context",
        "use_z_adapter": False,
        "note": "Height ablation A0 24ep: shared 3D output head, no xy/z decoupling, no z-adapter.",
    },
    "A1_xyaux_zlinear_24ep": {
        "backbone_type": "bimamba_context_xyaux_zlinear",
        "use_z_adapter": False,
        "note": "Height ablation A1 24ep: xy auxiliary fusion + independent linear z head, no z-adapter.",
    },
    "A2_xyaux_zlinear_zadapter_24ep": {
        "backbone_type": "bimamba_context_xyaux_zlinear_zadapter",
        "use_z_adapter": True,
        "note": "Height ablation A2 24ep: xy/z decoupling + residual z-adapter without gap-aware conditions.",
    },
    "A3_gapaware_small_24ep": {
        "backbone_type": "bimamba_context_xyaux_zlinear_zadapter_gapaware_small",
        "use_z_adapter": True,
        "note": "Height ablation A3 24ep: final gap-aware small residual z-adapter.",
    },
}


def main() -> int:
    base_text = BASE_CFG.read_text(encoding="utf-8")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, spec in ABLATIONS.items():
        text = base_text
        text = text.replace(
            "backbone_type: bimamba_context_xyaux_zlinear_zadapter_gapaware_small",
            f"backbone_type: {spec['backbone_type']}",
            1,
        )
        text = text.replace("use_z_adapter: true", f"use_z_adapter: {str(bool(spec['use_z_adapter'])).lower()}", 1)
        text = text.replace(
            "  run_dir: outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_gapaware_small",
            "  run_dir: "
            + "outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/"
            + f"ablation_height_24ep/{name}",
            1,
        )
        text = text.replace(
            "experiment_note: Final candidate ACT-BiMamba gapaware_small, 24-epoch formal training.",
            f"experiment_note: {spec['note']!r}",
            1,
        )

        out_path = OUT_DIR / f"{name}.yaml"
        out_path.write_text(text, encoding="utf-8")
        print(out_path.relative_to(ROOT))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
