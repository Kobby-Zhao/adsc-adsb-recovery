from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.utils import load_config
from src.models import TrajectoryRecoveryModel


MODELS = [
    {
        "name": "OurMethod",
        "config": "configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_proposed_24e.yaml",
        "run_dir": "outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_proposed_24e",
    },
    {
        "name": "UniLSTM",
        "config": "configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_unilstm_baseline_24e.yaml",
        "run_dir": "outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_unilstm_baseline_24e",
    },
    {
        "name": "Kalman Filter",
        "config": "configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_kalman_filter_baseline_24e.yaml",
        "run_dir": "outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_kalman_filter_baseline_24e",
    },
    {
        "name": "BiLSTM",
        "config": "configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_bilstm_baseline_24e.yaml",
        "run_dir": "outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_bilstm_baseline_24e",
    },
    {
        "name": "CNN+LSTM",
        "config": "configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_cnnlstm_baseline_24e.yaml",
        "run_dir": "outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_cnnlstm_baseline_24e",
    },
    {
        "name": "Transformer",
        "config": "configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_transformer_baseline_24e.yaml",
        "run_dir": "outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_transformer_baseline_24e",
    },
]


def build_model(cfg: dict) -> TrajectoryRecoveryModel:
    return TrajectoryRecoveryModel(
        exo_dim=len(cfg["data"].get("exo_cols", [])),
        vertical_exo_dim=len(cfg["data"].get("vertical_exo_cols", [])),
        quality_dim=len(cfg["data"].get("quality_cols", [])),
        backbone_type=str(cfg["model"].get("backbone_type", "bilstm")),
        hidden_size=int(cfg["model"]["hidden_size"]),
        num_layers=int(cfg["model"].get("num_layers", 1)),
        dropout=float(cfg["model"].get("dropout", 0.0)),
        transformer_num_heads=int(cfg["model"].get("transformer_num_heads", 4)),
        transformer_ff_multiplier=int(cfg["model"].get("transformer_ff_multiplier", 4)),
        fusion_hidden_size=int(cfg["model"].get("fusion_hidden_size", 32)),
        fusion_use_exo_quality=bool(cfg["model"].get("fusion_use_exo_quality", False)),
        alt_bias_enabled=bool(cfg["model"].get("alt_bias_enabled", False)),
        alt_bias_hidden_size=int(cfg["model"].get("alt_bias_hidden_size", 32)),
        alt_bias_use_exo_quality=bool(cfg["model"].get("alt_bias_use_exo_quality", True)),
        vertical_projector_enabled=bool(cfg["model"].get("vertical_projector_enabled", False)),
        vertical_projector_hidden_size=int(cfg["model"].get("vertical_projector_hidden_size", 32)),
        vertical_projector_use_vertical_exo=bool(cfg["model"].get("vertical_projector_use_vertical_exo", True)),
        vertical_tune_enabled=bool(cfg["model"].get("vertical_tune_enabled", False)),
        vertical_tune_hidden_size=int(cfg["model"].get("vertical_tune_hidden_size", 16)),
        vertical_tune_temperature=float(cfg["model"].get("vertical_tune_temperature", 1.0)),
        vertical_tune_mode=str(cfg["model"].get("vertical_tune_mode", "combined")),
        model_variant=str(cfg["model"].get("model_variant", "default")),
        dms_refiner_hidden_size=int(cfg["model"].get("dms_refiner_hidden_size", 64)),
        dms_refiner_latent_dim=int(cfg["model"].get("dms_refiner_latent_dim", 32)),
        dms_refiner_num_heads=int(cfg["model"].get("dms_refiner_num_heads", 2)),
        dms_refiner_ff_multiplier=int(cfg["model"].get("dms_refiner_ff_multiplier", 2)),
        dms_refiner_dropout=float(cfg["model"].get("dms_refiner_dropout", 0.0)),
        alt_base_builder_type=str(cfg["model"].get("alt_base_builder_type", "auto")),
        alt_base_residual_hidden_size=int(cfg["model"].get("alt_base_residual_hidden_size", 64)),
        alt_base_residual_dropout=float(cfg["model"].get("alt_base_residual_dropout", 0.0)),
        alt_base_residual_bounds=cfg["model"].get("alt_base_residual_bounds"),
        alt_base_residual_bound_enabled=bool(cfg["model"].get("alt_base_residual_bound_enabled", True)),
        alt_gate_enabled=bool(cfg["model"].get("alt_gate_enabled", False)),
        alt_gate_hidden_size=int(cfg["model"].get("alt_gate_hidden_size", 32)),
        alt_gate_mode=str(cfg["model"].get("alt_gate_mode", "learned")),
        alt_gate_fixed_value=float(cfg["model"].get("alt_gate_fixed_value", 1.0)),
        alt_anchor_hard_consistency=bool(cfg["model"].get("alt_anchor_hard_consistency", False)),
        use_left_edge_directional_constraint=bool(cfg["model"].get("use_left_edge_directional_constraint", False)),
        left_edge_direction_mode=str(cfg["model"].get("left_edge_direction_mode", "anchor_based")),
        left_edge_width=int(cfg["model"].get("left_edge_width", 2)),
        left_edge_direction_strength=float(cfg["model"].get("left_edge_direction_strength", 1.0)),
        left_edge_clip_mode=str(cfg["model"].get("left_edge_clip_mode", "hard")),
        alt_main_mode=str(cfg["model"].get("alt_main_mode", "absolute")),
        main_rmax_ft=float(cfg["model"].get("main_rmax_ft", 500.0)),
        v3_anchor_hard_consistency=bool(cfg["model"].get("v3_anchor_hard_consistency", True)),
        v3_edge_residual_damp_enabled=bool(cfg["model"].get("v3_edge_residual_damp_enabled", True)),
        v3_edge_residual_damp_strength=float(cfg["model"].get("v3_edge_residual_damp_strength", 0.7)),
        v3_edge_residual_damp_steps=int(cfg["model"].get("v3_edge_residual_damp_steps", 2)),
    )


def format_int(n: int) -> str:
    return f"{n:,}"


def format_float(v: float, digits: int = 1) -> str:
    return f"{v:.{digits}f}"


def draw_table(df: pd.DataFrame, title: str, out_path: Path, col_widths: list[float], font_size: int = 11) -> None:
    rows = df.astype(str).values.tolist()
    cols = df.columns.tolist()
    fig_h = 0.65 + 0.38 * (len(rows) + 2)
    fig, ax = plt.subplots(figsize=(sum(col_widths), fig_h))
    ax.axis("off")
    ax.text(0.02, 0.97, title, transform=ax.transAxes, va="top", ha="left", fontsize=15, fontweight="bold")
    table = ax.table(
        cellText=rows,
        colLabels=cols,
        cellLoc="center",
        colLoc="center",
        bbox=[0.02, 0.05, 0.96, 0.84],
        colWidths=[w / sum(col_widths) for w in col_widths],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.visible_edges = "BT"
            cell.set_linewidth(1.2)
        else:
            cell.visible_edges = ""
        if c == 0:
            cell.get_text().set_ha("left")
    # bottom rule
    last_row = len(rows)
    for c in range(len(cols)):
        cell = table[(last_row, c)]
        cell.visible_edges = "B"
        cell.set_linewidth(1.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    out_dir = ROOT / "outputs" / "experiments" / "curriculum_20260415_exp4cmp_s2v2" / "report_tables_20260428"
    out_dir.mkdir(parents=True, exist_ok=True)

    settings_rows = []
    complexity_rows = []

    for item in MODELS:
        cfg = load_config(item["config"])
        run_dir = ROOT / item["run_dir"]
        history_path = run_dir / "history.json"
        with history_path.open("r", encoding="utf-8") as f:
            hist = json.load(f)
        val_rows = hist["val"]
        best_idx = min(range(len(val_rows)), key=lambda i: val_rows[i]["val_gap_alt_rmse"])
        best_epoch = best_idx + 1
        total_train_time = float(sum(float(r["epoch_sec"]) for r in val_rows))
        avg_epoch_time = total_train_time / len(val_rows)
        time_to_best = float(sum(float(r["epoch_sec"]) for r in val_rows[: best_idx + 1]))

        model = build_model(cfg)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        backbone = str(cfg["model"].get("backbone_type", "default"))
        variant = str(cfg["model"].get("model_variant", "default"))
        if backbone == "transformer":
            heads = int(cfg["model"].get("transformer_num_heads", 4))
        elif variant != "default":
            heads = int(cfg["model"].get("dms_refiner_num_heads", 2))
        else:
            heads = "-"

        key_modules = []
        if bool(cfg["model"].get("alt_gate_enabled", False)):
            key_modules.append("alt gate")
        if variant != "default":
            key_modules.append("alt refiner")
        if bool(cfg["training"].get("curriculum", {}).get("enabled", False)):
            key_modules.append("curriculum")
        if not key_modules:
            key_modules.append("baseline")

        settings_rows.append(
            {
                "Model": item["name"],
                "Epochs": int(cfg["training"]["epochs"]),
                "Layers": int(cfg["model"].get("num_layers", 1)),
                "Hidden Dim": int(cfg["model"]["hidden_size"]),
                "Learning Rate": cfg["training"]["lr"],
                "Backbone": backbone,
                "Heads": heads,
                "Variant": variant,
                "Key Modules": ", ".join(key_modules),
            }
        )

        complexity_rows.append(
            {
                "Model": item["name"],
                "Parameters": format_int(n_params),
                "Avg Epoch Time (s)": format_float(avg_epoch_time, 1),
                "Total Time (s)": format_float(total_train_time, 1),
                "Best Epoch": best_epoch,
                "Time to Best (s)": format_float(time_to_best, 1),
            }
        )

    settings_df = pd.DataFrame(settings_rows)
    complexity_df = pd.DataFrame(complexity_rows)

    settings_csv = out_dir / "table1_model_parameter_settings_24e.csv"
    complexity_csv = out_dir / "table3_model_complexity_training_cost_24e.csv"
    settings_df.to_csv(settings_csv, index=False)
    complexity_df.to_csv(complexity_csv, index=False)

    draw_table(
        settings_df,
        "Table 1  Model Parameter Settings (24e Curriculum Group)",
        out_dir / "table1_model_parameter_settings_24e.png",
        [2.2, 0.9, 0.8, 1.1, 1.1, 1.2, 0.7, 1.4, 2.0],
        font_size=10,
    )
    draw_table(
        complexity_df,
        "Table 3  Complexity And Training Cost Comparison (24e Group)",
        out_dir / "table3_model_complexity_training_cost_24e.png",
        [2.2, 1.6, 1.5, 1.4, 1.0, 1.6],
        font_size=10,
    )
    print(f"[ok] settings_csv={settings_csv}")
    print(f"[ok] complexity_csv={complexity_csv}")
    print(f"[ok] settings_png={out_dir / 'table1_model_parameter_settings_24e.png'}")
    print(f"[ok] complexity_png={out_dir / 'table3_model_complexity_training_cost_24e.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
