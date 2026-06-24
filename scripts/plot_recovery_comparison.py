#!/usr/bin/env python3
"""Plot altitude recovery comparison: all 7 models vs truth on test samples."""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.utils import load_config, split_by_flight_id
from src.models import TrajectoryRecoveryModel

# ── Model definitions ──────────────────────────────────────────
MODELS = [
    ("OurMethod",   "configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_proposed_24e.yaml",
                    "outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_proposed_24e/best.pt"),
    ("BiLSTM",      "configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_bilstm_baseline_24e.yaml",
                    "outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_bilstm_baseline_24e/best.pt"),
    ("UniLSTM",     "configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_unilstm_baseline_24e.yaml",
                    "outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_unilstm_baseline_24e/best.pt"),
    ("Transformer", "configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_transformer_baseline_24e.yaml",
                    "outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_transformer_baseline_24e/best.pt"),
    ("CNN+LSTM",    "configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_cnnlstm_baseline_24e.yaml",
                    "outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_cnnlstm_baseline_24e/best.pt"),
    ("Kalman",      "configs/alt_focus/curriculum_20260415_exp4cmp_s2v2/exp_cur_kalman_filter_baseline_24e.yaml",
                    "outputs/experiments/curriculum_20260415_exp4cmp_s2v2/exp_cur_kalman_filter_baseline_24e/best.pt"),
]

COLORS = {
    "OurMethod":   "#E45756",  # red
    "BiLSTM":      "#4C78A8",  # blue
    "UniLSTM":     "#72B7B2",  # teal
    "Transformer": "#F58518",  # orange
    "CNN+LSTM":    "#B279A2",  # purple
    "LSTM+Attn":   "#54A24B",  # green
    "Kalman":      "#BAB0AC",  # gray
}


def build_model(cfg: dict) -> TrajectoryRecoveryModel:
    m = cfg["model"]
    return TrajectoryRecoveryModel(
        exo_dim=len(cfg["data"].get("exo_cols", [])),
        vertical_exo_dim=len(cfg["data"].get("vertical_exo_cols", [])),
        quality_dim=len(cfg["data"].get("quality_cols", [])),
        backbone_type=str(m.get("backbone_type", "bilstm")),
        hidden_size=int(m["hidden_size"]), num_layers=int(m.get("num_layers", 1)),
        dropout=float(m.get("dropout", 0.0)),
        transformer_num_heads=int(m.get("transformer_num_heads", 4)),
        transformer_ff_multiplier=int(m.get("transformer_ff_multiplier", 4)),
        fusion_hidden_size=int(m.get("fusion_hidden_size", 32)),
        fusion_use_exo_quality=bool(m.get("fusion_use_exo_quality", False)),
        alt_bias_enabled=bool(m.get("alt_bias_enabled", False)),
        alt_bias_hidden_size=int(m.get("alt_bias_hidden_size", 32)),
        alt_bias_use_exo_quality=bool(m.get("alt_bias_use_exo_quality", True)),
        vertical_projector_enabled=bool(m.get("vertical_projector_enabled", False)),
        vertical_projector_hidden_size=int(m.get("vertical_projector_hidden_size", 32)),
        vertical_projector_use_vertical_exo=bool(m.get("vertical_projector_use_vertical_exo", True)),
        vertical_tune_enabled=bool(m.get("vertical_tune_enabled", False)),
        vertical_tune_hidden_size=int(m.get("vertical_tune_hidden_size", 16)),
        vertical_tune_temperature=float(m.get("vertical_tune_temperature", 1.0)),
        vertical_tune_mode=str(m.get("vertical_tune_mode", "combined")),
        model_variant=str(m.get("model_variant", "default")),
        dms_refiner_hidden_size=int(m.get("dms_refiner_hidden_size", 64)),
        dms_refiner_latent_dim=int(m.get("dms_refiner_latent_dim", 32)),
        dms_refiner_num_heads=int(m.get("dms_refiner_num_heads", 2)),
        dms_refiner_ff_multiplier=int(m.get("dms_refiner_ff_multiplier", 2)),
        dms_refiner_dropout=float(m.get("dms_refiner_dropout", 0.0)),
        alt_base_builder_type=str(m.get("alt_base_builder_type", "auto")),
        alt_base_residual_hidden_size=int(m.get("alt_base_residual_hidden_size", 64)),
        alt_base_residual_dropout=float(m.get("alt_base_residual_dropout", 0.0)),
        alt_base_residual_bounds=m.get("alt_base_residual_bounds"),
        alt_base_residual_bound_enabled=bool(m.get("alt_base_residual_bound_enabled", True)),
        alt_gate_enabled=bool(m.get("alt_gate_enabled", False)),
        alt_gate_hidden_size=int(m.get("alt_gate_hidden_size", 32)),
        alt_gate_mode=str(m.get("alt_gate_mode", "learned")),
        alt_gate_fixed_value=float(m.get("alt_gate_fixed_value", 1.0)),
        alt_anchor_hard_consistency=bool(m.get("alt_anchor_hard_consistency", False)),
        use_left_edge_directional_constraint=bool(m.get("use_left_edge_directional_constraint", False)),
        left_edge_direction_mode=str(m.get("left_edge_direction_mode", "anchor_based")),
        left_edge_width=int(m.get("left_edge_width", 2)),
        left_edge_direction_strength=float(m.get("left_edge_direction_strength", 1.0)),
        left_edge_clip_mode=str(m.get("left_edge_clip_mode", "hard")),
        alt_main_mode=str(m.get("alt_main_mode", "absolute")),
        main_rmax_ft=float(m.get("main_rmax_ft", 500.0)),
        v3_anchor_hard_consistency=bool(m.get("v3_anchor_hard_consistency", True)),
        v3_edge_residual_damp_enabled=bool(m.get("v3_edge_residual_damp_enabled", True)),
        v3_edge_residual_damp_strength=float(m.get("v3_edge_residual_damp_strength", 0.7)),
        v3_edge_residual_damp_steps=int(m.get("v3_edge_residual_damp_steps", 2)),
    )


@torch.no_grad()
def infer_one(model, sdf, exo_cols):
    n = len(sdf)
    obs = np.stack([sdf[c].values for c in ["obs_lat","obs_lon","obs_alt"]], axis=-1).astype(np.float64)
    obs_t = torch.from_numpy(obs).float().unsqueeze(0)
    mask_t = torch.from_numpy(sdf['obs_mask'].values).float().unsqueeze(0)
    dp = torch.from_numpy(sdf['dt_prev'].values).float().unsqueeze(0)
    dn = torch.from_numpy(sdf['dt_next'].values).float().unsqueeze(0)
    exo = torch.from_numpy(sdf[exo_cols].values).float().unsqueeze(0)
    out = model(obs_pos=obs_t, obs_mask=mask_t, dt_prev=dp, dt_next=dn,
                exo=exo, vertical_exo=torch.zeros(1,n,len(exo_cols)),
                quality=torch.zeros(1,n,0), global_quality=torch.zeros(1,0))
    return out['pred_pos'][0,:,2].numpy()


def main():
    # Load test data
    df = pd.read_parquet('outputs/mvp_merged_nostage_20260415/stage_datasets_20260415_s2v2/stage3/samples.parquet')
    splits = split_by_flight_id(df, 'flight_id', 0.8, 0.1, 42)
    df_test = splits['test']

    # Pick representative samples: different gap lengths
    sample_gaps = {}
    for sid, sdf in df_test.groupby('sample_id'):
        mask = sdf['obs_mask'].values.astype(bool)
        anchors = np.where(mask)[0]
        gaps = [anchors[i+1]-anchors[i]-1 for i in range(len(anchors)-1) if anchors[i+1]-anchors[i]-1 > 0]
        if gaps:
            sample_gaps[sid] = max(gaps)

    # Pick samples across gap range
    sorted_sids = sorted(sample_gaps, key=sample_gaps.get)
    n_total = len(sorted_sids)
    selected = [
        sorted_sids[n_total // 3],       # short
        sorted_sids[2 * n_total // 3],   # medium
        sorted_sids[-1],                 # long
    ]
    selected = list(dict.fromkeys(selected))  # dedupe

    print(f"Selected samples: {selected}")
    for sid in selected:
        print(f"  {sid}: max_gap={sample_gaps[sid]} min")

    # Load models
    models = {}
    for name, cfg_path, ckpt_path in MODELS:
        cfg = load_config(cfg_path)
        model = build_model(cfg)
        ckpt = torch.load(str(ROOT / ckpt_path), map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()
        models[name] = (model, cfg)
        print(f"Loaded {name}")

    exo_cols = ["is_anchor","dt_prev","dt_next","gap_len","gap_pos_ratio",
                "vertical_speed","speed_delta","turn_rate"]

    # Plot
    n_plots = len(selected)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4*n_plots), squeeze=False)

    for i, sid in enumerate(selected):
        ax = axes[i][0]
        sdf = df_test[df_test['sample_id'] == sid].sort_values('minute_ts')
        times = np.arange(len(sdf))
        true_alt = sdf['alt'].values

        # Plot truth
        ax.plot(times, true_alt, 'k-', linewidth=2.5, label='ADS-B Truth', zorder=10)

        # Plot anchors
        anchor_idx = np.where(sdf['obs_mask'].values > 0.5)[0]
        ax.scatter(anchor_idx, true_alt[anchor_idx], c='black', marker='*',
                   s=120, zorder=11, label='ADS-C Anchors', edgecolors='white', linewidths=0.5)

        # Plot each model
        for name, (model, _cfg) in models.items():
            try:
                pred = infer_one(model, sdf, exo_cols)
                ax.plot(times, pred, color=COLORS[name], linewidth=1.2,
                        label=name, alpha=0.85)
            except Exception as e:
                print(f"  {name} FAILED: {e}")

        # Highlight gap regions
        mask = sdf['obs_mask'].values.astype(bool)
        in_gap = False
        gap_start = 0
        for t in range(len(mask)):
            if not mask[t] and not in_gap:
                gap_start = t
                in_gap = True
            elif mask[t] and in_gap:
                ax.axvspan(gap_start-0.5, t-0.5, alpha=0.08, color='blue')
                in_gap = False
        if in_gap:
            ax.axvspan(gap_start-0.5, len(mask)-0.5, alpha=0.08, color='blue')

        max_gap = sample_gaps[sid]
        anchors_n = len(anchor_idx)
        ax.set_title(f'{sid[:50]}...  (max_gap={max_gap}min, anchors={anchors_n})', fontsize=10)
        ax.set_ylabel('Altitude (ft)')
        ax.legend(loc='upper right', fontsize=8, ncol=4)
        ax.grid(True, alpha=0.3)

    ax.set_xlabel('Time step (minutes)')
    fig.tight_layout()
    out_path = ROOT / 'outputs/experiments/batch_eval_final/recovery_comparison.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"\n[ok] → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
