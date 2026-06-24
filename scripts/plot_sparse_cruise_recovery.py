"""
Inference + recovery plots: BiMamba vs BiLSTM on sparse cruise ADSB data.
Saves comparison plots to outputs/runs/0525/compare_1a_1da_adsb_sparse/bimamba_vs_bilstm/plots/
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, torch, yaml, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.models.full_model import TrajectoryRecoveryModel
from src.datasets import DatasetConfig, TrajectoryDataset, trajectory_collate_fn
from src.training.coords import (prepare_model_coordinates, build_anchor_pair_tracks,
                                   build_anchor_alt_tracks)
from src.training.target_norm import normalize_coords, load_target_stats
from torch.utils.data import DataLoader

DATA = Path("outputs/runs/0525/compare_1a_1da_adsb_sparse/sparse_cruise_samples.parquet")
OUT = Path("outputs/runs/0525/compare_1a_1da_adsb_sparse/bimamba_vs_bilstm")
PLOT_DIR = OUT / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_parquet(DATA)

# ── Load both models ────────────────────────────────────────────────────
models = {}
for mn in ["bimamba", "bilstm"]:
    cfg_path = f"outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_{mn}.yaml"
    with open(cfg_path) as f: cfg = yaml.safe_load(f)
    m = cfg["model"]
    model = TrajectoryRecoveryModel(
        exo_dim=len(cfg["data"]["exo_cols"]), quality_dim=0,
        backbone_type=m["backbone_type"], hidden_size=m.get("hidden_size",128),
        num_layers=m.get("num_layers",2), dropout=m.get("dropout",0.2),
        dms_refiner_hidden_size=m.get("dms_refiner_hidden_size",64),
        dms_refiner_latent_dim=m.get("dms_refiner_latent_dim",32),
        dms_refiner_num_heads=m.get("dms_refiner_num_heads",2),
        dms_refiner_ff_multiplier=m.get("dms_refiner_ff_multiplier",2),
        dms_refiner_dropout=m.get("dms_refiner_dropout",0.1),
        alt_target_mode=m.get("alt_target_mode","relative_to_left_anchor"),
        alt_main_mode=m.get("alt_main_mode","absolute"),
        alt_bias_enabled=m.get("alt_bias_enabled",False),
        vertical_projector_enabled=m.get("vertical_projector_enabled",False),
        vertical_exo_dim=len(cfg["data"].get("vertical_exo_cols",[])),
    )
    ckpt_path = f"outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_{mn}/best.pt"
    state = torch.load(ckpt_path, map_location=DEV, weights_only=False)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.to(DEV); model.eval()
    scaler_path = Path(cfg["outputs"]["run_dir"]) / "target_model_scaler.json"
    models[mn] = {"model": model, "scaler": load_target_stats(scaler_path)}
    print(f"Loaded {mn}")

# ── Build dataset ───────────────────────────────────────────────────────
dcfg = DatasetConfig(
    sample_id_col="sample_id", flight_id_col="flight_id", time_col="minute_ts",
    target_cols=["lat","lon","alt"], obs_cols=["obs_lat","obs_lon","obs_alt"],
    obs_mask_col="obs_mask",
    exo_cols=["is_anchor","dt_prev","dt_next","gap_len","gap_pos_ratio",
              "vertical_speed","speed_delta","turn_rate"],
    vertical_exo_cols=["is_anchor","dt_prev","dt_next","gap_len","gap_pos_ratio",
                       "vertical_speed","speed_delta","turn_rate"],
    quality_cols=[], max_time_gap_minutes=180.0, split_on_time_gap=False,
)
ds = TrajectoryDataset(df, dcfg)
loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=trajectory_collate_fn)

# ── Run inference + plot per sample ─────────────────────────────────────
colors = {"true": "black", "anchor": "lime", "bimamba": "#0072B2", "bilstm": "#D55E00"}
styles = {"bimamba": "-", "bilstm": "--"}

for batch in loader:
    B = {k: v.to(DEV) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    sid = B["sample_id"][0]
    om_cpu = B["obs_mask"].cpu(); sm_cpu = B["seq_mask"].cpu()

    # Target in model space
    tr, _, coord_ctx = prepare_model_coordinates(
        target_pos=B["target_pos"].cpu(), obs_pos=B["obs_pos"].cpu(),
        obs_mask=om_cpu, seq_mask=sm_cpu,
        mode="enu", u_relative_anchor=True, en_relative_anchor=True, en_incremental=True)
    true_alt = tr[0, :, 2].numpy()

    # True absolute altitude (for reference)
    true_abs = B["target_pos"].cpu()[0, :, 2].numpy()
    obs_abs = B["obs_pos"].cpu()[0, :, 2].numpy()
    anchor_mask = (om_cpu[0] > 0.5).numpy()

    preds = {}
    for mn in ["bimamba", "bilstm"]:
        scaler = models[mn]["scaler"]
        obs_for_model = normalize_coords(B["obs_pos"], scaler)

        anchor_left_raw, anchor_right_raw = build_anchor_pair_tracks(
            obs_pos=B["obs_pos"].cpu(), obs_mask=B["obs_mask"].cpu(),
            seq_mask=B["seq_mask"].cpu(), ctx=coord_ctx)
        anchor_left = normalize_coords(anchor_left_raw, scaler).to(DEV)
        anchor_right = normalize_coords(anchor_right_raw, scaler).to(DEV)
        anchor_alt = build_anchor_alt_tracks(
            obs_pos=B["obs_pos"].cpu(), obs_mask=B["obs_mask"].cpu(),
            seq_mask=B["seq_mask"].cpu())

        # BiLSTM may need different exo handling - use try/except with minimal args fallback
        Bsz, Tlen = B["obs_pos"].shape[:2]
        kwargs = dict(
            obs_pos=obs_for_model, obs_mask=B["obs_mask"], seq_mask=B["seq_mask"],
            dt_prev=B["dt_prev"], dt_next=B["dt_next"], exo=B["exo"],
            vertical_exo=B.get("vertical_exo", B["exo"]),
            quality=torch.zeros(Bsz, Tlen, 0, device=DEV),
            global_quality=torch.zeros(Bsz, 0, device=DEV),
            anchor_alt=anchor_alt.to(DEV),
            risk_flag=None, teacher_scale=None, risk_flag_teacher=None,
            segment_bucket=None, edge_weight=None,
            residual_rmax_m=None, residual_rmax_ft=None, gate_bias=None,
            left_boundary_alt=None, right_boundary_alt=None,
            anchor_left=anchor_left, anchor_right=anchor_right,
            target_pos=None, savca_beta_floor_mask=None, teacher_forcing_ratio=0.0,
        )

        with torch.no_grad():
            out = models[mn]["model"](**kwargs)
        preds[mn] = out["pred_pos"].cpu()[0, :, 2].numpy()

    # ── Plot ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    t = np.arange(len(true_alt))
    gap_mask_only = ~anchor_mask  # positions where model actually predicts

    for ax, title_suffix, alt_data in [
        (axes[0], "(model-relative space)", true_alt),
        (axes[1], "(absolute, meters)", true_abs)]:

        if "absolute" in title_suffix:
            # Build forward-filled anchor baseline for absolute conversion
            anchor_fill = np.zeros_like(anchor_mask, dtype=float)
            cur = 0.0
            for i in range(len(anchor_mask)):
                if anchor_mask[i]: cur = obs_abs[i]
                anchor_fill[i] = cur

            ax.plot(t, alt_data, color=colors["true"], linewidth=2.0, label="True", alpha=0.9, zorder=2)
            ax.scatter(t[anchor_mask], obs_abs[anchor_mask], color=colors["anchor"],
                      s=40, zorder=5, label="Anchor", edgecolors="darkgreen", linewidth=1)
            for mn, ls in [("bimamba", "-"), ("bilstm", "--")]:
                pred = preds[mn].copy()
                # Force anchor positions to match observed (as model does internally)
                pred[anchor_mask] = true_alt[anchor_mask]
                pred_abs = pred + anchor_fill
                # Compute gap-only RMSE (exclude anchors)
                gap_rmse = np.sqrt(np.mean((pred[gap_mask_only] - true_alt[gap_mask_only])**2))
                ax.plot(t, pred_abs, color=colors[mn], linestyle=ls, linewidth=2.0, alpha=0.85,
                       label=f"{mn.upper()} (gap_rmse={gap_rmse:.0f}m)")
        else:
            ax.plot(t, alt_data, color=colors["true"], linewidth=2.0, label="True (rel)", alpha=0.9, zorder=2)
            for mn, ls in [("bimamba", "-"), ("bilstm", "--")]:
                pred = preds[mn].copy()
                pred[anchor_mask] = alt_data[anchor_mask]
                gap_rmse = np.sqrt(np.mean((pred[gap_mask_only] - alt_data[gap_mask_only])**2))
                ax.plot(t, pred, color=colors[mn], linestyle=ls, linewidth=2.0, alpha=0.85,
                       label=f"{mn.upper()} (gap_rmse={gap_rmse:.0f}m)")

        ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
        ax.set_ylabel("Altitude")
        ax.set_title(f"{sid[:60]}  {title_suffix}", fontsize=10)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time step (minutes)")
    fig.tight_layout()

    # Safe filename
    safe_sid = sid.replace("/","_").replace(" ","_")[:80]
    fig.savefig(PLOT_DIR / f"{safe_sid}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok] {safe_sid}")

print(f"\nPlots saved to {PLOT_DIR}/")
