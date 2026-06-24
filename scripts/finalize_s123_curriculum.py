"""
Finalize S1 / S2_medium / S3 curriculum.
Produces: distribution audit, 24-epoch projection, 5-epoch + 24-epoch configs.
Does NOT train.

Usage:
  PYTHONNOUSERSITE=1 python scripts/finalize_s123_curriculum.py
"""
from __future__ import annotations

import copy, json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_curriculum_stage_distributions_20260528 import (
    _sample_stats_from_stage_frame,
    _sample_stats_from_real_adsc,
    _summarize,
)
from src.training import load_config, split_by_flight_id

# ═══════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════

S1_PATH    = Path("outputs/mvp_merged_250_20260514_clean/stage1_clean/samples.parquet")
S2_MED_PATH = Path("outputs/mvp_merged_250_20260514_clean/stage2_medium_clean/samples.parquet")
S3_PATH    = Path("outputs/mvp_merged_250_20260514_clean/stage3_clean/samples.parquet")
REAL_FEAT  = Path("outputs/runs/cross_ocean_adsc_anchor_routes_20260517/adsc_quartile_stats/cross_ocean_adsc_flight_level_features.csv")
REAL_RAW   = Path("outputs/mvp_global_202410_202503_full1000_20260414/adsc_parsed.parquet")
BASE_CFG   = "outputs/experiments/obs_conditioned_gaponly/bimamba_xyaux_zlinear_24e_v1/configs/C_bimamba_context_xyaux_zlinear.yaml"

OUT_ROOT = Path("outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum")
CFG_DIR  = OUT_ROOT / "configs"
AUDIT_DIR = OUT_ROOT / "audit"
for d in [OUT_ROOT, CFG_DIR, AUDIT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LG, FA = 20, 6
SEED = 42
TRAIN_SAMPLES_PER_EPOCH = 800

# ═══════════════════════════════════════════════════════════════════════════
# 1. Load data + split
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("1. LOADING DATA")
print("=" * 80)

s1_df = pd.read_parquet(S1_PATH)
s2m_df = pd.read_parquet(S2_MED_PATH)
s3_df = pd.read_parquet(S3_PATH)

s1_s = _sample_stats_from_stage_frame(s1_df)
s2m_s = _sample_stats_from_stage_frame(s2m_df)
s3_s = _sample_stats_from_stage_frame(s3_df)

base_cfg = load_config(BASE_CFG)
splits = split_by_flight_id(
    df=s3_df, flight_id_col=base_cfg["data"]["flight_id_col"],
    train_ratio=float(base_cfg["data"]["split"]["train_ratio"]),
    val_ratio=float(base_cfg["data"]["split"]["val_ratio"]),
    seed=SEED,
)
train_ids = set(splits["train"][base_cfg["data"]["flight_id_col"]].astype(str).unique())
val_ids   = set(splits["val"][base_cfg["data"]["flight_id_col"]].astype(str).unique())
test_ids  = set(splits["test"][base_cfg["data"]["flight_id_col"]].astype(str).unique())
print(f"Split: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} flights")

# Test set from S3 split
test_s = s3_s[s3_s["flight_id"].isin(test_ids)].copy()

# Real ADS-C
real_s = _sample_stats_from_real_adsc(REAL_FEAT, REAL_RAW)

# ═══════════════════════════════════════════════════════════════════════════
# 2. Full distribution audit
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("2. DISTRIBUTION AUDIT")
print("=" * 80)

all_datasets = [
    ("S1", s1_s),
    ("S2_medium", s2m_s),
    ("S3", s3_s),
    ("test", test_s),
    ("real_ADSC", real_s),
]

# Build full summary table
summary_rows = []
for name, sdf in all_datasets:
    r = _summarize(sdf, name, LG, FA)
    summary_rows.append(r)
summary_df = pd.DataFrame(summary_rows)

# Build comparison table (S3 vs test vs real)
compare_rows = []
for name in ["S3", "test", "real_ADSC"]:
    sdf = dict(all_datasets)[name]
    r = _summarize(sdf, name, LG, FA)
    compare_rows.append(r)
compare_df = pd.DataFrame(compare_rows)

# Save
summary_df.to_csv(AUDIT_DIR / "final_s123_distribution_audit.csv", index=False)
compare_df.to_csv(AUDIT_DIR / "final_s123_vs_test_real_adsc.csv", index=False)

# Print full tables
def _fmt(v, w=8, d=3):
    if isinstance(v, float):
        return f"{v:{w}.{d}f}"
    return f"{str(v):>{w}s}"

print("\n--- Full Distribution Table ---")
print(summary_df[["dataset","n_samples",
    "anchor_count_mean","anchor_count_p25","anchor_count_p50","anchor_count_p75","anchor_count_p90",
    "missing_ratio_mean","missing_ratio_p25","missing_ratio_p50","missing_ratio_p75","missing_ratio_p90",
    "gap_len_max_mean","gap_len_max_p50","gap_len_max_p75","gap_len_max_p90",
    "delta_z_gt100_ratio","delta_z_gt300_ratio","delta_z_gt600_ratio",
    "long_gap_ratio","few_anchor_ratio",
    "long_gap_large_dz300_ratio","few_anchor_large_dz300_ratio"]].to_string(index=False))

print("\n--- Key Progression Check ---")
for name in ["S1", "S2_medium", "S3", "test", "real_ADSC"]:
    sub = summary_df[summary_df["dataset"] == name].iloc[0]
    print(f"\n  {name} (n={int(sub['n_samples'])}):")
    print(f"    anchor:  m={sub['anchor_count_mean']:.1f}  p25={sub['anchor_count_p25']:.0f}  p50={sub['anchor_count_p50']:.0f}  p75={sub['anchor_count_p75']:.0f}  p90={sub['anchor_count_p90']:.0f}")
    print(f"    missing: m={sub['missing_ratio_mean']:.3f}  p25={sub['missing_ratio_p25']:.3f}  p50={sub['missing_ratio_p50']:.3f}  p75={sub['missing_ratio_p75']:.3f}  p90={sub['missing_ratio_p90']:.3f}")
    print(f"    gap_max: m={sub['gap_len_max_mean']:.1f}  p50={sub['gap_len_max_p50']:.0f}  p75={sub['gap_len_max_p75']:.0f}  p90={sub['gap_len_max_p90']:.0f}")
    print(f"    dz:      m={sub['delta_z_abs_mean']:.0f}  p25={sub['delta_z_abs_p25']:.0f}  p50={sub['delta_z_abs_p50']:.0f}  p75={sub['delta_z_abs_p75']:.0f}  p90={sub['delta_z_abs_p90']:.0f}")
    print(f"    dz>100={sub['delta_z_gt100_ratio']:.3f}  dz>300={sub['delta_z_gt300_ratio']:.3f}  dz>600={sub['delta_z_gt600_ratio']:.3f}")
    print(f"    long_gap={sub['long_gap_ratio']:.3f}  few_anchor={sub['few_anchor_ratio']:.3f}")
    print(f"    lg+dz300={sub['long_gap_large_dz300_ratio']:.3f}  fa+dz300={sub['few_anchor_large_dz300_ratio']:.3f}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. 24-epoch curriculum projection
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("3. 24-EPOCH CURRICULUM PROJECTION")
print("=" * 80)

# Train-only stats
s1_train_s = s1_s[s1_s["flight_id"].isin(train_ids)].copy()
s2m_train_s = s2m_s[s2m_s["flight_id"].isin(train_ids)].copy()
s3_train_s = s3_s[s3_s["flight_id"].isin(train_ids)].copy()
print(f"Train pools: S1={len(s1_train_s)} S2_medium={len(s2m_train_s)} S3={len(s3_train_s)}")

pools = {"S1": s1_train_s, "S2": s2m_train_s, "S3": s3_train_s}

schedule_24e = [
    ("epoch1-4",   1,  4, {"S1": 0.60, "S2": 0.30, "S3": 0.10}),
    ("epoch5-10",  5, 10, {"S1": 0.30, "S2": 0.45, "S3": 0.25}),
    ("epoch11-18", 11, 18, {"S1": 0.15, "S2": 0.35, "S3": 0.50}),
    ("epoch19-24", 19, 24, {"S1": 0.05, "S2": 0.25, "S3": 0.70}),
]

def _sample_pool(df, n, rng):
    if n <= 0 or df.empty:
        return df.iloc[0:0].copy()
    idx = rng.choice(len(df), size=min(n, len(df)), replace=(n > len(df)))
    return df.iloc[idx].copy()

proj_rows = []
for period_name, ep_s, ep_e, weights in schedule_24e:
    period_pieces = []
    for epoch in range(ep_s, ep_e + 1):
        rng = np.random.default_rng(SEED + 1009 * epoch)
        pieces = []
        counts = {k: int(np.floor(v * TRAIN_SAMPLES_PER_EPOCH)) for k, v in weights.items()}
        while sum(counts.values()) < TRAIN_SAMPLES_PER_EPOCH:
            k = max(weights, key=lambda x: weights[x] - counts.get(x, 0) / max(1, TRAIN_SAMPLES_PER_EPOCH))
            counts[k] = counts.get(k, 0) + 1
        for pool_name, cnt in counts.items():
            part = _sample_pool(pools[pool_name], cnt, rng)
            if not part.empty:
                part = part.copy()
                part["pool"] = pool_name
                pieces.append(part)
        if pieces:
            period_pieces.append(pd.concat(pieces, ignore_index=True))
    if period_pieces:
        period_df = pd.concat(period_pieces, ignore_index=True)
        row = _summarize(period_df, period_name, LG, FA)
        row["epoch_range"] = f"{ep_s}-{ep_e}"
        for k, v in weights.items():
            row[f"{k}_weight"] = float(v)
        proj_rows.append(row)

proj_df = pd.DataFrame(proj_rows)
proj_cols = ["dataset","epoch_range","n_samples",
    "anchor_count_mean","anchor_count_p50",
    "missing_ratio_mean","missing_ratio_p50",
    "gap_len_max_p90",
    "long_gap_ratio","few_anchor_ratio",
    "delta_z_gt300_ratio","long_gap_large_dz300_ratio",
    "S1_weight","S2_weight","S3_weight"]
print("\n" + proj_df[proj_cols].to_string(index=False))
proj_df.to_csv(AUDIT_DIR / "final_24epoch_curriculum_projected_distribution.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════════
# 4. Generate configs
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("4. GENERATING CONFIGS")
print("=" * 80)

# Build train-only parquet pools
s1_train_df = s1_df[s1_df["flight_id"].astype(str).isin(train_ids)].copy()
s2m_train_df = s2m_df[s2m_df["flight_id"].astype(str).isin(train_ids)].copy()
s3_train_df = s3_df[s3_df["flight_id"].astype(str).isin(train_ids)].copy()

POOL_DIR = OUT_ROOT / "pools"
POOL_DIR.mkdir(exist_ok=True)
s1_train_df.to_parquet(POOL_DIR / "S1_train.parquet", index=False)
s2m_train_df.to_parquet(POOL_DIR / "S2_medium_train.parquet", index=False)
s3_train_df.to_parquet(POOL_DIR / "S3_train.parquet", index=False)

with open(BASE_CFG) as f:
    base = yaml.safe_load(f)

def make_config(name, epochs, schedule):
    cfg = copy.deepcopy(base)
    cfg["outputs"]["run_dir"] = str(OUT_ROOT / name)
    cfg["training"]["epochs"] = epochs
    cfg["training"]["checkpoint_monitor_metric"] = "gap_alt_rmse"
    cfg["loss"]["alpha_vertical"] = 12.0
    cfg["training"]["curriculum"]["stage_paths"] = {
        "stage1": str(POOL_DIR / "S1_train.parquet"),
        "stage2": str(POOL_DIR / "S2_medium_train.parquet"),
        "stage3": str(POOL_DIR / "S3_train.parquet"),
    }
    cfg["training"]["curriculum"]["schedule"] = schedule
    cfg["experiment_note"] = (
        f"Final S1/S2_medium/S3 curriculum. "
        f"S2_medium: mask_ratios=[0.60,0.72,0.85] gap_buckets=[(25,35),(35,48),(48,62)]. "
        f"alpha_vertical=12."
    )
    return cfg

# 5-epoch sanity schedule
sched_5e = [
    {"end_epoch": 4, "weights": {"stage1": 0.60, "stage2": 0.30, "stage3": 0.10}},
    {"end_epoch": 5, "weights": {"stage1": 0.30, "stage2": 0.45, "stage3": 0.25}},
]

# 24-epoch formal schedule
sched_24e = [
    {"end_epoch": 4,  "weights": {"stage1": 0.60, "stage2": 0.30, "stage3": 0.10}},
    {"end_epoch": 10, "weights": {"stage1": 0.30, "stage2": 0.45, "stage3": 0.25}},
    {"end_epoch": 18, "weights": {"stage1": 0.15, "stage2": 0.35, "stage3": 0.50}},
    {"end_epoch": 24, "weights": {"stage1": 0.05, "stage2": 0.25, "stage3": 0.70}},
]

cfgs = {
    "sanity_5ep":  make_config("sanity_5ep",  5,  sched_5e),
    "formal_24ep": make_config("formal_24ep", 24, sched_24e),
}

for name, cfg in cfgs.items():
    path = CFG_DIR / f"{name}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    print(f"  {path}")

# ═══════════════════════════════════════════════════════════════════════════
# 5. Verify configs
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("5. CONFIG VERIFICATION")
print("=" * 80)

for name in ["sanity_5ep", "formal_24ep"]:
    with open(CFG_DIR / f"{name}.yaml") as f:
        cfg = yaml.safe_load(f)
    sp = cfg["training"]["curriculum"]["stage_paths"]
    sch = cfg["training"]["curriculum"]["schedule"]
    assert cfg["model"]["backbone_type"] == "bimamba_context_xyaux_zlinear"
    assert cfg["loss"]["alpha_vertical"] == 12.0
    assert cfg["training"]["batch_size"] == 32
    assert cfg["training"]["curriculum"]["train_samples_per_epoch"] == 800
    print(f"\n  {name}: epochs={cfg['training']['epochs']} alpha_vert={cfg['loss']['alpha_vertical']}")
    print(f"    backbone={cfg['model']['backbone_type']}")
    print(f"    batch_size={cfg['training']['batch_size']} samples_per_epoch={cfg['training']['curriculum']['train_samples_per_epoch']}")
    for s in sch:
        print(f"    end_epoch={s['end_epoch']:>2d}: {s['weights']}")
    # Check loss params unchanged
    assert cfg["loss"]["anchor_weight"] == 0.0
    assert cfg["loss"]["gap_weight"] == 1.0
    assert cfg["loss"]["gap_alt_weight"] == 1.5
    print(f"    loss: anchor_weight={cfg['loss']['anchor_weight']} gap_weight={cfg['loss']['gap_weight']} gap_alt_weight={cfg['loss']['gap_alt_weight']}")
    # Check target_norm unchanged
    assert cfg["training"]["target_norm"]["enabled"] == True
    assert cfg["training"]["target_norm"]["center_per_dim"] == [True, True, False]
    print(f"    target_norm: enabled center_per_dim={cfg['training']['target_norm']['center_per_dim']}")

# ═══════════════════════════════════════════════════════════════════════════
# 6. Print training commands
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("6. TRAINING COMMANDS")
print("=" * 80)

print("""
# ── 5-Epoch Sanity ─────────────────────────────────────────────────────
PYTHONNOUSERSITE=1 python scripts/train.py \\
  --config outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/sanity_5ep.yaml \\
  2>&1 | tee outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/sanity_5ep/train_console.log

# ── Evaluate Sanity ─────────────────────────────────────────────────────
# val
PYTHONNOUSERSITE=1 python scripts/evaluate.py \\
  --config outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/sanity_5ep.yaml \\
  --checkpoint outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/sanity_5ep/best.pt \\
  --split val --plot-count 0

# test
PYTHONNOUSERSITE=1 python scripts/evaluate.py \\
  --config outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/sanity_5ep.yaml \\
  --checkpoint outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/sanity_5ep/best.pt \\
  --split test --plot-count 0

# ── 24-Epoch Formal (only if sanity passes) ──────────────────────────────
PYTHONNOUSERSITE=1 python scripts/train.py \\
  --config outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep.yaml \\
  2>&1 | tee outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep/train_console.log
""")

# ═══════════════════════════════════════════════════════════════════════════
# 7. Judgment criteria summary
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("7. JUDGMENT CRITERIA (5-epoch sanity)")
print("=" * 80)
print("""
  RETAIN if ALL of:
    [ ] overall gap_alt_rmse not significantly worse
    [ ] hard-bucket gap_alt_rmse not degraded (ideally improved)
    [ ] lat/lon RMSE degradation <= 5%
    [ ] val_altrel_corr not degraded
    [ ] val_altrel_pred_std not collapsed
    [ ] sampled distribution matches design weights

  If sanity passes → run formal_24ep.
  If sanity fails  → report which metric failed and investigate.
""")

print(f"\nAll outputs in: {OUT_ROOT}/")
print(f"  Audit:     {AUDIT_DIR}/")
print(f"  Configs:   {CFG_DIR}/")
print(f"  Pools:     {POOL_DIR}/")
print("Done.")
