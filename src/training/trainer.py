from __future__ import annotations

import json
from pathlib import Path
import time

import torch

from src.training.engine import run_epoch


def _print_dim_diag(stats: dict, split: str, region: str) -> None:
    chunks = []
    for d in range(3):
        prefix = f"{region}_dim{d}"
        chunks.append(
            f"d{d}(loss={stats.get(prefix + '_loss', 0):.2f},"
            f"mae={stats.get(prefix + '_mae', 0):.2f},"
            f"rmse={stats.get(prefix + '_rmse', 0):.2f},"
            f"bias={stats.get(prefix + '_bias', 0):.2f},"
            f"pred_std={stats.get(prefix + '_pred_std', 0):.2f},"
            f"tgt_std={stats.get(prefix + '_target_std', 0):.2f},"
            f"ratio={stats.get(prefix + '_pred_over_target_std', 0):.2f})"
        )
    print(f"[diag][{split}][{region}] " + " | ".join(chunks))


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device,
        run_dir: str,
        checkpoint_name: str,
        grad_clip: float,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.run_dir / checkpoint_name
        self.grad_clip = grad_clip

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int,
        teacher_forcing_ratio: float,
        teacher_forcing_decay: float,
        coord_mode: str,
        start_epoch: int = 1,
        u_relative_anchor: bool = False,
        en_relative_anchor: bool = True,
        en_incremental: bool = False,
        long_gap_threshold: int = 20,
        checkpoint_monitor_metric: str = "gap_horizontal_rmse_m",
        target_norm_stats: dict | None = None,
        alt_target_transform_mode: str = "none",
        alt_target_clip_value: float = 3000.0,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.0,
        early_stopping_min_epochs: int = 0,
        train_loader_factory=None,
        epoch_context_factory=None,
        save_every_epoch: bool = False,
        save_epoch_interval: int = 1,
        verbose_epoch_diagnostics: bool = False,
        verbose_diag_first_epoch_only: bool = True,
        heartbeat_enabled: bool = True,
        heartbeat_interval: int = 200,
        use_segment_teacher: bool = True,
        use_alt_baseline_residual: bool = True,
        initial_history: dict | None = None,
        initial_best_val: float | None = None,
    ) -> dict:
        def _add_aliases(prefix: str, stats: dict) -> None:
            if not stats:
                return
            stats[f"{prefix}dim0_mae"] = float(stats.get("overall_dim0_mae", 0.0))
            stats[f"{prefix}dim1_mae"] = float(stats.get("overall_dim1_mae", 0.0))
            stats[f"{prefix}dim2_mae"] = float(stats.get("overall_dim2_mae", 0.0))
            stats[f"{prefix}dim0_rmse"] = float(stats.get("overall_dim0_rmse", 0.0))
            stats[f"{prefix}dim1_rmse"] = float(stats.get("overall_dim1_rmse", 0.0))
            stats[f"{prefix}dim2_rmse"] = float(stats.get("overall_dim2_rmse", 0.0))
            stats[f"{prefix}gap_dim0_mae"] = float(stats.get("gap_dim0_mae", 0.0))
            stats[f"{prefix}gap_dim1_mae"] = float(stats.get("gap_dim1_mae", 0.0))
            stats[f"{prefix}gap_dim2_mae"] = float(stats.get("gap_dim2_mae", 0.0))
            stats[f"{prefix}gap_dim0_rmse"] = float(stats.get("gap_dim0_rmse", 0.0))
            stats[f"{prefix}gap_dim1_rmse"] = float(stats.get("gap_dim1_rmse", 0.0))
            stats[f"{prefix}gap_dim2_rmse"] = float(stats.get("gap_dim2_rmse", 0.0))
            stats[f"{prefix}lat_mae"] = float(stats.get("overall_dim0_mae", 0.0))
            stats[f"{prefix}lon_mae"] = float(stats.get("overall_dim1_mae", 0.0))
            stats[f"{prefix}alt_mae"] = float(stats.get("altitude_mae", stats.get("overall_dim2_mae", 0.0)))
            stats[f"{prefix}lat_rmse"] = float(stats.get("overall_dim0_rmse", 0.0))
            stats[f"{prefix}lon_rmse"] = float(stats.get("overall_dim1_rmse", 0.0))
            stats[f"{prefix}alt_rmse"] = float(stats.get("altitude_rmse", stats.get("overall_dim2_rmse", 0.0)))
            stats[f"{prefix}altrel_mae"] = float(stats.get("altrel_mae", 0.0))
            stats[f"{prefix}altrel_rmse"] = float(stats.get("altrel_rmse", 0.0))
            stats[f"{prefix}gap_altrel_mae"] = float(stats.get("gap_altrel_mae", 0.0))
            stats[f"{prefix}gap_altrel_rmse"] = float(stats.get("gap_altrel_rmse", 0.0))
            stats[f"{prefix}altrel_pred_std"] = float(stats.get("altrel_pred_std", 0.0))
            stats[f"{prefix}altrel_true_std"] = float(stats.get("altrel_true_std", 0.0))
            stats[f"{prefix}altrel_corr"] = float(stats.get("altrel_corr", 0.0))
            stats[f"{prefix}altrel_bias_mean"] = float(stats.get("altrel_bias_mean", 0.0))
            stats[f"{prefix}anchor_altrel_mae"] = float(stats.get("anchor_altrel_mae", 0.0))
            stats[f"{prefix}anchor_altrel_rmse"] = float(stats.get("anchor_altrel_rmse", 0.0))
            stats[f"{prefix}gap_alt_mae"] = float(stats.get("gap_altitude_mae", 0.0))
            stats[f"{prefix}gap_alt_rmse"] = float(stats.get("gap_altitude_rmse", 0.0))
            stats[f"{prefix}anchor_alt_mae"] = float(stats.get("anchor_altitude_mae", 0.0))
            stats[f"{prefix}anchor_alt_rmse"] = float(stats.get("anchor_altitude_rmse", 0.0))
            stats[f"{prefix}anchor_dim2_rmse"] = float(stats.get("anchor_dim2_rmse", 0.0))
            stats[f"{prefix}altitude_mae"] = float(stats.get("altitude_mae", 0.0))
            stats[f"{prefix}altitude_rmse"] = float(stats.get("altitude_rmse", 0.0))
            stats[f"{prefix}planar_loss"] = float(stats.get("planar_loss", 0.0))
            stats[f"{prefix}step_increment_loss"] = float(stats.get("step_increment_loss", 0.0))
            stats[f"{prefix}horizontal_increment_loss"] = float(stats.get("horizontal_increment_loss", 0.0))
            stats[f"{prefix}horizontal_loss"] = float(stats.get("horizontal_loss", 0.0))
            stats[f"{prefix}vertical_loss"] = float(stats.get("vertical_loss", 0.0))
            stats[f"{prefix}vertical_smooth_loss"] = float(stats.get("vertical_smooth_loss", 0.0))
            stats[f"{prefix}alt_bias_abs_mean"] = float(stats.get("alt_bias_abs_mean", 0.0))
            stats[f"{prefix}gap_horizontal_rmse"] = float(stats.get("gap_horizontal_rmse", 0.0))
            stats[f"{prefix}anchor_horizontal_rmse"] = float(stats.get("anchor_horizontal_rmse", 0.0))
            stats[f"{prefix}long_gap_horizontal_rmse"] = float(stats.get("long_gap_horizontal_rmse", 0.0))
            stats[f"{prefix}overall_horizontal_rmse_m"] = float(stats.get("overall_horizontal_rmse_m", 0.0))
            stats[f"{prefix}gap_horizontal_rmse_m"] = float(stats.get("gap_horizontal_rmse_m", 0.0))
            stats[f"{prefix}anchor_horizontal_rmse_m"] = float(stats.get("anchor_horizontal_rmse_m", 0.0))
            stats[f"{prefix}long_gap_horizontal_rmse_m"] = float(stats.get("long_gap_horizontal_rmse_m", 0.0))
            stats[f"{prefix}fwd_gap_horizontal_rmse_m"] = float(stats.get("fwd_gap_horizontal_rmse_m", 0.0))
            stats[f"{prefix}bwd_gap_horizontal_rmse_m"] = float(stats.get("bwd_gap_horizontal_rmse_m", 0.0))
            stats[f"{prefix}fwd_long_gap_horizontal_rmse_m"] = float(stats.get("fwd_long_gap_horizontal_rmse_m", 0.0))
            stats[f"{prefix}bwd_long_gap_horizontal_rmse_m"] = float(stats.get("bwd_long_gap_horizontal_rmse_m", 0.0))
            stats[f"{prefix}anchor_obs_target_horizontal_rmse_m"] = float(
                stats.get("anchor_obs_target_horizontal_rmse_m", 0.0)
            )
            stats[f"{prefix}anchor_pred_obs_horizontal_rmse_m"] = float(
                stats.get("anchor_pred_obs_horizontal_rmse_m", 0.0)
            )
            stats[f"{prefix}dim2_pred_std"] = float(stats.get("overall_dim2_pred_std", 0.0))
            stats[f"{prefix}dim2_target_std"] = float(stats.get("overall_dim2_target_std", 0.0))
            stats[f"{prefix}dim2_pred_std_ratio"] = float(stats.get("overall_dim2_pred_over_target_std", 0.0))
            stats[f"{prefix}en_dim0_pred_std"] = float(stats.get("en_dim0_pred_std", 0.0))
            stats[f"{prefix}en_dim0_target_std"] = float(stats.get("en_dim0_target_std", 0.0))
            stats[f"{prefix}en_dim0_pred_std_ratio"] = float(stats.get("en_dim0_pred_over_target_std", 0.0))
            stats[f"{prefix}en_dim1_pred_std"] = float(stats.get("en_dim1_pred_std", 0.0))
            stats[f"{prefix}en_dim1_target_std"] = float(stats.get("en_dim1_target_std", 0.0))
            stats[f"{prefix}en_dim1_pred_std_ratio"] = float(stats.get("en_dim1_pred_over_target_std", 0.0))
            stats[f"{prefix}gap_en_dim0_pred_std_ratio"] = float(stats.get("gap_en_dim0_pred_over_target_std", 0.0))
            stats[f"{prefix}gap_en_dim1_pred_std_ratio"] = float(stats.get("gap_en_dim1_pred_over_target_std", 0.0))
            stats[f"{prefix}haversine_m"] = float(stats.get("haversine_m", 0.0))
            stats[f"{prefix}cruise_gap_points"] = float(stats.get("cruise_gap_points", 0.0))
            stats[f"{prefix}cruise_weight_mean"] = float(stats.get("cruise_weight_mean", 0.0))
            stats[f"{prefix}cruise_speed_smooth_loss"] = float(stats.get("cruise_speed_smooth_loss", 0.0))
            stats[f"{prefix}cruise_heading_rate_loss"] = float(stats.get("cruise_heading_rate_loss", 0.0))
            stats[f"{prefix}cruise_vertical_rate_loss"] = float(stats.get("cruise_vertical_rate_loss", 0.0))
            stats[f"{prefix}cruise_planar_accel_loss"] = float(stats.get("cruise_planar_accel_loss", 0.0))
            stats[f"{prefix}cruise_phys_loss"] = float(stats.get("cruise_phys_loss", 0.0))
            stats[f"{prefix}cruise_phys_over_total"] = float(stats.get("cruise_phys_over_total", 0.0))
            stats[f"{prefix}cruise_vertical_rate_over_total"] = float(stats.get("cruise_vertical_rate_over_total", 0.0))
            stats[f"{prefix}multi_scale_planar_loss"] = float(stats.get("multi_scale_planar_loss", 0.0))
            stats[f"{prefix}multi_scale_alt_loss"] = float(stats.get("multi_scale_alt_loss", 0.0))
            stats[f"{prefix}multi_scale_points"] = float(stats.get("multi_scale_points", 0.0))
            stats[f"{prefix}fusion_reg_loss"] = float(stats.get("fusion_reg_loss", 0.0))
            stats[f"{prefix}fusion_reg_over_total"] = float(stats.get("fusion_reg_over_total", 0.0))
            for k in [
                "fused_minus_fwd_mean_m",
                "fused_minus_bwd_mean_m",
                "fwd_minus_bwd_mean_m",
                "fused_minus_fwd_rmse_m",
                "fused_minus_bwd_rmse_m",
                "fwd_minus_bwd_rmse_m",
                "gap_fused_minus_fwd_mean_m",
                "gap_fused_minus_bwd_mean_m",
                "gap_fwd_minus_bwd_mean_m",
                "gap_fused_minus_fwd_rmse_m",
                "gap_fused_minus_bwd_rmse_m",
                "gap_fwd_minus_bwd_rmse_m",
                "grad_fusion_mlp_l2_total",
                "grad_fusion_mlp_l2_mean",
            ]:
                stats[f"{prefix}{k}"] = float(stats.get(k, 0.0))
            for k in [5, 10, 20]:
                stats[f"{prefix}multi_scale_k{k}_loss"] = float(stats.get(f"multi_scale_k{k}_loss", 0.0))
                stats[f"{prefix}multi_scale_k{k}_alt_loss"] = float(stats.get(f"multi_scale_k{k}_alt_loss", 0.0))
                stats[f"{prefix}multi_scale_k{k}_points"] = float(stats.get(f"multi_scale_k{k}_points", 0.0))
            for b in ["1_3", "4_8", "9_15", "16_30", "30_plus"]:
                stats[f"{prefix}gap_bucket_{b}_altrel_rmse"] = float(stats.get(f"gap_bucket_{b}_altrel_rmse", 0.0))
            stats[f"{prefix}dim_loss_contrib"] = [
                float(stats.get("pos_dim0_contrib_ratio", 0.0)),
                float(stats.get("pos_dim1_contrib_ratio", 0.0)),
                float(stats.get("pos_dim2_contrib_ratio", 0.0)),
            ]

        history = {"train": [], "val": []}
        if isinstance(initial_history, dict):
            history["train"] = list(initial_history.get("train", []))
            history["val"] = list(initial_history.get("val", []))
        best_val = float(initial_best_val) if initial_best_val is not None else float("inf")
        bad_epochs = 0

        tf_ratio = teacher_forcing_ratio
        for epoch in range(int(start_epoch), int(epochs) + 1):
            diag_verbose = bool(verbose_epoch_diagnostics) and (
                (not bool(verbose_diag_first_epoch_only)) or epoch == 1
            )
            if train_loader_factory is not None:
                train_loader = train_loader_factory(epoch)
            try:
                train_steps = int(len(train_loader))
            except Exception:
                train_steps = -1
            try:
                val_steps = int(len(val_loader))
            except Exception:
                val_steps = -1
            print(
                f"[epoch_start {epoch:03d}/{epochs:03d}] "
                f"train_steps={train_steps} val_steps={val_steps} tf_ratio={tf_ratio:.3f}",
                flush=True,
            )
            epoch_ctx = epoch_context_factory(epoch) if epoch_context_factory is not None else {}
            if hasattr(self.model, "set_runtime_savca_beta_max"):
                self.model.set_runtime_savca_beta_max(epoch_ctx.get("savca_beta_max"))
            epoch_t0 = time.perf_counter()
            train_stats = run_epoch(
                model=self.model,
                loader=train_loader,
                criterion=self.criterion,
                optimizer=self.optimizer,
                device=self.device,
                teacher_forcing_ratio=tf_ratio,
                train=True,
                grad_clip=self.grad_clip,
                coord_mode=coord_mode,
                u_relative_anchor=u_relative_anchor,
                en_relative_anchor=en_relative_anchor,
                en_incremental=en_incremental,
                long_gap_threshold=long_gap_threshold,
                target_norm_stats=target_norm_stats,
                alt_target_transform_mode=alt_target_transform_mode,
                alt_target_clip_value=alt_target_clip_value,
                enable_verbose_diag=diag_verbose,
                heartbeat_enabled=bool(heartbeat_enabled),
                heartbeat_interval=int(heartbeat_interval),
                use_segment_teacher=bool(use_segment_teacher),
                use_alt_baseline_residual=bool(use_alt_baseline_residual),
            )
            val_stats = run_epoch(
                model=self.model,
                loader=val_loader,
                criterion=self.criterion,
                optimizer=self.optimizer,
                device=self.device,
                teacher_forcing_ratio=0.0,
                train=False,
                grad_clip=self.grad_clip,
                coord_mode=coord_mode,
                u_relative_anchor=u_relative_anchor,
                en_relative_anchor=en_relative_anchor,
                en_incremental=en_incremental,
                long_gap_threshold=long_gap_threshold,
                target_norm_stats=target_norm_stats,
                alt_target_transform_mode=alt_target_transform_mode,
                alt_target_clip_value=alt_target_clip_value,
                enable_verbose_diag=diag_verbose,
                heartbeat_enabled=bool(heartbeat_enabled),
                heartbeat_interval=int(heartbeat_interval),
                use_segment_teacher=bool(use_segment_teacher),
                use_alt_baseline_residual=bool(use_alt_baseline_residual),
            )
            epoch_sec = time.perf_counter() - epoch_t0
            lr_cur = float(self.optimizer.param_groups[0].get("lr", 0.0))
            train_stats["lr"] = lr_cur
            train_stats["epoch_sec"] = epoch_sec
            val_stats["lr"] = lr_cur
            val_stats["epoch_sec"] = epoch_sec
            if epoch_ctx:
                for k, v in epoch_ctx.items():
                    train_stats[k] = v
                    val_stats[k] = v
            _add_aliases("train_", train_stats)
            _add_aliases("val_", val_stats)

            history["train"].append(train_stats)
            history["val"].append(val_stats)

            if val_stats:
                monitor_val = val_stats.get(checkpoint_monitor_metric, None)
                if monitor_val is None:
                    monitor_val = val_stats.get(f"val_{checkpoint_monitor_metric}", float("inf"))
            else:
                monitor_val = float("inf")
            if val_stats and monitor_val < (best_val - early_stopping_min_delta):
                best_val = monitor_val
                bad_epochs = 0
                torch.save({"model_state_dict": self.model.state_dict()}, self.checkpoint_path)
            else:
                bad_epochs += 1

            if bool(save_every_epoch) and int(save_epoch_interval) > 0 and (epoch % int(save_epoch_interval) == 0):
                ckpt_dir = self.run_dir / "checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"model_state_dict": self.model.state_dict(), "epoch": int(epoch)},
                    ckpt_dir / f"epoch_{epoch:03d}.pt",
                )

            print(
                f"[epoch_end {epoch:03d}/{epochs:03d}] "
                f"train_loss={train_stats.get('loss', 0):.6f} "
                f"val_loss={val_stats.get('loss', 0):.6f} "
                f"val_alt_rmse={val_stats.get('val_alt_rmse', 0):.3f} "
                f"val_gap_alt_rmse={val_stats.get('val_gap_alt_rmse', 0):.3f} "
                f"val_altrel_corr={val_stats.get('val_altrel_corr', 0):.3f} "
                f"val_pred_std={val_stats.get('val_altrel_pred_std', 0):.3f} "
                f"val_gap_horiz_rmse_m={val_stats.get('val_gap_horizontal_rmse_m', 0):.3f} "
                f"train_step_time={train_stats.get('step_time_sec', 0):.4f}s "
                f"val_step_time={val_stats.get('step_time_sec', 0):.4f}s "
                f"monitor={checkpoint_monitor_metric}:{monitor_val:.3f} "
                f"lr={lr_cur:.6g} epoch_sec={epoch_sec:.2f} tf_ratio={tf_ratio:.3f}",
                flush=True,
            )
            if diag_verbose:
                for region in ["overall", "anchor", "gap", "long_gap"]:
                    _print_dim_diag(train_stats, "train", region)
                    _print_dim_diag(val_stats, "val", region)
                contrib = []
                for d in range(3):
                    contrib.append(
                        f"d{d}(pos={val_stats.get(f'pos_dim{d}_contrib', 0):.3f},"
                        f"pos_ratio={val_stats.get(f'pos_dim{d}_contrib_ratio', 0):.3f},"
                        f"total_ratio={val_stats.get(f'total_dim{d}_contrib_ratio', 0):.3f})"
                    )
                print("[diag][val][loss_contrib] " + " | ".join(contrib))
                grad_chunks = []
                for d in range(3):
                    ksum = f"grad_sample_mu_head_dim{d}_w_l2_sum"
                    if ksum in train_stats:
                        grad_chunks.append(
                            f"d{d}(fw_w={train_stats.get(f'grad_sample_fw_mu_head_dim{d}_w_l2', 0):.3e},"
                            f"bw_w={train_stats.get(f'grad_sample_bw_mu_head_dim{d}_w_l2', 0):.3e},"
                            f"sum={train_stats.get(ksum, 0):.3e})"
                        )
                if grad_chunks:
                    print("[diag][train][grad_mu_head_sample] " + " | ".join(grad_chunks))
                for region in ["overall", "gap", "long_gap"]:
                    for d in range(3):
                        ratio_key = f"{region}_dim{d}_pred_over_target_std"
                        ratio = float(val_stats.get(ratio_key, 1.0))
                        if ratio < 0.2:
                            print(f"[warn][var_collapse] split=val region={region} dim={d} pred_std/target_std={ratio:.3f}")
                gap_buckets = ["1_3", "4_8", "9_15", "16_30", "30_plus"]
                for b in gap_buckets:
                    pref = f"gap_bucket_{b}"
                    if f"{pref}_num_segments" not in val_stats:
                        continue
                    print(
                        f"[gap_bucket][val][{b}] "
                        f"n_seg={val_stats.get(pref + '_num_segments', 0):.1f} "
                        f"n_pts={val_stats.get(pref + '_point_count', 0):.1f} "
                        f"h_rmse_m={val_stats.get(pref + '_horizontal_rmse_m', 0):.3f} "
                        f"altrel_rmse={val_stats.get(pref + '_altrel_rmse', 0):.3f} "
                        f"mean_err_m={val_stats.get(pref + '_mean_err_m', 0):.3f} "
                        f"end_err_m={val_stats.get(pref + '_end_err_m', 0):.3f} "
                        f"point_med/q90=({val_stats.get(pref + '_point_err_median_m', 0):.3f}/{val_stats.get(pref + '_point_err_q90_m', 0):.3f}) "
                        f"end_med/q90=({val_stats.get(pref + '_end_err_median_m', 0):.3f}/{val_stats.get(pref + '_end_err_q90_m', 0):.3f}) "
                        f"pred_dE(mean/std)=({val_stats.get(pref + '_pred_de_mean', 0):.3f}/{val_stats.get(pref + '_pred_de_std', 0):.3f}) "
                        f"true_dE(mean/std)=({val_stats.get(pref + '_true_de_mean', 0):.3f}/{val_stats.get(pref + '_true_de_std', 0):.3f}) "
                        f"pred_dN(mean/std)=({val_stats.get(pref + '_pred_dn_mean', 0):.3f}/{val_stats.get(pref + '_pred_dn_std', 0):.3f}) "
                        f"true_dN(mean/std)=({val_stats.get(pref + '_true_dn_mean', 0):.3f}/{val_stats.get(pref + '_true_dn_std', 0):.3f}) "
                        f"sum_pred_EN=({val_stats.get(pref + '_pred_sum_de', 0):.2f},{val_stats.get(pref + '_pred_sum_dn', 0):.2f}) "
                        f"sum_true_EN=({val_stats.get(pref + '_true_sum_de', 0):.2f},{val_stats.get(pref + '_true_sum_dn', 0):.2f}) "
                        f"wf/wb=({val_stats.get(pref + '_wf_mean', 0):.3f}/{val_stats.get(pref + '_wb_mean', 0):.3f})"
                    )
                for b in ["relpos_00_20", "relpos_20_40", "relpos_40_60", "relpos_60_80", "relpos_80_101"]:
                    if f"{b}_count" not in val_stats:
                        continue
                    print(
                        f"[fusion_relpos][val][{b}] "
                        f"count={val_stats.get(b + '_count', 0):.1f} "
                        f"wf_mean={val_stats.get(b + '_wf_mean', 0):.3f} "
                        f"wb_mean={val_stats.get(b + '_wb_mean', 0):.3f} "
                        f"fused_rmse_m={val_stats.get(b + '_fused_rmse_m', 0):.3f} "
                        f"fwd_rmse_m={val_stats.get(b + '_fwd_rmse_m', 0):.3f} "
                        f"bwd_rmse_m={val_stats.get(b + '_bwd_rmse_m', 0):.3f}"
                    )
            if (
                early_stopping_patience is not None
                and early_stopping_patience > 0
                and epoch >= max(1, int(early_stopping_min_epochs))
                and bad_epochs >= int(early_stopping_patience)
            ):
                print(
                    f"[early_stop] epoch={epoch} "
                    f"patience={early_stopping_patience} "
                    f"min_delta={early_stopping_min_delta}"
                )
                break
            tf_ratio = max(0.0, tf_ratio * teacher_forcing_decay)

        (self.run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        return history
