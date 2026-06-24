from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.datasets.segment_risk_rules import SegmentRiskRuleMatcher


@dataclass
class DatasetConfig:
    sample_id_col: str
    flight_id_col: str
    time_col: str
    target_cols: list[str]
    obs_cols: list[str]
    obs_mask_col: str
    exo_cols: list[str]
    vertical_exo_cols: list[str] | None = None
    quality_cols: list[str] | None = None
    max_time_gap_minutes: float = 5.0
    split_on_time_gap: bool = True
    short_segment_max_minutes: int = 15
    medium_segment_max_minutes: int = 60
    segment_risk_rules_path: str | None = None
    # Failure-mode driven reweighting (training-only; harmless for eval when disabled)
    use_failure_mode_reweighting: bool = False
    reweight_target_bucket: str = "medium"
    reweight_target_anchor_pattern: str = "two_anchor"
    reweight_target_feature: str = "high_last_two_step_geom"
    reweight_weight: float = 2.5
    # Threshold for |last_two_step_geom| high regime; if None/<=0 fallback to P75 estimated in split.
    reweight_last_two_step_geom_threshold: float = 0.0


class TrajectoryDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, cfg: DatasetConfig) -> None:
        self.cfg = cfg
        self.samples: list[dict] = []
        self.segment_rule_matcher = SegmentRiskRuleMatcher.from_file(cfg.segment_risk_rules_path)
        self._geom_abs_values: list[float] = []

        for sample_id, g in frame.groupby(cfg.sample_id_col):
            g = g.sort_values(cfg.time_col).reset_index(drop=True)
            chunks = self._split_discontinuous_chunks(g)
            for seg_idx, seg in enumerate(chunks):
                if len(seg) < 2:
                    continue
                if set(cfg.target_cols).issubset(seg.columns):
                    target_pos = seg[cfg.target_cols].to_numpy(dtype=np.float32)
                else:
                    target_pos = np.zeros((len(seg), 3), dtype=np.float32)
                obs_pos = seg[cfg.obs_cols].to_numpy(dtype=np.float32)
                obs_mask = seg[cfg.obs_mask_col].to_numpy(dtype=np.float32)
                dt_prev = seg["dt_prev"].to_numpy(dtype=np.float32)
                dt_next = seg["dt_next"].to_numpy(dtype=np.float32)
                # Keep model-side input numerically safe: proxy features may carry NaN
                # for "not available" semantics; mask/freshness columns should be provided
                # alongside these values to preserve missingness meaning.
                exo = seg[cfg.exo_cols].fillna(0.0).to_numpy(dtype=np.float32)
                vcols = list(cfg.vertical_exo_cols or [])
                if vcols:
                    vertical_exo = seg[vcols].fillna(0.0).to_numpy(dtype=np.float32)
                else:
                    vertical_exo = np.zeros((len(seg), 0), dtype=np.float32)
                qcols = list(cfg.quality_cols or [])
                if qcols:
                    quality = seg[qcols].fillna(0.0).to_numpy(dtype=np.float32)
                else:
                    quality = np.zeros((len(seg), 0), dtype=np.float32)
                global_quality = np.nanmean(quality, axis=0).astype(np.float32)
                seg_meta = self._build_segment_meta(
                    obs_mask=obs_mask,
                    target_alt=target_pos[:, 2],
                    dt_prev=dt_prev,
                    dt_next=dt_next,
                    short_max=int(cfg.short_segment_max_minutes),
                    medium_max=int(cfg.medium_segment_max_minutes),
                )
                if np.isfinite(float(seg_meta.get("last_two_step_geom_abs", np.nan))):
                    self._geom_abs_values.append(float(seg_meta["last_two_step_geom_abs"]))
                left_boundary_alt, right_boundary_alt = self._extract_boundary_alt(
                    obs_pos=obs_pos,
                    obs_mask=obs_mask,
                    dt_prev=dt_prev,
                    dt_next=dt_next,
                )
                sid = str(sample_id) if len(chunks) == 1 else f"{sample_id}__seg{seg_idx}"
                self.samples.append(
                    {
                        "sample_id": sid,
                        "flight_id": str(seg[cfg.flight_id_col].iloc[0]),
                        "times": seg[cfg.time_col].astype(str).tolist(),
                        "target_pos": torch.tensor(target_pos),
                        "obs_pos": torch.tensor(obs_pos),
                        "obs_mask": torch.tensor(obs_mask),
                        "dt_prev": torch.tensor(dt_prev),
                        "dt_next": torch.tensor(dt_next),
                        "exo": torch.tensor(exo),
                        "vertical_exo": torch.tensor(vertical_exo),
                        "quality": torch.tensor(quality),
                        "global_quality": torch.tensor(global_quality),
                        # Segment-level metadata for downstream training-time policy coupling.
                        # Keep both numeric ids and readable labels; legacy training can ignore.
                        "segment_len": torch.tensor(float(seg_meta["segment_len"]), dtype=torch.float32),
                        "segment_bucket": torch.tensor(int(seg_meta["segment_bucket"]), dtype=torch.long),
                        "segment_bucket_name": str(seg_meta["segment_bucket_name"]),
                        "anchor_pattern": torch.tensor(int(seg_meta["anchor_pattern"]), dtype=torch.long),
                        "anchor_pattern_name": str(seg_meta["anchor_pattern_name"]),
                        "risk_flag": torch.tensor(float(seg_meta["risk_flag"]), dtype=torch.float32),
                        "risk_level": str(seg_meta["risk_level"]),
                        "risk_flag_teacher": torch.tensor(float(seg_meta["risk_flag_teacher"]), dtype=torch.float32),
                        "teacher_scale": torch.tensor(float(seg_meta["teacher_scale"]), dtype=torch.float32),
                        "edge_weight": torch.tensor(float(seg_meta["edge_weight"]), dtype=torch.float32),
                        "residual_rmax_m": torch.tensor(float(seg_meta["residual_rmax_m"]), dtype=torch.float32),
                        "residual_rmax_ft": torch.tensor(float(seg_meta["residual_rmax_m"]) / 0.3048, dtype=torch.float32),
                        "gate_bias": torch.tensor(float(seg_meta["gate_bias"]), dtype=torch.float32),
                        "matched_risk_rule": str(seg_meta["matched_risk_rule"]),
                        "teacher_mode": torch.tensor(int(seg_meta["teacher_mode"]), dtype=torch.long),
                        "teacher_mode_name": str(seg_meta["teacher_mode_name"]),
                        "is_edge_sensitive": torch.tensor(float(seg_meta["is_edge_sensitive"]), dtype=torch.float32),
                        "last_two_step_geom": torch.tensor(float(seg_meta["last_two_step_geom"]), dtype=torch.float32),
                        "last_two_step_geom_abs": torch.tensor(float(seg_meta["last_two_step_geom_abs"]), dtype=torch.float32),
                        "is_medium_two_anchor_high_last_two_step_geom": torch.tensor(0.0, dtype=torch.float32),
                        "sample_weight": torch.tensor(1.0, dtype=torch.float32),
                        # Boundary altitudes for boundary-conditioned main altitude experiments.
                        "left_boundary_alt": torch.tensor(float(left_boundary_alt), dtype=torch.float32),
                        "right_boundary_alt": torch.tensor(float(right_boundary_alt), dtype=torch.float32),
                    }
                )

        # Second pass: compute failure-mode hard-shape flag + sample_weight.
        if self.samples:
            if float(cfg.reweight_last_two_step_geom_threshold) > 0.0:
                geom_thr = float(cfg.reweight_last_two_step_geom_threshold)
            else:
                geom_thr = float(np.quantile(self._geom_abs_values, 0.75)) if self._geom_abs_values else 0.0
            target_bucket = str(cfg.reweight_target_bucket).lower()
            target_pattern = str(cfg.reweight_target_anchor_pattern).lower()
            rw = float(max(1.0, cfg.reweight_weight))
            enabled = bool(cfg.use_failure_mode_reweighting)
            for s in self.samples:
                sb = str(s.get("segment_bucket_name", "")).lower()
                ap = str(s.get("anchor_pattern_name", "")).lower()
                gabs = float(s.get("last_two_step_geom_abs", torch.tensor(0.0)).item())
                hard = bool((sb == target_bucket) and (ap == target_pattern) and (gabs >= geom_thr))
                s["is_medium_two_anchor_high_last_two_step_geom"] = torch.tensor(float(hard), dtype=torch.float32)
                s["sample_weight"] = torch.tensor(float(rw if (enabled and hard) else 1.0), dtype=torch.float32)
            self.reweight_geom_threshold = float(geom_thr)
        else:
            self.reweight_geom_threshold = float(cfg.reweight_last_two_step_geom_threshold)

    def _split_discontinuous_chunks(self, g: pd.DataFrame) -> list[pd.DataFrame]:
        if not self.cfg.split_on_time_gap or len(g) <= 1:
            return [g]
        ts = pd.to_datetime(g[self.cfg.time_col], utc=True, errors="coerce")
        if ts.isna().any():
            return [g]
        dt_min = ts.diff().dt.total_seconds().div(60.0).fillna(1.0)
        split_idx = np.where(dt_min.to_numpy() > float(self.cfg.max_time_gap_minutes))[0]
        if len(split_idx) == 0:
            return [g]
        chunks: list[pd.DataFrame] = []
        start = 0
        for idx in split_idx.tolist():
            chunks.append(g.iloc[start:idx].reset_index(drop=True))
            start = idx
        chunks.append(g.iloc[start:].reset_index(drop=True))
        return [c for c in chunks if len(c) > 0]

    def _build_segment_meta(
        self,
        obs_mask: np.ndarray,
        target_alt: np.ndarray,
        dt_prev: np.ndarray,
        dt_next: np.ndarray,
        short_max: int,
        medium_max: int,
    ) -> dict:
        """Build minimal fill-segment metadata without changing existing sample semantics.

        Rules (first runnable version):
        - segment_len: maximum contiguous gap length in current sample.
        - segment_bucket: short/medium/long by segment_len.
        - anchor_pattern:
          - two_anchor: dominant gap has observed points on both sides.
          - asymmetric: only one side has observed anchor/context.
          - sparse_context: no clear boundary anchors around dominant gap.
        - risk_flag:
          - short + two_anchor => 1
          - any asymmetric/sparse_context => 1
        - teacher_scale in [0,1]:
          - two_anchor: short=0.35, medium=0.65, long=1.0
          - asymmetric: 0.45
          - sparse_context: 0.20
        - teacher_mode:
          - normal / conservative / off
        - is_edge_sensitive:
          - short segments or non-two_anchor patterns.
        """
        valid = np.isfinite(obs_mask)
        obs = np.where(valid, obs_mask, 0.0).astype(np.float32)
        gap = obs < 0.5
        t_len = int(len(obs))

        # Find contiguous gap runs and select the dominant run (max length).
        runs: list[tuple[int, int]] = []
        t = 0
        while t < t_len:
            if not gap[t]:
                t += 1
                continue
            s = t
            while t < t_len and gap[t]:
                t += 1
            e = t  # [s, e)
            runs.append((s, e))

        if not runs:
            # No missing run in this sample: keep safe defaults.
            return {
                "segment_len": 0.0,
                "segment_bucket": 0,
                "segment_bucket_name": "short",
                "anchor_pattern": 0,
                "anchor_pattern_name": "two_anchor",
                "risk_flag": 0,
                "risk_level": "low",
                "risk_flag_teacher": 0,
                "teacher_scale": 1.0,
                "edge_weight": 1.0,
                "residual_rmax_m": 365.76,
                "residual_rmax_ft": 1200.0,
                "gate_bias": 0.0,
                "matched_risk_rule": "no_gap_default",
                "teacher_mode": 0,
                "teacher_mode_name": "normal",
                "is_edge_sensitive": 0,
                "last_two_step_geom": 0.0,
                "last_two_step_geom_abs": 0.0,
            }

        s, e = max(runs, key=lambda x: x[1] - x[0])
        seg_len = int(e - s)

        # Boundary anchor/context availability around dominant gap.
        has_left = bool(s - 1 >= 0 and obs[s - 1] > 0.5)
        has_right = bool(e < t_len and obs[e] > 0.5)

        # Fallback from dt signals when direct neighbors are unavailable/noisy.
        if (not has_left) and (s < len(dt_prev)):
            has_left = bool(np.nan_to_num(dt_prev[s], nan=0.0) > 0.0)
        if (not has_right) and (min(e - 1, len(dt_next) - 1) >= 0):
            has_right = bool(np.nan_to_num(dt_next[min(e - 1, len(dt_next) - 1)], nan=0.0) > 0.0)

        if has_left and has_right:
            anchor_pattern_name = "two_anchor"
            anchor_pattern = 0
        elif has_left or has_right:
            anchor_pattern_name = "asymmetric"
            anchor_pattern = 1
        else:
            anchor_pattern_name = "sparse_context"
            anchor_pattern = 2

        if seg_len <= int(short_max):
            bucket_name = "short"
            bucket = 0
        elif seg_len <= int(medium_max):
            bucket_name = "medium"
            bucket = 1
        else:
            bucket_name = "long"
            bucket = 2

        risk_flag = int((bucket_name == "short" and anchor_pattern_name == "two_anchor") or (anchor_pattern_name != "two_anchor"))

        if anchor_pattern_name == "two_anchor":
            if bucket_name == "short":
                teacher_scale = 0.35
                teacher_mode_name = "conservative"
                teacher_mode = 1
            elif bucket_name == "medium":
                teacher_scale = 0.65
                teacher_mode_name = "conservative"
                teacher_mode = 1
            else:
                teacher_scale = 1.0
                teacher_mode_name = "normal"
                teacher_mode = 0
        elif anchor_pattern_name == "asymmetric":
            teacher_scale = 0.45
            teacher_mode_name = "conservative"
            teacher_mode = 1
        else:
            teacher_scale = 0.20
            teacher_mode_name = "off"
            teacher_mode = 2

        is_edge_sensitive = int((bucket_name == "short") or (anchor_pattern_name != "two_anchor"))
        # right_step2 local geometry proxy over dominant gap region
        if int(e - s) >= 2:
            last_two_step_geom = float(target_alt[e - 1] - target_alt[e - 2])
        else:
            last_two_step_geom = 0.0
        rule_out = None
        if self.segment_rule_matcher is not None:
            rule_out = self.segment_rule_matcher.resolve(
                segment_len=float(seg_len),
                segment_bucket=bucket_name,
                anchor_pattern=anchor_pattern_name,
            )
        risk_level = str(rule_out.risk_level) if rule_out is not None else ("high" if risk_flag else "low")
        risk_flag_teacher = int(rule_out.risk_flag_teacher) if rule_out is not None else int(risk_flag)
        teacher_scale_out = float(rule_out.teacher_scale) if rule_out is not None else float(np.clip(teacher_scale, 0.0, 1.0))
        edge_weight_out = float(rule_out.edge_weight) if rule_out is not None else (3.0 if is_edge_sensitive else 1.5)
        residual_rmax_m_out = float(rule_out.residual_rmax_m) if rule_out is not None else 182.88
        gate_bias_out = float(rule_out.gate_bias) if rule_out is not None else 0.0
        matched_rule = str(rule_out.matched_rule) if rule_out is not None else "legacy_meta"
        return {
            "segment_len": float(seg_len),
            "segment_bucket": int(bucket),
            "segment_bucket_name": bucket_name,
            "anchor_pattern": int(anchor_pattern),
            "anchor_pattern_name": anchor_pattern_name,
            "risk_flag": int(risk_flag),
            "risk_level": risk_level,
            "risk_flag_teacher": int(risk_flag_teacher),
            "teacher_scale": float(np.clip(teacher_scale_out, 0.0, 1.0)),
            "edge_weight": float(edge_weight_out),
            "residual_rmax_m": float(max(1.0, residual_rmax_m_out)),
            "residual_rmax_ft": float(max(1.0, residual_rmax_m_out) / 0.3048),
            "gate_bias": float(gate_bias_out),
            "matched_risk_rule": matched_rule,
            "teacher_mode": int(teacher_mode),
            "teacher_mode_name": teacher_mode_name,
            "is_edge_sensitive": int(is_edge_sensitive),
            "last_two_step_geom": float(last_two_step_geom),
            "last_two_step_geom_abs": float(abs(last_two_step_geom)),
        }

    def _extract_boundary_alt(
        self,
        obs_pos: np.ndarray,
        obs_mask: np.ndarray,
        dt_prev: np.ndarray,
        dt_next: np.ndarray,
    ) -> tuple[float, float]:
        """Extract dominant-gap left/right boundary altitude for current sample.

        Returns robust scalar boundaries for training-time boundary-conditioned
        main altitude parameterization. Falls back to nearest observed endpoints.
        """
        obs_alt = obs_pos[:, 2].astype(np.float32)
        obs = np.where(np.isfinite(obs_mask), obs_mask, 0.0).astype(np.float32)
        gap = obs < 0.5
        t_len = int(len(obs))
        runs: list[tuple[int, int]] = []
        t = 0
        while t < t_len:
            if not gap[t]:
                t += 1
                continue
            s = t
            while t < t_len and gap[t]:
                t += 1
            runs.append((s, t))
        if not runs:
            # No gap: use first/last observed as conservative default.
            return float(obs_alt[0]), float(obs_alt[-1])

        s, e = max(runs, key=lambda x: x[1] - x[0])  # dominant gap [s, e)
        left_idx = s - 1 if s - 1 >= 0 and obs[s - 1] > 0.5 else None
        right_idx = e if e < t_len and obs[e] > 0.5 else None

        # Fallback from dt hints to nearest available observed point.
        if left_idx is None:
            left_candidates = np.where(obs[:s] > 0.5)[0]
            if len(left_candidates) > 0:
                left_idx = int(left_candidates[-1])
        if right_idx is None:
            right_candidates = np.where(obs[e:] > 0.5)[0]
            if len(right_candidates) > 0:
                right_idx = int(e + right_candidates[0])

        if left_idx is None and right_idx is None:
            return float(obs_alt[0]), float(obs_alt[-1])
        if left_idx is None:
            lv = float(obs_alt[right_idx])  # type: ignore[index]
            return lv, float(obs_alt[right_idx])  # type: ignore[index]
        if right_idx is None:
            rv = float(obs_alt[left_idx])
            return float(obs_alt[left_idx]), rv
        return float(obs_alt[left_idx]), float(obs_alt[right_idx])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]
