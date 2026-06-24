from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SegmentRiskRuleResult:
    risk_level: str
    risk_flag_teacher: int
    teacher_scale: float
    edge_weight: float
    residual_rmax_m: float
    gate_bias: float
    matched_rule: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_level": str(self.risk_level),
            "risk_flag_teacher": int(self.risk_flag_teacher),
            "teacher_scale": float(self.teacher_scale),
            "edge_weight": float(self.edge_weight),
            "residual_rmax_m": float(self.residual_rmax_m),
            "residual_rmax_ft": float(self.residual_rmax_m / 0.3048),
            "gate_bias": float(self.gate_bias),
            "matched_rule": str(self.matched_rule),
        }


class SegmentRiskRuleMatcher:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg or {}
        self.defaults = dict(self.cfg.get("defaults", {}))
        self.priority_order = list(self.cfg.get("priority_order", [])) or [
            "high_risk_rules",
            "medium_risk_rules",
            "low_risk_rules",
        ]
        self.fallbacks = dict(self.cfg.get("fallbacks", {}))

    @staticmethod
    def _bucket_name(seg_len: float, short_max: int, medium_max: int) -> str:
        if seg_len <= float(short_max):
            return "short"
        if seg_len <= float(medium_max):
            return "medium"
        return "long"

    @staticmethod
    def _match_list(rule_vals: list[Any], target: str | None) -> bool:
        vals = {str(v) for v in (rule_vals or [])}
        if "*" in vals or "any" in vals:
            return True
        if target is None:
            return False
        return str(target) in vals

    @staticmethod
    def _residual_rmax_m(assign: dict[str, Any], default_m: float = 182.88) -> float:
        if "residual_rmax_m" in assign:
            return float(assign.get("residual_rmax_m", default_m))
        if "residual_rmax_ft" in assign:
            return float(assign.get("residual_rmax_ft", default_m / 0.3048)) * 0.3048
        return float(default_m)

    def _match_rule(self, rule: dict[str, Any], segment_bucket: str, anchor_pattern: str | None) -> bool:
        cond = dict(rule.get("when", {}))
        if "segment_bucket" in cond:
            if not self._match_list(list(cond.get("segment_bucket", [])), segment_bucket):
                return False
        if "anchor_pattern" in cond:
            if anchor_pattern is None and bool(
                self.fallbacks.get("missing_anchor_pattern", {}).get("use_segment_bucket_only", False)
            ):
                return True
            if not self._match_list(list(cond.get("anchor_pattern", [])), anchor_pattern):
                return False
        return True

    def resolve(self, *, segment_len: float, segment_bucket: str | None, anchor_pattern: str | None) -> SegmentRiskRuleResult:
        sb = str(segment_bucket) if segment_bucket else None
        if sb not in {"short", "medium", "long"}:
            if bool(self.fallbacks.get("unknown_segment_bucket", {}).get("assign_default_medium", True)):
                sb = "medium"
            else:
                sb = "short"

        best_assign: dict[str, Any] = dict(self.defaults)
        matched = "defaults"
        for group_name in self.priority_order:
            rules = list(self.cfg.get(group_name, []) or [])
            for rule in rules:
                if self._match_rule(rule, sb, anchor_pattern):
                    best_assign.update(dict(rule.get("assign", {})))
                    matched = str(rule.get("name", group_name))
                    return SegmentRiskRuleResult(
                        risk_level=str(best_assign.get("risk_level", "medium")),
                        risk_flag_teacher=int(best_assign.get("risk_flag_teacher", 0)),
                        teacher_scale=float(best_assign.get("teacher_scale", 0.55)),
                        edge_weight=float(best_assign.get("edge_weight", 2.0)),
                        residual_rmax_m=self._residual_rmax_m(best_assign),
                        gate_bias=float(best_assign.get("gate_bias", 0.0)),
                        matched_rule=matched,
                    )

        return SegmentRiskRuleResult(
            risk_level=str(best_assign.get("risk_level", "medium")),
            risk_flag_teacher=int(best_assign.get("risk_flag_teacher", 0)),
            teacher_scale=float(best_assign.get("teacher_scale", 0.55)),
            edge_weight=float(best_assign.get("edge_weight", 2.0)),
            residual_rmax_m=self._residual_rmax_m(best_assign),
            gate_bias=float(best_assign.get("gate_bias", 0.0)),
            matched_rule=matched,
        )

    @classmethod
    def from_file(cls, path: str | None) -> "SegmentRiskRuleMatcher | None":
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            return None
        with p.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cls(cfg)
