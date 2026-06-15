# pyright: reportMissingImports=none
"""Adaptive residual throughput gating."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .residual_errors import InvalidAccountingError


@dataclass(frozen=True)
class ResidualGateDecision:
    """Predicted throughput comparison for one residual opportunity."""

    should_run: bool
    baseline_throughput: float
    predicted_residual_throughput: float
    current_accepted: int
    estimated_residual_gain: float
    effective_estimated_residual_gain: float
    draft_seconds: float
    target_seconds: float
    residual_target_seconds: float
    residual_gain_scale: float
    min_residual_gain: float
    min_throughput_margin: float
    reason: str


def decide_residual_gate(
    *,
    current_accepted: int,
    draft_seconds: float,
    target_seconds: float,
    estimated_residual_gain: float,
    residual_target_seconds: float | None = None,
    residual_gain_scale: float = 1.0,
    min_residual_gain: float = 0.0,
    min_throughput_margin: float = 0.0,
) -> ResidualGateDecision:
    """Decide whether residual verification should run.

    First-version DFlash residual accounting sets `T_res = T` unless an explicit
    residual target cost is supplied by an integration test or benchmark.
    """

    if current_accepted < 0:
        raise InvalidAccountingError("current_accepted must be non-negative")
    for name, value in (
        ("estimated_residual_gain", estimated_residual_gain),
        ("residual_gain_scale", residual_gain_scale),
        ("min_residual_gain", min_residual_gain),
        ("min_throughput_margin", min_throughput_margin),
    ):
        if value < 0 or not isfinite(value):
            raise InvalidAccountingError(f"{name} must be finite and non-negative")

    effective_estimated_residual_gain = estimated_residual_gain * residual_gain_scale
    residual_target_seconds = (
        target_seconds if residual_target_seconds is None else residual_target_seconds
    )
    for name, value in (
        ("draft_seconds", draft_seconds),
        ("target_seconds", target_seconds),
        ("residual_target_seconds", residual_target_seconds),
    ):
        if value < 0 or not isfinite(value):
            raise InvalidAccountingError(f"{name} must be finite and non-negative")

    baseline_denominator = draft_seconds + target_seconds
    residual_denominator = baseline_denominator + residual_target_seconds
    if baseline_denominator <= 0 or residual_denominator <= 0:
        raise InvalidAccountingError("throughput denominators must be positive")

    baseline = current_accepted / baseline_denominator
    predicted = (
        current_accepted + effective_estimated_residual_gain
    ) / residual_denominator
    if predicted <= baseline * (1.0 + min_throughput_margin):
        reason = "predicted_not_profitable"
    else:
        reason = "predicted_improvement"
    should_run = reason == "predicted_improvement"
    return ResidualGateDecision(
        should_run=should_run,
        baseline_throughput=baseline,
        predicted_residual_throughput=predicted,
        current_accepted=current_accepted,
        estimated_residual_gain=estimated_residual_gain,
        effective_estimated_residual_gain=effective_estimated_residual_gain,
        draft_seconds=draft_seconds,
        target_seconds=target_seconds,
        residual_target_seconds=residual_target_seconds,
        residual_gain_scale=residual_gain_scale,
        min_residual_gain=min_residual_gain,
        min_throughput_margin=min_throughput_margin,
        reason=reason,
    )
