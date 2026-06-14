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
    draft_seconds: float
    target_seconds: float
    residual_target_seconds: float
    reason: str


def decide_residual_gate(
    *,
    current_accepted: int,
    draft_seconds: float,
    target_seconds: float,
    estimated_residual_gain: float,
    residual_target_seconds: float | None = None,
) -> ResidualGateDecision:
    """Decide whether residual verification should run.

    First-version DFlash residual accounting sets `T_res = T` unless an explicit
    residual target cost is supplied by an integration test or benchmark.
    """

    if current_accepted < 0:
        raise InvalidAccountingError("current_accepted must be non-negative")
    if estimated_residual_gain < 0 or not isfinite(estimated_residual_gain):
        raise InvalidAccountingError(
            "estimated_residual_gain must be finite and non-negative"
        )

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
    predicted = (current_accepted + estimated_residual_gain) / residual_denominator
    should_run = predicted > baseline
    return ResidualGateDecision(
        should_run=should_run,
        baseline_throughput=baseline,
        predicted_residual_throughput=predicted,
        current_accepted=current_accepted,
        estimated_residual_gain=estimated_residual_gain,
        draft_seconds=draft_seconds,
        target_seconds=target_seconds,
        residual_target_seconds=residual_target_seconds,
        reason="predicted_improvement" if should_run else "predicted_not_profitable",
    )
