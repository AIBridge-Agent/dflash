# pyright: reportMissingImports=none
import pytest

from dflash.residual_errors import InvalidAccountingError
from dflash.residual_gating import decide_residual_gate


def test_gate_triggers_on_strict_predicted_throughput_improvement() -> None:
    decision = decide_residual_gate(
        current_accepted=1,
        draft_seconds=1.0,
        target_seconds=1.0,
        estimated_residual_gain=2.0,
    )

    assert decision.should_run is True
    assert decision.reason == "predicted_improvement"
    assert decision.baseline_throughput == pytest.approx(0.5)
    assert decision.predicted_residual_throughput == pytest.approx(1.0)


def test_gate_does_not_trigger_on_equality() -> None:
    decision = decide_residual_gate(
        current_accepted=2,
        draft_seconds=1.0,
        target_seconds=1.0,
        estimated_residual_gain=1.0,
    )

    assert decision.baseline_throughput == pytest.approx(1.0)
    assert decision.predicted_residual_throughput == pytest.approx(1.0)
    assert decision.should_run is False
    assert decision.reason == "predicted_not_profitable"


def test_zero_current_acceptance_can_still_trigger() -> None:
    decision = decide_residual_gate(
        current_accepted=0,
        draft_seconds=1.0,
        target_seconds=1.0,
        estimated_residual_gain=1.0,
    )

    assert decision.baseline_throughput == 0.0
    assert decision.should_run is True


def test_gate_applies_gain_calibration_after_min_gain_filter() -> None:
    decision = decide_residual_gate(
        current_accepted=1,
        draft_seconds=1.0,
        target_seconds=1.0,
        estimated_residual_gain=4.0,
        residual_gain_scale=0.5,
        min_residual_gain=3.0,
        min_throughput_margin=0.2,
    )

    assert decision.should_run is True
    assert decision.reason == "predicted_improvement"
    assert decision.effective_estimated_residual_gain == pytest.approx(2.0)


def test_gate_requires_estimated_gain_above_minimum() -> None:
    decision = decide_residual_gate(
        current_accepted=1,
        draft_seconds=1.0,
        target_seconds=1.0,
        estimated_residual_gain=2.0,
        min_residual_gain=2.0,
    )

    assert decision.should_run is False
    assert decision.reason == "estimated_gain_below_min"


def test_gate_requires_minimum_throughput_margin() -> None:
    decision = decide_residual_gate(
        current_accepted=10,
        draft_seconds=1.0,
        target_seconds=1.0,
        estimated_residual_gain=6.0,
        min_residual_gain=1.0,
        min_throughput_margin=0.5,
    )

    assert decision.should_run is False
    assert decision.reason == "predicted_not_profitable"


def test_invalid_timing_denominator_is_rejected() -> None:
    with pytest.raises(InvalidAccountingError):
        decide_residual_gate(
            current_accepted=1,
            draft_seconds=0.0,
            target_seconds=0.0,
            estimated_residual_gain=1.0,
        )


def test_negative_estimated_gain_is_rejected() -> None:
    with pytest.raises(InvalidAccountingError):
        decide_residual_gate(
            current_accepted=1,
            draft_seconds=1.0,
            target_seconds=1.0,
            estimated_residual_gain=-0.1,
        )
