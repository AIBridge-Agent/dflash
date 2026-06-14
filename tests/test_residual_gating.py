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
