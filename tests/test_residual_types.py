# pyright: reportMissingImports=none
from dataclasses import FrozenInstanceError

import pytest

from dflash.residual_errors import InvalidAccountingError, InvalidCandidateError
from dflash.residual_types import (
    CandidateSource,
    CycleAccounting,
    ResidualCandidate,
    ResidualPath,
    VerificationAnchor,
)


def test_residual_candidate_is_immutable_and_records_provenance() -> None:
    candidate = ResidualCandidate(
        token_id=42,
        position=5,
        probability=0.25,
        source=CandidateSource.DRAFT_TAIL,
        source_block_position=3,
    )

    assert candidate.token_id == 42
    assert candidate.source is CandidateSource.DRAFT_TAIL
    with pytest.raises(FrozenInstanceError):
        candidate.token_id = 7  # type: ignore[misc]


def test_residual_candidate_rejects_invalid_values() -> None:
    with pytest.raises(InvalidCandidateError):
        ResidualCandidate(
            token_id=-1,
            position=1,
            probability=0.5,
            source=CandidateSource.DRAFT_TAIL,
            source_block_position=0,
        )
    with pytest.raises(InvalidCandidateError):
        ResidualCandidate(
            token_id=1,
            position=1,
            probability=1.5,
            source=CandidateSource.DRAFT_TAIL,
            source_block_position=0,
        )


def test_residual_path_requires_tail_after_anchor() -> None:
    anchor = VerificationAnchor(token_id=10, position=4)
    candidate = ResidualCandidate(
        token_id=11,
        position=5,
        probability=0.8,
        source=CandidateSource.DRAFT_TAIL,
        source_block_position=1,
    )

    path = ResidualPath(anchor=anchor, candidates=(candidate,))

    assert path.length == 1
    assert path.probability_product == pytest.approx(0.8)


def test_residual_path_rejects_candidate_before_anchor() -> None:
    anchor = VerificationAnchor(token_id=10, position=4)
    candidate = ResidualCandidate(
        token_id=11,
        position=4,
        probability=0.8,
        source=CandidateSource.DRAFT_TAIL,
        source_block_position=1,
    )

    with pytest.raises(InvalidCandidateError):
        ResidualPath(anchor=anchor, candidates=(candidate,))


def test_cycle_accounting_distinguishes_draft_target_and_residual_costs() -> None:
    accounting = CycleAccounting(
        current_accepted=2,
        realized_residual_accepted=1,
        draft_seconds=0.01,
        target_seconds=0.02,
        residual_target_seconds=0.02,
    )

    assert accounting.total_seconds == pytest.approx(0.05)
    assert accounting.total_accepted == 3


def test_cycle_accounting_rejects_negative_costs() -> None:
    with pytest.raises(InvalidAccountingError):
        CycleAccounting(
            current_accepted=1,
            realized_residual_accepted=0,
            draft_seconds=-0.01,
            target_seconds=0.02,
            residual_target_seconds=0.02,
        )
