# pyright: reportMissingImports=none
import pytest

from dflash.residual_errors import NoResidualOpportunity, ProvenanceViolation
from dflash.residual_reroot import assert_residual_provenance, extract_residual_path
from dflash.residual_types import (
    CandidateSource,
    ResidualCandidate,
    ResidualPath,
    VerificationAnchor,
)


def test_extract_residual_path_after_early_rejection() -> None:
    path = extract_residual_path(
        block_token_ids=(100, 11, 22, 33, 44),
        block_probabilities=(1.0, 0.9, 0.8, 0.7, 0.6),
        start_position=10,
        acceptance_length=1,
        verifier_mismatch_token_id=99,
    )

    assert path.anchor == VerificationAnchor(token_id=99, position=12)
    assert [candidate.token_id for candidate in path.candidates] == [33, 44]
    assert [candidate.position for candidate in path.candidates] == [13, 14]
    assert [candidate.source_block_position for candidate in path.candidates] == [3, 4]
    assert all(
        candidate.source is CandidateSource.DRAFT_TAIL for candidate in path.candidates
    )


def test_rejection_at_last_draft_position_has_no_residual_tail() -> None:
    with pytest.raises(NoResidualOpportunity):
        extract_residual_path(
            block_token_ids=(100, 11, 22),
            block_probabilities=(1.0, 0.9, 0.8),
            start_position=10,
            acceptance_length=1,
            verifier_mismatch_token_id=99,
        )


def test_no_early_rejection_has_no_residual_opportunity() -> None:
    with pytest.raises(NoResidualOpportunity):
        extract_residual_path(
            block_token_ids=(100, 11, 22),
            block_probabilities=(1.0, 0.9, 0.8),
            start_position=10,
            acceptance_length=2,
            verifier_mismatch_token_id=99,
        )


def test_residual_provenance_violation_is_rejected() -> None:
    rogue = ResidualCandidate(
        token_id=44,
        position=14,
        probability=0.6,
        source=CandidateSource.DRAFT_TAIL,
        source_block_position=9,
    )
    path = ResidualPath(
        anchor=VerificationAnchor(token_id=99, position=12), candidates=(rogue,)
    )

    with pytest.raises(ProvenanceViolation):
        assert_residual_provenance(path, allowed_source_block_positions={3, 4})
