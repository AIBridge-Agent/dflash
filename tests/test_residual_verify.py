# pyright: reportMissingImports=none
import pytest

from dflash.residual_errors import ExactnessViolation
from dflash.residual_types import (
    CandidateSource,
    ResidualCandidate,
    ResidualPath,
    VerificationAnchor,
)
from dflash.residual_verify import ResidualVerificationResult, verify_residual_path


def _path() -> ResidualPath:
    return ResidualPath(
        anchor=VerificationAnchor(token_id=99, position=12),
        candidates=(
            ResidualCandidate(
                token_id=33,
                position=13,
                probability=0.7,
                source=CandidateSource.DRAFT_TAIL,
                source_block_position=3,
            ),
            ResidualCandidate(
                token_id=44,
                position=14,
                probability=0.6,
                source=CandidateSource.DRAFT_TAIL,
                source_block_position=4,
            ),
        ),
    )


def test_residual_tokens_are_accepted_only_when_target_matches() -> None:
    result = verify_residual_path(_path(), target_token_ids=(33, 44))

    assert result.accepted_token_ids == (33, 44)
    assert result.accepted_count == 2
    assert result.mismatch_token_id is None
    assert result.target_calls == 2
    assert result.drafter_calls == 0
    assert result.exact is True


def test_residual_mismatch_reveals_target_token_and_stops() -> None:
    result = verify_residual_path(_path(), target_token_ids=(33, 55))

    assert result.accepted_token_ids == (33,)
    assert result.accepted_count == 1
    assert result.mismatch_token_id == 55
    assert result.target_calls == 2
    assert result.drafter_calls == 0


def test_accepting_more_tokens_than_target_calls_is_exactness_violation() -> None:
    with pytest.raises(ExactnessViolation):
        ResidualVerificationResult(
            accepted_token_ids=(33, 44),
            mismatch_token_id=None,
            target_calls=1,
            drafter_calls=0,
            exact=True,
        )


def test_target_sequence_must_cover_each_claimed_residual_candidate() -> None:
    with pytest.raises(ExactnessViolation):
        verify_residual_path(_path(), target_token_ids=(33,))
