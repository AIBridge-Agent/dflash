# pyright: reportMissingImports=none
"""Residual-tail extraction and re-rooting for DFlash draft blocks."""

from __future__ import annotations

from collections.abc import Iterable

from .residual_errors import (
    InvalidRerootError,
    NoResidualOpportunity,
    ProvenanceViolation,
)
from .residual_types import (
    CandidateSource,
    ResidualCandidate,
    ResidualPath,
    VerificationAnchor,
)


def extract_residual_path(
    *,
    block_token_ids: tuple[int, ...],
    block_probabilities: tuple[float, ...],
    start_position: int,
    acceptance_length: int,
    verifier_mismatch_token_id: int,
) -> ResidualPath:
    """Build a re-rooted residual path from an already-sampled DFlash block.

    `acceptance_length` follows DFlash's current accounting: it is the number of
    accepted draft tokens from `block_token_ids[1:]`. The mismatch draft token is
    therefore at `acceptance_length + 1`, and reusable residual tail candidates
    are strictly after that index.
    """

    if len(block_token_ids) != len(block_probabilities):
        raise InvalidRerootError(
            "block_token_ids and block_probabilities must have equal length"
        )
    if len(block_token_ids) < 2:
        raise InvalidRerootError(
            "DFlash block must include a root token and at least one draft token"
        )
    if start_position < 0:
        raise InvalidRerootError("start_position must be non-negative")
    if acceptance_length < 0:
        raise InvalidRerootError("acceptance_length must be non-negative")

    last_draft_acceptance = len(block_token_ids) - 1
    if acceptance_length >= last_draft_acceptance:
        raise NoResidualOpportunity("no early rejection occurred")

    mismatch_block_index = acceptance_length + 1
    residual_start_index = mismatch_block_index + 1
    if residual_start_index >= len(block_token_ids):
        raise NoResidualOpportunity("early rejection left no residual tail")

    anchor = VerificationAnchor(
        token_id=verifier_mismatch_token_id,
        position=start_position + mismatch_block_index,
    )
    candidates = tuple(
        ResidualCandidate(
            token_id=block_token_ids[block_index],
            position=start_position + block_index,
            probability=block_probabilities[block_index],
            source=CandidateSource.DRAFT_TAIL,
            source_block_position=block_index,
        )
        for block_index in range(residual_start_index, len(block_token_ids))
    )
    return ResidualPath(anchor=anchor, candidates=candidates)


def assert_residual_provenance(
    path: ResidualPath,
    *,
    allowed_source_block_positions: Iterable[int],
) -> None:
    """Fail if a residual path includes candidates outside the original tail."""

    allowed = set(allowed_source_block_positions)
    for candidate in path.candidates:
        if candidate.source is not CandidateSource.DRAFT_TAIL:
            raise ProvenanceViolation("residual candidate is not from draft tail")
        if candidate.source_block_position not in allowed:
            raise ProvenanceViolation(
                f"source block position {candidate.source_block_position} is not in the residual tail"
            )
