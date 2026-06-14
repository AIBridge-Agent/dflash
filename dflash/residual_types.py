# pyright: reportMissingImports=none
"""Typed value objects for zero-draft residual verification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import prod

from .residual_errors import InvalidAccountingError, InvalidCandidateError


class CandidateSource(str, Enum):
    """Provenance for a token used by residual verification."""

    DRAFT_TAIL = "draft_tail"
    VERIFIER_ANCHOR = "verifier_anchor"


@dataclass(frozen=True)
class VerificationAnchor:
    """Target-observed mismatch token used as the residual root."""

    token_id: int
    position: int

    def __post_init__(self) -> None:
        if self.token_id < 0:
            raise InvalidCandidateError("anchor token_id must be non-negative")
        if self.position < 0:
            raise InvalidCandidateError("anchor position must be non-negative")


@dataclass(frozen=True)
class ResidualCandidate:
    """A candidate token reused from the already-generated DFlash tail."""

    token_id: int
    position: int
    probability: float
    source: CandidateSource
    source_block_position: int

    def __post_init__(self) -> None:
        if self.token_id < 0:
            raise InvalidCandidateError("candidate token_id must be non-negative")
        if self.position < 0:
            raise InvalidCandidateError("candidate position must be non-negative")
        if not 0.0 <= self.probability <= 1.0:
            raise InvalidCandidateError("candidate probability must be in [0, 1]")
        if self.source_block_position < 0:
            raise InvalidCandidateError("source_block_position must be non-negative")


@dataclass(frozen=True)
class ResidualPath:
    """A re-rooted candidate path after the verifier-observed mismatch token."""

    anchor: VerificationAnchor
    candidates: tuple[ResidualCandidate, ...]

    def __post_init__(self) -> None:
        previous_position = self.anchor.position
        for candidate in self.candidates:
            if candidate.source is not CandidateSource.DRAFT_TAIL:
                raise InvalidCandidateError(
                    "residual candidates must come from draft tail"
                )
            if candidate.position <= self.anchor.position:
                raise InvalidCandidateError(
                    "residual candidates must follow anchor position"
                )
            if candidate.position <= previous_position:
                raise InvalidCandidateError(
                    "residual candidate positions must be increasing"
                )
            previous_position = candidate.position

    @property
    def length(self) -> int:
        return len(self.candidates)

    @property
    def probability_product(self) -> float:
        return prod(candidate.probability for candidate in self.candidates)


@dataclass(frozen=True)
class CycleAccounting:
    """Cycle-level timing and accepted-token accounting."""

    current_accepted: int
    realized_residual_accepted: int
    draft_seconds: float
    target_seconds: float
    residual_target_seconds: float

    def __post_init__(self) -> None:
        if self.current_accepted < 0 or self.realized_residual_accepted < 0:
            raise InvalidAccountingError("accepted token counts must be non-negative")
        for name, value in (
            ("draft_seconds", self.draft_seconds),
            ("target_seconds", self.target_seconds),
            ("residual_target_seconds", self.residual_target_seconds),
        ):
            if value < 0:
                raise InvalidAccountingError(f"{name} must be non-negative")

    @property
    def total_seconds(self) -> float:
        return self.draft_seconds + self.target_seconds + self.residual_target_seconds

    @property
    def total_accepted(self) -> int:
        return self.current_accepted + self.realized_residual_accepted
