# pyright: reportMissingImports=none
"""Exact residual verification helpers for DFlash residual paths."""

from __future__ import annotations

from dataclasses import dataclass

from .residual_errors import ExactnessViolation
from .residual_types import ResidualPath


@dataclass(frozen=True)
class ResidualVerificationResult:
    """Outcome of one exact zero-draft residual verification attempt."""

    accepted_token_ids: tuple[int, ...]
    mismatch_token_id: int | None
    target_calls: int
    drafter_calls: int
    exact: bool

    def __post_init__(self) -> None:
        if self.target_calls < 0 or self.drafter_calls < 0:
            raise ExactnessViolation("verification call counts must be non-negative")
        if len(self.accepted_token_ids) > self.target_calls:
            raise ExactnessViolation(
                "accepted residual tokens cannot exceed target verifications"
            )
        if self.drafter_calls != 0:
            raise ExactnessViolation(
                "zero-draft residual verification cannot call the drafter"
            )
        if not self.exact:
            raise ExactnessViolation("residual verification result must be exact")

    @property
    def accepted_count(self) -> int:
        return len(self.accepted_token_ids)


def verify_residual_path(
    path: ResidualPath,
    *,
    target_token_ids: tuple[int, ...],
) -> ResidualVerificationResult:
    """Verify residual candidates against target-observed token ids.

    The simulator requires a target token for every residual candidate so tests
    cannot accidentally count an unverified candidate as accepted.
    """

    if len(target_token_ids) < len(path.candidates):
        raise ExactnessViolation("target_token_ids must cover every residual candidate")

    accepted: list[int] = []
    target_calls = 0
    mismatch_token_id: int | None = None
    for candidate, target_token_id in zip(
        path.candidates, target_token_ids, strict=False
    ):
        target_calls += 1
        if candidate.token_id != target_token_id:
            mismatch_token_id = target_token_id
            break
        accepted.append(candidate.token_id)

    return ResidualVerificationResult(
        accepted_token_ids=tuple(accepted),
        mismatch_token_id=mismatch_token_id,
        target_calls=target_calls,
        drafter_calls=0,
        exact=True,
    )
