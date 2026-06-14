# pyright: reportMissingImports=none
"""Generation-loop accounting seam for DFlash residual verification."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .residual_gating import ResidualGateDecision, decide_residual_gate
from .residual_reroot import extract_residual_path
from .residual_surrogate import ResidualDDTree, build_residual_ddtree, estimate_residual_gain
from .residual_types import CandidateSource, ResidualCandidate, ResidualPath


@dataclass(frozen=True)
class ResidualOpportunity:
    """A gated residual opportunity built from one already-sampled DFlash block."""

    path: ResidualPath
    estimated_residual_gain: float
    gate: ResidualGateDecision
    residual_tree: ResidualDDTree | None = None
    drafter_calls: int = 0

    @property
    def should_run(self) -> bool:
        return self.gate.should_run

    @property
    def tree_nodes(self) -> int:
        return 0 if self.residual_tree is None else len(self.residual_tree.nodes)


def build_residual_candidate_groups(
    *,
    top_token_ids_by_block: Sequence[Sequence[int]],
    top_probabilities_by_block: Sequence[Sequence[float]],
    start_position: int,
    first_block_index: int,
) -> tuple[tuple[ResidualCandidate, ...], ...]:
    """Convert existing per-position draft top-k probabilities into tree candidates."""

    groups: list[tuple[ResidualCandidate, ...]] = []
    for offset, (token_ids, probabilities) in enumerate(
        zip(top_token_ids_by_block, top_probabilities_by_block, strict=True)
    ):
        block_index = first_block_index + offset
        group = tuple(
            ResidualCandidate(
                token_id=int(token_id),
                position=start_position + block_index,
                probability=float(probability),
                source=CandidateSource.DRAFT_TAIL,
                source_block_position=block_index,
            )
            for token_id, probability in zip(token_ids, probabilities, strict=True)
        )
        groups.append(group)
    return tuple(groups)


def build_residual_opportunity(
    *,
    block_token_ids: tuple[int, ...],
    block_probabilities: tuple[float, ...],
    start_position: int,
    acceptance_length: int,
    verifier_mismatch_token_id: int,
    current_accepted: int,
    draft_seconds: float,
    target_seconds: float,
    residual_budget: int,
    residual_candidate_groups: Sequence[Sequence[ResidualCandidate]] | None = None,
) -> ResidualOpportunity:
    """Build and gate a residual opportunity without invoking the drafter.

    When candidate groups are provided, the gate uses a residual DDTree over the
    whole residual tail distribution. Otherwise it falls back to the sampled-tail
    single-path expected prefix length.
    """

    path = extract_residual_path(
        block_token_ids=block_token_ids,
        block_probabilities=block_probabilities,
        start_position=start_position,
        acceptance_length=acceptance_length,
        verifier_mismatch_token_id=verifier_mismatch_token_id,
    )
    residual_tree = None
    if residual_candidate_groups:
        residual_tree = build_residual_ddtree(
            path.anchor, residual_candidate_groups, budget=residual_budget
        )
        estimated_gain = residual_tree.expected_accept_length
    else:
        estimated_gain = estimate_residual_gain([path], budget=residual_budget)

    gate = decide_residual_gate(
        current_accepted=current_accepted,
        draft_seconds=draft_seconds,
        target_seconds=target_seconds,
        estimated_residual_gain=estimated_gain,
    )
    return ResidualOpportunity(
        path=path,
        estimated_residual_gain=estimated_gain,
        gate=gate,
        residual_tree=residual_tree,
    )
