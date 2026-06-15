# pyright: reportMissingImports=none
"""Generation-loop accounting seam for DFlash residual verification."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .residual_gating import ResidualGateDecision, decide_residual_gate
from .residual_reroot import extract_residual_path
from .residual_surrogate import (
    ResidualDDTree,
    estimate_residual_gain,
    estimate_top1_path_eal,
)
from .residual_types import CandidateSource, ResidualCandidate, ResidualPath

ResidualEdgeRecord = dict[str, bool | float | int | str]


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


def build_residual_edge_records(
    path: ResidualPath, *, residual_acceptance_length: int
) -> tuple[ResidualEdgeRecord, ...]:
    """Build per-edge calibration labels from one residual verification run.

    Accepted residual-tail edges are observed as positive labels. The first edge
    after the accepted prefix is observed as the first rejection. Edges after the
    first rejection are unobserved and intentionally omitted.
    """

    if residual_acceptance_length < 0:
        raise ValueError("residual_acceptance_length must be non-negative")

    records: list[ResidualEdgeRecord] = []
    for edge_index, candidate in enumerate(path.candidates):
        accepted = edge_index < residual_acceptance_length
        first_reject = edge_index == residual_acceptance_length
        if not accepted and not first_reject:
            break
        records.append(
            {
                "edge_index": edge_index,
                "position": candidate.position,
                "source_block_position": candidate.source_block_position,
                "edge_probability": float(candidate.probability),
                "accepted": accepted,
                "first_reject": first_reject,
                "outcome": "accepted" if accepted else "first_rejected",
            }
        )
    return tuple(records)


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
    residual_target_seconds: float | None = None,
    residual_gain_scale: float = 1.0,
    residual_min_gain: float = 0.0,
    residual_min_margin: float = 0.0,
) -> ResidualOpportunity:
    """Build and gate a residual opportunity without invoking the drafter.

    When candidate groups are provided, the gate uses the top-1 residual tail
    sequence as a cheap EAL estimate. Actual verification integrations may still
    spend a DDTree budget after this gate passes. Otherwise it falls back to the
    sampled-tail single-path expected prefix length.
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
        estimated_gain = estimate_top1_path_eal(residual_candidate_groups)
    else:
        estimated_gain = estimate_residual_gain([path], budget=residual_budget)

    gate = decide_residual_gate(
        current_accepted=current_accepted,
        draft_seconds=draft_seconds,
        target_seconds=target_seconds,
        estimated_residual_gain=estimated_gain,
        residual_target_seconds=residual_target_seconds,
        residual_gain_scale=residual_gain_scale,
        min_residual_gain=residual_min_gain,
        min_throughput_margin=residual_min_margin,
    )
    return ResidualOpportunity(
        path=path,
        estimated_residual_gain=estimated_gain,
        gate=gate,
        residual_tree=residual_tree,
    )
