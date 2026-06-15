# pyright: reportMissingImports=none
"""DDTree-style residual surrogate scoring helpers."""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .residual_errors import InvalidBudgetError
from .residual_types import ResidualPath, VerificationAnchor


@dataclass(frozen=True)
class ResidualDDTree:
    """A residual DDTree represented by its selected prefix nodes."""

    nodes: tuple[ResidualPath, ...]

    @property
    def expected_accept_length(self) -> float:
        """Expected accepted length: sum of selected prefix probabilities."""

        return sum(node.probability_product for node in self.nodes)


def score_path(path: ResidualPath) -> float:
    """Return prefix expected accepted length for one residual path.

    A full-tail product estimates only the probability that the entire residual
    tail is accepted. Residual verification can still be useful when only a
    prefix is accepted, so the single-path gain estimate is the expected prefix
    length: q1 + q1*q2 + ... + q1*...*qL.
    """

    running_product = 1.0
    expected_length = 0.0
    for candidate in path.candidates:
        running_product *= candidate.probability
        expected_length += running_product
    return expected_length


def estimate_depth_mass_eal(candidate_groups: Sequence[Sequence]) -> float:
    """Cheap residual EAL upper estimate from per-depth top-k probability mass.

    For each residual-tail depth d, let p_d be the sum of the selected top-k
    edge probabilities at that depth, clipped to 1. The estimate includes the
    verifier bonus token and prefix expected accepted length:
    1 + p_1 + p_1 p_2 + ... . This avoids constructing a DDTree before the
    gate decides whether residual verification is worthwhile.
    """

    if not candidate_groups:
        return 0.0

    running_product = 1.0
    expected_length = 1.0
    for group in candidate_groups:
        depth_mass = min(1.0, sum(float(candidate.probability) for candidate in group))
        running_product *= depth_mass
        expected_length += running_product
    return expected_length


def build_residual_ddtree(
    anchor: VerificationAnchor,
    candidate_groups: Sequence[Sequence],
    *,
    budget: int,
) -> ResidualDDTree:
    """Greedily build a residual DDTree from per-position candidates.

    This uses the already-computed draft distribution for each residual tail
    position. It selects the highest-probability valid prefix nodes under a node
    budget; the resulting tree objective is the sum of selected node prefix
    probabilities, matching DDTree-style expected accepted length.
    """

    if budget <= 0:
        raise InvalidBudgetError("residual budget must be positive")
    if not candidate_groups:
        return ResidualDDTree(nodes=())

    frontier: list[
        tuple[float, tuple[int, ...], tuple[int, ...], int, int, tuple]
    ] = []
    insertion_order = 0
    for candidate in candidate_groups[0]:
        heapq.heappush(frontier, _frontier_item((candidate,), insertion_order))
        insertion_order += 1

    selected: list[ResidualPath] = []
    while frontier and len(selected) < budget:
        _, _, _, depth, _, candidates = heapq.heappop(frontier)
        selected.append(ResidualPath(anchor=anchor, candidates=candidates))
        if depth >= len(candidate_groups):
            continue
        for candidate in candidate_groups[depth]:
            heapq.heappush(
                frontier, _frontier_item((*candidates, candidate), insertion_order)
            )
            insertion_order += 1

    return ResidualDDTree(nodes=tuple(selected))


def _frontier_item(
    candidates: tuple,
    insertion_order: int,
) -> tuple[float, tuple[int, ...], tuple[int, ...], int, int, tuple]:
    probability_product = 1.0
    for candidate in candidates:
        probability_product *= float(candidate.probability)
    return (
        -probability_product,
        tuple(candidate.token_id for candidate in candidates),
        tuple(candidate.source_block_position for candidate in candidates),
        len(candidates),
        insertion_order,
        candidates,
    )


def estimate_residual_tree_gain(
    anchor: VerificationAnchor,
    candidate_groups: Sequence[Sequence],
    *,
    budget: int,
) -> float:
    """Estimate residual gain by constructing a residual DDTree."""

    return build_residual_ddtree(
        anchor, candidate_groups, budget=budget
    ).expected_accept_length


def select_residual_paths(
    paths: Iterable[ResidualPath], *, budget: int
) -> tuple[ResidualPath, ...]:
    """Select residual paths under a deterministic budget.

    Higher expected prefix accepted length paths come first. Exact ties are
    resolved by token sequence, then source block positions.
    """

    if budget <= 0:
        raise InvalidBudgetError("residual budget must be positive")

    def sort_key(path: ResidualPath) -> tuple[float, tuple[int, ...], tuple[int, ...]]:
        return (
            -score_path(path),
            tuple(candidate.token_id for candidate in path.candidates),
            tuple(candidate.source_block_position for candidate in path.candidates),
        )

    return tuple(sorted(paths, key=sort_key)[:budget])


def estimate_residual_gain(paths: Iterable[ResidualPath], *, budget: int) -> float:
    """Estimate residual accepted-token gain as selected path prefix EAL."""

    return sum(score_path(path) for path in select_residual_paths(paths, budget=budget))
