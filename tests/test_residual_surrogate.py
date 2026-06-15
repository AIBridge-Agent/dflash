# pyright: reportMissingImports=none
import pytest

from dflash.residual_errors import InvalidBudgetError
from dflash.residual_surrogate import (
    build_residual_ddtree,
    estimate_depth_mass_eal,
    estimate_residual_gain,
    select_residual_paths,
    score_path,
)
from dflash.residual_types import (
    CandidateSource,
    ResidualCandidate,
    ResidualPath,
    VerificationAnchor,
)


def _path(
    anchor_token: int, probs: tuple[float, ...], tokens: tuple[int, ...]
) -> ResidualPath:
    anchor = VerificationAnchor(token_id=anchor_token, position=10)
    candidates = tuple(
        ResidualCandidate(
            token_id=token,
            position=11 + idx,
            probability=prob,
            source=CandidateSource.DRAFT_TAIL,
            source_block_position=idx + 1,
        )
        for idx, (token, prob) in enumerate(zip(tokens, probs, strict=True))
    )
    return ResidualPath(anchor=anchor, candidates=candidates)


def test_single_path_score_is_prefix_expected_acceptance_length() -> None:
    path = _path(99, probs=(0.5, 0.25), tokens=(1, 2))

    # E[accepted residual length] = q1 + q1*q2.
    assert score_path(path) == pytest.approx(0.625)


def test_estimate_residual_gain_sums_selected_prefix_expected_lengths() -> None:
    paths = [
        _path(99, probs=(0.5,), tokens=(1,)),
        _path(99, probs=(0.25, 0.5), tokens=(2, 3)),
    ]

    assert estimate_residual_gain(paths, budget=2) == pytest.approx(0.875)


def test_budget_truncates_paths_deterministically_by_score_then_tokens() -> None:
    high = _path(99, probs=(0.9,), tokens=(9,))
    tie_a = _path(99, probs=(0.5,), tokens=(1,))
    tie_b = _path(99, probs=(0.5,), tokens=(2,))

    selected = select_residual_paths([tie_b, high, tie_a], budget=2)

    assert selected == (high, tie_a)


def test_depth_mass_eal_uses_topk_mass_without_tree_materialization() -> None:
    depth1 = (
        ResidualCandidate(1, 11, 0.4, CandidateSource.DRAFT_TAIL, 1),
        ResidualCandidate(2, 11, 0.3, CandidateSource.DRAFT_TAIL, 1),
    )
    depth2 = (
        ResidualCandidate(3, 12, 0.5, CandidateSource.DRAFT_TAIL, 2),
        ResidualCandidate(4, 12, 0.25, CandidateSource.DRAFT_TAIL, 2),
    )

    # EAL = 1 bonus token + (0.4 + 0.3) + (0.4 + 0.3) * (0.5 + 0.25).
    assert estimate_depth_mass_eal((depth1, depth2)) == pytest.approx(2.225)


def test_residual_ddtree_expected_accept_length_sums_tree_prefix_nodes() -> None:
    anchor = VerificationAnchor(token_id=99, position=10)
    depth1 = (
        ResidualCandidate(1, 11, 0.8, CandidateSource.DRAFT_TAIL, 1),
        ResidualCandidate(2, 11, 0.5, CandidateSource.DRAFT_TAIL, 1),
    )
    depth2 = (
        ResidualCandidate(3, 12, 0.7, CandidateSource.DRAFT_TAIL, 2),
        ResidualCandidate(4, 12, 0.6, CandidateSource.DRAFT_TAIL, 2),
    )

    tree = build_residual_ddtree(anchor, (depth1, depth2), budget=3)

    # Greedy DDTree nodes: [1] p=.8, [1,3] p=.56, [2] p=.5.
    assert [tuple(c.token_id for c in node.candidates) for node in tree.nodes] == [
        (1,),
        (1, 3),
        (2,),
    ]
    assert tree.expected_accept_length == pytest.approx(1.86)


def test_empty_paths_estimate_zero_gain() -> None:
    assert estimate_residual_gain([], budget=4) == 0.0


def test_invalid_budget_is_rejected() -> None:
    with pytest.raises(InvalidBudgetError):
        select_residual_paths([], budget=0)
