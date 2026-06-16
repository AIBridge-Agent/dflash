import pytest

from dflash.ddtree_termination import (
    compute_termination_predictions,
    rank_termination_predictions,
    termination_depth_distribution,
)
from dflash.residual_tree_verify import ResidualTreeNode
from dflash.residual_types import CandidateSource, ResidualCandidate


def _candidate(token_id: int, position: int, probability: float) -> ResidualCandidate:
    return ResidualCandidate(
        token_id=token_id,
        position=position,
        probability=probability,
        source=CandidateSource.DRAFT_TAIL,
        source_block_position=position,
    )


def test_compute_termination_predictions_for_branching_tree() -> None:
    nodes = (
        ResidualTreeNode(
            flat_index=1,
            parent_index=0,
            candidate=_candidate(10, 1, 0.6),
            path_tokens=(10,),
        ),
        ResidualTreeNode(
            flat_index=2,
            parent_index=0,
            candidate=_candidate(11, 1, 0.3),
            path_tokens=(11,),
        ),
        ResidualTreeNode(
            flat_index=3,
            parent_index=1,
            candidate=_candidate(20, 2, 0.5),
            path_tokens=(10, 20),
        ),
    )

    predictions = {
        prediction.flat_index: prediction
        for prediction in compute_termination_predictions(nodes)
    }

    assert predictions[0].termination_probability == pytest.approx(0.1)
    assert predictions[1].path_probability == pytest.approx(0.6)
    assert predictions[1].termination_probability == pytest.approx(0.3)
    assert predictions[2].termination_probability == pytest.approx(0.3)
    assert predictions[3].path_probability == pytest.approx(0.3)
    assert predictions[3].termination_probability == pytest.approx(0.3)
    assert sum(
        p.termination_probability for p in predictions.values()
    ) == pytest.approx(1.0)


def test_rank_termination_predictions_prefers_largest_mass() -> None:
    nodes = (
        ResidualTreeNode(
            flat_index=1,
            parent_index=0,
            candidate=_candidate(10, 1, 0.9),
            path_tokens=(10,),
        ),
        ResidualTreeNode(
            flat_index=2,
            parent_index=1,
            candidate=_candidate(20, 2, 0.2),
            path_tokens=(10, 20),
        ),
    )

    ranked = rank_termination_predictions(compute_termination_predictions(nodes))

    assert ranked[0].flat_index == 1
    assert ranked[0].termination_probability == pytest.approx(0.72)


def test_termination_depth_distribution() -> None:
    nodes = (
        ResidualTreeNode(
            flat_index=1,
            parent_index=0,
            candidate=_candidate(10, 1, 0.5),
            path_tokens=(10,),
        ),
    )

    distribution = termination_depth_distribution(
        compute_termination_predictions(nodes)
    )

    assert distribution == {0: 0.5, 1: 0.5}
