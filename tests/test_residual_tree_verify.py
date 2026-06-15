# pyright: reportMissingImports=none
from dflash.residual_surrogate import build_residual_ddtree
from dflash.residual_tree_verify import (
    children_by_parent,
    linearize_residual_tree,
    parent_indices,
    walk_residual_tree,
)
from dflash.residual_types import (
    CandidateSource,
    ResidualCandidate,
    VerificationAnchor,
)


def _candidate(token_id: int, depth: int, probability: float) -> ResidualCandidate:
    return ResidualCandidate(
        token_id=token_id,
        position=10 + depth,
        probability=probability,
        source=CandidateSource.DRAFT_TAIL,
        source_block_position=depth,
    )


def test_linearize_residual_tree_returns_topological_nodes() -> None:
    anchor = VerificationAnchor(token_id=99, position=10)
    tree = build_residual_ddtree(
        anchor,
        (
            (_candidate(1, 1, 0.9), _candidate(2, 1, 0.8)),
            (_candidate(3, 2, 0.7),),
        ),
        budget=4,
    )

    nodes = linearize_residual_tree(tree)

    assert [node.path_tokens for node in nodes] == [(1,), (2,), (1, 3), (2, 3)]
    assert parent_indices(nodes) == (None, 0, 0, 1, 2)
    assert [node.token_id for node in children_by_parent(nodes)[0]] == [1, 2]


def test_walk_residual_tree_accepts_path_until_first_reject_frontier() -> None:
    anchor = VerificationAnchor(token_id=99, position=10)
    tree = build_residual_ddtree(
        anchor,
        (
            (_candidate(1, 1, 0.9), _candidate(2, 1, 0.8)),
            (_candidate(3, 2, 0.7), _candidate(4, 2, 0.6)),
        ),
        budget=4,
    )
    nodes = linearize_residual_tree(tree)

    result = walk_residual_tree(nodes, target_token_ids_by_flat_index=(1, 5, 8, 9, 9))

    assert [node.token_id for node in result.accepted_nodes] == [1]
    assert result.mismatch_token_id == 5
    assert result.edge_records[0]["outcome"] == "accepted"
    rejected = [record for record in result.edge_records if record["first_reject"]]
    assert [record["token_id"] for record in rejected] == [3]


def test_walk_residual_tree_can_accept_multiple_edges() -> None:
    anchor = VerificationAnchor(token_id=99, position=10)
    tree = build_residual_ddtree(
        anchor,
        (
            (_candidate(1, 1, 0.9),),
            (_candidate(3, 2, 0.7),),
        ),
        budget=2,
    )
    nodes = linearize_residual_tree(tree)

    result = walk_residual_tree(nodes, target_token_ids_by_flat_index=(1, 3, 7))

    assert [node.token_id for node in result.accepted_nodes] == [1, 3]
    assert result.mismatch_token_id == 7
    assert [record["outcome"] for record in result.edge_records] == [
        "accepted",
        "accepted",
    ]
