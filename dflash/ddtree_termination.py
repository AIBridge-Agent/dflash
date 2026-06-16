# pyright: reportMissingImports=none
"""DDTree termination-probability diagnostics.

For a target-verified tree, a walk terminates at the first visited parent whose
next target token is not one of the selected children. If a node is reached with
probability P(path), and the selected outgoing child edge probabilities sum to
S, then the draft-model probability mass that verification terminates at that
node is P(path) * (1 - S). Leaves have S = 0.
"""

from __future__ import annotations

from dataclasses import dataclass

from .residual_tree_verify import ResidualTreeNode, children_by_parent


@dataclass(frozen=True)
class TerminationPrediction:
    """Predicted DDTree termination mass for one parent flat index."""

    flat_index: int
    depth: int
    path_probability: float
    child_edge_probability_sum: float
    termination_probability: float


def compute_termination_predictions(
    nodes: tuple[ResidualTreeNode, ...],
) -> tuple[TerminationPrediction, ...]:
    """Return termination-mass predictions for the anchor and all tree nodes.

    The anchor/root has flat index 0 and path probability 1.0. Node path
    probabilities are products of selected conditional edge probabilities along
    the unique root-to-node path. Child probability sums are clipped to [0, 1]
    before converting to residual stop mass, guarding against numerical or
    malformed top-k sums.
    """

    children = children_by_parent(nodes)
    node_by_flat_index = {node.flat_index: node for node in nodes}
    path_probability_by_flat_index: dict[int, float] = {0: 1.0}
    depth_by_flat_index: dict[int, int] = {0: 0}

    for node in sorted(nodes, key=lambda item: (item.depth, item.flat_index)):
        parent_probability = path_probability_by_flat_index.get(node.parent_index, 0.0)
        path_probability_by_flat_index[node.flat_index] = parent_probability * float(
            node.probability
        )
        depth_by_flat_index[node.flat_index] = int(node.depth)

    predictions: list[TerminationPrediction] = []
    for flat_index in sorted(path_probability_by_flat_index):
        outgoing = children.get(flat_index, ())
        child_sum = sum(float(child.probability) for child in outgoing)
        clipped_child_sum = min(1.0, max(0.0, child_sum))
        path_probability = path_probability_by_flat_index[flat_index]
        depth = depth_by_flat_index[flat_index]
        if flat_index in node_by_flat_index:
            depth = int(node_by_flat_index[flat_index].depth)
        predictions.append(
            TerminationPrediction(
                flat_index=int(flat_index),
                depth=depth,
                path_probability=float(path_probability),
                child_edge_probability_sum=float(child_sum),
                termination_probability=float(
                    path_probability * (1.0 - clipped_child_sum)
                ),
            )
        )
    return tuple(predictions)


def rank_termination_predictions(
    predictions: tuple[TerminationPrediction, ...],
) -> tuple[TerminationPrediction, ...]:
    """Sort termination predictions by descending mass with stable tie-breaks."""

    return tuple(
        sorted(
            predictions,
            key=lambda item: (
                -item.termination_probability,
                item.depth,
                item.flat_index,
            ),
        )
    )


def termination_depth_distribution(
    predictions: tuple[TerminationPrediction, ...],
) -> dict[int, float]:
    """Aggregate predicted termination mass by tree depth."""

    distribution: dict[int, float] = {}
    for prediction in predictions:
        distribution[prediction.depth] = distribution.get(
            prediction.depth, 0.0
        ) + float(prediction.termination_probability)
    return distribution
