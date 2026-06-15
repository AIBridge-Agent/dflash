# pyright: reportMissingImports=none
"""Residual DDTree verification planning helpers."""

from __future__ import annotations

from dataclasses import dataclass

from .residual_surrogate import ResidualDDTree
from .residual_types import ResidualCandidate


@dataclass(frozen=True)
class ResidualTreeNode:
    """One flattened residual DDTree node.

    `flat_index` is the token index in the tree-verification target call where
    index 0 is the verifier anchor and node indices start at 1.
    """

    flat_index: int
    parent_index: int
    candidate: ResidualCandidate
    path_tokens: tuple[int, ...]

    @property
    def token_id(self) -> int:
        return self.candidate.token_id

    @property
    def position(self) -> int:
        return self.candidate.position

    @property
    def probability(self) -> float:
        return self.candidate.probability

    @property
    def depth(self) -> int:
        return len(self.path_tokens)


def linearize_residual_tree(tree: ResidualDDTree) -> tuple[ResidualTreeNode, ...]:
    """Return unique DDTree nodes in deterministic topological order."""

    paths_by_signature = {
        tuple(candidate.token_id for candidate in path.candidates): path
        for path in tree.nodes
    }
    ordered_signatures = sorted(paths_by_signature, key=lambda sig: (len(sig), sig))
    signature_to_flat_index: dict[tuple[int, ...], int] = {}
    nodes: list[ResidualTreeNode] = []

    for signature in ordered_signatures:
        if not signature:
            continue
        parent_signature = signature[:-1]
        if parent_signature and parent_signature not in signature_to_flat_index:
            continue
        parent_index = signature_to_flat_index.get(parent_signature, 0)
        path = paths_by_signature[signature]
        flat_index = len(nodes) + 1
        signature_to_flat_index[signature] = flat_index
        nodes.append(
            ResidualTreeNode(
                flat_index=flat_index,
                parent_index=parent_index,
                candidate=path.candidates[-1],
                path_tokens=signature,
            )
        )
    return tuple(nodes)


def parent_indices(nodes: tuple[ResidualTreeNode, ...]) -> tuple[int | None, ...]:
    """Return parent indices for flattened anchor+node inputs."""

    return (None, *(node.parent_index for node in nodes))


def children_by_parent(
    nodes: tuple[ResidualTreeNode, ...],
) -> dict[int, tuple[ResidualTreeNode, ...]]:
    """Group flattened nodes by parent flat index."""

    grouped: dict[int, list[ResidualTreeNode]] = {}
    for node in nodes:
        grouped.setdefault(node.parent_index, []).append(node)
    return {
        parent: tuple(
            sorted(children, key=lambda node: (-node.probability, node.token_id))
        )
        for parent, children in grouped.items()
    }


@dataclass(frozen=True)
class ResidualTreeWalkResult:
    """Accepted path from exact target logits over a residual DDTree."""

    accepted_nodes: tuple[ResidualTreeNode, ...]
    mismatch_token_id: int
    edge_records: tuple[dict[str, bool | float | int | str], ...]

    @property
    def accepted_count(self) -> int:
        return len(self.accepted_nodes)


def walk_residual_tree(
    nodes: tuple[ResidualTreeNode, ...],
    *,
    target_token_ids_by_flat_index: tuple[int, ...],
) -> ResidualTreeWalkResult:
    """Walk the residual tree using exact target next-token choices.

    At each visited parent, the target next token chooses at most one child. If
    no child matches, the walk stops and the visited parent's target token is the
    mismatch token. Accepted edges and the first rejected frontier are logged for
    edge-probability calibration.
    """

    children = children_by_parent(nodes)
    parent_index = 0
    accepted: list[ResidualTreeNode] = []
    edge_records: list[dict[str, bool | float | int | str]] = []

    while True:
        if parent_index >= len(target_token_ids_by_flat_index):
            raise IndexError("target_token_ids_by_flat_index does not cover parent")
        target_token_id = int(target_token_ids_by_flat_index[parent_index])
        candidates = children.get(parent_index, ())
        matched = next(
            (
                candidate
                for candidate in candidates
                if candidate.token_id == target_token_id
            ),
            None,
        )
        if matched is None:
            for candidate in candidates:
                edge_records.append(
                    _edge_record(
                        candidate,
                        accepted=False,
                        first_reject=True,
                        target_token_id=target_token_id,
                    )
                )
            return ResidualTreeWalkResult(
                accepted_nodes=tuple(accepted),
                mismatch_token_id=target_token_id,
                edge_records=tuple(edge_records),
            )

        edge_records.append(
            _edge_record(
                matched,
                accepted=True,
                first_reject=False,
                target_token_id=target_token_id,
            )
        )
        accepted.append(matched)
        parent_index = matched.flat_index


def _edge_record(
    node: ResidualTreeNode,
    *,
    accepted: bool,
    first_reject: bool,
    target_token_id: int,
) -> dict[str, bool | float | int | str]:
    return {
        "edge_index": node.depth - 1,
        "flat_index": node.flat_index,
        "parent_index": node.parent_index,
        "position": node.position,
        "source_block_position": node.candidate.source_block_position,
        "token_id": node.token_id,
        "target_token_id": target_token_id,
        "edge_probability": float(node.probability),
        "accepted": accepted,
        "first_reject": first_reject,
        "outcome": "accepted" if accepted else "first_rejected",
    }
