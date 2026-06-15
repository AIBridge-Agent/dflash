# pyright: reportMissingImports=none
import ast
from pathlib import Path

from dflash.residual_generate import (
    build_residual_candidate_groups,
    build_residual_edge_records,
    build_residual_opportunity,
)


MODEL_PATH = Path(__file__).resolve().parents[1] / "dflash" / "model.py"


def test_build_residual_opportunity_from_dflash_block_accounting() -> None:
    opportunity = build_residual_opportunity(
        block_token_ids=(100, 11, 22, 33, 44),
        block_probabilities=(1.0, 0.9, 0.8, 0.95, 0.9),
        start_position=10,
        acceptance_length=1,
        verifier_mismatch_token_id=99,
        current_accepted=1,
        draft_seconds=1.0,
        target_seconds=1.0,
        residual_budget=1,
    )

    assert opportunity.should_run is True
    assert opportunity.drafter_calls == 0
    assert opportunity.estimated_residual_gain > opportunity.path.probability_product
    assert opportunity.path.anchor.token_id == 99


def test_build_residual_opportunity_uses_residual_ddtree_when_candidate_groups_exist() -> (
    None
):
    groups = build_residual_candidate_groups(
        top_token_ids_by_block=((33, 34), (44, 45)),
        top_probabilities_by_block=((0.95, 0.5), (0.9, 0.4)),
        start_position=10,
        first_block_index=3,
    )

    opportunity = build_residual_opportunity(
        block_token_ids=(100, 11, 22, 33, 44),
        block_probabilities=(1.0, 0.9, 0.8, 0.95, 0.9),
        start_position=10,
        acceptance_length=1,
        verifier_mismatch_token_id=99,
        current_accepted=1,
        draft_seconds=1.0,
        target_seconds=1.0,
        residual_budget=4,
        residual_candidate_groups=groups,
    )

    assert opportunity.residual_tree is not None
    assert opportunity.tree_nodes == 4
    assert (
        opportunity.estimated_residual_gain
        == opportunity.residual_tree.expected_accept_length
    )
    assert opportunity.should_run is True


def test_build_residual_edge_records_observes_accepts_and_first_reject() -> None:
    opportunity = build_residual_opportunity(
        block_token_ids=(100, 11, 22, 33, 44),
        block_probabilities=(1.0, 0.9, 0.8, 0.7, 0.6),
        start_position=10,
        acceptance_length=1,
        verifier_mismatch_token_id=99,
        current_accepted=1,
        draft_seconds=1.0,
        target_seconds=1.0,
        residual_budget=1,
    )

    records = build_residual_edge_records(
        opportunity.path, residual_acceptance_length=1
    )

    assert [record["outcome"] for record in records] == ["accepted", "first_rejected"]
    assert [record["edge_probability"] for record in records] == [0.7, 0.6]
    assert records[0]["accepted"] is True
    assert records[1]["first_reject"] is True


def test_build_residual_opportunity_passes_explicit_residual_target_cost() -> None:
    opportunity = build_residual_opportunity(
        block_token_ids=(100, 11, 22, 33),
        block_probabilities=(1.0, 0.9, 0.8, 0.7),
        start_position=10,
        acceptance_length=1,
        verifier_mismatch_token_id=99,
        current_accepted=1,
        draft_seconds=1.0,
        target_seconds=2.0,
        residual_budget=1,
        residual_target_seconds=0.5,
    )

    assert opportunity.gate.target_seconds == 2.0
    assert opportunity.gate.residual_target_seconds == 0.5


def test_build_residual_opportunity_can_gate_off() -> None:
    opportunity = build_residual_opportunity(
        block_token_ids=(100, 11, 22, 33),
        block_probabilities=(1.0, 0.9, 0.8, 0.01),
        start_position=10,
        acceptance_length=1,
        verifier_mismatch_token_id=99,
        current_accepted=4,
        draft_seconds=1.0,
        target_seconds=1.0,
        residual_budget=1,
    )

    assert opportunity.should_run is False


def test_dflash_generate_exposes_residual_opt_in_flags() -> None:
    module = ast.parse(MODEL_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "dflash_generate"
    )
    arg_names = [arg.arg for arg in function.args.args]

    assert "enable_residual" in arg_names
    assert "residual_budget" in arg_names
    assert "residual_draft_seconds" in arg_names
    assert "residual_target_seconds" in arg_names
    assert "residual_gain_scale" in arg_names
    assert "residual_min_gain" in arg_names
    assert "residual_min_margin" in arg_names


def test_dflash_generate_does_not_swallow_domain_errors() -> None:
    source = MODEL_PATH.read_text(encoding="utf-8")

    assert "except NoResidualOpportunity" in source
    assert "except (NoResidualOpportunity, ResidualDFlashError)" not in source


def test_dflash_generate_uses_residual_tail_without_second_draft_call() -> None:
    source = MODEL_PATH.read_text(encoding="utf-8")

    assert "build_residual_opportunity" in source
    assert (
        "extract_residual_path" not in source
    )  # keep model.py using the accounting seam
    assert "residual_ids" in source
    assert "residual_runs" in source
    assert "residual_gate_records" in source
    assert "residual_edge_records" in source
    assert "build_residual_edge_records" in source
    assert "linearize_residual_tree" in source
    assert "walk_residual_tree" in source
    assert "_tree_attention_mask" in source
    assert "compact_dynamic_cache" in source
    assert "verification_mode" in source
    assert "replay_output" not in source
    assert "measured_draft_seconds" in source
    assert "measured_target_seconds" in source
    assert "target_seconds_sum" in source
    assert "target_seconds_count" in source
    assert "residual_target_seconds_estimate" in source
    assert "residual_target_seconds" in source
    assert "residual_gain_scale" in source
    assert "residual_min_gain" in source
    assert "residual_min_margin" in source
    assert "effective_estimated_residual_gain" in source
    assert "prefix_hidden" in source
    assert "residual_hidden" in source
    assert "torch.cat" in source
    assert "model(" in source  # original draft call still exists
    assert "enable_residual" in source
