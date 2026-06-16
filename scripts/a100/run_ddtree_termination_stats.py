"""Measure whether DDTree termination mass predicts the first verifier stop.

For each primary DDTree verification call, every visited parent can be the first
termination point: the exact target next token is not among that parent's
selected DDTree children. The draft-model probability mass for that event is

    P(path_to_parent) * (1 - sum_child P(edge)).

This script runs ordinary primary-DDTree DFlash generation and records whether
that mass identifies the actual target termination node/depth.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from dflash.ddtree_termination import (
    compute_termination_predictions,
    rank_termination_predictions,
    termination_depth_distribution,
)
from dflash.model import (
    DFlashDraftModel,
    _tree_attention_mask,
    extract_context_feature,
    sample,
)
from dflash.residual_cache import compact_dynamic_cache
from dflash.residual_generate import build_residual_candidate_groups
from dflash.residual_surrogate import build_residual_ddtree
from dflash.residual_tree_verify import (
    linearize_residual_tree,
    parent_indices,
    walk_residual_tree,
)
from dflash.residual_types import VerificationAnchor
from scripts.a100.run_fair_residual_benchmark import (
    build_round_robin_plan,
    build_sequential_plan,
    load_prompts,
    make_input,
    parse_dataset_sample_counts,
    reset_gpu_state,
)

DEFAULT_DATASETS = ("alpaca", "humaneval", "mbpp", "gsm8k")


def _probability_bin(value: float) -> str:
    clipped = min(max(float(value), 0.0), 0.999999)
    index = int(clipped * 10)
    return f"{index / 10:.1f}-{(index + 1) / 10:.1f}"


def _new_bucket() -> dict[str, float | int]:
    return {
        "count": 0,
        "top1_exact": 0,
        "top3_exact": 0,
        "depth_exact": 0,
        "rank_sum": 0.0,
        "mrr_sum": 0.0,
        "actual_mass_sum": 0.0,
        "top_mass_sum": 0.0,
    }


def _add_bucket(
    bucket: dict[str, float | int],
    *,
    rank: int,
    depth_exact: bool,
    actual_mass: float,
    top_mass: float,
) -> None:
    bucket["count"] = int(bucket["count"]) + 1
    bucket["top1_exact"] = int(bucket["top1_exact"]) + int(rank == 1)
    bucket["top3_exact"] = int(bucket["top3_exact"]) + int(rank <= 3)
    bucket["depth_exact"] = int(bucket["depth_exact"]) + int(depth_exact)
    bucket["rank_sum"] = float(bucket["rank_sum"]) + float(rank)
    bucket["mrr_sum"] = float(bucket["mrr_sum"]) + 1.0 / float(rank)
    bucket["actual_mass_sum"] = float(bucket["actual_mass_sum"]) + float(actual_mass)
    bucket["top_mass_sum"] = float(bucket["top_mass_sum"]) + float(top_mass)


def _bucket_view(bucket: dict[str, float | int]) -> dict[str, float | int]:
    count = int(bucket["count"])
    if count == 0:
        return {
            **bucket,
            "top1_accuracy": 0.0,
            "top3_accuracy": 0.0,
            "depth_accuracy": 0.0,
        }
    return {
        **bucket,
        "top1_accuracy": int(bucket["top1_exact"]) / count,
        "top3_accuracy": int(bucket["top3_exact"]) / count,
        "depth_accuracy": int(bucket["depth_exact"]) / count,
        "mean_rank": float(bucket["rank_sum"]) / count,
        "mean_reciprocal_rank": float(bucket["mrr_sum"]) / count,
        "mean_actual_mass": float(bucket["actual_mass_sum"]) / count,
        "mean_top_mass": float(bucket["top_mass_sum"]) / count,
    }


class TerminationStats:
    def __init__(self) -> None:
        self.overall = _new_bucket()
        self.by_dataset: dict[str, dict[str, float | int]] = defaultdict(_new_bucket)
        self.by_top_mass_bin: dict[str, dict[str, float | int]] = defaultdict(
            _new_bucket
        )
        self.by_actual_depth: dict[str, dict[str, float | int]] = defaultdict(
            _new_bucket
        )
        self.actual_depth_hist: dict[str, int] = defaultdict(int)
        self.predicted_top_depth_hist: dict[str, int] = defaultdict(int)
        self.predicted_depth_mass_sum: dict[str, float] = defaultdict(float)
        self.actual_depth_sum = 0.0
        self.predicted_mean_depth_sum = 0.0
        self.depth_abs_error_sum = 0.0
        self.top_depth_abs_error_sum = 0.0
        self.mass_sum_values: list[float] = []
        self.nodes_per_tree: list[int] = []
        self.records = 0

    def add(
        self,
        *,
        dataset: str,
        rank: int,
        actual_depth: int,
        predicted_top_depth: int,
        predicted_mean_depth: float,
        actual_mass: float,
        top_mass: float,
        mass_sum: float,
        node_count: int,
        depth_distribution: dict[int, float],
    ) -> None:
        depth_exact = actual_depth == predicted_top_depth
        for bucket in (
            self.overall,
            self.by_dataset[dataset],
            self.by_top_mass_bin[_probability_bin(top_mass)],
            self.by_actual_depth[str(actual_depth)],
        ):
            _add_bucket(
                bucket,
                rank=rank,
                depth_exact=depth_exact,
                actual_mass=actual_mass,
                top_mass=top_mass,
            )
        self.records += 1
        self.actual_depth_hist[str(actual_depth)] += 1
        self.predicted_top_depth_hist[str(predicted_top_depth)] += 1
        for depth, probability in depth_distribution.items():
            self.predicted_depth_mass_sum[str(depth)] += float(probability)
        self.actual_depth_sum += float(actual_depth)
        self.predicted_mean_depth_sum += float(predicted_mean_depth)
        self.depth_abs_error_sum += abs(
            float(actual_depth) - float(predicted_mean_depth)
        )
        self.top_depth_abs_error_sum += abs(
            float(actual_depth) - float(predicted_top_depth)
        )
        self.mass_sum_values.append(float(mass_sum))
        self.nodes_per_tree.append(int(node_count))

    def to_json(self) -> dict[str, Any]:
        count = max(self.records, 1)
        return {
            "overall": _bucket_view(self.overall),
            "by_dataset": {
                key: _bucket_view(value)
                for key, value in sorted(self.by_dataset.items())
            },
            "by_top_mass_bin": {
                key: _bucket_view(value)
                for key, value in sorted(self.by_top_mass_bin.items())
            },
            "by_actual_depth": {
                key: _bucket_view(value)
                for key, value in sorted(
                    self.by_actual_depth.items(), key=lambda item: int(item[0])
                )
            },
            "actual_depth_hist": dict(
                sorted(self.actual_depth_hist.items(), key=lambda item: int(item[0]))
            ),
            "predicted_top_depth_hist": dict(
                sorted(
                    self.predicted_top_depth_hist.items(), key=lambda item: int(item[0])
                )
            ),
            "predicted_depth_mass_mean": {
                key: value / count
                for key, value in sorted(
                    self.predicted_depth_mass_sum.items(), key=lambda item: int(item[0])
                )
            },
            "mean_actual_depth": self.actual_depth_sum / count,
            "mean_predicted_depth": self.predicted_mean_depth_sum / count,
            "mean_abs_depth_error": self.depth_abs_error_sum / count,
            "mean_abs_top_depth_error": self.top_depth_abs_error_sum / count,
            "mean_mass_sum": statistics.mean(self.mass_sum_values)
            if self.mass_sum_values
            else 0.0,
            "max_mass_sum_deviation": max(
                (abs(value - 1.0) for value in self.mass_sum_values), default=0.0
            ),
            "mean_nodes_per_tree": statistics.mean(self.nodes_per_tree)
            if self.nodes_per_tree
            else 0.0,
            "trees": self.records,
        }


@torch.inference_mode()
def run_prompt(
    *,
    dataset: str,
    sample_index: int,
    prompt: str,
    draft: DFlashDraftModel,
    target: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    stats: TerminationStats,
    records_file,
    max_new_tokens: int,
    block_size: int,
    tree_width: int,
    budget: int,
    temperature: float,
) -> int:
    input_ids = make_input(tokenizer, prompt)
    num_input_tokens = int(input_ids.shape[1])
    max_length = num_input_tokens + max_new_tokens
    mask_token_id = int(draft.mask_token_id)
    output_ids = torch.full(
        (1, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=target.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
        output.logits, temperature
    )
    target_hidden = extract_context_feature(
        output.hidden_states, draft.target_layer_ids
    )

    start = num_input_tokens
    tree_index = 0
    stop_token_ids = (
        {int(tokenizer.eos_token_id)} if tokenizer.eos_token_id is not None else set()
    )
    tree_output_dtype = next(target.parameters()).dtype

    while start < max_length:
        bs = min(block_size, max_length - start + 1)
        if bs <= 1:
            break
        block_output_ids = output_ids[:, start : start + bs].clone()
        noise_embedding = target.model.embed_tokens(block_output_ids)
        draft_logits = target.lm_head(
            draft(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[
                    :, past_key_values_draft.get_seq_length() : start + bs
                ],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )[:, 1 - bs :, :]
        )
        past_key_values_draft.crop(start)

        tree_logits = draft_logits.float()
        topk = min(int(tree_width), int(tree_logits.shape[-1]))
        top_logits, top_token_ids = torch.topk(tree_logits, k=topk, dim=-1)
        log_z = torch.logsumexp(tree_logits, dim=-1, keepdim=True)
        top_probabilities = torch.exp(top_logits - log_z)
        candidate_groups = build_residual_candidate_groups(
            top_token_ids_by_block=tuple(
                tuple(int(x) for x in row)
                for row in top_token_ids[0].detach().cpu().tolist()
            ),
            top_probabilities_by_block=tuple(
                tuple(float(x) for x in row)
                for row in top_probabilities[0].detach().cpu().tolist()
            ),
            start_position=int(start),
            first_block_index=1,
        )
        anchor_token_id = int(output_ids[0, start].detach().cpu().item())
        tree = build_residual_ddtree(
            VerificationAnchor(token_id=anchor_token_id, position=int(start)),
            candidate_groups,
            budget=budget,
        )
        nodes = linearize_residual_tree(tree)
        if not nodes:
            break

        predictions = compute_termination_predictions(nodes)
        ranked = rank_termination_predictions(predictions)
        prediction_by_flat_index = {
            prediction.flat_index: prediction for prediction in predictions
        }
        rank_by_flat_index = {
            prediction.flat_index: index + 1 for index, prediction in enumerate(ranked)
        }
        depth_distribution = termination_depth_distribution(predictions)
        mass_sum = sum(prediction.termination_probability for prediction in predictions)
        predicted_mean_depth = sum(
            prediction.depth * prediction.termination_probability
            for prediction in predictions
        ) / max(mass_sum, 1e-12)
        predicted_top_depth = int(ranked[0].depth)

        past_key_values_target.crop(start)
        tree_ids = torch.tensor(
            [anchor_token_id, *(node.token_id for node in nodes)],
            dtype=torch.long,
            device=target.device,
        ).unsqueeze(0)
        tree_position_ids = torch.tensor(
            [int(start), *(node.position for node in nodes)],
            dtype=torch.long,
            device=target.device,
        ).unsqueeze(0)
        tree_output = target(
            tree_ids,
            position_ids=tree_position_ids,
            attention_mask=_tree_attention_mask(
                parent_indices(nodes),
                past_length=int(start),
                device=target.device,
                dtype=tree_output_dtype,
            ),
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )
        tree_posterior = sample(tree_output.logits, temperature)
        tree_walk = walk_residual_tree(
            nodes,
            target_token_ids_by_flat_index=tuple(
                int(token_id) for token_id in tree_posterior[0].detach().cpu().tolist()
            ),
        )
        actual_flat_index = (
            0
            if not tree_walk.accepted_nodes
            else int(tree_walk.accepted_nodes[-1].flat_index)
        )
        actual_depth = int(tree_walk.accepted_count)
        actual_prediction = prediction_by_flat_index[actual_flat_index]
        actual_rank = int(rank_by_flat_index[actual_flat_index])
        top_prediction = ranked[0]
        stats.add(
            dataset=dataset,
            rank=actual_rank,
            actual_depth=actual_depth,
            predicted_top_depth=predicted_top_depth,
            predicted_mean_depth=predicted_mean_depth,
            actual_mass=float(actual_prediction.termination_probability),
            top_mass=float(top_prediction.termination_probability),
            mass_sum=float(mass_sum),
            node_count=len(nodes),
            depth_distribution=depth_distribution,
        )
        records_file.write(
            json.dumps(
                {
                    "dataset": dataset,
                    "sample_index": sample_index,
                    "tree_index": tree_index,
                    "start": int(start),
                    "node_count": len(nodes),
                    "actual_flat_index": actual_flat_index,
                    "actual_depth": actual_depth,
                    "actual_rank": actual_rank,
                    "actual_mass": float(actual_prediction.termination_probability),
                    "top_flat_index": int(top_prediction.flat_index),
                    "top_depth": int(top_prediction.depth),
                    "top_mass": float(top_prediction.termination_probability),
                    "predicted_mean_depth": float(predicted_mean_depth),
                    "mass_sum": float(mass_sum),
                }
            )
            + "\n"
        )

        accepted_indices = [0, *(node.flat_index for node in tree_walk.accepted_nodes)]
        accepted_index_tensor = torch.tensor(
            accepted_indices, dtype=torch.long, device=tree_ids.device
        )
        accepted_tokens = tree_ids.index_select(1, accepted_index_tensor)
        output_ids[:, start : start + accepted_tokens.shape[1]] = accepted_tokens
        mismatch_position = start + accepted_tokens.shape[1]
        output_ids[:, mismatch_position] = int(tree_walk.mismatch_token_id)
        compact_dynamic_cache(past_key_values_target, int(start), accepted_indices)
        target_hidden = extract_context_feature(
            tree_output.hidden_states, draft.target_layer_ids
        ).index_select(1, accepted_index_tensor)
        start = mismatch_position
        tree_index += 1

        if stop_token_ids and any(
            int(token_id) in stop_token_ids
            for token_id in output_ids[0, num_input_tokens : start + 1]
            .detach()
            .cpu()
            .tolist()
        ):
            break

    return tree_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--dataset-samples", nargs="*", default=None)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--round-robin-datasets", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--tree-width", type=int, default=5)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--target-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-model", default="z-lab/Qwen3-8B-DFlash-b16")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%d %H:%M:%S %Z")

    print("loading models", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype="auto",
        device_map="cuda:0",
    ).eval()
    draft = DFlashDraftModel.from_pretrained(
        args.draft_model,
        torch_dtype="auto",
        device_map="cuda:0",
    ).eval()

    sample_counts = parse_dataset_sample_counts(
        args.datasets,
        args.dataset_samples,
        default=args.max_samples,
    )
    prompts_by_dataset = {
        dataset: load_prompts(
            dataset, max_samples=sample_counts[dataset], seed=args.seed
        )
        for dataset in args.datasets
    }
    plan = (
        build_round_robin_plan(prompts_by_dataset, args.datasets)
        if args.round_robin_datasets
        else build_sequential_plan(prompts_by_dataset, args.datasets)
    )

    stats = TerminationStats()
    prompt_tree_counts: dict[str, int] = defaultdict(int)
    records_path = output_dir / "termination_records.jsonl"
    with records_path.open("w") as records_file:
        for dataset, sample_index, prompt, pair_index in plan:
            reset_gpu_state()
            print(
                f"[{pair_index + 1}/{len(plan)}] {dataset}#{sample_index}", flush=True
            )
            tree_count = run_prompt(
                dataset=dataset,
                sample_index=sample_index,
                prompt=prompt,
                draft=draft,
                target=target,
                tokenizer=tokenizer,
                stats=stats,
                records_file=records_file,
                max_new_tokens=args.max_new_tokens,
                block_size=args.block_size,
                tree_width=args.tree_width,
                budget=args.budget,
                temperature=args.temperature,
            )
            prompt_tree_counts[dataset] += tree_count

    summary = {
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "args": vars(args),
        "prompt_count": len(plan),
        "prompt_tree_counts": dict(sorted(prompt_tree_counts.items())),
        "stats": stats.to_json(),
    }
    summary_path = output_dir / "termination_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
