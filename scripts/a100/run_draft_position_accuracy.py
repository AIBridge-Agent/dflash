"""Measure single-path DFlash draft token accuracy against final output.

This diagnostic answers: after the first verifier rejection in a normal DFlash
block, how often would later draft tokens have matched the final exact target
output at the same absolute positions anyway?

The script intentionally runs ordinary single-path DFlash verification
(block_size=16, no DDTree/residual) and aggregates draft-token correctness by
block offset, first-reject offset, and distance after rejection.
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

from dflash.model import DFlashDraftModel, extract_context_feature, sample
from scripts.a100.run_fair_residual_benchmark import (
    build_round_robin_plan,
    build_sequential_plan,
    load_prompts,
    make_input,
    parse_dataset_sample_counts,
    reset_gpu_state,
)

FULL_BALANCED_DATASETS = (
    "gsm8k",
    "math500",
    "aime25",
    "humaneval",
    "mbpp",
    "livecodebench",
    "mt_bench",
    "alpaca",
)
FULL_BALANCED_COUNTS = {
    "gsm8k": 110,
    "math500": 110,
    "aime25": 30,
    "humaneval": 110,
    "mbpp": 110,
    "livecodebench": 110,
    "mt_bench": 80,
    "alpaca": 110,
}

Bucket = dict[str, float | int]


def _new_bucket() -> Bucket:
    return {
        "attempts": 0,
        "comparable": 0,
        "correct": 0,
        "prob_sum": 0.0,
        "correct_prob_sum": 0.0,
        "incorrect_prob_sum": 0.0,
    }


def _add_bucket(bucket: Bucket, *, comparable: bool, correct: bool, probability: float) -> None:
    bucket["attempts"] = int(bucket["attempts"]) + 1
    bucket["prob_sum"] = float(bucket["prob_sum"]) + float(probability)
    if comparable:
        bucket["comparable"] = int(bucket["comparable"]) + 1
        if correct:
            bucket["correct"] = int(bucket["correct"]) + 1
            bucket["correct_prob_sum"] = float(bucket["correct_prob_sum"]) + float(probability)
        else:
            bucket["incorrect_prob_sum"] = float(bucket["incorrect_prob_sum"]) + float(probability)


def _merge_bucket(dst: Bucket, src: Bucket) -> None:
    for key in dst:
        if isinstance(dst[key], int):
            dst[key] = int(dst[key]) + int(src.get(key, 0))
        else:
            dst[key] = float(dst[key]) + float(src.get(key, 0.0))


def _bucket_view(bucket: Bucket) -> dict[str, float | int]:
    comparable = int(bucket["comparable"])
    correct = int(bucket["correct"])
    attempts = int(bucket["attempts"])
    prob_sum = float(bucket["prob_sum"])
    incorrect = comparable - correct
    return {
        **bucket,
        "accuracy": correct / comparable if comparable else 0.0,
        "coverage": comparable / attempts if attempts else 0.0,
        "mean_probability": prob_sum / attempts if attempts else 0.0,
        "mean_correct_probability": (
            float(bucket["correct_prob_sum"]) / correct if correct else 0.0
        ),
        "mean_incorrect_probability": (
            float(bucket["incorrect_prob_sum"]) / incorrect if incorrect else 0.0
        ),
    }


def _serialise_group(group: dict[str, Bucket]) -> dict[str, dict[str, float | int]]:
    return {key: _bucket_view(group[key]) for key in sorted(group, key=_sort_key)}


def _sort_key(value: str) -> tuple[int, str]:
    try:
        return (0, f"{int(value):08d}")
    except ValueError:
        return (1, value)


def _empty_stats() -> dict[str, Any]:
    return {
        "overall": _new_bucket(),
        "by_phase": defaultdict(_new_bucket),
        "by_offset": defaultdict(_new_bucket),
        "by_reject_offset": defaultdict(_new_bucket),
        "by_distance_after_reject": defaultdict(_new_bucket),
        "by_acceptance_length": defaultdict(_new_bucket),
        "first_reject_offset_hist": defaultdict(int),
        "acceptance_length_hist": defaultdict(int),
        "decode_seconds": [],
        "output_tokens": [],
        "rounds": [],
    }


def _record_attempt(
    stats: dict[str, Any],
    *,
    offset: int,
    acceptance_length: int,
    block_size: int,
    comparable: bool,
    correct: bool,
    probability: float,
) -> None:
    reject_offset = acceptance_length + 1
    if offset <= acceptance_length:
        phase = "accepted_prefix"
    elif offset == reject_offset:
        phase = "first_reject"
    else:
        phase = "after_reject"

    _add_bucket(stats["overall"], comparable=comparable, correct=correct, probability=probability)
    _add_bucket(
        stats["by_phase"][phase],
        comparable=comparable,
        correct=correct,
        probability=probability,
    )
    _add_bucket(
        stats["by_offset"][str(offset)],
        comparable=comparable,
        correct=correct,
        probability=probability,
    )
    _add_bucket(
        stats["by_acceptance_length"][str(acceptance_length)],
        comparable=comparable,
        correct=correct,
        probability=probability,
    )
    if reject_offset < block_size:
        _add_bucket(
            stats["by_reject_offset"][str(reject_offset)],
            comparable=comparable,
            correct=correct,
            probability=probability,
        )
    if offset >= reject_offset:
        _add_bucket(
            stats["by_distance_after_reject"][str(offset - reject_offset)],
            comparable=comparable,
            correct=correct,
            probability=probability,
        )


def _merge_stats(dst: dict[str, Any], src: dict[str, Any]) -> None:
    _merge_bucket(dst["overall"], src["overall"])
    for group_name in (
        "by_phase",
        "by_offset",
        "by_reject_offset",
        "by_distance_after_reject",
        "by_acceptance_length",
    ):
        for key, bucket in src[group_name].items():
            _merge_bucket(dst[group_name][key], bucket)
    for hist_name in ("first_reject_offset_hist", "acceptance_length_hist"):
        for key, value in src[hist_name].items():
            dst[hist_name][key] += value
    dst["decode_seconds"].extend(src["decode_seconds"])
    dst["output_tokens"].extend(src["output_tokens"])
    dst["rounds"].extend(src["rounds"])


def _stats_view(stats: dict[str, Any]) -> dict[str, Any]:
    output_tokens = [int(x) for x in stats["output_tokens"]]
    decode_seconds = [float(x) for x in stats["decode_seconds"]]
    rounds = [int(x) for x in stats["rounds"]]
    total_tokens = sum(output_tokens)
    total_decode = sum(decode_seconds)
    return {
        "overall": _bucket_view(stats["overall"]),
        "by_phase": _serialise_group(stats["by_phase"]),
        "by_offset": _serialise_group(stats["by_offset"]),
        "by_reject_offset": _serialise_group(stats["by_reject_offset"]),
        "by_distance_after_reject": _serialise_group(stats["by_distance_after_reject"]),
        "by_acceptance_length": _serialise_group(stats["by_acceptance_length"]),
        "first_reject_offset_hist": dict(sorted(stats["first_reject_offset_hist"].items(), key=lambda item: _sort_key(item[0]))),
        "acceptance_length_hist": dict(sorted(stats["acceptance_length_hist"].items(), key=lambda item: _sort_key(item[0]))),
        "num_prompts": len(output_tokens),
        "total_output_tokens": total_tokens,
        "total_decode_seconds": total_decode,
        "aggregate_tps": total_tokens / total_decode if total_decode else 0.0,
        "mean_tpot_ms": (
            statistics.fmean(
                sec / max(tok, 1) for sec, tok in zip(decode_seconds, output_tokens, strict=True)
            )
            * 1000.0
            if output_tokens
            else 0.0
        ),
        "mean_rounds": statistics.fmean(rounds) if rounds else 0.0,
    }


@torch.inference_mode()
def collect_prompt_accuracy(
    *,
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    stop_token_ids: list[int] | None,
    temperature: float,
    block_size: int,
    mask_token_id: int,
) -> dict[str, Any]:
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    output_ids = torch.full(
        (1, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=target.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)
    stop_token_ids_tensor = (
        None if stop_token_ids is None else torch.tensor(stop_token_ids, device=target.device)
    )
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    prefill_start = time.perf_counter()
    target_prefill = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )
    torch.cuda.synchronize()
    time_to_first_token = time.perf_counter() - prefill_start

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
        target_prefill.logits, temperature
    )
    target_hidden = extract_context_feature(target_prefill.hidden_states, model.target_layer_ids)

    decode_start = time.perf_counter()
    start = num_input_tokens
    acceptance_lengths: list[int] = []
    draft_attempts: list[dict[str, int | float]] = []

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]

        noise_embedding = target.model.embed_tokens(block_output_ids)
        draft_logits = target.lm_head(
            model(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[
                    :, past_key_values_draft.get_seq_length() : start + block_size
                ],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )[:, 1 - block_size :, :]
        )
        past_key_values_draft.crop(start)
        sampled_draft_ids = sample(draft_logits, temperature)
        block_output_ids[:, 1:] = sampled_draft_ids
        draft_probs = torch.softmax(draft_logits.float(), dim=-1)
        sampled_probs = torch.gather(
            draft_probs, -1, sampled_draft_ids.unsqueeze(-1)
        ).squeeze(-1)

        target_output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )
        posterior = sample(target_output.logits, temperature)
        acceptance_length = int(
            (block_output_ids[:, 1:] == posterior[:, :-1])
            .cumprod(dim=1)
            .sum(dim=1)[0]
            .item()
        )
        acceptance_lengths.append(acceptance_length + 1)

        for offset in range(1, block_size):
            draft_attempts.append(
                {
                    "abs_pos": int(start + offset),
                    "offset": int(offset),
                    "acceptance_length": int(acceptance_length),
                    "token_id": int(sampled_draft_ids[0, offset - 1].detach().cpu().item()),
                    "probability": float(sampled_probs[0, offset - 1].detach().cpu().item()),
                }
            )

        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[
            :, : acceptance_length + 1
        ]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
        start += acceptance_length + 1
        past_key_values_target.crop(start)
        target_hidden = extract_context_feature(
            target_output.hidden_states, model.target_layer_ids
        )[:, : acceptance_length + 1, :]

        if stop_token_ids_tensor is not None:
            generated = output_ids[:, num_input_tokens : start + 1]
            if torch.isin(generated[0], stop_token_ids_tensor).any():
                break

    torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - decode_start

    output_ids = output_ids[:, : min(start + 1, max_length)]
    if stop_token_ids_tensor is not None:
        stop_token_indices = torch.isin(
            output_ids[0][num_input_tokens:], stop_token_ids_tensor
        ).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + int(stop_token_indices[0]) + 1]

    final_tokens = output_ids[0].detach().cpu().tolist()
    prompt_stats = _empty_stats()
    for attempt in draft_attempts:
        abs_pos = int(attempt["abs_pos"])
        comparable = abs_pos < len(final_tokens)
        correct = comparable and int(attempt["token_id"]) == int(final_tokens[abs_pos])
        _record_attempt(
            prompt_stats,
            offset=int(attempt["offset"]),
            acceptance_length=int(attempt["acceptance_length"]),
            block_size=block_size,
            comparable=comparable,
            correct=bool(correct),
            probability=float(attempt["probability"]),
        )

    for accepted in acceptance_lengths:
        acceptance_length = accepted - 1
        prompt_stats["acceptance_length_hist"][str(acceptance_length)] += 1
        reject_offset = acceptance_length + 1
        if reject_offset < block_size:
            prompt_stats["first_reject_offset_hist"][str(reject_offset)] += 1
    prompt_stats["decode_seconds"].append(decode_seconds)
    prompt_stats["output_tokens"].append(max(len(final_tokens) - num_input_tokens, 0))
    prompt_stats["rounds"].append(len(acceptance_lengths))

    return {
        "output_tokens": max(len(final_tokens) - num_input_tokens, 0),
        "decode_seconds": decode_seconds,
        "time_to_first_token": time_to_first_token,
        "acceptance_lengths": acceptance_lengths,
        "prompt_stats": prompt_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(FULL_BALANCED_DATASETS))
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument(
        "--dataset-samples",
        nargs="*",
        default=[f"{name}={count}" for name, count in FULL_BALANCED_COUNTS.items()],
    )
    parser.add_argument("--round-robin-datasets", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-model", default="z-lab/Qwen3-8B-DFlash-b16")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/draft_position_accuracy")
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    jsonl_path = args.output_dir / f"draft_position_accuracy_{run_id}.jsonl"
    summary_path = args.output_dir / f"draft_position_accuracy_{run_id}_summary.json"

    print("loading models")
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    ).eval()
    draft = DFlashDraftModel.from_pretrained(
        args.draft_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)

    dataset_sample_counts = parse_dataset_sample_counts(
        args.datasets, args.dataset_samples, default=args.max_samples
    )
    prompts_by_dataset = {
        dataset_name: load_prompts(
            dataset_name,
            max_samples=dataset_sample_counts[dataset_name],
            seed=args.seed,
        )
        for dataset_name in args.datasets
    }
    prompt_plan = (
        build_round_robin_plan(prompts_by_dataset, args.datasets)
        if args.round_robin_datasets
        else build_sequential_plan(prompts_by_dataset, args.datasets)
    )

    overall = _empty_stats()
    by_dataset: dict[str, dict[str, Any]] = defaultdict(_empty_stats)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for dataset_name, sample_index, prompt, plan_index in prompt_plan:
            print(f"dataset={dataset_name} sample={sample_index} plan={plan_index}")
            input_ids = make_input(tokenizer, prompt)
            reset_gpu_state()
            result = collect_prompt_accuracy(
                model=draft,
                target=target,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
                block_size=args.block_size,
                mask_token_id=draft.mask_token_id,
            )
            prompt_stats = result.pop("prompt_stats")
            _merge_stats(overall, prompt_stats)
            _merge_stats(by_dataset[dataset_name], prompt_stats)
            row = {
                "dataset": dataset_name,
                "sample_index": sample_index,
                "plan_index": plan_index,
                "max_new_tokens": args.max_new_tokens,
                "block_size": args.block_size,
                **result,
                "stats": _stats_view(prompt_stats),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"  output_tokens={row['output_tokens']} rounds={len(row['acceptance_lengths'])} "
                f"post_reject_acc="
                f"{row['stats']['by_phase'].get('after_reject', {}).get('accuracy', 0.0):.3f}"
            )
            del input_ids, result, prompt_stats
            reset_gpu_state()

    summary = {
        "run_id": run_id,
        "jsonl_path": str(jsonl_path),
        "datasets": args.datasets,
        "dataset_sample_counts": dataset_sample_counts,
        "round_robin_datasets": bool(args.round_robin_datasets),
        "max_new_tokens": args.max_new_tokens,
        "block_size": args.block_size,
        "temperature": args.temperature,
        "overall": _stats_view(overall),
        "by_dataset": {
            dataset_name: _stats_view(stats)
            for dataset_name, stats in sorted(by_dataset.items())
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"jsonl={jsonl_path}")
    print(f"summary={summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
