"""Prompt-paired fair benchmark for DFlash residual verification on A100.

For each prompt, this runner executes dflash_block16 and
 dflash_block16_residual sequentially, resets CUDA allocator state before each
policy, and alternates policy order by prompt index to reduce order bias.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from dflash.model import DFlashDraftModel, dflash_generate

DATASETS: dict[str, dict[str, Any]] = {
    "gsm8k": {
        "load_args": ("openai/gsm8k", "main"),
        "load_kwargs": {"split": "test"},
        "format": lambda x: (
            f"{x['question']}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        ),
    },
    "math500": {
        "load_args": ("HuggingFaceH4/MATH-500",),
        "load_kwargs": {"split": "test"},
        "format": lambda x: (
            f"{x['problem']}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        ),
    },
    "humaneval": {
        "load_args": ("openai/openai_humaneval",),
        "load_kwargs": {"split": "test"},
        "format": lambda x: (
            "Write a solution to the following problem and make sure that it passes the tests:\n"
            f"```python\n{x['prompt']}\n```"
        ),
    },
    "mbpp": {
        "load_args": ("google-research-datasets/mbpp", "sanitized"),
        "load_kwargs": {"split": "test"},
        "format": lambda x: x["prompt"],
    },
    "mt-bench": {
        "load_args": ("HuggingFaceH4/mt_bench_prompts",),
        "load_kwargs": {"split": "train"},
        "format": lambda x: x["prompt"],
        "multi_turn": True,
    },
    "mt_bench": {
        "load_args": ("HuggingFaceH4/mt_bench_prompts",),
        "load_kwargs": {"split": "train"},
        "format": lambda x: x["prompt"],
        "multi_turn": True,
    },
    "aime25": {
        "load_args": ("math-ai/aime25",),
        "load_kwargs": {"split": "test"},
        "format": lambda x: (
            f"{x['problem']}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        ),
    },
    "livecodebench": {
        "load_args": ("livecodebench/code_generation_lite", "release_latest"),
        "load_kwargs": {"split": "test", "trust_remote_code": True},
        "format": lambda x: (
            "Write a complete solution for the following programming problem.\n"
            f"Title: {x['question_title']}\n\n{x['question_content']}\n\n"
            f"Starter code:\n```python\n{x['starter_code']}\n```"
        ),
    },
    "alpaca": {
        "load_args": ("tatsu-lab/alpaca",),
        "load_kwargs": {"split": "train"},
        "format": lambda x: (
            f"{x['instruction']}\n\nInput:\n{x['input']}"
            if x.get("input")
            else x["instruction"]
        ),
    },
}

DEFAULT_DATASETS = ("gsm8k", "math500", "humaneval", "mbpp", "mt-bench")
POLICIES = ("dflash_block16", "dflash_block16_residual")


def reset_gpu_state() -> None:
    """Reset transient CUDA allocator/cache state between policy runs."""

    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def load_prompts(dataset_name: str, *, max_samples: int, seed: int) -> list[str]:
    cfg = DATASETS[dataset_name]
    dataset = list(load_dataset(*cfg["load_args"], **cfg["load_kwargs"]))
    rng = random.Random(seed)
    rng.shuffle(dataset)
    prompts: list[str] = []
    for row in dataset[:max_samples]:
        formatted = cfg["format"](row)
        if cfg.get("multi_turn") and isinstance(formatted, list):
            formatted = formatted[0]
        prompts.append(str(formatted))
    return prompts


def make_input(tokenizer: AutoTokenizer, prompt: str) -> torch.Tensor:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer([text], return_tensors="pt").input_ids.to("cuda:0")


def policy_order(
    pair_index: int, *, start_with_residual: bool = False
) -> tuple[str, str]:
    first_order = tuple(reversed(POLICIES)) if start_with_residual else POLICIES
    if pair_index % 2 == 0:
        return first_order  # type: ignore[return-value]
    return tuple(reversed(first_order))  # type: ignore[return-value]


def parse_dataset_sample_counts(
    datasets: list[str], specs: list[str] | None, *, default: int
) -> dict[str, int]:
    counts = dict.fromkeys(datasets, default)
    if not specs:
        return counts
    for spec in specs:
        if "=" in spec:
            dataset, value = spec.split("=", 1)
        elif ":" in spec:
            dataset, value = spec.split(":", 1)
        else:
            raise ValueError(f"invalid dataset sample spec: {spec}")
        if dataset not in counts:
            raise ValueError(f"sample count specified for unloaded dataset: {dataset}")
        counts[dataset] = int(value)
    return counts


def build_round_robin_plan(
    prompts_by_dataset: dict[str, list[str]], datasets: list[str]
) -> list[tuple[str, int, str, int]]:
    plan: list[tuple[str, int, str, int]] = []
    max_count = max(
        (len(prompts) for prompts in prompts_by_dataset.values()), default=0
    )
    for sample_index in range(max_count):
        for dataset_name in datasets:
            prompts = prompts_by_dataset[dataset_name]
            if sample_index < len(prompts):
                plan.append(
                    (dataset_name, sample_index, prompts[sample_index], len(plan))
                )
    return plan


def build_sequential_plan(
    prompts_by_dataset: dict[str, list[str]], datasets: list[str]
) -> list[tuple[str, int, str, int]]:
    plan: list[tuple[str, int, str, int]] = []
    for dataset_name in datasets:
        for sample_index, prompt in enumerate(prompts_by_dataset[dataset_name]):
            plan.append((dataset_name, sample_index, prompt, len(plan)))
    return plan


def run_policy(
    *,
    policy: str,
    draft: DFlashDraftModel,
    target: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    residual_budget: int,
    residual_tree_width: int,
    residual_gain_scale: float,
    residual_min_gain: float,
    residual_min_margin: float,
    residual_draft_seconds: float | None,
    residual_target_seconds: float | None,
    residual_collect_diagnostics: bool,
) -> dict[str, Any]:
    reset_gpu_state()
    enable_residual = policy == "dflash_block16_residual"
    effective_residual_budget = 128
    start = time.perf_counter()
    stats = dflash_generate(
        draft,
        target=target,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id],
        temperature=0.0,
        block_size=16,
        return_stats=True,
        enable_primary_ddtree=True,
        enable_residual=enable_residual,
        residual_budget=effective_residual_budget,
        residual_tree_width=residual_tree_width,
        residual_gain_scale=residual_gain_scale,
        residual_min_gain=residual_min_gain,
        residual_min_margin=residual_min_margin,
        residual_draft_seconds=residual_draft_seconds,
        residual_target_seconds=residual_target_seconds,
        residual_collect_diagnostics=residual_collect_diagnostics,
    )
    torch.cuda.synchronize()
    end_to_end_wall_seconds = time.perf_counter() - start
    decode_wall_seconds = float(
        getattr(
            stats,
            "decode_seconds",
            float(stats.time_per_output_token) * int(stats.num_output_tokens),
        )
    )
    wall_seconds = decode_wall_seconds
    gate_records = list(getattr(stats, "residual_gate_records", []))
    edge_records = list(getattr(stats, "residual_edge_records", []))
    residual_attempt_count = int(
        getattr(
            stats,
            "residual_attempt_count",
            len(getattr(stats, "residual_attempts", [])),
        )
    )
    gate_off_count = int(
        getattr(
            stats,
            "residual_gate_off_count",
            len(
                [
                    record
                    for record in gate_records
                    if not record.get("should_run", False)
                ]
            ),
        )
    )
    residual_edge_count = int(getattr(stats, "residual_edge_count", len(edge_records)))
    residual_edge_accepted_count = int(
        getattr(
            stats,
            "residual_edge_accepted_count",
            sum(1 for record in edge_records if record.get("accepted")),
        )
    )
    residual_edge_probability_sum = float(
        getattr(
            stats,
            "residual_edge_probability_sum",
            sum(float(record.get("edge_probability", 0.0)) for record in edge_records),
        )
    )
    row = {
        "policy": policy,
        "output_tokens": int(stats.num_output_tokens),
        "wall_seconds": wall_seconds,
        "tokens_per_second": float(stats.num_output_tokens / max(wall_seconds, 1e-9)),
        "timing_scope": "decode_after_prefill",
        "decode_wall_seconds": decode_wall_seconds,
        "end_to_end_wall_seconds": end_to_end_wall_seconds,
        "time_to_first_token": float(getattr(stats, "time_to_first_token", 0.0) or 0.0),
        "acceptance_lengths": list(stats.acceptance_lengths),
        "residual_attempts": residual_attempt_count,
        "residual_runs": int(getattr(stats, "residual_runs", 0)),
        "residual_gate_records": gate_records,
        "residual_edge_records": edge_records,
        "gate_off_count": gate_off_count,
        "residual_edge_count": residual_edge_count,
        "residual_edge_accept_rate": (
            residual_edge_accepted_count / residual_edge_count
            if residual_edge_count
            else 0.0
        ),
        "mean_edge_probability": (
            residual_edge_probability_sum / residual_edge_count
            if residual_edge_count
            else 0.0
        ),
        "mean_delta_hat": _mean_stat_sum(
            stats,
            "residual_estimated_gain_sum",
            residual_attempt_count,
            gate_records,
            "estimated_residual_gain",
        ),
        "mean_theta_base": _mean_stat_sum(
            stats,
            "residual_baseline_throughput_sum",
            residual_attempt_count,
            gate_records,
            "baseline_throughput",
        ),
        "mean_theta_res_hat": _mean_stat_sum(
            stats,
            "residual_predicted_throughput_sum",
            residual_attempt_count,
            gate_records,
            "predicted_residual_throughput",
        ),
        "mean_effective_delta_hat": _mean_stat_sum(
            stats,
            "residual_effective_gain_sum",
            residual_attempt_count,
            gate_records,
            "effective_estimated_residual_gain",
        ),
        "mean_t_res": _mean_stat_sum(
            stats,
            "residual_target_seconds_sum",
            residual_attempt_count,
            gate_records,
            "residual_target_seconds",
        ),
        "mean_tree_nodes": _mean_stat_sum(
            stats,
            "residual_tree_node_sum",
            residual_attempt_count,
            gate_records,
            "tree_nodes",
        ),
        "residual_budget": effective_residual_budget,
        "residual_tree_width": residual_tree_width,
        "residual_gain_scale": residual_gain_scale,
        "residual_min_gain": residual_min_gain,
        "residual_min_margin": residual_min_margin,
        "residual_draft_seconds": residual_draft_seconds,
        "residual_target_seconds": residual_target_seconds,
        "residual_collect_diagnostics": residual_collect_diagnostics,
        "peak_memory_mib": float(torch.cuda.max_memory_allocated() / 1024 / 1024),
    }
    del stats
    reset_gpu_state()
    return row


def _mean_gate_field(records: list[dict[str, Any]], field: str) -> float:
    values = [float(record[field]) for record in records if field in record]
    return statistics.fmean(values) if values else 0.0


def _mean_stat_sum(
    stats: Any,
    sum_field: str,
    count: int,
    fallback_records: list[dict[str, Any]],
    fallback_field: str,
) -> float:
    if hasattr(stats, sum_field):
        return float(getattr(stats, sum_field)) / count if count else 0.0
    return _mean_gate_field(fallback_records, fallback_field)


def _edge_accept_rate(records: list[dict[str, Any]]) -> float:
    if not records:
        return 0.0
    return statistics.fmean(
        1.0 if record.get("accepted") else 0.0 for record in records
    )


def _weighted_edge_accept_rate(rows: list[dict[str, Any]]) -> float:
    accepted = sum(
        float(row.get("residual_edge_accept_rate", 0.0))
        * int(row.get("residual_edge_count", 0))
        for row in rows
    )
    total = sum(int(row.get("residual_edge_count", 0)) for row in rows)
    return accepted / total if total else 0.0


def _weighted_edge_probability(rows: list[dict[str, Any]]) -> float:
    probability = sum(
        float(row.get("mean_edge_probability", 0.0))
        * int(row.get("residual_edge_count", 0))
        for row in rows
    )
    total = sum(int(row.get("residual_edge_count", 0)) for row in rows)
    return probability / total if total else 0.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        summary[dataset] = {}
        for policy in POLICIES:
            policy_rows = [row for row in dataset_rows if row["policy"] == policy]
            tps = [float(row["tokens_per_second"]) for row in policy_rows]
            wall = [float(row["wall_seconds"]) for row in policy_rows]
            end_to_end_wall = [
                float(row.get("end_to_end_wall_seconds", row["wall_seconds"]))
                for row in policy_rows
            ]
            output_tokens = [int(row["output_tokens"]) for row in policy_rows]
            summary[dataset][policy] = {
                "num_runs": len(policy_rows),
                "mean_tps": statistics.fmean(tps) if tps else 0.0,
                "median_tps": statistics.median(tps) if tps else 0.0,
                "mean_wall_seconds": statistics.fmean(wall) if wall else 0.0,
                "mean_end_to_end_wall_seconds": (
                    statistics.fmean(end_to_end_wall) if end_to_end_wall else 0.0
                ),
                "total_output_tokens": sum(output_tokens),
                "residual_attempts": sum(
                    int(row.get("residual_attempts", 0)) for row in policy_rows
                ),
                "residual_runs": sum(
                    int(row.get("residual_runs", 0)) for row in policy_rows
                ),
                "gate_off_count": sum(
                    int(row.get("gate_off_count", 0)) for row in policy_rows
                ),
                "residual_edge_count": sum(
                    int(row.get("residual_edge_count", 0)) for row in policy_rows
                ),
                "residual_edge_accept_rate": _weighted_edge_accept_rate(policy_rows),
                "mean_edge_probability": _weighted_edge_probability(policy_rows),
                "mean_delta_hat": statistics.fmean(
                    float(row.get("mean_delta_hat", 0.0)) for row in policy_rows
                )
                if policy_rows
                else 0.0,
                "mean_theta_base": statistics.fmean(
                    float(row.get("mean_theta_base", 0.0)) for row in policy_rows
                )
                if policy_rows
                else 0.0,
                "mean_theta_res_hat": statistics.fmean(
                    float(row.get("mean_theta_res_hat", 0.0)) for row in policy_rows
                )
                if policy_rows
                else 0.0,
                "mean_effective_delta_hat": statistics.fmean(
                    float(row.get("mean_effective_delta_hat", 0.0))
                    for row in policy_rows
                )
                if policy_rows
                else 0.0,
                "mean_t_res": statistics.fmean(
                    float(row.get("mean_t_res", 0.0)) for row in policy_rows
                )
                if policy_rows
                else 0.0,
                "mean_tree_nodes": statistics.fmean(
                    float(row.get("mean_tree_nodes", 0.0)) for row in policy_rows
                )
                if policy_rows
                else 0.0,
            }
        base = summary[dataset]["dflash_block16"]["mean_tps"]
        residual = summary[dataset]["dflash_block16_residual"]["mean_tps"]
        summary[dataset]["residual_vs_base_speedup_pct"] = (
            (residual / base - 1.0) * 100.0 if base > 0 else 0.0
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument(
        "--dataset-samples",
        nargs="*",
        default=None,
        help="Per-dataset pair counts, e.g. gsm8k=110 math500=110 aime25=30.",
    )
    parser.add_argument("--round-robin-datasets", action="store_true")
    parser.add_argument("--start-with-residual", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--residual-budget", type=int, default=128)
    parser.add_argument("--residual-tree-width", type=int, default=5)
    parser.add_argument("--residual-gain-scale", type=float, default=1.0)
    parser.add_argument("--residual-min-gain", type=float, default=0.0)
    parser.add_argument("--residual-min-margin", type=float, default=0.2)
    parser.add_argument("--residual-draft-seconds", type=float, default=None)
    parser.add_argument("--residual-target-seconds", type=float, default=None)
    parser.add_argument(
        "--residual-diagnostics",
        action="store_true",
        help="Store per-gate and per-edge diagnostic records in JSONL rows.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/residual_fair_benchmark")
    )
    parser.add_argument("--target-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-model", default="z-lab/Qwen3-8B-DFlash-b16")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    jsonl_path = args.output_dir / f"fair_residual_{run_id}.jsonl"
    summary_path = args.output_dir / f"fair_residual_{run_id}_summary.json"

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
    if args.round_robin_datasets:
        prompt_plan = build_round_robin_plan(prompts_by_dataset, args.datasets)
    else:
        prompt_plan = build_sequential_plan(prompts_by_dataset, args.datasets)

    all_rows: list[dict[str, Any]] = []
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for dataset_name, sample_index, prompt, pair_index in prompt_plan:
            input_ids = make_input(tokenizer, prompt)
            order = policy_order(
                pair_index, start_with_residual=args.start_with_residual
            )
            print(
                f"dataset={dataset_name} sample={sample_index} pair={pair_index} order={list(order)}"
            )
            for policy in order:
                row = run_policy(
                    policy=policy,
                    draft=draft,
                    target=target,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    residual_budget=args.residual_budget,
                    residual_tree_width=args.residual_tree_width,
                    residual_gain_scale=args.residual_gain_scale,
                    residual_min_gain=args.residual_min_gain,
                    residual_min_margin=args.residual_min_margin,
                    residual_draft_seconds=args.residual_draft_seconds,
                    residual_target_seconds=args.residual_target_seconds,
                    residual_collect_diagnostics=args.residual_diagnostics,
                )
                row.update(
                    {
                        "dataset": dataset_name,
                        "sample_index": sample_index,
                        "pair_index": pair_index,
                        "order": list(order),
                        "max_new_tokens": args.max_new_tokens,
                        "residual_budget": args.residual_budget,
                        "residual_tree_width": args.residual_tree_width,
                        "residual_gain_scale": args.residual_gain_scale,
                        "residual_min_gain": args.residual_min_gain,
                        "residual_min_margin": args.residual_min_margin,
                        "residual_draft_seconds": args.residual_draft_seconds,
                        "residual_target_seconds": args.residual_target_seconds,
                        "fairness": {
                            "paired_prompt": True,
                            "cuda_cache_reset_before_each_policy": True,
                            "alternating_order_by_pair": True,
                            "round_robin_datasets": bool(args.round_robin_datasets),
                            "start_with_residual": bool(args.start_with_residual),
                            "hf_cache": str(Path.home() / "hf_cache" / "transformers"),
                        },
                    }
                )
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                all_rows.append(row)
                print(
                    f"  {policy}: tps={row['tokens_per_second']:.3f} "
                    f"wall={row['wall_seconds']:.3f}s residual_runs={row['residual_runs']} "
                    f"delta_hat={row['mean_delta_hat']:.3f}"
                )
            del input_ids
            reset_gpu_state()

    summary = summarize(all_rows)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"jsonl={jsonl_path}")
    print(f"summary={summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
