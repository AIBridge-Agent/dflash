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
}

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


def policy_order(sample_index: int) -> tuple[str, str]:
    if sample_index % 2 == 0:
        return POLICIES
    return tuple(reversed(POLICIES))  # type: ignore[return-value]


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
) -> dict[str, Any]:
    reset_gpu_state()
    enable_residual = policy == "dflash_block16_residual"
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
        enable_residual=enable_residual,
        residual_budget=residual_budget,
        residual_tree_width=residual_tree_width,
    )
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - start
    gate_records = list(getattr(stats, "residual_gate_records", []))
    gate_off_records = [record for record in gate_records if not record.get("should_run", False)]
    row = {
        "policy": policy,
        "output_tokens": int(stats.num_output_tokens),
        "wall_seconds": wall_seconds,
        "tokens_per_second": float(stats.num_output_tokens / max(wall_seconds, 1e-9)),
        "acceptance_lengths": list(stats.acceptance_lengths),
        "residual_attempts": len(getattr(stats, "residual_attempts", [])),
        "residual_runs": int(getattr(stats, "residual_runs", 0)),
        "residual_gate_records": gate_records,
        "gate_off_count": len(gate_off_records),
        "mean_delta_hat": _mean_gate_field(gate_records, "estimated_residual_gain"),
        "mean_theta_base": _mean_gate_field(gate_records, "baseline_throughput"),
        "mean_theta_res_hat": _mean_gate_field(gate_records, "predicted_residual_throughput"),
        "mean_tree_nodes": _mean_gate_field(gate_records, "tree_nodes"),
        "residual_budget": residual_budget,
        "residual_tree_width": residual_tree_width,
        "peak_memory_mib": float(torch.cuda.max_memory_allocated() / 1024 / 1024),
    }
    del stats
    reset_gpu_state()
    return row


def _mean_gate_field(records: list[dict[str, Any]], field: str) -> float:
    values = [float(record[field]) for record in records if field in record]
    return statistics.fmean(values) if values else 0.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        summary[dataset] = {}
        for policy in POLICIES:
            policy_rows = [row for row in dataset_rows if row["policy"] == policy]
            tps = [float(row["tokens_per_second"]) for row in policy_rows]
            wall = [float(row["wall_seconds"]) for row in policy_rows]
            output_tokens = [int(row["output_tokens"]) for row in policy_rows]
            summary[dataset][policy] = {
                "num_runs": len(policy_rows),
                "mean_tps": statistics.fmean(tps) if tps else 0.0,
                "median_tps": statistics.median(tps) if tps else 0.0,
                "mean_wall_seconds": statistics.fmean(wall) if wall else 0.0,
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
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--residual-budget", type=int, default=64)
    parser.add_argument("--residual-tree-width", type=int, default=4)
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

    all_rows: list[dict[str, Any]] = []
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for dataset_name in args.datasets:
            print(f"dataset={dataset_name}")
            prompts = load_prompts(
                dataset_name, max_samples=args.max_samples, seed=args.seed
            )
            for sample_index, prompt in enumerate(prompts):
                input_ids = make_input(tokenizer, prompt)
                order = policy_order(sample_index)
                print(
                    f"dataset={dataset_name} sample={sample_index} order={list(order)}"
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
                    )
                    row.update(
                        {
                            "dataset": dataset_name,
                            "sample_index": sample_index,
                            "order": list(order),
                            "max_new_tokens": args.max_new_tokens,
                            "residual_budget": args.residual_budget,
                            "residual_tree_width": args.residual_tree_width,
                            "fairness": {
                                "paired_prompt": True,
                                "cuda_cache_reset_before_each_policy": True,
                                "alternating_order_by_sample": True,
                                "hf_cache": str(
                                    Path.home() / "hf_cache" / "transformers"
                                ),
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
