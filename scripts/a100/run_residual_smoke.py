"""Small A100 smoke runner for opt-in residual DFlash generation.

This is intentionally a one-prompt smoke test, not a paper-valid benchmark.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dflash.model import DFlashDraftModel, dflash_generate

TARGET_MODEL = "Qwen/Qwen3-8B"
DRAFT_MODEL = "z-lab/Qwen3-8B-DFlash-b16"
PROMPT = "How many positive whole-number divisors does 196 have?"
MAX_NEW_TOKENS = 32


def _load_models() -> tuple[DFlashDraftModel, AutoModelForCausalLM, AutoTokenizer]:
    device = "cuda:0"
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="sdpa",
    ).eval()
    draft = DFlashDraftModel.from_pretrained(
        DRAFT_MODEL,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="sdpa",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
    return draft, target, tokenizer


def _prepare_input(tokenizer: AutoTokenizer) -> torch.Tensor:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer([text], return_tensors="pt").input_ids.to("cuda:0")


def _run_case(
    name: str,
    draft: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
    *,
    block_size: int,
    enable_residual: bool,
) -> SimpleNamespace:
    torch.cuda.synchronize()
    start = time.perf_counter()
    stats = dflash_generate(
        draft,
        target=target,
        input_ids=input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        stop_token_ids=[tokenizer.eos_token_id],
        temperature=0.0,
        block_size=block_size,
        return_stats=True,
        enable_residual=enable_residual,
        residual_budget=64,
        residual_tree_width=4,
    )
    torch.cuda.synchronize()
    wall = time.perf_counter() - start
    print(f"CASE {name}")
    print(f"  output_tokens={stats.num_output_tokens}")
    print(f"  wall_seconds={wall:.4f}")
    print(f"  tokens_per_second={stats.num_output_tokens / max(wall, 1e-9):.4f}")
    print(f"  acceptance_lengths={stats.acceptance_lengths}")
    print(f"  residual_attempts={len(getattr(stats, 'residual_attempts', []))}")
    print(f"  residual_runs={getattr(stats, 'residual_runs', 0)}")
    gate_records = getattr(stats, "residual_gate_records", [])
    if gate_records:
        print(f"  first_gate_record={gate_records[0]}")
    return stats


def main() -> None:
    print("loading models")
    draft, target, tokenizer = _load_models()
    input_ids = _prepare_input(tokenizer)
    print("running smoke cases")
    _run_case(
        "baseline_block1",
        draft,
        target,
        input_ids,
        tokenizer,
        block_size=1,
        enable_residual=False,
    )
    _run_case(
        "dflash_block16",
        draft,
        target,
        input_ids,
        tokenizer,
        block_size=16,
        enable_residual=False,
    )
    _run_case(
        "dflash_block16_residual",
        draft,
        target,
        input_ids,
        tokenizer,
        block_size=16,
        enable_residual=True,
    )


if __name__ == "__main__":
    main()
