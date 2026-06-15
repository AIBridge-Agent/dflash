# pyright: reportMissingImports=none, reportOptionalIterable=none, reportOperatorIssue=none, reportPossiblyUnboundVariable=none, reportGeneralTypeIssues=none, reportOptionalOperand=none, reportReturnType=none
import time
import torch
from types import SimpleNamespace
from typing import Callable
from typing_extensions import Unpack
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    Qwen3Config,
    Qwen3PreTrainedModel,
    Qwen3MLP,
    GradientCheckpointingLayer,
    FlashAttentionKwargs,
    rotate_half,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
)
from transformers import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache

from .residual_cache import compact_dynamic_cache
from .residual_errors import NoResidualOpportunity
from .residual_generate import (
    build_residual_candidate_groups,
    build_residual_edge_records,
    build_residual_opportunity,
)
from .residual_surrogate import build_residual_ddtree
from .residual_tree_verify import (
    linearize_residual_tree,
    parent_indices,
    walk_residual_tree,
)

# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: list[int] | None,
) -> torch.Tensor:
    offset = 1
    selected_states = [hidden_states[layer_id + offset] for layer_id in layer_ids]
    return torch.cat(selected_states, dim=-1)


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size) / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


def _cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def _tree_attention_mask(
    parents: tuple[int | None, ...],
    *,
    past_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build an additive tree-attention mask for flattened residual DDTree nodes."""

    q_len = len(parents)
    kv_len = past_length + q_len
    mask = torch.zeros((1, 1, q_len, kv_len), device=device, dtype=dtype)
    blocked_value = -1.0e4
    for query_index in range(q_len):
        allowed = set()
        cursor: int | None = query_index
        while cursor is not None:
            allowed.add(cursor)
            cursor = parents[cursor]
        for key_index in range(q_len):
            if key_index not in allowed:
                mask[:, :, query_index, past_length + key_index] = blocked_value
    return mask


@torch.inference_mode()
def dflash_generate(
    model: "DFlashDraftModel",
    target: nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    stop_token_ids: list[int] | None,
    temperature: float,
    block_size: int | None = None,
    mask_token_id: int | None = None,
    return_stats: bool = False,
    enable_residual: bool = False,
    residual_budget: int = 1,
    residual_tree_width: int = 5,
    residual_gain_scale: float = 0.6,
    residual_min_gain: float = 3.0,
    residual_min_margin: float = 0.2,
    residual_draft_seconds: float | None = None,
    residual_target_seconds: float | None = None,
    residual_collect_diagnostics: bool = True,
):
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    block_size = model.block_size if block_size is None else block_size
    mask_token_id = model.mask_token_id if mask_token_id is None else mask_token_id

    output_ids = torch.full(
        (1, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=target.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    prefill_start = _cuda_time() if return_stats else None
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=block_size > 1,
    )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
        output.logits, temperature
    )
    if block_size > 1:
        target_hidden = extract_context_feature(
            output.hidden_states, model.target_layer_ids
        )
    time_to_first_token = _cuda_time() - prefill_start if return_stats else None

    decode_start = _cuda_time() if return_stats else None
    acceptance_lengths = []
    residual_attempts = []
    residual_gate_records = []
    residual_edge_records = []
    residual_runs = 0
    residual_attempt_count = 0
    residual_gate_off_count = 0
    residual_edge_count = 0
    residual_edge_accepted_count = 0
    residual_edge_probability_sum = 0.0
    residual_estimated_gain_sum = 0.0
    residual_effective_gain_sum = 0.0
    residual_baseline_throughput_sum = 0.0
    residual_predicted_throughput_sum = 0.0
    residual_target_seconds_sum = 0.0
    residual_tree_node_sum = 0
    start = num_input_tokens
    draft_prefill = True
    target_seconds_sum = 0.0
    target_seconds_count = 0

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        block_probabilities = [1.0] * len(block_output_ids[0])
        sampled_draft_ids = None
        draft_logits = None
        measured_draft_seconds = residual_draft_seconds
        measured_target_seconds = residual_target_seconds
        if block_size > 1:
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_start = (
                _cuda_time()
                if enable_residual and residual_draft_seconds is None
                else None
            )
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
            if draft_start is not None:
                measured_draft_seconds = _cuda_time() - draft_start
            past_key_values_draft.crop(start)
            sampled_draft_ids = sample(draft_logits)
            block_output_ids[:, 1:] = sampled_draft_ids
            if draft_prefill and return_stats:
                draft_prefill = False
                decode_start = _cuda_time()

        target_start = (
            _cuda_time()
            if enable_residual and residual_target_seconds is None
            else None
        )
        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=block_size > 1,
        )
        if target_start is not None:
            measured_target_seconds = _cuda_time() - target_start
        current_target_seconds = float(measured_target_seconds or 1e-9)
        residual_target_seconds_estimate = (
            float(residual_target_seconds)
            if residual_target_seconds is not None
            else (
                target_seconds_sum / target_seconds_count
                if target_seconds_count > 0
                else current_target_seconds
            )
        )

        posterior = sample(output.logits, temperature)
        acceptance_length = (
            (block_output_ids[:, 1:] == posterior[:, :-1])
            .cumprod(dim=1)
            .sum(dim=1)[0]
            .item()
        )
        residual_ran = False
        if enable_residual and block_size > 2 and acceptance_length < block_size - 1:
            tail_start_row = int(acceptance_length + 1)
            residual_candidate_groups = ()
            if draft_logits is not None and sampled_draft_ids is not None:
                tail_logits = draft_logits[:, tail_start_row:, :]
                if tail_logits.shape[1] > 0:
                    draft_tail_probs = torch.softmax(tail_logits, dim=-1)
                    sampled_tail_ids = sampled_draft_ids[:, tail_start_row:]
                    sampled_tail_probs = torch.gather(
                        draft_tail_probs, -1, sampled_tail_ids.unsqueeze(-1)
                    ).squeeze(-1)
                    for offset, probability in enumerate(
                        sampled_tail_probs[0].detach().cpu().tolist()
                    ):
                        block_probabilities[tail_start_row + offset + 1] = float(
                            probability
                        )
                    topk = min(
                        int(residual_tree_width), int(draft_tail_probs.shape[-1])
                    )
                    residual_top_probs, residual_top_ids = torch.topk(
                        draft_tail_probs, k=topk, dim=-1
                    )
                    residual_candidate_groups = build_residual_candidate_groups(
                        top_token_ids_by_block=tuple(
                            tuple(int(x) for x in row)
                            for row in residual_top_ids[0].detach().cpu().tolist()
                        ),
                        top_probabilities_by_block=tuple(
                            tuple(float(x) for x in row)
                            for row in residual_top_probs[0].detach().cpu().tolist()
                        ),
                        start_position=int(start),
                        first_block_index=int(acceptance_length + 2),
                    )
            try:
                opportunity = build_residual_opportunity(
                    block_token_ids=tuple(
                        int(x) for x in block_output_ids[0].detach().cpu().tolist()
                    ),
                    block_probabilities=tuple(block_probabilities),
                    start_position=int(start),
                    acceptance_length=int(acceptance_length),
                    verifier_mismatch_token_id=int(
                        posterior[0, acceptance_length].detach().cpu().item()
                    ),
                    current_accepted=int(acceptance_length + 1),
                    draft_seconds=float(measured_draft_seconds or 1e-9),
                    target_seconds=current_target_seconds,
                    residual_budget=residual_budget,
                    residual_candidate_groups=residual_candidate_groups,
                    residual_target_seconds=residual_target_seconds_estimate,
                    residual_gain_scale=residual_gain_scale,
                    residual_min_gain=residual_min_gain,
                    residual_min_margin=residual_min_margin,
                )
            except NoResidualOpportunity:
                opportunity = None

            gate_record = None
            if opportunity is not None:
                residual_attempt_count += 1
                if not opportunity.should_run:
                    residual_gate_off_count += 1
                residual_estimated_gain_sum += float(
                    opportunity.estimated_residual_gain
                )
                residual_effective_gain_sum += float(
                    opportunity.gate.effective_estimated_residual_gain
                )
                residual_baseline_throughput_sum += float(
                    opportunity.gate.baseline_throughput
                )
                residual_predicted_throughput_sum += float(
                    opportunity.gate.predicted_residual_throughput
                )
                residual_target_seconds_sum += float(
                    opportunity.gate.residual_target_seconds
                )
                if residual_collect_diagnostics:
                    residual_attempts.append(opportunity)
                    gate_record = {
                        "start": int(start),
                        "acceptance_length": int(acceptance_length),
                        "current_accepted": int(acceptance_length + 1),
                        "estimated_residual_gain": float(
                            opportunity.estimated_residual_gain
                        ),
                        "effective_estimated_residual_gain": float(
                            opportunity.gate.effective_estimated_residual_gain
                        ),
                        "baseline_throughput": float(
                            opportunity.gate.baseline_throughput
                        ),
                        "predicted_residual_throughput": float(
                            opportunity.gate.predicted_residual_throughput
                        ),
                        "should_run": bool(opportunity.should_run),
                        "reason": opportunity.gate.reason,
                        "tree_nodes": 0,
                        "draft_seconds": float(opportunity.gate.draft_seconds),
                        "target_seconds": float(opportunity.gate.target_seconds),
                        "residual_target_seconds": float(
                            opportunity.gate.residual_target_seconds
                        ),
                        "residual_gain_scale": float(
                            opportunity.gate.residual_gain_scale
                        ),
                        "residual_min_gain": float(opportunity.gate.min_residual_gain),
                        "residual_min_margin": float(
                            opportunity.gate.min_throughput_margin
                        ),
                        "eal_estimator": "depth_mass",
                    }
                    residual_gate_records.append(gate_record)

            if opportunity is not None and opportunity.should_run:
                mismatch_block_index = acceptance_length + 1
                residual_tail = block_output_ids[:, mismatch_block_index + 1 :]
                residual_tree = (
                    build_residual_ddtree(
                        opportunity.path.anchor,
                        residual_candidate_groups,
                        budget=residual_budget,
                    )
                    if residual_candidate_groups
                    else opportunity.residual_tree
                )
                tree_nodes = (
                    linearize_residual_tree(residual_tree)
                    if residual_tree is not None
                    else ()
                )
                residual_tree_node_sum += len(tree_nodes)
                if gate_record is not None:
                    gate_record["tree_nodes"] = len(tree_nodes)
                if tree_nodes:
                    residual_ran = True
                    residual_runs += 1
                    prefix_width = int(acceptance_length + 1)
                    anchor_position = int(start + mismatch_block_index)
                    output_ids[:, start : start + prefix_width] = block_output_ids[
                        :, :prefix_width
                    ]
                    output_ids[:, anchor_position] = posterior[:, acceptance_length]

                    past_key_values_target.crop(anchor_position)
                    tree_ids = torch.tensor(
                        [
                            int(posterior[0, acceptance_length].detach().cpu().item()),
                            *(node.token_id for node in tree_nodes),
                        ],
                        dtype=torch.long,
                        device=target.device,
                    ).unsqueeze(0)
                    tree_position_ids = torch.tensor(
                        [anchor_position, *(node.position for node in tree_nodes)],
                        dtype=torch.long,
                        device=target.device,
                    ).unsqueeze(0)
                    tree_parents = parent_indices(tree_nodes)
                    tree_output_dtype = next(target.parameters()).dtype
                    tree_output = target(
                        tree_ids,
                        position_ids=tree_position_ids,
                        attention_mask=_tree_attention_mask(
                            tree_parents,
                            past_length=anchor_position,
                            device=target.device,
                            dtype=tree_output_dtype,
                        ),
                        past_key_values=past_key_values_target,
                        use_cache=True,
                        output_hidden_states=block_size > 1,
                    )
                    tree_posterior = sample(tree_output.logits, temperature)
                    tree_walk = walk_residual_tree(
                        tree_nodes,
                        target_token_ids_by_flat_index=tuple(
                            int(token_id)
                            for token_id in tree_posterior[0].detach().cpu().tolist()
                        ),
                    )
                    for edge_record in tree_walk.edge_records:
                        residual_edge_count += 1
                        residual_edge_accepted_count += int(
                            bool(edge_record.get("accepted", False))
                        )
                        residual_edge_probability_sum += float(
                            edge_record.get("edge_probability", 0.0)
                        )
                        if residual_collect_diagnostics:
                            residual_edge_records.append(
                                {
                                    **edge_record,
                                    "start": int(start),
                                    "acceptance_length": int(acceptance_length),
                                    "current_accepted": int(acceptance_length + 1),
                                    "residual_run_index": int(residual_runs),
                                    "estimated_residual_gain": float(
                                        opportunity.estimated_residual_gain
                                    ),
                                    "effective_estimated_residual_gain": float(
                                        opportunity.gate.effective_estimated_residual_gain
                                    ),
                                    "verification_mode": "tree",
                                }
                            )

                    accepted_indices = [
                        0,
                        *(node.flat_index for node in tree_walk.accepted_nodes),
                    ]
                    accepted_index_tensor = torch.tensor(
                        accepted_indices, dtype=torch.long, device=tree_ids.device
                    )
                    accepted_tokens = tree_ids.index_select(1, accepted_index_tensor)
                    residual_start = anchor_position
                    output_ids[
                        :, residual_start : residual_start + accepted_tokens.shape[1]
                    ] = accepted_tokens
                    mismatch_position = residual_start + accepted_tokens.shape[1]
                    output_ids[:, mismatch_position] = tree_walk.mismatch_token_id

                    compact_dynamic_cache(
                        past_key_values_target, anchor_position, accepted_indices
                    )
                    start = mismatch_position
                    acceptance_lengths.append(
                        acceptance_length + tree_walk.accepted_count + 2
                    )
                    if block_size > 1:
                        prefix_hidden = extract_context_feature(
                            output.hidden_states, model.target_layer_ids
                        )[:, : acceptance_length + 1, :]
                        residual_hidden = extract_context_feature(
                            tree_output.hidden_states, model.target_layer_ids
                        ).index_select(1, accepted_index_tensor)
                        target_hidden = torch.cat(
                            [prefix_hidden, residual_hidden], dim=1
                        )
                elif residual_tail.shape[1] > 0:
                    residual_ran = True
                    residual_runs += 1
                    output_ids[:, start : start + acceptance_length + 1] = (
                        block_output_ids[:, : acceptance_length + 1]
                    )
                    output_ids[:, start + acceptance_length + 1] = posterior[
                        :, acceptance_length
                    ]
                    past_key_values_target.crop(start + acceptance_length + 1)
                    residual_ids = torch.cat(
                        [
                            posterior[:, acceptance_length : acceptance_length + 1],
                            residual_tail,
                        ],
                        dim=1,
                    )
                    residual_position_ids = position_ids[
                        :, start + mismatch_block_index : start + block_size
                    ]
                    residual_output = target(
                        residual_ids,
                        position_ids=residual_position_ids,
                        past_key_values=past_key_values_target,
                        use_cache=True,
                        output_hidden_states=block_size > 1,
                    )
                    residual_posterior = sample(residual_output.logits, temperature)
                    residual_acceptance_length = (
                        (residual_ids[:, 1:] == residual_posterior[:, :-1])
                        .cumprod(dim=1)
                        .sum(dim=1)[0]
                        .item()
                    )
                    for edge_record in build_residual_edge_records(
                        opportunity.path,
                        residual_acceptance_length=int(residual_acceptance_length),
                    ):
                        residual_edge_count += 1
                        residual_edge_accepted_count += int(
                            bool(edge_record.get("accepted", False))
                        )
                        residual_edge_probability_sum += float(
                            edge_record.get("edge_probability", 0.0)
                        )
                        if residual_collect_diagnostics:
                            residual_edge_records.append(
                                {
                                    **edge_record,
                                    "start": int(start),
                                    "acceptance_length": int(acceptance_length),
                                    "current_accepted": int(acceptance_length + 1),
                                    "residual_run_index": int(residual_runs),
                                    "estimated_residual_gain": float(
                                        opportunity.estimated_residual_gain
                                    ),
                                    "effective_estimated_residual_gain": float(
                                        opportunity.gate.effective_estimated_residual_gain
                                    ),
                                    "verification_mode": "path",
                                }
                            )
                    residual_start = start + acceptance_length + 1
                    output_ids[
                        :,
                        residual_start : residual_start
                        + residual_acceptance_length
                        + 1,
                    ] = residual_ids[:, : residual_acceptance_length + 1]
                    output_ids[:, residual_start + residual_acceptance_length + 1] = (
                        residual_posterior[:, residual_acceptance_length]
                    )
                    start += acceptance_length + residual_acceptance_length + 2
                    past_key_values_target.crop(start)
                    acceptance_lengths.append(
                        acceptance_length + residual_acceptance_length + 2
                    )
                    if block_size > 1:
                        prefix_hidden = extract_context_feature(
                            output.hidden_states, model.target_layer_ids
                        )[:, : acceptance_length + 1, :]
                        residual_hidden = extract_context_feature(
                            residual_output.hidden_states, model.target_layer_ids
                        )[:, : residual_acceptance_length + 1, :]
                        target_hidden = torch.cat(
                            [prefix_hidden, residual_hidden], dim=1
                        )

        if enable_residual and residual_target_seconds is None:
            target_seconds_sum += current_target_seconds
            target_seconds_count += 1

        if not residual_ran:
            output_ids[:, start : start + acceptance_length + 1] = block_output_ids[
                :, : acceptance_length + 1
            ]
            output_ids[:, start + acceptance_length + 1] = posterior[
                :, acceptance_length
            ]
            start += acceptance_length + 1
            past_key_values_target.crop(start)
            acceptance_lengths.append(acceptance_length + 1)

            if block_size > 1:
                target_hidden = extract_context_feature(
                    output.hidden_states, model.target_layer_ids
                )[:, : acceptance_length + 1, :]

        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:]
            for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, : min(start + 1, max_length)]
    if stop_token_ids is not None:
        stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = torch.isin(
            output_ids[0][num_input_tokens:], stop_token_ids
        ).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    if not return_stats:
        return output_ids

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = _cuda_time() - decode_start
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=total_decode_time / num_output_tokens,
        acceptance_lengths=acceptance_lengths,
        residual_attempts=residual_attempts,
        residual_gate_records=residual_gate_records,
        residual_edge_records=residual_edge_records,
        residual_runs=residual_runs,
        residual_attempt_count=residual_attempt_count,
        residual_gate_off_count=residual_gate_off_count,
        residual_edge_count=residual_edge_count,
        residual_edge_accepted_count=residual_edge_accepted_count,
        residual_edge_probability_sum=residual_edge_probability_sum,
        residual_estimated_gain_sum=residual_estimated_gain_sum,
        residual_effective_gain_sum=residual_effective_gain_sum,
        residual_baseline_throughput_sum=residual_baseline_throughput_sum,
        residual_predicted_throughput_sum=residual_predicted_throughput_sum,
        residual_target_seconds_sum=residual_target_seconds_sum,
        residual_tree_node_sum=residual_tree_node_sum,
    )


# ---------------------------------------------------------------------------
# DFlash model
# ---------------------------------------------------------------------------


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3DFlashAttention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        v = torch.cat([v_ctx, v_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        target_hidden: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class DFlashDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList(
            [
                Qwen3DFlashDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.target_layer_ids = self.config.dflash_config.get(
            "target_layer_ids",
            build_target_layer_ids(config.num_target_layers, config.num_hidden_layers),
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = config.block_size
        self.mask_token_id = self.config.dflash_config.get("mask_token_id", None)
        self.post_init()

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        noise_embedding: torch.Tensor | None = None,
        target_hidden: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
    ):
        self.eval()
        return dflash_generate(
            self,
            target=target,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
        )
