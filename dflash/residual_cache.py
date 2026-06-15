# pyright: reportMissingImports=none
"""DynamicCache compaction helpers for residual DDTree verification.

Adapted from the official DDTree implementation's cache compaction strategy:
keep only the accepted tree nodes from the appended verification window and crop
away unselected siblings so the target cache remains an exact causal path.
"""

from __future__ import annotations

import torch
from transformers import DynamicCache


def _compact_appended_window(
    cache_tensor: torch.Tensor, past_length: int, keep_current_indices: torch.Tensor
) -> None:
    current_length = cache_tensor.shape[-2] - past_length
    if current_length <= 0:
        return

    keep_count = keep_current_indices.numel()
    if keep_count == 0 or keep_count == current_length:
        return

    kept_tail = cache_tensor.narrow(-2, past_length, current_length).index_select(
        -2, keep_current_indices
    )
    cache_tensor.narrow(-2, past_length, keep_count).copy_(kept_tail)


def compact_dynamic_cache(
    past_key_values: DynamicCache, past_length: int, keep_current_indices: list[int]
) -> None:
    """Keep accepted nodes from the appended tree window in-place.

    `past_length` is the cache length before the tree verification call. The
    keep indices are relative to the appended tree window, where index 0 is the
    verifier anchor and subsequent indices are flattened residual DDTree nodes.
    """

    if len(keep_current_indices) == 0:
        past_key_values.crop(past_length)
        return

    keep_tensor_by_device: dict[torch.device, torch.Tensor] = {}

    def get_keep_tensor(device: torch.device) -> torch.Tensor:
        if device not in keep_tensor_by_device:
            keep_tensor_by_device[device] = torch.tensor(
                keep_current_indices, dtype=torch.long, device=device
            )
        return keep_tensor_by_device[device]

    if hasattr(past_key_values, "key_cache") and hasattr(
        past_key_values, "value_cache"
    ):
        for layer_idx in range(len(past_key_values.key_cache)):
            key_cache = past_key_values.key_cache[layer_idx]
            value_cache = past_key_values.value_cache[layer_idx]
            keep_tensor = get_keep_tensor(key_cache.device)
            _compact_appended_window(key_cache, past_length, keep_tensor)
            _compact_appended_window(value_cache, past_length, keep_tensor)
        past_key_values.crop(past_length + len(keep_current_indices))
        return

    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            if (
                not hasattr(layer, "keys")
                or layer.keys is None
                or layer.keys.numel() == 0
            ):
                continue
            keep_tensor = get_keep_tensor(layer.keys.device)
            _compact_appended_window(layer.keys, past_length, keep_tensor)
            _compact_appended_window(layer.values, past_length, keep_tensor)
        past_key_values.crop(past_length + len(keep_current_indices))
        return

    raise RuntimeError(
        "Unsupported DynamicCache layout for residual DDTree cache compaction"
    )
