# pyright: reportMissingImports=none
import pytest

torch = pytest.importorskip("torch")

from dflash.residual_cache import compact_dynamic_cache  # noqa: E402


class _FakeCache:
    def __init__(self) -> None:
        self.key_cache = [torch.arange(6, dtype=torch.float32).view(1, 1, 6, 1)]
        self.value_cache = [torch.arange(10, 16, dtype=torch.float32).view(1, 1, 6, 1)]
        self.cropped_to = None

    def crop(self, max_length: int) -> None:
        self.cropped_to = max_length
        self.key_cache[0] = self.key_cache[0][..., :max_length, :]
        self.value_cache[0] = self.value_cache[0][..., :max_length, :]


def test_compact_dynamic_cache_keeps_only_accepted_tree_nodes() -> None:
    cache = _FakeCache()

    compact_dynamic_cache(cache, past_length=2, keep_current_indices=[0, 2])

    assert cache.cropped_to == 4
    assert cache.key_cache[0].flatten().tolist() == [0.0, 1.0, 2.0, 4.0]
    assert cache.value_cache[0].flatten().tolist() == [10.0, 11.0, 12.0, 14.0]
