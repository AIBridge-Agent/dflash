# pyright: reportMissingImports=none

import pytest


torch = pytest.importorskip("torch")


def test_tree_attention_mask_blocks_non_ancestors_without_blocking_past() -> None:
    from dflash.model import _tree_attention_mask

    mask = _tree_attention_mask(
        (None, 0, 1, 0),
        past_length=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert mask.shape == (1, 1, 4, 7)
    assert torch.all(mask[..., :3] == 0)

    local_mask = mask[0, 0, :, 3:]
    expected = torch.tensor(
        [
            [0.0, -1.0e4, -1.0e4, -1.0e4],
            [0.0, 0.0, -1.0e4, -1.0e4],
            [0.0, 0.0, 0.0, -1.0e4],
            [0.0, -1.0e4, -1.0e4, 0.0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(local_mask, expected)
