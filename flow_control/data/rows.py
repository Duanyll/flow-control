"""Row contract shared by every stage of the data stack.

A row is a plain ``dict``; the keys below are reserved (design §1). ``padding``
only ever exists in consumer memory (plan-time padding) and is never persisted.
"""

import re
from typing import Any

import torch

Row = dict[str, Any]

KEY = "key"
"""Unique within a dataset, written by the raw source; ``KEY_PATTERN`` chars only."""
COST = "cost"
"""Token total (latent + text + reference) computed by the processor; the sort key."""
IMAGE_SIZE = "image_size"
"""``(h, w)`` written by the processor."""
PADDING = "padding"
"""``True`` on plan-time padding rows: forwarded as usual, loss / outputs dropped."""

KEY_PATTERN = re.compile(r"[A-Za-z0-9._-]+")


def is_padding(row: Row) -> bool:
    return bool(row.get(PADDING, False))


def shape_signature(row: Row) -> str:
    """``name:shape`` of every top-level tensor and ``name:[len]`` of every list,
    sorted by name and joined. A sort tie-break after ``cost``, never a bucket."""
    parts: list[str] = []
    for name in sorted(row):
        value = row[name]
        if isinstance(value, torch.Tensor):
            parts.append(f"{name}:{'x'.join(str(d) for d in value.shape)}")
        elif isinstance(value, list):
            parts.append(f"{name}:[{len(value)}]")
    return ",".join(parts)


if __name__ == "__main__":
    from rich import print

    row: Row = {
        KEY: "a",
        COST: 3,
        "prompt_embeds": torch.zeros(1, 4, 8),
        "clean_latents": torch.zeros(2, 16, 4, 4),
        "reference_latents": [torch.zeros(1, 16, 2, 2)] * 3,
        IMAGE_SIZE: (32, 32),
    }
    print(shape_signature(row))
    assert shape_signature(row) == (
        "clean_latents:2x16x4x4,prompt_embeds:1x4x8,reference_latents:[3]"
    )
    assert not is_padding(row)
    row[PADDING] = True
    assert is_padding(row)
