"""Underscore-prefixed row fields are disposable adapter intermediates."""

from collections.abc import Mapping
from typing import Any, cast


def cache_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key.startswith("_")}


def clear_cache(row: Mapping[str, Any]) -> None:
    # Rows are dicts; Mapping also admits TypedDict callers with optional caches.
    data = cast(dict[str, Any], row)
    for key in tuple(data):
        if key.startswith("_"):
            del data[key]


if __name__ == "__main__":
    import torch
    from rich import print

    row = {"key": "example", "_img_ids": torch.zeros(4, 3)}
    cache = cache_fields(row)
    assert cache["_img_ids"] is row["_img_ids"]
    clear_cache(row)
    assert row == {"key": "example"}
    print(
        "[green]Runtime fields keep tensor identity and leave data fields intact[/green]"
    )
