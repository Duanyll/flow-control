"""Data stack: raw sources -> preprocess -> [random cache] -> pack -> [packed cache].

Public entry points: ``DatasetConfig`` (the ``{"type": ...}`` dict every trainer /
preprocess config carries) and ``open_store`` (design §7.1).
"""

from typing import Annotated, Any, Literal

from pydantic import WithJsonSchema

from .index import Index, IndexEntry
from .pack import PackConfig, pack
from .rows import COST, IMAGE_SIZE, KEY, PADDING, Row, is_padding, shape_signature
from .sources import RawSource, open_source, source_registry
from .store import (
    DirectoryStore,
    LmdbStore,
    OnlineStore,
    PackedStore,
    RowStore,
    open_cache,
)
from .writer import CacheOutputConfig, RandomCacheWriter, finalize_cache

DatasetConfig = Annotated[
    dict[str, Any],
    WithJsonSchema(
        {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "description": (
                        "Dataset type: 'cache' (any directory written by `flow-control "
                        "preprocess` / `flow-control pack`) or a raw source read online "
                        "(csv, jsonl, lines, parquet, inline, plain_directory, "
                        "raw_directory, huggingface, concat)"
                    ),
                },
                "path": {
                    "type": "string",
                    "description": "Cache directory for type 'cache' (or the file/directory of a raw source).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Optional maximum number of rows; truncates the cache index or the raw source.",
                },
            },
            "required": ["type"],
            "additionalProperties": True,
        }
    ),
]

CACHE_CONFIG_KEYS = {"type", "path", "limit"}


def open_store(
    config: DatasetConfig,
    *,
    processor_class: type,
    mode: Literal["training", "inference"],
) -> RowStore:
    """``"type": "cache"`` -> Directory / Lmdb / Packed store chosen by ``meta.json``;
    any other type -> ``OnlineStore`` over the raw source coerced to the processor's
    ``mode`` input TypedDict."""
    if "type" not in config:
        raise ValueError("Dataset config must contain a 'type' key.")
    if config["type"] == "cache":
        unknown = set(config) - CACHE_CONFIG_KEYS
        if unknown:
            raise ValueError(
                f"Unknown keys {sorted(unknown)} in a 'cache' dataset config; "
                f"allowed: {sorted(CACHE_CONFIG_KEYS)}."
            )
        return open_cache(config["path"], limit=config.get("limit"))

    # Imported here, not at module level: flow_control.processors imports this
    # package (coercion aliases), so a top-level import would be circular.
    from flow_control.processors import get_processor_input_typeddict

    coerce_to = get_processor_input_typeddict(processor_class, mode)
    return OnlineStore(open_source(config, coerce_to=coerce_to))


__all__ = [
    "COST",
    "IMAGE_SIZE",
    "KEY",
    "PADDING",
    "CacheOutputConfig",
    "DatasetConfig",
    "DirectoryStore",
    "Index",
    "IndexEntry",
    "LmdbStore",
    "OnlineStore",
    "PackConfig",
    "PackedStore",
    "RandomCacheWriter",
    "RawSource",
    "Row",
    "RowStore",
    "finalize_cache",
    "is_padding",
    "open_cache",
    "open_source",
    "open_store",
    "pack",
    "shape_signature",
    "source_registry",
]
