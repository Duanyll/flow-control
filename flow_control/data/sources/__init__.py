"""Raw sources (design §3). Importing the package registers every built-in reader."""

from . import (  # noqa: F401  (registration side effects)
    csv,
    huggingface,
    inline,
    jsonl,
    lines,
    parquet,
    plain_directory,
    raw_directory,
)
from .base import (
    CoercedSource,
    ConcatSource,
    LimitedSource,
    RawSource,
    open_source,
    source_registry,
)

__all__ = [
    "CoercedSource",
    "ConcatSource",
    "LimitedSource",
    "RawSource",
    "open_source",
    "source_registry",
]
