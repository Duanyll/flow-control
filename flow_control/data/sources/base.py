"""Raw sources: map-style readers that yield one ``Row`` per index (design §3).

Members register on ``source_registry`` under their ``"type"`` tag; ``open_source``
instantiates them from a dict config and applies the ``limit`` / ``attachment_dir``
/ ``concat`` handling. Coercion to the processor input TypedDict is applied here,
unconditionally, whenever the caller passes ``coerce_to``.
"""

import bisect
from itertools import accumulate
from typing import Any, Protocol

from flow_control.data.coercion import build_type_adapter, coerce_record
from flow_control.data.rows import Row
from flow_control.utils.logging import get_logger
from flow_control.utils.registry import Registry

logger = get_logger(__name__)


class RawSource(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Row: ...


source_registry: Registry[Any] = Registry("source")
"""Duck-typed members (no shared base class), instantiated with the config kwargs."""


class LimitedSource:
    """Truncates ``source`` to its first ``limit`` rows."""

    def __init__(self, source: RawSource, limit: int):
        self.source = source
        self.limit = limit

    def __len__(self) -> int:
        return min(len(self.source), self.limit)

    def __getitem__(self, index: int) -> Row:
        if index >= len(self):
            raise IndexError("Index out of range")
        return self.source[index]


class CoercedSource:
    """Applies pydantic coercion to each row via a TypedDict type.

    Pickle-safe: stores only the TypedDict class and rebuilds the ``TypeAdapter``
    lazily through the lru-cached ``build_type_adapter``.
    """

    def __init__(self, source: RawSource, coerce_to: type, attachment_dir: str):
        self.source = source
        self.coerce_to = coerce_to
        self.attachment_dir = attachment_dir
        logger.info(
            f"{source.__class__.__name__} wrapped with coercion to {coerce_to.__name__}"
        )

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> Row:
        adapter = build_type_adapter(self.coerce_to)
        return coerce_record(self.source[index], adapter, self.attachment_dir)


class ConcatSource:
    """Concatenates sources back to back; ``"type": "concat"`` in ``open_source``."""

    def __init__(self, sources: list[RawSource]):
        self.sources = sources
        self.cumulative = list(accumulate(len(s) for s in sources))

    def __len__(self) -> int:
        return self.cumulative[-1] if self.cumulative else 0

    def __getitem__(self, index: int) -> Row:
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")
        which = bisect.bisect_right(self.cumulative, index)
        offset = self.cumulative[which - 1] if which > 0 else 0
        return self.sources[which][index - offset]


def open_source(config: dict[str, Any], coerce_to: type | None) -> RawSource:
    """Instantiate the ``"type"`` member with the remaining keys as kwargs.

    ``coerce_to`` is the processor input TypedDict (``None`` = no coercion).
    ``"limit"`` truncates, ``"attachment_dir"`` resolves relative attachment paths
    during coercion, and ``"concat"`` concatenates every other value (each a source
    config of its own, coerced individually).
    """
    if "type" not in config:
        raise ValueError("Dataset config must contain a 'type' key.")
    config = dict(config)
    source_type = config.pop("type")
    limit = config.pop("limit", None)
    attachment_dir: str = config.pop("attachment_dir", ".")

    source: RawSource
    if source_type == "concat":
        source = ConcatSource([open_source(v, coerce_to) for v in config.values()])
    else:
        source_class = source_registry.get(source_type)
        if source_class is None:
            raise ValueError(
                f"Unknown dataset type {source_type!r}; registered: "
                f"{sorted(source_registry.members())} (plus 'concat' and 'cache')."
            )
        source = source_class(**config)
        if coerce_to is not None:
            source = CoercedSource(source, coerce_to, attachment_dir)

    if limit is not None and limit > 0:
        source = LimitedSource(source, limit)
    return source
