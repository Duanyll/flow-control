from typing import Any

from flow_control.data.rows import KEY, Row
from flow_control.utils.logging import get_logger

from .base import source_registry

logger = get_logger(__name__)


@source_registry.register("inline")
class InlineSource:
    """Records passed directly as a list of dicts; handy for tests and tiny sets."""

    def __init__(self, data: list[dict[str, Any]]):
        self.data = data
        logger.info(f"InlineSource: loaded {len(self.data)} rows")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Row:
        row = self.data[index].copy()
        row.setdefault(KEY, row.pop("__key__", str(index)))
        return row
