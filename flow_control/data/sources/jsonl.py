import json
import os
from typing import Any

from flow_control.data.rows import KEY, Row
from flow_control.utils.logging import get_logger

from .base import source_registry

logger = get_logger(__name__)

SIZE_LIMIT = 100 * 1024 * 1024  # 100 MB


@source_registry.register("jsonl")
class JsonlSource:
    """JSON Lines file (.jsonl); each line must be a JSON object."""

    def __init__(self, path: str):
        self.path = path
        self.data: list[dict[str, Any]] = []

        file_size = os.path.getsize(path)
        if file_size > SIZE_LIMIT:
            raise ValueError(
                f"JsonlSource is not suitable for files larger than {SIZE_LIMIT} bytes, but got {file_size} bytes. "
            )

        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

        logger.info(f"JsonlSource: loaded {len(self.data)} rows from {path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Row:
        row = self.data[index].copy()
        row.setdefault(KEY, row.pop("__key__", str(index)))
        return row
