import os

from flow_control.data.rows import KEY, Row
from flow_control.utils.logging import get_logger

from .base import source_registry

logger = get_logger(__name__)

SIZE_LIMIT = 100 * 1024 * 1024  # 100 MB


@source_registry.register("lines")
class LinesSource:
    """Text file where each non-empty line is one ``prompt``."""

    def __init__(self, path: str):
        self.path = path
        self.data: list[str] = []

        file_size = os.path.getsize(path)
        if file_size > SIZE_LIMIT:
            raise ValueError(
                f"LinesSource is not suitable for files larger than {SIZE_LIMIT} bytes, but got {file_size} bytes. "
            )

        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(line)

        logger.info(f"LinesSource: loaded {len(self.data)} rows from {path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Row:
        return {"prompt": self.data[index], KEY: str(index)}
