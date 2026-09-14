import os
from typing import Any

from flow_control.data.rows import KEY, Row
from flow_control.utils.logging import get_logger

from .base import source_registry

logger = get_logger(__name__)

SIZE_LIMIT = 1024 * 1024 * 1024  # 1 GB


@source_registry.register("parquet")
class ParquetSource:
    """Parquet file loaded eagerly into plain dicts; not for large files."""

    def __init__(self, path: str):
        import pyarrow.parquet as pq

        self.path = path
        file_size = os.path.getsize(path)
        if file_size > SIZE_LIMIT:
            raise ValueError(
                f"ParquetSource is not suitable for files larger than {SIZE_LIMIT} bytes, but got {file_size} bytes. "
            )

        table = pq.read_table(path)
        self.data: list[dict[str, Any]] = table.to_pylist()

        logger.info(f"ParquetSource: loaded {len(self.data)} rows from {path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Row:
        row = self.data[index].copy()
        if KEY not in row:
            row[KEY] = str(index)
        return row
