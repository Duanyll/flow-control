import os
from typing import Any

from torch.utils.data import Dataset

from flow_control.utils.logging import get_logger

logger = get_logger(__name__)


SIZE_LIMIT = 1024 * 1024 * 1024  # 1 GB


class ParquetDataset(Dataset):
    """Dataset backed by a Parquet file.

    Rows are loaded eagerly into memory as plain dicts, do not use this for large Parquet files.
    """

    def __init__(self, path: str):
        import pyarrow.parquet as pq

        self.path = path
        file_size = os.path.getsize(path)
        if file_size > SIZE_LIMIT:
            raise ValueError(
                f"ParquetDataset is not suitable for files larger than {SIZE_LIMIT} bytes, but got {file_size} bytes. "
            )

        table = pq.read_table(path)
        self.data: list[dict[str, Any]] = table.to_pylist()

        logger.info(f"ParquetDataset: loaded {len(self.data)} rows from {path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> dict[str, Any]:
        row = self.data[idx].copy()
        if "__key__" not in row:
            row["__key__"] = str(idx)
        return row
