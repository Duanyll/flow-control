import os
from typing import Any

from torch.utils.data import Dataset

from flow_control.utils.logging import get_logger

logger = get_logger(__name__)

SIZE_LIMIT = 100 * 1024 * 1024  # 100 MB


class LinesDataset(Dataset):
    """Dataset backed by a text file where each line is a separate sample"""

    def __init__(self, path: str):
        self.path = path
        self.data: list[str] = []

        file_size = os.path.getsize(path)
        if file_size > SIZE_LIMIT:
            raise ValueError(
                f"LinesDataset is not suitable for files larger than {SIZE_LIMIT} bytes, but got {file_size} bytes. "
            )

        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(line)

        logger.info(f"LinesDataset: loaded {len(self.data)} rows from {path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> dict[str, Any]:
        row = {"prompt": self.data[idx]}
        if "__key__" not in row:
            row["__key__"] = str(idx)
        return row
