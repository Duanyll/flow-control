from typing import Any

from torch.utils.data import Dataset

from flow_control.utils.logging import get_logger

logger = get_logger(__name__)


class InlineDataset(Dataset):
    """Dataset whose records are passed directly as a list of dicts.

    Useful for small datasets or testing without touching the filesystem.
    Type conversion is handled downstream by pydantic coercion.
    """

    def __init__(self, data: list[dict[str, Any]]):
        self.data = data
        logger.info(f"InlineDataset: loaded {len(self.data)} rows")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> dict[str, Any]:
        row = self.data[idx].copy()
        if "__key__" not in row:
            row["__key__"] = str(idx)
        return row
