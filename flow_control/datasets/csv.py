import csv

from torch.utils.data import Dataset

from flow_control.utils.logging import get_logger

logger = get_logger(__name__)


class CsvDataset(Dataset):
    """Simple CSV dataset that returns raw string dicts.

    Type conversion is handled downstream by pydantic coercion in the
    preprocessing pipeline (see ``flow_control.utils.coercion``).
    """

    def __init__(self, path: str):
        self.path = path
        self.data: list[dict[str, str]] = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row)
        logger.info(f"CsvDataset: loaded {len(self.data)} rows from {path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> dict[str, str]:
        row = self.data[idx].copy()
        if "__key__" not in row:
            row["__key__"] = str(idx)
        return row
