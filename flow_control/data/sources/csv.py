import csv

from flow_control.data.rows import KEY, Row
from flow_control.utils.logging import get_logger

from .base import source_registry

logger = get_logger(__name__)


@source_registry.register("csv")
class CsvSource:
    """CSV file; rows are raw string dicts, converted downstream by coercion."""

    def __init__(self, path: str):
        self.path = path
        self.data: list[dict[str, str]] = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row)
        logger.info(f"CsvSource: loaded {len(self.data)} rows from {path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Row:
        row: Row = self.data[index].copy()
        row.setdefault(KEY, row.pop("__key__", str(index)))
        return row
