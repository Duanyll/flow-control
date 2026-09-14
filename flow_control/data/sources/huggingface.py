from typing import Any

from flow_control.data.rows import KEY, Row
from flow_control.utils.logging import get_logger

from .base import source_registry

logger = get_logger(__name__)


@source_registry.register("huggingface")
class HuggingFaceSource:
    """``datasets.load_dataset(**kwargs)``; must resolve to a map-style split."""

    def __init__(self, **load_kwargs: Any):
        from datasets import load_dataset

        dataset = load_dataset(**load_kwargs)
        if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
            raise ValueError(
                "load_dataset did not return a map-style dataset; pass a single "
                f"split (got {type(dataset).__name__})."
            )
        self.dataset: Any = dataset
        logger.info(f"HuggingFaceSource: loaded {len(self.dataset)} rows")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Row:
        row = dict(self.dataset[index])
        if KEY not in row:
            row[KEY] = str(index)
        return row
