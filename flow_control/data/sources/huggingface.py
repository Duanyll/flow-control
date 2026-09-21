from typing import Any

from flow_control.data.rows import KEY, Row
from flow_control.utils.logging import get_logger

from .base import source_registry

logger = get_logger(__name__)


@source_registry.register("huggingface")
class HuggingFaceSource:
    """``datasets.load_dataset(**kwargs)``; must resolve to a map-style split."""

    def __init__(self, **load_kwargs: Any):
        import datasets

        dataset = datasets.load_dataset(**load_kwargs)
        # A DatasetDict (no ``split`` given) is a dict, so it has __len__ and
        # __getitem__; only the map-style Dataset class is acceptable.
        if not isinstance(dataset, datasets.Dataset):
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
        row.setdefault(KEY, row.pop("__key__", str(index)))
        return row


if __name__ == "__main__":
    from unittest.mock import patch

    import datasets
    from rich import print

    split = datasets.Dataset.from_dict({"prompt": ["a", "b"]})
    with patch("datasets.load_dataset", return_value=split):
        source = HuggingFaceSource(path="stub")
        print(source[1])
        assert len(source) == 2 and source[1] == {"prompt": "b", KEY: "1"}
    for bad in (datasets.DatasetDict({"train": split}), split.to_iterable_dataset()):
        with patch("datasets.load_dataset", return_value=bad):
            try:
                HuggingFaceSource(path="stub")
            except ValueError as e:
                print("rejected:", e)
            else:
                raise AssertionError(f"{type(bad).__name__} accepted")
