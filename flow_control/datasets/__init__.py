from typing import TYPE_CHECKING, Annotated, Any, TypeGuard, cast

from datasets import load_dataset
from pydantic import WithJsonSchema
from torch.utils.data import ConcatDataset, Dataset

from flow_control.datasets.coercion import build_type_adapter, coerce_record
from flow_control.utils.logging import get_logger
from flow_control.utils.pipeline import DataSink

from .bucket_directory import BucketDirectoryDataset, BucketDirectoryDatasink
from .csv import CsvDataset
from .inline import InlineDataset
from .jsonl import JsonlDataset
from .lines import LinesDataset
from .lmdb import LMDBDataset, LMDBDataSink
from .parquet import ParquetDataset
from .pickle_directory import PickleDirectoryDataset, PickleDirectoryDataSink
from .plain_directory import PlainDirectoryDataset
from .prism_layers_pro import PrismLayersProDataset
from .raw_directory import RawDirectoryDataset, RawDirectoryDataSink

logger = get_logger("flow_control.datasets")


class LimitedDataset(Dataset):
    """Wrapper dataset that limits the length of an underlying dataset."""

    def __init__(self, dataset: "MapDataset", limit: int):
        self.dataset = dataset
        self.limit = limit

    def __len__(self) -> int:
        return min(len(self.dataset), self.limit)

    def __getitem__(self, index: int):
        if index >= len(self):
            raise IndexError("Index out of range")
        return self.dataset[index]


class CoercedDataset(Dataset):
    """Wrapper that applies pydantic coercion to each sample via a TypedDict type.

    Pickle-safe: only stores the TypedDict class (not the TypeAdapter), and
    rebuilds the adapter lazily via the lru-cached ``build_type_adapter``.
    """

    def __init__(
        self,
        dataset: "MapDataset",
        coerce_to: type,
        attachment_dir: str = "",
    ):
        self.dataset = dataset
        self.coerce_to = coerce_to
        self.attachment_dir = attachment_dir
        logger.info(
            f"{dataset.__class__.__name__} wrapped with coercion to {coerce_to.__name__}"
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.dataset[index]
        adapter = build_type_adapter(self.coerce_to)
        return coerce_record(item, adapter, self.attachment_dir)


DatasetConfig = Annotated[
    dict[str, Any],
    WithJsonSchema(
        {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "description": "Dataset type (e.g. lmdb, plain_directory, pickle_directory, raw_directory, bucket_directory, csv, jsonl, parquet, inline, huggingface, concat)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Optional maximum length of the dataset. If specified, the dataset will be limited to this number of samples.",
                },
            },
            "required": ["type"],
            "additionalProperties": True,
        }
    ),
]

DATASET_REGISTRY = {
    "lmdb": LMDBDataset,
    "plain_directory": PlainDirectoryDataset,
    "pickle_directory": PickleDirectoryDataset,
    "raw_directory": RawDirectoryDataset,
    "bucket_directory": BucketDirectoryDataset,
    "prism_layers_pro": PrismLayersProDataset,
    "csv": CsvDataset,
    "inline": InlineDataset,
    "jsonl": JsonlDataset,
    "parquet": ParquetDataset,
    "lines": LinesDataset,
}

if TYPE_CHECKING:

    class MapDataset(Dataset):
        def __len__(self) -> int: ...
else:
    MapDataset = Dataset


def is_map_dataset(dataset: Dataset) -> TypeGuard[MapDataset]:
    return (
        isinstance(dataset, Dataset)
        and hasattr(dataset, "__getitem__")
        and hasattr(dataset, "__len__")
    )


def parse_dataset(
    dataset_config: DatasetConfig,
    coerce_to: type | None = None,
) -> MapDataset:
    if "type" not in dataset_config:
        raise ValueError("dataset_config must contain a 'type' key.")
    dataset_config = dataset_config.copy()
    dataset_type = dataset_config.pop("type")
    limit = dataset_config.pop("limit", None)
    attachment_dir: str = dataset_config.pop("attachment_dir", ".")

    dataset: Dataset
    if dataset_type == "huggingface":
        dataset = load_dataset(**dataset_config)
        if not is_map_dataset(dataset):
            raise ValueError(
                "The loaded dataset is not of type MapDataset. Make sure you have passed the correct parameters."
            )
    elif dataset_type == "concat":
        datasets = []
        for _, value in dataset_config.items():
            datasets.append(parse_dataset(value, coerce_to=coerce_to))
        dataset = ConcatDataset(datasets)
        assert is_map_dataset(dataset), "Concatenated dataset is not a MapDataset."
    elif dataset_type in DATASET_REGISTRY:
        dataset_class = DATASET_REGISTRY[dataset_type]
        dataset = dataset_class(**dataset_config)
        assert is_map_dataset(dataset), (
            f"Loaded dataset of type {dataset_type} is not a MapDataset."
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Wrap with coercion (skipped for concat — sub-datasets are already wrapped).
    if coerce_to is not None and dataset_type != "concat":
        dataset = CoercedDataset(dataset, coerce_to, attachment_dir)

    if limit is not None and limit > 0:
        limited = LimitedDataset(cast(MapDataset, dataset), limit)
        return cast(MapDataset, limited)

    return cast(MapDataset, dataset)


DatasinkConfig = Annotated[
    dict[str, Any],
    WithJsonSchema(
        {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "description": "Datasink type (e.g. lmdb, pickle_directory, raw_directory, bucket_directory)",
                },
            },
            "required": ["type"],
            "additionalProperties": True,
        }
    ),
]

DATASINK_REGISTRY = {
    "lmdb": LMDBDataSink,
    "pickle_directory": PickleDirectoryDataSink,
    "raw_directory": RawDirectoryDataSink,
    "bucket_directory": BucketDirectoryDatasink,
}


def parse_datasink(datasink_config: DatasinkConfig) -> DataSink:
    if "type" not in datasink_config:
        raise ValueError("datasink_config must contain a 'type' key.")
    datasink_config = datasink_config.copy()
    datasink_type = datasink_config.pop("type")

    if datasink_type in DATASINK_REGISTRY:
        datasink_class = DATASINK_REGISTRY[datasink_type]
        return datasink_class(**datasink_config)
    else:
        raise ValueError(f"Unknown datasink type: {datasink_type}")
