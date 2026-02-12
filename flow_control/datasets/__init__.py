from typing import TYPE_CHECKING, Any, TypeGuard

from datasets import load_dataset
from torch.utils.data import ConcatDataset, Dataset

from flow_control.utils.pipeline import DataSink

from .bucket_directory import BucketDirectoryDataset, BucketDirectoryDatasink
from .csv import CsvDataset
from .lmdb import LMDBDataset, LMDBDataSink
from .pickle_directory import PickleDirectoryDataset, PickleDirectoryDataSink
from .plain_directory import PlainDirectoryDataset
from .prism_layers_pro import PrismLayersProDataset
from .raw_directory import RawDirectoryDataset, RawDirectoryDataSink

DatasetConfig = dict[str, Any]

DATASET_REGISTRY = {
    "lmdb": LMDBDataset,
    "plain_directory": PlainDirectoryDataset,
    "pickle_directory": PickleDirectoryDataset,
    "raw_directory": RawDirectoryDataset,
    "bucket_directory": BucketDirectoryDataset,
    "prism_layers_pro": PrismLayersProDataset,
    "csv": CsvDataset,
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


def parse_dataset(dataset_config: DatasetConfig) -> MapDataset:
    if "type" not in dataset_config:
        raise ValueError("dataset_config must contain a 'type' key.")
    dataset_config = dataset_config.copy()
    dataset_type = dataset_config.pop("type")

    if dataset_type == "huggingface":
        dataset = load_dataset(**dataset_config)
        if is_map_dataset(dataset):
            return dataset
        else:
            raise ValueError(
                "The loaded dataset is not of type MapDataset. Make sure you have passed the correct parameters."
            )
    elif dataset_type == "multi":
        datasets = []
        for _, value in dataset_config.items():
            datasets.append(parse_dataset(value))
        dataset = ConcatDataset(datasets)
        assert is_map_dataset(dataset), "Concatenated dataset is not a MapDataset."
        return dataset
    elif dataset_type in DATASET_REGISTRY:
        dataset_class = DATASET_REGISTRY[dataset_type]
        dataset = dataset_class(**dataset_config)
        assert is_map_dataset(dataset), (
            f"Loaded dataset of type {dataset_type} is not a MapDataset."
        )
        return dataset
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


DatasinkConfig = dict[str, Any]

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
