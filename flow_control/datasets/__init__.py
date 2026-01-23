from typing import Any

from datasets import load_dataset
from torch.utils.data import ConcatDataset, Dataset

from flow_control.utils.pipeline import DataSink

from .bins_directory import BinsDirectoryDataset, BinsDirectoryDataSink
from .directory import (
    PickleDirectoryDataset,
    PickleDirectoryDataSink,
    RawDirectoryDataset,
    RawDirectoryDataSink,
)
from .lmdb import LMDBDataset, LMDBDataSink
from .plain import PlainDirectoryDataset
from .prism_layers_pro import PrismLayersProDataset

DatasetConfig = dict[str, Any]

DATASET_REGISTRY = {
    "lmdb": LMDBDataset,
    "plain_directory": PlainDirectoryDataset,
    "pickle_directory": PickleDirectoryDataset,
    "raw_directory": RawDirectoryDataset,
    "bins_directory": BinsDirectoryDataset,
    "prism_layers_pro": PrismLayersProDataset,
}


def parse_dataset(dataset_config: DatasetConfig) -> Dataset:
    if "type" not in dataset_config:
        raise ValueError("dataset_config must contain a 'type' key.")
    dataset_type = dataset_config.pop("type")

    if dataset_type == "huggingface":
        dataset = load_dataset(**dataset_config)
        if isinstance(dataset, Dataset):
            return dataset
        else:
            raise ValueError(
                "The loaded dataset is not of type Dataset. Make sure you have passed the correct parameters."
            )
    elif dataset_type == "multi":
        datasets = []
        for _, value in dataset_config.items():
            datasets.append(parse_dataset(value))
        return ConcatDataset(datasets)
    elif dataset_type in DATASET_REGISTRY:
        dataset_class = DATASET_REGISTRY[dataset_type]
        return dataset_class(**dataset_config)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


DatasinkConfig = dict[str, Any]

DATASINK_REGISTRY = {
    "lmdb": LMDBDataSink,
    "pickle_directory": PickleDirectoryDataSink,
    "raw_directory": RawDirectoryDataSink,
    "bins_directory": BinsDirectoryDataSink,
}


def parse_datasink(datasink_config: DatasinkConfig) -> DataSink:
    if "type" not in datasink_config:
        raise ValueError("datasink_config must contain a 'type' key.")
    datasink_type = datasink_config.pop("type")

    if datasink_type in DATASINK_REGISTRY:
        datasink_class = DATASINK_REGISTRY[datasink_type]
        return datasink_class(**datasink_config)
    else:
        raise ValueError(f"Unknown datasink type: {datasink_type}")


# This library is designed to work with batch size 1 datasets.
# For larger batch sizes, use gradient accumulation.
# Dataset should return tensors with batch dimension 1.
def collate_fn(batch: list[dict]) -> dict:
    if len(batch) != 1:
        raise ValueError(
            "Batch size greater than 1 is not supported. Use gradient accumulation instead."
        )
    item = batch[0]
    return item
