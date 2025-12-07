from typing import Any
from torch.utils.data import Dataset, ConcatDataset
from datasets import load_dataset

from .lmdb import LMDBDataset, LMDBDataSink
from .civitai import CivitaiDataset
from .directory import DirectoryDataset, DirectoryDataSink

DatasetConfig = dict[str, Any]

DATASET_REGISTRY = {
    "lmdb": LMDBDataset,
    "civitai": CivitaiDataset,
    "directory": DirectoryDataset,
}

def parse_dataset(dataset_config: DatasetConfig) -> Dataset:
    if not isinstance(dataset_config, dict):
        raise ValueError("dataset_config must be a dictionary.")
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
    

DATASINK_REGISTRY = {
    "lmdb": LMDBDataSink,
    "directory": DirectoryDataSink,
}