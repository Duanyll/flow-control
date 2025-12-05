from typing import Any
from torch.utils.data import Dataset, ConcatDataset
from datasets import load_dataset
from .lmdb import LMDBDataset


DatasetConfig = dict[str, Any]


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
    elif dataset_type == "lmdb":
        return LMDBDataset(**dataset_config)
    elif dataset_type == "multi":
        datasets = []
        for _, value in dataset_config.items():
            datasets.append(parse_dataset(value))
        return ConcatDataset(datasets)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
