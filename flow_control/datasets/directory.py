import gzip
import os
import uuid

import torch
from torch.utils.data import Dataset

from flow_control.utils.logging import get_logger
from flow_control.utils.pipeline import DataSink

logger = get_logger(__name__)


class DirectoryDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        self.files = []
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file() and (
                    entry.name.endswith(".pt") or entry.name.endswith(".pt.gz")
                ):
                    self.files.append(entry.name)
        self.files.sort()
        logger.info(f"Loaded {len(self.files)} files from directory: {path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> dict:
        file_name = self.files[index]
        file_path = os.path.join(self.path, file_name)
        if file_name.endswith(".gz"):
            with gzip.open(file_path, "rb") as f:
                sample = torch.load(f)  # type: ignore
        else:
            sample = torch.load(file_path)
        return sample


class DirectoryDataSink(DataSink):
    def __init__(self, worker_id, path: str, use_compression: bool = False):
        self.worker_id = worker_id
        self.path = path
        self.use_compression = use_compression
        os.makedirs(path, exist_ok=True)
        logger.info(
            f"Initialized DirectoryDataSink for worker {worker_id} at path: {path}"
        )

    def write(self, item: dict):
        key = str(uuid.uuid4()) if "__key__" not in item else item["__key__"]
        file_path = os.path.join(
            self.path, f"{key}.pt" if not self.use_compression else f"{key}.pt.gz"
        )
        if self.use_compression:
            with gzip.open(file_path, "wb") as f:
                torch.save(item, f)  # type: ignore
        else:
            torch.save(item, file_path)
        return True
