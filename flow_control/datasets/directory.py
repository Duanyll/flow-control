import os
import torch
import uuid
import gzip
from torch.utils.data import Dataset

from flow_control.utils.pipeline import DataSink

class DirectoryDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        self.files = []
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file() and (entry.name.endswith(".pt") or entry.name.endswith(".pt.gz")):
                    self.files.append(entry.name)
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> dict:
        file_name = self.files[index]
        file_path = os.path.join(self.path, file_name)
        if file_name.endswith(".gz"):
            with gzip.open(file_path, "rb") as f:
                sample = torch.load(f) # type: ignore
        else:
            sample = torch.load(file_path)
        return sample


class DirectoryDataSink(DataSink):
    def __init__(self, path: str, use_compression: bool = False):
        self.path = path
        self.use_compression = use_compression
        os.makedirs(path, exist_ok=True)

    def write(self, item: dict):
        if "__key__" not in item:
            key = str(uuid.uuid4())
        else:
            key = item["__key__"]
        file_path = os.path.join(self.path, f"{key}.pt" if not self.use_compression else f"{key}.pt.gz")
        if self.use_compression:
            with gzip.open(file_path, "wb") as f:
                torch.save(item, f) # type: ignore
        else:
            torch.save(item, file_path)
        return True