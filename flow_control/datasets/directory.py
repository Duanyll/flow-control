import gzip
import json
import os
import uuid

import torch
from PIL import Image
from torch.utils.data import Dataset

from flow_control.utils.common import pil_to_tensor, tensor_to_pil
from flow_control.utils.logging import get_logger
from flow_control.utils.pipeline import DataSink

logger = get_logger(__name__)


class PickleDirectoryDataset(Dataset):
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
        if "__key__" not in sample:
            sample["__key__"] = file_name.rsplit(".", 1)[0]
        return sample


class PickleDirectoryDataSink(DataSink):
    def __init__(self, worker_id, path: str, use_compression: bool = False):
        self.worker_id = worker_id
        self.path = path
        self.use_compression = use_compression
        os.makedirs(path, exist_ok=True)
        logger.info(
            f"Initialized PickleDirectoryDataSink for worker {worker_id} at path: {path}"
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


class RawDirectoryDataset(Dataset):
    """
    Dataset directory structure:

    ```
    root/
        sample_key_1/
            some_image.png
            some_tensor.pt
            some_text.txt
            some_metadata.json
            array_of_images.0.png
            array_of_images.1.png
            ...
        sample_key_2/
            ...
    ```

    Each sample is stored in its own subdirectory named by its unique key. The above
    example loads to a dictionary like:

    ```
    {
        "__key__": "sample_key_1",
        "some_image": PIL.Image,
        "some_tensor": torch.Tensor,
        "some_text": str,
        "some_metadata": dict,
        "array_of_images": list[PIL.Image],
    }
    ```
    """

    def __init__(self, path: str, allowed_fields: list[str] | None = None):
        self.path = path
        self.sample_dirs = []
        self.allowed_fields = allowed_fields
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_dir():
                    self.sample_dirs.append(entry.name)
        self.sample_dirs.sort()
        logger.info(f"Loaded {len(self.sample_dirs)} samples from directory: {path}")

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, index) -> dict:
        sample_dir_name = self.sample_dirs[index]
        sample_dir_path = os.path.join(self.path, sample_dir_name)
        sample = {}
        with os.scandir(sample_dir_path) as it:
            for entry in it:
                if entry.is_file():
                    file_name, file_ext = os.path.splitext(entry.name)
                    file_path = os.path.join(sample_dir_path, entry.name)
                    field_name = file_name.rsplit(".", 1)[0]
                    if (
                        self.allowed_fields is not None
                        and field_name not in self.allowed_fields
                    ):
                        continue

                    data = None
                    if file_ext.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
                        pil_image = Image.open(file_path).convert("RGB")
                        data = pil_to_tensor(pil_image)
                    elif file_ext.lower() == ".pt":
                        data = torch.load(file_path)
                    elif file_ext.lower() == ".txt":
                        with open(file_path, encoding="utf-8") as f:
                            data = f.read()
                    elif file_ext.lower() == ".json":
                        with open(file_path, encoding="utf-8") as f:
                            data = json.load(f)
                    else:
                        logger.warning(
                            f"Unsupported file type {file_ext} in sample {sample_dir_name}, skipping."
                        )
                        continue

                    # Handle array files
                    if "." in file_name:
                        base_name, index_str = file_name.rsplit(".", 1)
                        if index_str.isdigit():
                            index = int(index_str)
                            if base_name not in sample:
                                sample[base_name] = []
                            while len(sample[base_name]) <= index:
                                sample[base_name].append(None)
                            sample[base_name][index] = data
                        else:
                            logger.warning(
                                f"Unexpected file name format {file_name} in sample {sample_dir_name}, skipping."
                            )
                    else:
                        sample[file_name] = data
        sample["__key__"] = sample_dir_name
        return sample


class RawDirectoryDataSink(DataSink):
    """
    DataSink that writes samples to a directory in raw file formats.
    Each sample is stored in its own subdirectory named by its unique key.
    Supported types are:

    - PIL.Image, or torch.Tensor and field name ends with "_image" or "_images": .png
    - torch.Tensor: .pt
    - str: .txt
    - dict: .json
    - list of above types: saved as multiple files with .0, .1, ... suffixes
    """

    def __init__(self, worker_id, path: str, allowed_fields: list[str] | None = None):
        self.worker_id = worker_id
        self.path = path
        self.allowed_fields = allowed_fields
        os.makedirs(path, exist_ok=True)
        logger.info(
            f"Initialized RawDirectoryDataSink for worker {worker_id} at path: {path}"
        )

    def write(self, item: dict):
        key = str(uuid.uuid4()) if "__key__" not in item else item["__key__"]
        sample_dir_path = os.path.join(self.path, key)
        os.makedirs(sample_dir_path, exist_ok=True)

        for field_name, data in item.items():
            if field_name == "__key__":
                continue

            if (
                self.allowed_fields is not None
                and field_name not in self.allowed_fields
            ):
                continue

            if isinstance(data, list):
                for index, element in enumerate(data):
                    self._write_single_field(
                        sample_dir_path, f"{field_name}.{index}", element
                    )
            else:
                self._write_single_field(sample_dir_path, field_name, data)

        return True

    def _write_single_field(self, sample_dir_path: str, field_name: str, data):
        if isinstance(data, torch.Tensor):
            if field_name.endswith("_image") or field_name.endswith("_images"):
                pil_image = tensor_to_pil(data)
                file_path = os.path.join(sample_dir_path, f"{field_name}.png")
                pil_image.save(file_path)
            else:
                file_path = os.path.join(sample_dir_path, f"{field_name}.pt")
                torch.save(data, file_path)
        elif isinstance(data, str):
            file_path = os.path.join(sample_dir_path, f"{field_name}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(data)
        elif isinstance(data, dict):
            file_path = os.path.join(sample_dir_path, f"{field_name}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        elif isinstance(data, Image.Image):
            file_path = os.path.join(sample_dir_path, f"{field_name}.png")
            data.save(file_path)
        else:
            logger.warning(
                f"Unsupported data type {type(data)} for field {field_name}, skipping."
            )
