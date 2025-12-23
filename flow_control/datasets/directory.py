import gzip
import json
import os
import uuid
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from flow_control.utils.common import pil_to_tensor, tensor_to_pil
from flow_control.utils.logging import get_logger
from flow_control.utils.pipeline import DataSink

logger = get_logger(__name__)


def _serialize_with_field_names(obj: Any, base_path: str, field_path: str = "") -> Any:
    """
    Recursively serialize an object, saving special types to external files.

    Args:
        obj: The object to serialize
        base_path: The directory path where external files will be saved
        field_path: The current field path (e.g., "latents", "images.0", "metadata.tags")

    Returns:
        A JSON-serializable representation of the object
    """
    # Handle tuple - must be checked before other types
    if isinstance(obj, tuple):
        serialized_items = [
            _serialize_with_field_names(item, base_path, f"{field_path}.{i}")
            for i, item in enumerate(obj)
        ]
        return {"__type__": "tuple", "value": serialized_items}

    # Handle torch.Tensor
    if isinstance(obj, torch.Tensor):
        # Clean field_path for filename (remove leading dot)
        clean_path = field_path.lstrip(".")
        file_name = clean_path if clean_path else "tensor"

        # Check if this tensor should be saved as an image
        if clean_path.endswith("image") or clean_path.endswith("images"):
            # Convert tensor to PIL image
            pil_image = tensor_to_pil(obj)
            file_path = os.path.join(base_path, f"{file_name}.png")
            pil_image.save(file_path)
            return {"__type__": "image", "file": f"{file_name}.png"}
        else:
            # Save as tensor
            file_path = os.path.join(base_path, f"{file_name}.pt")
            torch.save(obj, file_path)
            return {"__type__": "tensor", "file": f"{file_name}.pt"}

    # Handle PIL.Image
    if isinstance(obj, Image.Image):
        clean_path = field_path.lstrip(".")
        file_name = clean_path if clean_path else "image"
        file_path = os.path.join(base_path, f"{file_name}.png")
        obj.save(file_path)
        return {"__type__": "image", "file": f"{file_name}.png"}

    # Handle numpy.ndarray
    if isinstance(obj, np.ndarray):
        clean_path = field_path.lstrip(".")
        file_name = clean_path if clean_path else "array"
        file_path = os.path.join(base_path, f"{file_name}.npy")
        np.save(file_path, obj)
        return {"__type__": "ndarray", "file": f"{file_name}.npy"}

    # Handle dict
    if isinstance(obj, dict):
        return {
            key: _serialize_with_field_names(
                value, base_path, f"{field_path}.{key}" if field_path else key
            )
            for key, value in obj.items()
        }

    # Handle list
    if isinstance(obj, list):
        return [
            _serialize_with_field_names(item, base_path, f"{field_path}.{i}")
            for i, item in enumerate(obj)
        ]

    # Handle basic JSON-serializable types
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj

    # Fallback for unknown types
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def custom_json_decode_hook(dct: dict) -> Any:
    """Custom JSON decoder hook that handles special types with __type__ markers."""
    if "__type__" not in dct:
        return dct

    type_name = dct["__type__"]

    if type_name == "tuple":
        return tuple(dct["value"])

    # For tensor, image, and ndarray, we return the dict with type and file info
    # The actual loading is done in RawDirectoryDataset.__getitem__
    return dct


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
            index.json          # Contains all data with references to external files
            tensor_0.pt         # External tensor file
            image_0.png         # External image file
            array_0.npy         # External ndarray file
            ...
        sample_key_2/
            ...
    ```

    Each sample is stored in its own subdirectory named by its unique key.
    The index.json file contains the complete data structure, with special types
    (torch.Tensor, PIL.Image, numpy.ndarray) stored in separate files and referenced
    via __type__ markers.

    Example index.json:
    ```json
    {
        "some_text": "hello",
        "some_number": 42,
        "some_tuple": {"__type__": "tuple", "value": [1, 2, 3]},
        "some_tensor": {"__type__": "tensor", "file": "tensor_0.pt"},
        "some_image": {"__type__": "image", "file": "image_0.png"},
        "some_array": {"__type__": "ndarray", "file": "array_0.npy"}
    }
    ```

    Loads to:
    ```python
    {
        "__key__": "sample_key_1",
        "some_text": "hello",
        "some_number": 42,
        "some_tuple": (1, 2, 3),
        "some_tensor": torch.Tensor(...),
        "some_image": torch.Tensor(...),  # Images are converted to tensors
        "some_array": numpy.ndarray(...)
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

        # Load index.json
        index_json_path = os.path.join(sample_dir_path, "index.json")
        if not os.path.exists(index_json_path):
            logger.warning(
                f"index.json not found in sample {sample_dir_name}, returning empty sample"
            )
            return {"__key__": sample_dir_name}

        with open(index_json_path, encoding="utf-8") as f:
            sample = json.load(f, object_hook=custom_json_decode_hook)

        # Recursively load external files
        sample = self._load_external_files(sample, sample_dir_path)

        # Apply allowed_fields filter
        if self.allowed_fields is not None:
            filtered_sample = {}
            for field_name in self.allowed_fields:
                if field_name in sample:
                    filtered_sample[field_name] = sample[field_name]
            sample = filtered_sample

        sample["__key__"] = sample_dir_name
        return sample

    def _load_external_files(self, obj: Any, sample_dir_path: str) -> Any:
        """Recursively load external files referenced in the data structure."""
        if isinstance(obj, dict):
            if "__type__" in obj:
                type_name = obj["__type__"]
                if type_name == "tensor" and "file" in obj:
                    file_path = os.path.join(sample_dir_path, obj["file"])
                    return torch.load(file_path)
                elif type_name == "image" and "file" in obj:
                    file_path = os.path.join(sample_dir_path, obj["file"])
                    pil_image = Image.open(file_path)
                    return pil_to_tensor(pil_image)
                elif type_name == "ndarray" and "file" in obj:
                    file_path = os.path.join(sample_dir_path, obj["file"])
                    return np.load(file_path)
                # tuple is already handled by custom_json_decode_hook
                return obj
            else:
                return {
                    k: self._load_external_files(v, sample_dir_path)
                    for k, v in obj.items()
                }
        elif isinstance(obj, list):
            return [self._load_external_files(item, sample_dir_path) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(
                self._load_external_files(item, sample_dir_path) for item in obj
            )
        else:
            return obj


class RawDirectoryDataSink(DataSink):
    """
    DataSink that writes samples to a directory with index.json format.
    Each sample is stored in its own subdirectory named by its unique key.

    The sample data is serialized to an index.json file, with special types
    (torch.Tensor, PIL.Image, numpy.ndarray) stored in separate files:

    - torch.Tensor: saved as .pt files
    - PIL.Image: saved as .png files
    - numpy.ndarray: saved as .npy files
    - tuple: serialized with __type__ marker
    - Other types (str, int, float, dict, list, etc.): directly in JSON

    Directory structure:
    ```
    root/
        sample_key_1/
            index.json          # Contains all data with references to external files
            tensor_0.pt         # External tensor file
            image_0.png         # External image file
            array_0.npy         # External ndarray file
            ...
    ```
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

        # Filter fields if needed
        data_to_write = {}
        for field_name, data in item.items():
            if field_name == "__key__":
                continue

            if (
                self.allowed_fields is not None
                and field_name not in self.allowed_fields
            ):
                continue

            data_to_write[field_name] = data

        # Serialize data with field names and write to index.json
        serialized_data = _serialize_with_field_names(data_to_write, sample_dir_path)
        index_json_path = os.path.join(sample_dir_path, "index.json")
        with open(index_json_path, "w", encoding="utf-8") as f:
            json.dump(serialized_data, f, indent=2)

        return True
