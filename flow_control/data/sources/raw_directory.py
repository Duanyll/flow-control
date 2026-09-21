import json
import os
from typing import Any

import numpy as np
import torch
from PIL import Image

from flow_control.data.rows import KEY, Row
from flow_control.utils.logging import get_logger
from flow_control.utils.tensor import pil_to_tensor

from .base import source_registry

logger = get_logger(__name__)


def custom_json_decode_hook(dct: dict) -> Any:
    """JSON object hook that materializes ``__type__: tuple``; tensor / image /
    ndarray markers stay dicts and are loaded by ``RawDirectorySource``."""
    if dct.get("__type__") == "tuple":
        return tuple(dct["value"])
    return dct


@source_registry.register("raw_directory")
class RawDirectorySource:
    """Read-only reader for the ``raw_directory`` layout (one sub-directory per key).

    ```
    root/
        sample_key_1/
            index.json          # full record; special values reference the files below
            tensor_0.pt         # {"__type__": "tensor", "file": ..., "dtype": ...}
            image_0.png         # {"__type__": "image", "file": ...} -> tensor via pil_to_tensor
            array_0.npy         # {"__type__": "ndarray", "file": ...}
        sample_key_2/
            ...
    ```

    Tuples are stored as ``{"__type__": "tuple", "value": [...]}``. The writer for
    this layout was removed with the cache rework; existing directories stay readable.
    """

    def __init__(self, path: str, allowed_fields: list[str] | None = None):
        self.path = path
        self.sample_dirs: list[str] = []
        self.allowed_fields = allowed_fields
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_dir():
                    self.sample_dirs.append(entry.name)
        self.sample_dirs.sort()
        logger.info(f"Loaded {len(self.sample_dirs)} samples from directory: {path}")

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, index: int) -> Row:
        sample_dir_name = self.sample_dirs[index]
        sample_dir_path = os.path.join(self.path, sample_dir_name)

        index_json_path = os.path.join(sample_dir_path, "index.json")
        if not os.path.exists(index_json_path):
            logger.warning(
                f"index.json not found in sample {sample_dir_name}, returning empty sample"
            )
            return {KEY: sample_dir_name}

        with open(index_json_path, encoding="utf-8") as f:
            sample = json.load(f, object_hook=custom_json_decode_hook)

        sample = self._load_external_files(sample, sample_dir_path)

        if self.allowed_fields is not None:
            sample = {k: sample[k] for k in self.allowed_fields if k in sample}

        sample.pop("__key__", None)
        sample[KEY] = sample_dir_name
        return sample

    def _load_external_files(self, obj: Any, sample_dir_path: str) -> Any:
        """Recursively load external files referenced in the data structure."""
        if isinstance(obj, dict):
            if "__type__" in obj:
                type_name = obj["__type__"]
                if type_name == "tensor" and "file" in obj:
                    file_path = os.path.join(sample_dir_path, obj["file"])
                    tensor = torch.load(file_path, weights_only=True)
                    if tensor.is_floating_point() and "dtype" in obj:
                        dtype_name = obj["dtype"].split(".")[-1]
                        stored_dtype = getattr(torch, dtype_name, None)
                        if stored_dtype is not None:
                            tensor = tensor.to(stored_dtype)
                    return tensor
                elif type_name == "image" and "file" in obj:
                    file_path = os.path.join(sample_dir_path, obj["file"])
                    return pil_to_tensor(Image.open(file_path))
                elif type_name == "ndarray" and "file" in obj:
                    file_path = os.path.join(sample_dir_path, obj["file"])
                    return np.load(file_path)
                # tuple is already handled by custom_json_decode_hook
                return obj
            return {
                k: self._load_external_files(v, sample_dir_path) for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [self._load_external_files(item, sample_dir_path) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(
                self._load_external_files(item, sample_dir_path) for item in obj
            )
        return obj
