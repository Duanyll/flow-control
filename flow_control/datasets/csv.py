import csv
import json
import os
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from flow_control.utils.common import pil_to_tensor
from flow_control.utils.logging import get_logger

logger = get_logger(__name__)


CsvColumnType = Literal["int", "float", "str", "json", "attachment", "attachment_list"]


class CsvDataset(Dataset):
    def __init__(
        self,
        path: str,
        column_types: dict[str, CsvColumnType],
        attachments_dir: str | None = None,
    ):
        self.path = path
        self.column_types = column_types
        if not attachments_dir:
            attachments_dir = os.path.dirname(path)
        self.attachments_dir = attachments_dir

        self.data = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        processed_row = {}
        for column, value in row.items():
            column_type = self.column_types.get(column, "str")
            if column_type == "int":
                processed_row[column] = int(value)
            elif column_type == "float":
                processed_row[column] = float(value)
            elif column_type == "json":
                processed_row[column] = json.loads(value)
            elif column_type == "attachment":
                attachment_path = os.path.join(self.attachments_dir, value)
                processed_row[column] = self.load_attachment(attachment_path)
            elif column_type == "attachment_list":
                attachment_paths = [
                    os.path.join(self.attachments_dir, v.strip())
                    for v in value.split(";")
                ]
                processed_row[column] = [
                    self.load_attachment(p) for p in attachment_paths
                ]
            else:
                processed_row[column] = value
        if "__key__" not in processed_row:
            processed_row["__key__"] = str(idx)
        return processed_row

    def load_attachment(self, path: str):
        extname = os.path.splitext(path)[1].lower()
        if extname in [".npy", ".npz"]:
            return torch.from_numpy(np.load(path)).clone()
        elif extname in [".pt", ".pth"]:
            return torch.load(path)
        else:
            try:
                image = Image.open(path)
                return pil_to_tensor(image)
            except Exception as e:
                logger.error(f"Failed to load attachment at {path}: {e}")
                return None
