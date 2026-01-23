import os

import torch
from PIL import Image
from torch.utils.data import Dataset

from flow_control.utils.common import pil_to_tensor
from flow_control.utils.logging import get_logger

logger = get_logger(__name__)


class PlainDirectoryDataset(Dataset):
    """
    Plain directory dataset (Civitai format)

    Folder structure:

    - `000.jpg/png` -> `clean_image`
    - `000.txt` -> `prompt`
    - `001.jpg/png` -> `clean_image`
    - `001.txt` -> `prompt`
    - ...
    """

    def __init__(self, path: str):
        self.path = path
        self.image_files = []
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file() and entry.name.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                ):
                    self.image_files.append(entry.name)
        self.image_files.sort()
        logger.info(
            f"PlainDirectoryDataset initialized with {len(self.image_files)} items from {path}"
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index) -> dict:
        image_file = self.image_files[index]
        base_name = os.path.splitext(image_file)[0]
        text_file = f"{base_name}.txt"

        # Load image
        image_path = os.path.join(self.path, image_file)
        image = Image.open(image_path).convert("RGB")
        image = pil_to_tensor(image).to(torch.bfloat16)

        data = {"__key__": image_file, "clean_image": image}

        # Load prompt
        text_path = os.path.join(self.path, text_file)
        if os.path.isfile(text_path):
            with open(text_path, encoding="utf-8") as f:
                prompt = f.read().strip()
            data["prompt"] = prompt

        return data
