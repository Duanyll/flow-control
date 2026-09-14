import os

import torch
from PIL import Image

from flow_control.data.rows import KEY, Row
from flow_control.utils.logging import get_logger
from flow_control.utils.tensor import pil_to_tensor

from .base import source_registry

logger = get_logger(__name__)


@source_registry.register("plain_directory")
class PlainDirectorySource:
    """Plain directory (Civitai layout): ``000.jpg/png`` -> ``clean_image``,
    ``000.txt`` -> ``prompt``, keyed by the image file name."""

    def __init__(self, path: str, is_rgba: bool | None = None):
        self.path = path
        self.image_files: list[str] = []
        self.is_rgba = is_rgba
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file() and entry.name.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                ):
                    self.image_files.append(entry.name)
        self.image_files.sort()
        logger.info(
            f"PlainDirectorySource initialized with {len(self.image_files)} items from {path}"
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> Row:
        image_file = self.image_files[index]
        base_name = os.path.splitext(image_file)[0]

        image = Image.open(os.path.join(self.path, image_file))
        if self.is_rgba is True:
            image = image.convert("RGBA")
        elif self.is_rgba is False:
            image = image.convert("RGB")
        row: Row = {
            KEY: image_file,
            "clean_image": pil_to_tensor(image).to(torch.bfloat16),
        }

        text_path = os.path.join(self.path, f"{base_name}.txt")
        if os.path.isfile(text_path):
            with open(text_path, encoding="utf-8") as f:
                row["prompt"] = f.read().strip()
        return row
