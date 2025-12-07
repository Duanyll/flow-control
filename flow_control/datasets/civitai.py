import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from einops import rearrange

from flow_control.utils.logging import get_logger
logger = get_logger(__name__)

class CivitaiDataset(Dataset):
    """
    Simple Civitai format dataset.

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
                if entry.is_file() and entry.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_files.append(entry.name)
        self.image_files.sort()
        logger.info(f"CivitaiDataset initialized with {len(self.image_files)} items from {path}")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index) -> dict:
        image_file = self.image_files[index]
        base_name = os.path.splitext(image_file)[0]
        text_file = f"{base_name}.txt"
        
        # Load image
        image_path = os.path.join(self.path, image_file)
        image = Image.open(image_path).convert("RGB")
        image = torch.from_numpy(np.array(image)) / 255.0  # Normalize to [0, 1]
        image = rearrange(image, "h w c -> 1 c h w")
        
        # Load prompt
        text_path = os.path.join(self.path, text_file)
        if not os.path.exists(text_path):
            logger.warning(f"Image {image_file} has no corresponding text file. Using empty prompt.")
            prompt = ""
        else:
            with open(text_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
        
        return {
            "__key__": image_file,
            "clean_image": image,
            "prompt": prompt
        }