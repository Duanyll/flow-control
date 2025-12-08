from collections.abc import MutableMapping
from typing import Tuple, Any

import kornia
import torch
import numpy as np
from einops import rearrange
from PIL import Image


def flatten_dict(dictionary, parent_key="", separator="_"):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def deep_merge_dicts(dict1, dict2):
    """
    Recursively merge two dictionaries.
    """
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def meshgrid_to_ij(grid: torch.Tensor, h: int, w: int):
    """
    Convert a meshgrid for F.grid_sample to the subscript index [i, j].

    Args:
        grid: (N, 2) tensor with the meshgrid coordinates, ranging from -1 to 1.
        h: height of the image.
        w: width of the image.
    Returns:
        ij: (N, 2) tensor with the subscript indices.
    """
    # Ensure grid is of shape (N, 2)
    assert grid.shape[1] == 2, f"Expected grid shape (N, 2), got {grid.shape}"

    # Extract x and y coordinates
    x = grid[:, 0]  # (N,)
    y = grid[:, 1]  # (N,)

    # Convert from [-1, 1] to [0, w-1] for x and [0, h-1] for y
    i = ((y + 1) / 2) * (h - 1)  # (N,)
    j = ((x + 1) / 2) * (w - 1)  # (N,)

    # Stack to form (N, 2) tensor [i, j]
    ij = torch.stack([i, j], dim=1)

    return ij


def make_grid(h, w, device: torch.device = torch.device("cuda")):
    grid_array = kornia.utils.create_meshgrid(
        h, w, normalized_coordinates=True, device=device
    )
    grid = rearrange(grid_array, "1 h w c -> c h w")  # Shape: 2 H W
    return grid


def ensure_trainable(module):
    """
    Ensure that all parameters of the module are trainable.
    """
    # Check if LoRA is installed
    for name, param in module.named_parameters():
        if "lora" in name:
            break
    else:
        module.requires_grad_(True)


def parse_checkpoint_step(checkpoint_name: str) -> int:
    """
    Parse the checkpoint step from the checkpoint directory name.
    Assumes the checkpoint name is in the format 'checkpoint-{step}'.
    """
    try:
        step_str = checkpoint_name.split("-")[-1]
        return int(step_str)
    except (IndexError, ValueError):
        return -1


def deep_apply_tensor_fn(data, fn) -> Any:
    """
    Recursively apply a function to all tensors in a nested structure.
    Args:
        data: A nested structure (dict, list, tuple) containing tensors.
        fn: A function to apply to each tensor.
    Returns:
        The nested structure with the function applied to all tensors.
    """
    if isinstance(data, torch.Tensor):
        return fn(data)
    elif isinstance(data, dict):
        return {k: deep_apply_tensor_fn(v, fn) for k, v in data.items()}
    elif isinstance(data, list):
        return [deep_apply_tensor_fn(v, fn) for v in data]
    elif isinstance(data, tuple):
        return tuple(deep_apply_tensor_fn(v, fn) for v in data)
    else:
        return data


def deep_move_to_device(data, device: torch.device):
    return deep_apply_tensor_fn(data, lambda x: x.to(device))


def deep_move_to_shared_memory(data):
    return deep_apply_tensor_fn(data, lambda x: x.cpu().share_memory_())


def pil_to_tensor(image) -> torch.Tensor:
    """
    Convert a PIL Image to a normalized torch Tensor.
    Args:
        image: PIL Image
    Returns:
        Tensor of shape (1, C, H, W) with values in [0, 1]
    """
    image = torch.from_numpy(np.array(image)) / 255.0  # Normalize to [0, 1]
    image = rearrange(image, "h w c -> 1 c h w")
    return image


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a normalized torch Tensor to a PIL Image.
    Args:
        tensor: Tensor of shape (1, C, H, W) or (C, H, W) with values in [0, 1]
    Returns:
        PIL Image
    """
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.clamp(0, 1)  # Ensure values are in [0, 1]
    tensor = rearrange(tensor, "c h w -> h w c")
    array = (tensor.detach().float().cpu().numpy() * 255).astype(np.uint8)
    image = Image.fromarray(array)
    return image
