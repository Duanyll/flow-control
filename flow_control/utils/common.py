import json
import tomllib
from collections.abc import MutableMapping
from typing import Any, Literal

import kornia
import numpy as np
import torch
import yaml
from einops import rearrange, repeat
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


def make_grid(h, w, device: torch.device) -> torch.Tensor:
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
    for name, _param in module.named_parameters():
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


def deep_cast_float_dtype(data, dtype: torch.dtype):
    return deep_apply_tensor_fn(
        data, lambda x: x.to(dtype) if x.is_floating_point() else x
    )


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


def ensure_alpha_channel(image: torch.Tensor) -> torch.Tensor:
    """
    Ensure that the input image tensor has an alpha channel.
    If it has 3 channels (RGB), add an alpha channel with value 1.
    Args:
        image: Tensor of shape (B, 3, H, W) or (B, 4, H, W)
    Returns:
        Tensor of shape (B, 4, H, W)
    """
    if image.shape[1] == 3:
        alpha_channel = torch.ones(
            (image.shape[0], 1, image.shape[2], image.shape[3]),
            dtype=image.dtype,
            device=image.device,
        )
        image = torch.cat([image, alpha_channel], dim=1)
    return image


def remove_alpha_channel(
    image: torch.Tensor,
    background_color: tuple[float, float, float] | Literal["auto"] = "auto",
) -> torch.Tensor:
    """
    Remove the alpha channel from the input image tensor by compositing it over a background color.
    Args:
        image: Tensor of shape (B, 4, H, W) or (B, 3, H, W)
        background_color: Tuple of (R, G, B) values in [0, 1] or "auto" to use content-aware background
    Returns:
        Tensor of shape (B, 3, H, W)
    """
    if image.shape[1] < 4:
        return image  # No alpha channel to remove
    rgb = image[:, :3, :, :]
    alpha = image[:, 3:4, :, :]
    if background_color == "auto":
        # Get average brightness of non-transparent pixels
        mask = alpha > 0.01
        if mask.sum() == 0:
            background_color = (1.0, 1.0, 1.0)  # Default to white if fully transparent
        else:
            avg_brightness = (rgb * mask).sum(dim=(0, 2, 3)) / mask.sum()
            # Use black or white based on brightness
            background_color = (
                (0.0, 0.0, 0.0) if avg_brightness.mean() > 0.5 else (1.0, 1.0, 1.0)
            )
    background = torch.tensor(background_color, device=image.device)
    background = repeat(
        background, "c -> b c h w", b=image.shape[0], h=image.shape[2], w=image.shape[3]
    )
    composited = rgb * alpha + background * (1 - alpha)
    return composited


def load_config_file(path: str) -> dict:
    """
    Load JSON, YAML or TOML configuration file.
    """
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    elif path.endswith((".yaml", ".yml")):
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    elif path.endswith(".toml"):
        with open(path, "rb") as f:
            return tomllib.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {path}")
