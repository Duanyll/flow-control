from collections.abc import MutableMapping
from typing import Tuple

import kornia
import torch
from einops import rearrange


def flatten_dict(dictionary, parent_key='', separator='_'):
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


PackedBoolTensor = Tuple[torch.Tensor, torch.Size]
MaybePackedBoolTensor = PackedBoolTensor | torch.Tensor

def pack_bool_tensor(tensor: torch.Tensor) -> PackedBoolTensor:
    if torch.is_floating_point(tensor):
        tensor = tensor > 0.5
    tensor = tensor.to(torch.bool)  # 转换为bool类型
    
    flat = tensor.flatten()
    pad_len = (8 - flat.numel() % 8) % 8  # 需要填充的位数，使其能整除8
    if pad_len > 0:
        flat = torch.cat([flat, torch.zeros(pad_len, dtype=torch.bool)])

    flat = flat.view(-1, 8)  # 每组8个bool
    packed = torch.sum(flat * (1 << torch.arange(8, dtype=torch.uint8)), dim=1)
    return packed, tensor.shape  # 返回原始形状用于解码


def unpack_bool_tensor(
    packed: torch.Tensor, original_shape: torch.Size
) -> torch.Tensor:
    unpacked = ((packed.unsqueeze(1) >> torch.arange(8)) & 1).to(torch.bool)
    unpacked = unpacked.view(-1)[: torch.tensor(original_shape).prod()]
    return unpacked.view(original_shape)


def find_closest_resolution(h, w, all_resolutions) -> Tuple[int, int]:
    """Find the closest supported resolution to the image's dimensions."""
    aspect_ratio = w / h
    min_diff = float("inf")
    best_res = (0, 0)
    
    all_resolutions = [(res[1], res[0]) for res in all_resolutions] + list(all_resolutions)

    for res in all_resolutions:
        res_height, res_width = res
        res_aspect = res_width / res_height
        # Calculate difference in aspect ratio and total pixels
        aspect_diff = abs(aspect_ratio - res_aspect)
        size_diff = abs(w * h - res_width * res_height)
        total_diff = aspect_diff + size_diff / (
            w * h
        )  # Normalize size difference

        if total_diff < min_diff:
            min_diff = total_diff
            best_res = (res_height, res_width)

    return best_res


def crop_and_resize_image(image, target_size):
    """
    Crop and resize image to target size while maintaining aspect ratio.
    image: (1, C, H, W) or (C, H, W) or (H, W)
    """
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    elif len(image.shape) == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    
    img_height, img_width = image.shape[-2:]
    target_height, target_width = target_size
    # Calculate aspect ratios
    img_aspect = img_width / img_height
    target_aspect = target_width / target_height
    if img_aspect > target_aspect:
        # Image is wider than target, crop width
        new_width = int(img_height * target_aspect)
        left = (img_width - new_width) // 2
        image = image[:, :, :, left : left + new_width]
    else:
        # Image is taller than target, crop height
        new_height = int(img_width / target_aspect)
        top = (img_height - new_height) // 2
        image = image[:, :, top : top + new_height, :]
    # Resize to target size
    image = torch.nn.functional.interpolate(
        image, size=(target_height, target_width), mode="bilinear", align_corners=False
    )
    return image.squeeze()  # Remove batch dimension

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

def cast_trainable_parameters(module: torch.nn.Module, dtype: torch.dtype) -> None:
    r"""
    Casts the trainable parameters of a given module to a specified dtype.
    Args:
        module (`torch.nn.Module`):
            The module to cast the parameters of.
        dtype (`torch.dtype`):
            The dtype to cast the parameters to.
    """
    for name, param in module.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(dtype=dtype)

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