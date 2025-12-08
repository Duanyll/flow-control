from typing import Annotated
import torch
from pydantic import BeforeValidator, PlainSerializer

def parse_resolution_list(resolutions: list[str]) -> list[tuple[int, int]]:
    result = []
    for res in resolutions:
        try:
            width_str, height_str = res.lower().split('x')
            width = int(width_str)
            height = int(height_str)
            result.append((width, height))
        except ValueError:
            raise ValueError(f"Invalid resolution format: {res}. Expected format 'WIDTHxHEIGHT'.")
    return result

def serialize_resolution_list(resolutions: list[tuple[int, int]]) -> list[str]:
    return [f"{width}x{height}" for width, height in resolutions]

ResolutionList = Annotated[
    list[tuple[int, int]],
    BeforeValidator(parse_resolution_list),
    PlainSerializer(serialize_resolution_list, return_type=list),
]

def resize_to_resolution(
    image: torch.Tensor, target_resolution: tuple[int, int], crop: bool = True
) -> torch.Tensor:
    """
    Resize an image to a specific resolution.
    If crop is True, the image will be center-cropped to match the target aspect ratio before resizing.
    """
    if image.dim() != 4:
        raise ValueError("Image tensor must have 4 dimensions (B, C, H, W).")
    
    _, _, orig_height, orig_width = image.shape
    target_width, target_height = target_resolution
    orig_aspect_ratio = orig_width / orig_height
    target_aspect_ratio = target_width / target_height
    if crop:
        if orig_aspect_ratio > target_aspect_ratio:
            new_width = int(orig_height * target_aspect_ratio)
            left = (orig_width - new_width) // 2
            image = image[:, :, :, left:left + new_width]
        else:
            new_height = int(orig_width / target_aspect_ratio)
            top = (orig_height - new_height) // 2
            image = image[:, :, top:top + new_height, :]
    resized_image = torch.nn.functional.interpolate(
        image,
        size=(target_height, target_width),
        mode='bilinear',
        align_corners=False
    )
    return resized_image


def resize_to_closest_resolution(
    image: torch.Tensor,
    target_resolutions: list[tuple[int, int]],
    crop: bool = True
) -> torch.Tensor:
    """
    Resize an image to the closest aspect ratio from a list of target resolutions.
    If crop is True, the image will be center-cropped to match the target aspect ratio before resizing.
    """

    if image.dim() != 4:
        raise ValueError("Image tensor must have 4 dimensions (B, C, H, W).")
    
    _, _, orig_height, orig_width = image.shape
    orig_aspect_ratio = orig_width / orig_height

    closest_resolution = (0, 0)
    smallest_diff = float('inf')

    for target_width, target_height in target_resolutions:
        target_aspect_ratio = target_width / target_height
        aspect_ratio_diff = abs(orig_aspect_ratio - target_aspect_ratio)

        if aspect_ratio_diff < smallest_diff:
            smallest_diff = aspect_ratio_diff
            closest_resolution = (target_width, target_height)

    target_width, target_height = closest_resolution

    return resize_to_resolution(
        image,
        target_resolution=(target_width, target_height),
        crop=crop
    )


def resize_to_multiple_of(image: torch.Tensor, multiple: int, crop: bool = True) -> torch.Tensor:
    """
    Resize an image so that its dimensions are multiples of a given number.
    If crop is True, the image will be center-cropped to the nearest multiple before resizing.
    """

    if image.dim() != 4:
        raise ValueError("Image tensor must have 4 dimensions (B, C, H, W).")
    
    _, _, orig_height, orig_width = image.shape

    target_width = (orig_width // multiple) * multiple
    target_height = (orig_height // multiple) * multiple

    if crop:
        left = (orig_width - target_width) // 2
        top = (orig_height - target_height) // 2
        image = image[:, :, top:top + target_height, left:left + target_width]

    resized_image = torch.nn.functional.interpolate(
        image,
        size=(target_height, target_width),
        mode='bilinear',
        align_corners=False
    )

    return resized_image