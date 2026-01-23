import math
from typing import Annotated, Any

import torch
import torch.nn.functional as F
from pydantic import BeforeValidator, PlainSerializer


def parse_resolution_list(resolutions: Any) -> list[tuple[int, int]]:
    """
    Parses a list of resolutions. Handles strings like "1024x1024"
    and passes through already valid tuples/lists for idempotency.
    """
    if not isinstance(resolutions, list):
        # Handle case where user might pass a single string or tuple by mistake
        if isinstance(resolutions, (str, tuple)):
            resolutions = [resolutions]
        else:
            raise ValueError(f"Expected a list, got {type(resolutions)}")

    result = []
    for res in resolutions:
        # Pydantic validation is idempotent; if it's already a tuple/list, pass it through.
        if isinstance(res, (tuple, list)) and len(res) == 2:
            result.append((int(res[0]), int(res[1])))
            continue

        try:
            height_str, width_str = str(res).lower().split("x")
            width = int(width_str)
            height = int(height_str)
            result.append((height, width))
        except ValueError as e:
            raise ValueError(
                f"Invalid resolution format: {res}. Expected format 'HEIGHTxWIDTH'."
            ) from e
    return result


def serialize_resolution_list(resolutions: list[tuple[int, int]]) -> list[str]:
    return [f"{height}x{width}" for height, width in resolutions]


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
    target_resolution should be in (height, width) format.
    If crop is True, the image will be center-cropped to match the target aspect ratio before resizing.
    If crop is False, the image will be stretched to the target resolution.
    """
    if image.dim() != 4:
        raise ValueError("Image tensor must have 4 dimensions (B, C, H, W).")

    _, _, orig_height, orig_width = image.shape
    target_height, target_width = target_resolution

    # Avoid division by zero
    if target_height == 0 or target_width == 0:
        raise ValueError("Target resolution dimensions must be > 0")

    orig_aspect_ratio = orig_width / orig_height
    target_aspect_ratio = target_width / target_height

    if crop:
        # Determine crop window in the original image space
        if orig_aspect_ratio > target_aspect_ratio:
            # Image is wider than target: Crop width
            # We want new_width / orig_height = target_width / target_height
            new_width = int(round(orig_height * target_aspect_ratio))
            # Ensure we don't exceed original dimensions due to rounding
            new_width = min(new_width, orig_width)
            left = (orig_width - new_width) // 2
            image = image[:, :, :, left : left + new_width]
        else:
            # Image is taller than target: Crop height
            # We want orig_width / new_height = target_width / target_height
            new_height = int(round(orig_width / target_aspect_ratio))
            new_height = min(new_height, orig_height)
            top = (orig_height - new_height) // 2
            image = image[:, :, top : top + new_height, :]

    # Interpolate to the exact target resolution
    # Note: align_corners=False is standard for image resizing to avoid shifting
    resized_image = F.interpolate(
        image, size=(target_height, target_width), mode="bilinear", align_corners=False
    )
    return resized_image


def resize_to_closest_resolution(
    image: torch.Tensor, target_resolutions: ResolutionList, crop: bool = True
) -> torch.Tensor:
    """
    Resize an image to the closest aspect ratio from a list of target resolutions.
    """
    if image.dim() != 4:
        raise ValueError("Image tensor must have 4 dimensions (B, C, H, W).")

    if not target_resolutions:
        raise ValueError("target_resolutions list cannot be empty.")

    _, _, orig_height, orig_width = image.shape
    orig_aspect_ratio = orig_width / orig_height

    closest_resolution = target_resolutions[0]
    smallest_diff = float("inf")

    for target_height, target_width in target_resolutions:
        target_aspect_ratio = target_width / target_height
        aspect_ratio_diff = abs(orig_aspect_ratio - target_aspect_ratio)

        if aspect_ratio_diff < smallest_diff:
            smallest_diff = aspect_ratio_diff
            closest_resolution = (target_height, target_width)

    return resize_to_resolution(image, target_resolution=closest_resolution, crop=crop)


def resize_to_multiple_of(
    image: torch.Tensor, multiple: int, crop: bool = True, pixels: int = 0
) -> torch.Tensor:
    """
    Resize an image so that its dimensions are multiples of a given number.

    - If pixels > 0: Resizes the image so total pixel count is approx 'pixels',
      dimensions are multiples of 'multiple', and aspect ratio is preserved (cropped if crop=True).
    - If pixels == 0: Resizes (or crops) the image to the nearest multiple of the ORIGINAL dimensions.
    """
    if image.dim() != 4:
        raise ValueError("Image tensor must have 4 dimensions (B, C, H, W).")

    _, _, orig_height, orig_width = image.shape

    # --- Step 1: Calculate Target Height/Width ---
    if pixels > 0:
        # Scale to match approximate pixel count while preserving AR
        aspect_ratio = orig_width / orig_height
        # h * (h * ar) = pixels  =>  h = sqrt(pixels / ar)
        target_height_float = math.sqrt(pixels / aspect_ratio)
        target_width_float = target_height_float * aspect_ratio

        # Round to nearest multiple
        target_height = int(round(target_height_float / multiple) * multiple)
        target_width = int(round(target_width_float / multiple) * multiple)

        # Safety: ensure at least 1 block exists
        target_height = max(multiple, target_height)
        target_width = max(multiple, target_width)

    else:
        # Just snap original dimensions to the nearest multiple
        # Usually we floor or round. Flooring is safer to avoid creating "fake" pixels if not intended.
        # But rounding is better for maintaining size. Let's use standard rounding logic.
        target_height = int(round(orig_height / multiple) * multiple)
        target_width = int(round(orig_width / multiple) * multiple)

        if target_height == 0:
            target_height = multiple
        if target_width == 0:
            target_width = multiple

    # --- Step 2: Delegate to resize_to_resolution ---
    # We reuse the logic in resize_to_resolution to handle the Aspect Ratio cropping
    # and interpolation correctly.
    return resize_to_resolution(
        image, target_resolution=(target_height, target_width), crop=crop
    )
