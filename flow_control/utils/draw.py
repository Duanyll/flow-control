import colorsys
import hashlib
import os
from typing import Literal

import torch
from PIL import Image, ImageDraw, ImageFont

from .common import pil_to_tensor, tensor_to_pil
from .logging import get_logger, warn_once

logger = get_logger(__name__)

DEFAULT_FONT = os.path.join(os.path.dirname(__file__), "../third_party/hei.TTF")


def load_font(font_size):
    try:
        font = ImageFont.truetype(DEFAULT_FONT, font_size)
    except Exception as e:
        font = ImageFont.load_default()
        warn_once(
            logger, f"Failed to load font from {DEFAULT_FONT}. Using default font."
        )
        logger.exception(e)
    return font


def generate_color(identifier: str | int) -> tuple[int, int, int]:
    """
    Generate a unique RGB color for a given identifier (string or integer).
    """
    if isinstance(identifier, str):
        hash_object = hashlib.md5(identifier.encode())
        hex_dig = hash_object.hexdigest()
        seed = int(hex_dig[:8], 16)
    else:
        seed = identifier

    hue = (seed * 0.618033988749895) % 1

    saturation = 0.95
    value = 0.95

    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return int(r * 255), int(g * 255), int(b * 255)


def draw_bbox_on_image(
    image: torch.Tensor,
    boxes: list[tuple[int, int, int, int]],
    labels: list[str] | None = None,
    mode: Literal["outline", "fill"] = "outline",
    line_thickness: int = 4,
    opacity: float = 1.0,
    font_size: int = 25,
) -> torch.Tensor:
    """
    Draw bounding boxes on an image tensor with unique colors for each box.
    """
    if len(boxes) == 0:
        return image

    pil_img = tensor_to_pil(image).convert("RGBA")
    img_w, img_h = pil_img.size

    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font = load_font(font_size)

    for i, box in enumerate(boxes):
        top, bottom, left, right = box

        top = max(0, min(img_h, top))
        bottom = max(0, min(img_h, bottom))
        left = max(0, min(img_w, left))
        right = max(0, min(img_w, right))

        if right <= left or bottom <= top:
            continue

        label = labels[i] if labels and i < len(labels) else None
        color_seed = label if label else i
        rgb = generate_color(color_seed)

        alpha_int = int(255 * opacity)
        rgba_color = (*rgb, alpha_int)

        xy = [left, top, right, bottom]

        if mode == "fill":
            draw.rectangle(xy, fill=rgba_color, outline=None)
        else:
            draw.rectangle(xy, outline=rgba_color, width=line_thickness)

        if label:
            text_bbox = draw.textbbox((0, 0), str(label), font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]

            padding = 4
            full_text_w = text_w + padding * 2
            full_text_h = text_h + padding * 2

            text_x = left
            text_y = top - full_text_h

            if text_y < 0:
                text_y = top

            if text_y < 0:
                text_y = 0

            if text_x + full_text_w > img_w:
                text_x = right - full_text_w
                if text_x < 0:
                    text_x = img_w - full_text_w

            text_bg_opacity = min(255, int(alpha_int * 1.5)) if opacity < 1.0 else 255
            text_bg_color = (*rgb, text_bg_opacity)

            draw.rectangle(
                [text_x, text_y, text_x + full_text_w, text_y + full_text_h],
                fill=text_bg_color,
            )

            draw.text(
                (text_x + padding, text_y + padding),
                label,
                fill=(255, 255, 255, 255),
                font=font,
            )

    result = Image.alpha_composite(pil_img, overlay)

    return pil_to_tensor(result.convert("RGB"))
