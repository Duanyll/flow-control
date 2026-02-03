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
    基于字符串或索引生成唯一的、视觉上明显的 RGB 颜色。
    """
    if isinstance(identifier, str):
        # 使用 md5 hash 确保同一个 label 总是对应同一个颜色
        hash_object = hashlib.md5(identifier.encode())
        hex_dig = hash_object.hexdigest()
        # 【修正关键点】只取前 8 位 16 进制字符 (32-bit 整数)
        # 这避免了超大整数乘以浮点数时的精度丢失问题
        seed = int(hex_dig[:8], 16)
    else:
        seed = identifier

    # 使用黄金分割比生成分布均匀的色相 (Hue)
    # 0.618... 能让相邻的 seed (0, 1, 2...) 产生色相上差异最大的颜色
    hue = (seed * 0.618033988749895) % 1

    # 提高饱和度(S)和亮度(V)以确保颜色鲜艳
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

    # 1. 转换图像为 PIL RGBA 模式以便处理透明度
    pil_img = tensor_to_pil(image).convert("RGBA")
    img_w, img_h = pil_img.size

    # 创建一个用于绘制 Box 的透明图层
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # 加载字体
    font = load_font(font_size)

    for i, box in enumerate(boxes):
        # 解析坐标 (Prompt 定义为: top, bottom, left, right)
        top, bottom, left, right = box

        # 2. 坐标裁剪 (Clipping)
        # 确保坐标在图像范围内
        top = max(0, min(img_h, top))
        bottom = max(0, min(img_h, bottom))
        left = max(0, min(img_w, left))
        right = max(0, min(img_w, right))

        # 如果 box 无效（宽度或高度 <= 0），跳过
        if right <= left or bottom <= top:
            continue

        # 获取 Label 和 颜色
        label = labels[i] if labels and i < len(labels) else None
        color_seed = label if label else i
        rgb = generate_color(color_seed)

        # 计算带透明度的颜色
        alpha_int = int(255 * opacity)
        rgba_color = (*rgb, alpha_int)

        # PIL rectangle 需要 (x0, y0, x1, y1) -> (left, top, right, bottom)
        xy = [left, top, right, bottom]

        # 3. 绘制 Box
        if mode == "fill":
            draw.rectangle(xy, fill=rgba_color, outline=None)
        else:
            # Outline 模式：为了保持透明度叠加正确，我们在 overlay 上绘制
            # 注意：PIL 的 width 参数是向内和向外延伸的，如果贴边可能会被切掉一半
            # 这里如果不做偏移，边缘的线条可能会变细。
            # 简单的处理是不做偏移，依赖 overlay 合成。
            draw.rectangle(xy, outline=rgba_color, width=line_thickness)

        # 4. 绘制 Label (确保位置可见)
        if label:
            # 获取文字大小
            text_bbox = draw.textbbox((0, 0), str(label), font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]

            # 增加一点 Padding
            padding = 4
            full_text_w = text_w + padding * 2
            full_text_h = text_h + padding * 2

            # 默认位置：Box 左上角上方
            text_x = left
            text_y = top - full_text_h

            # 边界检查与调整：

            # Case A: 如果上方空间不足，画在 Box 内部（Top 往下）
            if text_y < 0:
                text_y = top

            # Case B: 如果 Box 本身就在最顶端且非常窄，画在 Box 下方？
            # 简单策略：如果放在 Box 内部还是 < 0 (极少见)，强制置为 0
            if text_y < 0:
                text_y = 0

            # Case C: 如果右侧超出图片，向左对齐
            if text_x + full_text_w > img_w:
                text_x = right - full_text_w
                # 如果 Box 很窄，导致向左对齐后跑出左边界，则强制靠最右侧
                if text_x < 0:
                    text_x = img_w - full_text_w

            # 绘制文字背景 (通常使用不透明背景以保证文字清晰，或者使用带透明度的背景)
            # 这里为了清晰度，使用略微透明的背景，但比 Box 更不透明
            text_bg_opacity = min(255, int(alpha_int * 1.5)) if opacity < 1.0 else 255
            text_bg_color = (*rgb, text_bg_opacity)

            # 绘制背景矩形
            draw.rectangle(
                [text_x, text_y, text_x + full_text_w, text_y + full_text_h],
                fill=text_bg_color,
            )

            # 绘制文字 (白色通常对比度最高，如果背景太亮可以考虑自动反色，这里简单使用白色)
            draw.text(
                (text_x + padding, text_y + padding),
                label,
                fill=(255, 255, 255, 255),
                font=font,
            )

    # 5. 合成图像 (Alpha Composite)
    # 将 overlay 叠加到原图上
    result = Image.alpha_composite(pil_img, overlay)

    # 转回 Tensor 并保留原始格式 (RGB)
    return pil_to_tensor(result.convert("RGB"))
