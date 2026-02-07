import random
from typing import Any, overload

import torch
from PIL import Image, ImageDraw

from .common import pil_to_tensor, tensor_to_pil
from .draw import generate_color, load_font


class BinPacker:
    """
    二维装箱算法 (保持不变)
    """

    def __init__(self):
        self.root = {"x": 0, "y": 0, "w": 0, "h": 0, "used": False}

    def fit(self, blocks):
        len_blocks = len(blocks)
        w = 0 if len_blocks == 0 else blocks[0]["w"]
        h = 0 if len_blocks == 0 else blocks[0]["h"]
        self.root = {"x": 0, "y": 0, "w": w, "h": h, "used": False}
        for block in blocks:
            node = self.find_node(self.root, block["w"], block["h"])
            if node:
                block["fit"] = self.split_node(node, block["w"], block["h"])
            else:
                block["fit"] = self.grow_node(block["w"], block["h"])

    def find_node(self, root, w, h):
        if root["used"]:
            return self.find_node(root["right"], w, h) or self.find_node(
                root["down"], w, h
            )
        elif w <= root["w"] and h <= root["h"]:
            return root
        else:
            return None

    def split_node(self, node, w, h):
        node["used"] = True
        node["down"] = {
            "x": node["x"],
            "y": node["y"] + h,
            "w": node["w"],
            "h": node["h"] - h,
            "used": False,
        }
        node["right"] = {
            "x": node["x"] + w,
            "y": node["y"],
            "w": node["w"] - w,
            "h": h,
            "used": False,
        }
        return node

    def grow_node(self, w, h):
        can_grow_down = w <= self.root["w"]
        can_grow_right = h <= self.root["h"]
        should_grow_right = can_grow_right and (self.root["h"] >= (self.root["w"] + w))
        should_grow_down = can_grow_down and (self.root["w"] >= (self.root["h"] + h))

        if should_grow_right:
            return self.grow_right(w, h)
        elif should_grow_down:
            return self.grow_down(w, h)
        elif can_grow_right:
            return self.grow_right(w, h)
        elif can_grow_down:
            return self.grow_down(w, h)
        else:
            return (
                self.grow_right(w, h)
                if self.root["w"] < self.root["h"]
                else self.grow_down(w, h)
            )

    def grow_right(self, w, h):
        new_root = {
            "x": 0,
            "y": 0,
            "w": self.root["w"] + w,
            "h": self.root["h"],
            "used": True,
            "down": self.root,
            "right": {
                "x": self.root["w"],
                "y": 0,
                "w": w,
                "h": self.root["h"],
                "used": False,
            },
        }
        self.root = new_root
        node = self.find_node(self.root, w, h)
        return self.split_node(node, w, h) if node else None

    def grow_down(self, w, h):
        new_root = {
            "x": 0,
            "y": 0,
            "w": self.root["w"],
            "h": self.root["h"] + h,
            "used": True,
            "down": {
                "x": 0,
                "y": self.root["h"],
                "w": self.root["w"],
                "h": h,
                "used": False,
            },
            "right": self.root,
        }
        self.root = new_root
        node = self.find_node(self.root, w, h)
        return self.split_node(node, w, h) if node else None


def _render_pil(
    blocks, canvas_w, canvas_h, bg_color, border_w, draw_labels=False, label_size=20
):
    """PIL 渲染逻辑"""
    canvas = Image.new("RGBA", (canvas_w, canvas_h), bg_color)
    draw = ImageDraw.Draw(canvas) if (border_w > 0 or draw_labels) else None

    # Load font for labels if needed
    font = None
    if draw_labels and draw:
        try:
            font = load_font(label_size)
        except Exception:
            font = None  # Fall back to default font

    for block in blocks:
        if block.get("fit"):
            x, y = block["fit"]["x"], block["fit"]["y"]
            w, h = block["w"], block["h"]
            canvas.paste(block["img"], (x, y))
            if border_w > 0 and draw:
                # Generate unique color for each block's border
                border_color = generate_color(block["index"])
                draw.rectangle(
                    [x, y, x + w - 1, y + h - 1], outline=border_color, width=border_w
                )
            # Draw label with index
            if draw_labels and draw:
                label_text = str(block["index"])
                # Generate the same color used for border
                label_bg_color = generate_color(block["index"])
                # Add a semi-transparent background for better readability
                padding = 2
                if font:
                    bbox = draw.textbbox((0, 0), label_text, font=font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                else:
                    # Rough estimate for default font
                    text_w = len(label_text) * 8
                    text_h = 12

                # Draw background rectangle with border color
                bg_rect = [
                    x + padding,
                    y + padding,
                    x + padding + text_w + padding * 2,
                    y + padding + text_h + padding * 2,
                ]
                draw.rectangle(bg_rect, fill=(*label_bg_color, 220))

                # Draw text
                text_pos = (x + padding * 2, y + padding * 2)
                draw.text(text_pos, label_text, fill=(255, 255, 255, 255), font=font)
    return canvas


def _make_border_color_tensor(
    border_color: tuple[int, ...],
    channels: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Convert an RGB border color tuple to a (1, C, 1, 1) tensor."""
    if channels == 4:
        border_color = (*border_color, 255)
    elif channels > 4:
        border_color = border_color + (255,) * (channels - 3)

    values = list(border_color[:channels])
    if dtype.is_floating_point:
        values = [c / 255.0 for c in values]
    return torch.tensor(values, dtype=dtype, device=device).view(1, -1, 1, 1)


def _draw_tensor_border(
    canvas: torch.Tensor,
    x: int,
    y: int,
    w: int,
    h: int,
    border_w: int,
    bc_val: torch.Tensor,
) -> None:
    """Draw a border on the tensor canvas via slice assignment."""
    canvas[:, :, y : y + border_w, x : x + w] = bc_val
    canvas[:, :, y + h - border_w : y + h, x : x + w] = bc_val
    canvas[:, :, y : y + h, x : x + border_w] = bc_val
    canvas[:, :, y : y + h, x + w - border_w : x + w] = bc_val


def _draw_labels_on_tensor(
    blocks,
    canvas: torch.Tensor,
    label_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Draw index labels on a tensor canvas by round-tripping through PIL."""
    pil_canvas = tensor_to_pil(canvas)
    draw = ImageDraw.Draw(pil_canvas)
    try:
        font = load_font(label_size)
    except Exception:
        font = None

    for block in blocks:
        if not block.get("fit"):
            continue
        x, y = block["fit"]["x"], block["fit"]["y"]
        label_text = str(block["index"])
        label_bg_color = generate_color(block["index"])
        padding = 2

        if font:
            bbox = draw.textbbox((0, 0), label_text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        else:
            text_w = len(label_text) * 8
            text_h = 12

        bg_rect = [
            x + padding,
            y + padding,
            x + padding + text_w + padding * 2,
            y + padding + text_h + padding * 2,
        ]
        draw.rectangle(bg_rect, fill=(*label_bg_color, 220))
        text_pos = (x + padding * 2, y + padding * 2)
        draw.text(text_pos, label_text, fill=(255, 255, 255, 255), font=font)

    return pil_to_tensor(pil_canvas).to(device=device, dtype=dtype)


def _render_tensor(
    blocks, canvas_w, canvas_h, bg_color, border_w, draw_labels=False, label_size=20
):
    """PyTorch Tensor 渲染逻辑"""
    ref_tensor = blocks[0]["img"]
    dtype = ref_tensor.dtype
    device = ref_tensor.device
    channels = ref_tensor.shape[1]  # shape is (1, C, H, W)

    canvas = torch.zeros((1, channels, canvas_h, canvas_w), dtype=dtype, device=device)

    for block in blocks:
        if not block.get("fit"):
            continue
        x, y = block["fit"]["x"], block["fit"]["y"]
        w, h = block["w"], block["h"]
        canvas[:, :, y : y + h, x : x + w] = block["img"]

        if border_w > 0:
            border_color = generate_color(block["index"])
            bc_val = _make_border_color_tensor(border_color, channels, dtype, device)
            _draw_tensor_border(canvas, x, y, w, h, border_w, bc_val)

    if draw_labels:
        canvas = _draw_labels_on_tensor(blocks, canvas, label_size, device, dtype)

    return canvas


@overload
def merge_images(
    images: list[Image.Image],
    background_color: tuple = (0, 0, 0, 0),
    border_width: int = 0,
    draw_labels: bool = False,
    label_size=20,
) -> Image.Image: ...


@overload
def merge_images(
    images: list[torch.Tensor],
    background_color: tuple = (0, 0, 0, 0),
    border_width: int = 0,
    draw_labels: bool = False,
    label_size=20,
) -> torch.Tensor: ...


def merge_images(
    images,
    background_color=(0, 0, 0, 0),
    border_width: int = 0,
    draw_labels: bool = False,
    label_size: int = 20,
):
    """
    接受 PIL Image 或 torch.Tensor 列表，将其拼接进一个矩形画布。
    支持输入 Tensor 形状为 (1, C, H, W)。

    Args:
        images: PIL Image 或 torch.Tensor 列表
        background_color: 背景颜色
        border_width: 边框宽度（每个图像的边框颜色将自动根据索引生成）
        draw_labels: 是否在每个图像左上角绘制索引标签
        label_size: 标签字体大小
    """
    if not images:
        raise ValueError("Input image list is empty")

    is_tensor = False
    if isinstance(images[0], torch.Tensor):
        is_tensor = True

    # 1. 统一提取尺寸信息
    blocks = []
    for i, img in enumerate(images):
        if isinstance(img, torch.Tensor):
            # Tensor shape: (1, C, H, W) -> w=dim3, h=dim2
            if img.dim() != 4:
                raise ValueError(f"Expected tensor shape (1, C, H, W), got {img.shape}")
            h, w = img.shape[2], img.shape[3]
        else:
            w, h = img.width, img.height

        blocks.append({"w": w, "h": h, "img": img, "index": i})

    # 2. 排序 (确定性)
    blocks.sort(key=lambda x: (x["h"], x["w"], -x["index"]), reverse=True)

    # 3. 运行装箱算法
    packer = BinPacker()
    packer.fit(blocks)

    canvas_w = packer.root["w"]
    canvas_h = packer.root["h"]

    # 4. 根据类型分发渲染
    if is_tensor:
        return _render_tensor(
            blocks,
            canvas_w,
            canvas_h,
            background_color,
            border_width,
            draw_labels,
            label_size,
        )
    else:
        return _render_pil(
            blocks,
            canvas_w,
            canvas_h,
            background_color,
            border_width,
            draw_labels,
            label_size,
        )


# --- 测试用例 ---
if __name__ == "__main__":
    from rich import print

    # --- 测试 1: PIL 模式 ---
    print("--- Test 1: PIL Images ---")
    pil_images = []
    for _ in range(5):
        w, h = random.randint(50, 100), random.randint(50, 100)
        img = Image.new(
            "RGB",
            (w, h),
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
        )
        pil_images.append(img)

    res_pil = merge_images(pil_images, border_width=2, draw_labels=True, label_size=24)
    print(f"PIL Result Size: {res_pil.size}")
    res_pil.save("pil_packed_result.png")
    print("PIL result saved to 'pil_packed_result.png'")
    # res_pil.show()

    # --- 测试 2: PyTorch Tensor 模式 ---
    print("\n--- Test 2: PyTorch Tensors ---")
    tensor_images = []

    # 创建一些随机 tensor: (1, 4, H, W) 模拟 RGBA
    # 注意：这里使用 float (0-1)
    for _ in range(5):
        h, w = random.randint(50, 100), random.randint(50, 100)
        # 随机颜色 tensor
        t = torch.rand((1, 3, h, w), dtype=torch.float32)
        t = torch.cat([t, torch.ones((1, 1, h, w), dtype=torch.float32)], dim=1)
        tensor_images.append(t)

    try:
        res_tensor: Any = merge_images(
            tensor_images,
            border_width=2,
            draw_labels=True,
            label_size=24,
        )

        print(f"Input Tensor Shape: {tensor_images[0].shape}")
        print(f"Output Tensor Shape: {res_tensor.shape}")
        print(f"Output Device: {res_tensor.device}")
        print(f"Output Dtype: {res_tensor.dtype}")

        # 验证结果 (转换回 PIL 看看)
        # For float tensors in 0-1 range, convert to uint8 for PIL
        if res_tensor.dtype.is_floating_point:
            tensor_uint8 = (res_tensor.squeeze(0) * 255).clamp(0, 255).byte()
            disp_img = Image.fromarray(tensor_uint8.permute(1, 2, 0).numpy())
        else:
            disp_img = Image.fromarray(res_tensor.squeeze(0).permute(1, 2, 0).numpy())
        # disp_img.show()
        disp_img.save("tensor_packed_result.png")
        print("Tensor result saved to 'tensor_packed_result.png'")

    except Exception as e:
        print(f"Tensor error: {e}")
        import traceback

        traceback.print_exc()
