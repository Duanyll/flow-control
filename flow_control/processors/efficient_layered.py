import asyncio
import math
from typing import Any, Literal, NotRequired

import torch
from pydantic import PrivateAttr

from flow_control.utils.common import ensure_alpha_channel, remove_alpha_channel
from flow_control.utils.llm import LLMClient
from flow_control.utils.loaders import HfModelLoader
from flow_control.utils.merge_images import merge_images
from flow_control.utils.resize import (
    resize_to_closest_resolution,
    resize_to_multiple_of,
    resize_to_resolution,
)

from .base import BaseProcessor
from .qwen import QwenImageProcessor

_DEFAULT_CAPTION_PROMPT_FG_CN = "请你给我给出的图片生成一句话的描述。你要描述的图片是从平面设计作品中提取出的部分设计元素，你只用关注图片的前景部分。如果图片中包含文字，你必须在描述中用双引号完整地给出图片中的文字内容。直接输出最终结果，不要加额外的解释。"
_DEFAULT_CAPTION_PROMPT_FG_EN = 'Task: Describe the image in exactly one sentence. Context: The image is a specific design element extracted from a larger graphic design work. Requirements: 1. Focus exclusively on the foreground. 2. If text is present, include the text content verbatim inside "double quotes". 3. Output ONLY the description string. Do not include introductory or concluding remarks.'
_DEFAULT_CAPTION_PROMPT_BG_CN = "请你给我给出的图片生成一句话的描述。你要描述的图片是从平面设计作品中提取出的背景部分，它可能是纯色背景，也可能有一些图案。直接输出最终结果，不要加额外的解释。"
_DEFAULT_CAPTION_PROMPT_BG_EN = "Task: Describe the image in exactly one sentence. Context: The image is the background layer extracted from a graphic design work. Requirements: 1. Analyze the visual style, noting whether it is a solid color, a gradient, a texture, or contains specific patterns. 2. Output ONLY the description string. Do not include introductory or concluding remarks."


class EfficientLayeredQwenImageProcessor(QwenImageProcessor):
    class BatchType(BaseProcessor.BatchType):  # type: ignore
        # Required condition inputs
        whole_image: torch.Tensor
        layer_boxes: list[tuple[int, int, int, int]]

        # Optional inputs for training
        base_image: NotRequired[torch.Tensor]
        layer_images: NotRequired[list[torch.Tensor]]

        # Derived condition inputs
        base_caption: NotRequired[str]
        layer_captions: NotRequired[list[str]]

        # Encoded conditions
        image_latents: NotRequired[torch.Tensor]
        text_lengths: NotRequired[list[int]]
        prompt_embeds: NotRequired[torch.Tensor]

    vae: HfModelLoader = HfModelLoader(
        type="diffusers",
        class_name="AutoencoderKLQwenImage",
        pretrained_model_id="Qwen/Qwen-Image-Layered",
        subfolder="vae",
        dtype=torch.bfloat16,
    )
    _vae: Any = PrivateAttr()

    llm: LLMClient

    default_resolution: tuple[int, int] = (1024, 1024)
    resize_mode: Literal["multiple_of", "list"] = "multiple_of"
    multiple_of: int = 32
    pixels: int = 1024 * 1024
    fg_prompt: str = "en"
    bg_prompt: str = "en"

    async def generate_background_caption(self, image: torch.Tensor) -> str:
        prompt = (
            _DEFAULT_CAPTION_PROMPT_BG_EN
            if self.bg_prompt == "en"
            else _DEFAULT_CAPTION_PROMPT_BG_CN
            if self.bg_prompt == "cn"
            else self.bg_prompt
        )
        response, _ = await self.llm.generate(
            prompt, images=[remove_alpha_channel(image)]
        )
        return response.strip()

    async def generate_layer_caption(self, image: torch.Tensor) -> str:
        prompt = (
            _DEFAULT_CAPTION_PROMPT_FG_EN
            if self.fg_prompt == "en"
            else _DEFAULT_CAPTION_PROMPT_FG_CN
            if self.fg_prompt == "cn"
            else self.fg_prompt
        )
        response, _ = await self.llm.generate(
            prompt, images=[remove_alpha_channel(image)]
        )
        return response.strip()

    def resize_image(self, image: torch.Tensor) -> torch.Tensor:
        # Cropping is disabled to make resizing layer box calculation easier
        if self.resize_mode == "list":
            return resize_to_closest_resolution(
                image,
                self.preferred_resolutions,
                crop=False,
            )
        else:
            return resize_to_multiple_of(
                image, self.multiple_of, crop=False, pixels=self.pixels
            )

    def _scale_and_align_layer_boxes(self, layer_boxes, orig_size, new_size):
        orig_h, orig_w = orig_size
        new_h, new_w = new_size
        scale_h = new_h / orig_h
        scale_w = new_w / orig_w
        align = self.multiple_of
        return [
            (
                int(math.floor(top * scale_h / align) * align),
                int(math.ceil(bottom * scale_h / align) * align),
                int(math.floor(left * scale_w / align) * align),
                int(math.ceil(right * scale_w / align) * align),
            )
            for (top, bottom, left, right) in layer_boxes
        ]

    def _stack_all_images(self, whole_image, layer_boxes, layer_images):
        n_layers = len(layer_boxes)
        h, w = whole_image.shape[-2:]
        stacked_images = torch.zeros((n_layers + 1, 4, h, w), device=whole_image.device)
        stacked_images[0] = ensure_alpha_channel(whole_image)[0]
        for i in range(n_layers):
            top, bottom, left, right = layer_boxes[i]
            layer_img = layer_images[i]
            if layer_img.shape[-2:] != (bottom - top, right - left):
                layer_img = resize_to_resolution(
                    layer_img,
                    (bottom - top, right - left),
                    crop=False,
                )
            if layer_img.shape[1] == 3:
                # Insert alpha channel
                alpha_channel = torch.ones(
                    (1, 1, layer_img.shape[2], layer_img.shape[3]),
                    device=layer_img.device,
                )
                layer_img = torch.cat([layer_img, alpha_channel], dim=1)
            stacked_images[i + 1, :, top:bottom, left:right] = layer_img[0]
        return stacked_images

    def _crop_stacked_images(self, stacked_images, layer_boxes):
        cropped_layers = []
        for i in range(1, stacked_images.shape[0]):
            top, bottom, left, right = layer_boxes[i - 1]
            cropped_layer = stacked_images[i : i + 1, :, top:bottom, left:right]
            cropped_layers.append(cropped_layer)
        return cropped_layers

    async def preprocess_batch(self, batch: BatchType):
        # 0. Check length consistency
        if "layer_images" in batch:
            assert len(batch["layer_images"]) == len(batch["layer_boxes"]), (
                "Number of layer images and layer boxes must be the same."
            )
        if "layer_captions" in batch:
            assert len(batch["layer_captions"]) == len(batch["layer_boxes"]), (
                "Number of layer captions and layer boxes must be the same."
            )

        # Start generating `base_caption` (Future)
        base_caption_task = None
        if "base_image" in batch and "base_caption" not in batch:
            base_caption_task = asyncio.create_task(
                self.generate_background_caption(batch["base_image"])
            )

        # Start generating `layer_captions` (Future)
        layer_captions_task = None
        if "layer_images" in batch and "layer_captions" not in batch:
            layer_captions_task = asyncio.gather(
                *[self.generate_layer_caption(img) for img in batch["layer_images"]]
            )

        # Merge `base_image` into `layer_images` (ONLY IMAGES & BOXES PART)
        orig_h, orig_w = batch["whole_image"].shape[-2:]
        has_base_image = "base_image" in batch

        if has_base_image:
            if "layer_images" not in batch:
                batch["layer_images"] = []
            # 立即合并图片，供后续 resize/crop 使用
            batch["layer_images"].insert(0, batch["base_image"])
            batch["layer_boxes"].insert(0, (0, orig_h, 0, orig_w))

            del batch["base_image"]

        # Handle resizing and cropping of `whole_image` (Heavy CPU/GPU)
        if "layer_images" in batch:
            stacked_images = self._stack_all_images(
                batch["whole_image"],
                batch["layer_boxes"],
                batch["layer_images"],
            )
        else:
            stacked_images = batch["whole_image"]

        stacked_images = self.resize_image(stacked_images)
        batch["whole_image"] = stacked_images[0:1]
        new_h, new_w = stacked_images.shape[-2:]
        batch["image_size"] = (new_h, new_w)

        batch["layer_boxes"] = self._scale_and_align_layer_boxes(
            batch["layer_boxes"], (orig_h, orig_w), (new_h, new_w)
        )

        if "layer_images" in batch:
            batch["layer_images"] = self._crop_stacked_images(
                stacked_images, batch["layer_boxes"]
            )
        del stacked_images

        # Encode image latents (Heavy GPU)
        batch["image_latents"] = self.encode_latents(batch["whole_image"])
        if "layer_images" in batch:
            clean_latents_list = []
            for layer_img in batch["layer_images"]:
                latents = self.encode_latents(layer_img)
                clean_latents_list.append(latents)
            batch["clean_latents"] = torch.cat(clean_latents_list, dim=1)

        # Resolve base_caption
        if base_caption_task:
            batch["base_caption"] = await base_caption_task

        # Resolve layer_captions
        if layer_captions_task:
            batch["layer_captions"] = await layer_captions_task

        # Merge `base_caption` into `layer_captions` (CAPTION PART)
        if has_base_image and "base_caption" in batch:
            if "layer_captions" not in batch:
                batch["layer_captions"] = []
            batch["layer_captions"].insert(0, batch["base_caption"])
            del batch["base_caption"]

        # Encode text prompts (Depends on Captions)
        prompt_embeds_list = []
        text_lengths = []
        if "layer_captions" in batch:
            for caption in batch["layer_captions"]:
                embeds = self.encode_prompt(caption)
                prompt_embeds_list.append(embeds)
                text_lengths.append(embeds.shape[1])
        batch["prompt_embeds"] = torch.cat(prompt_embeds_list, dim=1)
        batch["text_lengths"] = text_lengths

        # Compute latent length
        ratio = (self.vae_scale_factor * self.patch_size) ** 2
        batch["latent_length"] = (
            batch["prompt_embeds"].shape[1]
            + batch["image_size"][0] * batch["image_size"][1] // ratio
            + sum(
                (top - bottom) * (right - left) // ratio
                for (top, bottom, left, right) in batch["layer_boxes"]
            )
        )
        return batch

    def make_negative_batch(self, batch):
        raise NotImplementedError("Negative prompts are not supported yet.")

    def decode_output(
        self, output_latent: torch.Tensor, batch: BatchType
    ) -> torch.Tensor:
        ratio = (self.vae_scale_factor * self.patch_size) ** 2
        latent_len_per_image = [
            (bottom - top) * (right - left) // ratio
            for (top, bottom, left, right) in batch["layer_boxes"]
        ]
        split_latents = torch.split(
            output_latent,
            latent_len_per_image,
            dim=1,
        )
        decoded_layers: list[torch.Tensor] = []
        for i, latents in enumerate(split_latents):
            layer_size = batch["layer_boxes"][i]
            decoded_layer = self.decode_latents(
                latents, (layer_size[1] - layer_size[0], layer_size[3] - layer_size[2])
            )
            decoded_layers.append(decoded_layer)
        batch["layer_images"] = decoded_layers
        merged_image = merge_images(decoded_layers)
        return merged_image

    def initialize_latents(self, batch: BatchType, generator=None, device=None):
        ratio = (self.vae_scale_factor * self.patch_size) ** 2
        latent_len_per_image = [
            (bottom - top) * (right - left) // ratio
            for (top, bottom, left, right) in batch["layer_boxes"]
        ]
        total_latent_len = sum(latent_len_per_image)
        latents = torch.randn(
            (1, total_latent_len, 64),
            generator=generator,
            device=device or self.device,
        )
        batch["noisy_latents"] = latents
        return latents
