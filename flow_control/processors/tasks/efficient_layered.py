import asyncio
import math
from typing import Literal, NotRequired

import torch
from einops import repeat

from flow_control.utils.common import ensure_alpha_channel, remove_alpha_channel
from flow_control.utils.logging import get_logger, warn_once
from flow_control.utils.merge_images import merge_images
from flow_control.utils.resize import (
    resize_to_closest_resolution,
    resize_to_multiple_of,
    resize_to_resolution,
)

from ..base import BaseProcessor, InputBatch, ProcessedBatch, TrainInputBatch
from ..components.llm import parse_llm_json_output
from ..components.prompts import PromptStr, parse_prompt

logger = get_logger(__name__)


class EfficientLayeredInputBatch(InputBatch):
    clean_image: torch.Tensor
    layer_boxes: NotRequired[list[tuple[int, int, int, int]] | None]
    layer_prompts: NotRequired[list[str] | None]


class EfficientLayeredTrainInputBatch(TrainInputBatch):
    clean_image: torch.Tensor
    layer_boxes: list[tuple[int, int, int, int]]
    layer_images: list[torch.Tensor]
    layer_prompts: NotRequired[list[str] | None]


class EfficientLayeredProcessedBatch(ProcessedBatch):
    prompt_embeds: torch.Tensor
    layer_boxes: list[tuple[int, int, int, int]]
    image_latents: torch.Tensor
    text_lengths: list[int]


class EfficientLayeredProcessor(BaseProcessor):
    encoder_prompt: PromptStr
    fg_caption_prompt: PromptStr = parse_prompt("@efficient_layered_caption_fg_en")
    bg_caption_prompt: PromptStr = parse_prompt("@efficient_layered_caption_bg_en")
    default_negative_prompt: str = " "
    save_negative: bool = False

    detection_prompt: PromptStr = parse_prompt("@efficient_layered_detection_en")
    detection_coord_type: Literal["qwen25vl", "qwen3vl"] = "qwen3vl"

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
                image, self.multiple_of, crop=False, pixels=self.total_pixels
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

    async def genearte_layer_boxes_prompts(
        self, image: torch.Tensor
    ) -> tuple[list[tuple[int, int, int, int]], list[str]]:
        h, w = image.shape[2], image.shape[3]
        result_text = await self.chat_completion(
            self.detection_prompt, [remove_alpha_channel(image)]
        )
        try:
            result_json = parse_llm_json_output(result_text)
            layer_boxes: list[tuple[int, int, int, int]] = [(0, h, 0, w)]
            layer_prompts: list[str] = [result_json["background"]]
            for item in result_json["foreground"]:
                xmin, ymin, xmax, ymax = item["bbox_2d"]
                if self.detection_coord_type == "qwen3vl":
                    # Qwen3-VL use normized coordinates [0, 1000]
                    xmin = int(xmin / 1000 * w)
                    xmax = int(xmax / 1000 * w)
                    ymin = int(ymin / 1000 * h)
                    ymax = int(ymax / 1000 * h)
                layer_boxes.append((ymin, ymax, xmin, xmax))
                layer_prompts.append(item["label"])
            return layer_boxes, layer_prompts
        except Exception:
            logger.error(
                "Failed to parse detection output, this is likely due to LLM returning invalid JSON."
            )
            logger.debug("Dumping raw LLM response:")
            logger.debug(result_text)
            raise

    def generate_negative(self, num_layers: int):
        negative_prompt_embeds = self.encoder.encode(
            self.default_negative_prompt,
            system_prompt=self.encoder_prompt,
        )
        text_lengths = [negative_prompt_embeds.shape[1]] * num_layers
        prompt_embeds = repeat(
            negative_prompt_embeds, "b l d -> b (n l) d", n=num_layers
        )
        return {
            "prompt_embeds": prompt_embeds,
            "text_lengths": text_lengths,
        }

    async def prepare_inference_batch(
        self, batch: EfficientLayeredInputBatch
    ) -> EfficientLayeredProcessedBatch:
        orig_size = batch["clean_image"].shape[2], batch["clean_image"].shape[3]
        batch["clean_image"] = clean_image = self.resize_image(
            ensure_alpha_channel(batch["clean_image"])
        )
        image_size = clean_image.shape[2], clean_image.shape[3]
        image_latents = self.encode_latents(clean_image)

        layer_boxes = batch.get("layer_boxes", None)
        layer_prompts = batch.get("layer_prompts", None)
        if (layer_boxes is None) ^ (layer_prompts is None):
            warn_once(
                logger,
                "Either both or neither of layer_boxes and layer_prompts should be provided. Ignoring both.",
            )
        if layer_boxes is None or layer_prompts is None:
            batch["layer_boxes"], batch["layer_prompts"] = (
                layer_boxes,
                layer_prompts,
            ) = await self.genearte_layer_boxes_prompts(clean_image)
            # No need to scale layer boxes as image is already resized
            batch["layer_boxes"] = layer_boxes = self._scale_and_align_layer_boxes(
                layer_boxes, image_size, image_size
            )
        else:
            batch["layer_boxes"] = layer_boxes = self._scale_and_align_layer_boxes(
                layer_boxes, orig_size, image_size
            )
        prompt_embeds_list = [
            self.encoder.encode(prompt, system_prompt=self.encoder_prompt)
            for prompt in layer_prompts
        ]
        prompt_embeds = torch.cat(prompt_embeds_list, dim=1)
        text_lengths = [embed.shape[1] for embed in prompt_embeds_list]

        result = EfficientLayeredProcessedBatch(
            image_size=image_size,
            image_latents=image_latents,
            prompt_embeds=prompt_embeds,
            layer_boxes=layer_boxes,
            text_lengths=text_lengths,
        )

        if self.save_negative:
            result["negative"] = self.generate_negative(len(layer_boxes))

        return result

    async def prepare_training_batch(
        self, batch: EfficientLayeredTrainInputBatch
    ) -> EfficientLayeredProcessedBatch:
        if (layer_prompts := batch.get("layer_prompts", None)) is None:
            batch["layer_prompts"] = layer_prompts = await asyncio.gather(
                *(
                    [
                        self.chat_completion(
                            self.bg_caption_prompt,
                            [remove_alpha_channel(batch["layer_images"][0])],
                        )
                    ]
                    + [
                        self.chat_completion(
                            self.fg_caption_prompt, [remove_alpha_channel(img)]
                        )
                        for img in batch["layer_images"][1:]
                    ]
                )
            )
        stacked_images = self._stack_all_images(
            batch["clean_image"], batch["layer_boxes"], batch["layer_images"]
        )
        orig_size = batch["clean_image"].shape[2], batch["clean_image"].shape
        resized_images = self.resize_image(stacked_images)
        new_size = resized_images.shape[2], resized_images.shape[3]
        batch["clean_image"] = clean_image = resized_images[0:1]
        image_latents = self.encode_latents(clean_image)
        batch["layer_boxes"] = layer_boxes = self._scale_and_align_layer_boxes(
            batch["layer_boxes"], orig_size, new_size
        )
        batch["layer_images"] = layer_images = self._crop_stacked_images(
            resized_images, layer_boxes
        )
        clean_latents = torch.cat(
            [self.encode_latents(img) for img in layer_images], dim=1
        )
        prompt_embeds_list = [
            self.encoder.encode(prompt, system_prompt=self.encoder_prompt)
            for prompt in layer_prompts
        ]
        prompt_embeds = torch.cat(prompt_embeds_list, dim=1)
        text_lengths = [embed.shape[1] for embed in prompt_embeds_list]

        result = EfficientLayeredProcessedBatch(
            image_size=new_size,
            image_latents=image_latents,
            prompt_embeds=prompt_embeds,
            layer_boxes=layer_boxes,
            text_lengths=text_lengths,
            clean_latents=clean_latents,
        )

        if self.save_negative:
            result["negative"] = self.generate_negative(len(layer_boxes))

        return result

    def get_latent_length(self, batch: EfficientLayeredProcessedBatch):
        ratio = (self.vae_scale_factor * self.patch_size) ** 2
        return (
            super().get_latent_length(batch)
            + batch["prompt_embeds"].shape[1]
            + sum(
                (bottom - top) * (right - left) // ratio
                for (top, bottom, left, right) in batch["layer_boxes"]
            )
        )

    def decode_output(
        self, output_latent: torch.Tensor, batch: EfficientLayeredProcessedBatch
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
        batch["layer_images"] = decoded_layers  # type: ignore
        merged_image = merge_images(decoded_layers)
        return merged_image

    def initialize_latents(
        self,
        batch: EfficientLayeredProcessedBatch,
        generator=None,
        device=None,
        dtype=torch.bfloat16,
    ) -> torch.Tensor:
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
            dtype=dtype,
        )
        batch["noisy_latents"] = latents
        return latents
