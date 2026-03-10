from typing import Literal, NotRequired, TypedDict

import torch

from flow_control.utils.coercion import ImageTensor, ImageTensorList
from flow_control.utils.logging import get_logger, warn_once
from flow_control.utils.resize import (
    resize_to_closest_resolution,
    resize_to_multiple_of,
    resize_to_resolution,
)

from ..base import BaseProcessor, InputBatch, ProcessedBatch, TrainInputBatch
from ..components.prompts import PromptStr, parse_prompt

logger = get_logger(__name__)


class TIEInputBatch(InputBatch):
    prompt: str
    negative_prompt: NotRequired[str | None]
    reference_images: ImageTensorList


class TIETrainInputBatch(TrainInputBatch):
    prompt: str
    reference_images: ImageTensorList
    clean_image: ImageTensor


class TIEProcessedBatch(ProcessedBatch):
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor | None
    reference_latents: list[torch.Tensor]
    reference_sizes: list[tuple[int, int]]


class TIEProcessor(BaseProcessor[TIEInputBatch, TIETrainInputBatch, TIEProcessedBatch]):
    task: Literal["tie"] = "tie"
    encoder_prompt: PromptStr = ""
    tie_enhance_prompt: PromptStr = parse_prompt("@default_tie_enhance")
    default_negative_prompt: str = " "
    save_negative: bool = False
    enable_enhance: bool = False
    max_reference_images: int = 0

    reference_image_resize_mode: Literal["multiple_of", "list", "match_latent"] = (
        "match_latent"
    )

    class _EncodeRefenenceResult(TypedDict):
        reference_latents: list[torch.Tensor]
        reference_sizes: list[tuple[int, int]]

    def trim_reference_images(
        self, reference_images: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        if (
            self.max_reference_images > 0
            and len(reference_images) > self.max_reference_images
        ):
            warn_once(
                logger,
                f"Provided {len(reference_images)} reference images, but max_reference_images "
                f"is set to {self.max_reference_images}. Discarding extra reference images.",
            )
            return reference_images[: self.max_reference_images]
        return reference_images

    def resize_reference_images(
        self, reference_images: list[torch.Tensor], image_size: tuple[int, int]
    ) -> list[torch.Tensor]:
        resized_images = []
        for img in reference_images:
            if self.reference_image_resize_mode == "multiple_of":
                img = resize_to_multiple_of(
                    img, multiple=self.multiple_of, no_upscale=self.no_upscale
                )
            elif self.reference_image_resize_mode == "list":
                img = resize_to_closest_resolution(img, self.preferred_resolutions)
            elif self.reference_image_resize_mode == "match_latent":
                img = resize_to_resolution(img, image_size)
            resized_images.append(img)
        return resized_images

    def encode_reference_images(
        self, reference_images: list[torch.Tensor]
    ) -> _EncodeRefenenceResult:
        return {
            "reference_latents": [self.encode_latents(img) for img in reference_images],
            "reference_sizes": [
                (img.shape[2], img.shape[3]) for img in reference_images
            ],
        }

    async def enhance_prompt(
        self, prompt: str, reference_images: list[torch.Tensor]
    ) -> str:
        if not self.enable_enhance:
            return prompt
        return await self.chat_completion(
            prompt=prompt,
            system_prompt=self.tie_enhance_prompt,
            images=reference_images,
        )

    async def prepare_inference_batch(self, batch: TIEInputBatch) -> TIEProcessedBatch:
        batch["reference_images"] = self.trim_reference_images(
            batch["reference_images"]
        )
        if (image_size := batch.get("image_size", None)) is None:
            if len(batch["reference_images"]) > 0:
                batch["reference_images"][0] = self.resize_image(
                    batch["reference_images"][0]
                )
                batch["image_size"] = image_size = (
                    batch["reference_images"][0].shape[2],
                    batch["reference_images"][0].shape[3],
                )
            else:
                batch["image_size"] = image_size = self.default_resolution
        batch["reference_images"] = self.resize_reference_images(
            batch["reference_images"], image_size
        )

        batch["prompt"] = await self.enhance_prompt(
            batch["prompt"], batch["reference_images"]
        )

        result = TIEProcessedBatch(
            image_size=image_size,
            **self.encode_prompt(
                batch["prompt"],
                images=batch["reference_images"],
                system_prompt=self.encoder_prompt,
            ),
            **self.encode_reference_images(batch["reference_images"]),
        )

        if self.save_negative:
            result["negative"] = self.encode_prompt(
                batch.get("negative_prompt", None) or self.default_negative_prompt,
                system_prompt=self.encoder_prompt,
            )

        return result

    async def prepare_training_batch(
        self, batch: TIETrainInputBatch
    ) -> TIEProcessedBatch:
        batch["reference_images"] = self.trim_reference_images(
            batch["reference_images"]
        )

        batch["clean_image"] = clean_image = self.resize_image(batch["clean_image"])
        image_size = clean_image.shape[2], clean_image.shape[3]
        clean_latents = self.encode_latents(clean_image)

        batch["reference_images"] = self.resize_reference_images(
            batch["reference_images"], image_size
        )

        batch["prompt"] = await self.enhance_prompt(
            batch["prompt"], batch["reference_images"]
        )

        result = TIEProcessedBatch(
            image_size=image_size,
            clean_latents=clean_latents,
            **self.encode_prompt(
                batch["prompt"],
                images=batch["reference_images"],
                system_prompt=self.encoder_prompt,
            ),
            **self.encode_reference_images(batch["reference_images"]),
        )

        if self.save_negative:
            result["negative"] = self.encode_prompt(
                self.default_negative_prompt,
                system_prompt=self.encoder_prompt,
            )
        return result

    def get_latent_length(self, batch: TIEProcessedBatch):
        ratio = (self.vae_scale_factor * self.patch_size) ** 2
        return (
            super().get_latent_length(batch)
            + batch["prompt_embeds"].shape[1]
            + sum((h * w) // ratio for h, w in batch["reference_sizes"])
        )
