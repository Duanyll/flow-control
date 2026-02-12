from typing import Literal, NotRequired, TypedDict

import torch

from flow_control.utils.resize import (
    resize_to_closest_resolution,
    resize_to_multiple_of,
    resize_to_resolution,
)

from ..base import BaseProcessor, InputBatch, ProcessedBatch, TrainInputBatch
from ..components.prompts import PromptStr


class TIEInputBatch(InputBatch):
    image_size: NotRequired[tuple[int, int] | None]
    prompt: str
    negative_prompt: NotRequired[str | None]
    reference_images: list[torch.Tensor]


class TIETrainInputBatch(TrainInputBatch):
    prompt: str
    reference_images: list[torch.Tensor]
    clean_image: torch.Tensor


class TIEProcessedBatch(ProcessedBatch):
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor | None
    reference_latents: list[torch.Tensor]
    reference_sizes: list[tuple[int, int]]


class TIEProcessor(BaseProcessor[TIEInputBatch, TIETrainInputBatch, TIEProcessedBatch]):
    encoder_prompt: PromptStr
    default_negative_prompt: str = " "
    save_negative: bool = False

    reference_image_resize_mode: Literal["multiple_of", "list", "match_latent"] = (
        "match_latent"
    )

    class _EncodeRefenenceResult(TypedDict):
        reference_latents: list[torch.Tensor]
        reference_sizes: list[tuple[int, int]]

    def encode_reference_images(
        self, reference_images: list[torch.Tensor], image_size: tuple[int, int]
    ) -> _EncodeRefenenceResult:
        reference_latents = []
        reference_sizes = []
        for img in reference_images:
            if self.reference_image_resize_mode == "multiple_of":
                img = resize_to_multiple_of(img, multiple=self.multiple_of)
            elif self.reference_image_resize_mode == "list":
                img = resize_to_closest_resolution(img, self.preferred_resolutions)
            elif self.reference_image_resize_mode == "match_latent":
                img = resize_to_resolution(img, image_size)
            reference_sizes.append((img.shape[2], img.shape[3]))
            latents = self.encode_latents(img)
            reference_latents.append(latents)
        return {
            "reference_latents": reference_latents,
            "reference_sizes": reference_sizes,
        }

    async def prepare_inference_batch(self, batch: TIEInputBatch) -> TIEProcessedBatch:
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

        result = TIEProcessedBatch(
            image_size=image_size,
            **self.encode_prompt(
                batch["prompt"],
                images=batch["reference_images"],
                system_prompt=self.encoder_prompt,
            ),
            **self.encode_reference_images(batch["reference_images"], image_size),
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
        batch["clean_image"] = clean_image = self.resize_image(batch["clean_image"])
        image_size = clean_image.shape[2], clean_image.shape[3]
        clean_latents = self.encode_latents(clean_image)
        result = TIEProcessedBatch(
            image_size=image_size,
            clean_latents=clean_latents,
            **self.encode_prompt(
                batch["prompt"],
                images=batch["reference_images"],
                system_prompt=self.encoder_prompt,
            ),
            **self.encode_reference_images(batch["reference_images"], image_size),
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
