from typing import ClassVar, Literal, NotRequired, TypedDict

import torch

from flow_control.data.coercion import ImageTensor, ImageTensorList
from flow_control.utils.logging import get_logger, warn_once
from flow_control.utils.merge_images import merge_images
from flow_control.utils.resize import (
    resize_to_closest_resolution,
    resize_to_multiple_of,
    resize_to_resolution,
)

from ..base import (
    BaseProcessor,
    DecodedRow,
    InputRow,
    ProcessedRow,
    TrainInputRow,
    task_registry,
)
from ..components.prompts import PromptStr, parse_prompt

logger = get_logger(__name__)


class TIEInputRow(InputRow):
    prompt: str
    negative_prompt: NotRequired[str | None]
    reference_images: ImageTensorList


class TIETrainInputRow(TrainInputRow):
    prompt: str
    reference_images: ImageTensorList
    clean_image: ImageTensor


class TIEProcessedRow(ProcessedRow):
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor | None
    reference_latents: list[torch.Tensor]
    reference_sizes: list[tuple[int, int]]


@task_registry.register("tie")
class TIEProcessor(BaseProcessor[TIEInputRow, TIETrainInputRow, TIEProcessedRow]):
    task: Literal["tie"] = "tie"
    encoder_prompt: PromptStr = ""
    tie_enhance_prompt: PromptStr = parse_prompt("@default_tie_enhance")
    default_negative_prompt: str = " "
    save_negative: bool = False
    negative_with_images: bool = False
    """Encode the negative prompt together with the reference images. Models
    whose unconditional pass still conditions on the references (e.g.
    HiDream-O1) need this so the negative overlay stays consistent with the
    positive row's image tensors."""
    enable_enhance: bool = False
    max_reference_images: int = 0
    posterior_fields: ClassVar[tuple[str, ...]] = ("clean_latents", "reference_latents")

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
            "reference_latents": [
                self.encode_latents(img, posterior=self.condition_posterior)
                for img in reference_images
            ],
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

    async def prepare_inference_row(self, row: TIEInputRow) -> TIEProcessedRow:
        row["reference_images"] = self.trim_reference_images(row["reference_images"])
        if (image_size := row.get("image_size", None)) is None:
            if len(row["reference_images"]) > 0:
                row["reference_images"][0] = self.resize_image(
                    row["reference_images"][0]
                )
                row["image_size"] = image_size = (
                    row["reference_images"][0].shape[2],
                    row["reference_images"][0].shape[3],
                )
            else:
                row["image_size"] = image_size = self.default_resolution
        row["reference_images"] = self.resize_reference_images(
            row["reference_images"], image_size
        )

        row["prompt"] = await self.enhance_prompt(
            row["prompt"], row["reference_images"]
        )

        result = TIEProcessedRow(
            image_size=image_size,
            **self.encode_prompt(
                row["prompt"],
                images=row["reference_images"],
                system_prompt=self.encoder_prompt,
            ),
            **self.encode_reference_images(row["reference_images"]),
        )

        if self.save_negative:
            result["negative"] = self.encode_prompt(
                row.get("negative_prompt", None) or self.default_negative_prompt,
                images=row["reference_images"] if self.negative_with_images else None,
                system_prompt=self.encoder_prompt,
            )

        return result

    async def prepare_training_row(self, row: TIETrainInputRow) -> TIEProcessedRow:
        row["reference_images"] = self.trim_reference_images(row["reference_images"])

        row["clean_image"] = clean_image = self.resize_image(row["clean_image"])
        image_size = clean_image.shape[2], clean_image.shape[3]
        clean_latents = self.encode_latents(
            clean_image, posterior=self.target_posterior
        )

        row["reference_images"] = self.resize_reference_images(
            row["reference_images"], image_size
        )

        row["prompt"] = await self.enhance_prompt(
            row["prompt"], row["reference_images"]
        )

        result = TIEProcessedRow(
            image_size=image_size,
            clean_latents=clean_latents,
            **self.encode_prompt(
                row["prompt"],
                images=row["reference_images"],
                system_prompt=self.encoder_prompt,
            ),
            **self.encode_reference_images(row["reference_images"]),
        )

        if self.save_negative:
            result["negative"] = self.encode_prompt(
                self.default_negative_prompt,
                images=row["reference_images"] if self.negative_with_images else None,
                system_prompt=self.encoder_prompt,
            )
        return result

    def get_cost(self, row: TIEProcessedRow) -> int:
        ratio = (self.vae_scale_factor * self.patch_size) ** 2
        return (
            super().get_cost(row)
            + row["prompt_embeds"].shape[1]
            + sum((h * w) // ratio for h, w in row["reference_sizes"])
        )

    def annotate_output(
        self, decoded: DecodedRow, row: TIEProcessedRow
    ) -> torch.Tensor:
        references = row.get("reference_images")
        if not references:
            return decoded["clean_image"]
        return merge_images(
            [*references, decoded["clean_image"]],
            border_width=4,
            draw_labels=True,
        )
