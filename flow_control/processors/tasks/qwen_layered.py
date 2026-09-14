from typing import ClassVar, Literal, NotRequired, cast

import torch
from einops import rearrange

from flow_control.data.coercion import ImageTensor, ImageTensorList
from flow_control.utils.merge_images import merge_images
from flow_control.utils.resize import resize_to_resolution

from ..base import (
    BaseProcessor,
    DecodedRow,
    InputRow,
    ProcessedRow,
    TrainInputRow,
    task_registry,
)
from ..components.prompts import PromptStr, parse_prompt
from ..components.vae import PosteriorMode


class QwenLayeredInputRow(InputRow):
    clean_image: ImageTensor
    prompt: NotRequired[str | None]
    negative_prompt: NotRequired[str | None]
    num_layers: NotRequired[int]


class QwenLayeredTrainInputRow(TrainInputRow):
    clean_image: ImageTensor
    prompt: NotRequired[str | None]
    layer_images: ImageTensorList


class QwenLayeredProcessedRow(ProcessedRow):
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor | None
    image_latents: torch.Tensor
    num_layers: int


class QwenLayeredDecodedRow(DecodedRow):
    base_image: torch.Tensor
    layer_images: list[torch.Tensor]


@task_registry.register("qwen_layered")
class QwenImageLayeredProcessor(
    BaseProcessor[
        QwenLayeredInputRow, QwenLayeredTrainInputRow, QwenLayeredProcessedRow
    ]
):
    task: Literal["qwen_layered"] = "qwen_layered"
    default_num_layers: int = 4
    encoder_prompt: PromptStr = ""
    caption_prompt: PromptStr = parse_prompt("@qwen_layered_caption_en")
    default_negative_prompt: str = " "
    save_negative: bool = False
    posterior_fields: ClassVar[tuple[str, ...]] = ("clean_latents", "image_latents")

    @torch.no_grad()
    def _encode_latents_layered(
        self, images: list[torch.Tensor], posterior: PosteriorMode = "sample"
    ) -> torch.Tensor:
        """Encode multiple images one-by-one and cat along the sequence dim."""
        return torch.cat(
            [self.encode_latents(img, posterior=posterior) for img in images],
            dim=1,
        )

    def _pack_latents_layered(self, latents: torch.Tensor) -> torch.Tensor:
        return rearrange(
            latents,
            "f c (h ph) (w pw) -> 1 (f h w) (c ph pw)",
            ph=self.patch_size,
            pw=self.patch_size,
        )

    @torch.no_grad()
    def decode_latents_layered(
        self, latents: torch.Tensor, size: tuple[int, int]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        latents = self._unpack_latents_layered(latents, size)
        images = self.vae.decode(latents)
        base_image = images[0:1]
        layer_images = [images[i + 1 : i + 2] for i in range(images.shape[0] - 1)]
        return base_image, layer_images

    def _unpack_latents_layered(
        self, latents: torch.Tensor, size: tuple[int, int]
    ) -> torch.Tensor:
        h, w = size
        h = h // self.vae_scale_factor
        w = w // self.vae_scale_factor
        return rearrange(
            latents,
            "1 (f h w) (c ph pw) -> f c (h ph) (w pw)",
            h=h // self.patch_size,
            w=w // self.patch_size,
            ph=self.patch_size,
            pw=self.patch_size,
        )

    async def prepare_inference_row(
        self, row: QwenLayeredInputRow
    ) -> QwenLayeredProcessedRow:
        row["clean_image"] = clean_image = self.resize_image(row["clean_image"])
        image_size = clean_image.shape[2], clean_image.shape[3]
        if (prompt := row.get("prompt", None)) is None:
            row["prompt"] = prompt = await self.chat_completion(
                self.caption_prompt, images=[clean_image]
            )
        image_latents = self.encode_latents(
            clean_image, posterior=self.condition_posterior
        )
        num_layers = row.get("num_layers")
        if num_layers is None:
            num_layers = self.default_num_layers

        result = QwenLayeredProcessedRow(
            image_size=image_size,
            image_latents=image_latents,
            **self.encode_prompt(prompt, system_prompt=self.encoder_prompt),
            num_layers=num_layers,
        )

        if self.save_negative:
            result["negative"] = self.encode_prompt(
                row.get("negative_prompt", None) or self.default_negative_prompt,
                system_prompt=self.encoder_prompt,
            )

        return result

    async def prepare_training_row(
        self, row: QwenLayeredTrainInputRow
    ) -> QwenLayeredProcessedRow:
        row["clean_image"] = clean_image = self.resize_image(row["clean_image"])
        image_size = clean_image.shape[2], clean_image.shape[3]
        if (prompt := row.get("prompt", None)) is None:
            row["prompt"] = prompt = await self.chat_completion(
                self.caption_prompt, images=[clean_image]
            )
        image_latents = self.encode_latents(
            clean_image, posterior=self.condition_posterior
        )
        num_layers = len(row["layer_images"])
        for i in range(num_layers):
            row["layer_images"][i] = resize_to_resolution(
                row["layer_images"][i], image_size
            )
        clean_latents = self._encode_latents_layered(
            [row["clean_image"], *row["layer_images"]],
            posterior=self.target_posterior,
        )
        result = QwenLayeredProcessedRow(
            image_size=image_size,
            image_latents=image_latents,
            clean_latents=clean_latents,
            **self.encode_prompt(prompt, system_prompt=self.encoder_prompt),
            num_layers=num_layers,
        )
        if self.save_negative:
            result["negative"] = self.encode_prompt(
                self.default_negative_prompt,
                system_prompt=self.encoder_prompt,
            )
        return result

    def get_cost(self, row: QwenLayeredProcessedRow) -> int:
        # Source image + (num_layers + 1) generated images share one sequence.
        return (row["num_layers"] + 2) * super().get_cost(row) + row[
            "prompt_embeds"
        ].shape[1]

    def decode_output(
        self,
        output_latent: torch.Tensor,
        row: ProcessedRow,
    ) -> QwenLayeredDecodedRow:
        base_image, layer_images = self.decode_latents_layered(
            output_latent, row["image_size"]
        )
        return QwenLayeredDecodedRow(
            clean_image=merge_images([base_image, *layer_images]),
            base_image=base_image,
            layer_images=layer_images,
        )

    def annotate_output(self, decoded: DecodedRow, row: ProcessedRow) -> torch.Tensor:
        layered = cast(QwenLayeredDecodedRow, decoded)
        return merge_images(
            [layered["base_image"], *layered["layer_images"]],
            border_width=4,
            draw_labels=True,
        )

    def initialize_latents(
        self,
        row: QwenLayeredProcessedRow,
        generator=None,
        device=None,
        dtype=torch.bfloat16,
    ):
        if device is None:
            device = self.device
        h, w = row["image_size"]
        c = self.latent_channels
        h = h // self.vae_scale_factor
        w = w // self.vae_scale_factor
        f = row["num_layers"] + 1
        latents = torch.randn(
            (f, c, h, w), generator=generator, device=device, dtype=dtype
        )
        row["noisy_latents"] = self._pack_latents_layered(latents)
        return row["noisy_latents"]
