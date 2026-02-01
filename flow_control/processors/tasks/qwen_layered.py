from typing import NotRequired

import torch
from einops import rearrange

from flow_control.utils.common import ensure_alpha_channel
from flow_control.utils.merge_images import merge_images
from flow_control.utils.resize import resize_to_resolution

from ..base import BaseProcessor, InputBatch, ProcessedBatch, TrainInputBatch
from ..components.prompts import PromptStr, parse_prompt


class QwenLayeredInputBatch(InputBatch):
    clean_image: torch.Tensor
    prompt: NotRequired[str | None]
    negative_prompt: NotRequired[str | None]
    num_layers: NotRequired[int]


class QwenLayeredTrainInputBatch(TrainInputBatch):
    clean_image: torch.Tensor
    prompt: NotRequired[str | None]
    layer_images: list[torch.Tensor]


class QwenLayeredProcessedBatch(ProcessedBatch):
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor | None
    image_latents: torch.Tensor
    num_layers: int


class QwenImageLayeredProcessor(BaseProcessor):
    default_num_layers: int = 4
    encoder_prompt: PromptStr
    caption_prompt: PromptStr = parse_prompt("@qwen_layered_caption_en")
    default_negative_prompt: str = " "
    save_negative: bool = False

    @torch.no_grad()
    def encode_latents(self, images: torch.Tensor | list[torch.Tensor]):
        if not isinstance(images, list):
            images = [images]

        all_images = torch.cat([ensure_alpha_channel(image) for image in images], dim=0)
        latents = self.vae.encode(all_images)
        latents = self._pack_latents_layered(latents)
        return latents

    def _pack_latents_layered(self, latents: torch.Tensor) -> torch.Tensor:
        return rearrange(
            latents,
            "f c (h ph) (w pw) -> 1 (f h w) (c ph pw)",
            ph=self.patch_size,
            pw=self.patch_size,
        )

    @torch.no_grad()
    def decode_latents(
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

    async def prepare_inference_batch(
        self, batch: QwenLayeredInputBatch
    ) -> QwenLayeredProcessedBatch:
        batch["clean_image"] = clean_image = self.resize_image(batch["clean_image"])
        image_size = clean_image.shape[2], clean_image.shape[3]
        if (prompt := batch.get("prompt", None)) is None:
            batch["prompt"] = prompt = await self.chat_completion(
                self.caption_prompt, images=[clean_image]
            )
        image_latents = self.encode_latents(clean_image)
        num_layers = batch.get("num_layers", None) or self.default_num_layers

        result = QwenLayeredProcessedBatch(
            image_size=image_size,
            image_latents=image_latents,
            **self.encode_prompt(prompt, system_prompt=self.encoder_prompt),
            num_layers=num_layers,
        )

        if self.save_negative:
            result["negative"] = self.encode_prompt(
                batch.get("negative_prompt", None) or self.default_negative_prompt,
                system_prompt=self.encoder_prompt,
            )

        return result

    async def prepare_training_batch(
        self, batch: QwenLayeredTrainInputBatch
    ) -> QwenLayeredProcessedBatch:
        batch["clean_image"] = clean_image = self.resize_image(batch["clean_image"])
        image_size = clean_image.shape[2], clean_image.shape[3]
        if (prompt := batch.get("prompt", None)) is None:
            batch["prompt"] = prompt = await self.chat_completion(
                self.caption_prompt, images=[clean_image]
            )
        image_latents = self.encode_latents(clean_image)
        num_layers = len(batch["layer_images"])
        for i in range(num_layers):
            batch["layer_images"][i] = resize_to_resolution(
                batch["layer_images"][i], image_size
            )
        clean_latents = self.encode_latents(
            [batch["clean_image"], *batch["layer_images"]]
        )
        result = QwenLayeredProcessedBatch(
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

    def get_latent_length(self, batch: QwenLayeredProcessedBatch):
        return (batch["num_layers"] + 1) * super().get_latent_length(batch)

    def decode_output(
        self, output_latent: torch.Tensor, batch: ProcessedBatch
    ) -> torch.Tensor:
        base_image, layer_images = self.decode_latents(
            output_latent, batch["image_size"]
        )
        batch["clean_image"] = base_image  # type: ignore
        batch["layer_images"] = layer_images  # type: ignore
        return merge_images([base_image, *layer_images])

    def initialize_latents(
        self, batch: ProcessedBatch, generator=None, device=None, dtype=torch.bfloat16
    ):
        if device is None:
            device = self.device
        h, w = batch["image_size"]
        c = self.latent_channels
        h = h // self.vae_scale_factor
        w = w // self.vae_scale_factor
        f = batch.get("num_layers", 0) + 1
        latents = torch.randn(
            (f, c, h, w), generator=generator, device=device, dtype=dtype
        )
        batch["noisy_latents"] = self._pack_latents_layered(latents)
        return batch["noisy_latents"]
