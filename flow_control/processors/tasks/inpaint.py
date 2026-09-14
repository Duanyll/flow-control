from typing import ClassVar, Literal, NotRequired

import torch
import torch.nn.functional as F

from flow_control.data.coercion import ImageTensor
from flow_control.utils.resize import resize_to_resolution

from ..base import (
    BaseProcessor,
    InputBatch,
    ProcessedBatch,
    TrainInputBatch,
    task_registry,
)
from ..components.prompts import PromptStr, parse_prompt


class InpaintInputBatch(InputBatch):
    prompt: str
    negative_prompt: NotRequired[str | None]
    inpaint_image: ImageTensor
    """Source image; masked regions are regenerated, the rest is kept."""
    inpaint_mask: ImageTensor


class InpaintTrainInputBatch(TrainInputBatch):
    prompt: NotRequired[str | None]
    inpaint_mask: ImageTensor
    clean_image: ImageTensor


class InpaintProcessedBatch(ProcessedBatch):
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor | None
    inpaint_latents: torch.Tensor
    inpaint_mask: torch.Tensor
    """`[B, H, W]` Luminance mask, where white is the editable region."""
    inpaint_mask_latents: torch.Tensor
    """`[B, N, P]` Mask packed like latents, with `P = patch_size ** 2`."""


@task_registry.register("inpaint")
class InpaintProcessor(
    BaseProcessor[InpaintInputBatch, InpaintTrainInputBatch, InpaintProcessedBatch]
):
    task: Literal["inpaint"] = "inpaint"
    encoder_prompt: PromptStr = ""
    caption_prompt: PromptStr = parse_prompt("@default_t2i_caption")
    default_negative_prompt: str = " "
    save_negative: bool = False
    posterior_fields: ClassVar[tuple[str, ...]] = ("clean_latents", "inpaint_latents")

    def _prepare_inpaint_mask(
        self,
        mask: torch.Tensor,
        image_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize an image mask and preserve its intra-token spatial detail."""
        mask = resize_to_resolution(mask, image_size)
        channels = mask.shape[1]
        if channels in (1, 2):
            luminance = mask[:, 0]
        elif channels in (3, 4):
            luminance = mask[:, :3].mean(dim=1)
        else:
            raise ValueError(
                "Inpaint masks must have 1 (L), 2 (LA), 3 (RGB), or 4 (RGBA) "
                f"channels, got shape {tuple(mask.shape)}."
            )

        latent_size = (
            image_size[0] // self.vae_scale_factor,
            image_size[1] // self.vae_scale_factor,
        )
        latent_mask = F.interpolate(
            luminance.unsqueeze(1),
            size=latent_size,
            mode="bilinear",
            align_corners=False,
        )
        return luminance, self._pack_latents(latent_mask)

    async def prepare_inference_batch(
        self, batch: InpaintInputBatch
    ) -> InpaintProcessedBatch:
        inpaint_image = batch["inpaint_image"] = self.resize_image(
            batch["inpaint_image"]
        )
        image_size = (inpaint_image.shape[2], inpaint_image.shape[3])
        inpaint_mask, inpaint_mask_latents = self._prepare_inpaint_mask(
            batch["inpaint_mask"], image_size
        )
        batch["inpaint_mask"] = inpaint_mask
        inpaint_latents = self.encode_latents(
            inpaint_image, posterior=self.condition_posterior
        )
        result = InpaintProcessedBatch(
            image_size=image_size,
            inpaint_latents=inpaint_latents,
            inpaint_mask=inpaint_mask,
            inpaint_mask_latents=inpaint_mask_latents,
            **self.encode_prompt(batch["prompt"], system_prompt=self.encoder_prompt),
        )

        if self.save_negative:
            result["negative"] = self.encode_prompt(
                batch.get("negative_prompt", None) or self.default_negative_prompt,
                system_prompt=self.encoder_prompt,
            )

        return result

    async def prepare_training_batch(
        self, batch: InpaintTrainInputBatch
    ) -> InpaintProcessedBatch:
        batch["clean_image"] = clean_image = self.resize_image(batch["clean_image"])
        image_size = clean_image.shape[2], clean_image.shape[3]
        if (prompt := batch.get("prompt", None)) is None:
            batch["prompt"] = prompt = await self.chat_completion(
                self.caption_prompt, images=[clean_image]
            )
        clean_latents = self.encode_latents(
            clean_image, posterior=self.target_posterior
        )
        # Training is self-supervised: the clean target is also the inpaint source.
        inpaint_latents = self.encode_latents(
            clean_image, posterior=self.condition_posterior
        )
        inpaint_mask, inpaint_mask_latents = self._prepare_inpaint_mask(
            batch["inpaint_mask"], image_size
        )
        batch["inpaint_mask"] = inpaint_mask

        result = InpaintProcessedBatch(
            image_size=image_size,
            clean_latents=clean_latents,
            inpaint_latents=inpaint_latents,
            inpaint_mask=inpaint_mask,
            inpaint_mask_latents=inpaint_mask_latents,
            **self.encode_prompt(prompt, system_prompt=self.encoder_prompt),
        )

        if self.save_negative:
            result["negative"] = self.encode_prompt(
                self.default_negative_prompt,
                system_prompt=self.encoder_prompt,
            )

        return result

    def get_cost(self, batch: InpaintProcessedBatch) -> int:
        return super().get_cost(batch) + batch["prompt_embeds"].shape[1]
