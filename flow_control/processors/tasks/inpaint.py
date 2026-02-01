from typing import NotRequired

import torch

from flow_control.utils.resize import resize_to_resolution

from ..base import BaseProcessor, InputBatch, ProcessedBatch, TrainInputBatch
from ..components.prompts import PromptStr, parse_prompt


class InpaintInputBatch(InputBatch):
    prompt: str
    negative_prompt: NotRequired[str | None]
    inpaint_mask: torch.Tensor
    clean_image: torch.Tensor


class InpaintTrainInputBatch(TrainInputBatch):
    prompt: NotRequired[str | None]
    inpaint_mask: torch.Tensor
    clean_image: torch.Tensor


class InpaintProcessedBatch(ProcessedBatch):
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor | None
    inpaint_latents: torch.Tensor
    inpaint_mask: torch.Tensor


class InpaintProcessor(BaseProcessor):
    encoder_prompt: PromptStr
    caption_prompt: PromptStr = parse_prompt("@default_t2i_caption")
    default_negative_prompt: str = " "
    save_negative: bool = False

    async def prepare_inference_batch(
        self, batch: InpaintInputBatch
    ) -> InpaintProcessedBatch:
        inpaint_image = batch["clean_image"] = self.resize_image(batch["clean_image"])
        image_size = (inpaint_image.shape[2], inpaint_image.shape[3])
        inpaint_mask = batch["inpaint_mask"] = resize_to_resolution(
            batch["inpaint_mask"], image_size
        )
        inpaint_latents = self.encode_latents(inpaint_image)
        result = InpaintProcessedBatch(
            image_size=image_size,
            inpaint_latents=inpaint_latents,
            inpaint_mask=inpaint_mask,
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
        clean_latents = self.encode_latents(clean_image)
        inpaint_mask = batch["inpaint_mask"] = resize_to_resolution(
            batch["inpaint_mask"], image_size
        )

        result = InpaintProcessedBatch(
            image_size=image_size,
            clean_latents=clean_latents,
            inpaint_latents=clean_latents,
            inpaint_mask=inpaint_mask,
            **self.encode_prompt(prompt, system_prompt=self.encoder_prompt),
        )

        if self.save_negative:
            result["negative"] = self.encode_prompt(
                self.default_negative_prompt,
                system_prompt=self.encoder_prompt,
            )

        return result

    def get_latent_length(self, batch: InpaintProcessedBatch):
        return super().get_latent_length(batch) + batch["prompt_embeds"].shape[1]
