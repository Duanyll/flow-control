from typing import NotRequired

import torch

from flow_control.utils.resize import resize_to_resolution

from ..base import BaseProcessor, InputBatch, ProcessedBatch, TrainInputBatch
from ..components.prompts import PromptStr, parse_prompt


class T2IControlInputBatch(InputBatch):
    prompt: str
    negative_prompt: NotRequired[str | None]
    control_image: torch.Tensor


class T2IControlTrainInputBatch(TrainInputBatch):
    prompt: NotRequired[str | None]
    control_image: NotRequired[torch.Tensor]
    clean_image: torch.Tensor


class T2IControlProcessedBatch(ProcessedBatch):
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor | None
    control_latents: torch.Tensor


class T2IControlProcessor(BaseProcessor):
    encoder_prompt: PromptStr
    caption_prompt: PromptStr = parse_prompt("@default_t2i_caption")
    default_negative_prompt: str = " "
    save_negative: bool = False

    async def prepare_inference_batch(
        self, batch: T2IControlInputBatch
    ) -> T2IControlProcessedBatch:
        control_image = batch["control_image"] = self.resize_image(
            batch["control_image"]
        )
        image_size = (control_image.shape[2], control_image.shape[3])
        control_latents = self.encode_latents(control_image)
        result = T2IControlProcessedBatch(
            image_size=image_size,
            control_latents=control_latents,
            **self.encode_prompt(batch["prompt"], system_prompt=self.encoder_prompt),
        )

        if self.save_negative:
            result["negative"] = self.encode_prompt(
                batch.get("negative_prompt", None) or self.default_negative_prompt,
                system_prompt=self.encoder_prompt,
            )

        return result

    def generate_control_image(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Control image generation not implemented.")

    async def prepare_training_batch(
        self, batch: T2IControlTrainInputBatch
    ) -> T2IControlProcessedBatch:
        batch["clean_image"] = clean_image = self.resize_image(batch["clean_image"])
        image_size = clean_image.shape[2], clean_image.shape[3]
        if (prompt := batch.get("prompt", None)) is None:
            batch["prompt"] = prompt = await self.chat_completion(
                self.caption_prompt, images=[clean_image]
            )
        clean_latents = self.encode_latents(clean_image)
        if (control_image := batch.get("control_image", None)) is None:
            batch["control_image"] = control_image = self.generate_control_image(
                clean_image
            )
        control_image = resize_to_resolution(control_image, image_size)
        control_latents = self.encode_latents(control_image)

        result = T2IControlProcessedBatch(
            image_size=image_size,
            clean_latents=clean_latents,
            control_latents=control_latents,
            **self.encode_prompt(prompt, system_prompt=self.encoder_prompt),
        )
        if self.save_negative:
            result["negative"] = self.encode_prompt(
                self.default_negative_prompt,
                system_prompt=self.encoder_prompt,
            )
        return result

    def get_latent_length(self, batch: T2IControlProcessedBatch):
        return super().get_latent_length(batch) + batch["prompt_embeds"].shape[1]
