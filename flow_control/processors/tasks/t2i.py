from typing import NotRequired

import torch

from ..base import BaseProcessor, InputBatch, ProcessedBatch, TrainInputBatch
from ..components.prompts import PromptStr, parse_prompt


class T2IInputBatch(InputBatch):
    prompt: str
    negative_prompt: NotRequired[str | None]


class T2ITrainInputBatch(TrainInputBatch):
    prompt: NotRequired[str | None]
    clean_image: torch.Tensor


class T2IProcessedBatch(ProcessedBatch):
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor | None


class T2IProcessor(BaseProcessor):
    encoder_prompt: PromptStr
    caption_prompt: PromptStr = parse_prompt("@default_t2i_caption")
    default_negative_prompt: str = " "
    save_negative: bool = False

    async def prepare_inference_batch(self, batch: T2IInputBatch) -> T2IProcessedBatch:
        image_size = batch.get("image_size", None) or self.default_resolution

        result = T2IProcessedBatch(
            image_size=image_size,
            **self.encode_prompt(batch["prompt"], system_prompt=self.encoder_prompt),
        )

        if self.save_negative:
            result["negative"] = self.encode_prompt(
                batch.get("negative_prompt", None) or self.default_negative_prompt,
                system_prompt=self.encoder_prompt,
            )

        return result

    async def prepare_training_batch(
        self, batch: T2ITrainInputBatch
    ) -> T2IProcessedBatch:
        batch["clean_image"] = clean_image = self.resize_image(batch["clean_image"])
        image_size = clean_image.shape[2], clean_image.shape[3]
        if (prompt := batch.get("prompt", None)) is None:
            batch["prompt"] = prompt = await self.chat_completion(
                self.caption_prompt, images=[clean_image]
            )
        clean_latents = self.encode_latents(clean_image)

        result = T2IProcessedBatch(
            image_size=image_size,
            clean_latents=clean_latents,
            **self.encode_prompt(prompt, system_prompt=self.encoder_prompt),
        )
        if self.save_negative:
            result["negative"] = self.encode_prompt(
                self.default_negative_prompt,
                system_prompt=self.encoder_prompt,
            )
        return result

    def get_latent_length(self, batch: T2IProcessedBatch):
        return super().get_latent_length(batch) + batch["prompt_embeds"].shape[1]
