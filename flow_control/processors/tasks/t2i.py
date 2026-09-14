from typing import Literal, NotRequired

import torch

from flow_control.data.coercion import ImageTensor

from ..base import (
    BaseProcessor,
    InputRow,
    ProcessedRow,
    TrainInputRow,
    task_registry,
)
from ..components.prompts import PromptStr, parse_prompt


class T2IInputRow(InputRow):
    prompt: str
    negative_prompt: NotRequired[str | None]


class T2ITrainInputRow(TrainInputRow):
    prompt: NotRequired[str | None]
    clean_image: ImageTensor


class T2IProcessedRow(ProcessedRow):
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor | None


@task_registry.register("t2i")
class T2IProcessor(BaseProcessor[T2IInputRow, T2ITrainInputRow, T2IProcessedRow]):
    task: Literal["t2i"] = "t2i"
    encoder_prompt: PromptStr = ""
    caption_prompt: PromptStr = parse_prompt("@default_t2i_caption")
    t2i_enhance_prompt: PromptStr = parse_prompt("@default_t2i_enhance")
    default_negative_prompt: str = " "
    save_negative: bool = False
    enable_enhance: bool = False

    prepend_trigger_words: str | None = None

    async def enhance_prompt(self, prompt: str) -> str:
        if self.enable_enhance:
            prompt = await self.chat_completion(
                prompt=prompt, system_prompt=self.t2i_enhance_prompt
            )
        if self.prepend_trigger_words is not None and not prompt.startswith(
            self.prepend_trigger_words
        ):
            prompt = self.prepend_trigger_words + prompt
        return prompt

    async def prepare_inference_row(self, row: T2IInputRow) -> T2IProcessedRow:
        image_size = row.get("image_size", None) or self.default_resolution

        row["prompt"] = await self.enhance_prompt(row["prompt"])

        result = T2IProcessedRow(
            image_size=image_size,
            **self.encode_prompt(row["prompt"], system_prompt=self.encoder_prompt),
        )

        if self.save_negative:
            result["negative"] = self.encode_prompt(
                row.get("negative_prompt", None) or self.default_negative_prompt,
                system_prompt=self.encoder_prompt,
            )

        return result

    async def prepare_training_row(self, row: T2ITrainInputRow) -> T2IProcessedRow:
        row["clean_image"] = clean_image = self.resize_image(row["clean_image"])
        image_size = clean_image.shape[2], clean_image.shape[3]
        if (prompt := row.get("prompt", None)) is None:
            row["prompt"] = prompt = await self.chat_completion(
                self.caption_prompt, images=[clean_image]
            )
        clean_latents = self.encode_latents(
            clean_image, posterior=self.target_posterior
        )

        row["prompt"] = prompt = await self.enhance_prompt(prompt)

        result = T2IProcessedRow(
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

    def get_cost(self, row: T2IProcessedRow) -> int:
        return super().get_cost(row) + row["prompt_embeds"].shape[1]
