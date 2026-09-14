from typing import ClassVar, Literal, NotRequired

import torch

from flow_control.data.coercion import ImageTensor
from flow_control.utils.resize import resize_to_resolution

from ..base import (
    BaseProcessor,
    InputRow,
    ProcessedRow,
    TrainInputRow,
    task_registry,
)
from ..components.prompts import PromptStr, parse_prompt


class T2IControlInputRow(InputRow):
    prompt: str
    negative_prompt: NotRequired[str | None]
    control_image: ImageTensor


class T2IControlTrainInputRow(TrainInputRow):
    prompt: NotRequired[str | None]
    control_image: NotRequired[ImageTensor | None]
    clean_image: ImageTensor


class T2IControlProcessedRow(ProcessedRow):
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor | None
    control_latents: torch.Tensor


@task_registry.register("t2i_control")
class T2IControlProcessor(
    BaseProcessor[T2IControlInputRow, T2IControlTrainInputRow, T2IControlProcessedRow]
):
    task: Literal["t2i_control"] = "t2i_control"
    encoder_prompt: PromptStr = ""
    caption_prompt: PromptStr = parse_prompt("@default_t2i_caption")
    default_negative_prompt: str = " "
    save_negative: bool = False
    posterior_fields: ClassVar[tuple[str, ...]] = ("clean_latents", "control_latents")

    async def prepare_inference_row(
        self, row: T2IControlInputRow
    ) -> T2IControlProcessedRow:
        control_image = row["control_image"] = self.resize_image(row["control_image"])
        image_size = (control_image.shape[2], control_image.shape[3])
        control_latents = self.encode_latents(
            control_image, posterior=self.condition_posterior
        )
        result = T2IControlProcessedRow(
            image_size=image_size,
            control_latents=control_latents,
            **self.encode_prompt(row["prompt"], system_prompt=self.encoder_prompt),
        )

        if self.save_negative:
            result["negative"] = self.encode_prompt(
                row.get("negative_prompt", None) or self.default_negative_prompt,
                system_prompt=self.encoder_prompt,
            )

        return result

    def generate_control_image(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Control image generation not implemented.")

    async def prepare_training_row(
        self, row: T2IControlTrainInputRow
    ) -> T2IControlProcessedRow:
        row["clean_image"] = clean_image = self.resize_image(row["clean_image"])
        image_size = clean_image.shape[2], clean_image.shape[3]
        if (prompt := row.get("prompt", None)) is None:
            row["prompt"] = prompt = await self.chat_completion(
                self.caption_prompt, images=[clean_image]
            )
        clean_latents = self.encode_latents(
            clean_image, posterior=self.target_posterior
        )
        if (control_image := row.get("control_image", None)) is None:
            row["control_image"] = control_image = self.generate_control_image(
                clean_image
            )
        control_image = resize_to_resolution(control_image, image_size)
        control_latents = self.encode_latents(
            control_image, posterior=self.condition_posterior
        )

        result = T2IControlProcessedRow(
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

    def get_cost(self, row: T2IControlProcessedRow) -> int:
        return super().get_cost(row) + row["prompt_embeds"].shape[1]
