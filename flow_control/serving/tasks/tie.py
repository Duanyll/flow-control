"""Gradio UI template for the tie (text-image exemplar) task."""

from __future__ import annotations

from typing import Any

import gradio as gr
from PIL import Image

from flow_control.datasets.coercion import _coerce_to_image_tensor

from . import register_task


@register_task("tie")
class TIETemplate:
    """Text-image exemplar: prompt + reference images."""

    prompt: gr.Textbox
    negative_prompt: gr.Textbox
    reference_images: gr.File

    def render(self) -> list[gr.components.Component]:
        self.prompt = gr.Textbox(
            label="Prompt",
            placeholder="Describe the image to generate using the reference style...",
            lines=3,
        )
        self.negative_prompt = gr.Textbox(
            label="Negative Prompt (optional)",
            placeholder="What to avoid...",
            lines=2,
        )
        self.reference_images = gr.File(
            label="Reference Images",
            file_count="multiple",
            file_types=["image"],
        )
        return [self.prompt, self.negative_prompt, self.reference_images]

    def coerce(
        self,
        prompt: str,
        negative_prompt: str,
        reference_images: list[Any] | None,
    ) -> dict[str, Any]:
        if not reference_images:
            raise gr.Error("At least one reference image is required.")
        images = []
        for file_data in reference_images:
            pil_img = Image.open(file_data)
            images.append(_coerce_to_image_tensor(pil_img))
        batch: dict[str, Any] = {
            "prompt": prompt,
            "reference_images": images,
        }
        if negative_prompt:
            batch["negative_prompt"] = negative_prompt
        return batch
