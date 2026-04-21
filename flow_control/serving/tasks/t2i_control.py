"""Gradio UI template for the t2i_control (text-to-image with control) task."""

from __future__ import annotations

from typing import Any

import gradio as gr
from PIL import Image

from flow_control.datasets.coercion import _coerce_to_image_tensor

from . import register_task


@register_task("t2i_control")
class T2IControlTemplate:
    """Text-to-image with control image: prompt + control image + optional size."""

    prompt: gr.Textbox
    negative_prompt: gr.Textbox
    control_image: gr.Image
    height: gr.Number
    width: gr.Number

    def render(self) -> list[gr.components.Component]:
        self.prompt = gr.Textbox(
            label="Prompt",
            placeholder="Describe the image you want to generate...",
            lines=3,
        )
        self.negative_prompt = gr.Textbox(
            label="Negative Prompt (optional)",
            placeholder="What to avoid in the image...",
            lines=2,
        )
        self.control_image = gr.Image(label="Control Image", type="pil", height=384)
        with gr.Row():
            self.height = gr.Number(label="Height", value=0, precision=0)
            self.width = gr.Number(label="Width", value=0, precision=0)
        return [
            self.prompt,
            self.negative_prompt,
            self.control_image,
            self.height,
            self.width,
        ]

    def coerce(
        self,
        prompt: str,
        negative_prompt: str,
        control_image: Image.Image | None,
        height: float,
        width: float,
    ) -> dict[str, Any]:
        if control_image is None:
            raise gr.Error("Control image is required.")
        batch: dict[str, Any] = {
            "prompt": prompt,
            "control_image": _coerce_to_image_tensor(control_image),
        }
        if negative_prompt:
            batch["negative_prompt"] = negative_prompt
        if height > 0 and width > 0:
            batch["image_size"] = (int(height), int(width))
        return batch
