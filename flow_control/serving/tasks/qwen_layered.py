"""Gradio UI template for the qwen_layered task."""

from __future__ import annotations

from typing import Any

import gradio as gr
from PIL import Image

from flow_control.datasets.coercion import _coerce_to_image_tensor

from . import register_task


@register_task("qwen_layered")
class QwenLayeredTemplate:
    """Qwen layered decomposition: image + optional prompt and layer count."""

    clean_image: gr.Image
    prompt: gr.Textbox
    negative_prompt: gr.Textbox
    num_layers: gr.Number

    def render(self) -> list[gr.components.Component]:
        self.clean_image = gr.Image(label="Input Image", type="pil", height=384)
        self.prompt = gr.Textbox(
            label="Prompt (optional, auto-captioned if empty)",
            placeholder="Describe the image...",
            lines=3,
        )
        self.negative_prompt = gr.Textbox(
            label="Negative Prompt (optional)",
            placeholder="What to avoid...",
            lines=2,
        )
        self.num_layers = gr.Number(
            label="Number of Layers",
            value=4,
            precision=0,
            minimum=1,
            maximum=16,
        )
        return [
            self.clean_image,
            self.prompt,
            self.negative_prompt,
            self.num_layers,
        ]

    def coerce(
        self,
        clean_image: Image.Image | None,
        prompt: str,
        negative_prompt: str,
        num_layers: float,
    ) -> dict[str, Any]:
        if clean_image is None:
            raise gr.Error("Input image is required.")
        batch: dict[str, Any] = {
            "clean_image": _coerce_to_image_tensor(clean_image),
        }
        if prompt:
            batch["prompt"] = prompt
        if negative_prompt:
            batch["negative_prompt"] = negative_prompt
        if num_layers > 0:
            batch["num_layers"] = int(num_layers)
        return batch
