"""Gradio UI template for the t2i (text-to-image) task."""

from __future__ import annotations

from typing import Any

import gradio as gr

from . import register_task


@register_task("t2i")
class T2ITemplate:
    """Text-to-image: prompt + optional negative prompt + image size."""

    prompt: gr.Textbox
    negative_prompt: gr.Textbox
    width: gr.Number
    height: gr.Number

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
        with gr.Row():
            self.height = gr.Number(label="Height", value=1024, precision=0)
            self.width = gr.Number(label="Width", value=1024, precision=0)
        return [self.prompt, self.negative_prompt, self.height, self.width]

    def coerce(
        self,
        prompt: str,
        negative_prompt: str,
        height: float,
        width: float,
    ) -> dict[str, Any]:
        batch: dict[str, Any] = {"prompt": prompt}
        if negative_prompt:
            batch["negative_prompt"] = negative_prompt
        if height > 0 and width > 0:
            batch["image_size"] = (int(height), int(width))
        return batch
