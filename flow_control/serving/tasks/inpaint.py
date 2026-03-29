"""Gradio UI template for the inpaint task."""

from __future__ import annotations

from typing import Any

import gradio as gr
from PIL import Image

from flow_control.datasets.coercion import _coerce_to_image_tensor

from . import register_task


@register_task("inpaint")
class InpaintTemplate:
    """Inpainting: prompt + source image + mask image."""

    prompt: gr.Textbox
    clean_image: gr.Image
    inpaint_mask: gr.Image

    def render(self) -> list[gr.components.Component]:
        self.prompt = gr.Textbox(
            label="Prompt",
            placeholder="Describe what should fill the masked area...",
            lines=3,
        )
        with gr.Row():
            self.clean_image = gr.Image(label="Source Image", type="pil", height=384)
            self.inpaint_mask = gr.Image(
                label="Inpaint Mask (white = inpaint)", type="pil", height=384
            )
        return [self.prompt, self.clean_image, self.inpaint_mask]

    def coerce(
        self,
        prompt: str,
        clean_image: Image.Image | None,
        inpaint_mask: Image.Image | None,
    ) -> dict[str, Any]:
        if clean_image is None:
            raise gr.Error("Source image is required.")
        if inpaint_mask is None:
            raise gr.Error("Inpaint mask is required.")
        return {
            "prompt": prompt,
            "clean_image": _coerce_to_image_tensor(clean_image),
            "inpaint_mask": _coerce_to_image_tensor(inpaint_mask),
        }
