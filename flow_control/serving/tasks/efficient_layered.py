"""Gradio UI template for the efficient_layered task."""

from __future__ import annotations

from typing import Any

import gradio as gr
from PIL import Image

from flow_control.datasets.coercion import _coerce_to_image_tensor

from . import register_task


@register_task("efficient_layered")
class EfficientLayeredTemplate:
    """Efficient layered decomposition: input image only.

    Layer boxes and prompts are auto-detected by the processor's LLM pipeline.
    """

    clean_image: gr.Image

    def render(self) -> list[gr.components.Component]:
        self.clean_image = gr.Image(label="Input Image", type="pil", height=384)
        return [self.clean_image]

    def coerce(self, clean_image: Image.Image | None) -> dict[str, Any]:
        if clean_image is None:
            raise gr.Error("Input image is required.")
        return {"clean_image": _coerce_to_image_tensor(clean_image)}
