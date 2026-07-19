"""Gradio UI template for the experimental ``efficient_layered`` task.

Kept separate from the package ``__init__`` so training/preprocess runs that opt
into ``flow_control.contrib.efficient_layered`` (adapter + processor) do not pull
in the Gradio/serving stack. Serving configs import this submodule explicitly:

    imports = ["flow_control.contrib.efficient_layered.serving"]

which transitively registers the adapter + processor via the package ``__init__``.
"""

from __future__ import annotations

from typing import Any

import gradio as gr
from PIL import Image

from flow_control.datasets.coercion import _coerce_to_image_tensor
from flow_control.serving.tasks.base import task_template_registry


@task_template_registry.register("efficient_layered")
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
