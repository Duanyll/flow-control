"""Per-task Gradio UI templates.

Each task module exports:
- ``render()``: creates Gradio components, returns a list of component references.
- ``coerce(…)``: converts Gradio outputs into a dict suitable for
  ``processor.prepare_inference_batch()``.
"""

from __future__ import annotations

from . import (
    inpaint,
    qwen_layered,
    t2i,
    t2i_control,
    tie,
)
from .base import (
    TaskTemplate,
    task_template_registry,
)

__all__ = [
    "TaskTemplate",
    "inpaint",
    "qwen_layered",
    "t2i",
    "t2i_control",
    "task_template_registry",
    "tie",
]
