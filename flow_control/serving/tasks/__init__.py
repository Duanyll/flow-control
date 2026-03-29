"""Per-task Gradio UI templates.

Each task module exports:
- ``render()``: creates Gradio components, returns a list of component references.
- ``coerce(…)``: converts Gradio outputs into a dict suitable for
  ``processor.prepare_inference_batch()``.
"""

from __future__ import annotations

from typing import Any, Protocol


class TaskTemplate(Protocol):
    """Protocol for a task UI template.

    ``render()`` creates Gradio components and returns them as a list.
    ``coerce()`` takes the Gradio component values and returns an InputBatch dict.
    Concrete implementations define typed signatures; the protocol uses ``Any``
    for compatibility.
    """

    def render(self) -> list: ...
    def coerce(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...


TASK_TEMPLATE_REGISTRY: dict[str, type[TaskTemplate]] = {}


def register_task(name: str):
    def decorator(cls: type[TaskTemplate]):
        TASK_TEMPLATE_REGISTRY[name] = cls
        return cls

    return decorator


def get_task_template(name: str) -> TaskTemplate:
    if name not in TASK_TEMPLATE_REGISTRY:
        raise ValueError(
            f"No Gradio template for task '{name}'. "
            f"Available: {list(TASK_TEMPLATE_REGISTRY.keys())}"
        )
    return TASK_TEMPLATE_REGISTRY[name]()


# Import task modules to trigger registration
from . import inpaint as _inpaint  # noqa: E402, F401
from . import t2i as _t2i  # noqa: E402, F401
from . import tie as _tie  # noqa: E402, F401
