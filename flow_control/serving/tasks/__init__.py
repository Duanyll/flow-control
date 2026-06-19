"""Per-task Gradio UI templates.

Each task module exports:
- ``render()``: creates Gradio components, returns a list of component references.
- ``coerce(…)``: converts Gradio outputs into a dict suitable for
  ``processor.prepare_inference_batch()``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from flow_control.utils.registry import Registry


class TaskTemplate(Protocol):
    """Protocol for a task UI template.

    ``render()`` creates Gradio components and returns them as a list.
    ``coerce()`` takes the Gradio component values and returns an InputBatch dict.
    Concrete implementations define typed signatures; the protocol uses ``Any``
    for compatibility.
    """

    def render(self) -> list: ...
    def coerce(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...


# Duck-typed family: ``TaskTemplate`` is a Protocol, so ``base=None`` skips the
# nominal subclass guard.
TASK_TEMPLATE_REGISTRY: Registry[TaskTemplate] = Registry("serving_task")


def register_task[C](name: str) -> Callable[[type[C]], type[C]]:
    """Thin wrapper over ``TASK_TEMPLATE_REGISTRY.register`` (identity-preserving)."""
    return TASK_TEMPLATE_REGISTRY.register(name)


def get_task_template(name: str) -> TaskTemplate:
    template = TASK_TEMPLATE_REGISTRY.get(name)
    if template is None:
        raise ValueError(
            f"No Gradio template for task '{name}'. "
            f"Available: {sorted(TASK_TEMPLATE_REGISTRY.members())}"
        )
    return template()


# Import task modules to trigger registration
from . import efficient_layered as _efficient_layered  # noqa: E402, F401
from . import inpaint as _inpaint  # noqa: E402, F401
from . import qwen_layered as _qwen_layered  # noqa: E402, F401
from . import t2i as _t2i  # noqa: E402, F401
from . import t2i_control as _t2i_control  # noqa: E402, F401
from . import tie as _tie  # noqa: E402, F401
