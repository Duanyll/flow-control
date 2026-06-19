"""Serving task registry + the ``TaskTemplate`` protocol.

Lives in its own module (not the package ``__init__``) so the per-task modules can
import ``register_task`` without an import cycle — which lets ``__init__`` import
them at the top (no E402) and re-export via ``__all__`` (no F401), matching the
registration style of the other families.
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
task_template_registry: Registry[TaskTemplate] = Registry("serving_task")


def register_task[C](name: str) -> Callable[[type[C]], type[C]]:
    """Thin wrapper over ``task_template_registry.register`` (identity-preserving)."""
    return task_template_registry.register(name)


def get_task_template(name: str) -> TaskTemplate:
    template = task_template_registry.get(name)
    if template is None:
        raise ValueError(
            f"No Gradio template for task '{name}'. "
            f"Available: {sorted(task_template_registry.members())}"
        )
    return template()
