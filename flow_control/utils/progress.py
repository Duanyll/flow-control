"""Lightweight progress reporting via :mod:`contextvars`.

Library code (samplers, engine, etc.) calls :func:`report_progress` at key
points.  The caller decides what to *do* with those reports by setting
:data:`progress_var` to a callback — e.g. a Gradio ``gr.Progress`` wrapper.
When no callback is registered the call is a no-op.
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar

ProgressCallback = Callable[[float, str], None]
"""``(fraction, description) -> None``.

*fraction* is in ``[0, 1]`` for determinate progress.
"""

progress_var: ContextVar[ProgressCallback | None] = ContextVar(
    "progress_callback", default=None
)


def report_progress(fraction: float, description: str) -> None:
    """Report progress if a callback is registered in the current context."""
    cb = progress_var.get(None)
    if cb is not None:
        cb(fraction, description)
