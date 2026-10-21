"""Gradio serving for interactive single-request testing.

Outdated: its interfaces lag the current adapter, variant and sampler APIs and
need a refactor before it is relied on (``docs/scripts/serving.md``).
"""

from .config import ServeConfig
from .engine import ServingEngine
from .gradio_app import create_gradio_app

__all__ = ["ServeConfig", "ServingEngine", "create_gradio_app"]
