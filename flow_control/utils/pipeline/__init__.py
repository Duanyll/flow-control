"""
Multi-stage data processing pipeline framework with multiprocessing support.

This module provides a flexible pipeline framework for processing large datasets
through multiple stages, with support for:
- Multiple CPU/GPU workers per stage
- Async process methods with configurable concurrency (ideal for LLM API calls)
- Data filtering (return empty list) and splitting (return multiple items)
- Dynamic progress tracking across stages
- Backpressure management via bounded queues
- Centralized logging from worker processes
- Graceful termination without sentinel propagation issues

Usage:
    See the `examples/pipeline.py` section for a complete example.
"""

from .base import DataSink, DataSource, PipelineStage
from .config import (
    PipelineResult,
    SinkConfig,
    SourceConfig,
    StageConfig,
)
from .pipeline import Pipeline

__all__ = [
    "PipelineStage",
    "DataSource",
    "DataSink",
    "SourceConfig",
    "StageConfig",
    "SinkConfig",
    "PipelineResult",
    "Pipeline",
]
