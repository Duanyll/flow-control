"""
Configuration classes and data structures for pipeline framework.
"""

from dataclasses import dataclass, field

from .base import DataSink, DataSource, PipelineStage


@dataclass
class SourceConfig:
    """Configuration for the data source."""

    source: type[DataSource]
    name: str = "Scanning"
    queue_size: int = 8
    init_kwargs: dict = field(default_factory=dict)


@dataclass
class StageConfig:
    """Configuration for a processing stage."""

    stage: type[PipelineStage]
    num_workers: int = 1
    gpu_ids: list[int] | None = None  # None=CPU, [0,1,2]=assign to GPUs
    num_threads: int = 8  # torch.set_num_threads per worker
    queue_size: int = 4  # Output queue size for backpressure
    max_concurrency: int = 1  # Max concurrent async process calls per worker
    name: str = ""  # Display name for progress bar (defaults to class name)
    init_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            self.name = self.stage.__name__


@dataclass
class SinkConfig:
    """Configuration for the data sink."""

    sink: type[DataSink]
    num_workers: int = 1
    num_threads: int = 8
    queue_size: int = 4
    name: str = "Writing"
    init_kwargs: dict = field(default_factory=dict)


@dataclass
class StageStats:
    """Statistics for a single stage."""

    pending: int = 0
    completed: int = 0
    filtered: int = 0
    total_output: int = 0


@dataclass
class PipelineResult:
    """Result of pipeline execution."""

    source_total: int
    stage_stats: list[dict]
    sink_success: int
    sink_skipped: int
    elapsed_time: float
    aborted: bool = False  # True if pipeline was aborted due to error/interrupt
    error_message: str | None = None
