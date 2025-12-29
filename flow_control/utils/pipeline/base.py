"""
Abstract base classes for pipeline framework.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any


class PipelineStage(ABC):
    """Abstract base class for pipeline processing stages.

    The process method can be either synchronous or asynchronous (async).
    Use async process for I/O-bound tasks like LLM API calls to increase throughput.
    """

    @abstractmethod
    def __init__(self, worker_id: int, device: int | None = None, **kwargs):
        """
        Initialize the stage in the worker process.

        This is called once when the worker starts, use it to load models,
        open database connections, etc.

        Args:
            worker_id: Unique identifier for this worker.
            device: GPU device ID if applicable, None for CPU workers.
            **kwargs: Additional arguments from StageConfig.init_kwargs.
        """
        pass

    @abstractmethod
    def process(self, item: Any) -> list[Any]:
        """
        Process a single input item.

        This method can be either sync or async:
        - Sync: def process(self, item) -> list[Any]
        - Async: async def process(self, item) -> list[Any]

        Use async for I/O-bound tasks (e.g., LLM API calls) to enable concurrent
        execution within a single worker. Configure StageConfig.max_concurrency
        to control the number of concurrent async calls per worker.

        Args:
            item: Input data item from the previous stage.

        Returns:
            - [] (empty list): Item is filtered/skipped.
            - [result]: Normal output, one item.
            - [r1, r2, ...]: Split into multiple outputs.
        """
        pass

    def cleanup(self):
        """Clean up resources when the worker exits."""
        pass


class DataSource(ABC):
    """Abstract base class for data sources."""

    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initialize the data source in the worker process.

        Args:
            **kwargs: Arguments from the pipeline configuration.
        """
        pass

    @abstractmethod
    def scan(self) -> Iterator[tuple[Any, int | None]]:
        """
        Lazily scan and yield data items.

        This should be a generator that yields items one by one,
        suitable for scanning 100k+ items without loading all into memory.

        Yields:
            Tuple (Any, int | None): Data item, and predicted total item count (or None if unknown).
        """
        pass


class DataSink(ABC):
    """Abstract base class for data sinks (writers)."""

    @abstractmethod
    def __init__(self, worker_id: int, **kwargs):
        """
        Initialize the sink in the worker process.

        Args:
            worker_id: Unique identifier for this worker.
            **kwargs: Additional arguments from SinkConfig.init_kwargs.
        """
        pass

    @abstractmethod
    def write(self, item: Any) -> bool:
        """
        Write a single item.

        Args:
            item: Data item to write.

        Returns:
            True if successful, False if skipped/failed.
        """
        pass

    def cleanup(self):
        """Clean up resources when the worker exits."""
        pass
