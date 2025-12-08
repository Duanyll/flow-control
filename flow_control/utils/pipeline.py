"""
Multi-stage data processing pipeline framework with multiprocessing support.

This module provides a flexible pipeline framework for processing large datasets
through multiple stages, with support for:
- Multiple CPU/GPU workers per stage
- Data filtering (return empty list) and splitting (return multiple items)
- Dynamic progress tracking across stages
- Backpressure management via bounded queues
- Centralized logging from worker processes
- Graceful termination without sentinel propagation issues

Usage:
    See the `pipeline_demo.py` section for a complete example.
"""

import queue
import time
import torch
import torch.multiprocessing as mp
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional, Type, Tuple
from logging.handlers import QueueListener, QueueHandler
from multiprocessing.synchronize import Event as EventType
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TaskID,
)

from .logging import get_logger, setup_global_handler, rich_handler, console

logger = get_logger(__name__)


# =============================================================================
# Abstract Base Classes
# =============================================================================


class PipelineStage(ABC):
    """Abstract base class for pipeline processing stages."""

    @abstractmethod
    def __init__(self, worker_id: int, device: Optional[int] = None, **kwargs):
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
    def process(self, item: Any) -> List[Any]:
        """
        Process a single input item.

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


# =============================================================================
# Configuration Classes
# =============================================================================


@dataclass
class SourceConfig:
    """Configuration for the data source."""

    source: Type[DataSource]
    name: str = "Scanning"
    queue_size: int = 8
    init_kwargs: dict = field(default_factory=dict)


@dataclass
class StageConfig:
    """Configuration for a processing stage."""

    stage: Type[PipelineStage]
    num_workers: int = 1
    gpu_ids: Optional[List[int]] = None  # None=CPU, [0,1,2]=assign to GPUs
    num_threads: int = 8  # torch.set_num_threads per worker
    queue_size: int = 4  # Output queue size for backpressure
    name: str = ""  # Display name for progress bar (defaults to class name)
    init_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            self.name = self.stage.__name__


@dataclass
class SinkConfig:
    """Configuration for the data sink."""

    sink: Type[DataSink]
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
    stage_stats: List[dict]
    sink_success: int
    sink_skipped: int
    elapsed_time: float
    aborted: bool = False  # True if pipeline was aborted due to error/interrupt
    error_message: Optional[str] = None


# =============================================================================
# Worker Functions
# =============================================================================


def _setup_worker_logging(name: str, log_queue: mp.Queue):
    """Set up logging for a worker process."""
    handler = QueueHandler(log_queue)
    setup_global_handler(handler)
    return get_logger(name)


def _safe_put(
    q: mp.Queue,
    item: Any,
    timeout: float = 1.0,
    logger=None,
    shutdown_event: Optional[EventType] = None,
) -> bool:
    """
    Put item to queue with timeout and backpressure handling.

    Returns:
        True if item was put, False if shutdown was requested.
    """
    total_wait = 0
    while True:
        if shutdown_event is not None and shutdown_event.is_set():
            return False
        try:
            q.put(item, timeout=timeout)
            return True
        except queue.Full:
            total_wait += timeout
            if total_wait > 120 and logger:
                logger.warning(f"Queue put blocked for {total_wait:.0f}s")
                total_wait = 0
            time.sleep(0.1)


def _safe_get(
    q: mp.Queue,
    timeout: float = 0.5,
    shutdown_event: Optional[EventType] = None,
) -> Tuple[bool, Any]:
    """
    Get item from queue with timeout.

    Returns:
        (success, item): success=True if item retrieved, False if timeout or shutdown.
    """
    if shutdown_event is not None and shutdown_event.is_set():
        return False, None
    try:
        item = q.get(timeout=timeout)
        return True, item
    except queue.Empty:
        return False, None


def _source_worker(
    source_class: Type[DataSource],
    source_kwargs: dict,
    output_queue: Any,
    progress_queue: Any,
    log_queue: Any,
    done_event: EventType,
    shutdown_event: EventType,
):
    """Worker that scans the data source and feeds items to the pipeline."""
    worker_logger = _setup_worker_logging("Source", log_queue)
    worker_logger.info("Source worker started")

    try:
        # Instantiate the source in the worker process
        source = source_class(**source_kwargs)

        last_total = None
        for item, total in source.scan():
            if shutdown_event.is_set():
                worker_logger.info("Source worker received shutdown signal")
                break
            if not _safe_put(
                output_queue, item, logger=worker_logger, shutdown_event=shutdown_event
            ):
                break
            _safe_put(progress_queue, ("source", 0, "produced", 1))
            if total != last_total:
                last_total = total
                _safe_put(progress_queue, ("source", 0, "total", total))

        # Mark scanning as complete immediately
        done_event.set()

        # But don't exit yet - wait for shutdown signal
        # to ensure downstream stages finish consuming our outputs
        if not shutdown_event.is_set():
            worker_logger.info("Scanning completed, waiting for pipeline shutdown")
            while not shutdown_event.is_set():
                time.sleep(0.5)
    except Exception as e:
        worker_logger.error(f"Source worker fatal error: {e}", exc_info=True)
        # Notify main process of fatal error
        try:
            progress_queue.put(("fatal", 0, "error", str(e)), timeout=5)
        except queue.Full:
            pass
    finally:
        worker_logger.info("Source worker finished")


def _stage_worker(
    stage_index: int,
    worker_id: int,
    stage_class: Type[PipelineStage],
    input_queue: Any,
    output_queue: Any,
    progress_queue: Any,
    log_queue: Any,
    upstream_done: EventType,
    shutdown_event: EventType,
    num_threads: int,
    device: Optional[int],
    init_kwargs: dict,
):
    """Worker that processes items through a pipeline stage."""
    torch.set_num_threads(num_threads)
    worker_name = f"Stage{stage_index}-W{worker_id}"
    worker_logger = _setup_worker_logging(worker_name, log_queue)
    worker_logger.info(f"Stage worker started (device={device}, threads={num_threads})")

    try:
        # Instantiate the stage in the worker process
        stage = stage_class(worker_id, device=device, **init_kwargs)

        work_done = False
        while not shutdown_event.is_set():
            success, item = _safe_get(input_queue, shutdown_event=shutdown_event)
            if not success:
                # Queue empty or shutdown, check if we should exit
                if shutdown_event.is_set():
                    worker_logger.info("Stage worker received shutdown signal")
                    break
                if upstream_done.is_set() and input_queue.empty():
                    # Work is done, but don't exit yet - wait for shutdown signal
                    # to ensure downstream stages finish consuming our outputs
                    if not work_done:
                        worker_logger.info("Work completed, waiting for pipeline shutdown")
                        work_done = True
                    time.sleep(0.5)
                continue

            results = stage.process(item)
            output_count = len(results)

            for result in results:
                if not _safe_put(
                    output_queue,
                    result,
                    logger=worker_logger,
                    shutdown_event=shutdown_event,
                ):
                    break

            if output_count == 0:
                _safe_put(progress_queue, ("stage", stage_index, "filtered", 1))
            else:
                _safe_put(
                    progress_queue, ("stage", stage_index, "produced", output_count)
                )

    except Exception as e:
        # Fatal error (e.g., in __init__) - notify main process
        worker_logger.error(f"Stage worker fatal error: {e}", exc_info=True)
        try:
            progress_queue.put(
                (
                    "fatal",
                    stage_index,
                    "error",
                    f"Stage{stage_index}-W{worker_id}: {e}",
                ),
                timeout=5,
            )
        except queue.Full:
            pass
    finally:
        stage.cleanup()
        worker_logger.info("Stage worker finished")


def _sink_worker(
    worker_id: int,
    sink_class: Type[DataSink],
    input_queue: Any,
    progress_queue: Any,
    log_queue: Any,
    upstream_done: EventType,
    shutdown_event: EventType,
    num_threads: int,
    init_kwargs: dict,
):
    """Worker that writes items to the sink."""
    torch.set_num_threads(num_threads)
    worker_name = f"Sink-W{worker_id}"
    worker_logger = _setup_worker_logging(worker_name, log_queue)
    worker_logger.info(f"Sink worker started (threads={num_threads})")

    try:
        # Instantiate the sink in the worker process
        sink = sink_class(worker_id, **init_kwargs)

        work_done = False
        while not shutdown_event.is_set():
            success, item = _safe_get(input_queue, shutdown_event=shutdown_event)
            if not success:
                if shutdown_event.is_set():
                    worker_logger.info("Sink worker received shutdown signal")
                    break
                if upstream_done.is_set() and input_queue.empty():
                    # Work is done, but don't exit yet - wait for shutdown signal
                    # to ensure upstream stages don't exit before we finish
                    if not work_done:
                        worker_logger.info("Work completed, waiting for pipeline shutdown")
                        work_done = True
                    time.sleep(0.5)
                continue

            result = sink.write(item)
            if result:
                _safe_put(progress_queue, ("sink", 0, "produced", 1))
            else:
                _safe_put(progress_queue, ("sink", 0, "filtered", 1))

    except Exception as e:
        # Fatal error (e.g., in __init__) - notify main process
        worker_logger.error(f"Sink worker fatal error: {e}", exc_info=True)
        try:
            progress_queue.put(
                ("fatal", 0, "error", f"Sink-W{worker_id}: {e}"),
                timeout=5,
            )
        except queue.Full:
            pass
    finally:
        sink.cleanup()
        worker_logger.info("Sink worker finished")


# =============================================================================
# Pipeline Class
# =============================================================================


class Pipeline:
    """
    Multi-stage data processing pipeline.

    The pipeline processes data through multiple stages:
    DataSource -> Stage[0] -> Stage[1] -> ... -> Sink

    Each stage can have multiple workers, and stages are connected via queues.
    Workers report progress to the main process, which displays progress bars
    and manages graceful termination.
    """

    def __init__(
        self,
        source: SourceConfig,
        stages: List[StageConfig],
        sink: SinkConfig,
    ):
        """
        Initialize the pipeline.

        Args:
            source: Data source configuration.
            stages: List of processing stage configurations.
            sink: Sink configuration for writing results.
        """
        self.source = source
        self.stages = stages
        self.sink = sink

    def run(self) -> PipelineResult:
        """
        Run the pipeline.

        Returns:
            PipelineResult with execution statistics.
        """
        mp.set_start_method("spawn", force=True)
        ctx = mp.get_context("spawn")
        start_time = time.time()

        # Create queues
        log_queue = ctx.Queue()
        progress_queue = ctx.Queue()

        # Source -> Stage[0]
        stage_queues = [ctx.Queue(maxsize=self.source.queue_size)]
        # Stage[i] -> Stage[i+1] or Sink
        for stage_cfg in self.stages:
            stage_queues.append(ctx.Queue(maxsize=stage_cfg.queue_size))

        # Done events for each stage
        source_done = ctx.Event()
        stage_done_events = [ctx.Event() for _ in self.stages]
        sink_done = ctx.Event()

        # Shutdown event for graceful termination
        shutdown_event = ctx.Event()

        # Start log listener
        log_listener = QueueListener(
            log_queue, rich_handler, respect_handler_level=True
        )
        log_listener.start()

        processes = []

        try:
            # Start source worker
            source_proc = ctx.Process(
                target=_source_worker,
                args=(
                    self.source.source,
                    self.source.init_kwargs,
                    stage_queues[0],
                    progress_queue,
                    log_queue,
                    source_done,
                    shutdown_event,
                ),
            )
            source_proc.start()
            processes.append(source_proc)
            logger.info("Started source worker")

            # Start stage workers
            for stage_idx, stage_cfg in enumerate(self.stages):
                upstream_done = (
                    source_done if stage_idx == 0 else stage_done_events[stage_idx - 1]
                )
                input_queue = stage_queues[stage_idx]
                output_queue = stage_queues[stage_idx + 1]

                for worker_id in range(stage_cfg.num_workers):
                    # Determine device
                    device = None
                    if stage_cfg.gpu_ids:
                        device = stage_cfg.gpu_ids[worker_id % len(stage_cfg.gpu_ids)]

                    proc = ctx.Process(
                        target=_stage_worker,
                        args=(
                            stage_idx,
                            worker_id,
                            stage_cfg.stage,
                            input_queue,
                            output_queue,
                            progress_queue,
                            log_queue,
                            upstream_done,
                            shutdown_event,
                            stage_cfg.num_threads,
                            device,
                            stage_cfg.init_kwargs,
                        ),
                    )
                    proc.start()
                    processes.append(proc)

                logger.info(
                    f"Started {stage_cfg.num_workers} workers for stage '{stage_cfg.name}'"
                )

            # Start sink workers
            last_stage_done = stage_done_events[-1] if self.stages else source_done
            sink_queue = stage_queues[-1]

            for worker_id in range(self.sink.num_workers):
                proc = ctx.Process(
                    target=_sink_worker,
                    args=(
                        worker_id,
                        self.sink.sink,
                        sink_queue,
                        progress_queue,
                        log_queue,
                        last_stage_done,
                        shutdown_event,
                        self.sink.num_threads,
                        self.sink.init_kwargs,
                    ),
                )
                proc.start()
                processes.append(proc)

            logger.info(f"Started {self.sink.num_workers} sink workers")

            # Monitor progress
            result = self._monitor_progress(
                progress_queue,
                source_done,
                stage_done_events,
                sink_done,
                shutdown_event,
            )

        except KeyboardInterrupt:
            logger.warning("Received keyboard interrupt, shutting down...")
            shutdown_event.set()
            result = PipelineResult(
                source_total=0,
                stage_stats=[],
                sink_success=0,
                sink_skipped=0,
                elapsed_time=0,
                aborted=True,
                error_message="Interrupted by user",
            )

        finally:
            # Wait for all processes to finish
            for proc in processes:
                proc.join(timeout=10)
                if proc.is_alive():
                    logger.warning(f"Process {proc.pid} did not terminate, killing")
                    proc.terminate()
                    proc.join(timeout=5)

            log_listener.stop()

        elapsed = time.time() - start_time
        result.elapsed_time = elapsed

        # Print summary
        if result.aborted:
            console.print(
                f"\n[bold red]Pipeline aborted after {elapsed:.1f}s[/bold red]"
            )
            if result.error_message:
                console.print(f"  Error: {result.error_message}")
        else:
            console.print(
                f"\n[bold green]Pipeline completed in {elapsed:.1f}s[/bold green]"
            )

        for i, stats in enumerate(result.stage_stats):
            name = self.stages[i].name if i < len(self.stages) else self.sink.name
            console.print(
                f"  {name}: completed={stats['completed']}, filtered={stats['filtered']}"
            )
        if result.stage_stats:
            console.print(
                f"  Sink: success={result.sink_success}, skipped={result.sink_skipped}"
            )

        return result

    def _monitor_progress(
        self,
        progress_queue: Any,
        source_done: EventType,
        stage_done_events: List[EventType],
        sink_done: EventType,
        shutdown_event: EventType,
    ) -> PipelineResult:
        """Monitor progress and update progress bars."""
        source_total = None
        source_count = 0

        # Initialize stats
        stage_stats = [StageStats() for _ in self.stages]
        sink_stats = StageStats()

        # Track pending counts
        # pending[i] = items sent to stage[i] but not yet processed
        pending = [0] * (len(self.stages) + 1)  # +1 for sink

        # Track abort state
        aborted = False
        error_message = None

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>5.1f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[extra]}"),
            console=console,
        ) as progress:
            # Create tasks
            source_task = progress.add_task(
                f"[cyan]{self.source.name}",
                total=source_total,
                extra="",
            )

            stage_tasks: List[TaskID] = []
            for stage_cfg in self.stages:
                task = progress.add_task(
                    f"[green]{stage_cfg.name}",
                    total=source_total,
                    extra="",
                )
                stage_tasks.append(task)

            sink_task = progress.add_task(
                f"[yellow]{self.sink.name}",
                total=source_total,
                extra="",
            )

            all_done = False
            try:
                while not all_done:
                    try:
                        msg = progress_queue.get(timeout=0.5)
                    except queue.Empty:
                        # Check if shutdown was requested
                        if shutdown_event.is_set():
                            all_done = True
                            aborted = True
                            continue
                        # Check if everything is done
                        if (
                            source_done.is_set()
                            and all(e.is_set() for e in stage_done_events)
                            and all(p == 0 for p in pending)
                        ):
                            all_done = True
                        continue

                    stage_type, stage_idx, event_type, count = msg

                    # Handle fatal errors from workers
                    if stage_type == "fatal":
                        error_message = str(count)  # count contains error message
                        logger.error(f"Fatal error from worker: {error_message}")
                        shutdown_event.set()
                        aborted = True
                        all_done = True
                        continue

                    if stage_type == "source":
                        if event_type == "total":
                            source_total = count
                            progress.update(source_task, total=source_total)
                            if self.stages:
                                progress.update(
                                    stage_tasks[0], total=source_total
                                )
                        elif event_type == "produced":
                            source_count += count
                            pending[0] += count
                            progress.update(source_task, completed=source_count)

                    elif stage_type == "stage":
                        stats = stage_stats[stage_idx]
                        pending[stage_idx] -= 1

                        if event_type == "produced":
                            stats.completed += 1
                            stats.total_output += count
                            pending[stage_idx + 1] += count
                        elif event_type == "filtered":
                            stats.completed += 1
                            stats.filtered += 1

                        # Update progress bar
                        extra = f"⊘{stats.filtered}" if stats.filtered > 0 else ""
                        progress.update(
                            stage_tasks[stage_idx],
                            completed=stats.completed,
                            extra=extra,
                        )

                        # Check if this stage is done
                        upstream_done_flag = (
                            source_done.is_set()
                            if stage_idx == 0
                            else stage_done_events[stage_idx - 1].is_set()
                        )
                        if upstream_done_flag and pending[stage_idx] == 0:
                            stage_done_events[stage_idx].set()

                        delta = 0
                        for i in range(len(self.stages)):
                            delta += stage_stats[i].total_output - stage_stats[i].completed
                            display_total = (
                                source_total + delta
                                if source_total is not None
                                else stage_stats[i].total_output
                            )
                            if i + 1 < len(self.stages):
                                progress.update(
                                    stage_tasks[i + 1],
                                    total=display_total,
                                )
                            else:
                                progress.update(
                                    sink_task,
                                    total=display_total,
                                )

                    elif stage_type == "sink":
                        pending[-1] -= 1

                        if event_type == "produced":
                            sink_stats.completed += 1
                        elif event_type == "filtered":
                            sink_stats.completed += 1
                            sink_stats.filtered += 1

                        extra = (
                            f"⊘{sink_stats.filtered}" if sink_stats.filtered > 0 else ""
                        )
                        progress.update(
                            sink_task,
                            completed=sink_stats.completed,
                            extra=extra,
                        )

                        # Check if sink is done
                        last_stage_done = (
                            stage_done_events[-1].is_set()
                            if self.stages
                            else source_done.is_set()
                        )
                        if last_stage_done and pending[-1] == 0:
                            sink_done.set()
                            # Signal all workers to shutdown now that work is complete
                            shutdown_event.set()
                            all_done = True

            except KeyboardInterrupt:
                logger.warning("Keyboard interrupt in monitor, shutting down...")
                shutdown_event.set()
                aborted = True
                error_message = "Interrupted by user"

        return PipelineResult(
            source_total=source_count,
            stage_stats=[
                {
                    "completed": s.completed,
                    "filtered": s.filtered,
                    "total_output": s.total_output,
                }
                for s in stage_stats
            ],
            sink_success=sink_stats.completed - sink_stats.filtered,
            sink_skipped=sink_stats.filtered,
            elapsed_time=0,
            aborted=aborted,
            error_message=error_message,
        )


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
