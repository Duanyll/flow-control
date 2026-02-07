"""
Main Pipeline class for multi-stage data processing.
"""

import queue
import time
from dataclasses import dataclass, field
from logging.handlers import QueueListener
from multiprocessing.synchronize import Event as EventType
from typing import Any

import torch.multiprocessing as mp
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from ..logging import console, get_logger, rich_handler
from .config import (
    PipelineResult,
    SinkConfig,
    SourceConfig,
    StageConfig,
    StageStats,
)
from .workers import _sink_worker, _source_worker, _stage_worker

logger = get_logger(__name__)


@dataclass
class _MonitorState:
    """Mutable state for the progress monitor loop."""

    source_total: int | None = None
    source_count: int = 0
    stage_stats: list[StageStats] = field(default_factory=list)
    sink_stats: StageStats = field(default_factory=StageStats)
    pending: list[int] = field(default_factory=list)
    aborted: bool = False
    error_message: str | None = None
    all_done: bool = False

    def __init__(self, num_stages: int):
        self.source_total = None
        self.source_count = 0
        self.stage_stats = [StageStats() for _ in range(num_stages)]
        self.sink_stats = StageStats()
        self.pending = [0] * (num_stages + 1)
        self.aborted = False
        self.error_message = None
        self.all_done = False


@dataclass
class _ProgressTasks:
    """Rich progress task IDs."""

    source: TaskID
    stages: list[TaskID]
    sink: TaskID


def _check_idle_done(
    shutdown_event: EventType,
    source_done: EventType,
    stage_done_events: list[EventType],
    state: _MonitorState,
) -> bool:
    """Check if the pipeline is done when the queue is empty."""
    if shutdown_event.is_set():
        state.aborted = True
        return True
    return (
        source_done.is_set()
        and all(e.is_set() for e in stage_done_events)
        and all(p == 0 for p in state.pending)
    )


def _handle_source_msg(
    state: _MonitorState,
    event_type: str,
    count: int,
    progress: Progress,
    tasks: _ProgressTasks,
    has_stages: bool,
) -> None:
    """Handle a progress message from the source worker."""
    if event_type == "total":
        state.source_total = count
        progress.update(tasks.source, total=count)
        if has_stages:
            progress.update(tasks.stages[0], total=count)
    elif event_type == "produced":
        state.source_count += count
        state.pending[0] += count
        progress.update(tasks.source, completed=state.source_count)


def _handle_stage_msg(
    state: _MonitorState,
    stage_idx: int,
    event_type: str,
    count: int,
    progress: Progress,
    tasks: _ProgressTasks,
    source_done: EventType,
    stage_done_events: list[EventType],
) -> None:
    """Handle a progress message from a stage worker."""
    stats = state.stage_stats[stage_idx]
    state.pending[stage_idx] -= 1

    if event_type == "produced":
        stats.completed += 1
        stats.total_output += count
        state.pending[stage_idx + 1] += count
    elif event_type == "filtered":
        stats.completed += 1
        stats.filtered += 1

    extra = f"⊘{stats.filtered}" if stats.filtered > 0 else ""
    progress.update(tasks.stages[stage_idx], completed=stats.completed, extra=extra)

    # Check if this stage is done
    upstream_done = (
        source_done.is_set()
        if stage_idx == 0
        else stage_done_events[stage_idx - 1].is_set()
    )
    if upstream_done and state.pending[stage_idx] == 0:
        stage_done_events[stage_idx].set()

    # Update downstream totals
    _update_downstream_totals(state, progress, tasks)


def _update_downstream_totals(
    state: _MonitorState,
    progress: Progress,
    tasks: _ProgressTasks,
) -> None:
    """Recalculate and update the total for each downstream task."""
    delta = 0
    num_stages = len(state.stage_stats)
    for i in range(num_stages):
        delta += state.stage_stats[i].total_output - state.stage_stats[i].completed
        display_total = (
            state.source_total + delta
            if state.source_total is not None
            else state.stage_stats[i].total_output
        )
        target_task = tasks.stages[i + 1] if i + 1 < num_stages else tasks.sink
        progress.update(target_task, total=display_total)


def _handle_sink_msg(
    state: _MonitorState,
    event_type: str,
    progress: Progress,
    tasks: _ProgressTasks,
    source_done: EventType,
    stage_done_events: list[EventType],
    sink_done: EventType,
    shutdown_event: EventType,
    has_stages: bool,
) -> None:
    """Handle a progress message from a sink worker."""
    state.pending[-1] -= 1

    if event_type == "produced":
        state.sink_stats.completed += 1
    elif event_type == "filtered":
        state.sink_stats.completed += 1
        state.sink_stats.filtered += 1

    extra = f"⊘{state.sink_stats.filtered}" if state.sink_stats.filtered > 0 else ""
    progress.update(tasks.sink, completed=state.sink_stats.completed, extra=extra)

    last_done = stage_done_events[-1].is_set() if has_stages else source_done.is_set()
    if last_done and state.pending[-1] == 0:
        sink_done.set()
        shutdown_event.set()
        state.all_done = True


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
        stages: list[StageConfig],
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

    def _spawn_workers(
        self,
        ctx,
        stage_queues,
        progress_queue,
        log_queue,
        source_done,
        stage_done_events,
        shutdown_event,
    ):
        """Spawn source, stage, and sink worker processes."""
        processes = []

        # Source worker
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

        # Stage workers
        for stage_idx, stage_cfg in enumerate(self.stages):
            upstream_done = (
                source_done if stage_idx == 0 else stage_done_events[stage_idx - 1]
            )
            for worker_id in range(stage_cfg.num_workers):
                device = None
                if stage_cfg.gpu_ids:
                    device = stage_cfg.gpu_ids[worker_id % len(stage_cfg.gpu_ids)]
                proc = ctx.Process(
                    target=_stage_worker,
                    args=(
                        stage_idx,
                        worker_id,
                        stage_cfg.stage,
                        stage_queues[stage_idx],
                        stage_queues[stage_idx + 1],
                        progress_queue,
                        log_queue,
                        upstream_done,
                        shutdown_event,
                        stage_cfg.num_threads,
                        device,
                        stage_cfg.init_kwargs,
                        stage_cfg.max_concurrency,
                    ),
                )
                proc.start()
                processes.append(proc)
            logger.info(
                f"Started {stage_cfg.num_workers} workers for stage '{stage_cfg.name}'"
            )

        # Sink workers
        last_stage_done = stage_done_events[-1] if self.stages else source_done
        for worker_id in range(self.sink.num_workers):
            proc = ctx.Process(
                target=_sink_worker,
                args=(
                    worker_id,
                    self.sink.sink,
                    stage_queues[-1],
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

        return processes

    def _print_summary(self, result: PipelineResult) -> None:
        """Print pipeline execution summary."""
        elapsed = result.elapsed_time
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

        stage_queues = [ctx.Queue(maxsize=self.source.queue_size)]
        for stage_cfg in self.stages:
            stage_queues.append(ctx.Queue(maxsize=stage_cfg.queue_size))

        # Done events
        source_done = ctx.Event()
        stage_done_events = [ctx.Event() for _ in self.stages]
        sink_done = ctx.Event()
        shutdown_event = ctx.Event()

        # Start log listener
        log_listener = QueueListener(
            log_queue, rich_handler, respect_handler_level=True
        )
        log_listener.start()

        processes: list = []

        try:
            processes = self._spawn_workers(
                ctx,
                stage_queues,
                progress_queue,
                log_queue,
                source_done,
                stage_done_events,
                shutdown_event,
            )

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
            for proc in processes:
                proc.join(timeout=10)
                if proc.is_alive():
                    logger.warning(f"Process {proc.pid} did not terminate, killing")
                    proc.terminate()
                    proc.join(timeout=5)
            log_listener.stop()

        result.elapsed_time = time.time() - start_time
        self._print_summary(result)
        return result

    def _monitor_progress(
        self,
        progress_queue: Any,
        source_done: EventType,
        stage_done_events: list[EventType],
        sink_done: EventType,
        shutdown_event: EventType,
    ) -> PipelineResult:
        """Monitor progress and update progress bars."""
        state = _MonitorState(len(self.stages))

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
            source_task = progress.add_task(
                f"[cyan]{self.source.name}", total=None, extra=""
            )
            stage_tasks = [
                progress.add_task(f"[green]{cfg.name}", total=None, extra="")
                for cfg in self.stages
            ]
            sink_task = progress.add_task(
                f"[yellow]{self.sink.name}", total=None, extra=""
            )

            tasks = _ProgressTasks(source_task, stage_tasks, sink_task)

            try:
                while not state.all_done:
                    try:
                        msg = progress_queue.get(timeout=0.5)
                    except queue.Empty:
                        state.all_done = _check_idle_done(
                            shutdown_event, source_done, stage_done_events, state
                        )
                        continue

                    stage_type, stage_idx, event_type, count = msg

                    if stage_type == "fatal":
                        state.error_message = str(count)
                        logger.error(f"Fatal error from worker: {state.error_message}")
                        shutdown_event.set()
                        state.aborted = True
                        state.all_done = True
                    elif stage_type == "source":
                        _handle_source_msg(
                            state, event_type, count, progress, tasks, bool(self.stages)
                        )
                    elif stage_type == "stage":
                        _handle_stage_msg(
                            state,
                            stage_idx,
                            event_type,
                            count,
                            progress,
                            tasks,
                            source_done,
                            stage_done_events,
                        )
                    elif stage_type == "sink":
                        _handle_sink_msg(
                            state,
                            event_type,
                            progress,
                            tasks,
                            source_done,
                            stage_done_events,
                            sink_done,
                            shutdown_event,
                            bool(self.stages),
                        )

            except KeyboardInterrupt:
                logger.warning("Keyboard interrupt in monitor, shutting down...")
                shutdown_event.set()
                state.aborted = True
                state.error_message = "Interrupted by user"

        return PipelineResult(
            source_total=state.source_count,
            stage_stats=[
                {
                    "completed": s.completed,
                    "filtered": s.filtered,
                    "total_output": s.total_output,
                }
                for s in state.stage_stats
            ],
            sink_success=state.sink_stats.completed - state.sink_stats.filtered,
            sink_skipped=state.sink_stats.filtered,
            elapsed_time=0,
            aborted=state.aborted,
            error_message=state.error_message,
        )
