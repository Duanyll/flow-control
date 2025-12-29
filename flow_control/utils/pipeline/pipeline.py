"""
Main Pipeline class for multi-stage data processing.
"""

import queue
import time
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
                            stage_cfg.max_concurrency,
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
        stage_done_events: list[EventType],
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

            stage_tasks: list[TaskID] = []
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
                                progress.update(stage_tasks[0], total=source_total)
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
                            delta += (
                                stage_stats[i].total_output - stage_stats[i].completed
                            )
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
