"""
Worker functions for pipeline framework.
"""

import asyncio
import contextlib
import inspect
import queue
import time
from logging.handlers import QueueHandler
from multiprocessing.synchronize import Event as EventType
from typing import Any

import torch
import torch.multiprocessing as mp

from ..logging import get_logger, setup_global_handler
from .base import DataSink, DataSource, PipelineStage


def _setup_worker_logging(name: str, log_queue: mp.Queue):
    """Set up logging for a worker process."""
    handler = QueueHandler(log_queue)
    setup_global_handler(handler, include_name=False)
    return get_logger(name)


def _safe_put(
    q: mp.Queue,
    item: Any,
    timeout: float = 1.0,
    logger=None,
    shutdown_event: EventType | None = None,
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
    shutdown_event: EventType | None = None,
) -> tuple[bool, Any]:
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
    source_class: type[DataSource],
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
        with contextlib.suppress(queue.Full):
            progress_queue.put(("fatal", 0, "error", str(e)), timeout=5)
    finally:
        worker_logger.info("Source worker finished")


def _stage_worker(
    stage_index: int,
    worker_id: int,
    stage_class: type[PipelineStage],
    input_queue: Any,
    output_queue: Any,
    progress_queue: Any,
    log_queue: Any,
    upstream_done: EventType,
    shutdown_event: EventType,
    num_threads: int,
    device: int | None,
    init_kwargs: dict,
    max_concurrency: int = 1,
):
    """Worker that processes items through a pipeline stage."""
    torch.set_num_threads(num_threads)
    worker_name = f"Stage{stage_index}-W{worker_id}"
    worker_logger = _setup_worker_logging(worker_name, log_queue)
    worker_logger.info(f"Stage worker started (device={device}, threads={num_threads})")
    stage = None

    try:
        # Instantiate the stage in the worker process
        stage = stage_class(worker_id, device=device, **init_kwargs)

        # Check if process is async
        is_async = inspect.iscoroutinefunction(stage.process)

        if is_async:
            worker_logger.info(
                f"Using async mode with max_concurrency={max_concurrency}"
            )
            asyncio.run(
                _async_stage_loop(
                    stage,
                    stage_index,
                    input_queue,
                    output_queue,
                    progress_queue,
                    upstream_done,
                    shutdown_event,
                    max_concurrency,
                    worker_logger,
                )
            )
        else:
            _sync_stage_loop(
                stage,
                stage_index,
                input_queue,
                output_queue,
                progress_queue,
                upstream_done,
                shutdown_event,
                worker_logger,
            )

    except Exception as e:
        # Fatal error (e.g., in __init__) - notify main process
        worker_logger.error(f"Stage worker fatal error: {e}", exc_info=True)
        with contextlib.suppress(queue.Full):
            progress_queue.put(
                (
                    "fatal",
                    stage_index,
                    "error",
                    f"Stage{stage_index}-W{worker_id}: {e}",
                ),
                timeout=5,
            )
    finally:
        if stage is not None:
            stage.cleanup()
        worker_logger.info("Stage worker finished")


def _sync_stage_loop(
    stage: PipelineStage,
    stage_index: int,
    input_queue: Any,
    output_queue: Any,
    progress_queue: Any,
    upstream_done: EventType,
    shutdown_event: EventType,
    worker_logger,
):
    """Synchronous processing loop for non-async stages."""
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
            _safe_put(progress_queue, ("stage", stage_index, "produced", output_count))


async def _async_stage_loop(
    stage: PipelineStage,
    stage_index: int,
    input_queue: Any,
    output_queue: Any,
    progress_queue: Any,
    upstream_done: EventType,
    shutdown_event: EventType,
    max_concurrency: int,
    worker_logger,
):
    """Asynchronous processing loop for async stages with concurrency control.

    IMPORTANT: To maintain proper backpressure, we must limit how many items
    we pull from the input queue. The semaphore must guard queue access, not
    just the process() call. Otherwise, items accumulate in pending_tasks
    instead of staying in the queue, bypassing backpressure.
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    pending_tasks: set[asyncio.Task] = set()
    # Track active slots to properly manage backpressure
    active_slots = 0
    active_slots_lock = asyncio.Lock()

    async def process_item(item):
        """Process a single item. Semaphore is already acquired before calling."""
        nonlocal active_slots
        try:
            if shutdown_event.is_set():
                return
            try:
                results = await stage.process(item)  # type: ignore[misc]
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
                worker_logger.error(f"Error processing item: {e}", exc_info=True)
                _safe_put(progress_queue, ("stage", stage_index, "filtered", 1))
        finally:
            # Release the semaphore after processing is complete
            semaphore.release()
            async with active_slots_lock:
                active_slots -= 1

    work_done = False
    while not shutdown_event.is_set():
        # First, try to acquire a semaphore slot before pulling from queue
        # This ensures we only pull items we can actually process
        try:
            # Use wait_for with a short timeout to allow checking shutdown
            await asyncio.wait_for(semaphore.acquire(), timeout=0.1)
        except TimeoutError:
            # Couldn't acquire semaphore, check conditions and retry
            if shutdown_event.is_set():
                worker_logger.info("Stage worker received shutdown signal")
                break
            # Check if we're done (no pending work and upstream finished)
            async with active_slots_lock:
                no_active = active_slots == 0
            if upstream_done.is_set() and input_queue.empty() and no_active:
                if not work_done:
                    worker_logger.info("Work completed, waiting for pipeline shutdown")
                    work_done = True
                await asyncio.sleep(0.1)
            continue

        # We have a semaphore slot, now try to get an item
        success, item = _safe_get(
            input_queue, timeout=0.1, shutdown_event=shutdown_event
        )

        if success:
            # Track active slot and create task
            async with active_slots_lock:
                active_slots += 1
            task = asyncio.create_task(process_item(item))
            pending_tasks.add(task)
            task.add_done_callback(pending_tasks.discard)
        else:
            # No item available, release the semaphore we acquired
            semaphore.release()

            if shutdown_event.is_set():
                worker_logger.info("Stage worker received shutdown signal")
                break

            # Check if all work is done
            async with active_slots_lock:
                no_active = active_slots == 0
            if upstream_done.is_set() and input_queue.empty() and no_active:
                if not work_done:
                    worker_logger.info("Work completed, waiting for pipeline shutdown")
                    work_done = True
                await asyncio.sleep(0.1)
            else:
                # Brief sleep to avoid busy loop when queue is temporarily empty
                await asyncio.sleep(0.01)

    # Wait for all pending tasks to complete on shutdown
    if pending_tasks:
        worker_logger.info(
            f"Waiting for {len(pending_tasks)} pending tasks to complete"
        )
        await asyncio.gather(*pending_tasks, return_exceptions=True)


def _sink_worker(
    worker_id: int,
    sink_class: type[DataSink],
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
    sink = None

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
                        worker_logger.info(
                            "Work completed, waiting for pipeline shutdown"
                        )
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
        with contextlib.suppress(queue.Full):
            progress_queue.put(
                ("fatal", 0, "error", f"Sink-W{worker_id}: {e}"),
                timeout=5,
            )
    finally:
        if sink is not None:
            sink.cleanup()
        worker_logger.info("Sink worker finished")
