"""
IMPORTANT for other modules:
1. Always use get_logger from this module to get loggers. Never call plain `print`.
2. Explicitly pass `console` from this module to Rich progress bars and other Rich
   components.

Logging behavior:

- Single process or distributed local rank 0:
  - Use RichHandler for pretty console output.
  - Save logs to ``/tmp/flow-control/rankXXXX.log``.
- Non-zero distributed ranks:
  - Silence the console and save logs to that rank's log file.
- Multiprocessing workers spawned inside a rank:
  - Reconfigure themselves with ``setup_global_handler(QueueHandler(...))`` so their
    logs flow back to the parent process.
- Every rank keeps uncaught Python tracebacks in
  ``/tmp/flow-control/rankXXXX.traceback.log``.
- Ray workers (``FLOW_CONTROL_RAY_WORKER`` set):
  - Skip all logging customization; Ray handles logging itself.
"""

from __future__ import annotations

import contextlib
import faulthandler
import logging
import os
import signal
import subprocess
import sys
import threading
import traceback
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from logging.handlers import QueueHandler
from pathlib import Path
from types import TracebackType
from typing import Any

import diffusers
import torch
import torch.multiprocessing as mp
import transformers
from rich import print
from rich.console import Console
from rich.logging import RichHandler

from .describe import describe

LOG_DIR = Path(os.getenv("LOG_DIR", "/tmp/flow-control"))
LOG_FILE_TEMPLATE = "rank{rank:04d}.log"
TRACEBACK_FILE_TEMPLATE = "rank{rank:04d}.traceback.log"
TRACEBACK_SEPARATOR = "=" * 80


@dataclass(frozen=True, slots=True)
class ProcessContext:
    process_type: str
    rank: int
    local_rank: int
    pid: int

    @property
    def console_enabled(self) -> bool:
        return self.process_type == "main"

    @property
    def has_default_handlers(self) -> bool:
        return self.process_type != "mp_spawn_child"

    @property
    def log_path(self) -> Path | None:
        if self.process_type in ("mp_spawn_child", "ray_worker"):
            return None
        return LOG_DIR / LOG_FILE_TEMPLATE.format(rank=self.rank)

    @property
    def traceback_path(self) -> Path:
        return LOG_DIR / TRACEBACK_FILE_TEMPLATE.format(rank=self.rank)

    @property
    def resets_traceback_file(self) -> bool:
        return self.process_type != "mp_spawn_child"


def _get_env_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None:
        return None
    with contextlib.suppress(ValueError):
        return int(value)
    return None


def _build_process_context() -> ProcessContext:
    if os.getenv("FLOW_CONTROL_RAY_WORKER"):
        return ProcessContext("ray_worker", 0, 0, os.getpid())

    rank = _get_env_int("RANK") or 0
    local_rank = _get_env_int("LOCAL_RANK") or 0

    if local_rank > 0:
        return ProcessContext("mpi_child", rank or local_rank, local_rank, os.getpid())

    current = mp.current_process()
    if current.name == "MainProcess" and mp.parent_process() is None:
        return ProcessContext("main", rank, local_rank, os.getpid())

    if current.name == "MainProcess" and current.authkey == b"\x00" * 32:
        return ProcessContext("main", rank, local_rank, os.getpid())

    return ProcessContext("mp_spawn_child", rank, local_rank, os.getpid())


def get_process_type() -> str:
    return process_context.process_type


def _configure_record_factory() -> None:
    previous_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = previous_factory(*args, **kwargs)
        record.flow_rank = process_context.rank
        record.flow_local_rank = process_context.local_rank
        record.flow_process_type = process_context.process_type
        return record

    logging.setLogRecordFactory(record_factory)


def _ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _reset_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _reset_traceback_file() -> None:
    if process_context.resets_traceback_file and traceback_file_path is not None:
        _reset_file(traceback_file_path)


def _create_message_formatter(include_name: bool) -> logging.Formatter:
    if include_name:
        return logging.Formatter("<%(name)s> %(message)s")
    return logging.Formatter("%(message)s")


def _create_file_handler(path: Path) -> logging.FileHandler:
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            (
                "%(asctime)s | %(levelname)s | rank=%(flow_rank)s | "
                "local_rank=%(flow_local_rank)s | %(flow_process_type)s | "
                "pid=%(process)d | %(name)s | %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    return handler


def _apply_handler_formatter(
    handler: logging.Handler, include_name: bool = True
) -> logging.Handler:
    if isinstance(handler, logging.FileHandler):
        return handler
    if isinstance(handler, (QueueHandler, RichHandler)):
        handler.setFormatter(_create_message_formatter(include_name))
        return handler
    if isinstance(handler, logging.StreamHandler):
        return handler
    handler.setFormatter(_create_message_formatter(include_name))
    return handler


def _replace_handlers(logger: logging.Logger, handlers: list[logging.Handler]) -> None:
    for existing in list(logger.handlers):
        logger.removeHandler(existing)
    for handler in handlers:
        logger.addHandler(handler)


def _configure_library_logger(
    name: str,
    level: int,
    handlers: list[logging.Handler],
) -> None:
    library_logger = logging.getLogger(name)
    library_logger.setLevel(level)
    _replace_handlers(library_logger, handlers)


def _configure_root_logger(handlers: list[logging.Handler]) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(global_log_level)
    _replace_handlers(root_logger, handlers)
    logging.getLogger("flow_control").setLevel(log_level)


def _configure_third_party_logging(handlers: list[logging.Handler]) -> None:
    transformers.utils.logging.disable_default_handler()
    transformers.utils.logging.disable_progress_bar()
    transformers.utils.logging.set_verbosity(global_log_level)

    diffusers.utils.logging.disable_default_handler()
    diffusers.utils.logging.disable_progress_bar()
    diffusers.utils.logging.set_verbosity(global_log_level)

    _configure_library_logger("transformers", global_log_level, handlers)
    _configure_library_logger("diffusers", global_log_level, handlers)


def _configure_logging(handlers: list[logging.Handler]) -> None:
    _configure_root_logger(handlers)
    _configure_third_party_logging(handlers)
    logging.captureWarnings(True)

    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.WARNING)


def _build_default_handlers() -> list[logging.Handler]:
    if not process_context.has_default_handlers:
        return []

    handlers: list[logging.Handler] = []
    if process_context.console_enabled and _rich_handler is not None:
        handlers.append(_rich_handler)

    if log_file_path is not None:
        handlers.append(_create_file_handler(log_file_path))

    return handlers


def _write_traceback_file(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_traceback: TracebackType | None,
    source: str,
) -> None:
    header = (
        f"{TRACEBACK_SEPARATOR}\n"
        f"source={source} pid={os.getpid()} rank={process_context.rank} "
        f"local_rank={process_context.local_rank} "
        f"process_type={process_context.process_type}\n"
    )
    if traceback_file_path is None:
        return
    with traceback_file_path.open("a", encoding="utf-8") as handle:
        handle.write(header)
        handle.writelines(
            traceback.format_exception(exc_type, exc_value, exc_traceback)
        )
        handle.write(f"{TRACEBACK_SEPARATOR}\n")
        print(
            f"[red]>>> Got {exc_type.__name__} and logged to {traceback_file_path} <<<[/red]"
        )


def _install_exception_hooks() -> None:
    def sys_hook(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            return
        with contextlib.suppress(Exception):
            _write_traceback_file(
                exc_type,
                exc_value,
                exc_traceback,
                source="sys.excepthook",
            )
        with contextlib.suppress(Exception):
            logging.getLogger("flow_control.traceback").critical(
                "Uncaught exception",
                exc_info=(exc_type, exc_value, exc_traceback),
            )

    def thread_hook(args: threading.ExceptHookArgs) -> None:
        thread_name = args.thread.name if args.thread is not None else "unknown"
        if issubclass(args.exc_type, KeyboardInterrupt):
            return
        if args.exc_value is None:
            return
        with contextlib.suppress(Exception):
            _write_traceback_file(
                args.exc_type,
                args.exc_value,
                args.exc_traceback,
                source=f"thread:{thread_name}",
            )
        with contextlib.suppress(Exception):
            logging.getLogger("flow_control.traceback").critical(
                f"Uncaught exception in thread {thread_name}",
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )

    sys.excepthook = sys_hook
    threading.excepthook = thread_hook


_faulthandler_file: Any = None  # kept alive so faulthandler can write to it


def _install_faulthandler() -> None:
    """Register faulthandler to dump all-thread tracebacks on SIGQUIT.

    When a training process is stuck in a C-level call (NCCL, CUDA, etc.),
    SIGINT cannot interrupt it.  ``torchrun`` forwards SIGQUIT to workers,
    so ``kill -QUIT <pid>`` (or ``kill -3``) will dump every thread's Python
    stack into the traceback log file.  The process keeps running so you can
    send the signal repeatedly.
    """
    global _faulthandler_file
    if traceback_file_path is None:
        return
    if not hasattr(faulthandler, "register"):  # Unix only
        return
    _faulthandler_file = traceback_file_path.open("a")
    faulthandler.register(
        signal.SIGQUIT,
        file=_faulthandler_file,
        all_threads=False,
        chain=False,  # don't core-dump / terminate after dumping
    )


def setup_global_handler(handler: logging.Handler, include_name: bool = True) -> None:
    configured_handler = _apply_handler_formatter(handler, include_name=include_name)
    _configure_logging([configured_handler])


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: The configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    return logger


@lru_cache(None)
def _warn_once(logger: logging.Logger, message: str) -> None:
    logger.warning(message)


def warn_once(logger: logging.Logger, message: str) -> None:
    """Log a warning message only once per unique message.

    Args:
        logger (logging.Logger): The logger to use.
        message (str): The warning message to log.
    """
    _warn_once(logger, message)


_version_cache: str | None = None


def get_version() -> str:
    """Get the current version of the flow_control package.

    Returns:
        str: The version string.
    """
    global _version_cache
    if _version_cache is not None:
        return _version_cache

    result = "unknown"
    with contextlib.suppress(PackageNotFoundError):
        result = version("flow_control")
    with contextlib.suppress(Exception):
        git_commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("ascii")
            .strip()
        )
        result += f"+git.{git_commit}"

        has_changes = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            ).strip()
            != b""
        )
        if has_changes:
            result += ".wip"
    _version_cache = result
    return result


@contextlib.contextmanager
def dump_if_failed(logger: logging.Logger, obj: Any, save: bool = False):
    try:
        yield
    except Exception as e:
        logger.debug(
            f"Dumping the {type(obj).__name__} that caused {type(e).__name__}..."
        )
        logger.debug(describe(obj, console=False), extra={"markup": True})
        if save:
            rank = os.getenv("RANK", "0")
            path = f"dump/debug_dump_{rank}.pt"
            torch.save(obj, path)
            logger.info(
                f"Saved the {type(obj).__name__} that caused {type(e).__name__} to {path}"
            )
        raise


process_context = _build_process_context()
process_type = process_context.process_type
log_level: int = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
global_log_level: int = getattr(
    logging, os.getenv("GLOBAL_LOG_LEVEL", "WARNING").upper(), logging.WARNING
)

if process_context.process_type == "ray_worker":
    # Ray has its own logging mechanism; skip all customization.
    console = Console(quiet=True)
    _rich_handler = None
    log_file_path = None
    traceback_file_path = None
    queue_listener_handlers: tuple[logging.Handler, ...] = ()
else:
    _ensure_log_dir()
    log_file_path = process_context.log_path
    traceback_file_path = process_context.traceback_path
    _reset_traceback_file()
    _configure_record_factory()

    console = Console(quiet=not process_context.console_enabled)
    _rich_handler = RichHandler(
        console=console, rich_tracebacks=True, enable_link_path=False
    )

    default_handlers = [
        _apply_handler_formatter(handler) for handler in _build_default_handlers()
    ]
    queue_listener_handlers = tuple(default_handlers)
    if default_handlers:
        _configure_logging(default_handlers)

    _install_exception_hooks()
    _install_faulthandler()

__all__ = [
    "console",
    "get_logger",
    "get_process_type",
    "get_version",
    "log_file_path",
    "process_context",
    "queue_listener_handlers",
    "setup_global_handler",
    "traceback_file_path",
    "warn_once",
]


if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.info("flow_control logging smoke test")
    print(f"process_type: {process_type}")
    print(f"log file: {log_file_path}")
    print(f"traceback file: {traceback_file_path}")
