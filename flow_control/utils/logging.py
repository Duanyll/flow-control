"""
IMPORTANT for other modules:
1. Always use get_logger from this module to get loggers. Never call plain `print`.
2. Explicitly pass `console` from this module to Rich progress bars and other Rich components.

Logging behavior:

- Single process or main process in multiprocessing or accelerate:
  - Use RichHandler for pretty console output.
  - Set up global logging handlers for transformers and diffusers.
- Child processes spawned by multiprocessing:
  - Should call setup_global_handler with a QueueHandler to route logs to the main process.
  - Import and call setup_global_handler with include_name=False as early as possible in the child process.
    Pass a QueueHandler connected to the main process's listener.
- Child processes launched by accelerate or torchrun:
  - Due to limitations, just silence all logs to avoid clutter.
"""

import contextlib
import logging
import os
import subprocess
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import diffusers
import torch
import torch.multiprocessing as mp
import transformers
from accelerate.state import PartialState
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

from .describe import describe


def get_process_type():
    """
    Detect if current process is:
    - 'main': Normal main process import
    - 'mpi_child': Subprocess launched by accelerate
    - 'mp_spawn_child': Subprocess spawned by multiprocessing
    - 'ray_worker': Ray Data worker (set via FLOW_CONTROL_RAY_WORKER env var)
    """
    # Check for Ray worker env var before PartialState() which can have side effects
    if os.getenv("FLOW_CONTROL_RAY_WORKER"):
        return "ray_worker"

    # Check for LOCAL_RANK environment variable set by torchrun
    if os.getenv("LOCAL_RANK") is not None:
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        return "main" if local_rank == 0 else "mpi_child"

    state = PartialState()
    if not state.is_local_main_process:
        return "mpi_child"

    current = mp.current_process()
    if (
        current.name == "MainProcess" and mp.parent_process() is None
    ):  # Python 3.8+ for parent_process
        return "main"

    # For older Python, fallback to name and authkey
    if current.name == "MainProcess" and current.authkey == b"\x00" * 32:
        return "main"

    return "mp_spawn_child"


def setup_global_handler(handler, include_name: bool = True):
    transformers.utils.logging.add_handler(handler)
    diffusers.utils.logging.add_handler(handler)
    logging.basicConfig(
        level=global_log_level,
        handlers=[handler],
        format="<%(name)s> %(message)s" if include_name else "%(message)s",
    )
    package_logger = logging.getLogger("flow_control")
    package_logger.setLevel(log_level)


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
def _warn_once(logger: logging.Logger, message: str):
    # lru_cache prevents pyright from checking input types strictly, so we add explicit
    # type hints by wrapping this function.
    logger.warning(message)


def warn_once(logger: logging.Logger, message: str):
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


BANNER = r"""
  _____ _                  ____            _             _ 
 |  ___| | _____      __  / ___|___  _ __ | |_ _ __ ___ | |
 | |_  | |/ _ \ \ /\ / / | |   / _ \| '_ \| __| '__/ _ \| |
 |  _| | | (_) \ V  V /  | |__| (_) | | | | |_| | | (_) | |
 |_|   |_|\___/ \_/\_/    \____\___/|_| |_|\__|_|  \___/|_|
"""


def _print_banner():
    console.print(
        Panel.fit(BANNER, subtitle=f"[bold green]{get_version()}[/bold green]")
    )


@contextlib.contextmanager
def dump_if_failed(logger: logging.Logger, obj: Any, save: bool = False):
    try:
        yield
    except Exception as e:
        logger.debug(
            f"Dumping the {type(obj).__name__} that caused {type(e).__name__}..."
        )
        logger.debug(describe(obj, console=False), {"markup": True})
        if save:
            rank = os.getenv("RANK", "0")
            path = f"dump/debug_dump_{rank}.pt"
            torch.save(obj, path)
            logger.info(
                f"Saved the {type(obj).__name__} that caused {type(e).__name__} to {path}"
            )
        raise


process_type = get_process_type()
enable_console = process_type == "main"

console = Console(quiet=not enable_console)
log_level: int = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
global_log_level: int = getattr(
    logging, os.getenv("GLOBAL_LOG_LEVEL", "WARNING").upper(), logging.WARNING
)
rich_handler = RichHandler(console=console, rich_tracebacks=True)

transformers.utils.logging.disable_default_handler()
transformers.utils.logging.disable_progress_bar()
transformers.utils.logging.set_verbosity(global_log_level)

diffusers.utils.logging.disable_default_handler()
diffusers.utils.logging.disable_progress_bar()
diffusers.utils.logging.set_verbosity(global_log_level)

if process_type == "ray_worker":
    _basic_handler = logging.StreamHandler()
    _basic_handler.setFormatter(
        logging.Formatter("%(levelname)s - %(name)s - %(message)s")
    )
    setup_global_handler(_basic_handler)
elif process_type != "mp_spawn_child":
    setup_global_handler(rich_handler)

logging.captureWarnings(True)

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)  # Httpx info logs are too verbose

if enable_console:
    _print_banner()


__all__ = [
    "get_logger",
    "setup_global_handler",
    "console",
    "rich_handler",
    "get_version",
]
