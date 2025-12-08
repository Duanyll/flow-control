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
  - Import and call setup_global_handler as early as possible in the child process.
- Child processes launched by accelerate:
  - Due to limitations, just silence all logs to avoid clutter.
"""

import logging
import os

import diffusers
import transformers
import torch.multiprocessing as mp
from accelerate.state import PartialState
from rich.console import Console
from rich.logging import RichHandler

def get_process_type():
    """
    Detect if current process is:
    - 'main': Normal main process import
    - 'accelerate_child': Subprocess launched by accelerate
    - 'mp_spawn_child': Subprocess spawned by multiprocessing
    """
    state = PartialState()
    if not state.is_local_main_process:
        return 'accelerate_child'
    
    current = mp.current_process()
    if current.name == 'MainProcess' and mp.parent_process() is None:  # Python 3.8+ for parent_process
        return 'main'
    
    # For older Python, fallback to name and authkey
    if current.name == 'MainProcess' and current.authkey == b'\x00' * 32:
        return 'main'
    
    return 'mp_spawn_child'

process_type = get_process_type()

console = Console(quiet=process_type != 'main')
log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
rich_handler = RichHandler(console=console, rich_tracebacks=True)

transformers.utils.logging.disable_default_handler()
transformers.utils.logging.disable_progress_bar()
transformers.utils.logging.set_verbosity_warning()

diffusers.utils.logging.disable_default_handler()
diffusers.utils.logging.disable_progress_bar()
diffusers.utils.logging.set_verbosity_warning()

def setup_global_handler(handler):
    transformers.utils.logging.add_handler(handler)
    diffusers.utils.logging.add_handler(handler)
    logging.basicConfig(
        level=log_level, 
        handlers=[handler],
        format="%(name)s | %(message)s",
    )

if process_type != 'mp_spawn_child':
    setup_global_handler(rich_handler)

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

__all__ = ["get_logger", "setup_global_handler", "console", "rich_handler"]