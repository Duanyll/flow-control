import logging
import os

import diffusers
import transformers
from accelerate.state import PartialState
from rich import Console
from rich.logging import RichHandler

_state = PartialState()

console = Console(quiet=not _state.is_local_main_process)
log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
rich_handler = RichHandler(console=console, rich_tracebacks=True)

transformers.utils.logging.disable_default_handler()
transformers.utils.logging.disable_progress_bar()
transformers.utils.logging.add_handler(rich_handler)
transformers.utils.logging.set_verbosity_warning()

diffusers.utils.logging.disable_default_handler()
diffusers.utils.logging.disable_progress_bar()
diffusers.utils.logging.add_handler(rich_handler)
diffusers.utils.logging.set_verbosity_warning()

logging.basicConfig(level=log_level, handlers=[rich_handler])

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

