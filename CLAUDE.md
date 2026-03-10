# Instructions for agents working on the `flow_control` module

This is the `flow_control` module, which provides training utilities for flow-matching Diffusion Transformers (DiTs).

## Package management

Use `uv` for package management. Rules to follow:

1. Use `uv add`, `uv remove` commands to manage dependencies, never edit `pyproject.toml` directly.
2. To run python scripts, use `uv run <script>` or `uv run -m <module>`, never directly call `python <script>` or `python -m <module>`.

## Type hints and linting

Use `pyright` for type checking, and `ruff` for linting and formatting, with commands:

```bash
uv run pyright <path_to_check>
uv run ruff format <path_to_format>
uv run ruff check --fix <path_to_check>
```

Rules to follow:

1. Always try to maintain type hints in the code. We use Python 3.12, so prefer using `dict`, `list`, `|`, etc. over `Dict`, `List`, `Union` from `typing`.
2. Try to use `TypedDict` for dicts with fixed structure, instead of using `dict[str, Any]`.
3. Use Ruff to check and fix style issues by first running `uv run ruff format <file>` and then `uv run ruff check --fix <file>`, then manually fix remaining issues if any. Try avoid `# noqa` as much as possible.
4. The automated diagnostics from IDE after you edit the code may be stale or incorrect. Call above CLI commands to get accurate diagnostics.

## Code structure

1. The main code for the `flow_control` module is located in the `flow_control` directory.
2. The entrypoint scripts facing users are located in the `flow_control/scripts` directory.

## Coding conventions

### Pydantic patterns

We prefer to: **Pass data through function arguments, and store configuration in Pydantic models.**

1. We extend `pydantic.BaseModel` to define classes for configuration, avoid spread untyped kwargs and dicts in the code.
2. Behaviours are preferred to be methods of the config class. This means we can override the behaviour by extending the super class, making the code more modular. 

### Logging and printing

Rules to follow:

1. Avoid using `print()` directly in library code. Instead, do the following:
   ```python
   from flow_control.utils.logging import get_logger, console
   logger = get_logger(__name__)
   logger.info("This is an info message")             # logger is a `logging.Logger` instance 
   console.print("This is a message to the console")  # console is a `rich.console.Console` instance
   ```
   To print something in `if __name__ == "__main__":` block for quick testing and debugging, use `from rich import print` ONLY in that block.
2. We use `rich` for logging and printing. Import `get_logger` for logging messages, and explicitly pass the `console` instance for any rich printing (especially for progress bars) in the code. 
3. `flow_control.utils.logging` correctly works with multiprocessing, so you can safely use the logger and console in any subprocess as in the main process, without worrying about messing up the terminal output.

### Batch processing and Tensor operations

1. Prefer `einops` for tensor operations, instead of directly using `view()`, `reshape()`, `permute()`, etc. 
2. Prefer to pass images as `torch.Tensor` in `BCHW` format and in `[0, 1]` range. You may find `pil_to_tensor` and `tensor_to_pil` functions in `flow_control.utils.common` useful.
3. If it brings extra complexity to support batch size > 1, just support batch size of 1. 

## Workflow

1. Read related existing code if you want to add a new feature, and try to follow the existing patterns. 
2. Write self-contained, minimal test code in `if __name__ == "__main__":` block to verify your code works as expected, and to provide usage examples. 
3. Make sure you fix all linting and type errors.
4. You don't have to run tests that requires actual weights / datasets, unless the user instructs you to do so. 