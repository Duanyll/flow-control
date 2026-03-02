# Instructions for agents working on the `flow_control` module

This is the `flow_control` module, which provides training utilities for flow-matching Diffusion Transformers (DiTs).

## Package management

Use `uv` for package management. Rules to follow:

1. When adding, removing, or modifying dependencies, never edit `pyproject.toml` directly. Instead, use `uv add`, `uv remove` commands to manage dependencies.
2. To run python scripts in correct environment, use `uv run <script>` or `uv run -m <module>` instead of `python <script>` or `python -m <module>`.

## Type hints and linting

Use `pyright` for type checking, and `ruff` for linting and formatting. To manually run type checking and linting, use the following commands:

```bash
uv run pyright <path_to_check>
uv run ruff format <path_to_format>
uv run ruff check --fix <path_to_check>
```

Rules to follow:

1. Always try to maintain type hints in the code. We use Python 3.12, so prefer using `dict`, `list`, `|`, etc. over `Dict`, `List`, `Union` from `typing`.
2. If you encounter type errors, try to fix them by adding or adjusting type hints. If you have to use `Any` or `# type: ignore`, please add a comment describing why it's necessary and what the expected types are.
3. Try to follow style guidelines enforced by `ruff`. Most style issues can be automatically fixed with first running `uv run ruff format <file>` and then `uv run ruff check --fix <file>`, then manually fix remaining issues if any. Try avoid `# noqa` as much as possible.

## Code structure

1. The main code for the `flow_control` module is located in the `flow_control` directory.
2. Entrypoints: The entrypoint scripts facing users are located in the `flow_control/scripts` directory. Read code here to understand how to use the module, and add new entrypoint scripts here if needed. 
3. Testing and examples: we do not maintain traditional unit tests. Any code outside `flow_control/scripts` may contain a simple `if __name__ == "__main__":` block for quick testing and debugging. More complicated examples and usage demonstrations (e.g., requires multiple mock classes or multiprocessing) of individual modules should be placed in the `examples` directory at the root of the repository.

## Coding conventions

### Pydantic patterns

For anything ouside `flow_control.utils`, we prefer: **Pass data through function arguments, and store configuration in Pydantic models.**

1. We extend `pydantic.BaseModel` to define classes for configuration, avoid spread untyped kwargs and dicts in the code.
2. Behaviours are preferred to be methods of the config class. This means we can override the behaviour by extending the super class, making the code more modular. 
   - See `flow_control.training.weighting` and `flow_control.utils.types` for examples of this pattern.
   - `HsdpEngine` in `flow_control.training.hsdp_engine` is an exception, since it has to hold many non-pydantic states (e.g., dataloaders, optimizers, etc.), it has a separated `HsdpEngineConfig` class for configuration, and the engine itself is a `Stateful` class that holds all the states and behaviours.
3. Read related existing code if you want to add a new feature, and try to follow the existing patterns. 

### Logging and printing

Rules to follow:

1. Avoid using `print()` directly in the code. Instead, do the following:
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
2. Prefer to pass images as `torch.Tensor` in `BCHW` format and in `[0, 1]` range. To support images in different resolutions, we use `list` of `1CHW` tensors. You may find `pil_to_tensor` and `tensor_to_pil` functions in `flow_control.utils.image` useful.
3. If it brings extra complexity to support batch size > 1, just support batch size of 1. 