# Instructions for agents working on the `flow_control` module

This is the `flow_control` module, which provides training utilities for flow-matching based Diffusion Transformers (DiTs).

## Package management

This repository uses `uv` for package management. Rules to follow:

1. When adding, removing, or modifying dependencies, never edit `pyproject.toml` directly. Instead, use `uv add`, `uv remove` commands to manage dependencies.
2. To run python scripts in correct environment, use `uv run <script>` or `uv run -m <module>` instead of `python <script>` or `python -m <module>`.

## Type hints and linting

We use `pyright` for type checking, and `ruff` for linting and formatting. VSCode intergration is configured to automatically provide type checking and linting feedback to the agent after editing the code. To manually run type checking and linting, use the following commands:

```bash
uv run pyright <path_to_check>
uv run ruff format <path_to_format>
uv run ruff check --fix <path_to_check>
```

Rules to follow:

1. Always try to maintain type hints in the code. We use Python 3.12, so prefer using `dict`, `list`, `|`, etc. over `Dict`, `List`, `Union` from `typing`.
2. If you encounter type errors, try to fix them by adding or adjusting type hints. If you have to use `Any` or `# type: ignore`, please add a comment describing why it's necessary and what the expected types are.
3. Try to follow style guidelines enforced by `ruff`. Most style issues can be automatically fixed with first running `uv run ruff format <file>` and then `uv run ruff check --fix <file>`. If there are any remaining issues that cannot be automatically fixed, please address them manually. If you have to use `# noqa`, ask the user before doing so.

## Code structure

1. The main code for the `flow_control` module is located in the `flow_control` directory.
2. Entrypoints: The entrypoint scripts facing users are located in the `flow_control/scripts` directory. All modules in this directory export a `main()` function as the main entrypoint, defined in `[project.scripts]` in `pyproject.toml`.
   - Almost every option for these scripts should be loaded from a TOML file, avoid CLI arguments and environment variables as much as possible.
   - The only exception is the `LOG_LEVEL` environment variable, which can be used to override logging level for debugging purposes. Default is `INFO`.
3. Testing and examples: Due to the nature of machine learning code, we do not maintain traditional unit tests. Any code outside `flow_control/scripts` may contain a simple `if __name__ == "__main__":` block for quick testing and debugging. More complicated examples and usage demonstrations (e.g., requires multiple mock classes or multiprocessing) of individual modules should be placed in the `examples` directory at the root of the repository.

## Coding conventions

### Code quality

Rules to follow:

1. AVOID excessive code nesting (e.g., nested loops, nested conditionals, etc.) and long functions (e.g., > 100 lines). A McCabe complexity check with a threshold of 10 is in place, please listen to it. 

### Pydantic patterns

For anything ouside `flow_control.utils`, we prefer: **Pass data through function arguments, and store configuration in Pydantic models.**

1. We extend `pydantic.BaseModel` to define classes for configuration, Pydantic will help us load all the configuration from a TOML file to the closest place it is needed, without having to spread kwargs across the codebase.
2. Behaviours are preferred to be methods of the config class. This means we can override the behaviour by extending the super class, making the code more modular. Most modern DiT checkpoints have variants for different subtasks, this pattern allows us to clearly show the differences and commons between different variants by putting them in different classes, and using inheritance to reuse the common code.
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
   A ruff rule is in place to forbid using `print()` directly. If you hope to print something in `if __name__ == "__main__":` block for quick testing and debugging, use `from rich import print` ONLY in that block.
2. We use `rich` for logging and printing, which provides better formatting and color support. Use `logger` for logging messages, and explicitly pass the `console` instance for any rich printing (especially for progress bars) in the code. 
3. `flow_control.utils.logging` correctly works with multiprocessing, so you can safely use the logger and console in any subprocess as in the main process, without worrying about messing up the terminal output.

### Batch processing and Tensor operations

In `flow_control`: **Only need to support batch size of 1.**

Why: Modern DiTs are very large in terms of memory footprint, and deal with sequences of thousands of tokens. Batch size of 1 can already saturate the GPU. Multimodal data records may vary significantly in shape and size (e.g., different image resolutions, different length of text, etc.), supporting batch size > 1 would require complex padding and batching logic, which is not worth the effort given the current state of hardware and models.

Rules to follow:

1. You can assume batch size of 1 in the code, and no need to add extra logic to support batch size > 1. However, if you find it easier to write the code in a way that supports batch size > 1, you can do so. For example, the image reizing logic in `flow_control.utils.resize` can handle batch size > 1 with no extra effort, so it's fine to keep that logic in place. 
   - However, most Tensors will have a dummy batch dimension, don't `squeeze()` it away, just keep it as is, since most torch operators and modules will expect a batch dimension. For example, an image tensor should be in `1CHW` format, instead of `CHW` format, even if the batch size is 1.
2. We prefer `einops` for tensor operations, instead of directly using `view()`, `reshape()`, `permute()`, etc. 
3. We prefer to pass images as `torch.Tensor` in `BCHW` format and in `[0, 1]` range. To support images in different resolutions, we use `list` of `1CHW` tensors. You may find `pil_to_tensor` and `tensor_to_pil` functions in `flow_control.utils.image` useful.