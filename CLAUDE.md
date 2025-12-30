# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

flow-control is a PyTorch-based training framework for diffusion and flow models, specifically designed for fine-tuning models like Flux1 and Qwen. It uses diffusers' model layers but implements custom training loops and sampling logic.

## Commands

```bash
# Install dependencies
uv sync

# Install with dev dependencies (linting, type checking)
uv sync --group dev

# Run training (TOML config required)
uv run launch <config.toml>

# Run preprocessing
uv run preprocess <config.toml>

# Linting and formatting
uv run ruff format .
uv run ruff check --fix .

# Type checking (optional, not enforced in pre-commit)
uv run pyright
```

Pre-commit hooks (lefthook) automatically run ruff format and check on commit.

## Architecture

### Plugin Registry Pattern

All major components use a registration pattern with Pydantic validators. When adding new implementations:

- **Adapters** (`flow_control/adapters/`): Register via `model_type` field in `__init__.py` validators
- **Processors** (`flow_control/processors/`): Register via `type` field
- **Samplers** (`flow_control/samplers/`): Register via `type` field
- **Datasets** (`flow_control/datasets/`): Register via `type` field

### Key Abstractions

- `BaseModelAdapter` (`adapters/base.py`): Abstract class for model-specific training logic. Implements `predict_velocity()` for the training step.
- `Processor` (`processors/base.py`): Handles offline data preprocessing (VAE encoding, etc.)
- `Sampler` (`samplers/base.py`): Diffusion/flow sampling methods
- `DataSource/PipelineStage/DataSink` (`utils/pipeline/base.py`): Concurrent data pipeline framework

### Training Backends

Two training modes selected via `launch.mode` in config:
- `accelerate`: Uses HuggingFace Accelerate + DDP (`training/accelerate_ddp.py`)
- `torchrun`: Uses HSDP (Hybrid Sharded Data Parallel) (`training/hsdp.py`)

### Model Loading

`HfModelLoader` (`utils/loaders.py`) handles model loading with:
- Transparent caching
- bitsandbytes 8-bit quantization support
- Layerwise upcasting for mixed precision

## Configuration

Training configs are TOML files with sections:
- `[launch]`: mode (torchrun/accelerate), num_processes, accelerate_config
- `[model]`: adapter configuration
- `[sampler]`: sampling method
- `[processor]`: data preprocessing
- `[dataset]`: training data
- `[optimizer]`/`[scheduler]`: training hyperparameters

## Directory Structure

- `flow_control/adapters/flux1/`, `qwen/`: Model-specific adapter implementations
- `flow_control/utils/pipeline/`: Concurrent multi-stage data processing
- `config/accelerate/`: Pre-configured Accelerate YAML files for multi-GPU

## Code Style

- Python 3.12 required
- Ruff for linting (E, F, UP, B, SIM, I rules)
- Pyright in standard mode for type checking
- `reportPrivateImportUsage` disabled (HuggingFace uses private imports)