"""Top-level configuration types for schema generation.

This module re-exports LaunchConfig and HsdpEngineConfig so that they can be
imported without pulling in torch-heavy training modules, and provides a
single place to generate JSON schemas for the full configuration files.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class LaunchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["sft", "grpo", "inference"]
    devices: int | list[int]
    generate_dcp_seed: bool = False
    preprocess_config: str | None = None
    env: dict[str, str] = {}


class HsdpEngineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    launch: LaunchConfig | None = None
    seed: int = 42
    hsdp_shard_dim: int = 1
    gradient_checkpointing: bool = True
    async_save: bool = False
    """
    Whether to use `dcp.async_save` to save checkpoints. This is still experimental and
    may cause race conditions when using collectives. Use with caution.
    """
