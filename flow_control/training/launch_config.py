"""
LaunchConfig stays in a separate file so that launch.py does not need to import torch
and other heavy dependencies from training, making it faster to parse the config and
avoid os.execvp issues.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class LaunchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["sft", "grpo", "nft", "inference"]
    devices: int | list[int]
    generate_dcp_seed: bool = False
    preprocess_config: str | None = None
    env: dict[str, str] = {}
