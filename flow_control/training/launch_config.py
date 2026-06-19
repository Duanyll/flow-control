"""
LaunchConfig stays in a separate file so that launch.py does not need to import torch
and other heavy dependencies from training, making it faster to parse the config and
avoid os.execvp issues.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class LaunchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Open: any tag registered in ``trainer_registry`` (see training/registry.py),
    # whether built-in or a plugin trainer named in the config's ``imports``.
    # Validated against the registry at dispatch time (launch._run_child), not
    # here, so a new trainer needs no edit to this file.
    type: str
    devices: int | list[int] | Literal["all"] = "all"
    generate_dcp_seed: bool = False
    preprocess_config: str | list[str] | None = None
    trackio_dir: str | None = "./runs/.trackio"
    """Local Trackio storage directory exported as ``TRACKIO_DIR`` by launcher."""
    env: dict[str, str] = {}
