"""
LaunchConfig stays in a separate file so that launch.py does not need to import torch
and other heavy dependencies from training, making it faster to parse the config and
avoid os.execvp issues.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from flow_control.utils.registry import Registry

# tag -> trainer class. Core trainers register at import of their module (e.g.
# ``@trainer_registry.register("sft")`` on ``SftTrainer``); out-of-tree trainers
# register when loaded via a config's ``imports``. The registry lives here, in
# this deliberately torch-free module, so the launch parent and the ``schema``
# command can read it without importing torch (``registry`` is torch-free too).
trainer_registry: Registry[Any] = Registry("trainer")

# Built-in trainers: declared lazily so importing this (torch-free) module does
# not pull torch. ``trainer_registry.get(tag)`` imports the module on demand,
# running its ``@trainer_registry.register`` decorator. Plugin trainers register
# the same way by being listed in a config's ``imports`` (e.g. the RGBA VAE
# trainer at ``flow_control.contrib.rgba_vae_training``).
for _tag, _module in {
    "sft": "flow_control.training.sft",
    "grpo": "flow_control.training.grpo",
    "nft": "flow_control.training.nft",
    "inference": "flow_control.training.inference",
}.items():
    trainer_registry.register_lazy(_tag, _module)


class LaunchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Open: any tag registered in ``trainer_registry`` (core or plugin). Validated
    # against the registry at dispatch time (launch._run_child), not here, so a
    # plugin trainer named in ``imports`` is accepted without editing this file.
    type: str
    devices: int | list[int] | Literal["all"] = "all"
    generate_dcp_seed: bool = False
    preprocess_config: str | list[str] | None = None
    trackio_dir: str | None = "./runs/.trackio"
    """Local Trackio storage directory exported as ``TRACKIO_DIR`` by launcher."""
    env: dict[str, str] = {}
