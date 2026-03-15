import os
import random
from functools import wraps
from typing import Literal

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from pydantic import BaseModel, ConfigDict
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
)
from torch.distributed.fsdp import fully_shard

from flow_control.adapters.base import BaseModelAdapter
from flow_control.utils.logging import get_logger

logger = get_logger(__name__)


class LaunchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["sft", "grpo", "inference"]
    devices: int | list[int]
    generate_dcp_seed: bool = False
    preprocess_config: str | None = None
    env: dict[str, str] = {}


class HsdpMixin(BaseModel):
    # ---------------------------------- Configs --------------------------------- #
    launch: LaunchConfig
    seed: int = 42
    hsdp_shard_dim: int = 1
    gradient_checkpointing: bool = True

    # -------------------------------- Properties -------------------------------- #
    _world_size: int = 1

    @property
    def world_size(self) -> int:
        return self._world_size

    _rank: int = 0

    @property
    def rank(self) -> int:
        return self._rank

    _local_rank: int = 0

    @property
    def local_rank(self) -> int:
        return self._local_rank

    _mesh: dist.device_mesh.DeviceMesh | None = None

    @property
    def mesh(self):
        return self._mesh

    @property
    def is_main_process(self):
        return self.rank == 0

    @property
    def is_local_main_process(self):
        return self.local_rank == 0

    @property
    def device(self):
        return torch.device(f"cuda:{self.local_rank}")

    # ---------------------------------- Methods --------------------------------- #

    def init_device_mesh(self):
        if not dist.is_torchelastic_launched():
            raise RuntimeError("HSDPTrainer requires torchelastic launch.")
        self._world_size = int(os.environ["WORLD_SIZE"])
        self._rank = int(os.environ["RANK"])
        self._local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.local_rank)
        dist.init_process_group(backend="nccl", device_id=self.local_rank)
        self._mesh = dist.device_mesh.init_device_mesh(
            "cuda",
            mesh_shape=(
                self.world_size // self.hsdp_shard_dim,
                self.hsdp_shard_dim,
            ),
            mesh_dim_names=("replicate", "shard"),
        )
        logger.info(
            f"Initialized device mesh: "
            f"world_size={self.world_size}, "
            f"replicate_dim={self.world_size // self.hsdp_shard_dim}, "
            f"shard_dim={self.hsdp_shard_dim}"
        )

    def set_seed(self):
        seed = self.seed + self.rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Random seed set to {seed} for rank {self.rank}.")

    def load_transformer_from_seed(
        self,
        model: BaseModelAdapter,
        seed_checkpoint_dir: str | None = None,
    ) -> BaseModelAdapter:
        if seed_checkpoint_dir is not None:
            with torch.device("meta"):
                model.load_transformer(device=torch.device("meta"))
        else:
            if self.hsdp_shard_dim > 1:
                logger.warning(
                    "HSDP sharding is enabled but loading transformer without DCP seed "
                    "checkpoint. The transformer will be loaded on CPU and then sharded, "
                    "which may be slow and memory-intensive for large models. "
                )
            load_device = (
                torch.device("cpu") if self.hsdp_shard_dim > 1 else self.device
            )
            model.load_transformer(device=load_device)

        if self.gradient_checkpointing:
            if (
                hasattr(model.transformer, "_supports_gradient_checkpointing")
                and model.transformer._supports_gradient_checkpointing
            ):
                model.transformer.enable_gradient_checkpointing()
            else:
                logger.warning(
                    f"Gradient checkpointing is enabled but {model.transformer.__class__.__name__} does not support it."
                )

        fsdp_layers: list[str] = (
            model.transformer._no_split_modules or model.transformer._repeated_blocks
        )
        if not fsdp_layers:
            raise ValueError(
                "Model transformer must specify _no_split_modules or _repeated_blocks for FSDP sharding."
            )
        count = 0
        for _, module in model.transformer.named_modules():
            module_type = type(module).__name__
            if module_type in fsdp_layers:
                fully_shard(module, mesh=self.mesh)
                count += 1
        fully_shard(model.transformer, mesh=self.mesh)
        logger.info(f"Transformer is sharded with {count} FSDP layers.")

        if seed_checkpoint_dir is not None:
            if not os.path.exists(os.path.join(seed_checkpoint_dir, ".metadata")):
                raise ValueError(
                    f"Seed checkpoint directory {seed_checkpoint_dir} does not exist or "
                    "is not a valid DCP checkpoint. Set launch.generate_seed = true or "
                    "manually generate the seed checkpoint with `flow-control seed` command."
                )
            model.transformer.to_empty(device=self.device)
            logger.info("Transformer is materialized on to GPU.")
            logger.info(
                f"Loading seed checkpoint from {seed_checkpoint_dir} into transformer..."
            )
            model_sd, _ = get_state_dict(
                model.transformer, [], options=StateDictOptions(strict=False)
            )
            dcp.load(
                model_sd,
                checkpoint_id=seed_checkpoint_dir,
                planner=dcp.default_planner.DefaultLoadPlanner(allow_partial_load=True),
            )
            logger.info("Seed checkpoint loaded into transformer.")
        return model

    def save_transformer_to_seed(
        self, model: BaseModelAdapter, seed_checkpoint_dir: str
    ):
        logger.info(f"Saving DCP seed checkpoint to {seed_checkpoint_dir}...")
        model_sd, _ = get_state_dict(
            model.transformer, [], options=StateDictOptions(strict=False)
        )
        dcp.save(model_sd, checkpoint_id=seed_checkpoint_dir, no_dist=True)
        logger.info(f"Saved DCP seed checkpoint to {seed_checkpoint_dir}.")

    def cleanup(self):
        if self.mesh is not None:
            dist.barrier()
            dist.destroy_process_group()
        logger.info("Cleaned up distributed resources.")


def distributed_main(func):
    @wraps(func)
    def wrapper(self: HsdpMixin, *args, **kwargs):
        try:
            self.init_device_mesh()
            return func(self, *args, **kwargs)
        finally:
            self.cleanup()

    return wrapper


def main_process_only(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        is_main: bool = getattr(self, "is_main_process", True)
        return func(self, *args, **kwargs) if is_main else None

    return wrapper


def main_process_first(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        world_size = getattr(self, "world_size", 1)
        is_main: bool = getattr(self, "is_main_process", True)
        if world_size <= 1:
            return func(self, *args, **kwargs)

        if is_main:
            result = func(self, *args, **kwargs)
            dist.barrier()
            return result
        else:
            dist.barrier()
            return func(self, *args, **kwargs)

    return wrapper
