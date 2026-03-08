import os
import random
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from pydantic import BaseModel, ConfigDict
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.fsdp import fully_shard

from flow_control.adapters import ModelAdapter
from flow_control.utils.logging import get_logger

logger = get_logger(__name__)


class HsdpEngineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    hsdp_shard_dim: int = 1
    gradient_checkpointing: bool = True
    async_save: bool = True


class HsdpEngine[TConfig: HsdpEngineConfig](Stateful):
    conf: TConfig

    world_size: int
    rank: int
    local_rank: int
    mesh: dist.device_mesh.DeviceMesh | None = None

    _checkpoint_future: Any = None

    @property
    def is_main_process(self):
        return self.rank == 0

    @property
    def is_local_main_process(self):
        return self.local_rank == 0

    @property
    def device(self):
        return torch.device(f"cuda:{self.local_rank}")

    def init_device_mesh(self):
        if not dist.is_torchelastic_launched():
            raise RuntimeError("HSDPTrainer requires torchelastic launch.")
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.local_rank)
        backend = "cpu:gloo,cuda:nccl" if self.conf.async_save else "nccl"
        dist.init_process_group(backend=backend, device_id=self.local_rank)
        self.mesh = dist.device_mesh.init_device_mesh(
            "cuda",
            mesh_shape=(
                self.world_size // self.conf.hsdp_shard_dim,
                self.conf.hsdp_shard_dim,
            ),
            mesh_dim_names=("replicate", "shard"),
        )
        logger.info(
            f"Initialized device mesh: "
            f"world_size={self.world_size}, "
            f"replicate_dim={self.world_size // self.conf.hsdp_shard_dim}, "
            f"shard_dim={self.conf.hsdp_shard_dim}"
        )

    def set_seed(self):
        seed = self.conf.seed + self.rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Random seed set to {seed} for rank {self.rank}.")

    def load_transformer(
        self,
        model: ModelAdapter,
        seed_checkpoint_dir: str | None = None,
    ) -> ModelAdapter:
        if seed_checkpoint_dir is not None:
            with torch.device("meta"):
                model.load_transformer(device=torch.device("meta"))
        else:
            if self.conf.hsdp_shard_dim > 1:
                logger.warning(
                    "HSDP sharding is enabled but loading transformer without DCP seed "
                    "checkpoint. The transformer will be loaded on CPU and then sharded, "
                    "which may be slow and memory-intensive for large models. "
                )
            load_device = (
                torch.device("cpu") if self.conf.hsdp_shard_dim > 1 else self.device
            )
            model.load_transformer(device=load_device)

        if self.conf.gradient_checkpointing:
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

    def save_transformer_to_seed(self, model: ModelAdapter, seed_checkpoint_dir: str):
        logger.info(f"Saving DCP seed checkpoint to {seed_checkpoint_dir}...")
        model_sd, _ = get_state_dict(
            model.transformer, [], options=StateDictOptions(strict=False)
        )
        dcp.save(model_sd, checkpoint_id=seed_checkpoint_dir, no_dist=True)
        logger.info(f"Saved DCP seed checkpoint to {seed_checkpoint_dir}.")

    def state_dict(self):
        raise NotImplementedError(
            "state_dict() must be implemented in subclasses to save and load checkpoints."
        )

    def load_state_dict(self, state_dict: dict[str, Any]):
        raise NotImplementedError(
            "load_state_dict() must be implemented in subclasses to save and load checkpoints."
        )

    def load_dcp_checkpoint(self, checkpoint_dir: str):
        state_dict = {"app": self}
        # will call self.state_dict() and self.load_state_dict() internally
        dcp.load(
            state_dict,
            checkpoint_id=checkpoint_dir,
            planner=dcp.default_planner.DefaultLoadPlanner(allow_partial_load=True),
        )
        logger.info(f"Resumed DCP checkpoint from {checkpoint_dir}.")

    def save_dcp_checkpoint(self, checkpoint_dir: str):
        if self._checkpoint_future is not None:
            if not self._checkpoint_future.done():
                logger.warning("Still waiting for previous checkpoint to finish.")
            self._checkpoint_future.result()
        state_dict = {"app": self}
        # will call self.state_dict() internally
        if self.conf.async_save:
            self._checkpoint_future = dcp.async_save(
                state_dict, checkpoint_id=checkpoint_dir
            )
            logger.info(f"Started async DCP save to {checkpoint_dir}.")
        else:
            dcp.save(state_dict, checkpoint_id=checkpoint_dir)
            logger.info(f"Saved DCP checkpoint to {checkpoint_dir}.")

    def cleanup(self):
        if self._checkpoint_future is not None:
            logger.info("Waiting for checkpoint to finish before cleanup...")
            self._checkpoint_future.result()
        if self.mesh is not None:
            dist.barrier()
            dist.destroy_process_group()
        logger.info("Cleaned up distributed resources.")


def distributed_main(func):
    def wrapper(self: HsdpEngine, *args, **kwargs):
        try:
            self.init_device_mesh()
            return func(self, *args, **kwargs)
        finally:
            self.cleanup()

    return wrapper
