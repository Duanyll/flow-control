import os
import random
import signal
import sys
from contextlib import ContextDecorator
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from pydantic import BaseModel
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.fsdp import fully_shard

from flow_control.adapters import ModelAdapter
from flow_control.utils.logging import console, get_logger, warn_once

logger = get_logger(__name__)


class HsdpEngineConfig(BaseModel):
    seed: int = 42
    hsdp_shard_dim: int = 1
    gradient_checkpointing: bool = True


class HsdpEngine(Stateful):
    raw_conf: dict[str, Any]
    conf: HsdpEngineConfig

    world_size: int
    rank: int
    local_rank: int
    mesh: dist.device_mesh.DeviceMesh | None = None

    @property
    def is_main_process(self):
        return self.rank == 0

    @property
    def is_local_main_process(self):
        return self.local_rank == 0

    @property
    def device(self):
        return torch.device(f"cuda:{self.local_rank}")

    def __init__(self, **kwargs):
        self.raw_conf = deepcopy(kwargs)

    def init_device_mesh(self):
        if not dist.is_torchelastic_launched():
            raise RuntimeError("HSDPTrainer requires torchelastic launch.")
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.local_rank)
        dist.init_process_group(backend="nccl", device_id=self.local_rank)
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

    def load_transformer_from_seed(
        self, model: ModelAdapter, seed_checkpoint_dir: str | None = None
    ) -> ModelAdapter:
        with torch.device("meta"):
            model.load_transformer(device=torch.device("meta"))
            if self.conf.gradient_checkpointing:
                if (
                    hasattr(model.transformer, "_supports_gradient_checkpointing")
                    and model.transformer._supports_gradient_checkpointing
                ):
                    model.transformer.enable_gradient_checkpointing()
                else:
                    warn_once(
                        logger,
                        "Gradient checkpointing is enabled in the config, "
                        "but the transformer model does not support it.",
                    )
        if not hasattr(model.transformer, "_no_split_modules"):
            raise ValueError(
                "The transformer model must define _no_split_modules for HSDP."
            )
        fsdp_layers: list[str] = model.transformer._no_split_modules
        count = 0
        for _, module in model.transformer.named_modules():
            module_type = type(module).__name__
            if module_type in fsdp_layers:
                fully_shard(module, mesh=self.mesh)
                count += 1
        fully_shard(model.transformer, mesh=self.mesh)
        logger.info(f"Transformer is sharded with {count} FSDP layers.")
        model.transformer.to_empty(device=self.device)
        logger.info("Transformer is materialized on to GPU.")

        if seed_checkpoint_dir is not None:
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
        elif not model.all_trainable:
            warn_once(
                logger,
                "Model is not fully trainable and no seed checkpoint is provided. "
                "This may lead to uninitialized parameters.",
            )
        return model

    def save_transformer_to_seed(self, model: ModelAdapter, seed_checkpoint_dir: str):
        logger.info(f"Saving DCP seed checkpoint to {seed_checkpoint_dir}...")
        model_sd, _ = get_state_dict(
            model.transformer, [], options=StateDictOptions(strict=False)
        )
        dcp.save(model_sd, checkpoint_id=seed_checkpoint_dir)
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
        if self.is_main_process:
            os.makedirs(checkpoint_dir, exist_ok=True)
        dist.barrier()
        state_dict = {"app": self}
        # will call self.state_dict() internally
        dcp.save(state_dict, checkpoint_id=checkpoint_dir)
        logger.info(f"Saved DCP checkpoint to {checkpoint_dir}.")

    def cleanup(self):
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


class DistributedExitSignal(ContextDecorator):
    def __init__(self, hsdp_engine: HsdpEngine):
        super().__init__()

        self.sigint_received = False
        self.original_handler = None
        self.hsdp_engine = hsdp_engine

    def __enter__(self):
        self.original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.handle_sigint)
        logger.info("Registered SIGINT handler for distributed exit signal.")
        return self

    def __exit__(self, *exc):
        signal.signal(signal.SIGINT, self.original_handler)
        logger.info("Restored original SIGINT handler.")
        return False

    def handle_sigint(self, signum, frame):
        if self.sigint_received:
            logger.error("Second SIGINT received. Exiting immediately.")
            # Do not catch further SIGINTs
            signal.signal(signal.SIGINT, self.original_handler)
            sys.exit(1)
        else:
            self.sigint_received = True
            console.rule("[red]SIGINT[/red]")
            logger.warning(
                "SIGINT received. Waiting to save state after current step... Press Ctrl+C again to force exit."
            )

    def __bool__(self):
        exit_flag_tensor = torch.tensor(
            int(self.sigint_received), device=self.hsdp_engine.device
        )
        dist.all_reduce(exit_flag_tensor, op=dist.ReduceOp.MAX)
        return bool(exit_flag_tensor.item())
