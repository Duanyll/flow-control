import os
import pickle
import random
from functools import wraps

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from pydantic import BaseModel
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
)
from torch.distributed.fsdp import fully_shard

from flow_control.adapters.base import BaseModelAdapter
from flow_control.utils import device as devutil
from flow_control.utils.logging import get_logger
from flow_control.utils.registry import Registry

from ..launch_config import LaunchConfig

logger = get_logger(__name__)


class BaseTrainer(BaseModel):
    # ---------------------------------- Configs --------------------------------- #
    launch: LaunchConfig
    imports: list[str] = []
    """Out-of-tree plugin modules to import (for registry side effects) before
    constructing this config. Loaded explicitly by the launcher / CLI, never via
    an env var."""
    seed: int = 42
    hsdp_shard_dim: int = 1
    """
    The dimension along which to shard the model parameters in FSDP2. If set to 1, this
    means no sharding and equivalent to DDP. Usually you don't want to shard across 
    NUMA nodes or NVLink domain because of the communication overhead.
    """
    gradient_checkpointing: bool = True

    # ------------------------------ Lifecycle hooks ----------------------------- #
    # The launchable entry points (launch / seed / export) construct a trainer and
    # call these uniformly; subclasses override as needed. No external code
    # special-cases a trainer type.

    def run(self) -> None:
        """Run the trainer's main loop. Subclasses implement this."""
        raise NotImplementedError(f"{type(self).__name__} does not implement run().")

    def seed_checkpoint(self) -> None:
        """Generate a DCP seed checkpoint (``flow-control seed``).

        Default: seed the transformer adapter (``self.model``) — the common case
        for the diffusion trainers. Trainers that train something else (e.g. the
        VAE trainer) override this.
        """
        # ``model`` / ``seed_checkpoint_dir`` are declared by the diffusion
        # trainers that use this default; the VAE trainer overrides instead.
        model = self.model  # type: ignore[attr-defined]
        model.load_transformer(device=torch.device("cpu"))
        self.save_transformer_to_seed(model, self.seed_checkpoint_dir)  # type: ignore[attr-defined]

    def export_checkpoint(self, checkpoint_dir: str, output_dir: str) -> None:
        """Export a trained DCP checkpoint to HuggingFace format (``export``).

        Not supported by default; trainers that can export (e.g. the VAE trainer)
        override this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support checkpoint export."
        )

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
        return torch.device(devutil.current_device_type(), self.local_rank)

    # ---------------------------------- Methods --------------------------------- #

    def init_device_mesh(self):
        if not dist.is_torchelastic_launched():
            raise RuntimeError("HSDPTrainer requires torchelastic launch.")
        self._world_size = int(os.environ["WORLD_SIZE"])
        self._rank = int(os.environ["RANK"])
        self._local_rank = int(os.environ["LOCAL_RANK"])
        dev = self.device
        devutil.set_device(dev)
        dist.init_process_group(backend=devutil.dist_backend(dev), device_id=dev)
        self._mesh = dist.device_mesh.init_device_mesh(
            devutil.current_device_type(),
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
        devutil.manual_seed_all(seed)
        logger.info(f"Random seed set to {seed} for rank {self.rank}.")

    def get_rng_state_bytes(self) -> bytes:
        """Serialize this rank's global RNG state (torch/accelerator/numpy/random).

        Packed as a single pickled blob so DCP treats it as a per-rank bytes
        object (each rank saves/loads its own copy, not a broadcast of rank 0).
        """
        state: dict[str, object] = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
        acc_state = devutil.get_rng_state(self.device)
        if acc_state is not None:
            state[devutil.current_device_type()] = acc_state
        return pickle.dumps(state)

    def load_rng_state_bytes(self, data: bytes | None) -> None:
        if not data:
            return
        state = pickle.loads(data)
        torch.set_rng_state(state["torch"])
        np.random.set_state(state["numpy"])
        random.setstate(state["python"])
        acc_state = state.get(devutil.current_device_type())
        if acc_state is not None:
            devutil.set_rng_state(acc_state, self.device)
        logger.info("Restored RNG state from checkpoint.")

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

    def cleanup(self) -> None:
        if self.mesh is None or not dist.is_available() or not dist.is_initialized():
            logger.info(
                "Distributed resources were not initialized or already cleaned up."
            )
            return

        try:
            dist.destroy_process_group()
        except Exception:
            logger.exception("dist.destroy_process_group() failed during cleanup.")
        finally:
            self._mesh = None

        logger.info("Cleaned up distributed resources.")


# tag -> trainer class, mapping a config's ``launch.type`` to its trainer. Lives
# here next to ``BaseTrainer`` (like every other family's registry lives next to
# its base). ``base=BaseTrainer`` enforces that registered trainers implement the
# lifecycle interface. The launch *parent* never imports this module (it imports
# only ``launch_config``, staying torch-free); the child resolves trainers here.
trainer_registry: Registry[BaseTrainer] = Registry("trainer", base=BaseTrainer)


def distributed_main(func):
    @wraps(func)
    def wrapper(self: BaseTrainer, *args, **kwargs):
        try:
            self.init_device_mesh()
            return func(self, *args, **kwargs)
        except Exception:
            # Log the traceback to this rank's flushed log file BEFORE the
            # finally's cleanup(): destroy_process_group() can block (other
            # ranks still mid-collective), which would otherwise stop the
            # exception from ever reaching sys.excepthook -> the real error is
            # silently lost behind an NCCL-timeout hang.
            logger.exception("Uncaught exception in distributed_main run()")
            raise
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
