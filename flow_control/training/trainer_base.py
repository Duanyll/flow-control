"""Shared base class for HSDP-based trainers (SFT, GRPO, etc.)."""

import os
import shutil
from abc import abstractmethod
from collections.abc import Callable
from typing import Any

import aim
import numpy as np
import torch
import torch.distributed as dist
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torchdata.stateful_dataloader import StatefulDataLoader

from flow_control.adapters import ModelAdapter
from flow_control.datasets import DatasetConfig, parse_dataset
from flow_control.processors import Processor
from flow_control.samplers import Sampler
from flow_control.utils.common import (
    deep_cast_float_dtype,
    deep_move_to_device,
    tensor_to_pil,
)
from flow_control.utils.ema import EMAConfig, apply_ema_maybe
from flow_control.utils.logging import (
    console,
    dump_if_failed,
    get_logger,
    get_version,
)
from flow_control.utils.types import (
    OptimizerConfig,
    SchedulerConfig,
    parse_optimizer,
    parse_scheduler,
)

from .data import DistributedBucketSampler, PaddingAwareDatasetWrapper, collate_fn
from .hsdp_engine import HsdpEngine, HsdpEngineConfig

logger = get_logger(__name__)


class HsdpTrainerBaseConfig(HsdpEngineConfig):
    """Shared configuration fields for all HSDP trainers."""

    model: ModelAdapter
    sampler: Sampler
    processor: Processor

    dataset: DatasetConfig
    validation_dataset: DatasetConfig | None = None

    optimizer: OptimizerConfig = {"class_name": "AdamW", "lr": 1e-4}
    scheduler: SchedulerConfig = {"class_name": "ConstantLR", "factor": 1.0}

    checkpoint_root: str
    seed_checkpoint_dir: str
    aim_repo: str = "."
    experiment_name: str
    resume_from_dir: str | None = None
    checkpoint_limit: int = 5

    num_dataloader_workers: int = 4

    ema: EMAConfig | None = None
    clip_grad_norm: float = 1.0


class HsdpTrainerBase[TConfig: HsdpTrainerBaseConfig](HsdpEngine[TConfig], Stateful):
    """Shared functionality for HSDP trainers.

    Subclasses must set ``conf`` to their specific config type and implement
    the abstract methods.
    """

    conf: TConfig

    # --- Common properties ---

    @property
    def model(self):
        return self.conf.model

    @property
    def transformer(self):
        return self.conf.model.transformer

    @property
    def sampler(self):
        return self.conf.sampler

    @property
    def processor(self):
        return self.conf.processor

    # --- Common instance state ---

    dataloader: StatefulDataLoader
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler
    validation_dataloader: StatefulDataLoader | None = None
    tracker: aim.Run

    current_step: int = 0

    # --- Initialization ---

    def init_tracker(self):
        if not self.is_main_process:
            return
        self.tracker = aim.Run(
            repo=self.conf.aim_repo, experiment=self.conf.experiment_name
        )
        conf_dump = self.conf.model_dump(mode="json", warnings="none")
        conf_dump["__version__"] = get_version()
        self.tracker["hparams"] = conf_dump
        logger.info(
            f"Initialized Aim tracker at {self.conf.aim_repo}, "
            f"experiment={self.conf.experiment_name}."
        )

    def make_optimizer_and_scheduler(self, enable_init_backup: bool = False):
        params = [p for p in self.transformer.parameters() if p.requires_grad]
        num_trainable_params = sum(p.numel() for p in params)
        if num_trainable_params == 0:
            raise RuntimeError("No trainable parameters found in the model.")
        self.optimizer = parse_optimizer(
            self.conf.optimizer,
            params,
            ema_config=self.conf.ema,
            enable_init_backup=enable_init_backup,
        )
        logger.info(
            f"Created optimizer with {num_trainable_params / 1e6:.2f}M trainable parameters."
        )
        self.scheduler = parse_scheduler(self.conf.scheduler, self.optimizer)

    def make_validation_dataloader_maybe(self):
        if self.conf.validation_dataset is None:
            logger.info(
                "No validation dataset provided, skipping validation dataloader."
            )
            self.validation_dataloader = None
            return
        self.processor.load_models("decode", device=self.device)
        dataset = PaddingAwareDatasetWrapper(
            parse_dataset(self.conf.validation_dataset)
        )
        sampler = DistributedBucketSampler(
            dataset=dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
            seed=self.conf.seed,
            grad_acc_steps=1,
        )
        self.validation_dataloader = StatefulDataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=self.conf.num_dataloader_workers,
            collate_fn=collate_fn,
        )

    @abstractmethod
    def make_train_dataloader(self):
        """Create the training dataloader. Subclasses must implement."""
        ...

    # --- Checkpointing ---

    def get_checkpoint_dir(self, step: int) -> str:
        return os.path.join(self.conf.checkpoint_root, f"step_{step:07d}")

    def state_dict(self):
        transformer_state_dict, optimizer_state_dict = get_state_dict(
            self.transformer,
            [self.optimizer],
            options=StateDictOptions(strict=False, ignore_frozen_params=True),
        )
        if len(transformer_state_dict) == 0:
            raise RuntimeError("Nothing to save in transformer state dict.")
        state = {
            "transformer": transformer_state_dict,
            "optimizer": optimizer_state_dict,
            "dataloader": self.dataloader.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "current_step": self.current_step,
        }
        self._save_extra_state(state)
        return state

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.transformer,
            [self.optimizer],
            model_state_dict=state_dict["transformer"],
            optim_state_dict=state_dict["optimizer"],
            options=StateDictOptions(strict=False, ignore_frozen_params=True),
        )
        self.dataloader.load_state_dict(state_dict["dataloader"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.current_step = state_dict["current_step"]
        self._load_extra_state(state_dict)

    def _save_extra_state(self, state: dict) -> None:
        """Hook for subclasses to add extra state to the checkpoint."""

    def _load_extra_state(self, state_dict: dict) -> None:
        """Hook for subclasses to load extra state from the checkpoint."""

    def rotate_checkpoints_maybe(self):
        if not self.is_main_process:
            return
        if self.conf.checkpoint_limit <= 0:
            return
        checkpoint_dirs = []
        for name in os.listdir(self.conf.checkpoint_root):
            if name.startswith("step_"):
                checkpoint_dirs.append(name)
        if len(checkpoint_dirs) <= self.conf.checkpoint_limit:
            return
        checkpoint_dirs.sort()
        num_to_remove = len(checkpoint_dirs) - self.conf.checkpoint_limit
        for i in range(num_to_remove):
            dir_to_remove = os.path.join(self.conf.checkpoint_root, checkpoint_dirs[i])
            shutil.rmtree(dir_to_remove)
            logger.info(f"Removed old checkpoint: {dir_to_remove}")

    # --- Logging ---

    def log_metrics(self, metrics: dict[str, float]):
        if self.is_main_process:
            for key, value in metrics.items():
                self.tracker.track(value, name=key, step=self.current_step)

    def log_images(self, image, key):
        image = np.array(image)
        if self.is_main_process:
            images: list[Any] = [None] * self.world_size
            dist.gather_object(image, images, dst=0)
            keys = [None] * self.world_size
            dist.gather_object(key, keys, dst=0)
            for k, img in zip(keys, images, strict=True):
                if k == "__padding__":
                    continue
                self.tracker.track(
                    aim.Image(img), name=f"samples/{k}", step=self.current_step
                )
        else:
            dist.gather_object(image, None, dst=0)
            dist.gather_object(key, None, dst=0)

    # --- Validation ---

    def make_validation_progress_bar(self):
        if self.validation_dataloader is None:
            raise RuntimeError("Validation dataloader is not initialized.")
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn(" Batch: {task.completed}/{task.total}"),
            BarColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,
            disable=len(self.validation_dataloader) <= 1,
        )
        task = progress.add_task(
            "Validating...",
            total=len(self.validation_dataloader),
        )
        return progress, task

    def _default_on_sample(self, batch: dict[str, Any]) -> None:
        """Default per-sample callback: log generated images."""
        image = tensor_to_pil(batch["clean_image"])
        key = batch.get("__key__", "unknown")
        self.log_images(image, key)

    @torch.no_grad()
    def validate_and_log(
        self,
        on_sample: Callable[[dict[str, Any]], None] | None = None,
    ):
        if self.validation_dataloader is None:
            return

        if on_sample is None:
            on_sample = self._default_on_sample

        logger.info(f"Validating at step {self.current_step}...")
        progress, task = self.make_validation_progress_bar()
        self.transformer.eval()
        with apply_ema_maybe(self.optimizer), progress:
            for batch in self.validation_dataloader:
                with dump_if_failed(logger, batch):
                    batch = deep_cast_float_dtype(batch, self.model.dtype)
                    batch = deep_move_to_device(batch, self.device)
                    negative_batch: Any = (
                        self.processor.get_negative_batch(batch)
                        if self.sampler.cfg_scale > 1.0
                        else None
                    )
                    generator = torch.Generator(device=self.device).manual_seed(
                        self.conf.seed
                    )
                    self.processor.initialize_latents(
                        batch,
                        generator=generator,
                        device=self.device,
                        dtype=self.model.dtype,
                    )
                    clean_latents = self.sampler.sample(
                        self.model, batch, negative_batch=negative_batch
                    )
                    if not isinstance(clean_latents, torch.Tensor):
                        raise RuntimeError(
                            "validate_and_log expects sampler.sample(..., return_trajectory=False) "
                            "to return a tensor."
                        )
                    decoded = self.processor.decode_output(clean_latents, batch)
                    batch.update(decoded)
                    on_sample(batch)
                    progress.advance(task)
        self.transformer.train()
        logger.info(f"Completed validation at step {self.current_step}.")
