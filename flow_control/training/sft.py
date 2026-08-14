import math
import os
import time
from typing import Any

import torch
from pydantic import ConfigDict, PositiveInt
from rich.panel import Panel
from rich.progress import Progress
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torchdata.stateful_dataloader import StatefulDataLoader

from flow_control.adapters import ModelAdapter
from flow_control.datasets import DatasetConfig
from flow_control.processors import Processor
from flow_control.samplers import Sampler
from flow_control.utils.logging import (
    console,
    dump_if_failed,
    get_logger,
    warn_once,
)
from flow_control.utils.tensor import (
    deep_cast_float_dtype,
    deep_move_to_device,
)
from flow_control.utils.types import (
    OptimizerConfig,
    SchedulerConfig,
    parse_optimizer,
    parse_scheduler,
)

from .data import (
    DistributedBucketSampler,
    PaddingAwareDatasetWrapper,
    collate_fn,
    seed_worker,
)
from .ema import EMAConfig, EMAOptimizer, apply_ema_maybe
from .mixins import (
    CheckpointingMixin,
    ValidationMixin,
    distributed_main,
    trainer_registry,
)
from .weighting import (
    LogitNormalTimestepWeighting,
    LossWeighting,
    TimestepWeighting,
    UniformLossWeighting,
)

logger = get_logger(__name__)


@trainer_registry.register("sft")
class SftTrainer(ValidationMixin, CheckpointingMixin):
    model_config = ConfigDict(extra="forbid")
    training_type: str = "sft"

    # ---------------------------------- Configs --------------------------------- #
    model: ModelAdapter
    validation_sampler: Sampler
    processor: Processor

    dataset: DatasetConfig
    seed_checkpoint_dir: str
    resume_from_dir: str | None = None
    num_dataloader_workers: int = 1

    optimizer_config: OptimizerConfig = {"class_name": "AdamW", "lr": 1e-4}
    scheduler_config: SchedulerConfig = {"class_name": "ConstantLR", "factor": 1.0}

    global_batch_size: int = 16
    micro_batch_size: PositiveInt = 1
    """Number of logical samples in each per-rank physical forward."""
    train_steps: int = 10000
    checkpoint_interval: int = 500
    """Archival checkpoint cadence in optimizer steps."""
    validation_steps: int = 1000

    timestep_weighting: TimestepWeighting = LogitNormalTimestepWeighting()
    loss_weighting: LossWeighting = UniformLossWeighting()

    ema: EMAConfig | None = None
    clip_grad_norm: float = 1.0

    cfg_drop_prob: float = 0.0
    latent_length_test_mode: bool = False

    # --------------------------------- Status bar ------------------------------- #
    _status_fields: dict[str, str] = {
        "train/loss": "Loss: {v:.4f}",
        "train/lr": "LR: {v:.6f}",
    }

    # ------------------------------- Lazy state --------------------------------- #
    _dataloader: StatefulDataLoader
    _optimizer: torch.optim.Optimizer
    _scheduler: Any
    _ema_optimizer: EMAOptimizer | None = None
    _current_step: int = 0

    @property
    def transformer(self):
        return self.model.transformer

    @property
    def grad_acc_steps(self):
        if self.global_batch_size % self.world_size != 0:
            raise ValueError(
                f"global_batch_size ({self.global_batch_size}) must be divisible "
                f"by world_size ({self.world_size})."
            )
        local_update_batch = self.global_batch_size // self.world_size
        if local_update_batch % self.micro_batch_size != 0:
            raise ValueError(
                f"Per-rank update batch ({local_update_batch}) must be divisible "
                f"by micro_batch_size ({self.micro_batch_size})."
            )
        return local_update_batch // self.micro_batch_size

    @property
    def total_epochs(self):
        return math.ceil(
            self.train_steps / (len(self._dataloader) // self.grad_acc_steps)
        )

    @property
    def current_epoch(self):
        return self._current_step // (len(self._dataloader) // self.grad_acc_steps)

    # ------------------------------- Setup methods ------------------------------ #

    def make_optimizer_and_scheduler(self):
        params = [p for p in self.transformer.parameters() if p.requires_grad]
        num_trainable_params = sum(p.numel() for p in params)
        if num_trainable_params == 0:
            raise RuntimeError("No trainable parameters found in the model.")
        self._optimizer = parse_optimizer(self.optimizer_config, params)
        logger.info(
            f"Created optimizer with {num_trainable_params / 1e6:.2f}M trainable parameters."
        )
        self._scheduler = parse_scheduler(self.scheduler_config, self._optimizer)
        if self.ema is not None:
            self._ema_optimizer = EMAOptimizer(params, self.ema)

    def make_train_dataloader(self):
        # The sampler pads logical items to a complete optimizer update, while
        # the DataLoader groups those items into physical microbatches.
        physical_grad_acc_steps = self.grad_acc_steps
        dataset = PaddingAwareDatasetWrapper(self.parse_training_dataset(self.dataset))
        sampler = DistributedBucketSampler(
            dataset=dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            seed=self.seed,
            grad_acc_steps=physical_grad_acc_steps * self.micro_batch_size,
        )
        self._dataloader = StatefulDataLoader(
            dataset,
            batch_size=self.micro_batch_size,
            sampler=sampler,
            num_workers=self.num_dataloader_workers,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
        )

    # ------------------------------- Checkpointing ------------------------------ #

    def state_dict(self):
        opts = StateDictOptions(strict=False, ignore_frozen_params=True)
        transformer_sd = get_model_state_dict(self.transformer, options=opts)
        if len(transformer_sd) == 0:
            raise RuntimeError("Nothing to save in transformer state dict.")
        state: dict[str, Any] = {
            "transformer": transformer_sd,
            "optimizer": get_optimizer_state_dict(
                self.transformer, self._optimizer, options=opts
            ),
            "dataloader": self._dataloader.state_dict(),
            "scheduler": self._scheduler.state_dict(),
            "current_step": self._current_step,
            "rng": self.get_rng_state_bytes(),
        }
        if self._ema_optimizer is not None:
            state["optim_ema"] = get_optimizer_state_dict(
                self.transformer, self._ema_optimizer, options=opts
            )
        return state

    def load_state_dict(self, state_dict: dict[str, Any]):
        opts = StateDictOptions(strict=False, ignore_frozen_params=True)
        set_model_state_dict(self.transformer, state_dict["transformer"], options=opts)
        set_optimizer_state_dict(
            self.transformer,
            self._optimizer,
            state_dict["optimizer"],
            options=opts,
        )
        if self._ema_optimizer is not None and "optim_ema" in state_dict:
            set_optimizer_state_dict(
                self.transformer,
                self._ema_optimizer,
                state_dict["optim_ema"],
                options=opts,
            )
            self._ema_optimizer.coerce_buffer_dtype()
        self._dataloader.load_state_dict(state_dict["dataloader"])
        self._scheduler.load_state_dict(state_dict["scheduler"])
        self._current_step = state_dict["current_step"]
        self.load_rng_state_bytes(state_dict.get("rng"))

    # ------------------------------- Training ----------------------------------- #

    def train_step(self, batches: list[Any]) -> torch.Tensor:
        timesteps: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []
        model_batches: list[Any] = []

        for original_batch in batches:
            batch: Any = original_batch
            if self.cfg_drop_prob > 0.0 and torch.rand(1).item() < self.cfg_drop_prob:
                negative_batch = self.processor.get_negative_batch(batch)
                if negative_batch is not None:
                    batch = negative_batch
                else:
                    warn_once(
                        logger,
                        f"CFG drop prob is set to {self.cfg_drop_prob}, but no negative (unconditional) batch available.",
                    )

            timestep = self.timestep_weighting.sample_timesteps(1).to(
                device=self.device, dtype=torch.float32
            )
            clean = batch["clean_latents"].float()
            noise = torch.randn_like(clean, dtype=torch.float32)
            batch["noisy_latents"] = (1.0 - timestep) * clean + timestep * noise

            model_batches.append(batch)
            timesteps.append(timestep)
            targets.append(noise - clean)
            weights.append(
                self.loss_weighting.get_weights(timestep).to(
                    device=self.device, dtype=torch.float32
                )
            )

        predictions = self.model.predict_velocity_batched(model_batches, timesteps)
        per_sample_losses = [
            ((prediction - target) ** 2).mean() * weight.mean()
            for prediction, target, weight in zip(
                predictions, targets, weights, strict=True
            )
        ]
        return torch.stack(per_sample_losses).mean()

    def _after_sync_step(self, total_loss: float):
        """Handle optimizer step, logging, checkpointing after a gradient sync."""
        if self.clip_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.transformer.parameters(), self.clip_grad_norm
            )

        self._optimizer.step()
        if self._ema_optimizer is not None:
            self._ema_optimizer.step()
        self._scheduler.step()
        self._optimizer.zero_grad()

        self._current_step += 1
        self.log_metrics(
            {
                "train/loss": total_loss,
                "train/lr": float(self._scheduler.get_last_lr()[0]),
            },
            step=self._current_step,
        )

        self.save_maybe(
            self._current_step,
            force_archival=self._current_step == self.train_steps,
        )

        if self._current_step % self.validation_steps == 0:
            with apply_ema_maybe(self._ema_optimizer):
                self.validate_and_log(self.model, self._current_step)

    def check_loss(self, loss: torch.Tensor):
        if not torch.isfinite(loss):
            logger.error(
                f"Non-finite loss detected (loss={loss.item()}). Stopping training."
            )
            raise RuntimeError("Non-finite loss detected.")

    # ------------------------------- Main loop ---------------------------------- #

    @distributed_main
    def run(self):
        if self.latent_length_test_mode:
            self.run_latent_length_test()
            return

        self.set_seed()
        self.resolve_run_context()
        self.init_tracker()
        self.load_transformer_from_seed(self.model, self.seed_checkpoint_dir)
        self.make_optimizer_and_scheduler()
        self.load_processor()
        self.make_train_dataloader()
        self.make_validation_dataloader()
        os.makedirs(self.checkpoint_root, exist_ok=True)

        self.maybe_auto_resume(self.resume_from_dir)

        with apply_ema_maybe(self._ema_optimizer):
            self.validate_and_log(self.model, self._current_step)

        progress = Progress(
            *self.get_progress_columns(),
            console=console,
        )
        task = progress.add_task(
            "Training",
            total=self.train_steps,
            completed=self._current_step,
        )

        with self.status_bar("SFT Training"), progress:
            starting_epoch = self.current_epoch
            accumulated_loss = 0.0
            for _ in range(starting_epoch, self.total_epochs):
                if hasattr(self._dataloader.sampler, "set_epoch"):
                    self._dataloader.sampler.set_epoch(self.current_epoch)  # type: ignore[union-attr]
                for i, items in enumerate(self._dataloader):
                    with dump_if_failed(logger, items):
                        is_sync_step = (i + 1) % self.grad_acc_steps == 0
                        self.transformer.set_requires_gradient_sync(is_sync_step)

                        items = deep_move_to_device(items, self.device)
                        batches = [
                            deep_cast_float_dtype(
                                self.preprocess_for_training(item), self.model.dtype
                            )
                            for item in items
                        ]

                        loss = self.train_step(batches)
                        self.check_loss(loss)
                        scaled_loss = loss / self.grad_acc_steps
                        scaled_loss.backward()
                        accumulated_loss += scaled_loss.item()

                    if not is_sync_step:
                        continue

                    self._after_sync_step(accumulated_loss)
                    accumulated_loss = 0.0
                    progress.advance(task)

                    if self._current_step >= self.train_steps:
                        break

        with apply_ema_maybe(self._ema_optimizer):
            self.save_dcp_checkpoint(
                self.get_checkpoint_dir(self._current_step) + "_final"
            )

        console.rule("[bold green]Training completed[/bold green]")

    def run_latent_length_test(self):
        logger.warning(
            "Running in latent length test mode since enabled in config. This will not perform training, but will test "
            "increasing latent lengths until OOM. This is useful for finding the maximum latent length that fits in memory."
        )

        self.set_seed()
        self.load_transformer_from_seed(self.model)
        self.make_optimizer_and_scheduler()
        self.load_processor()

        console.rule("[bold blue]Starting latent length test[/bold blue]")

        current_len = 0
        best_len = 0
        try:
            for batch in self.model.latent_length_test():
                current_len = batch["latent_length"]
                start_time = time.time()
                logger.info(f"Testing latent length: {current_len}")
                batch = deep_cast_float_dtype(batch, self.model.dtype)
                batch = deep_move_to_device(batch, self.device)
                loss = self.train_step([batch])
                loss.backward()
                self._optimizer.step()
                self._optimizer.zero_grad()
                elapsed_time = time.time() - start_time
                logger.info(
                    f"Successfully trained with latent length {current_len} in {elapsed_time:.2f} seconds."
                )
                best_len = current_len
            logger.info(
                f"Latent length test completed successfully up to length {current_len}."
            )
        except torch.OutOfMemoryError:
            logger.error(
                f"Out of memory error encountered at latent length {current_len}."
            )
        finally:
            console.rule("[bold red]Latent length test completed[/bold red]")
            console.print(Panel.fit(f"Maximum latent length: [bold]{best_len}[/bold]"))
