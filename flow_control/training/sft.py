import math
import os
import time
from typing import Any

import torch
from einops import reduce
from pydantic import ConfigDict
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
from .mixins import CheckpointingMixin, ValidationMixin, distributed_main
from .weighting import (
    LogitNormalTimestepWeighting,
    LossWeighting,
    TimestepWeighting,
    UniformLossWeighting,
)

logger = get_logger(__name__)


class SftTrainer(ValidationMixin, CheckpointingMixin):
    model_config = ConfigDict(extra="forbid")

    # ---------------------------------- Configs --------------------------------- #
    model: ModelAdapter
    validation_sampler: Sampler
    processor: Processor

    dataset: DatasetConfig
    seed_checkpoint_dir: str
    resume_from_dir: str | None = None
    num_dataloader_workers: int = 4

    optimizer_config: OptimizerConfig = {"class_name": "AdamW", "lr": 1e-4}
    scheduler_config: SchedulerConfig = {"class_name": "ConstantLR", "factor": 1.0}

    global_batch_size: int = 16
    train_steps: int = 10000
    checkpoint_steps: int = 500
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
        return self.global_batch_size // self.world_size

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
        dataset = PaddingAwareDatasetWrapper(self.parse_training_dataset(self.dataset))
        sampler = DistributedBucketSampler(
            dataset=dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            seed=self.seed,
            grad_acc_steps=self.grad_acc_steps,
        )
        self._dataloader = StatefulDataLoader(
            dataset,
            batch_size=1,
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
        self._dataloader.load_state_dict(state_dict["dataloader"])
        self._scheduler.load_state_dict(state_dict["scheduler"])
        self._current_step = state_dict["current_step"]

    # ------------------------------- Training ----------------------------------- #

    def train_step(self, batch: Any):
        if self.cfg_drop_prob > 0.0 and torch.rand(1).item() < self.cfg_drop_prob:
            negative_batch = self.processor.get_negative_batch(batch)
            if negative_batch is not None:
                batch = negative_batch
            else:
                warn_once(
                    logger,
                    f"CFG drop prob is set to {self.cfg_drop_prob}, but no negative (unconditional) batch available.",
                )

        timesteps = self.timestep_weighting.sample_timesteps(1)
        timesteps = timesteps.to(device=self.device, dtype=torch.float32)
        clean = batch["clean_latents"].float()
        noise = torch.randn_like(clean, dtype=torch.float32)
        batch["noisy_latents"] = (1.0 - timesteps) * clean + timesteps * noise

        model_pred = self.model.predict_velocity(
            batch, timesteps.to(dtype=self.model.dtype)
        ).float()
        target = noise - clean
        loss = (model_pred - target) ** 2
        loss = reduce(loss, "b n d -> 1", reduction="mean")

        weighting = self.loss_weighting.get_weights(timesteps)
        weighting = weighting.to(device=loss.device, dtype=loss.dtype)
        weighted_loss = (loss * weighting).mean()
        return weighted_loss

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

        if (self._current_step % self.checkpoint_steps == 0) or (
            self._current_step == self.train_steps
        ):
            self.save(self._current_step)

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
        self.init_tracker()
        self.load_transformer_from_seed(self.model, self.seed_checkpoint_dir)
        self.make_optimizer_and_scheduler()
        self.load_processor()
        self.make_train_dataloader()
        self.make_validation_dataloader()
        os.makedirs(self.checkpoint_root, exist_ok=True)

        if self.resume_from_dir is not None:
            self.load_dcp_checkpoint(self.resume_from_dir)

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
            for _ in range(starting_epoch, self.total_epochs):
                if hasattr(self._dataloader.sampler, "set_epoch"):
                    self._dataloader.sampler.set_epoch(self.current_epoch)  # type: ignore[union-attr]
                for i, batch in enumerate(self._dataloader):
                    with dump_if_failed(logger, batch):
                        is_sync_step = (i + 1) % self.grad_acc_steps == 0
                        self.transformer.set_requires_gradient_sync(is_sync_step)

                        batch = deep_move_to_device(batch, self.device)
                        batch: Any = self.preprocess_for_training(batch)
                        batch = deep_cast_float_dtype(batch, self.model.dtype)

                        loss = self.train_step(batch) / self.grad_acc_steps
                        self.check_loss(loss)
                        loss.backward()

                        total_loss = loss.item()

                    if not is_sync_step:
                        continue

                    self._after_sync_step(total_loss)
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
                loss = self.train_step(batch)
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
        except torch.cuda.OutOfMemoryError:
            logger.error(
                f"Out of memory error encountered at latent length {current_len}."
            )
        finally:
            console.rule("[bold red]Latent length test completed[/bold red]")
            console.print(Panel.fit(f"Maximum latent length: [bold]{best_len}[/bold]"))
