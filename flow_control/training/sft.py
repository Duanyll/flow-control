import os
import time
from typing import Any

import torch
from pydantic import ConfigDict
from rich.panel import Panel
from rich.progress import Progress

from flow_control.adapters import ModelAdapter
from flow_control.data import is_padding
from flow_control.rewards.base import load_rewards
from flow_control.samplers import Row
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
)

from .mixins import (
    CheckpointingMixin,
    EpochLoopMixin,
    MicrobatchTrainMixin,
    TrainingPredictionMixin,
    ValidationMixin,
    distributed_main,
    trainer_registry,
)
from .mixins.base import (
    load_model_state_dict,
    load_optimizer_state_dict,
    make_optimizer,
    model_state_dict,
    optimizer_state_dict,
    optimizer_step,
)
from .weighting import (
    LogitNormalTimestepWeighting,
    LossWeighting,
    TimestepWeighting,
    UniformLossWeighting,
)

logger = get_logger(__name__)


@trainer_registry.register("sft")
class SftTrainer(
    TrainingPredictionMixin,
    ValidationMixin,
    MicrobatchTrainMixin,
    EpochLoopMixin,
    CheckpointingMixin,
):
    model_config = ConfigDict(extra="forbid")
    training_type: str = "sft"

    # ---------------------------------- Configs --------------------------------- #
    model: ModelAdapter

    seed_checkpoint_dir: str
    resume_from_dir: str | None = None

    optimizer_config: OptimizerConfig = {"class_name": "AdamW", "lr": 1e-4}
    scheduler_config: SchedulerConfig = {"class_name": "ConstantLR", "factor": 1.0}

    train_batch_size: int = 16
    train_steps: int = 10000
    checkpoint_interval: int = 500
    """Archival checkpoint cadence in optimizer steps."""
    validation_steps: int = 1000

    timestep_weighting: TimestepWeighting = LogitNormalTimestepWeighting()
    loss_weighting: LossWeighting = UniformLossWeighting()

    clip_grad_norm: float = 1.0

    cfg_drop_prob: float = 0.0
    cost_test_mode: bool = False
    """Memory probe instead of training: feed the adapter's synthetic ``cost_test``
    batches of increasing token count until OOM."""

    # --------------------------------- Status bar ------------------------------- #
    _status_fields: dict[str, str] = {
        "train/loss": "Loss: {v:.4f}",
        "train/lr": "LR: {v:.6f}",
    }

    # ------------------------------- Lazy state --------------------------------- #
    _optimizer: torch.optim.Optimizer
    _scheduler: Any

    @property
    def transformer(self):
        return self.model.transformer

    # ------------------------------- Setup methods ------------------------------ #

    def make_optimizer_and_scheduler(self):
        self._optimizer, self._scheduler = make_optimizer(
            self.model, self.optimizer_config, self.scheduler_config
        )

    def make_train_dataloader(self):
        # The plan hands every rank the same number of rows per epoch (padding
        # rows included); the DataLoader cuts them into physical microbatches.
        self._store = self.open_train_store()
        self.make_train_loader(
            self._store,
            self.make_planner(
                self._store,
                shuffle=True,
                micro_batch_size=self.train_micro_batch_size,
            ),
            micro_batch_size=self.train_micro_batch_size,
            num_workers=self.num_dataloader_workers,
        )

    # ------------------------------- Checkpointing ------------------------------ #

    def state_dict(self):
        return {
            "transformer": model_state_dict(self.model),
            "optimizer": optimizer_state_dict(self.model, self._optimizer),
            "dataloader": self._dataloader.state_dict(),
            "scheduler": self._scheduler.state_dict(),
            "current_step": self._current_step,
            "rng": self.get_rng_state_bytes(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        load_model_state_dict(self.model, state_dict["transformer"])
        load_optimizer_state_dict(self.model, self._optimizer, state_dict["optimizer"])
        self._dataloader.load_state_dict(state_dict["dataloader"])
        self._scheduler.load_state_dict(state_dict["scheduler"])
        self._current_step = state_dict["current_step"]
        self.load_rng_state_bytes(state_dict.get("rng"))

    # ------------------------------- Training ----------------------------------- #

    def train_step(self, rows: list[Any]) -> torch.Tensor:
        points: list[Row] = []
        weights: list[torch.Tensor] = []
        for row in rows:
            if self.cfg_drop_prob > 0.0 and torch.rand(1).item() < self.cfg_drop_prob:
                negative = self.processor.get_negative_row(row)
                if negative is None:
                    warn_once(
                        logger,
                        f"CFG drop prob is set to {self.cfg_drop_prob}, but no negative (unconditional) row available.",
                    )
                else:  # the dropped row keeps its negative for a guided train_predictor
                    row = {**negative, "negative": row["negative"]}
            timestep = self.timestep_weighting.sample_timesteps(1)
            noise = torch.randn_like(row["clean_latents"], dtype=torch.float32)
            points.append(self.noised_point(row, float(timestep), noise))
            weights.append(
                self.loss_weighting.get_weights(timestep).to(
                    device=self.device, dtype=torch.float32
                )
            )

        # Padding rows are forwarded (FSDP collectives stay balanced) but weigh
        # nothing; the mean is over the real rows of the microbatch.
        per_sample_losses = [
            ((prediction - (point["noise"] - point["clean_latents"])) ** 2).mean()
            * weight.mean()
            * (0.0 if is_padding(row) else 1.0)
            for prediction, point, weight, row in zip(
                self.predict_training(points), points, weights, rows, strict=True
            )
        ]
        real = max(1, sum(not is_padding(row) for row in rows))
        return torch.stack(per_sample_losses).sum() / real

    def _after_sync_step(self, total_loss: float):
        """Handle optimizer step, logging, checkpointing after a gradient sync."""
        optimizer_step(
            self.model, self._optimizer, self._scheduler, self.clip_grad_norm
        )
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
            self._validate()

    def _validate(self) -> None:
        self.validate_variants(self.model, self._current_step, None)

    def check_loss(self, loss: torch.Tensor):
        if not torch.isfinite(loss):
            logger.error(
                f"Non-finite loss detected (loss={loss.item()}). Stopping training."
            )
            raise RuntimeError("Non-finite loss detected.")

    # ------------------------------- Main loop ---------------------------------- #

    @distributed_main
    def run(self):
        if self.cost_test_mode:
            self.run_cost_test()
            return

        for name in self.validation_variants:
            self.model.resolve(name)
        self.set_seed()
        self.resolve_run_context()
        self.init_tracker()
        self.load_transformer_from_seed(self.model, self.seed_checkpoint_dir)
        self.make_optimizer_and_scheduler()
        self.load_processor()
        self.make_train_dataloader()
        self.make_validation_dataloader()
        self.make_runner(self.model)
        load_rewards(self, self.device)
        os.makedirs(self.checkpoint_root, exist_ok=True)

        self.maybe_auto_resume(self.resume_from_dir)

        if self.validation_at_start:
            self._validate()

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
            accumulated_loss = 0.0
            for epoch, items, is_sync_step in self.epoch_microbatches():
                with dump_if_failed(logger, items):
                    self.transformer.set_requires_gradient_sync(is_sync_step)

                    rows = [
                        deep_cast_float_dtype(
                            self.prepare_row(item, mode="training", epoch=epoch),
                            self.model.dtype,
                        )
                        for item in items
                    ]

                    loss = self.train_step(rows)
                    self.check_loss(loss)
                    scaled_loss = loss / self.grad_acc_steps
                    scaled_loss.backward()
                    accumulated_loss += scaled_loss.item()

                if not is_sync_step:
                    continue

                self._after_sync_step(accumulated_loss)
                accumulated_loss = 0.0
                progress.advance(task)

        self.save_dcp_checkpoint(self.get_checkpoint_dir(self._current_step) + "_final")

        console.rule("[bold green]Training completed[/bold green]")

    def run_cost_test(self):
        logger.warning(
            "Running in cost test mode since enabled in config. This will not perform training, but will test "
            "increasing sequence costs (token totals) until OOM. This is useful for finding the maximum cost that fits in memory."
        )

        self.set_seed()
        self.load_transformer_from_seed(self.model)
        self.make_optimizer_and_scheduler()
        self.load_processor()

        console.rule("[bold blue]Starting cost test[/bold blue]")

        current_len = 0
        best_len = 0
        try:
            for batch in self.model.cost_test():
                current_len = batch["cost"]
                start_time = time.time()
                logger.info(f"Testing cost: {current_len}")
                batch = deep_cast_float_dtype(batch, self.model.dtype)
                batch = deep_move_to_device(batch, self.device)
                loss = self.train_step([batch])
                loss.backward()
                self._optimizer.step()
                self._optimizer.zero_grad()
                elapsed_time = time.time() - start_time
                logger.info(
                    f"Successfully trained with cost {current_len} in {elapsed_time:.2f} seconds."
                )
                best_len = current_len
            logger.info(f"Cost test completed successfully up to cost {current_len}.")
        except torch.OutOfMemoryError:
            logger.error(f"Out of memory error encountered at cost {current_len}.")
        finally:
            console.rule("[bold red]Cost test completed[/bold red]")
            console.print(Panel.fit(f"Maximum cost: [bold]{best_len}[/bold]"))
