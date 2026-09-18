import os
import time
from typing import Any

import torch
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

from flow_control.adapters import ModelAdapter
from flow_control.adapters.base import Batch
from flow_control.data import is_padding
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

from .ema import EMAConfig, EMAOptimizer, apply_ema_maybe
from .mixins import (
    CheckpointingMixin,
    EpochLoopMixin,
    MicrobatchTrainMixin,
    TrainingPredictionMixin,
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
    validation_sampler: Sampler

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

    ema: EMAConfig | None = None
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
    _ema_optimizer: EMAOptimizer | None = None

    @property
    def transformer(self):
        return self.model.transformer

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
            # DCP keeps only ``state`` / ``param_groups``; the warmup schedule
            # needs the step counter too.
            state["ema_step_count"] = self._ema_optimizer.ema_step_count
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
            # Stepped once per optimizer step, so older checkpoints without the
            # counter resume it from the step count.
            self._ema_optimizer.ema_step_count = state_dict.get(
                "ema_step_count", state_dict["current_step"]
            )
        self._dataloader.load_state_dict(state_dict["dataloader"])
        self._scheduler.load_state_dict(state_dict["scheduler"])
        self._current_step = state_dict["current_step"]
        self.load_rng_state_bytes(state_dict.get("rng"))

    # ------------------------------- Training ----------------------------------- #

    def train_step(self, rows: list[Any]) -> torch.Tensor:
        timesteps: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []
        model_rows: list[Any] = []
        negative_rows: list[Batch | None] = []

        for original_batch in rows:
            row: Any = original_batch
            # Preserve the original negative condition before CFG dropout can
            # replace the positive row (negative rows have no overlay).
            negative = self.training_negative(original_batch)
            negative_rows.append(negative)
            if self.cfg_drop_prob > 0.0 and torch.rand(1).item() < self.cfg_drop_prob:
                negative_row = (
                    negative
                    if negative is not None
                    else self.processor.get_negative_row(row)
                )
                if negative_row is not None:
                    row = negative_row
                else:
                    warn_once(
                        logger,
                        f"CFG drop prob is set to {self.cfg_drop_prob}, but no negative (unconditional) row available.",
                    )

            timestep = self.timestep_weighting.sample_timesteps(1).to(
                device=self.device, dtype=torch.float32
            )
            clean = row["clean_latents"].float()
            noise = torch.randn_like(clean, dtype=torch.float32)
            row["noisy_latents"] = (1.0 - timestep) * clean + timestep * noise

            model_rows.append(row)
            timesteps.append(timestep)
            targets.append(noise - clean)
            weights.append(
                self.loss_weighting.get_weights(timestep).to(
                    device=self.device, dtype=torch.float32
                )
            )

        predictions = self.predict_training(model_rows, timesteps, negative_rows)
        # Padding rows are forwarded (FSDP collectives stay balanced) but weigh
        # nothing; the mean is over the real rows of the microbatch.
        per_sample_losses = [
            ((prediction - target) ** 2).mean()
            * weight.mean()
            * (0.0 if is_padding(row) else 1.0)
            for prediction, target, weight, row in zip(
                predictions, targets, weights, rows, strict=True
            )
        ]
        real = max(1, sum(not is_padding(row) for row in rows))
        return torch.stack(per_sample_losses).sum() / real

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
        if self.cost_test_mode:
            self.run_cost_test()
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

        with apply_ema_maybe(self._ema_optimizer):
            self.save_dcp_checkpoint(
                self.get_checkpoint_dir(self._current_step) + "_final"
            )

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
