"""The rollout -> advantage -> update loop shared by the RL trainers.

:class:`RolloutTrainerBase` owns everything that is method-independent: the
config surface, optimizer / scheduler / validation EMA / reference backup,
checkpoint state, the micro-update loop and the outer epoch loop. Subclasses
fill in how rollouts become train items
(:meth:`RolloutTrainerBase._build_train_plan`) and how a microbatch of items
becomes a loss (:meth:`RolloutTrainerBase._loss_batched`):
:class:`~flow_control.training.grpo.GrpoTrainer` replays the recorded rollout
steps, :class:`~flow_control.training.endpoint.EndpointTrainer` re-noises the
rollout endpoints.
"""

import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any

import torch
from pydantic import ConfigDict
from rich.progress import Progress
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)

from flow_control.adapters import ModelAdapter
from flow_control.processors import Processor
from flow_control.rewards import Reward
from flow_control.samplers import Sampler
from flow_control.utils import device as devutil
from flow_control.utils.logging import console, get_logger
from flow_control.utils.types import (
    OptimizerConfig,
    SchedulerConfig,
    parse_optimizer,
    parse_scheduler,
)

from .ema import (
    EMAConfig,
    EMAOptimizer,
    InitBackupOptimizer,
    apply_ema_maybe,
    apply_init_maybe,
)
from .mixins import (
    CheckpointingMixin,
    MicrobatchTrainMixin,
    Rollout,
    RolloutMixin,
    TrainingPredictionMixin,
    ValidationMixin,
    distributed_main,
)
from .mixins.microbatch import RolloutIndexedItem

logger = get_logger(__name__)


class RolloutTrainerBase[ItemT: RolloutIndexedItem](
    TrainingPredictionMixin,
    RolloutMixin,
    ValidationMixin,
    MicrobatchTrainMixin,
    CheckpointingMixin,
    ABC,
):
    """Rollout / advantage / update loop; subclasses define items and their loss."""

    model_config = ConfigDict(extra="forbid")

    # ---------------------------------- Configs --------------------------------- #
    model: ModelAdapter
    rollout_sampler: Sampler
    processor: Processor
    reward: Reward

    seed_checkpoint_dir: str
    resume_from_dir: str | None = None

    optimizer_config: OptimizerConfig = {"class_name": "AdamW", "lr": 3e-4}
    scheduler_config: SchedulerConfig = {"class_name": "ConstantLR", "factor": 1.0}

    num_inner_epochs: int = 1
    """Passes over each epoch's rollouts. Whether a pass draws fresh train
    items or reorders the same ones is up to :meth:`_build_train_plan`."""

    ema: EMAConfig | None = None
    """Validation EMA config (stepped per gradient step)."""
    precompute_aux_model_outputs: bool = False
    """Precompute the auxiliary (old / reference) model outputs for the whole
    train plan once per outer epoch and reuse them during optimization. Fewer
    weight switches at the cost of accelerator memory for the cached tensors."""

    clip_grad_norm: float = 1.0
    """Gradient-norm clip before each optimizer step; ``0.0`` disables it."""

    # Optimization / training loop
    train_epochs: int = 100
    validation_epochs: int = 20
    validation_non_ema: bool = False
    """Also log validation for the current weights before applying the
    validation EMA (``val/non_ema``)."""

    # --------------------------------- Status bar ------------------------------- #
    _status_fields: dict[str, str] = {
        "rollout/reward_mean": "R̄: {v:.3f}",
        "rollout/reward_std": "σ: {v:.3f}",
        "train/loss": "Loss: {v:.4f}",
        "val/reward_mean": "Val R̄: {v:.3f}",
    }

    # ------------------------------- Lazy state --------------------------------- #
    _optimizer: torch.optim.Optimizer
    _scheduler: Any
    _ema_optimizer: EMAOptimizer | None = None
    _init_backup_optimizer: InitBackupOptimizer | None = None
    _current_step: int = 0
    _current_epoch: int = 0

    # ------------------------------- Properties --------------------------------- #

    @property
    def transformer(self):
        return self.model.transformer

    # --------------------------------- Hooks ------------------------------------ #

    @abstractmethod
    def _needs_reference(self) -> bool:
        """Whether the loss evaluates the frozen reference (initial) weights."""
        ...

    @abstractmethod
    def _build_train_plan(self, rollouts: list[Rollout]) -> list[list[ItemT]]:
        """One item list per inner epoch; every rank must build equal counts."""
        ...

    @abstractmethod
    def _loss_batched(
        self, items: list[ItemT], rollouts: list[Rollout], advantages: torch.Tensor
    ) -> torch.Tensor:
        """Mean loss of one microbatch, attached to the graph."""
        ...

    def _precompute(
        self,
        rollouts: list[Rollout],
        train_plan: list[list[ItemT]],
        advantages: torch.Tensor,
    ) -> None:
        """Fill per-item caches before the update loop (``precompute_aux_model_outputs``)."""

    def _make_aux_optimizers(self, params: list[torch.nn.Parameter]) -> None:
        """Create subclass-owned optimizers (e.g. an old-policy EMA)."""

    def _aux_state_dict(self, opts: StateDictOptions) -> dict[str, Any]:
        return {}

    def _load_aux_state_dict(
        self, state_dict: dict[str, Any], opts: StateDictOptions
    ) -> None:
        pass

    def _rollout_scope(self) -> AbstractContextManager[None]:
        """Weights the rollout sampler runs under; default: the current policy."""
        return nullcontext()

    def _check_rollouts(self, rollouts: list[Rollout]) -> None:
        """Fail fast on rollouts the trainer cannot train on."""

    def _after_train_epoch(self) -> None:
        """Runs once per outer epoch after the update loop (e.g. EMA steps)."""

    # ------------------------------- Setup methods ------------------------------ #

    def make_optimizer_and_scheduler(self):
        params = [p for p in self.transformer.parameters() if p.requires_grad]
        num_trainable_params = sum(p.numel() for p in params)
        if num_trainable_params == 0:
            raise RuntimeError("No trainable parameters found in the model.")
        self._optimizer = parse_optimizer(self.optimizer_config, params)
        logger.info(
            f"Created optimizer with {num_trainable_params / 1e6:.2f}M trainable "
            "parameters."
        )
        self._scheduler = parse_scheduler(self.scheduler_config, self._optimizer)

        self._make_aux_optimizers(params)

        # Validation EMA (stepped per gradient step)
        if self.ema is not None:
            self._ema_optimizer = EMAOptimizer(params, self.ema)

        # Reference model (frozen initial weights). For LoRA the reference is
        # reached by disabling adapters, so a backup is only required for full
        # fine-tuning.
        if self._needs_reference() and self.model.peft_lora_rank == 0:
            self._init_backup_optimizer = InitBackupOptimizer(params)
            logger.info("Init backup enabled for the reference model.")

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
            "cursor": self._cursor.state_dict(),
            "scheduler": self._scheduler.state_dict(),
            "current_step": self._current_step,
            "current_epoch": self._current_epoch,
            "rng": self.get_rng_state_bytes(),
        }
        state.update(self._aux_state_dict(opts))
        if self._ema_optimizer is not None:
            state["optim_ema"] = get_optimizer_state_dict(
                self.transformer, self._ema_optimizer, options=opts
            )
        if self._init_backup_optimizer is not None:
            state["optim_init_backup"] = get_optimizer_state_dict(
                self.transformer, self._init_backup_optimizer, options=opts
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
        self._load_aux_state_dict(state_dict, opts)
        if self._ema_optimizer is not None and "optim_ema" in state_dict:
            set_optimizer_state_dict(
                self.transformer,
                self._ema_optimizer,
                state_dict["optim_ema"],
                options=opts,
            )
            self._ema_optimizer.coerce_buffer_dtype()
        if (
            self._init_backup_optimizer is not None
            and "optim_init_backup" in state_dict
        ):
            set_optimizer_state_dict(
                self.transformer,
                self._init_backup_optimizer,
                state_dict["optim_init_backup"],
                options=opts,
            )
        if "cursor" in state_dict:
            self._cursor.load_state_dict(state_dict["cursor"])
        else:
            logger.warning(
                "Checkpoint has no prompt cursor state (written before the data "
                "rework); rollout prompts restart from the beginning of the plan."
            )
        self._scheduler.load_state_dict(state_dict["scheduler"])
        self._current_step = state_dict["current_step"]
        self._current_epoch = state_dict.get("current_epoch", 0)
        self.load_rng_state_bytes(state_dict.get("rng"))

    # ------------------------------- Reference model ---------------------------- #

    @contextmanager
    def reference_model(self):
        """Temporarily switch to reference (frozen initial) model weights."""
        if self.model.peft_lora_rank > 0:
            with self.model.use_variant("base"):
                yield
        else:
            with apply_init_maybe(self._init_backup_optimizer):
                yield

    # ----------------------------- Training phase ------------------------------- #

    def _optimizer_step(self):
        """Clip gradients (optional), step the per-step optimizers, zero grads."""
        if self.clip_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.transformer.parameters(), self.clip_grad_norm
            )
        self._optimizer.step()
        if self._ema_optimizer is not None:
            self._ema_optimizer.step()
        if self._init_backup_optimizer is not None:
            self._init_backup_optimizer.step()
        self._scheduler.step()
        self._optimizer.zero_grad()

    @contextmanager
    def _precompute_scope(self) -> Iterator[Progress]:
        """Eval-mode, ``no_grad`` scope for :meth:`_precompute` with a transient
        progress bar; task timings are logged under ``profile/precompute``."""
        was_training = self.transformer.training
        self.transformer.eval()
        progress = Progress(
            *self.get_progress_columns(),
            console=console,
            transient=True,
        )
        with progress, torch.no_grad():
            yield progress
        if was_training:
            self.transformer.train()
        self.log_progress_timing(
            progress, self._current_step, prefix="profile/precompute"
        )

    def _train_on_rollouts(
        self,
        rollouts: list[Rollout],
        advantages: torch.Tensor,
    ):
        """Training phase: update model using collected rollouts and advantages."""
        self.transformer.train()

        train_plan = self._build_train_plan(rollouts)
        total_items = sum(len(items) for items in train_plan)
        if self.precompute_aux_model_outputs:
            self._precompute(rollouts, train_plan, advantages)

        progress = Progress(
            *self.get_progress_columns(),
            console=console,
            transient=True,
        )
        train_task = progress.add_task("Training", total=total_items)

        with progress:
            for train_items in train_plan:
                for update in self.iter_micro_updates(train_items):
                    self.transformer.set_requires_gradient_sync(update.is_sync_step)
                    loss = self._loss_batched(update.items, rollouts, advantages)
                    self._check_finite_loss(loss, update.items)
                    (loss * update.loss_scale).backward()
                    progress.advance(train_task, advance=len(update.items))

                    if update.is_sync_step:
                        self._optimizer_step()
                        self._current_step += 1
                        self.flush_aggregated_metrics(self._current_step)

        self.log_progress_timing(progress, self._current_step, prefix="profile/train")

    # -------------------------------- Main loop --------------------------------- #

    def _validate_current_and_ema(self) -> None:
        if self.validation_non_ema and self._ema_optimizer is not None:
            self.validate_and_log(
                self.model,
                self._current_step,
                reward=self.reward,
                metric_prefix="val/non_ema",
                image_name="validation_non_ema",
                profile_prefix="profile/validation_non_ema",
            )
        with apply_ema_maybe(self._ema_optimizer):
            self.validate_and_log(self.model, self._current_step, reward=self.reward)

    @distributed_main
    def run(self):
        self.set_seed()
        self.resolve_run_context()
        self.init_tracker()
        self.load_transformer_from_seed(self.model, self.seed_checkpoint_dir)
        self.make_optimizer_and_scheduler()
        self.load_processor()
        self.make_rollout_cursor()
        self.make_validation_dataloader()

        self.reward.load_model(self.device)
        if self.validation_reward:
            self.validation_reward.load_model(self.device)

        os.makedirs(self.checkpoint_root, exist_ok=True)
        self.maybe_auto_resume(self.resume_from_dir)

        self._validate_current_and_ema()

        name = self.training_type.upper()
        logger.info(
            "%s draws %d distinct prompts per epoch (%s sampling) with %d rollouts "
            "each: %d rollouts per epoch, %d per rank.",
            name,
            self.num_prompts_per_epoch,
            self.prompt_sampling,
            self.num_rollouts_per_prompt,
            self.num_prompts_per_epoch * self.num_rollouts_per_prompt,
            self.rollouts_per_rank,
        )
        logger.info(
            "%s optimization uses train_batch_size=%d, world_size=%d, "
            "grad_acc_steps=%d.",
            name,
            self.train_batch_size,
            self.world_size,
            self.grad_acc_steps,
        )

        progress = Progress(
            *self.get_progress_columns(),
            console=console,
        )
        task = progress.add_task(
            f"{name} Training", total=self.train_epochs, completed=self._current_epoch
        )

        with self.status_bar(f"{name} Training"), progress:
            while self._current_epoch < self.train_epochs:
                logger.debug(f"Epoch {self._current_epoch}: starting rollout phase...")
                with self._rollout_scope():
                    rollouts = self._collect_rollouts(self._current_epoch)
                self._check_rollouts(rollouts)
                advantages = self._compute_advantages(rollouts, step=self._current_step)

                logger.debug(f"Epoch {self._current_epoch}: starting training phase...")
                self._train_on_rollouts(rollouts, advantages)
                self._after_train_epoch()

                self._current_epoch += 1
                progress.update(task, completed=self._current_epoch)

                del rollouts, advantages
                devutil.empty_cache()

                self.save_maybe(
                    self._current_step,
                    progress=self._current_epoch,
                    force_archival=self._current_epoch == self.train_epochs,
                )

                if (
                    self.validation_epochs > 0
                    and self._current_epoch % self.validation_epochs == 0
                ):
                    self._validate_current_and_ema()

        with apply_ema_maybe(self._ema_optimizer):
            self.save_dcp_checkpoint(
                self.get_checkpoint_dir(self._current_step) + "_final"
            )
