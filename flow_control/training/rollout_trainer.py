"""The rollout -> advantage -> update loop shared by the RL trainers.

:class:`RolloutTrainerBase` owns everything that is method-independent: the
config surface, optimizer / scheduler, checkpoint state, the micro-update loop
and the outer epoch loop. Subclasses fill in how rollouts become train items
(:meth:`RolloutTrainerBase._build_train_plan`) and how a microbatch of items
becomes a loss (:meth:`RolloutTrainerBase._loss_batched`):
:class:`~flow_control.training.grpo.GrpoTrainer` replays the recorded rollout
steps, :class:`~flow_control.training.endpoint.EndpointTrainer` re-noises the
rollout endpoints.
"""

import os
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Self

import torch
from pydantic import ConfigDict, NonNegativeInt, PrivateAttr, model_validator
from rich.progress import Progress
from torch.profiler import record_function

from flow_control.adapters import ModelAdapter
from flow_control.rewards import Reward, _has_pairwise_child
from flow_control.samplers import Trajectory
from flow_control.utils import device as devutil
from flow_control.utils.logging import console, get_logger
from flow_control.utils.types import (
    OptimizerConfig,
    SchedulerConfig,
)

from .mixins import (
    CheckpointingMixin,
    MicrobatchTrainMixin,
    PendingRollouts,
    Rollout,
    RolloutMixin,
    TrainingPredictionMixin,
    ValidationMixin,
    distributed_main,
)
from .mixins.base import (
    load_model_state_dict,
    load_optimizer_state_dict,
    make_optimizer,
    model_state_dict,
    optimizer_state_dict,
    optimizer_step,
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
    rollout_sampler: Trajectory
    reward: Reward

    seed_checkpoint_dir: str
    resume_from_dir: str | None = None

    optimizer_config: OptimizerConfig = {"class_name": "AdamW", "lr": 3e-4}
    scheduler_config: SchedulerConfig = {"class_name": "ConstantLR", "factor": 1.0}

    num_inner_epochs: int = 1
    """Passes over each epoch's rollouts. Whether a pass draws fresh train
    items or reorders the same ones is up to :meth:`_build_train_plan`."""
    rollout_lookahead: NonNegativeInt = 0
    """Batches sampled ahead of training (``N``). ``0`` is the synchronous loop:
    sample, wait for rewards, train. With ``N > 0`` epoch ``k`` samples batch
    ``k + N`` before waiting for batch ``k``'s rewards, so the reward backend
    scores one batch while the accelerator produces the next; the price is that
    every batch trains under a policy ``N`` epochs newer than the one that
    sampled it (``rollout/policy_lag``)."""

    clip_grad_norm: float = 1.0
    """Gradient-norm clip before each optimizer step; ``0.0`` disables it."""

    # Optimization / training loop
    train_epochs: int = 100
    validation_epochs: int = 20

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
    _current_step: int = 0
    _current_epoch: int = 0
    _pending_rollouts: deque[PendingRollouts] = PrivateAttr(default_factory=deque)
    """Batches sampled but not yet trained on, oldest first."""

    @model_validator(mode="after")
    def check_validation_reward(self) -> Self:
        """Validation scores one image per prompt, so it cannot use a pairwise
        reward (also not the training reward it falls back to)."""
        if self.validation_dataset is None or not self.validation_log_rewards:
            return self
        metric_reward = (
            None
            if self.validation_reward is False
            else self.validation_reward or self.reward
        )
        if metric_reward is not None and _has_pairwise_child(metric_reward):
            raise ValueError(
                "A pairwise reward compares the rollouts of one prompt and cannot "
                "score validation samples; set validation_reward to a pointwise "
                "reward that compares each sample against an anchor (for example "
                "reference_compare in anchored mode), or validation_reward=false."
            )
        return self

    # ------------------------------- Properties --------------------------------- #

    @property
    def transformer(self):
        return self.model.transformer

    # --------------------------------- Hooks ------------------------------------ #

    def _rollout_variant(self) -> str:
        """The model variant that samples the rollouts."""
        return "default"

    def _required_variants(self) -> frozenset[str]:
        """Auxiliary variants the loss evaluates besides the current policy."""
        return frozenset()

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

    def _check_rollouts(self, rollouts: list[Rollout]) -> None:
        """Fail fast on rollouts the trainer cannot train on."""

    # ------------------------------- Setup methods ------------------------------ #

    def cleanup(self) -> None:
        # Runs on every exit of run(), an exception included: batches sampled
        # ahead keep issuing reward requests until their loop is stopped.
        self.close_reward_loop()
        super().cleanup()

    def make_optimizer_and_scheduler(self):
        self._optimizer, self._scheduler = make_optimizer(
            self.model, self.optimizer_config, self.scheduler_config
        )

    # ------------------------------- Checkpointing ------------------------------ #

    def state_dict(self):
        return {
            "transformer": model_state_dict(self.model),
            "optimizer": optimizer_state_dict(self.model, self._optimizer),
            # In-flight batches are dropped on resume and sampled again, so the
            # cursor must rewind to before the oldest of them took its prompts.
            "cursor": (
                self._pending_rollouts[0].cursor_state_before
                if self._pending_rollouts
                else self._cursor.state_dict()
            ),
            "scheduler": self._scheduler.state_dict(),
            "current_step": self._current_step,
            "current_epoch": self._current_epoch,
            "rng": self.get_rng_state_bytes(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        load_model_state_dict(self.model, state_dict["transformer"])
        load_optimizer_state_dict(self.model, self._optimizer, state_dict["optimizer"])
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

    # ----------------------------- Training phase ------------------------------- #

    def _optimizer_step(self):
        optimizer_step(
            self.model, self._optimizer, self._scheduler, self.clip_grad_norm
        )

    def _train_on_rollouts(self, rollouts: list[Rollout], advantages: torch.Tensor):
        """Training phase: update model using collected rollouts and advantages."""
        self.transformer.train()
        train_plan = self._build_train_plan(rollouts)
        total_items = sum(len(items) for items in train_plan)

        progress = Progress(
            *self.get_progress_columns(),
            console=console,
            transient=True,
        )
        train_task = progress.add_task("Training", total=total_items)

        with progress, record_function("phase/train"):
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

    def _validate(self) -> None:
        self.validate_variants(
            self.model,
            self._current_step,
            reward=self.reward,
            reward_loop=self.reward_loop,
        )

    def _launch_rollouts(self, epoch: int) -> None:
        """Sample batch *epoch* under the rollout variant and queue it behind the
        batches already in flight."""
        with self.model.use_variant(self._rollout_variant()):
            self._pending_rollouts.append(self._start_rollouts(epoch))

    def _prime_rollouts(self) -> None:
        """Sample the first ``rollout_lookahead`` batches under the current
        weights. A resume starts here too, re-sampling whatever was in flight."""
        first, ahead = self._current_epoch, self.rollout_lookahead
        for epoch in range(first, min(first + ahead, self.train_epochs)):
            self._launch_rollouts(epoch)

    def _run_epoch(self) -> None:
        """One outer epoch: top up the lookahead, then wait for the oldest
        batch's rewards and train on it. The ``phase/*`` ranges mark the
        phases in a profiler trace."""
        epoch = self._current_epoch
        with self.profile_scope(f"epoch{epoch}", epoch=epoch):
            logger.debug(f"Epoch {epoch}: starting rollout phase...")
            if epoch + self.rollout_lookahead < self.train_epochs:
                with record_function("phase/rollout"):
                    self._launch_rollouts(epoch + self.rollout_lookahead)
            pending = self._pending_rollouts.popleft()
            with record_function("phase/reward_wait"):
                rollouts = self._finish_rollouts(pending)
            self._check_rollouts(rollouts)
            advantages = self._compute_advantages(rollouts, step=self._current_step)

            logger.debug(f"Epoch {epoch}: starting training phase...")
            self._train_on_rollouts(rollouts, advantages)
            self.model.update_variants("epoch")

        self._current_epoch += 1

    @distributed_main
    def run(self):
        for name in {
            self._rollout_variant(),
            *self._required_variants(),
            *self.validation_variants,
        }:
            self.model.resolve(name)
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

        if self.validation_at_start:
            self._validate()

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
            self._prime_rollouts()
            while self._current_epoch < self.train_epochs:
                # The epoch's rollouts and advantages are released on return.
                self._run_epoch()
                progress.update(task, completed=self._current_epoch)
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
                    self._validate()

        self.save_dcp_checkpoint(self.get_checkpoint_dir(self._current_step) + "_final")
