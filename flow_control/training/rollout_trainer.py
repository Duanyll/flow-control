"""The rollout -> advantage -> update loop shared by the RL trainers.

Two layers:

- :class:`RolloutTrainerBase` owns everything that is method-independent:
  the config surface, optimizer / scheduler / validation EMA / reference
  backup, checkpoint state, the micro-update loop and the outer epoch loop.
  Subclasses fill in how rollouts become train items
  (:meth:`RolloutTrainerBase._build_train_plan`) and how a microbatch of items
  becomes a loss (:meth:`RolloutTrainerBase._loss_batched`).
- :class:`EndpointTrainer` is the endpoint family (NFT / RAM / AWM /
  weighted flow matching): the clean rollout endpoint is re-noised at a
  training timestep and an :class:`~flow_control.training.objective.Objective`
  turns the velocities of the current / old / reference policies there into a
  loss. The concrete trainers are presets over ``objective`` and
  ``train_timesteps``; their math lives on the objective classes.

Forward rule for endpoint items: a *grid* timestep (``grid_index`` set) is
evaluated through the rollout plan's own step
(``SampleRun.guided_velocity`` with ``train_predictor``), so per-step variant
schedules and CFG++ see the executed transition; a *continuous* timestep has no
transition and goes through ``train_predictor.velocity`` directly.
"""

import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
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
from flow_control.adapters.base import Batch
from flow_control.processors import Processor
from flow_control.rewards import Reward
from flow_control.samplers import Executor, Sampler, SampleRequest
from flow_control.samplers.calls import Calls
from flow_control.utils import device as devutil
from flow_control.utils.logging import console, get_logger
from flow_control.utils.tensor import deep_move_to_device
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
    LinearRampWarmup,
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
from .objective import Objective, PolicyRole, PolicyVelocities, TrainPoint
from .train_timesteps import TrainTimesteps

logger = get_logger(__name__)

POLICY_ROLE_ORDER: tuple[PolicyRole, ...] = ("old", "ref")
"""Auxiliary forwards run in this fixed order on every rank. Iterating the
objective's ``frozenset`` directly would follow per-process string hashing and
could desynchronize the collective forward sequence across ranks."""


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
    """Passes over each epoch's rollouts; every pass draws fresh train items."""

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
            "dataloader": self._dataloader.state_dict(),
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
        self._dataloader.load_state_dict(state_dict["dataloader"])
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
        self.make_rollout_dataloader()
        self.make_validation_dataloader()

        self.reward.load_model(self.device)
        if self.validation_reward:
            self.validation_reward.load_model(self.device)

        os.makedirs(self.checkpoint_root, exist_ok=True)
        self.maybe_auto_resume(self.resume_from_dir)

        self._validate_current_and_ema()

        name = self.training_type.upper()
        logger.info(
            "%s rollouts in each epoch will randomly select %d unique prompts "
            "for %d times, and generate %d rollouts for each prompt. That is "
            "%d rollouts in total (may have duplicates across batches).",
            name,
            self.num_prompts_per_batch,
            self.num_batches_per_epoch,
            self.num_rollouts_per_prompt,
            self.num_batches_per_epoch
            * self.num_prompts_per_batch
            * self.num_rollouts_per_prompt,
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


# =============================================================================== #
#                               Endpoint trainers                                 #
# =============================================================================== #


@dataclass(slots=True)
class EndpointTrainItem:
    rollout_idx: int
    sigma: float
    """Training timestep, ``1`` = pure noise."""
    grid_index: int | None
    """Index of ``sigma`` on the rollout plan, or ``None`` for a continuous draw."""
    noise: torch.Tensor | None = None
    """The re-noising ``eps``; drawn once by precompute so the cached velocities
    match the training input, otherwise fresh per loss evaluation."""
    cache: dict[str, torch.Tensor] = field(default_factory=dict)
    """Detached auxiliary velocities by policy role (``old`` / ``ref``)."""


@dataclass(slots=True)
class _Prepared:
    """One item on device: the batch already carries ``noisy_latents = x_t``."""

    item: EndpointTrainItem
    rollout: Rollout
    batch: Batch
    negative: Batch | None
    point: TrainPoint


class EndpointTrainer(RolloutTrainerBase[EndpointTrainItem]):
    """Re-noise rollout endpoints and train an :class:`Objective` on them.

    ``objective`` decides which policy samples the rollouts and which auxiliary
    velocities the loss needs; ``train_timesteps`` decides where each endpoint
    is re-noised. The old policy is a lagged EMA (``ema_old``) stepped once per
    outer epoch; the reference policy is the frozen initial weights.
    """

    objective: Objective
    train_timesteps: TrainTimesteps
    ema_old: EMAConfig = EMAConfig(
        decay=0.5, warmup=LinearRampWarmup(flat_steps=0, ramp_rate=0.001)
    )
    """Old / lagged policy EMA (stepped once per outer epoch)."""

    _old_ema: EMAOptimizer

    # --------------------------------- Hooks ------------------------------------ #

    def _needs_reference(self) -> bool:
        return "ref" in self.objective.required_policies()

    def _make_aux_optimizers(self, params: list[torch.nn.Parameter]) -> None:
        self._old_ema = EMAOptimizer(params, self.ema_old)
        logger.info(
            f"Old-policy EMA created (decay={self.ema_old.decay}, "
            f"warmup={self.ema_old.warmup.type})."
        )

    def _aux_state_dict(self, opts: StateDictOptions) -> dict[str, Any]:
        return {
            "optim_ema_old": get_optimizer_state_dict(
                self.transformer, self._old_ema, options=opts
            )
        }

    def _load_aux_state_dict(
        self, state_dict: dict[str, Any], opts: StateDictOptions
    ) -> None:
        if "optim_ema_old" in state_dict:
            set_optimizer_state_dict(
                self.transformer,
                self._old_ema,
                state_dict["optim_ema_old"],
                options=opts,
            )
            self._old_ema.coerce_buffer_dtype()

    def _rollout_scope(self) -> AbstractContextManager[None]:
        if self.objective.rollout_policy() == "old":
            return apply_ema_maybe(self._old_ema)
        return nullcontext()

    def _after_train_epoch(self) -> None:
        self._old_ema.step()

    def _policy_scope(self, role: PolicyRole) -> AbstractContextManager[None]:
        if role == "old":
            return apply_ema_maybe(self._old_ema)
        return self.reference_model()

    # ------------------------------- Train plan --------------------------------- #

    def _build_train_plan(
        self, rollouts: list[Rollout]
    ) -> list[list[EndpointTrainItem]]:
        # Resolve the scalar plan data once per rollout. For device-backed
        # storage this is one host transfer here, never one sync per model eval.
        rollout_sigmas = [
            [transition.sigma for transition in rollout.sampling_plan]
            for rollout in rollouts
        ]
        train_plan: list[list[EndpointTrainItem]] = []
        for _ in range(self.num_inner_epochs):
            items: list[EndpointTrainItem] = []
            for rollout_idx in torch.randperm(len(rollouts)).tolist():
                items.extend(
                    EndpointTrainItem(rollout_idx, sigma, grid_index)
                    for sigma, grid_index in self.train_timesteps.draw(
                        rollout_sigmas[rollout_idx]
                    )
                )
            train_plan.append(items)
        return train_plan

    # ------------------------------- Prediction --------------------------------- #

    def _make_point(
        self, item: EndpointTrainItem, batch: Batch, advantage: torch.Tensor
    ) -> TrainPoint:
        x0 = batch["clean_latents"].float()
        noise = (
            torch.randn_like(x0)
            if item.noise is None
            else item.noise.to(device=self.device, dtype=torch.float32)
        )
        return TrainPoint(
            x0=x0,
            noise=noise,
            t=torch.tensor([item.sigma], device=self.device, dtype=torch.float32),
            advantage=advantage.to(device=self.device, dtype=torch.float32).view(1),
            grid_index=item.grid_index,
        )

    def _prepare(
        self,
        items: list[EndpointTrainItem],
        rollouts: list[Rollout],
        advantages: torch.Tensor,
    ) -> list[_Prepared]:
        prepared: list[_Prepared] = []
        for item in items:
            rollout = rollouts[item.rollout_idx]
            batch = deep_move_to_device(rollout.batch, self.device)
            point = self._make_point(item, batch, advantages[item.rollout_idx])
            batch["noisy_latents"] = point.xt
            prepared.append(
                _Prepared(
                    item,
                    rollout,
                    batch,
                    deep_move_to_device(rollout.negative_batch, self.device),
                    point,
                )
            )
        return prepared

    def _predict(self, prepared: list[_Prepared]) -> list[torch.Tensor]:
        """``train_predictor`` velocities at every point, in one collective pass.

        Grid points run through the rollout plan's step so the predictor sees
        the executed transition; continuous points are independent evaluations.
        """
        gens: list[Calls[torch.Tensor]] = []
        variants: dict[str | None, None] = {}
        for entry in prepared:
            batch, negative, point = entry.batch, entry.negative, entry.point
            if point.grid_index is None:
                gens.append(
                    self.train_predictor.velocity(
                        batch,
                        point.t,
                        self.training_negative(batch, negative_batch=negative),
                    )
                )
                variants.update(dict.fromkeys(self.train_predictor.variant_keys(1)))
                continue
            plan = entry.rollout.sampling_plan
            run = self.rollout_sampler.make_run(
                SampleRequest(
                    batch=batch,
                    negative_batch=self.training_negative(batch, len(plan), negative),
                ),
                plan=plan,
                predictor=self.train_predictor,
            )
            gens.append(run.guided_velocity(point.xt, point.grid_index))
            variants.update(dict.fromkeys(self.train_predictor.variant_keys(len(plan))))
        return Executor(self.model, list(variants) or [None]).evaluate(gens)

    def _required_roles(self) -> Iterator[PolicyRole]:
        required = self.objective.required_policies()
        return (role for role in POLICY_ROLE_ORDER if role in required)

    def _loss_batched(
        self,
        items: list[EndpointTrainItem],
        rollouts: list[Rollout],
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        prepared = self._prepare(items, rollouts, advantages)
        aux: dict[PolicyRole, list[torch.Tensor]] = {}
        for role in self._required_roles():
            if all(role in entry.item.cache for entry in prepared):
                aux[role] = [
                    entry.item.cache[role].to(device=self.device, dtype=torch.float32)
                    for entry in prepared
                ]
            else:
                with torch.no_grad(), self._policy_scope(role):
                    aux[role] = [v.detach() for v in self._predict(prepared)]

        losses: list[torch.Tensor] = []
        for index, (entry, current) in enumerate(
            zip(prepared, self._predict(prepared), strict=True)
        ):
            out = self.objective.compute(
                entry.point,
                PolicyVelocities(
                    current=current,
                    old=aux["old"][index] if "old" in aux else None,
                    ref=aux["ref"][index] if "ref" in aux else None,
                ),
            )
            self.log_aggregated_metrics(
                {f"train/{key}": value for key, value in out.metrics.items()}
            )
            losses.append(out.loss)
        return torch.stack(losses).mean()

    # ------------------------------- Precompute --------------------------------- #

    def _precompute(
        self,
        rollouts: list[Rollout],
        train_plan: list[list[EndpointTrainItem]],
        advantages: torch.Tensor,
    ) -> None:
        flat_items = [item for items in train_plan for item in items]
        if len(flat_items) == 0:
            return

        was_training = self.transformer.training
        self.transformer.eval()
        progress = Progress(
            *self.get_progress_columns(),
            console=console,
            transient=True,
        )
        prepare_task = progress.add_task("Prepare cache", total=len(flat_items))

        with progress, torch.no_grad():
            # Draw the noise once so every cached velocity sees the exact x_t
            # the update loop will train on.
            for item in flat_items:
                item.noise = torch.randn_like(
                    rollouts[item.rollout_idx].batch["clean_latents"],
                    device=self.device,
                    dtype=torch.float32,
                )
                progress.advance(prepare_task)
            for role in self._required_roles():
                task = progress.add_task(f"Precompute {role}", total=len(flat_items))
                with self._policy_scope(role):
                    for micro_items in self.iter_train_micro_batches(flat_items):
                        prepared = self._prepare(micro_items, rollouts, advantages)
                        velocities = self._predict(prepared)
                        for item, velocity in zip(micro_items, velocities, strict=True):
                            item.cache[role] = velocity.detach()
                        progress.advance(task, advance=len(micro_items))

        if was_training:
            self.transformer.train()

        self.log_progress_timing(
            progress, self._current_step, prefix="profile/precompute"
        )
