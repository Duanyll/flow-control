"""RAM (Reinforce Adjoint Matching) trainer.

RAM is a trajectory-free RL algorithm for flow-matching models.  Instead of
policy-gradients over the sampling chain (GRPO) or positive/negative prediction
interpolation (NFT), RAM regresses the policy velocity onto a closed-form target
derived from KL-regularized optimal control:

    x_t    = (1 - t) * x0 + t * eps
    target = v_ref(x_t) + m * A * ((eps - x0) - v_old(x_t))
    loss   = || v_theta(x_t) - sg[target] ||^2

where ``x0`` is a clean endpoint sampled by the (lagged) policy, ``A`` is the
group-relative advantage, ``m`` is ``reward_multiplier``, ``v_ref`` is the frozen
base model, and ``v_old`` is a lagged EMA of the policy.  There is no explicit KL
term: regularization is implicit through the ``v_ref`` anchor and the reward
scale.  All three velocity forwards are plain *conditional* passes (no CFG) even
though rollouts are generated with CFG.

Reference: arXiv:2605.10759, ``AndreasBergmeister/ram`` (``scripts/training_sd3.py``).

Two ``EMAOptimizer`` instances mirror the reference's lagged adapters:
- **old-EMA** (``_old_ema``): samples the rollout endpoints *and* supplies
  ``v_old`` in the loss target.  Stepped once per epoch.
- **validation-EMA** (``_ema_optimizer``): standard EMA for validation, stepped
  per gradient step.
"""

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

import torch
from pydantic import ConfigDict
from rich.progress import Progress, TaskID
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
    Rollout,
    RolloutMixin,
    ValidationMixin,
    distributed_main,
    trainer_registry,
)
from .weighting import PowerLawTimestepWeighting, TimestepWeighting

logger = get_logger(__name__)


@dataclass(slots=True)
class RamCachedTargets:
    timestep: torch.Tensor
    noise: torch.Tensor
    base_prediction: torch.Tensor
    old_prediction: torch.Tensor


@dataclass(slots=True)
class RamTrainItem:
    rollout_idx: int
    cached_targets: RamCachedTargets | None = None


@trainer_registry.register("ram")
class RamTrainer(RolloutMixin, ValidationMixin, CheckpointingMixin):
    model_config = ConfigDict(extra="forbid")
    training_type: str = "ram"

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
    train_batch_size: int = 4
    """
    How many (rollout, timestep) items should the optimizer see before each
    update step. Must be divisible by world_size.
    """

    # RAM-specific hyperparameters
    reward_multiplier: float = 100.0
    """The ``m`` in ``target = v_ref + m * A * ((eps - x0) - v_old)``. Doubles as
    the implicit KL knob: larger ``m`` = weaker regularization. Reference uses 100
    for GenEval/OCR and 1000 for PickScore."""
    adv_clip_max: float | None = None
    """Optional symmetric advantage clip. ``None`` (reference default) disables
    clipping; the normalized advantage enters the target unclipped."""

    num_train_timesteps: int = 8
    """``K`` = how many times each endpoint is reused, each with a fresh timestep
    and noise draw (reference ``num_loss_targets_per_sample``)."""
    timestep_weighting: TimestepWeighting = PowerLawTimestepWeighting(alpha=1.0)
    """Distribution the per-item training timestep is drawn from. RAM uses a
    power law ``p(t) ∝ t^alpha`` biased toward the noise end."""

    ema_old: EMAConfig = EMAConfig(
        decay=0.9, warmup=LinearRampWarmup(flat_steps=0, ramp_rate=0.01)
    )
    """Old/lagged EMA config (stepped once per epoch): samples endpoints and
    supplies ``v_old`` in the loss target."""
    ema: EMAConfig | None = None
    """Validation EMA config (stepped per gradient step)."""
    precompute_aux_model_outputs: bool = False
    """
    Precompute base/old velocities once per outer epoch and reuse them during
    optimization. Reduces repeated model switching at the cost of extra
    accelerator memory for cached per-item tensors.
    """

    clip_grad_norm: float = 0.0
    """Reference RAM does not clip gradients (0.0 disables clipping)."""

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
    _rollout_needs_trajectory: bool = False

    # ------------------------------- Lazy state --------------------------------- #
    _optimizer: torch.optim.Optimizer
    _scheduler: Any
    _ema_optimizer: EMAOptimizer | None = None
    _old_ema: EMAOptimizer
    _init_backup_optimizer: InitBackupOptimizer | None = None
    _current_step: int = 0
    _current_epoch: int = 0

    # ------------------------------- Properties --------------------------------- #

    @property
    def transformer(self):
        return self.model.transformer

    @property
    def grad_acc_steps(self) -> int:
        world_size: int = getattr(self, "world_size", 1)
        if self.train_batch_size % world_size != 0:
            raise ValueError(
                f"global_batch_size ({self.train_batch_size}) must be divisible "
                f"by world_size ({world_size})."
            )
        return self.train_batch_size // world_size

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

        # Old/lagged EMA (stepped once per epoch)
        self._old_ema = EMAOptimizer(params, self.ema_old)
        logger.info(
            f"Old/lagged EMA created (decay={self.ema_old.decay}, "
            f"warmup={self.ema_old.warmup.type})."
        )

        # Validation EMA (stepped per gradient step)
        if self.ema is not None:
            self._ema_optimizer = EMAOptimizer(params, self.ema)

        # Reference (frozen base) model. RAM always needs ``v_ref``; for LoRA the
        # reference is reached by disabling adapters, so a backup is only required
        # for full fine-tuning.
        need_init = self.model.peft_lora_rank == 0
        if need_init:
            self._init_backup_optimizer = InitBackupOptimizer(params)
            logger.info("Init backup enabled for reference (base) model.")

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
        state["optim_ema_old"] = get_optimizer_state_dict(
            self.transformer, self._old_ema, options=opts
        )
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
        if "optim_ema_old" in state_dict:
            set_optimizer_state_dict(
                self.transformer,
                self._old_ema,
                state_dict["optim_ema_old"],
                options=opts,
            )
            self._old_ema.coerce_buffer_dtype()
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
        """Temporarily switch to reference (frozen base) model weights."""
        if self.model.peft_lora_rank > 0:
            self.transformer.disable_adapters()
            try:
                yield
            finally:
                self.transformer.enable_adapters()
        else:
            with apply_init_maybe(self._init_backup_optimizer):
                yield

    # ---------------------------------- Predict --------------------------------- #

    def _predict(self, batch: Any, t: torch.Tensor) -> torch.Tensor:
        """Plain *conditional* velocity prediction (no CFG).

        RAM's loss target is defined against conditional velocities even though
        rollouts are sampled with CFG, so this deliberately bypasses
        ``get_guided_velocity``.
        """
        return self.model.predict_velocity(batch, t.to(dtype=self.model.dtype)).float()

    def _sample_timestep(self) -> torch.Tensor:
        t = self.timestep_weighting.sample_timesteps(1)
        return t.to(device=self.device, dtype=torch.float32)

    # -------------------------------- RAM loss ---------------------------------- #

    def ram_loss(
        self,
        rollout: Rollout,
        rollout_advantages: torch.Tensor,
        cached_targets: RamCachedTargets | None = None,
    ) -> torch.Tensor:
        """Compute the RAM regression loss for one (rollout, timestep) item."""
        batch = deep_move_to_device(rollout.batch, self.device)
        x0 = batch["clean_latents"].float()
        base_prediction = torch.empty_like(x0)
        old_prediction = torch.empty_like(x0)

        if cached_targets is not None:
            t = cached_targets.timestep.to(device=self.device, dtype=torch.float32)
            noise = cached_targets.noise.to(device=self.device)
            base_prediction = cached_targets.base_prediction.to(device=self.device)
            old_prediction = cached_targets.old_prediction.to(device=self.device)
        else:
            t = self._sample_timestep()
            noise = torch.randn_like(x0)

        t_expanded = t.view(-1, *([1] * (x0.ndim - 1)))
        xt = (1.0 - t_expanded) * x0 + t_expanded * noise
        batch["noisy_latents"] = xt

        if cached_targets is None:
            with torch.no_grad(), self.reference_model():
                base_prediction = self._predict(batch, t).detach()
            with torch.no_grad(), apply_ema_maybe(self._old_ema):
                old_prediction = self._predict(batch, t).detach()

        # Current policy velocity (with grad)
        forward_prediction = self._predict(batch, t)

        # Advantage-scaled reward direction toward the flow-matching target.
        adv = rollout_advantages.to(device=self.device)
        if self.adv_clip_max is not None:
            adv = torch.clamp(adv, -self.adv_clip_max, self.adv_clip_max)
        scaled_adv = self.reward_multiplier * adv.view(-1, *([1] * (x0.ndim - 1)))

        reward_direction = noise - x0
        target = base_prediction + scaled_adv * (reward_direction - old_prediction)
        loss = ((forward_prediction - target.detach()) ** 2).mean()

        metrics: dict[str, torch.Tensor] = {
            "train/loss": loss.detach(),
            "train/target_norm": (target**2).mean().detach(),
            "train/base_deviate": ((forward_prediction - base_prediction) ** 2)
            .mean()
            .detach(),
        }
        self.log_aggregated_metrics(metrics)
        return loss

    # ----------------------------- Training phase ------------------------------- #

    def _build_inner_epoch_train_items(
        self, rollouts: list[Rollout]
    ) -> list[RamTrainItem]:
        train_items: list[RamTrainItem] = []
        for rollout_idx in torch.randperm(len(rollouts)).tolist():
            train_items.extend(
                RamTrainItem(rollout_idx=rollout_idx)
                for _ in range(self.num_train_timesteps)
            )
        return train_items

    def _build_train_plan(self, rollouts: list[Rollout]) -> list[list[RamTrainItem]]:
        return [
            self._build_inner_epoch_train_items(rollouts)
            for _ in range(self.num_inner_epochs)
        ]

    def _prepare_cached_targets(
        self,
        rollouts: list[Rollout],
        flat_items: list[RamTrainItem],
        progress: Progress,
        task_id: TaskID,
    ) -> None:
        for item in flat_items:
            rollout = rollouts[item.rollout_idx]
            batch = deep_move_to_device(rollout.batch, self.device)
            x0 = batch["clean_latents"].float()
            t = self._sample_timestep()
            noise = torch.randn_like(x0)
            item.cached_targets = RamCachedTargets(
                timestep=t.detach(),
                noise=noise.detach(),
                base_prediction=torch.empty_like(x0),
                old_prediction=torch.empty_like(x0),
            )
            progress.advance(task_id)

    def _precompute_predictions(
        self,
        rollouts: list[Rollout],
        flat_items: list[RamTrainItem],
        progress: Progress,
        task_id: TaskID,
        field: Literal["base_prediction", "old_prediction"],
    ) -> None:
        for item in flat_items:
            rollout = rollouts[item.rollout_idx]
            batch = deep_move_to_device(rollout.batch, self.device)
            x0 = batch["clean_latents"].float()
            cached_targets = item.cached_targets
            if cached_targets is None:
                raise RuntimeError("Missing cached RAM targets.")
            t = cached_targets.timestep.to(device=self.device, dtype=torch.float32)
            noise = cached_targets.noise.to(device=self.device)
            t_expanded = t.view(-1, *([1] * (x0.ndim - 1)))
            batch["noisy_latents"] = (1.0 - t_expanded) * x0 + t_expanded * noise
            prediction = self._predict(batch, t).detach()
            setattr(cached_targets, field, prediction)
            progress.advance(task_id)

    def _precompute_aux_model_outputs_for_plan(
        self,
        rollouts: list[Rollout],
        train_plan: list[list[RamTrainItem]],
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
        base_task = progress.add_task("Precompute base", total=len(flat_items))
        old_task = progress.add_task("Precompute old", total=len(flat_items))

        with progress, torch.no_grad():
            self._prepare_cached_targets(rollouts, flat_items, progress, prepare_task)
            with self.reference_model():
                self._precompute_predictions(
                    rollouts, flat_items, progress, base_task, field="base_prediction"
                )
            with apply_ema_maybe(self._old_ema):
                self._precompute_predictions(
                    rollouts, flat_items, progress, old_task, field="old_prediction"
                )

        if was_training:
            self.transformer.train()

        self.log_progress_timing(
            progress, self._current_step, prefix="profile/precompute"
        )

    def _optimizer_step(self):
        """Clip gradients (optional), step all optimizers (except old-EMA), zero."""
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
            self._precompute_aux_model_outputs_for_plan(rollouts, train_plan)

        progress = Progress(
            *self.get_progress_columns(),
            console=console,
            transient=True,
        )
        train_task = progress.add_task("Training", total=total_items)

        with progress:
            for train_items in train_plan:
                if len(train_items) == 0:
                    raise RuntimeError(
                        "No training items were selected in RAM inner epoch."
                    )

                if (
                    len(train_items) % self.grad_acc_steps != 0
                    and self.is_main_process
                    and self._current_epoch == 0
                ):
                    logger.warning(
                        "Local loss count (%d) is not divisible by "
                        "grad_acc_steps (%d). The tail update uses a smaller "
                        "effective batch.",
                        len(train_items),
                        self.grad_acc_steps,
                    )

                for chunk_start in range(0, len(train_items), self.grad_acc_steps):
                    chunk = train_items[chunk_start : chunk_start + self.grad_acc_steps]
                    chunk_size = len(chunk)

                    for micro_idx, item in enumerate(chunk):
                        is_sync_step = micro_idx == chunk_size - 1
                        self.transformer.set_requires_gradient_sync(is_sync_step)

                        loss = self.ram_loss(
                            rollout=rollouts[item.rollout_idx],
                            rollout_advantages=advantages[item.rollout_idx],
                            cached_targets=item.cached_targets,
                        )

                        if not torch.isfinite(loss):
                            raise RuntimeError(
                                f"Non-finite RAM loss detected: {loss.item()}. "
                                f"(rollout_idx={item.rollout_idx})"
                            )

                        (loss / chunk_size).backward()
                        progress.advance(train_task)

                    self._optimizer_step()
                    self._current_step += 1
                    self.flush_aggregated_metrics(self._current_step)

        self.log_progress_timing(progress, self._current_step, prefix="profile/train")

    # -------------------------------- Main loop --------------------------------- #

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

        with apply_ema_maybe(self._ema_optimizer):
            self.validate_and_log(self.model, self._current_step, reward=self.reward)

        logger.info(
            "RAM rollouts in each epoch will randomly select %d unique prompts "
            "for %d times, and generate %d rollouts for each prompt. That is "
            "%d rollouts in total (may have duplicates across batches).",
            self.num_prompts_per_batch,
            self.num_batches_per_epoch,
            self.num_rollouts_per_prompt,
            self.num_batches_per_epoch
            * self.num_prompts_per_batch
            * self.num_rollouts_per_prompt,
        )
        logger.info(
            "RAM optimization uses train_batch_size=%d, world_size=%d, "
            "grad_acc_steps=%d.",
            self.train_batch_size,
            self.world_size,
            self.grad_acc_steps,
        )

        progress = Progress(
            *self.get_progress_columns(),
            console=console,
        )
        task = progress.add_task(
            "RAM Training", total=self.train_epochs, completed=self._current_epoch
        )

        with self.status_bar("RAM Training"), progress:
            while self._current_epoch < self.train_epochs:
                logger.debug(f"Epoch {self._current_epoch}: starting rollout phase...")
                # Endpoints are sampled by the lagged policy (old-EMA), with CFG
                # driven by ``rollout_sampler.cfg_scale``.
                with apply_ema_maybe(self._old_ema):
                    rollouts = self._collect_rollouts(self._current_epoch)
                advantages = self._compute_advantages(rollouts, step=self._current_step)

                logger.debug(f"Epoch {self._current_epoch}: starting training phase...")
                self._train_on_rollouts(rollouts, advantages)

                # Step old/lagged EMA once per epoch
                self._old_ema.step()

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
                    with apply_ema_maybe(self._ema_optimizer):
                        self.validate_and_log(
                            self.model, self._current_step, reward=self.reward
                        )

        with apply_ema_maybe(self._ema_optimizer):
            self.save_dcp_checkpoint(
                self.get_checkpoint_dir(self._current_step) + "_final"
            )
