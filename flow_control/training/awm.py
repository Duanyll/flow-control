"""AWM (Advantage Weighted Matching) trainer.

AWM reframes diffusion RL so the policy-gradient objective *is* the pretraining
flow-matching loss, weighted by the (group-relative) advantage.  For a clean
endpoint ``x0`` sampled by the policy, re-noise it at a training timestep
``x_t = (1 - t) * x0 + t * eps`` and evaluate the *conditional* velocity (no
CFG).  The flow-matching log-likelihood surrogate is

    log_p(x0) ∝ - w(t) * || v_theta(x_t) - (eps - x0) ||^2

and the GRPO-style loss is

    ratio       = exp(log_p - sg[log_p_old])
    policy_loss = mean( max(-A * ratio, -A * clip(ratio, 1-eps, 1+eps)) )
    loss        = policy_loss
                + beta     * || v_theta - v_ref ||^2      (KL to frozen base)
                + ema_beta * || v_theta - v_ema ||^2      (TRPO-style KL to lagged EMA)

On-policy (the default), ``log_p_old = sg[log_p]`` so the ratio is 1 in value and
its gradient reduces to the advantage-weighted flow-matching gradient.  Off-policy
draws ``log_p_old`` (and the rollout endpoints) from the lagged EMA policy.

Reference: arXiv:2509.25050, ``scxue/advantage_weighted_matching``
(``scripts/train_sd3_awm.py``).

Two ``EMAOptimizer`` instances mirror the reference's EMA machinery:
- **TRPO-EMA** (``_old_ema``): supplies ``v_ema`` for the TRPO KL term (and the
  off-policy behaviour policy).  Stepped once per epoch, ``decay = min(0.3,
  0.001 * step)``.
- **validation-EMA** (``_ema_optimizer``): standard EMA (decay 0.99) for
  validation, stepped per gradient step.
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
from flow_control.utils.logging import console, get_logger, warn_once
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
from .weighting import LogitNormalTimestepWeighting, TimestepWeighting

logger = get_logger(__name__)

AwmWeighting = Literal["uniform", "t", "t2", "huber", "ghuber"]
AwmKlWeighting = Literal["uniform", "elbo"]


@dataclass(slots=True)
class AwmCachedTargets:
    noise: torch.Tensor
    ref_prediction: torch.Tensor | None = None
    ema_prediction: torch.Tensor | None = None


@dataclass(slots=True)
class AwmTrainItem:
    rollout_idx: int
    timestep: torch.Tensor
    cached_targets: AwmCachedTargets | None = None


@trainer_registry.register("awm")
class AwmTrainer(RolloutMixin, ValidationMixin, CheckpointingMixin):
    model_config = ConfigDict(extra="forbid")
    training_type: str = "awm"

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

    # AWM-specific hyperparameters
    beta: float = 0.001
    """KL-to-reference (frozen base) coefficient."""
    ema_beta: float = 1.0
    """TRPO-style KL-to-EMA coefficient. ``0`` disables the term."""
    adv_clip_max: float = 5.0
    advantage_max: float = 1.0
    """Advantage is clipped to ``[-adv_clip_max, adv_clip_max]`` then rescaled so
    its magnitude is bounded by ``advantage_max``."""
    clip_range: float = 1.0
    """PPO ratio clip epsilon (reference uses 1.0, i.e. effectively unclipped)."""

    weighting: AwmWeighting = "ghuber"
    """Flow-matching log-prob weighting ``w(t)``. ``ghuber`` is the reference
    default; the paper reports ``uniform`` works best."""
    ghuber_power: float = 0.25
    kl_weight: AwmKlWeighting = "uniform"
    kl_ema_weight: AwmKlWeighting = "uniform"

    num_train_timesteps: int = 6
    """Training timesteps drawn per endpoint."""
    timestep_fraction: float = 0.9
    """For discrete sampling, restrict indices to ``[lo, steps * fraction]``."""
    train_timestep_sampling: Literal["discrete_wo_init", "discrete", "weighting"] = (
        "discrete_wo_init"
    )
    """``discrete_wo_init``/``discrete`` draw stratified indices from the rollout
    sigma grid (excluding / including the pure-noise index 0). ``weighting`` draws
    continuous timesteps from :pyattr:`timestep_weighting`."""
    timestep_weighting: TimestepWeighting = LogitNormalTimestepWeighting()
    """Used only when ``train_timestep_sampling == "weighting"``."""

    off_policy: bool = False
    """Sample rollout endpoints and the ratio denominator from the lagged EMA
    policy instead of the current policy."""

    ema_old: EMAConfig = EMAConfig(
        decay=0.3, warmup=LinearRampWarmup(flat_steps=0, ramp_rate=0.001)
    )
    """TRPO-EMA config (stepped once per epoch)."""
    ema: EMAConfig | None = None
    """Validation EMA config (stepped per gradient step)."""
    precompute_aux_model_outputs: bool = False
    """
    Precompute reference/EMA velocities once per outer epoch and reuse them
    during optimization, at the cost of extra accelerator memory.
    """

    clip_grad_norm: float = 1.0

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

    @property
    def _needs_ema_prediction(self) -> bool:
        return self.ema_beta > 0 or self.off_policy

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

        # TRPO-EMA (stepped once per epoch)
        self._old_ema = EMAOptimizer(params, self.ema_old)
        logger.info(
            f"TRPO-EMA created (decay={self.ema_old.decay}, "
            f"warmup={self.ema_old.warmup.type})."
        )

        # Validation EMA (stepped per gradient step)
        if self.ema is not None:
            self._ema_optimizer = EMAOptimizer(params, self.ema)

        # Reference (frozen base) model for the KL-to-ref term.
        need_init = self.beta > 0 and self.model.peft_lora_rank == 0
        if need_init:
            self._init_backup_optimizer = InitBackupOptimizer(params)
            logger.info("Init backup enabled for reference model (beta > 0).")

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

        AWM's flow-matching loss is defined against conditional velocities, so
        this deliberately bypasses ``get_guided_velocity`` even when rollouts use
        CFG (``off_policy`` still samples endpoints via the sampler's CFG path).
        """
        return self.model.predict_velocity(batch, t.to(dtype=self.model.dtype)).float()

    # ------------------------------- Loss helpers ------------------------------- #

    def _flow_matching_logp(
        self,
        velocity: torch.Tensor,
        noise: torch.Tensor,
        x0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted flow-matching log-likelihood surrogate, per sample ``[B]``."""
        target = noise - x0
        mse = ((velocity - target) ** 2).mean(dim=tuple(range(1, velocity.ndim)))
        t_flat = t.view(-1)
        p = self.ghuber_power
        eps = 1e-10
        if self.weighting == "uniform":
            return -mse
        if self.weighting == "t":
            return -mse * t_flat
        if self.weighting == "t2":
            return -mse * t_flat**2
        if self.weighting == "huber":
            return -(torch.sqrt(mse + eps) - 1e-5) * t_flat
        # ghuber
        base = torch.tensor(eps, device=mse.device, dtype=mse.dtype)
        return -(torch.pow(mse + eps, p) - torch.pow(base, p)) * t_flat / p

    def _kl_term(
        self,
        velocity: torch.Tensor,
        reference: torch.Tensor,
        t: torch.Tensor,
        mode: AwmKlWeighting,
    ) -> torch.Tensor:
        diff = ((velocity - reference) ** 2).mean(dim=tuple(range(1, velocity.ndim)))
        if mode == "elbo":
            sigma = t.view(-1)
            std_dev = torch.sqrt(sigma / (1 - torch.clamp(sigma, 0, 0.99))) * 0.7
            diff = diff / (2 * std_dev**2)
        return diff.mean()

    # -------------------------------- AWM loss ---------------------------------- #

    def awm_loss(
        self,
        rollout: Rollout,
        rollout_advantages: torch.Tensor,
        timestep: torch.Tensor,
        cached_targets: AwmCachedTargets | None = None,
    ) -> torch.Tensor:
        """Compute the AWM loss for one (rollout, timestep) item."""
        batch = deep_move_to_device(rollout.batch, self.device)
        x0 = batch["clean_latents"].float()
        t = timestep.to(device=self.device, dtype=torch.float32)

        if cached_targets is not None:
            noise = cached_targets.noise.to(device=self.device)
            ref_prediction = (
                cached_targets.ref_prediction.to(device=self.device)
                if cached_targets.ref_prediction is not None
                else None
            )
            ema_prediction = (
                cached_targets.ema_prediction.to(device=self.device)
                if cached_targets.ema_prediction is not None
                else None
            )
        else:
            noise = torch.randn_like(x0)
            ref_prediction = None
            ema_prediction = None

        t_expanded = t.view(-1, *([1] * (x0.ndim - 1)))
        xt = (1.0 - t_expanded) * x0 + t_expanded * noise
        batch["noisy_latents"] = xt

        if cached_targets is None:
            if self.beta > 0:
                with torch.no_grad(), self.reference_model():
                    ref_prediction = self._predict(batch, t).detach()
            if self._needs_ema_prediction:
                with torch.no_grad(), apply_ema_maybe(self._old_ema):
                    ema_prediction = self._predict(batch, t).detach()

        # Current policy velocity (with grad)
        forward_prediction = self._predict(batch, t)
        log_prob = self._flow_matching_logp(forward_prediction, noise, x0, t)

        if self.off_policy:
            if ema_prediction is None:
                raise RuntimeError("off_policy requires an EMA prediction.")
            old_log_prob = self._flow_matching_logp(ema_prediction, noise, x0, t)
        else:
            old_log_prob = log_prob.detach()
        ratio = torch.exp(log_prob - old_log_prob.detach())

        adv = rollout_advantages.to(device=self.device)
        adv = torch.clamp(adv, -self.adv_clip_max, self.adv_clip_max)
        adv = adv / self.adv_clip_max * self.advantage_max
        adv = adv.view(-1)

        unclipped_loss = -adv * ratio
        clipped_loss = -adv * torch.clamp(
            ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
        )
        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

        loss = policy_loss
        kl_loss = torch.tensor(0.0, device=self.device)
        ema_kl_loss = torch.tensor(0.0, device=self.device)

        if self.beta > 0 and ref_prediction is not None:
            kl_loss = self._kl_term(
                forward_prediction, ref_prediction, t, self.kl_weight
            )
            loss = loss + self.beta * kl_loss
        if self.ema_beta > 0 and ema_prediction is not None:
            ema_kl_loss = self._kl_term(
                forward_prediction, ema_prediction, t, self.kl_ema_weight
            )
            loss = loss + self.ema_beta * ema_kl_loss

        metrics: dict[str, torch.Tensor] = {
            "train/policy_loss": policy_loss.detach(),
            "train/kl_loss": kl_loss.detach(),
            "train/ema_kl_loss": ema_kl_loss.detach(),
            "train/loss": loss.detach(),
            "train/ratio_mean": ratio.detach().mean(),
            "train/clipfrac": (torch.abs(ratio - 1.0) > self.clip_range)
            .float()
            .mean()
            .detach(),
        }
        self.log_aggregated_metrics(metrics)
        return loss

    # ----------------------------- Timestep sampling ---------------------------- #

    def _sample_train_timesteps(self, rollout: Rollout) -> torch.Tensor:
        """Return ``num_train_timesteps`` training timesteps for one rollout."""
        T = self.num_train_timesteps
        if self.train_timestep_sampling == "weighting":
            return self.timestep_weighting.sample_timesteps(T).to(
                device=self.device, dtype=torch.float32
            )

        grid = rollout.trajectory.timesteps
        if grid is None:
            warn_once(
                logger,
                "Rollout sigma grid unavailable; falling back to timestep_weighting.",
            )
            return self.timestep_weighting.sample_timesteps(T).to(
                device=self.device, dtype=torch.float32
            )

        grid = grid.to(device=self.device, dtype=torch.float32)
        steps = grid.shape[0]
        lo = 1 if self.train_timestep_sampling == "discrete_wo_init" else 0
        hi = max(lo + 1, int(steps * self.timestep_fraction))
        boundaries = torch.linspace(lo, hi, T + 1, device=self.device)
        lower = boundaries[:-1].long()
        upper = boundaries[1:].long()
        rand_u = torch.rand(T, device=self.device)
        indices = lower + (rand_u * (upper - lower)).long()
        indices = indices.clamp(0, steps - 1)
        return grid[indices]

    # ----------------------------- Training phase ------------------------------- #

    def _build_inner_epoch_train_items(
        self, rollouts: list[Rollout]
    ) -> list[AwmTrainItem]:
        train_items: list[AwmTrainItem] = []
        for rollout_idx in torch.randperm(len(rollouts)).tolist():
            timesteps = self._sample_train_timesteps(rollouts[rollout_idx])
            train_items.extend(
                AwmTrainItem(
                    rollout_idx=rollout_idx,
                    timestep=timesteps[i : i + 1].detach(),
                )
                for i in range(self.num_train_timesteps)
            )
        return train_items

    def _build_train_plan(self, rollouts: list[Rollout]) -> list[list[AwmTrainItem]]:
        return [
            self._build_inner_epoch_train_items(rollouts)
            for _ in range(self.num_inner_epochs)
        ]

    def _prepare_cached_targets(
        self,
        rollouts: list[Rollout],
        flat_items: list[AwmTrainItem],
        progress: Progress,
        task_id: TaskID,
    ) -> None:
        for item in flat_items:
            rollout = rollouts[item.rollout_idx]
            batch = deep_move_to_device(rollout.batch, self.device)
            x0 = batch["clean_latents"].float()
            item.cached_targets = AwmCachedTargets(noise=torch.randn_like(x0).detach())
            progress.advance(task_id)

    def _precompute_predictions(
        self,
        rollouts: list[Rollout],
        flat_items: list[AwmTrainItem],
        progress: Progress,
        task_id: TaskID,
        field: Literal["ref_prediction", "ema_prediction"],
    ) -> None:
        for item in flat_items:
            rollout = rollouts[item.rollout_idx]
            batch = deep_move_to_device(rollout.batch, self.device)
            x0 = batch["clean_latents"].float()
            cached_targets = item.cached_targets
            if cached_targets is None:
                raise RuntimeError("Missing cached AWM targets.")
            t = item.timestep.to(device=self.device, dtype=torch.float32)
            noise = cached_targets.noise.to(device=self.device)
            t_expanded = t.view(-1, *([1] * (x0.ndim - 1)))
            batch["noisy_latents"] = (1.0 - t_expanded) * x0 + t_expanded * noise
            prediction = self._predict(batch, t).detach()
            setattr(cached_targets, field, prediction)
            progress.advance(task_id)

    def _precompute_aux_model_outputs_for_plan(
        self,
        rollouts: list[Rollout],
        train_plan: list[list[AwmTrainItem]],
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
            self._prepare_cached_targets(rollouts, flat_items, progress, prepare_task)
            if self.beta > 0:
                ref_task = progress.add_task("Precompute ref", total=len(flat_items))
                with self.reference_model():
                    self._precompute_predictions(
                        rollouts, flat_items, progress, ref_task, field="ref_prediction"
                    )
            if self._needs_ema_prediction:
                ema_task = progress.add_task("Precompute ema", total=len(flat_items))
                with apply_ema_maybe(self._old_ema):
                    self._precompute_predictions(
                        rollouts, flat_items, progress, ema_task, field="ema_prediction"
                    )

        if was_training:
            self.transformer.train()

        self.log_progress_timing(
            progress, self._current_step, prefix="profile/precompute"
        )

    def _optimizer_step(self):
        """Clip gradients, step all optimizers (except TRPO-EMA), and zero grads."""
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
                        "No training items were selected in AWM inner epoch."
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

                        loss = self.awm_loss(
                            rollout=rollouts[item.rollout_idx],
                            rollout_advantages=advantages[item.rollout_idx],
                            timestep=item.timestep,
                            cached_targets=item.cached_targets,
                        )

                        if not torch.isfinite(loss):
                            raise RuntimeError(
                                f"Non-finite AWM loss detected: {loss.item()}. "
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
            "AWM rollouts in each epoch will randomly select %d unique prompts "
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
            "AWM optimization uses train_batch_size=%d, world_size=%d, "
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
            "AWM Training", total=self.train_epochs, completed=self._current_epoch
        )

        with self.status_bar("AWM Training"), progress:
            while self._current_epoch < self.train_epochs:
                logger.debug(f"Epoch {self._current_epoch}: starting rollout phase...")
                # On-policy rollouts use the current model; off-policy uses the
                # lagged TRPO-EMA policy.
                if self.off_policy:
                    with apply_ema_maybe(self._old_ema):
                        rollouts = self._collect_rollouts(self._current_epoch)
                else:
                    rollouts = self._collect_rollouts(self._current_epoch)
                advantages = self._compute_advantages(rollouts, step=self._current_step)

                logger.debug(f"Epoch {self._current_epoch}: starting training phase...")
                self._train_on_rollouts(rollouts, advantages)

                # Step TRPO-EMA once per epoch
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
