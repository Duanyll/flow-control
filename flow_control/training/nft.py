"""Diffusion NFT (Negative-aware FineTuning) trainer.

NFT is an RL variant for diffusion models that avoids log-probability
computation.  Instead of PPO-style policy gradients, NFT uses the forward
(noising) process with implicit positive/negative objectives weighted by
normalized advantages.

Two ``EMAOptimizer`` instances are used:
- **old-EMA** (``_old_ema``): teacher model, stepped once per epoch.
- **validation-EMA** (``_ema_optimizer``, optional): standard EMA for
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
from torchdata.stateful_dataloader import StatefulDataLoader

from flow_control.adapters import ModelAdapter
from flow_control.processors import Processor
from flow_control.rewards import Reward
from flow_control.samplers import Sampler
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
)
from .weighting import LogitNormalTimestepWeighting, TimestepWeighting

logger = get_logger(__name__)


@dataclass(slots=True)
class NftCachedTargets:
    timestep: torch.Tensor
    noisy_latents: torch.Tensor
    old_prediction: torch.Tensor
    ref_prediction: torch.Tensor | None = None


@dataclass(slots=True)
class NftTrainItem:
    rollout_idx: int
    timestep_idx: int
    cached_targets: NftCachedTargets | None = None


class NftTrainer(RolloutMixin, ValidationMixin, CheckpointingMixin):
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
    train_batch_size: int = 4
    """
    How many (rollout, timestep) items should the optimizer see before each
    update step. Must be divisible by world_size.
    """

    # NFT-specific hyperparameters
    beta: float = 1.0
    """Positive/negative prediction interpolation weight."""
    kl_beta: float = 0.01
    """KL (MSE) loss coefficient for reference model regularisation."""
    adv_clip_max: float = 5.0
    adv_mode: Literal["all", "positive_only", "negative_only", "binary"] = "all"
    """Optional advantage clipping mode."""

    timestep_weighting: TimestepWeighting = LogitNormalTimestepWeighting()
    """Legacy fallback for sampling timesteps when rollout timesteps are unavailable."""
    num_train_timesteps: int | None = None
    """Timesteps per sample per inner epoch. ``None`` means derive from rollout
    timesteps and ``timestep_fraction``."""
    timestep_fraction: float = 1.0
    """Fraction of rollout timesteps to train on for each sample."""

    ema_old: EMAConfig = EMAConfig(
        decay=0.5, warmup=LinearRampWarmup(flat_steps=0, ramp_rate=0.001)
    )
    """Old-teacher EMA config (stepped once per epoch)."""
    ema: EMAConfig | None = None
    """Validation EMA config (stepped per gradient step)."""
    precompute_aux_model_outputs: bool = False
    """
    Precompute old/reference model outputs once per outer epoch and reuse them
    during optimization. This reduces repeated model switching at the cost of
    extra accelerator memory for cached per-item tensors.
    """

    clip_grad_norm: float = 1.0

    # Optimization / training loop
    train_epochs: int = 100
    checkpoint_epochs: int = 5
    validation_epochs: int = 20

    # --------------------------------- Status bar ------------------------------- #
    _status_fields: dict[str, str] = {
        "reward/mean": "R̄: {v:.3f}",
        "reward/std": "σ: {v:.3f}",
        "train/loss": "Loss: {v:.4f}",
        "val/reward_mean": "Val R̄: {v:.3f}",
    }
    _rollout_needs_trajectory: bool = False

    # ------------------------------- Lazy state --------------------------------- #
    _optimizer: torch.optim.Optimizer | None = None
    _scheduler: Any = None
    _ema_optimizer: EMAOptimizer | None = None
    _old_ema: EMAOptimizer | None = None
    _init_backup_optimizer: InitBackupOptimizer | None = None
    _current_step: int = 0
    _current_epoch: int = 0

    # ------------------------------- Properties --------------------------------- #

    @property
    def transformer(self):
        return self.model.transformer

    @property
    def dataloader(self) -> StatefulDataLoader:
        if self._dataloader is None:
            raise RuntimeError("Dataloader not created yet.")
        return self._dataloader

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        if self._optimizer is None:
            raise RuntimeError("Optimizer not created yet.")
        return self._optimizer

    @property
    def scheduler(self):
        if self._scheduler is None:
            raise RuntimeError("Scheduler not created yet.")
        return self._scheduler

    @property
    def current_step(self) -> int:
        return self._current_step

    @current_step.setter
    def current_step(self, value: int):
        self._current_step = value

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    @current_epoch.setter
    def current_epoch(self, value: int):
        self._current_epoch = value

    @property
    def old_ema(self) -> EMAOptimizer:
        if self._old_ema is None:
            raise RuntimeError("Old EMA optimizer not created yet.")
        return self._old_ema

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
        self._scheduler = parse_scheduler(self.scheduler_config, self.optimizer)

        # Old-teacher EMA (stepped once per epoch)
        self._old_ema = EMAOptimizer(params, self.ema_old)
        logger.info(
            f"Old-teacher EMA created (decay={self.ema_old.decay}, "
            f"warmup={self.ema_old.warmup.type})."
        )

        # Validation EMA (stepped per gradient step)
        if self.ema is not None:
            self._ema_optimizer = EMAOptimizer(params, self.ema)

        # Reference model (frozen initial weights)
        need_init = self.kl_beta > 0 and self.model.peft_lora_rank == 0
        if need_init:
            self._init_backup_optimizer = InitBackupOptimizer(params)
            logger.info("Init backup enabled for reference model (kl_beta > 0).")

    # ------------------------------- Checkpointing ------------------------------ #

    def state_dict(self):
        opts = StateDictOptions(strict=False, ignore_frozen_params=True)
        transformer_sd = get_model_state_dict(self.transformer, options=opts)
        if len(transformer_sd) == 0:
            raise RuntimeError("Nothing to save in transformer state dict.")
        state: dict[str, Any] = {
            "transformer": transformer_sd,
            "optimizer": get_optimizer_state_dict(
                self.transformer, self.optimizer, options=opts
            ),
            "dataloader": self.dataloader.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "current_step": self.current_step,
            "current_epoch": self.current_epoch,
        }
        state["optim_ema_old"] = get_optimizer_state_dict(
            self.transformer, self.old_ema, options=opts
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
            self.optimizer,
            state_dict["optimizer"],
            options=opts,
        )
        if "optim_ema_old" in state_dict:
            set_optimizer_state_dict(
                self.transformer,
                self.old_ema,
                state_dict["optim_ema_old"],
                options=opts,
            )
        if self._ema_optimizer is not None and "optim_ema" in state_dict:
            set_optimizer_state_dict(
                self.transformer,
                self._ema_optimizer,
                state_dict["optim_ema"],
                options=opts,
            )
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
        self.dataloader.load_state_dict(state_dict["dataloader"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.current_step = state_dict["current_step"]
        self.current_epoch = state_dict.get("current_epoch", 0)

    # ------------------------------- Reference model ---------------------------- #

    @contextmanager
    def reference_model(self):
        """Temporarily switch to reference model weights."""
        if self.model.peft_lora_rank > 0:
            with self.transformer.disable_adapter():
                yield
        else:
            with apply_init_maybe(self._init_backup_optimizer):
                yield

    def _resolve_training_timestep(
        self,
        rollout: Rollout,
        timestep_idx: int,
    ) -> torch.Tensor:
        rollout_timesteps = rollout.trajectory.timesteps
        if rollout_timesteps is None:
            t = self.timestep_weighting.sample_timesteps(1)
            return t.to(device=self.device, dtype=torch.float32)

        if timestep_idx < 0 or timestep_idx >= rollout_timesteps.shape[0]:
            raise IndexError(
                f"Timestep index {timestep_idx} is out of range for rollout "
                f"with {rollout_timesteps.shape[0]} timesteps."
            )
        return rollout_timesteps[timestep_idx : timestep_idx + 1].to(
            device=self.device, dtype=torch.float32
        )

    # -------------------------------- NFT loss ---------------------------------- #

    def nft_loss(
        self,
        rollout: Rollout,
        rollout_advantages: torch.Tensor,
        timestep_idx: int,
        cached_targets: NftCachedTargets | None = None,
    ) -> torch.Tensor:
        """Compute NFT loss for one (rollout, timestep) item.

        1. Select a rollout timestep and sample noise to create the noisy input.
        2. Get predictions from old-teacher, current model, and reference model.
        3. Normalise advantages, build positive/negative predictions.
        4. Compute adaptive-weighted MSE loss + KL regularisation.
        """
        batch = deep_move_to_device(rollout.batch, self.device)
        negative_batch = (
            deep_move_to_device(rollout.negative_batch, self.device)
            if rollout.negative_batch is not None
            else None
        )
        x0 = batch["clean_latents"].float()
        old_prediction = torch.empty_like(x0)
        ref_prediction: torch.Tensor | None = None

        if cached_targets is not None:
            t = cached_targets.timestep.to(device=self.device, dtype=torch.float32)
            batch["noisy_latents"] = cached_targets.noisy_latents.to(device=self.device)
            xt = batch["noisy_latents"].float()
            old_prediction = cached_targets.old_prediction.to(device=self.device)
            if cached_targets.ref_prediction is not None:
                ref_prediction = cached_targets.ref_prediction.to(device=self.device)
        else:
            t = self._resolve_training_timestep(rollout, timestep_idx)
            xt = torch.empty_like(x0)

        neg_for_cfg = negative_batch if negative_batch is not None else None
        t_expanded = t.view(-1, *([1] * (x0.ndim - 1)))
        if cached_targets is None:
            noise = torch.randn_like(x0)
            xt = (1.0 - t_expanded) * x0 + t_expanded * noise
            batch["noisy_latents"] = xt

            # Old-teacher prediction (no grad)
            with torch.no_grad(), apply_ema_maybe(self.old_ema):
                old_prediction = self._predict(batch, t, neg_for_cfg).detach()

            # Reference model prediction (no grad)
            if self.kl_beta > 0:
                with torch.no_grad(), self.reference_model():
                    ref_prediction = self._predict(batch, t, neg_for_cfg).detach()
            else:
                ref_prediction = None

        # Current model prediction (with grad)
        forward_prediction = self._predict(batch, t, neg_for_cfg)

        # Normalise advantages to r in [0, 1]
        adv = rollout_advantages.to(device=self.device)
        adv = torch.clamp(adv, -self.adv_clip_max, self.adv_clip_max)
        if self.adv_mode == "positive_only":
            adv = torch.clamp(adv, 0, self.adv_clip_max)
        elif self.adv_mode == "negative_only":
            adv = torch.clamp(adv, -self.adv_clip_max, 0)
        elif self.adv_mode == "binary":
            adv = torch.sign(adv)

        r = (adv / self.adv_clip_max) / 2.0 + 0.5
        r = torch.clamp(r, 0.0, 1.0)
        # Expand r to match spatial dims
        r = r.view(-1, *([1] * (x0.ndim - 1)))

        beta = self.beta

        # Positive & negative predictions
        positive_pred = beta * forward_prediction + (1 - beta) * old_prediction
        negative_pred = (1 + beta) * old_prediction - beta * forward_prediction

        # Predicted x0 from positive prediction
        x0_pos = xt - t_expanded * positive_pred
        with torch.no_grad():
            weight_pos = (
                torch.abs(x0_pos.double() - x0.double())
                .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                .clip(min=1e-5)
            )
        pos_loss = ((x0_pos - x0) ** 2 / weight_pos).mean(dim=tuple(range(1, x0.ndim)))

        # Predicted x0 from negative prediction
        x0_neg = xt - t_expanded * negative_pred
        with torch.no_grad():
            weight_neg = (
                torch.abs(x0_neg.double() - x0.double())
                .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                .clip(min=1e-5)
            )
        neg_loss = ((x0_neg - x0) ** 2 / weight_neg).mean(dim=tuple(range(1, x0.ndim)))

        # Flatten r for per-sample weighting
        r_flat = r.view(r.shape[0], -1)[:, 0]
        policy_loss_per_sample = (
            r_flat * pos_loss / beta + (1.0 - r_flat) * neg_loss / beta
        )
        policy_loss = (policy_loss_per_sample * self.adv_clip_max).mean()

        loss = policy_loss

        # KL regularisation (MSE to reference)
        if ref_prediction is not None:
            kl_loss = ((forward_prediction - ref_prediction) ** 2).mean(
                dim=tuple(range(1, x0.ndim))
            )
            kl_loss = torch.mean(kl_loss)
            loss = loss + self.kl_beta * kl_loss
        else:
            kl_loss = torch.tensor(0.0, device=self.device)

        metrics: dict[str, torch.Tensor] = {
            "train/policy_loss": policy_loss.detach(),
            "train/kl_loss": kl_loss.detach(),
            "train/loss": loss.detach(),
            "train/old_deviate": ((forward_prediction - old_prediction) ** 2)
            .mean()
            .detach(),
        }
        self.log_aggregated_metrics(metrics)
        return loss

    def _predict(
        self,
        batch: Any,
        t: torch.Tensor,
        negative_batch: Any | None,
    ) -> torch.Tensor:
        """Get model velocity prediction, optionally with CFG."""
        sampler = self.rollout_sampler
        if negative_batch is not None and sampler.cfg_scale > 1.0:
            return sampler.get_guided_velocity(
                self.model,
                batch["noisy_latents"],
                t,
                batch,
                negative_batch,
            )
        return self.model.predict_velocity(batch, t.to(dtype=self.model.dtype)).float()

    # ----------------------------- Training phase ------------------------------- #

    def _resolve_num_train_timesteps(self, rollout: Rollout) -> int:
        rollout_timesteps = rollout.trajectory.timesteps
        total_timesteps = (
            rollout_timesteps.shape[0]
            if rollout_timesteps is not None
            else self.rollout_sampler.steps
        )
        if total_timesteps <= 0:
            raise RuntimeError(
                "NFT rollout must contain at least one training timestep."
            )
        if not 0.0 < self.timestep_fraction <= 1.0:
            raise ValueError(
                f"timestep_fraction must be in (0, 1], got {self.timestep_fraction}."
            )
        if self.num_train_timesteps is not None:
            if self.num_train_timesteps <= 0:
                raise ValueError(
                    "num_train_timesteps must be positive when explicitly set."
                )
            return min(self.num_train_timesteps, total_timesteps)
        return max(1, int(total_timesteps * self.timestep_fraction))

    def _build_inner_epoch_train_items(
        self, rollouts: list[Rollout]
    ) -> list[NftTrainItem]:
        train_items: list[NftTrainItem] = []
        for rollout_idx in torch.randperm(len(rollouts)).tolist():
            rollout = rollouts[rollout_idx]
            total_timesteps = (
                rollout.trajectory.timesteps.shape[0]
                if rollout.trajectory.timesteps is not None
                else self.rollout_sampler.steps
            )
            selected_timesteps = self._resolve_num_train_timesteps(rollout)
            timestep_perm = torch.randperm(total_timesteps)[
                :selected_timesteps
            ].tolist()
            train_items.extend(
                NftTrainItem(rollout_idx=rollout_idx, timestep_idx=timestep_idx)
                for timestep_idx in timestep_perm
            )
        return train_items

    def _build_train_plan(self, rollouts: list[Rollout]) -> list[list[NftTrainItem]]:
        return [
            self._build_inner_epoch_train_items(rollouts)
            for _ in range(self.num_inner_epochs)
        ]

    def _prepare_cached_targets(
        self,
        rollouts: list[Rollout],
        flat_items: list[NftTrainItem],
        progress: Progress,
        task_id: TaskID,
    ) -> None:
        # Prepare per-item noisy latents once so the cached teacher/reference
        # outputs match the exact inputs used later in optimization.
        for item in flat_items:
            rollout = rollouts[item.rollout_idx]
            batch = deep_move_to_device(rollout.batch, self.device)
            x0 = batch["clean_latents"].float()
            t = self._resolve_training_timestep(rollout, item.timestep_idx)
            t_expanded = t.view(-1, *([1] * (x0.ndim - 1)))
            noise = torch.randn_like(x0)
            xt = (1.0 - t_expanded) * x0 + t_expanded * noise
            item.cached_targets = NftCachedTargets(
                timestep=t.detach(),
                noisy_latents=xt.detach(),
                old_prediction=torch.empty_like(xt),
            )
            progress.advance(task_id)

    def _precompute_old_predictions(
        self,
        rollouts: list[Rollout],
        flat_items: list[NftTrainItem],
        progress: Progress,
        task_id: TaskID,
    ) -> None:
        with apply_ema_maybe(self.old_ema):
            for item in flat_items:
                rollout = rollouts[item.rollout_idx]
                batch = deep_move_to_device(rollout.batch, self.device)
                negative_batch = (
                    deep_move_to_device(rollout.negative_batch, self.device)
                    if rollout.negative_batch is not None
                    else None
                )
                cached_targets = item.cached_targets
                if cached_targets is None:
                    raise RuntimeError("Missing cached NFT targets.")
                batch["noisy_latents"] = cached_targets.noisy_latents
                neg_for_cfg = negative_batch if negative_batch is not None else None
                cached_targets.old_prediction = self._predict(
                    batch,
                    cached_targets.timestep,
                    neg_for_cfg,
                ).detach()
                progress.advance(task_id)

    def _precompute_reference_predictions(
        self,
        rollouts: list[Rollout],
        flat_items: list[NftTrainItem],
        progress: Progress,
        task_id: TaskID,
    ) -> None:
        with self.reference_model():
            for item in flat_items:
                rollout = rollouts[item.rollout_idx]
                batch = deep_move_to_device(rollout.batch, self.device)
                negative_batch = (
                    deep_move_to_device(rollout.negative_batch, self.device)
                    if rollout.negative_batch is not None
                    else None
                )
                cached_targets = item.cached_targets
                if cached_targets is None:
                    raise RuntimeError("Missing cached NFT targets.")
                batch["noisy_latents"] = cached_targets.noisy_latents
                neg_for_cfg = negative_batch if negative_batch is not None else None
                cached_targets.ref_prediction = self._predict(
                    batch,
                    cached_targets.timestep,
                    neg_for_cfg,
                ).detach()
                progress.advance(task_id)

    def _precompute_aux_model_outputs_for_plan(
        self,
        rollouts: list[Rollout],
        train_plan: list[list[NftTrainItem]],
    ) -> None:
        flat_items = [item for items in train_plan for item in items]
        if len(flat_items) == 0:
            return

        logger.info(
            "Precomputing NFT old/reference model outputs for %d train items.",
            len(flat_items),
        )
        was_training = self.transformer.training
        self.transformer.eval()
        progress = Progress(
            *self.get_progress_columns(),
            console=console,
            transient=True,
        )
        prepare_task = progress.add_task("Prepare cache", total=len(flat_items))
        old_task = progress.add_task("Precompute old", total=len(flat_items))
        ref_task = None
        if self.kl_beta > 0:
            ref_task = progress.add_task("Precompute ref", total=len(flat_items))

        with progress, torch.no_grad():
            self._prepare_cached_targets(rollouts, flat_items, progress, prepare_task)
            self._precompute_old_predictions(rollouts, flat_items, progress, old_task)

            if self.kl_beta > 0:
                if ref_task is None:
                    raise RuntimeError("Missing NFT reference precompute task.")
                self._precompute_reference_predictions(
                    rollouts,
                    flat_items,
                    progress,
                    ref_task,
                )

        if was_training:
            self.transformer.train()

    def _optimizer_step(self):
        """Clip gradients, step all optimizers (except old-EMA), and zero grads."""
        if self.clip_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.transformer.parameters(), self.clip_grad_norm
            )
        self.optimizer.step()
        if self._ema_optimizer is not None:
            self._ema_optimizer.step()
        if self._init_backup_optimizer is not None:
            self._init_backup_optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

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
                        "No training items were selected in NFT inner epoch."
                    )

                if (
                    len(train_items) % self.grad_acc_steps != 0
                    and self.is_main_process
                    and self.current_epoch == 0
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

                        loss = self.nft_loss(
                            rollout=rollouts[item.rollout_idx],
                            rollout_advantages=advantages[item.rollout_idx],
                            timestep_idx=item.timestep_idx,
                            cached_targets=item.cached_targets,
                        )

                        if not torch.isfinite(loss):
                            raise RuntimeError(
                                f"Non-finite NFT loss detected: {loss.item()}. "
                                f"(rollout_idx={item.rollout_idx})"
                            )

                        (loss / chunk_size).backward()
                        progress.advance(train_task)

                    self._optimizer_step()
                    self.current_step += 1
                    self.flush_aggregated_metrics(self.current_step)

    # -------------------------------- Main loop --------------------------------- #

    @distributed_main
    def run(self):
        self.set_seed()
        self.init_tracker()
        self.load_transformer_from_seed(self.model, self.seed_checkpoint_dir)
        self.make_optimizer_and_scheduler()
        self.processor.load_models("decode", device=self.device)
        self.make_rollout_dataloader()
        self.make_validation_dataloader()

        self.reward.load_model(self.device)

        os.makedirs(self.checkpoint_root, exist_ok=True)
        if self.resume_from_dir is not None:
            self.load_dcp_checkpoint(self.resume_from_dir)

        with apply_ema_maybe(self._ema_optimizer):
            self.validate_and_log(self.model, self.current_step, reward=self.reward)

        logger.info(
            "NFT rollouts in each epoch will randomly select %d unique prompts "
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
            "NFT optimization uses train_batch_size=%d, world_size=%d, "
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
            "NFT Training", total=self.train_epochs, completed=self.current_epoch
        )

        with self.status_bar("NFT Training"), progress:
            while self.current_epoch < self.train_epochs:
                logger.debug(f"Epoch {self.current_epoch}: starting rollout phase...")
                with apply_ema_maybe(self.old_ema):
                    rollouts = self._collect_rollouts(self.current_epoch)
                advantages = self._compute_advantages(rollouts, step=self.current_step)

                logger.debug(f"Epoch {self.current_epoch}: starting training phase...")
                self._train_on_rollouts(rollouts, advantages)

                # Step old-teacher EMA once per epoch
                self.old_ema.step()

                self.current_epoch += 1
                progress.update(task, completed=self.current_epoch)

                del rollouts, advantages
                torch.cuda.empty_cache()

                if self.checkpoint_epochs > 0 and (
                    self.current_epoch % self.checkpoint_epochs == 0
                    or self.current_epoch == self.train_epochs
                ):
                    self.save(self.current_step)

                if (
                    self.validation_epochs > 0
                    and self.current_epoch % self.validation_epochs == 0
                ):
                    with apply_ema_maybe(self._ema_optimizer):
                        self.validate_and_log(
                            self.model, self.current_step, reward=self.reward
                        )

        with apply_ema_maybe(self._ema_optimizer):
            self.save_dcp_checkpoint(
                self.get_checkpoint_dir(self.current_step) + "_final"
            )
