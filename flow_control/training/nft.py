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
    MicrobatchTrainMixin,
    Rollout,
    RolloutMixin,
    ValidationMixin,
    distributed_main,
    trainer_registry,
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


@dataclass(slots=True)
class _NftLossInput:
    batch: Any
    negative_batch: Any | None
    timestep: torch.Tensor
    x0: torch.Tensor
    noisy_latents: torch.Tensor
    advantage: torch.Tensor
    old_prediction: torch.Tensor | None
    ref_prediction: torch.Tensor | None


@trainer_registry.register("nft")
class NftTrainer(
    RolloutMixin, ValidationMixin, MicrobatchTrainMixin, CheckpointingMixin
):
    model_config = ConfigDict(extra="forbid")
    training_type: str = "nft"

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
    """Timesteps per sample per inner epoch. ``None`` means derive from the
    eligible rollout timesteps and ``timestep_fraction``."""
    timestep_fraction: float = 1.0
    """Fraction of *eligible* rollout timesteps to train on for each sample."""
    timestep_range: float = 1.0
    """Restrict NFT training to the noisiest fraction of the trajectory.

    Rollout timesteps live in ``[0, 1]`` (1 = pure noise). Only timesteps with
    ``t >= 1 - timestep_range`` are eligible for training, so ``0.8`` keeps the
    noisiest 80% and skips the cleanest 20% (matches Flow-Factory's
    ``timestep_range``). ``1.0`` keeps the whole grid."""
    train_timestep_sampling: Literal["random", "stratified"] = "random"
    """How to draw the training timesteps from the eligible window each inner
    epoch. ``random`` = uniform subset (legacy). ``stratified`` = one draw per
    equal bin across the window, giving even noise coverage (matches
    Flow-Factory's discrete stratified sampling)."""

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
        """Temporarily switch to reference model weights."""
        if self.model.peft_lora_rank > 0:
            # diffusers LoRA models expose enable/disable_adapters() toggles;
            # PEFT's disable_adapter() context manager is PeftModel-only and is
            # absent on the FSDP-wrapped diffusers transformer.
            self.transformer.disable_adapters()
            try:
                yield
            finally:
                self.transformer.enable_adapters()
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

    def _prepare_nft_loss_input(
        self,
        rollout: Rollout,
        rollout_advantages: torch.Tensor,
        timestep_idx: int,
        cached_targets: NftCachedTargets | None = None,
    ) -> _NftLossInput:
        batch = deep_move_to_device(rollout.batch, self.device)
        negative_batch = (
            deep_move_to_device(rollout.negative_batch, self.device)
            if rollout.negative_batch is not None
            else None
        )
        x0 = batch["clean_latents"].float()
        old_prediction: torch.Tensor | None = None
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
            noise = torch.randn_like(x0)
            t_expanded = t.view(-1, *([1] * (x0.ndim - 1)))
            xt = (1.0 - t_expanded) * x0 + t_expanded * noise
            batch["noisy_latents"] = xt

        return _NftLossInput(
            batch=batch,
            negative_batch=negative_batch,
            timestep=t,
            x0=x0,
            noisy_latents=xt,
            advantage=rollout_advantages.to(device=self.device),
            old_prediction=old_prediction,
            ref_prediction=ref_prediction,
        )

    def _nft_objective(
        self,
        prepared: _NftLossInput,
        forward_prediction: torch.Tensor,
    ) -> torch.Tensor:
        old_prediction = prepared.old_prediction
        if old_prediction is None:
            raise RuntimeError("Missing NFT old-teacher prediction.")
        x0 = prepared.x0
        xt = prepared.noisy_latents
        t_expanded = prepared.timestep.view(-1, *([1] * (x0.ndim - 1)))

        adv = prepared.advantage
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
        if prepared.ref_prediction is not None:
            kl_loss = ((forward_prediction - prepared.ref_prediction) ** 2).mean(
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

    def nft_loss_batched(
        self,
        items: list[NftTrainItem],
        rollouts: list[Rollout],
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        prepared = [
            self._prepare_nft_loss_input(
                rollouts[item.rollout_idx],
                advantages[item.rollout_idx],
                item.timestep_idx,
                item.cached_targets,
            )
            for item in items
        ]
        batches = [item.batch for item in prepared]
        timesteps = [item.timestep for item in prepared]
        negative_batches = [item.negative_batch for item in prepared]

        if any(item.old_prediction is None for item in prepared):
            with torch.no_grad(), apply_ema_maybe(self._old_ema):
                old_predictions = self._predict_batched(
                    batches, timesteps, negative_batches
                )
            for item, prediction in zip(prepared, old_predictions, strict=True):
                if item.old_prediction is None:
                    item.old_prediction = prediction.detach()

        if self.kl_beta > 0 and any(item.ref_prediction is None for item in prepared):
            with torch.no_grad(), self.reference_model():
                ref_predictions = self._predict_batched(
                    batches, timesteps, negative_batches
                )
            for item, prediction in zip(prepared, ref_predictions, strict=True):
                if item.ref_prediction is None:
                    item.ref_prediction = prediction.detach()

        forward_predictions = self._predict_batched(
            batches, timesteps, negative_batches
        )
        return torch.stack(
            [
                self._nft_objective(item, prediction)
                for item, prediction in zip(prepared, forward_predictions, strict=True)
            ]
        ).mean()

    def _predict_batched(
        self,
        batches: list[Any],
        timesteps: list[torch.Tensor],
        negative_batches: list[Any | None],
    ) -> list[torch.Tensor]:
        """Get model velocity prediction, optionally with CFG."""
        sampler = self.rollout_sampler
        if sampler.cfg_scale > 1.0:
            return sampler.get_guided_velocity(
                model=self.model,
                batches=batches,
                negative_batches=negative_batches,
                latents=[batch["noisy_latents"] for batch in batches],
                timesteps=timesteps,
            )
        return self.model.predict_velocity_batched(batches, timesteps)

    # ----------------------------- Training phase ------------------------------- #

    def _eligible_timestep_indices(self, rollout: Rollout) -> list[int]:
        """Rollout-grid indices eligible for NFT training.

        ``timestep_range`` keeps only the noisiest fraction of the trajectory
        (``t >= 1 - timestep_range`` with ``t in [0, 1]``); ``1.0`` keeps all.

        Distributed invariant: every rank must build the same number of train
        items per inner epoch, or the per-microbatch model collectives stop
        lining up across ranks. With ``timestep_range < 1`` the eligible count
        is derived from the rollout's sigma grid, which a resolution-dependent
        ``shift`` makes image-size dependent — mixed-resolution data could then
        skew per-rank totals. Fixed-resolution workloads are unaffected.
        """
        rollout_timesteps = rollout.trajectory.timesteps
        if rollout_timesteps is None:
            return list(range(self.rollout_sampler.steps))
        total = rollout_timesteps.shape[0]
        if not 0.0 < self.timestep_range <= 1.0:
            raise ValueError(
                f"timestep_range must be in (0, 1], got {self.timestep_range}."
            )
        if self.timestep_range >= 1.0:
            return list(range(total))
        threshold = 1.0 - self.timestep_range
        eligible = (
            (rollout_timesteps.float() >= threshold).nonzero(as_tuple=True)[0].tolist()
        )
        if not eligible:
            warn_once(
                logger,
                f"timestep_range={self.timestep_range} excludes all rollout "
                "timesteps; falling back to the full grid.",
            )
            return list(range(total))
        return eligible

    def _resolve_num_train_timesteps(self, num_eligible: int) -> int:
        if num_eligible <= 0:
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
            return min(self.num_train_timesteps, num_eligible)
        return max(1, int(num_eligible * self.timestep_fraction))

    def _select_timestep_positions(self, n: int, k: int) -> list[int]:
        """Pick ``k`` positions in ``[0, n)`` from the eligible window.

        ``stratified`` splits ``[0, n - 1]`` into ``k`` equal bins and draws one
        position uniformly per bin (even noise coverage); ``random`` draws a
        uniform distinct subset. ``k`` is assumed ``<= n``.
        """
        if self.train_timestep_sampling == "stratified":
            boundaries = torch.linspace(0, n - 1, k + 1)
            lower = boundaries[:-1].long()
            upper = boundaries[1:].long()
            offsets = (torch.rand(k) * (upper - lower)).long()
            return (lower + offsets).tolist()
        return torch.randperm(n)[:k].tolist()

    def _build_inner_epoch_train_items(
        self, rollouts: list[Rollout]
    ) -> list[NftTrainItem]:
        train_items: list[NftTrainItem] = []
        for rollout_idx in torch.randperm(len(rollouts)).tolist():
            rollout = rollouts[rollout_idx]
            eligible = self._eligible_timestep_indices(rollout)
            selected_timesteps = self._resolve_num_train_timesteps(len(eligible))
            positions = self._select_timestep_positions(
                len(eligible), selected_timesteps
            )
            train_items.extend(
                NftTrainItem(rollout_idx=rollout_idx, timestep_idx=eligible[p])
                for p in positions
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

    def _precompute_predictions(
        self,
        rollouts: list[Rollout],
        flat_items: list[NftTrainItem],
        progress: Progress,
        task_id: TaskID,
        field: Literal["old_prediction", "ref_prediction"] = "old_prediction",
    ) -> None:
        for micro_items in self.iter_train_micro_batches(flat_items):
            batches: list[Any] = []
            timesteps: list[torch.Tensor] = []
            negative_batches: list[Any | None] = []
            cached_targets_list: list[NftCachedTargets] = []
            for item in micro_items:
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
                batches.append(batch)
                timesteps.append(cached_targets.timestep)
                negative_batches.append(negative_batch)
                cached_targets_list.append(cached_targets)

            predictions = self._predict_batched(batches, timesteps, negative_batches)
            for cached_targets, prediction in zip(
                cached_targets_list, predictions, strict=True
            ):
                setattr(cached_targets, field, prediction.detach())
            progress.advance(task_id, advance=len(micro_items))

    def _precompute_aux_model_outputs_for_plan(
        self,
        rollouts: list[Rollout],
        train_plan: list[list[NftTrainItem]],
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
        old_task = progress.add_task("Precompute old", total=len(flat_items))

        with progress, torch.no_grad():
            self._prepare_cached_targets(rollouts, flat_items, progress, prepare_task)
            with apply_ema_maybe(self._old_ema):
                self._precompute_predictions(
                    rollouts, flat_items, progress, old_task, field="old_prediction"
                )

            if self.kl_beta > 0:
                ref_task = progress.add_task("Precompute ref", total=len(flat_items))
                with self.reference_model():
                    self._precompute_predictions(
                        rollouts,
                        flat_items,
                        progress,
                        ref_task,
                        field="ref_prediction",
                    )

        if was_training:
            self.transformer.train()

        self.log_progress_timing(
            progress, self._current_step, prefix="profile/precompute"
        )

    def _optimizer_step(self):
        """Clip gradients, step all optimizers (except old-EMA), and zero grads."""
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
                for update in self.iter_micro_updates(train_items):
                    self.transformer.set_requires_gradient_sync(update.is_sync_step)
                    loss = self.nft_loss_batched(update.items, rollouts, advantages)
                    self._check_finite_loss(loss, update.items)
                    (loss * update.loss_scale).backward()
                    progress.advance(train_task, advance=len(update.items))

                    if update.is_sync_step:
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
            "NFT Training", total=self.train_epochs, completed=self._current_epoch
        )

        with self.status_bar("NFT Training"), progress:
            while self._current_epoch < self.train_epochs:
                logger.debug(f"Epoch {self._current_epoch}: starting rollout phase...")
                with apply_ema_maybe(self._old_ema):
                    rollouts = self._collect_rollouts(self._current_epoch)
                advantages = self._compute_advantages(rollouts, step=self._current_step)

                logger.debug(f"Epoch {self._current_epoch}: starting training phase...")
                self._train_on_rollouts(rollouts, advantages)

                # Step old-teacher EMA once per epoch
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
