import os
from contextlib import contextmanager
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
from flow_control.utils.logging import console, get_logger
from flow_control.utils.tensor import (
    deep_move_to_device,
)
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
    Rollout,
    RolloutMixin,
    ValidationMixin,
    distributed_main,
)

logger = get_logger(__name__)


class GrpoTrainer(RolloutMixin, ValidationMixin, CheckpointingMixin):
    model_config = ConfigDict(extra="forbid")
    training_type: str = "grpo"

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
    How many (rollout, timestep) items should the optimizer see before each update step.
    Must be divisible by world_size. Gradient accumulation steps will be automatically
    set to train_batch_size // world_size.
    """
    clip_range: float = 1e-4
    adv_clip_max: float = 5.0
    kl_beta: float = 0.0

    ema: EMAConfig | None = None
    precompute_aux_model_outputs: bool = False
    """
    Precompute reference-model replay outputs once per outer epoch and reuse
    them during optimization. This reduces repeated model switching at the cost
    of extra accelerator memory for cached per-item tensors.
    """
    clip_grad_norm: float = 1.0

    # Optimization / training loop
    train_epochs: int = 100
    checkpoint_epochs: int = 5
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
    _ema_optimizer: EMAOptimizer | None = None
    _init_backup_optimizer: InitBackupOptimizer | None = None
    _current_step: int = 0
    _current_epoch: int = 0

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
            f"Created optimizer with {num_trainable_params / 1e6:.2f}M trainable parameters."
        )
        self._scheduler = parse_scheduler(self.scheduler_config, self._optimizer)
        if self.ema is not None:
            self._ema_optimizer = EMAOptimizer(params, self.ema)
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
        }
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
        self._dataloader.load_state_dict(state_dict["dataloader"])
        self._scheduler.load_state_dict(state_dict["scheduler"])
        self._current_step = state_dict["current_step"]
        self._current_epoch = state_dict.get("current_epoch", 0)

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

    # --- GRPO loss ---

    def grpo_loss(
        self,
        log_prob: torch.Tensor,
        old_log_prob: torch.Tensor,
        advantages: torch.Tensor,
        mean: torch.Tensor | None = None,
        ref_mean: torch.Tensor | None = None,
        std_dev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute GRPO loss.

        L = E[max(-adv * ratio, -adv * clip(ratio, 1-eps, 1+eps))] + beta * KL
        """
        adv_clip_max = self.adv_clip_max
        clip_range = self.clip_range
        kl_beta = self.kl_beta

        advantages = torch.clamp(advantages, -adv_clip_max, adv_clip_max)
        ratio = torch.exp(log_prob - old_log_prob)

        unclipped_loss = -advantages * ratio
        clipped_loss = -advantages * torch.clamp(
            ratio, 1.0 - clip_range, 1.0 + clip_range
        )
        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

        metrics: dict[str, torch.Tensor] = {
            "train/policy_loss": policy_loss.detach(),
            "train/ratio_mean": ratio.detach().mean(),
            "train/approx_kl": (
                0.5 * torch.mean((log_prob - old_log_prob) ** 2)
            ).detach(),
            "train/clipfrac": (
                torch.mean((torch.abs(ratio - 1.0) > clip_range).float())
            ).detach(),
        }

        loss = policy_loss

        if (
            kl_beta > 0
            and mean is not None
            and ref_mean is not None
            and std_dev is not None
        ):
            kl_loss = ((mean - ref_mean) ** 2).mean(dim=tuple(range(1, mean.ndim))) / (
                2 * std_dev**2
            )
            kl_loss = torch.mean(kl_loss)
            loss = loss + kl_beta * kl_loss
            metrics["train/kl_loss"] = kl_loss.detach()

        metrics["train/loss"] = loss.detach()
        self.log_aggregated_metrics(metrics)
        return loss

    def _compute_loss_at_item(
        self,
        rollout: Rollout,
        rollout_advantages: torch.Tensor,
        timestep_idx: int,
        cached_ref_mean: torch.Tensor | None = None,
    ) -> torch.Tensor:
        trajectory = rollout.trajectory
        batch = deep_move_to_device(rollout.batch, self.device)
        negative_batch = (
            deep_move_to_device(rollout.negative_batch, self.device)
            if rollout.negative_batch is not None
            else None
        )
        if trajectory.log_probs is None or trajectory.timesteps is None:
            raise RuntimeError("Rollout is missing log-prob trajectory data.")
        if trajectory.latents is None:
            raise RuntimeError("Rollout is missing latent trajectory data.")
        solver_state = None
        if trajectory.solver_states is not None:
            solver_state = trajectory.solver_states[timestep_idx]

        old_log_probs = trajectory.log_probs[0].to(device=self.device)  # [T]
        sigmas = trajectory.timesteps.to(device=self.device)  # [T+1]

        latent_t = trajectory.latents[:, timestep_idx].to(device=self.device)
        latent_next = trajectory.latents[:, timestep_idx + 1].to(device=self.device)
        sigma = sigmas[timestep_idx : timestep_idx + 1]
        sigma_next = sigmas[timestep_idx + 1 : timestep_idx + 2]
        advantage = rollout_advantages.to(device=self.device)

        log_prob, mean, std_dev = self.rollout_sampler.compute_logprob_at_step(
            self.model,
            batch,
            latent_t,
            latent_next,
            sigma,
            sigma_next,
            negative_batch=negative_batch,
            solver_state=solver_state,
        )

        ref_mean = None
        if cached_ref_mean is not None:
            ref_mean = cached_ref_mean.to(device=self.device)
        elif self.kl_beta > 0:
            with torch.no_grad(), self.reference_model():
                _, ref_mean, _ = self.rollout_sampler.compute_logprob_at_step(
                    self.model,
                    batch,
                    latent_t,
                    latent_next,
                    sigma,
                    sigma_next,
                    negative_batch=negative_batch,
                    solver_state=solver_state,
                )

        loss = self.grpo_loss(
            log_prob=log_prob,
            old_log_prob=old_log_probs[timestep_idx : timestep_idx + 1],
            advantages=advantage,
            mean=mean,
            ref_mean=ref_mean,
            std_dev=std_dev,
        )
        return loss

    def _precompute_reference_means(
        self,
        rollouts: list[Rollout],
    ) -> list[list[torch.Tensor]] | None:
        if not self.precompute_aux_model_outputs or self.kl_beta <= 0:
            return None

        was_training = self.transformer.training
        self.transformer.eval()

        total_items = 0
        for rollout in rollouts:
            log_probs = rollout.trajectory.log_probs
            if log_probs is None:
                raise RuntimeError("Rollout is missing log-prob trajectory data.")
            total_items += log_probs.shape[1]

        progress = Progress(
            *self.get_progress_columns(),
            console=console,
            transient=True,
        )
        precompute_task = progress.add_task("Precompute ref", total=total_items)
        ref_means: list[list[torch.Tensor]] = []

        with progress, torch.no_grad(), self.reference_model():
            for rollout in rollouts:
                trajectory = rollout.trajectory
                batch = deep_move_to_device(rollout.batch, self.device)
                negative_batch = (
                    deep_move_to_device(rollout.negative_batch, self.device)
                    if rollout.negative_batch is not None
                    else None
                )
                if trajectory.timesteps is None or trajectory.latents is None:
                    raise RuntimeError("Rollout is missing trajectory data.")
                solver_states = trajectory.solver_states
                sigmas = trajectory.timesteps.to(device=self.device)
                rollout_ref_means: list[torch.Tensor] = []
                num_timesteps = trajectory.latents.shape[1] - 1

                for timestep_idx in range(num_timesteps):
                    solver_state = (
                        solver_states[timestep_idx]
                        if solver_states is not None
                        else None
                    )
                    latent_t = trajectory.latents[:, timestep_idx].to(
                        device=self.device
                    )
                    latent_next = trajectory.latents[:, timestep_idx + 1].to(
                        device=self.device
                    )
                    sigma = sigmas[timestep_idx : timestep_idx + 1]
                    sigma_next = sigmas[timestep_idx + 1 : timestep_idx + 2]
                    _, ref_mean, _ = self.rollout_sampler.compute_logprob_at_step(
                        self.model,
                        batch,
                        latent_t,
                        latent_next,
                        sigma,
                        sigma_next,
                        negative_batch=negative_batch,
                        solver_state=solver_state,
                    )
                    rollout_ref_means.append(ref_mean.detach())
                    progress.advance(precompute_task)
                ref_means.append(rollout_ref_means)

        if was_training:
            self.transformer.train()
        return ref_means

    def _build_train_items(
        self, num_rollouts: int, num_timesteps: int
    ) -> list[tuple[int, int]]:
        train_items: list[tuple[int, int]] = []
        for rollout_idx in range(num_rollouts):
            train_items.extend((rollout_idx, j) for j in range(num_timesteps))
        return train_items

    def _warn_if_non_divisible(self, num_train_items: int) -> None:
        if (
            num_train_items % self.grad_acc_steps != 0
            and self.is_main_process
            and self._current_epoch == 0
        ):
            logger.warning(
                "Local loss count (%d) is not divisible by grad_acc_steps (%d). "
                "The tail update uses a smaller effective batch.",
                num_train_items,
                self.grad_acc_steps,
            )

    # --- Core training phases ---

    def _optimizer_step(self):
        """Clip gradients, step all optimizers, and zero gradients."""
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

        ref_means = self._precompute_reference_means(rollouts)
        first_log_probs = rollouts[0].trajectory.log_probs
        if first_log_probs is None:
            raise RuntimeError("Rollout is missing log-prob trajectory data.")
        num_timesteps = first_log_probs.shape[1]
        total_items = self.num_inner_epochs * len(rollouts) * num_timesteps

        progress = Progress(
            *self.get_progress_columns(),
            console=console,
            transient=True,
        )
        train_task = progress.add_task("Training", total=total_items)

        with progress:
            for _inner_epoch in range(self.num_inner_epochs):
                perm = torch.randperm(len(rollouts))
                shuffled_rollouts = [rollouts[perm[i]] for i in range(len(rollouts))]
                shuffled_advantages = advantages[perm]
                shuffled_ref_means = (
                    [ref_means[perm[i]] for i in range(len(rollouts))]
                    if ref_means is not None
                    else None
                )

                train_items = self._build_train_items(
                    len(shuffled_rollouts), num_timesteps
                )

                if len(train_items) == 0:
                    raise RuntimeError(
                        "No training items were selected in GRPO inner epoch."
                    )

                self._warn_if_non_divisible(len(train_items))

                for chunk_start in range(0, len(train_items), self.grad_acc_steps):
                    chunk = train_items[chunk_start : chunk_start + self.grad_acc_steps]
                    chunk_size = len(chunk)

                    for micro_idx, (rollout_idx, j) in enumerate(chunk):
                        is_sync_step = micro_idx == chunk_size - 1
                        self.transformer.set_requires_gradient_sync(is_sync_step)

                        loss = self._compute_loss_at_item(
                            rollout=shuffled_rollouts[rollout_idx],
                            rollout_advantages=shuffled_advantages[rollout_idx],
                            timestep_idx=j,
                            cached_ref_mean=(
                                shuffled_ref_means[rollout_idx][j]
                                if shuffled_ref_means is not None
                                else None
                            ),
                        )

                        if not torch.isfinite(loss):
                            raise RuntimeError(
                                f"Non-finite GRPO loss detected: {loss.item()}. "
                                f"(rollout_idx={rollout_idx}, timestep={j})"
                            )

                        (loss / chunk_size).backward()
                        progress.advance(train_task)

                    self._optimizer_step()
                    self._current_step += 1
                    self.flush_aggregated_metrics(self._current_step)

    # --- Main loop ---

    @distributed_main
    def run(self):
        self.set_seed()
        self.init_tracker()
        self.load_transformer_from_seed(self.model, self.seed_checkpoint_dir)
        self.make_optimizer_and_scheduler()
        self.load_processor()
        self.make_rollout_dataloader()
        self.make_validation_dataloader()

        self.reward.load_model(self.device)

        os.makedirs(self.checkpoint_root, exist_ok=True)
        if self.resume_from_dir is not None:
            self.load_dcp_checkpoint(self.resume_from_dir)

        with apply_ema_maybe(self._ema_optimizer):
            self.validate_and_log(self.model, self._current_step, reward=self.reward)
        logger.info(
            f"GRPO rollouts in each epoch will randomly select {self.num_prompts_per_batch}"
            f" unique prompts for {self.num_batches_per_epoch} times, and generate"
            f" {self.num_rollouts_per_prompt} rollouts for each prompt. That is "
            f"{self.num_batches_per_epoch * self.num_prompts_per_batch * self.num_rollouts_per_prompt}"
            " rollouts in total (may have duplicates across batches)."
        )
        logger.info(
            "GRPO optimization uses train_batch_size=%d, world_size=%d, grad_acc_steps=%d.",
            self.train_batch_size,
            self.world_size,
            self.grad_acc_steps,
        )

        progress = Progress(
            *self.get_progress_columns(),
            console=console,
        )
        task = progress.add_task(
            "GRPO Training", total=self.train_epochs, completed=self._current_epoch
        )

        with self.status_bar("GRPO Training"), progress:
            while self._current_epoch < self.train_epochs:
                logger.debug(f"Epoch {self._current_epoch}: starting rollout phase...")
                rollouts = self._collect_rollouts(self._current_epoch)

                advantages = self._compute_advantages(rollouts, step=self._current_step)

                logger.debug(f"Epoch {self._current_epoch}: starting training phase...")
                self._train_on_rollouts(rollouts, advantages)

                self._current_epoch += 1
                progress.update(task, advance=1)

                del rollouts, advantages
                torch.cuda.empty_cache()

                if self.checkpoint_epochs > 0 and (
                    self._current_epoch % self.checkpoint_epochs == 0
                    or self._current_epoch == self.train_epochs
                ):
                    self.save(self._current_step)

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
