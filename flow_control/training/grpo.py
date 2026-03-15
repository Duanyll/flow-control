import os
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.distributed as dist
from pydantic import ConfigDict
from rich.progress import Progress
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torchdata.stateful_dataloader import StatefulDataLoader

from flow_control.adapters import ModelAdapter
from flow_control.adapters.base import Batch
from flow_control.datasets import DatasetConfig, parse_dataset
from flow_control.processors import Processor
from flow_control.rewards import Reward, execute_reward
from flow_control.rewards.base import BaseReward
from flow_control.samplers import SampleOutput, Sampler
from flow_control.utils.common import (
    deep_cast_float_dtype,
    deep_move_to_device,
)
from flow_control.utils.logging import console, get_logger
from flow_control.utils.types import (
    OptimizerConfig,
    SchedulerConfig,
    parse_optimizer,
    parse_scheduler,
)

from .advantage import Advantage, PerPromptAdvantage
from .data import DistributedKRepeatSampler, PaddingAwareDatasetWrapper, collate_fn
from .ema import (
    EMAConfig,
    EMAOptimizer,
    InitBackupOptimizer,
    apply_ema_maybe,
    apply_init_maybe,
)
from .mixins import CheckpointingMixin, ValidationMixin, distributed_main

logger = get_logger(__name__)


@dataclass
class Rollout:
    trajectory: SampleOutput
    reward: torch.Tensor
    key: str
    batch: Batch
    negative_batch: Batch | None


class HsdpGrpoTrainer(ValidationMixin, CheckpointingMixin):
    model_config = ConfigDict(extra="forbid")

    # ---------------------------------- Configs --------------------------------- #
    model: ModelAdapter
    sampler: Sampler
    processor: Processor
    reward: Reward
    rollout_sampler: Sampler | None = None

    dataset: DatasetConfig
    seed_checkpoint_dir: str
    resume_from_dir: str | None = None
    num_dataloader_workers: int = 0

    optimizer_config: OptimizerConfig = {"class_name": "AdamW", "lr": 3e-4}
    scheduler_config: SchedulerConfig = {"class_name": "ConstantLR", "factor": 1.0}

    # GRPO hyperparameters
    num_batches_per_epoch: int = 2
    """
    Number of "batches" to generate per epoch. The actual micro batch size on GPU is
    always 1. This means in each epoch, we will select `num_prompts_per_batch` unique
    prompts for `num_batches_per_epoch` times, that is `num_batches_per_epoch *
    num_prompts_per_batch` prompts in total (may have duplicates across batches).
    """
    num_prompts_per_batch: int = 4
    """
    Number of unique prompts to select for each batch. See `num_batches_per_epoch` for details.
    """
    num_rollouts_per_prompt: int = 4
    """
    Number of rollouts to generate for each prompt.
    """

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
    advantage: Advantage = PerPromptAdvantage()

    ema: EMAConfig | None = None
    clip_grad_norm: float = 1.0

    # Optimization / training loop
    train_epochs: int = 100
    checkpoint_epochs: int = 5
    validation_epochs: int = 20

    # Validation mode: what to log during validation
    validation_mode: Literal["images", "reward", "both"] = "both"

    # --------------------------------- Status bar ------------------------------- #
    _status_fields: dict[str, str] = {
        "reward/mean": "R̄: {v:.3f}",
        "reward/std": "σ: {v:.3f}",
        "train/loss": "Loss: {v:.4f}",
    }

    # ------------------------------- Lazy state --------------------------------- #
    _dataloader: StatefulDataLoader | None = None
    _optimizer: torch.optim.Optimizer | None = None
    _scheduler: Any = None
    _ema_optimizer: EMAOptimizer | None = None
    _init_backup_optimizer: InitBackupOptimizer | None = None
    _current_step: int = 0
    _current_epoch: int = 0

    @property
    def validation_sampler(self) -> Sampler:
        return self.sampler

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
    def rollout_batch_per_rank(self) -> int:
        """Number of rollouts per rank per batch (derived from global settings)."""
        total = self.num_prompts_per_batch * self.num_rollouts_per_prompt
        if total % self.world_size != 0:
            raise ValueError(
                f"num_prompts_per_batch * num_rollouts_per_prompt ({total}) "
                f"must be divisible by world_size ({self.world_size})."
            )
        return total // self.world_size

    @property
    def grad_acc_steps(self) -> int:
        if self.train_batch_size % self.world_size != 0:
            raise ValueError(
                f"global_batch_size ({self.train_batch_size}) must be divisible by world_size ({self.world_size})."
            )
        return self.train_batch_size // self.world_size

    @property
    def grpo_sampler(self) -> Sampler:
        sampler = self.rollout_sampler if self.rollout_sampler else self.sampler
        if not sampler.solver.supports_step_log_prob:
            raise TypeError(
                "GRPO training requires a stochastic solver with replayable step log-prob."
            )
        return sampler

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
        self._scheduler = parse_scheduler(self.scheduler_config, self.optimizer)
        if self.ema is not None:
            self._ema_optimizer = EMAOptimizer(params, self.ema)
        need_init = self.kl_beta > 0 and self.model.peft_lora_rank == 0
        if need_init:
            self._init_backup_optimizer = InitBackupOptimizer(params)
            logger.info("Init backup enabled for reference model (kl_beta > 0).")

    def make_train_dataloader(self):
        dataset = PaddingAwareDatasetWrapper(parse_dataset(self.dataset))
        sampler = DistributedKRepeatSampler(
            dataset=dataset,
            num_batches_per_epoch=self.num_batches_per_epoch,
            num_prompts_per_batch=self.num_prompts_per_batch,
            num_rollouts_per_prompt=self.num_rollouts_per_prompt,
            num_replicas=self.world_size,
            rank=self.rank,
            seed=self.seed,
        )
        self._dataloader = StatefulDataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=self.num_dataloader_workers,
            collate_fn=collate_fn,
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
                self.transformer, self.optimizer, options=opts
            ),
            "dataloader": self.dataloader.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "current_step": self.current_step,
            "current_epoch": self.current_epoch,
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
            self.optimizer,
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

        log_prob, mean, std_dev = self.grpo_sampler.compute_logprob_at_step(
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
        if self.kl_beta > 0:
            with torch.no_grad(), self.reference_model():
                _, ref_mean, _ = self.grpo_sampler.compute_logprob_at_step(
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
            and self.current_epoch == 0
        ):
            logger.warning(
                "Local loss count (%d) is not divisible by grad_acc_steps (%d). "
                "The tail update uses a smaller effective batch.",
                num_train_items,
                self.grad_acc_steps,
            )

    # --- Core training phases ---

    def _collect_rollouts(self) -> list[Rollout]:
        """Rollout phase: generate images, decode, then score rewards concurrently."""
        rollouts: list[Rollout] = []
        self.transformer.eval()

        total_rollouts = self.num_batches_per_epoch * self.rollout_batch_per_rank

        progress = Progress(
            *self.get_progress_columns(),
            console=console,
            transient=True,
        )
        rollout_task = progress.add_task("Rollout", total=total_rollouts)

        def rollout_submitter() -> Generator[tuple[dict[str, Any], int]]:
            with progress:
                for batch_idx, batch in enumerate(self.dataloader):
                    batch = deep_move_to_device(batch, self.device)
                    batch = deep_cast_float_dtype(batch, self.model.dtype)

                    generator = torch.Generator(device=self.device).manual_seed(
                        self.seed
                        + self.current_epoch * 10000
                        + batch_idx * self.world_size
                        + self.rank
                    )
                    self.processor.initialize_latents(
                        batch,
                        generator=generator,
                        device=self.device,
                    )

                    negative_batch: Any = (
                        self.processor.get_negative_batch(batch)
                        if self.grpo_sampler.cfg_scale > 1.0
                        else None
                    )

                    with torch.no_grad():
                        rollout_out = self.grpo_sampler.sample(
                            self.model,
                            batch,
                            negative_batch=negative_batch,
                            return_trajectory=True,
                        )

                    with torch.no_grad():
                        decoded = self.processor.decode_output(
                            rollout_out.final_latents,
                            batch,
                        )
                    batch.update(decoded)

                    rollouts.append(
                        Rollout(
                            trajectory=deep_move_to_device(
                                rollout_out, torch.device("cpu")
                            ),
                            reward=torch.tensor(0.0),  # placeholder
                            key=batch.get("__key__", "unknown"),
                            batch=deep_move_to_device(batch, torch.device("cpu")),
                            negative_batch=(
                                deep_move_to_device(
                                    negative_batch,
                                    torch.device("cpu"),
                                )
                                if negative_batch is not None
                                else None
                            ),
                        )
                    )

                    progress.advance(rollout_task)
                    yield batch, len(rollouts) - 1

        def reward_handler(idx: int, reward_tensor: torch.Tensor) -> None:
            rollouts[idx].reward = (
                reward_tensor.view(1) if reward_tensor.ndim == 0 else reward_tensor
            ).cpu()

        execute_reward(self.reward, rollout_submitter(), reward_handler)
        return rollouts

    def _compute_advantages(self, rollouts: list[Rollout]) -> torch.Tensor:
        """Compute advantages using gathered rewards across all GPUs."""
        local_rewards = torch.cat([s.reward for s in rollouts], dim=0).to(self.device)

        gathered_rewards_list: list[torch.Tensor] = [
            torch.zeros_like(local_rewards) for _ in range(self.world_size)
        ]
        dist.all_gather(gathered_rewards_list, local_rewards)
        gathered_rewards = torch.cat(gathered_rewards_list, dim=0)

        local_keys = [s.key for s in rollouts]
        all_keys_nested: list[Any] = [None] * self.world_size
        dist.all_gather_object(all_keys_nested, local_keys)
        all_keys: list[str] = []
        for key_list in all_keys_nested:
            all_keys.extend(key_list)

        unique_keys = list(dict.fromkeys(all_keys))
        key_to_id = {k: i for i, k in enumerate(unique_keys)}
        prompt_ids = torch.tensor(
            [key_to_id[k] for k in all_keys],
            device=gathered_rewards.device,
        )

        advantages = self.advantage.compute(gathered_rewards, prompt_ids)

        reward_mean = gathered_rewards.mean().item()
        reward_std = gathered_rewards.std().item()
        self.log_metrics(
            {
                "reward/mean": reward_mean,
                "reward/std": reward_std,
                "reward/adv_abs_mean": advantages.abs().mean().item(),
            },
            step=self.current_step,
        )
        logger.debug(
            f"Epoch {self.current_epoch}: reward_mean={reward_mean:.4f}, "
            f"reward_std={reward_std:.4f}"
        )

        local_b = local_rewards.shape[0]
        start = self.rank * local_b
        end = start + local_b
        return advantages[start:end].cpu()

    def _optimizer_step(self):
        """Clip gradients, step all optimizers, and zero gradients."""
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
                        )

                        if not torch.isfinite(loss):
                            raise RuntimeError(
                                f"Non-finite GRPO loss detected: {loss.item()}. "
                                f"(rollout_idx={rollout_idx}, timestep={j})"
                            )

                        (loss / chunk_size).backward()
                        progress.advance(train_task)

                    self._optimizer_step()
                    self.current_step += 1
                    self.flush_aggregated_metrics(self.current_step)

    # --- Validation ---

    def _grpo_validate(self):
        """Run validation with mode-specific behavior."""
        mode = self.validation_mode
        should_log_reward = mode in ("reward", "both")
        reward_for_val: BaseReward | None = self.reward if should_log_reward else None

        if mode == "reward":
            # reward-only: skip image logging by not calling validate_and_log
            # (ValidationMixin always logs images; instead just score reward directly)
            pass

        with apply_ema_maybe(self._ema_optimizer):
            self.validate_and_log(self.model, self.current_step, reward=reward_for_val)

    # --- Main loop ---

    @distributed_main
    def run(self):
        self.set_seed()
        self.init_tracker()
        self.load_transformer_from_seed(self.model, self.seed_checkpoint_dir)
        self.make_optimizer_and_scheduler()
        self.make_train_dataloader()
        self.make_validation_dataloader(self.processor)

        self.reward.load_model(self.device)

        os.makedirs(self.checkpoint_root, exist_ok=True)
        if self.resume_from_dir is not None:
            self.load_dcp_checkpoint(self.resume_from_dir)

        self._grpo_validate()
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

        with self.status_bar("GRPO Training"):
            while self.current_epoch < self.train_epochs:
                if hasattr(self.dataloader.sampler, "set_epoch"):
                    self.dataloader.sampler.set_epoch(self.current_epoch)  # type: ignore[union-attr]

                logger.debug(f"Epoch {self.current_epoch}: starting rollout phase...")
                rollouts = self._collect_rollouts()

                advantages = self._compute_advantages(rollouts)

                logger.debug(f"Epoch {self.current_epoch}: starting training phase...")
                self._train_on_rollouts(rollouts, advantages)

                self.current_epoch += 1

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
                    self._grpo_validate()

        with apply_ema_maybe(self._ema_optimizer):
            self.save_dcp_checkpoint(
                self.get_checkpoint_dir(self.current_step) + "_final"
            )
