import asyncio
import concurrent.futures
import os
import threading
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, Literal, TypedDict

import torch
import torch.distributed as dist
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torchdata.stateful_dataloader import StatefulDataLoader

from flow_control.adapters.base import Batch
from flow_control.datasets import parse_dataset
from flow_control.rewards import Reward
from flow_control.samplers import Sampler
from flow_control.samplers.euler import EulerSampler, SdeTrajectory
from flow_control.utils.common import (
    deep_cast_float_dtype,
    deep_move_to_device,
)
from flow_control.utils.ema import apply_ema_maybe, apply_init_maybe
from flow_control.utils.logging import console, get_logger

from .advantage import Advantage, PerPromptAdvantage
from .data import DistributedKRepeatSampler, PaddingAwareDatasetWrapper, collate_fn
from .hsdp_engine import distributed_main
from .trainer_base import HsdpTrainerBase, HsdpTrainerBaseConfig

logger = get_logger(__name__)


class HsdpGrpoTrainerConfig(HsdpTrainerBaseConfig):
    reward: Reward
    train_sampler: Sampler | None = None

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
    timestep_selection: Literal[
        "all_except_last", "random_contiguous", "random_fraction"
    ] = "all_except_last"
    timestep_fraction: float = 1.0
    timestep_window_size: int | None = None
    advantage: Advantage = PerPromptAdvantage()

    # Optimization / training loop
    train_epochs: int = 100
    checkpoint_epochs: int = 5
    validation_epochs: int = 20

    # Validation mode: what to log during validation
    validation_mode: Literal["images", "reward", "both"] = "both"

    # Override defaults from base
    optimizer: dict[str, Any] = {"class_name": "AdamW", "lr": 3e-4}
    num_dataloader_workers: int = 0


class Rollout(TypedDict):
    trajectory: SdeTrajectory
    reward: torch.Tensor
    key: str
    batch: Batch
    negative_batch: Batch | None


class RewardLoopThread:
    """Run async reward requests in a dedicated event loop thread."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="grpo-reward-loop",
            daemon=True,
        )
        self._started = False

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def submit(self, coro: Any) -> concurrent.futures.Future[torch.Tensor]:
        if not self._started:
            self._thread.start()
            self._started = True
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def close(self) -> None:
        if not self._started:
            self._loop.close()
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()


class HsdpGrpoTrainer(HsdpTrainerBase[HsdpGrpoTrainerConfig]):
    conf: HsdpGrpoTrainerConfig

    current_epoch: int = 0

    @property
    def rollout_batch_per_rank(self) -> int:
        """Number of rollouts per rank per batch (derived from global settings)."""
        total = self.conf.num_prompts_per_batch * self.conf.num_rollouts_per_prompt
        if total % self.world_size != 0:
            raise ValueError(
                f"num_prompts_per_batch * num_rollouts_per_prompt ({total}) "
                f"must be divisible by world_size ({self.world_size})."
            )
        return total // self.world_size

    @property
    def grad_acc_steps(self) -> int:
        if self.conf.train_batch_size % self.world_size != 0:
            raise ValueError(
                f"global_batch_size ({self.conf.train_batch_size}) must be divisible by world_size ({self.world_size})."
            )
        return self.conf.train_batch_size // self.world_size

    def __init__(self, **kwargs):
        self.conf = HsdpGrpoTrainerConfig(**kwargs)
        super().__init__(**kwargs)

    @property
    def grpo_sampler(self) -> EulerSampler:
        sampler = self.conf.train_sampler if self.conf.train_sampler else self.sampler
        if not isinstance(sampler, EulerSampler):
            raise TypeError(
                "GRPO training requires an EulerSampler-compatible train_sampler."
            )
        return sampler

    def make_optimizer_and_scheduler(self, enable_init_backup: bool = False):
        need_init = self.conf.kl_beta > 0 and self.model.peft_lora_rank == 0
        super().make_optimizer_and_scheduler(enable_init_backup=need_init)
        if need_init:
            logger.info("Init backup enabled for reference model (kl_beta > 0).")

    def make_train_dataloader(self):
        dataset = PaddingAwareDatasetWrapper(parse_dataset(self.conf.dataset))
        sampler = DistributedKRepeatSampler(
            dataset=dataset,
            num_batches_per_epoch=self.conf.num_batches_per_epoch,
            num_prompts_per_batch=self.conf.num_prompts_per_batch,
            num_rollouts_per_prompt=self.conf.num_rollouts_per_prompt,
            num_replicas=self.world_size,
            rank=self.rank,
            seed=self.conf.seed,
        )
        self.dataloader = StatefulDataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=self.conf.num_dataloader_workers,
            collate_fn=collate_fn,
        )

    def _save_extra_state(self, state: dict) -> None:
        state["current_epoch"] = self.current_epoch

    def _load_extra_state(self, state_dict: dict) -> None:
        self.current_epoch = state_dict.get("current_epoch", 0)

    @contextmanager
    def reference_model(self):
        """Temporarily switch to reference model weights."""
        if self.model.peft_lora_rank > 0:
            with self.transformer.disable_adapter():
                yield
        else:
            with apply_init_maybe(self.optimizer):
                yield

    # --- GRPO loss ---

    @staticmethod
    def grpo_loss(
        log_prob: torch.Tensor,
        old_log_prob: torch.Tensor,
        advantages: torch.Tensor,
        clip_range: float,
        adv_clip_max: float,
        mean: torch.Tensor | None = None,
        ref_mean: torch.Tensor | None = None,
        std_dev: torch.Tensor | None = None,
        kl_beta: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute GRPO loss.

        L = E[max(-adv * ratio, -adv * clip(ratio, 1-eps, 1+eps))] + beta * KL
        """
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
        return loss, metrics

    def _num_train_timesteps(self, num_timesteps: int) -> int:
        if num_timesteps <= 1:
            return 1

        max_train_timesteps = num_timesteps - 1  # Skip the last step by default.
        if self.conf.timestep_selection == "all_except_last":
            return max_train_timesteps
        if self.conf.timestep_selection == "random_contiguous":
            if self.conf.timestep_window_size is None:
                return max_train_timesteps
            if self.conf.timestep_window_size <= 0:
                raise ValueError("timestep_window_size must be > 0.")
            return min(self.conf.timestep_window_size, max_train_timesteps)
        if self.conf.timestep_selection == "random_fraction":
            if not (0.0 < self.conf.timestep_fraction <= 1.0):
                raise ValueError("timestep_fraction must be in (0, 1].")
            return max(1, int(max_train_timesteps * self.conf.timestep_fraction))
        raise ValueError(f"Unknown timestep_selection: {self.conf.timestep_selection}")

    def _select_train_timesteps(self, num_timesteps: int) -> list[int]:
        if num_timesteps <= 1:
            return [0]

        trainable = list(range(num_timesteps - 1))
        if self.conf.timestep_selection == "all_except_last":
            return trainable

        num_train_timesteps = self._num_train_timesteps(num_timesteps)
        if self.conf.timestep_selection == "random_contiguous":
            max_start = len(trainable) - num_train_timesteps
            start = torch.randint(0, max_start + 1, (1,)).item()
            return trainable[start : start + num_train_timesteps]

        perm = torch.randperm(len(trainable))[:num_train_timesteps].tolist()
        return sorted(trainable[i] for i in perm)

    def _gather_mean_scalar(self, value: float) -> float:
        values: list[Any] = [None] * self.world_size
        dist.all_gather_object(values, value)
        return sum(float(v) for v in values) / len(values)

    def _log_step_metrics(self, metric_tensors: dict[str, list[torch.Tensor]]) -> None:
        local_metrics: dict[str, float] = {}
        for key, values in metric_tensors.items():
            if len(values) == 0:
                continue
            local_metrics[key] = torch.stack(values).mean().item()
        local_metrics["train/lr"] = float(self.scheduler.get_last_lr()[0])
        gathered_metrics = {
            key: self._gather_mean_scalar(value) for key, value in local_metrics.items()
        }
        self.log_metrics(gathered_metrics)

    def _compute_loss_at_item(
        self,
        rollout: Rollout,
        rollout_advantages: torch.Tensor,
        timestep_idx: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        trajectory = rollout["trajectory"]
        batch = deep_move_to_device(rollout["batch"], self.device)
        negative_batch = (
            deep_move_to_device(rollout["negative_batch"], self.device)
            if rollout["negative_batch"] is not None
            else None
        )
        old_log_probs = trajectory["log_probs"][0].to(device=self.device)  # [T]
        sigmas = trajectory["timesteps"].to(device=self.device)  # [T+1]

        latent_t = trajectory["latents"][:, timestep_idx].to(device=self.device)
        latent_next = trajectory["latents"][:, timestep_idx + 1].to(device=self.device)
        sigma = sigmas[timestep_idx : timestep_idx + 1]
        sigma_next = sigmas[timestep_idx + 1 : timestep_idx + 2]
        advantage_slice = rollout_advantages[timestep_idx : timestep_idx + 1].to(
            device=self.device
        )

        log_prob, mean, std_dev = self.grpo_sampler.compute_logprob_at_step(
            self.model,
            batch,
            latent_t,
            latent_next,
            sigma,
            sigma_next,
            negative_batch=negative_batch,
        )

        ref_mean = None
        if self.conf.kl_beta > 0:
            with torch.no_grad(), self.reference_model():
                _, ref_mean, _ = self.grpo_sampler.compute_logprob_at_step(
                    self.model,
                    batch,
                    latent_t,
                    latent_next,
                    sigma,
                    sigma_next,
                    negative_batch=negative_batch,
                )

        loss, metrics = self.grpo_loss(
            log_prob=log_prob,
            old_log_prob=old_log_probs[timestep_idx : timestep_idx + 1],
            advantages=advantage_slice,
            clip_range=self.conf.clip_range,
            adv_clip_max=self.conf.adv_clip_max,
            mean=mean,
            ref_mean=ref_mean,
            std_dev=std_dev,
            kl_beta=self.conf.kl_beta,
        )
        return loss, metrics

    def _build_train_items(
        self, num_rollouts: int, num_timesteps: int
    ) -> list[tuple[int, int]]:
        train_items: list[tuple[int, int]] = []
        for rollout_idx in range(num_rollouts):
            step_indices = self._select_train_timesteps(num_timesteps)
            train_items.extend((rollout_idx, j) for j in step_indices)
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
        """Rollout phase: generate images, decode, then score rewards concurrently.

        Reward scoring is launched asynchronously right after each sample's VAE
        decode completes, so network-bound reward requests (e.g. LLM judges)
        overlap with subsequent GPU rollouts.
        """
        rollouts: list[Rollout] = []
        self.transformer.eval()

        total_rollouts = self.conf.num_batches_per_epoch * self.rollout_batch_per_rank

        rollout_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )
        rollout_task = rollout_progress.add_task("Rollout...", total=total_rollouts)

        overlap_reward = self.conf.reward.supports_rollout_overlap()
        reward_loop = RewardLoopThread() if overlap_reward else None
        reward_futures: list[concurrent.futures.Future[torch.Tensor]] = []

        try:
            with rollout_progress:
                for batch_idx, batch in enumerate(self.dataloader):
                    batch = deep_move_to_device(batch, self.device)
                    batch = deep_cast_float_dtype(batch, self.model.dtype)

                    # Initialize noise
                    generator = torch.Generator(device=self.device).manual_seed(
                        self.conf.seed
                        + self.current_epoch * 10000
                        + batch_idx * self.world_size
                        + self.rank
                    )
                    self.processor.initialize_latents(
                        batch,
                        generator=generator,
                        device=self.device,
                        dtype=self.model.dtype,
                    )

                    negative_batch: Any = (
                        self.processor.get_negative_batch(batch)
                        if self.grpo_sampler.cfg_scale > 1.0
                        else None
                    )

                    # Run rollout and record trajectory for GRPO.
                    with torch.no_grad():
                        rollout_out = self.grpo_sampler.sample(
                            self.model,
                            batch,
                            negative_batch=negative_batch,
                            return_trajectory=True,
                        )
                    if not isinstance(rollout_out, dict):
                        raise RuntimeError("GRPO rollout expected trajectory output.")
                    trajectory: SdeTrajectory = rollout_out

                    # Decode final latents to images and merge into batch
                    final_latents = trajectory["latents"][:, -1]
                    with torch.no_grad():
                        decoded = self.processor.decode_output(final_latents, batch)
                    batch.update(decoded)

                    rollout: Rollout = {
                        "trajectory": {
                            "latents": trajectory["latents"].cpu(),
                            "log_probs": trajectory["log_probs"].cpu(),
                            "timesteps": trajectory["timesteps"].cpu(),
                        },
                        "reward": torch.tensor(0.0),  # placeholder, filled below
                        "key": batch.get("__key__", "unknown"),
                        "batch": deep_move_to_device(batch, torch.device("cpu")),
                        "negative_batch": (
                            deep_move_to_device(
                                negative_batch,
                                torch.device("cpu"),
                            )
                            if negative_batch is not None
                            else None
                        ),
                    }
                    rollouts.append(rollout)

                    if reward_loop is not None:
                        reward_batch = self.conf.reward.prepare_batch_for_async(batch)
                        reward_futures.append(
                            reward_loop.submit(
                                self.conf.reward.async_score(reward_batch)
                            )
                        )
                    else:
                        reward = self._score_reward_blocking(batch)
                        rollout["reward"] = (
                            reward.view(1) if reward.ndim == 0 else reward
                        )
                        rollout["reward"] = rollout["reward"].cpu()

                    rollout_progress.advance(rollout_task)

            if reward_loop is not None:
                rewards = [future.result() for future in reward_futures]
                for rollout, reward in zip(rollouts, rewards, strict=True):
                    rollout["reward"] = (
                        reward.view(1) if reward.ndim == 0 else reward
                    ).cpu()
        finally:
            if reward_loop is not None:
                reward_loop.close()

        reward_mean = torch.stack([r["reward"] for r in rollouts]).mean().item()
        logger.info(
            f"Epoch {self.current_epoch}: rollout reward_mean={reward_mean:.4f}"
        )

        return rollouts

    def _compute_advantages(self, rollouts: list[Rollout]) -> torch.Tensor:
        """Compute advantages using gathered rewards across all GPUs.

        Uses the ``__key__`` field from each rollout's batch to group rollouts
        that belong to the same prompt for per-prompt advantage estimation.
        """
        local_rewards = torch.cat([s["reward"] for s in rollouts], dim=0).to(
            self.device
        )

        # Gather rewards across all processes
        gathered_rewards_list: list[torch.Tensor] = [
            torch.zeros_like(local_rewards) for _ in range(self.world_size)
        ]
        dist.all_gather(gathered_rewards_list, local_rewards)
        gathered_rewards = torch.cat(gathered_rewards_list, dim=0)

        # Gather keys across processes to identify prompt groups
        local_keys = [s["key"] for s in rollouts]
        all_keys_nested: list[Any] = [None] * self.world_size
        dist.all_gather_object(all_keys_nested, local_keys)
        all_keys: list[str] = []
        for key_list in all_keys_nested:
            all_keys.extend(key_list)

        # Map keys to integer IDs for the advantage estimator
        unique_keys = list(dict.fromkeys(all_keys))
        key_to_id = {k: i for i, k in enumerate(unique_keys)}
        prompt_ids = torch.tensor(
            [key_to_id[k] for k in all_keys],
            device=gathered_rewards.device,
        )

        num_timesteps = rollouts[0]["trajectory"]["log_probs"].shape[1]

        # Compute advantages [global_B, T]
        advantages = self.conf.advantage.compute(
            gathered_rewards, prompt_ids, num_timesteps
        )

        # Log reward stats
        reward_mean = gathered_rewards.mean().item()
        reward_std = gathered_rewards.std().item()
        self.log_metrics(
            {
                "reward/mean": reward_mean,
                "reward/std": reward_std,
                "reward/adv_abs_mean": advantages.abs().mean().item(),
            }
        )
        logger.info(
            f"Epoch {self.current_epoch}: reward_mean={reward_mean:.4f}, "
            f"reward_std={reward_std:.4f}"
        )

        # Ungather: keep only this rank's slice
        local_b = local_rewards.shape[0]
        start = self.rank * local_b
        end = start + local_b
        return advantages[start:end].cpu()

    def _train_on_rollouts(
        self,
        rollouts: list[Rollout],
        advantages: torch.Tensor,
    ):
        """Training phase: update model using collected rollouts and advantages."""
        self.transformer.train()

        num_timesteps = rollouts[0]["trajectory"]["log_probs"].shape[1]
        num_train_timesteps = self._num_train_timesteps(num_timesteps)
        total_items = self.conf.num_inner_epochs * len(rollouts) * num_train_timesteps

        train_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("• Loss: {task.fields[loss]:.4f}"),
            console=console,
        )
        train_task = train_progress.add_task("Training...", total=total_items, loss=0.0)

        with train_progress:
            for _inner_epoch in range(self.conf.num_inner_epochs):
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
                    metrics_accum: dict[str, list[torch.Tensor]] = {}

                    for micro_idx, (rollout_idx, j) in enumerate(chunk):
                        is_sync_step = micro_idx == chunk_size - 1
                        self.transformer.set_requires_gradient_sync(is_sync_step)

                        loss, metrics = self._compute_loss_at_item(
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
                        for key, value in metrics.items():
                            metrics_accum.setdefault(key, []).append(value.detach())
                        train_progress.advance(train_task)

                    if self.conf.clip_grad_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            self.transformer.parameters(), self.conf.clip_grad_norm
                        )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.current_step += 1
                    self._log_step_metrics(metrics_accum)
                    if "train/loss" in metrics_accum:
                        avg_loss = (
                            torch.stack(metrics_accum["train/loss"]).mean().item()
                        )
                        train_progress.update(train_task, loss=avg_loss)

    # --- Validation ---

    def validate_and_log(
        self,
        on_sample: Callable[[dict[str, Any]], None] | None = None,
    ):
        mode = self.conf.validation_mode
        should_log_images = mode in ("images", "both")
        should_log_reward = mode in ("reward", "both")

        overlap_reward = (
            should_log_reward and self.conf.reward.supports_rollout_overlap()
        )
        reward_loop = RewardLoopThread() if overlap_reward else None
        reward_futures: list[concurrent.futures.Future[torch.Tensor]] = []
        reward_values: list[torch.Tensor] = []

        def grpo_on_sample(batch: dict[str, Any]) -> None:
            if should_log_images:
                self._default_on_sample(batch)
            if not should_log_reward:
                return
            if reward_loop is not None:
                reward_batch = self.conf.reward.prepare_batch_for_async(batch)
                reward_futures.append(
                    reward_loop.submit(self.conf.reward.async_score(reward_batch))
                )
                return
            reward_values.append(self._score_reward_blocking(batch))

        try:
            super().validate_and_log(on_sample=grpo_on_sample)
            if reward_loop is not None:
                reward_values.extend(future.result() for future in reward_futures)
        finally:
            if reward_loop is not None:
                reward_loop.close()

        if reward_values:
            local_mean = torch.stack(reward_values).mean().item()
            val_reward_mean = self._gather_mean_scalar(local_mean)
            self.log_metrics({"val/reward_mean": val_reward_mean})
            logger.info(
                f"Validation at step {self.current_step}: "
                f"reward_mean={val_reward_mean:.4f}"
            )

    def _score_reward_blocking(self, batch: dict[str, Any]) -> torch.Tensor:
        try:
            return self.conf.reward.score(batch)
        except NotImplementedError:
            return asyncio.run(self.conf.reward.async_score(batch))

    # --- Main loop ---

    @distributed_main
    def run(self):
        self.set_seed()
        self.init_tracker()
        self.load_transformer(self.model, self.conf.seed_checkpoint_dir)
        self.make_optimizer_and_scheduler()
        self.make_train_dataloader()
        self.make_validation_dataloader_maybe()

        # Load processor for VAE decoding
        self.processor.load_models("decode", device=self.device)

        # Load reward model
        self.conf.reward.load_model(self.device)

        os.makedirs(self.conf.checkpoint_root, exist_ok=True)
        if self.conf.resume_from_dir is not None:
            self.load_dcp_checkpoint(self.conf.resume_from_dir)

        self.validate_and_log()
        logger.info(
            f"GRPO rollouts in each epoch will randomly select {self.conf.num_prompts_per_batch}"
            f" unique prompts for {self.conf.num_batches_per_epoch} times, and generate"
            f" {self.conf.num_rollouts_per_prompt} rollouts for each prompt. That is "
            f"{self.conf.num_batches_per_epoch * self.conf.num_prompts_per_batch * self.conf.num_rollouts_per_prompt}"
            " rollouts in total (may have duplicates across batches)."
        )
        logger.info(
            "GRPO optimization uses train_batch_size=%d, world_size=%d, grad_acc_steps=%d.",
            self.conf.train_batch_size,
            self.world_size,
            self.grad_acc_steps,
        )

        console.rule("[bold blue]Starting GRPO training[/bold blue]")

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn(" Epoch: {task.fields[epoch]}/{task.fields[total_epochs]}"),
            TextColumn(" Step: {task.fields[step]}"),
            BarColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )
        task = progress.add_task(
            "GRPO Training...",
            total=self.conf.train_epochs,
            completed=self.current_epoch,
            epoch=self.current_epoch,
            total_epochs=self.conf.train_epochs,
            step=self.current_step,
        )

        with progress:
            while self.current_epoch < self.conf.train_epochs:
                if hasattr(self.dataloader.sampler, "set_epoch"):
                    self.dataloader.sampler.set_epoch(self.current_epoch)  # type: ignore[union-attr]

                # Rollout phase
                logger.info(f"Epoch {self.current_epoch}: starting rollout phase...")
                rollouts = self._collect_rollouts()

                # Advantage computation
                advantages = self._compute_advantages(rollouts)

                # Training phase
                logger.info(f"Epoch {self.current_epoch}: starting training phase...")
                self._train_on_rollouts(rollouts, advantages)

                self.current_epoch += 1
                progress.update(
                    task,
                    completed=self.current_epoch,
                    epoch=self.current_epoch,
                    step=self.current_step,
                )

                del rollouts, advantages
                torch.cuda.empty_cache()

                if self.conf.checkpoint_epochs > 0 and (
                    self.current_epoch % self.conf.checkpoint_epochs == 0
                    or self.current_epoch == self.conf.train_epochs
                ):
                    self.save_dcp_checkpoint(self.get_checkpoint_dir(self.current_step))
                    self.rotate_checkpoints_maybe()

                if (
                    self.conf.validation_epochs > 0
                    and self.current_epoch % self.conf.validation_epochs == 0
                ):
                    self.validate_and_log()

        with apply_ema_maybe(self.optimizer):
            self.save_dcp_checkpoint(
                self.get_checkpoint_dir(self.current_step) + "_final"
            )

        console.rule("[bold green]GRPO Training completed[/bold green]")
