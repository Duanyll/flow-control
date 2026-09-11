"""Rollout collection and advantage computation mixin.

Extracted from GrpoTrainer so that NFT and other RL trainers can reuse the
rollout / reward / advantage pipeline.
"""

from collections.abc import Generator, Iterator
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, cast

import torch
import torch.distributed as dist
from pydantic import BaseModel
from rich.progress import Progress
from torchdata.stateful_dataloader import StatefulDataLoader

from flow_control.adapters import ModelAdapter
from flow_control.adapters.base import Batch
from flow_control.datasets import DatasetConfig
from flow_control.processors import Processor
from flow_control.rewards import (
    Reward,
    RewardProfile,
    _has_pairwise_child,
    execute_pairwise_reward,
    execute_reward,
    reduce_reward_profiles,
)
from flow_control.rewards.base import RewardResult
from flow_control.samplers import Sampler, SampleRequest, derive_seed
from flow_control.samplers.plan import Transition
from flow_control.utils.logging import console
from flow_control.utils.tensor import (
    deep_cast_float_dtype,
    deep_move_to_device,
)

from ..advantage import Advantage, PerPromptAdvantage
from ..data import (
    DistributedKRepeatSampler,
    PaddingAwareDatasetWrapper,
    collate_fn,
    seed_worker,
)
from ..grpo_sampling import GrpoCollector, RecordedStep
from .base import BaseTrainer
from .logging import LoggingMixin
from .preprocess import PreprocessMixin


@dataclass
class Rollout:
    sampling_plan: list[Transition]
    """The plan the sampler executed; ``batch["clean_latents"]`` is its endpoint."""
    reward: torch.Tensor
    raw_reward: torch.Tensor
    reward_weights: torch.Tensor
    reward_labels: list[str]
    key: str
    batch: Batch
    negative_batch: Batch | None
    recorded_steps: list[RecordedStep] | None = None


class RolloutMixin(PreprocessMixin, LoggingMixin, BaseTrainer, BaseModel):
    """Mixin providing rollout collection and advantage computation.

    Subclasses must implement :pyattr:`rollout_sampler_instance` and provide the
    fields consumed here (``num_batches_per_epoch``, ``reward``, etc.).
    """

    # ---------------------------------- Configs --------------------------------- #
    num_batches_per_epoch: int
    """
    Number of "batches" to generate per epoch. The actual micro batch size on GPU is
    always 1. This means in each epoch, we will select `num_prompts_per_batch` unique
    prompts for `num_batches_per_epoch` times, that is `num_batches_per_epoch *
    num_prompts_per_batch` prompts in total (may have duplicates across batches).
    """
    num_prompts_per_batch: int
    """
    Number of unique prompts to select for each batch. See `num_batches_per_epoch` for details.
    """
    num_rollouts_per_prompt: int
    """
    Number of rollouts to generate for each prompt.
    """
    rollout_storage_device: Literal["cpu", "device"] = "cpu"
    """
    Where to store collected rollout results between the rollout and training phases.
    ``"cpu"`` reduces accelerator memory pressure. ``"device"`` keeps rollout
    tensors on the current training device to avoid host-device transfers.
    """
    advantage: Advantage = PerPromptAdvantage()

    dataset: DatasetConfig
    num_dataloader_workers: int = 1

    model: ModelAdapter
    processor: Processor
    reward: Reward
    rollout_sampler: Sampler
    _ROLLOUT_RECORD_STEPS: ClassVar[bool] = False
    """Whether rollouts keep GRPO's stochastic-step records for replay."""

    _dataloader: StatefulDataLoader

    # -------------------------------- Properties -------------------------------- #

    @property
    def rollout_batch_per_rank(self) -> int:
        """Number of rollouts per rank per batch (derived from global settings)."""
        total = self.num_prompts_per_batch * self.num_rollouts_per_prompt
        world_size: int = getattr(self, "world_size", 1)
        if total % world_size != 0:
            raise ValueError(
                f"num_prompts_per_batch * num_rollouts_per_prompt ({total}) "
                f"must be divisible by world_size ({world_size})."
            )
        return total // world_size

    # ----------------------------- Rollout phase ----------------------------- #

    def make_rollout_dataloader(self):
        dataset = PaddingAwareDatasetWrapper(self.parse_inference_dataset(self.dataset))
        use_pairwise = _has_pairwise_child(self.reward)
        sampler = DistributedKRepeatSampler(
            dataset=dataset,
            num_batches_per_epoch=self.num_batches_per_epoch,
            num_prompts_per_batch=self.num_prompts_per_batch,
            num_rollouts_per_prompt=self.num_rollouts_per_prompt,
            num_replicas=self.world_size,
            rank=self.rank,
            seed=self.seed,
            keep_prompt_local=use_pairwise,
        )
        self._dataloader = StatefulDataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=self.num_dataloader_workers,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
        )

    def _collect_rollouts(self, epoch: int) -> list[Rollout]:
        """Rollout phase: sample, decode, then score rewards as each sample finishes."""
        rollouts: list[Rollout] = []
        model = self.model
        processor = self.processor
        sampler = self.rollout_sampler
        collector = GrpoCollector(sampler) if self._ROLLOUT_RECORD_STEPS else None
        device = self.device
        rollout_storage = (
            device if self.rollout_storage_device == "device" else torch.device("cpu")
        )
        dataloader = self._dataloader

        model.transformer.eval()

        total_rollouts = self.num_batches_per_epoch * self.rollout_batch_per_rank
        progress = Progress(
            *self.get_progress_columns(),
            console=console,
            transient=True,
        )
        rollout_task = progress.add_task("Rollout", total=total_rollouts)

        # ``make_rollout_dataloader`` above always builds this sampler.
        cast(DistributedKRepeatSampler, dataloader.sampler).set_epoch(epoch)

        def requests() -> Iterator[SampleRequest]:
            """Preprocess lazily; the sampler pulls a request when it has room."""
            ordinal = 0
            for items in dataloader:
                for item in items:
                    batch = deep_move_to_device(item, device)
                    batch = self.preprocess_for_inference(batch, save_extra=True)
                    batch = deep_cast_float_dtype(batch, model.dtype)
                    key = batch.get("__key__", "unknown")
                    generator = torch.Generator(device=device).manual_seed(
                        derive_seed(
                            self.seed, f"rollout:{epoch}:{self.rank}:{ordinal}:{key}"
                        )
                    )
                    ordinal += 1
                    processor.initialize_latents(
                        batch, generator=generator, device=device, dtype=model.dtype
                    )
                    yield self.build_sample_request(sampler, batch, generator)

        def rollout_submitter() -> Generator[tuple[dict[str, Any], int]]:
            with progress, torch.no_grad():
                for run in sampler.sample(model, requests(), collector=collector):
                    batch: Any = run.batch
                    batch["clean_latents"] = run.ctx.latents
                    batch.update(processor.decode_output(run.ctx.latents, batch))
                    rollouts.append(
                        Rollout(
                            sampling_plan=run.plan,
                            recorded_steps=(
                                deep_move_to_device(
                                    collector.take(run),
                                    rollout_storage,
                                    preserve_aliases=True,
                                )
                                if collector is not None
                                else None
                            ),
                            reward=torch.zeros(1),  # placeholder
                            raw_reward=torch.zeros(1),  # placeholder
                            reward_weights=torch.ones(1),  # placeholder
                            reward_labels=["reward"],  # placeholder
                            key=batch.get("__key__", "unknown"),
                            batch=deep_move_to_device(batch, rollout_storage),
                            negative_batch=deep_move_to_device(
                                run.negative_batch, rollout_storage
                            ),
                        )
                    )
                    progress.advance(rollout_task)
                    yield batch, len(rollouts) - 1

        def reward_handler(idx: int, result: RewardResult) -> None:
            # Result is [1, C] for the single rollout sample.
            rollouts[idx].reward = result.normalized.squeeze(0).detach().cpu()
            rollouts[idx].raw_reward = result.raw.squeeze(0).detach().cpu()
            rollouts[idx].reward_weights = result.weights.detach().cpu()
            rollouts[idx].reward_labels = result.labels

        reward_profile = RewardProfile()
        if _has_pairwise_child(self.reward):
            execute_pairwise_reward(
                self.reward,
                rollout_submitter(),
                reward_handler,
                num_rollouts_per_prompt=self.num_rollouts_per_prompt,
            )
        else:
            execute_reward(
                self.reward, rollout_submitter(), reward_handler, profile=reward_profile
            )

        # Drain the (now-finished) Rollout progress bar -> GPU production timing,
        # plus the async reward profile (the part the progress bar cannot see).
        step = getattr(self, "_current_step", 0)
        self.log_progress_timing(progress, step, prefix="profile/rollout")
        self.log_reduced_metrics(
            reward_profile.local_payload(), reduce_reward_profiles, step
        )

        return rollouts

    # ----------------------------- Advantages -------------------------------- #

    def _compute_advantages(self, rollouts: list[Rollout], step: int) -> torch.Tensor:
        """Compute advantages using gathered rewards across all GPUs."""
        device = self.device
        rank: int = self.rank
        world_size: int = self.world_size
        log_metrics = self.log_metrics

        # Stack per-sample [C] rewards into [B_local, C]
        local_rewards = torch.stack([s.reward for s in rollouts], dim=0).to(device)
        local_raw_rewards = torch.stack([s.raw_reward for s in rollouts], dim=0).to(
            device
        )

        gathered_rewards_list: list[torch.Tensor] = [
            torch.zeros_like(local_rewards) for _ in range(world_size)
        ]
        dist.all_gather(gathered_rewards_list, local_rewards)
        gathered_rewards = torch.cat(gathered_rewards_list, dim=0)  # [B_global, C]

        gathered_raw_rewards_list: list[torch.Tensor] = [
            torch.zeros_like(local_raw_rewards) for _ in range(world_size)
        ]
        dist.all_gather(gathered_raw_rewards_list, local_raw_rewards)
        gathered_raw_rewards = torch.cat(gathered_raw_rewards_list, dim=0)

        local_keys = [s.key for s in rollouts]
        all_keys_nested: list[Any] = [None] * world_size
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

        component_weights = rollouts[0].reward_weights.to(
            device=gathered_rewards.device,
            dtype=gathered_rewards.dtype,
        )
        component_labels = rollouts[0].reward_labels
        advantages = self.advantage.compute(
            gathered_rewards, prompt_ids, component_weights
        )

        # Log aggregate reward plus per-component raw/normalized stats.
        combined = (gathered_rewards * component_weights).sum(dim=-1)
        metrics: dict[str, float] = {
            "rollout/reward_mean": combined.mean().item(),
            "rollout/reward_std": combined.std(correction=0).item(),
            "rollout/adv_abs_mean": advantages.abs().mean().item(),
        }
        for i, label in enumerate(component_labels):
            metrics[f"rollout/raw/{label}_mean"] = (
                gathered_raw_rewards[:, i].mean().item()
            )
            metrics[f"rollout/raw/{label}_std"] = (
                gathered_raw_rewards[:, i].std(correction=0).item()
            )
            metrics[f"rollout/normalized/{label}_mean"] = (
                gathered_rewards[:, i].mean().item()
            )
            metrics[f"rollout/normalized/{label}_std"] = (
                gathered_rewards[:, i].std(correction=0).item()
            )
        log_metrics(metrics, step=step)

        local_b = local_rewards.shape[0]
        start = rank * local_b
        end = start + local_b
        return advantages[start:end].cpu()
