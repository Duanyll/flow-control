"""Rollout collection and advantage computation mixin.

Extracted from GrpoTrainer so that NFT and other RL trainers can reuse the
rollout / reward / advantage pipeline.
"""

from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from pydantic import BaseModel
from rich.progress import Progress
from torchdata.stateful_dataloader import StatefulDataLoader

from flow_control.adapters import ModelAdapter
from flow_control.adapters.base import Batch
from flow_control.datasets import DatasetConfig, parse_dataset
from flow_control.processors import Processor
from flow_control.rewards import Reward, execute_reward
from flow_control.samplers import SampleOutput, Sampler
from flow_control.utils.logging import console, get_logger
from flow_control.utils.tensor import (
    deep_cast_float_dtype,
    deep_move_to_device,
)

from ..advantage import Advantage, PerPromptAdvantage
from ..data import DistributedKRepeatSampler, PaddingAwareDatasetWrapper, collate_fn
from .hsdp import HsdpMixin
from .logging import LoggingMixin

logger = get_logger(__name__)


@dataclass
class Rollout:
    trajectory: SampleOutput
    reward: torch.Tensor
    key: str
    batch: Batch
    negative_batch: Batch | None


class RolloutMixin(LoggingMixin, HsdpMixin, BaseModel):
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
    advantage: Advantage = PerPromptAdvantage()

    dataset: DatasetConfig
    num_dataloader_workers: int = 1

    model: ModelAdapter
    processor: Processor
    reward: Reward
    rollout_sampler: Sampler

    _rollout_needs_trajectory: bool = True
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

    def _collect_rollouts(self, epoch: int) -> list[Rollout]:
        """Rollout phase: generate images, decode, then score rewards concurrently."""
        rollouts: list[Rollout] = []
        transformer = self.model.transformer
        model = self.model
        processor = self.processor
        reward: Reward = self.reward
        device = self.device
        seed: int = self.seed
        rank: int = self.rank
        world_size: int = self.world_size
        dataloader = self._dataloader
        get_progress_columns = self.get_progress_columns

        transformer.eval()

        total_rollouts = self.num_batches_per_epoch * self.rollout_batch_per_rank

        progress = Progress(
            *get_progress_columns(),
            console=console,
            transient=True,
        )
        rollout_task = progress.add_task("Rollout", total=total_rollouts)

        dataloader.sampler.set_epoch(epoch)  # type: ignore[union-attr]

        def rollout_submitter() -> Generator[tuple[dict[str, Any], int]]:
            with progress:
                for batch_idx, batch in enumerate(dataloader):
                    batch = deep_move_to_device(batch, device)
                    batch = deep_cast_float_dtype(batch, model.dtype)

                    generator = torch.Generator(device=device).manual_seed(
                        seed + epoch * 10000 + batch_idx * world_size + rank
                    )
                    processor.initialize_latents(
                        batch,
                        generator=generator,
                        device=device,
                    )

                    negative_batch: Any = (
                        processor.get_negative_batch(batch)
                        if self.rollout_sampler.cfg_scale > 1.0
                        else None
                    )

                    with torch.no_grad():
                        rollout_out = self.rollout_sampler.sample(
                            model,
                            batch,
                            negative_batch=negative_batch,
                            return_trajectory=self._rollout_needs_trajectory,
                        )
                        batch["clean_latents"] = rollout_out.final_latents
                        decoded = processor.decode_output(
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

        execute_reward(reward, rollout_submitter(), reward_handler)

        return rollouts

    # ----------------------------- Advantages -------------------------------- #

    def _compute_advantages(self, rollouts: list[Rollout], step: int) -> torch.Tensor:
        """Compute advantages using gathered rewards across all GPUs."""
        device = self.device
        rank: int = self.rank
        world_size: int = self.world_size
        log_metrics = self.log_metrics

        local_rewards = torch.cat([s.reward for s in rollouts], dim=0).to(device)

        gathered_rewards_list: list[torch.Tensor] = [
            torch.zeros_like(local_rewards) for _ in range(world_size)
        ]
        dist.all_gather(gathered_rewards_list, local_rewards)
        gathered_rewards = torch.cat(gathered_rewards_list, dim=0)

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

        advantages = self.advantage.compute(gathered_rewards, prompt_ids)

        reward_mean = gathered_rewards.mean().item()
        reward_std = gathered_rewards.std().item()
        log_metrics(
            {
                "reward/mean": reward_mean,
                "reward/std": reward_std,
                "reward/adv_abs_mean": advantages.abs().mean().item(),
            },
            step=step,
        )

        local_b = local_rewards.shape[0]
        start = rank * local_b
        end = start + local_b
        return advantages[start:end].cpu()
