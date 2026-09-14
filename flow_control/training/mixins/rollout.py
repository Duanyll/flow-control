"""Rollout collection and advantage computation mixin.

Extracted from GrpoTrainer so that NFT and other RL trainers can reuse the
rollout / reward / advantage pipeline. Prompts come from a ``RowCursor`` over
the trainer's store (design §7.3): every RL epoch takes ``num_prompts_per_epoch``
distinct prompts, identical on every rank, and ``expand_rollouts`` (§9.3)
stripes their ``num_rollouts_per_prompt`` rollouts across the ranks.
"""

import copy
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from itertools import groupby
from operator import itemgetter
from typing import Any, ClassVar, Literal

import torch
import torch.distributed as dist
from pydantic import BaseModel
from rich.progress import Progress

from flow_control.adapters import ModelAdapter
from flow_control.adapters.base import Batch
from flow_control.data import (
    KEY,
    OnlineStore,
    PromptSampling,
    RowCursor,
    expand_rollouts,
)
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
from flow_control.utils.logging import console, get_logger
from flow_control.utils.tensor import (
    deep_cast_float_dtype,
    deep_move_to_device,
)

from ..advantage import Advantage, PerPromptAdvantage
from ..grpo_sampling import GrpoCollector, RecordedStep
from .base import BaseTrainer
from .data import DataMixin
from .logging import LoggingMixin

logger = get_logger(__name__)


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


class RolloutMixin(DataMixin, LoggingMixin, BaseTrainer, BaseModel):
    """Mixin providing rollout collection and advantage computation.

    Subclasses must implement :pyattr:`rollout_sampler_instance` and provide the
    fields consumed here (``num_prompts_per_epoch``, ``reward``, etc.).
    """

    # ---------------------------------- Configs --------------------------------- #
    num_prompts_per_epoch: int
    """
    Distinct prompts drawn per epoch (``M``). Each gets ``num_rollouts_per_prompt``
    rollouts, so an epoch collects ``M * K`` rollouts globally and ``M * K /
    world_size`` per rank; ``M * K`` must be divisible by ``world_size`` (``M``
    itself when the reward is pairwise, which keeps a prompt's rollouts on one rank).
    """
    num_rollouts_per_prompt: int
    """
    Number of rollouts to generate for each prompt (``K``).
    """
    prompt_sampling: PromptSampling = "chunked"
    """
    How the epoch's prompts are drawn. ``chunked``: consecutive rows of the
    grouped plan, no repeats within a pass over the dataset, cost-adjacent.
    ``independent``: a fresh random subset every epoch (repeats across epochs;
    baseline reproduction), random / raw-source datasets only.
    """
    rollout_storage_device: Literal["cpu", "device"] = "cpu"
    """
    Where to store collected rollout results between the rollout and training phases.
    ``"cpu"`` reduces accelerator memory pressure. ``"device"`` keeps rollout
    tensors on the current training device to avoid host-device transfers.
    """
    advantage: Advantage = PerPromptAdvantage()

    model: ModelAdapter
    reward: Reward
    rollout_sampler: Sampler
    _ROLLOUT_RECORD_STEPS: ClassVar[bool] = False
    """Whether rollouts keep GRPO's stochastic-step records for replay."""

    _cursor: RowCursor

    # -------------------------------- Properties -------------------------------- #

    @property
    def rollouts_per_rank(self) -> int:
        """``M * K / world_size`` after checking that the rollouts tile the ranks."""
        prompts, per_prompt = self.num_prompts_per_epoch, self.num_rollouts_per_prompt
        total = prompts * per_prompt
        if _has_pairwise_child(self.reward):
            if prompts % self.world_size != 0:
                raise ValueError(
                    f"A pairwise reward keeps all {per_prompt} rollouts of a prompt on "
                    f"one rank, so num_prompts_per_epoch ({prompts}) must be divisible "
                    f"by world_size ({self.world_size})."
                )
        elif total % self.world_size != 0:
            raise ValueError(
                f"num_prompts_per_epoch * num_rollouts_per_prompt ({total}) must be "
                f"divisible by world_size ({self.world_size}) so every rank collects "
                "the same number of rollouts."
            )
        return total // self.world_size

    # ----------------------------- Rollout phase ----------------------------- #

    def make_rollout_cursor(self) -> None:
        """Open the prompt store and the cursor over it; fails on a layout the
        ranks cannot share before any model work starts."""
        store = self.open_inference_store(self.dataset)
        self._store = store
        if self.num_prompts_per_epoch > len(store):
            raise ValueError(
                f"num_prompts_per_epoch ({self.num_prompts_per_epoch}) exceeds the "
                f"{len(store)} rows of the dataset; prompts within an epoch are distinct."
            )
        _ = self.rollouts_per_rank
        self._cursor = RowCursor(
            store,
            planner=self.make_planner(store, shuffle=True),
            seed=self.seed,
            sampling=self.prompt_sampling,
        )
        short = len(store) % self.group_size
        if (
            self.prompt_sampling == "chunked"
            and not isinstance(store, OnlineStore)
            and short % self.world_size != 0
            and (
                _has_pairwise_child(self.reward)
                or self.num_rollouts_per_prompt % self.world_size != 0
            )
        ):
            # See the RowCursor alignment caveat: throughput only.
            logger.warning(
                f"{len(store)} rows leave a plan group of {short} real rows "
                f"(group_size {self.group_size}), not a multiple of world_size "
                f"({self.world_size}): once a pass walks past it, one block of "
                "rollouts per group straddles two cost groups. Set dataset.limit "
                "to a multiple of group_size to keep every rank on one group."
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
        store = self._store
        assert store is not None, "make_rollout_cursor() runs before the first epoch"

        model.transformer.eval()

        # Same seed and cursor state everywhere -> every rank draws the same ids.
        prompt_ids = self._cursor.take(self.num_prompts_per_epoch, epoch=epoch)
        mine = expand_rollouts(
            prompt_ids,
            self.num_rollouts_per_prompt,
            self.rank,
            self.world_size,
            whole_prompts=_has_pairwise_child(self.reward),
        )

        progress = Progress(
            *self.get_progress_columns(),
            console=console,
            transient=True,
        )
        rollout_task = progress.add_task("Rollout", total=len(mine))

        def requests() -> Iterator[SampleRequest]:
            """Prepare lazily; the sampler pulls a request when it has room.

            A prompt is fetched and resampled once and its rollouts share that
            batch through shallow copies: ``initialize_latents`` writes a fresh
            ``noisy_latents`` into each copy and the sampler rebuilds the dict
            per run, so nothing downstream mutates the shared tensors.
            """
            for prompt_id, group in groupby(mine, key=itemgetter(0)):
                batch = self.prepare_row(
                    store.get(prompt_id), mode="inference", epoch=epoch
                )
                batch = deep_cast_float_dtype(batch, model.dtype)
                for _, k in group:
                    rollout_batch = copy.copy(batch)
                    generator = torch.Generator(device=device).manual_seed(
                        derive_seed(self.seed, f"rollout:{epoch}:{prompt_id}:{k}")
                    )
                    processor.initialize_latents(
                        rollout_batch,
                        generator=generator,
                        device=device,
                        dtype=model.dtype,
                    )
                    yield self.build_sample_request(sampler, rollout_batch, generator)

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
                            key=batch[KEY],
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
