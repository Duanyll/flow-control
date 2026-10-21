"""Rollout collection and advantage computation mixin.

Extracted from GrpoTrainer so that NFT and other RL trainers can reuse the
rollout / reward / advantage pipeline. Prompts come from a ``RowCursor`` over
the trainer's store (``docs/data/data.md``): every RL epoch takes ``num_prompts_per_epoch``
distinct prompts, identical on every rank, and each rank keeps every
``world_size``-th prompt whole; the rollout program (``docs/training/rollouts.md``) turns a
prompt into its ``num_rollouts_per_prompt`` scored rows through the ``Runner``.
"""

import json
import time
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.distributed as dist
from pydantic import BaseModel

from flow_control.adapters import ModelAdapter
from flow_control.data import KEY, OnlineStore, PromptSampling, RowCursor
from flow_control.rewards import RewardProfile, reduce_reward_profiles
from flow_control.samplers import Observer, Row
from flow_control.training.rollouts import Pending, Rollout
from flow_control.utils.logging import get_logger
from flow_control.utils.tensor import deep_cast_float_dtype

from ..advantage import Advantage, PerPromptAdvantage
from .base import BaseTrainer
from .logging import LoggingMixin
from .runner import RunnerMixin

logger = get_logger(__name__)


@dataclass
class PendingRollouts:
    """One epoch's rollouts whose rewards may still be in flight."""

    epoch: int
    """The batch's own epoch number: seeds its prompts and rollout noise."""
    policy_version: int
    """``_current_epoch`` when sampled; ``rollout/policy_lag`` is the gap to the
    epoch that trains on it (0 without ``rollout_lookahead``)."""
    groups: Pending | None
    """The batch's prompt groups on the runner; None for a batch restored from
    a checkpoint, which arrives scored."""
    cursor_state_before: dict[str, int]
    """``RowCursor`` state before this batch took its prompts; the checkpoint
    saves the oldest in-flight batch's so a resume re-takes the same prompts."""
    profile: RewardProfile
    """The batch's remote reward requests."""
    rows: list[Row] | None = None
    """The scored rows, once waited for."""

    def wait(self) -> list[Row]:
        """The scored rows in submission order; collective until they are in."""
        if self.rows is None:
            assert self.groups is not None
            self.rows = [row for rows in self.groups.wait() for row in rows]
        return self.rows


class RolloutMixin(RunnerMixin, LoggingMixin, BaseTrainer, BaseModel):
    """Mixin providing rollout collection and advantage computation."""

    # ---------------------------------- Configs --------------------------------- #
    num_prompts_per_epoch: int
    """
    Distinct prompts drawn per epoch (``M``). Each gets ``num_rollouts_per_prompt``
    rollouts, so an epoch collects ``M * K`` rollouts globally. A prompt's
    rollouts stay on one rank, so ``M`` must be divisible by ``world_size``.
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
    profile_dump: bool = False
    """
    Also write every finished batch's raw reward submit/done timestamps as one
    JSON per (epoch, rank) under ``<run_dir>/reward_profiles`` for offline
    in-flight plots; ``profile/reward/*`` only logs the reduced summary.
    """
    rollout_dump: bool = False
    """
    Also write every epoch's per-rollout rewards (key, raw and normalized
    components, combined reward, advantage) as JSON lines to
    ``<run_dir>/rollout_rows/step_XXXXXXX.jsonl`` on rank 0, for offline
    per-row analysis; ``rollout/*`` only logs aggregates.
    """

    model: ModelAdapter
    rollout: Rollout

    _cursor: RowCursor

    # -------------------------------- Properties -------------------------------- #

    @property
    def rollouts_per_rank(self) -> int:
        """``M * K / world_size`` after checking that the prompts tile the ranks."""
        prompts, per_prompt = self.num_prompts_per_epoch, self.num_rollouts_per_prompt
        if prompts % self.world_size != 0:
            raise ValueError(
                f"A prompt's {per_prompt} rollouts stay on one rank, so "
                f"num_prompts_per_epoch ({prompts}) must be divisible by world_size "
                f"({self.world_size})."
            )
        return prompts // self.world_size * per_prompt

    @property
    def rollout_storage(self) -> torch.device:
        """Where a batch's rollouts live between sampling and training
        (``rollout_storage_device``)."""
        if self.rollout_storage_device == "device":
            return self.device
        return torch.device("cpu")

    def _rollout_observers(self) -> list[tuple[type, Observer]]:
        """Observers on every rollout's chain root (what the trainer records)."""
        return []

    def _rollout_variant(self) -> str:
        """The model variant that samples the rollouts."""
        return "default"

    def _reward_labels(self) -> list[str]:
        """Labels of the ``row["reward"]`` columns the advantage sees: the
        program's, plus any column a trainer appends before the advantage."""
        return self.rollout.labels

    def _reward_weights(self) -> list[float]:
        """Weights of the :meth:`_reward_labels` columns."""
        return self.rollout.weights

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
        ):
            # See the RowCursor alignment caveat: throughput only.
            logger.warning(
                f"{len(store)} rows leave a plan group of {short} real rows "
                f"(group_size {self.group_size}), not a multiple of world_size "
                f"({self.world_size}): once a pass walks past it, one block of "
                "prompts per group straddles two cost groups. Set dataset.limit "
                "to a multiple of group_size to keep every rank on one group."
            )

    def _start_rollouts(self, epoch: int) -> PendingRollouts:
        """Submit batch *epoch*'s prompt groups to the runner and return once
        every group has left ``sample()``, so every model call of the batch ran
        under the current weights; its rewards may still be in flight until
        :meth:`_finish_rollouts`."""
        model = self.model
        store = self._store
        assert store is not None, "make_rollout_cursor() runs before the first epoch"
        model.transformer.eval()

        # Same seed and cursor state everywhere -> every rank draws the same ids,
        # and keeps every world_size-th prompt whole.
        cursor_state_before = self._cursor.state_dict()
        ids = self._cursor.take(self.num_prompts_per_epoch, epoch=epoch)
        mine = ids[self.rank :: self.world_size]

        def prepare(item: Row) -> Row:
            return deep_cast_float_dtype(
                self.prepare_row(item, mode="inference", epoch=epoch), model.dtype
            )

        started = time.perf_counter()
        groups = self.runner.submit(
            ((i, store.get(i)) for i in mine),
            self.rollout.sample,
            self.rollout.score,
            prepare=prepare,
            k=self.num_rollouts_per_prompt,
            epoch=epoch,
            variant=self._rollout_variant(),
            sampler=self.rollout.sampler,
            observers=self._rollout_observers(),
        )
        groups.sampled()
        step = getattr(self, "_current_step", 0)
        self.log_phase_timing(
            "Rollout",
            time.perf_counter() - started,
            len(mine) * self.num_rollouts_per_prompt,
            step,
            prefix="profile/rollout",
        )
        return PendingRollouts(
            epoch=epoch,
            policy_version=getattr(self, "_current_epoch", epoch),
            groups=groups,
            cursor_state_before=cursor_state_before,
            profile=groups.profile,
        )

    def _finish_rollouts(self, pending: PendingRollouts) -> list[Row]:
        """Wait for the batch's rewards and log the reward profile (the part the
        rollout timing cannot see) plus how long the wait actually blocked and
        how many epochs the batch's policy is behind. All of it lands on the
        training epoch's step; the same batch's ``profile/rollout/*`` went out
        at the step it was sampled."""
        started = time.perf_counter()
        rows = pending.wait()
        wait_s = time.perf_counter() - started

        step = getattr(self, "_current_step", 0)
        self.log_reduced_metrics(
            pending.profile.local_payload(), reduce_reward_profiles, step
        )
        current_epoch: int = getattr(self, "_current_epoch", pending.epoch)
        self.log_metrics(
            {
                "rollout/policy_lag": current_epoch - pending.policy_version,
                "profile/reward/wait_at_train_s": wait_s,
            },
            step=step,
        )
        if self.profile_dump:
            out_dir = self.run_dir / "reward_profiles"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"epoch{pending.epoch:05d}_rank{self.rank}.json").write_text(
                json.dumps(
                    {
                        "epoch": pending.epoch,
                        "policy_version": pending.policy_version,
                        "step": step,
                        "wait_at_train_s": wait_s,
                        **pending.profile.timestamps(),
                    }
                )
            )
        return rows

    # ----------------------------- Advantages -------------------------------- #

    def _compute_advantages(self, rows: list[Row], step: int) -> torch.Tensor:
        """Compute advantages using gathered rewards across all GPUs."""
        device = self.device
        rank: int = self.rank
        world_size: int = self.world_size
        log_metrics = self.log_metrics

        # Stack per-sample [C] rewards into [B_local, C]
        local_rewards = torch.stack([row["reward"] for row in rows], dim=0).to(device)
        labels = self._reward_labels()
        if local_rewards.shape[1] != len(labels):
            raise ValueError(
                f"Rows carry {local_rewards.shape[1]} reward components but the "
                f"advantage expects {len(labels)} (the {self.rollout.type!r} "
                f"program's, plus any the trainer appends): {labels}."
            )
        local_raw_rewards = torch.stack([row["reward_raw"] for row in rows], dim=0).to(
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

        local_keys = [row[KEY] for row in rows]
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

        weights = torch.tensor(
            self._reward_weights(),
            device=gathered_rewards.device,
            dtype=gathered_rewards.dtype,
        )
        advantages = self.advantage.compute(gathered_rewards, prompt_ids, weights)

        # Log aggregate reward plus per-component raw/normalized stats. The
        # ``group_std`` entries are the mean over prompts of the within-group
        # std: the spread group-relative advantages actually see, and where a
        # collapse of rollout diversity shows first while the global std holds.
        group_ids = prompt_ids.unique()

        def group_std(values: torch.Tensor) -> float:
            stds = [values[prompt_ids == g].std(correction=0) for g in group_ids]
            return torch.stack(stds).mean().item()

        combined = (gathered_rewards * weights).sum(dim=-1)
        metrics: dict[str, float] = {
            "rollout/reward_mean": combined.mean().item(),
            "rollout/reward_std": combined.std(correction=0).item(),
            "rollout/reward_group_std": group_std(combined),
            "rollout/adv_abs_mean": advantages.abs().mean().item(),
        }
        for i, label in enumerate(labels):
            metrics[f"rollout/raw/{label}_mean"] = (
                gathered_raw_rewards[:, i].mean().item()
            )
            metrics[f"rollout/raw/{label}_std"] = (
                gathered_raw_rewards[:, i].std(correction=0).item()
            )
            metrics[f"rollout/raw/{label}_group_std"] = group_std(
                gathered_raw_rewards[:, i]
            )
            metrics[f"rollout/normalized/{label}_mean"] = (
                gathered_rewards[:, i].mean().item()
            )
            metrics[f"rollout/normalized/{label}_std"] = (
                gathered_rewards[:, i].std(correction=0).item()
            )
        log_metrics(metrics, step=step)
        if self.rollout_dump and rank == 0:
            out_dir = self.run_dir / "rollout_rows"
            out_dir.mkdir(parents=True, exist_ok=True)
            raw, norm = gathered_raw_rewards.tolist(), gathered_rewards.tolist()
            with open(out_dir / f"step_{step:07d}.jsonl", "w") as f:
                for i, key in enumerate(all_keys):
                    record = {
                        "step": step,
                        "key": key,
                        "raw": dict(zip(labels, raw[i], strict=True)),
                        "normalized": dict(zip(labels, norm[i], strict=True)),
                        "weights": weights.tolist(),
                        "combined": combined[i].item(),
                        "advantage": advantages[i].item(),
                    }
                    f.write(json.dumps(record) + "\n")

        local_b = local_rewards.shape[0]
        start = rank * local_b
        end = start + local_b
        return advantages[start:end].cpu()
