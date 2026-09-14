"""GRPO (Group Relative Policy Optimization) over recorded rollout steps.

The loop is :class:`~flow_control.training.rollout_trainer.RolloutTrainerBase`.
What sets GRPO apart from the endpoint family: the train points are the
stochastic transitions recorded during the rollout (replayed through
``train_predictor``), the old policy is the log-prob recorded at rollout time
rather than an EMA, and the loss is the clipped policy ratio plus an optional
KL to the reference model's step mean.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch

from flow_control.samplers import SampleRequest
from flow_control.utils.logging import get_logger
from flow_control.utils.tensor import deep_move_to_device

from .grpo_sampling import RecordedStep, ReplayItem, replay_steps
from .mixins import Rollout, trainer_registry
from .rollout_trainer import RolloutTrainerBase

logger = get_logger(__name__)


@dataclass(slots=True)
class GrpoTrainItem:
    rollout_idx: int
    timestep_idx: int
    cached_ref_mean: torch.Tensor | None = None


@trainer_registry.register("grpo")
class GrpoTrainer(RolloutTrainerBase[GrpoTrainItem]):
    training_type: str = "grpo"

    _ROLLOUT_RECORD_STEPS: ClassVar[bool] = True

    clip_range: float = 1e-4
    adv_clip_max: float = 5.0
    kl_beta: float = 0.0

    # --------------------------------- Hooks ------------------------------------ #

    def _needs_reference(self) -> bool:
        return self.kl_beta > 0

    def _check_rollouts(self, rollouts: list[Rollout]) -> None:
        """Fail fast on rollouts that GRPO's step replay cannot train on."""
        for rollout in rollouts:
            trajectory = rollout.recorded_steps
            if not trajectory:
                raise RuntimeError(
                    "GRPO rollout produced no recorded trajectory steps; step "
                    "replay needs at least one recorded transition. Check the "
                    "rollout_sampler solver eta and sde_window transforms."
                )
            if not any(bool((step.log_prob != 0).any()) for step in trajectory):
                raise RuntimeError(
                    "GRPO rollout recorded only deterministic steps (all step "
                    "log-probs are zero), which cannot train a policy ratio. "
                    f"Solver '{self.rollout_sampler.solver.type}' ran with "
                    f"eta={self.rollout_sampler.solver.eta}; set eta > 0 and "
                    "make sure the sde_window covers stochastic steps."
                )

    def _build_train_plan(self, rollouts: list[Rollout]) -> list[list[GrpoTrainItem]]:
        """Every inner epoch shuffles the rollout order while keeping each
        rollout's recorded steps contiguous and in trajectory order."""
        item_groups: list[list[GrpoTrainItem]] = []
        for rollout_idx, rollout in enumerate(rollouts):
            trajectory = rollout.recorded_steps
            assert trajectory, "validated by _check_rollouts"
            item_groups.append(
                [
                    GrpoTrainItem(rollout_idx=rollout_idx, timestep_idx=timestep_idx)
                    for timestep_idx in range(len(trajectory))
                ]
            )
        train_plan: list[list[GrpoTrainItem]] = []
        for _ in range(self.num_inner_epochs):
            perm = torch.randperm(len(item_groups)).tolist()
            train_plan.append([item for index in perm for item in item_groups[index]])
        return train_plan

    # --------------------------------- Loss ------------------------------------- #

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

        log_prob, old_log_prob = log_prob.float(), old_log_prob.float()
        advantages = torch.clamp(advantages.float(), -adv_clip_max, adv_clip_max)
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
            mean, ref_mean, std_dev = mean.float(), ref_mean.float(), std_dev.float()
            kl_loss = ((mean - ref_mean) ** 2).mean(dim=tuple(range(1, mean.ndim))) / (
                2 * std_dev**2
            )
            kl_loss = torch.mean(kl_loss)
            loss = loss + kl_beta * kl_loss
            metrics["train/kl_loss"] = kl_loss.detach()

        metrics["train/loss"] = loss.detach()
        self.log_aggregated_metrics(metrics)
        return loss

    def _make_replay_item(
        self,
        rollout: Rollout,
        timestep_idx: int,
    ) -> tuple[ReplayItem, torch.Tensor]:
        trajectory = rollout.recorded_steps
        assert trajectory, "validated by _check_rollouts"
        recorded: RecordedStep = deep_move_to_device(
            trajectory[timestep_idx], self.device
        )
        row = deep_move_to_device(rollout.row, self.device)
        run = self.rollout_sampler.make_run(
            SampleRequest(
                row=row,
                negative_row=self.training_negative(
                    row,
                    len(rollout.sampling_plan),
                    deep_move_to_device(rollout.negative_row, self.device),
                ),
            ),
            plan=rollout.sampling_plan,
            predictor=self.train_predictor,
        )
        return ReplayItem(run, recorded), recorded.log_prob

    def _loss_batched(
        self,
        items: list[GrpoTrainItem],
        rollouts: list[Rollout],
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        replay_items: list[ReplayItem] = []
        old_log_probs: list[torch.Tensor] = []
        for item in items:
            replay_item, old_log_prob = self._make_replay_item(
                rollouts[item.rollout_idx], item.timestep_idx
            )
            replay_items.append(replay_item)
            old_log_probs.append(old_log_prob)

        replay_outputs = replay_steps(self.model, replay_items)
        uncached_reference_outputs = None
        if self.kl_beta > 0 and any(item.cached_ref_mean is None for item in items):
            with torch.no_grad(), self.reference_model():
                uncached_reference_outputs = replay_steps(self.model, replay_items)

        losses: list[torch.Tensor] = []
        for index, (item, replay_output, old_log_prob) in enumerate(
            zip(items, replay_outputs, old_log_probs, strict=True)
        ):
            ref_mean = (
                item.cached_ref_mean.to(device=self.device)
                if item.cached_ref_mean is not None
                else (
                    uncached_reference_outputs[index].mean
                    if uncached_reference_outputs is not None
                    else None
                )
            )
            losses.append(
                self.grpo_loss(
                    log_prob=replay_output.log_prob,
                    old_log_prob=old_log_prob,
                    advantages=advantages[item.rollout_idx].to(device=self.device),
                    mean=replay_output.mean,
                    ref_mean=ref_mean,
                    std_dev=replay_output.std_dev,
                )
            )
        return torch.stack(losses).mean()

    # ------------------------------- Precompute --------------------------------- #

    def _precompute(
        self,
        rollouts: list[Rollout],
        train_plan: list[list[GrpoTrainItem]],
        advantages: torch.Tensor,
    ) -> None:
        """Fill ``item.cached_ref_mean`` for every item using the reference model."""
        if self.kl_beta <= 0:
            return
        # Inner epochs reorder the same item objects, so cache each item once.
        items = list(
            {id(item): item for items in train_plan for item in items}.values()
        )

        with self._precompute_scope() as progress, self.reference_model():
            precompute_task = progress.add_task("Precompute ref", total=len(items))
            for micro_items in self.iter_train_micro_batches(items):
                replay_items = [
                    self._make_replay_item(
                        rollouts[item.rollout_idx], item.timestep_idx
                    )[0]
                    for item in micro_items
                ]
                outputs = replay_steps(self.model, replay_items)
                for item, output in zip(micro_items, outputs, strict=True):
                    item.cached_ref_mean = output.mean.detach()
                progress.advance(precompute_task, advance=len(micro_items))
