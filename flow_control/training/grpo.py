"""GRPO (Group Relative Policy Optimization) over recorded rollout steps.

The loop is :class:`~flow_control.training.rollout_trainer.RolloutTrainerBase`.
What sets GRPO apart from the endpoint family: the train points are the
stochastic transitions recorded during the rollout (replayed through
``train_predictor``), the old policy is the log-prob recorded at rollout time,
and the loss is the clipped policy ratio plus an optional KL to the ``ref``
variant's step mean.
"""

from dataclasses import dataclass
from typing import cast

import torch

from flow_control.samplers import (
    BaseTransition,
    Observer,
    RootFrame,
    Row,
    StepFrame,
    StepLogProb,
    data,
)
from flow_control.utils.logging import get_logger
from flow_control.utils.tensor import deep_move_to_device

from .mixins import trainer_registry
from .rollout_trainer import RolloutTrainerBase

logger = get_logger(__name__)


@dataclass(slots=True)
class GrpoTrainItem:
    rollout_idx: int
    timestep_idx: int


def record(row: Row, step: StepFrame) -> None:
    """Observer: keep each stochastic step's point and likelihood on the row."""
    node = cast(BaseTransition, step.node)
    if not node.is_stochastic(step):
        return
    assert step.velocity is not None
    point = {
        "noisy_latents": step.latents,
        "sigma": step.sigma,
        "next_latents": step.next_latents,
        "_frame": step.snapshot(),
    }
    point["log_prob"] = node.log_prob({**data(row), **point}, step.velocity).log_prob
    row.setdefault("trajectory", []).append(point)


@trainer_registry.register("grpo")
class GrpoTrainer(RolloutTrainerBase[GrpoTrainItem]):
    training_type: str = "grpo"

    clip_range: float = 1e-4
    adv_clip_max: float = 5.0
    kl_beta: float = 0.0

    # --------------------------------- Hooks ------------------------------------ #

    def _required_variants(self) -> frozenset[str]:
        return frozenset({"ref"} if self.kl_beta > 0 else ())

    def _rollout_observers(self) -> list[tuple[type, Observer]]:
        return [(BaseTransition, record)]

    def _check_rollouts(self, rows: list[Row]) -> None:
        for row in rows:
            if not row.get("trajectory"):
                raise RuntimeError(
                    "GRPO replays stochastic steps but the rollout recorded none; "
                    "give rollout.sampler's transition eta > 0 (or an sde_window_grid "
                    "with a nonempty window)."
                )

    def _build_train_plan(self, rows: list[Row]) -> list[list[GrpoTrainItem]]:
        """Every inner epoch shuffles the rollout order while keeping each
        rollout's recorded steps contiguous and in trajectory order."""
        item_groups = [
            [
                GrpoTrainItem(rollout_idx, timestep_idx)
                for timestep_idx in range(len(row["trajectory"]))
            ]
            for rollout_idx, row in enumerate(rows)
        ]
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
        mean: torch.Tensor,
        std_dev: torch.Tensor,
        ref_mean: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute GRPO loss.

        L = E[max(-adv * ratio, -adv * clip(ratio, 1-eps, 1+eps))] + beta * KL
        """
        adv_clip_max = self.adv_clip_max
        clip_range = self.clip_range

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
        if ref_mean is not None:
            mean, ref_mean, std_dev = mean.float(), ref_mean.float(), std_dev.float()
            kl_loss = ((mean - ref_mean) ** 2).mean(dim=tuple(range(1, mean.ndim))) / (
                2 * std_dev**2
            )
            kl_loss = torch.mean(kl_loss)
            loss = loss + self.kl_beta * kl_loss
            metrics["train/kl_loss"] = kl_loss.detach()

        metrics["train/loss"] = loss.detach()
        self.log_aggregated_metrics(metrics)
        return loss

    def _point(self, row: Row, timestep_idx: int) -> Row:
        """The recorded step as a row of its own, on the device, its chain
        re-rooted at the trainer: the rollout's root names the variant that
        sampled it, while the replay runs under the active one (``default``,
        or ``ref`` for the KL term)."""
        sample = {key: value for key, value in row.items() if key != "trajectory"}
        point = {**sample, **row["trajectory"][timestep_idx]}
        point["_frame"] = point["_frame"].rerooted(RootFrame(self))
        return deep_move_to_device(point, self.device)

    def _replay(self, points: list[Row]) -> list[StepLogProb]:
        """Each recorded step's likelihood under ``train_predictor``'s velocity."""
        return [
            point["_frame"].node.log_prob(point, velocity)
            for point, velocity in zip(
                points, self.predict_training(points), strict=True
            )
        ]

    def _loss_batched(
        self,
        items: list[GrpoTrainItem],
        rows: list[Row],
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        points = [
            self._point(rows[item.rollout_idx], item.timestep_idx) for item in items
        ]
        outputs = self._replay(points)
        ref_means: list[torch.Tensor | None] = [None] * len(points)
        if self.kl_beta > 0:
            with torch.no_grad(), self.model.use_variant("ref"):
                ref_means = [output.mean for output in self._replay(points)]
        losses = [
            self.grpo_loss(
                log_prob=output.log_prob,
                old_log_prob=point["log_prob"],
                advantages=advantages[item.rollout_idx].to(device=self.device),
                mean=output.mean,
                std_dev=output.std_dev,
                ref_mean=ref_mean,
            )
            for item, point, output, ref_mean in zip(
                items, points, outputs, ref_means, strict=True
            )
        ]
        return torch.stack(losses).mean()
