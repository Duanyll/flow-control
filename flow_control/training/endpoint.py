"""Endpoint training: the objective contract and the trainer loop over it.

:class:`EndpointTrainer` is the endpoint family (NFT / RAM / AWM / weighted
flow matching): the clean rollout endpoint is re-noised at a training
timestep and an :class:`Objective` turns the velocities of the current and
auxiliary variants there into a loss. The concrete trainers are presets over
``objective`` and ``train_timesteps``.

Conventions (flow_control): ``sigma`` in ``[0, 1]`` with ``1`` = pure noise,
``x_t = (1 - sigma) * x0 + sigma * noise`` and the flow-matching target is
``noise - x0``. A training point is a row with ``clean_latents``,
``noisy_latents``, ``noise``, ``sigma`` (float) and ``advantage`` (``[1]``).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Annotated

import torch
from pydantic import BaseModel, ConfigDict

from flow_control.samplers import BaseGrid, GridFrame, Observer, Row
from flow_control.utils.registry import Registry, RegistryUnion
from flow_control.utils.tensor import deep_move_to_device

from .mixins import Rollout
from .rollout_trainer import RolloutTrainerBase
from .train_timesteps import TrainTimesteps


@dataclass(slots=True)
class LossOutput:
    loss: torch.Tensor
    """0-dim, attached to the graph."""
    metrics: dict[str, torch.Tensor]
    """0-dim detached values without prefix; the trainer adds ``train/``."""


def sigma_like(point: Row, like: torch.Tensor) -> torch.Tensor:
    """The point's ``sigma`` as a float32 tensor broadcastable over ``like``."""
    return like.new_tensor(point["sigma"]).view(-1, *([1] * (like.ndim - 1)))


class BaseObjective(BaseModel, ABC):
    type: str
    model_config = ConfigDict(extra="forbid")

    @abstractmethod
    def rollout_variant(self) -> str:
        """The model variant that samples the rollout endpoints."""
        ...

    @abstractmethod
    def required_variants(self) -> frozenset[str]:
        """Auxiliary variants whose velocities :meth:`compute` reads."""
        ...

    @abstractmethod
    def compute(
        self, point: Row, current: torch.Tensor, aux: dict[str, torch.Tensor]
    ) -> LossOutput:
        """Per-item loss and detached metrics; ``aux`` holds the required variants' velocities."""
        ...


objective_registry: Registry[BaseObjective] = Registry("objective", base=BaseObjective)
Objective = Annotated[BaseObjective, RegistryUnion(objective_registry, "type")]


@dataclass(slots=True)
class EndpointTrainItem:
    rollout_idx: int
    sigma: float


def keep_sigmas(row: Row, grid: GridFrame) -> None:
    """Observer: the executed grid, concatenated over sequential grids."""
    row["sigmas"] = [*row.get("sigmas", [])[:-1], *grid.sigmas]


class EndpointTrainer(RolloutTrainerBase[EndpointTrainItem]):
    """Re-noise rollout endpoints and train an :class:`Objective` on them."""

    # The objective members register from the method modules (nft / ram / awm /
    # weighted_fm), which import this class; defer the schema build past import
    # so the ``Objective`` union is not materialized from an empty registry.
    model_config = ConfigDict(extra="forbid", defer_build=True)

    objective: Objective
    train_timesteps: TrainTimesteps

    def _rollout_variant(self) -> str:
        return self.objective.rollout_variant()

    def _required_variants(self) -> frozenset[str]:
        return self.objective.required_variants()

    def _rollout_observers(self) -> list[tuple[type, Observer]]:
        return [(BaseGrid, keep_sigmas)]

    def _build_train_plan(
        self, rollouts: list[Rollout]
    ) -> list[list[EndpointTrainItem]]:
        train_plan: list[list[EndpointTrainItem]] = []
        for _ in range(self.num_inner_epochs):
            items: list[EndpointTrainItem] = []
            for rollout_idx in torch.randperm(len(rollouts)).tolist():
                items.extend(
                    EndpointTrainItem(rollout_idx, sigma)
                    for sigma in self.train_timesteps.draw(
                        rollouts[rollout_idx].row["sigmas"]
                    )
                )
            train_plan.append(items)
        return train_plan

    def _loss_batched(
        self,
        items: list[EndpointTrainItem],
        rollouts: list[Rollout],
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        points: list[Row] = []
        for item in items:
            row = deep_move_to_device(rollouts[item.rollout_idx].row, self.device)
            point = self.noised_point(
                row, item.sigma, torch.randn_like(row["clean_latents"].float())
            )
            point["advantage"] = (
                advantages[item.rollout_idx]
                .to(device=self.device, dtype=torch.float32)
                .view(1)
            )
            points.append(point)
        aux: dict[str, list[torch.Tensor]] = {}
        for name in sorted(self.objective.required_variants()):
            with torch.no_grad(), self.model.use_variant(name):
                aux[name] = self.predict_training(points)
        losses: list[torch.Tensor] = []
        for index, (point, current) in enumerate(
            zip(points, self.predict_training(points), strict=True)
        ):
            out = self.objective.compute(
                point, current, {name: values[index] for name, values in aux.items()}
            )
            self.log_aggregated_metrics(
                {f"train/{key}": value for key, value in out.metrics.items()}
            )
            losses.append(out.loss)
        return torch.stack(losses).mean()
