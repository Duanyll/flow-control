from typing import Literal

import torch

from flow_control.adapters.base import BaseModelAdapter, Batch

from .euler import EulerSampler, SdeTrajectory


class MomentumGuidedSampler(EulerSampler):
    type: Literal["momentum"] = "momentum"
    alpha: float
    beta: float

    _momentum: torch.Tensor | None = None

    def _sample(
        self,
        model: BaseModelAdapter,
        batch: Batch,
        negative_batch: Batch | None = None,
        t_start: float = 1,
        t_end: float = 0,
        return_trajectory: bool = False,
    ) -> torch.Tensor | SdeTrajectory:
        self._momentum = None
        return super()._sample(
            model,
            batch,
            negative_batch,
            t_start,
            t_end,
            return_trajectory=return_trajectory,
        )

    def get_guided_velocity(
        self,
        model: BaseModelAdapter,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: Batch,
        negative_batch: Batch | None,
    ) -> torch.Tensor:
        velocity = super().get_guided_velocity(
            model, latents, timestep, batch, negative_batch
        )
        if self._momentum is None:
            self._momentum = velocity
        assert self._momentum is not None
        guided_velocity = velocity + self.alpha * (velocity - self._momentum)
        self._momentum = (1 - self.beta) * velocity + self.beta * self._momentum
        return guided_velocity
