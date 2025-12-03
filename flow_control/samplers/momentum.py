import torch
from typing import Callable
from pydantic import PrivateAttr
from flow_control.adapters.base import BaseModelAdapter
from .base import BaseSampler


class MomentumGuidedSampler(BaseSampler):
    alpha: float
    beta: float

    _momentum: torch.Tensor | None = PrivateAttr(None)

    def sample(
        self,
        model: BaseModelAdapter,
        batch: dict,
        negative_batch: dict | None = None,
        t_start=1,
        t_end=0,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> torch.Tensor:
        self._momentum = None
        return super().sample(
            model, batch, negative_batch, t_start, t_end, progress_callback
        )

    def get_guided_velocity(
        self,
        model: BaseModelAdapter,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: dict,
        negative_batch: dict | None,
    ) -> torch.Tensor:
        velocity = super().get_guided_velocity(
            model, latents, timestep, batch, negative_batch
        )
        if self._momentum is None:
            self._momentum = velocity
        else:
            guided_velocity = velocity + self.alpha * (velocity - self._momentum)
        self._momentum = (1 - self.beta) * velocity + self.beta * self._momentum
        return guided_velocity
