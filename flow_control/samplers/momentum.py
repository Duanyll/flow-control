import torch

from flow_control.adapters import Batch, ModelAdapter

from .euler import EulerSampler, SdeTrajectory


class MomentumGuidedSampler(EulerSampler):
    alpha: float
    beta: float

    _momentum: torch.Tensor | None = None

    def _sample(
        self,
        model: ModelAdapter,
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
        model: ModelAdapter,
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
