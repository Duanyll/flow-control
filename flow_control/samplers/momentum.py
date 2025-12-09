import torch

from flow_control.adapters import ModelAdapter

from .simple_euler import SimpleEulerSampler


class MomentumGuidedSampler(SimpleEulerSampler):
    alpha: float
    beta: float

    _momentum: torch.Tensor | None = None

    def sample(
        self,
        model: ModelAdapter,
        batch: dict,
        negative_batch: dict | None = None,
        t_start=1,
        t_end=0,
    ) -> torch.Tensor:
        self._momentum = None
        return super().sample(model, batch, negative_batch, t_start, t_end)

    def get_guided_velocity(
        self,
        model: ModelAdapter,
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
        guided_velocity = velocity + self.alpha * (velocity - self._momentum)
        self._momentum = (1 - self.beta) * velocity + self.beta * self._momentum
        return guided_velocity
