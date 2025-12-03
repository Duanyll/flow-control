import torch
from typing import Callable, Any, cast
from einops import repeat

from flow_control.adapters.base import BaseModelAdapter
from .base import BaseSampler


class SimpleEulerSampler(BaseSampler):
    steps: int = 50

    def sample(
        self,
        model: BaseModelAdapter,
        batch: dict,
        negative_batch: dict | None = None,
        t_start=1.0,
        t_end=0.0,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> torch.Tensor:
        sigmas = torch.linspace(t_start, t_end, self.steps + 1)
        return self._euler_sample(
            model, batch, sigmas, negative_batch, progress_callback
        )

    def _euler_sample(
        self,
        model: BaseModelAdapter,
        batch: dict,
        sigmas: torch.Tensor,
        negative_batch: dict | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> torch.Tensor:
        device = batch["noisy_latents"].device
        dtype = batch["noisy_latents"].dtype
        b, c, h, w = batch["noisy_latents"].shape

        latents = batch["noisy_latents"].float()

        for i in range(self.steps):
            t_cur = sigmas[i]
            t_next = sigmas[i + 1]
            timestep = repeat(t_cur, "1 -> b", b=b).to(device)
            velocity = self.get_guided_velocity(
                model, latents, timestep, batch, negative_batch
            )
            dt = t_next - t_cur
            latents = latents + velocity * dt

            if progress_callback is not None:
                progress_callback(i + 1, self.steps)

        return latents.to(dtype)
