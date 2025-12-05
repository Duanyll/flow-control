import torch
from einops import repeat

from flow_control.adapters.base import BaseModelAdapter

from .base import BaseSampler, make_sample_progress


class SimpleEulerSampler(BaseSampler):
    steps: int = 50

    def sample(
        self,
        model: BaseModelAdapter,
        batch: dict,
        negative_batch: dict | None = None,
        t_start=1.0,
        t_end=0.0,
    ) -> torch.Tensor:
        sigmas = torch.linspace(t_start, t_end, self.steps + 1)
        return self._euler_sample(
            model, batch, sigmas, negative_batch
        )

    def _euler_sample(
        self,
        model: BaseModelAdapter,
        batch: dict,
        sigmas: torch.Tensor,
        negative_batch: dict | None = None,
    ) -> torch.Tensor:
        device = batch["noisy_latents"].device
        dtype = batch["noisy_latents"].dtype
        b, c, h, w = batch["noisy_latents"].shape

        latents = batch["noisy_latents"].float()

        with make_sample_progress() as progress:
            task = progress.add_task("Sampling", total=self.steps)
            
            for i in range(self.steps):
                t_cur = sigmas[i]
                t_next = sigmas[i + 1]
                timestep = repeat(t_cur, "1 -> b", b=b).to(device)
                velocity = self.get_guided_velocity(
                    model, latents, timestep, batch, negative_batch
                )
                dt = t_next - t_cur
                latents = latents + velocity * dt

                progress.advance(task)


        return latents.to(dtype)
