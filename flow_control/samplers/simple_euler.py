import torch

from flow_control.adapters import Batch, ModelAdapter

from .base import BaseSampler, make_sample_progress


class SimpleEulerSampler(BaseSampler):
    steps: int = 50

    def sample(
        self,
        model: ModelAdapter,
        batch: Batch,
        negative_batch: Batch | None = None,
        t_start: float = 1.0,
        t_end: float = 0.0,
    ) -> torch.Tensor:
        sigmas = torch.linspace(t_start, t_end, self.steps + 1)
        return self._euler_sample(model, batch, sigmas, negative_batch)

    def _euler_sample(
        self,
        model: ModelAdapter,
        batch: Batch,
        sigmas: torch.Tensor,
        negative_batch: Batch | None = None,
    ) -> torch.Tensor:
        device = model.device
        dtype = model.dtype

        sigmas = sigmas.to(device=device, dtype=torch.float32)
        latents = batch["noisy_latents"].float()

        with make_sample_progress() as progress:
            task = progress.add_task("Sampling", total=self.steps)

            for i in range(self.steps):
                t_cur = sigmas[i : i + 1]
                t_next = sigmas[i + 1 : i + 2]
                timestep = t_cur.to(dtype=dtype)
                velocity = self.get_guided_velocity(
                    model, latents, timestep, batch, negative_batch
                )
                dt = t_next - t_cur
                latents = latents + velocity * dt

                progress.advance(task)

        return latents.to(dtype)
