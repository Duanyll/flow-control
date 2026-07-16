from typing import Literal

import torch

from .base import BaseSolver, SolverState, StepResult, solver_registry


@solver_registry.register("dance")
class DanceSolver(BaseSolver):
    type: Literal["dance"] = "dance"

    @property
    def supports_step_log_prob(self) -> bool:
        return self.eta > 0.0

    def step(
        self,
        velocity: torch.Tensor,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        prev_sample: torch.Tensor | None = None,
        eta: float | None = None,
        state: SolverState | None = None,
        generator: torch.Generator | None = None,
    ) -> StepResult:
        step_eta = self.eta if eta is None else eta
        dsigma = sigma_next - sigma
        mean = latents + dsigma * velocity

        pred_original_sample = latents - sigma * velocity
        if step_eta > 0.0:
            delta_t = sigma - sigma_next
            std_dev_t = step_eta * torch.sqrt(delta_t)
            score_estimate = -(latents - pred_original_sample * (1 - sigma)) / sigma**2
            mean = mean - 0.5 * step_eta**2 * score_estimate * dsigma
            next_latents = prev_sample
            if next_latents is None:
                noise = torch.randn(
                    latents.shape,
                    dtype=latents.dtype,
                    device=latents.device,
                    generator=generator,
                )
                next_latents = mean + std_dev_t * noise
            log_prob = self._normal_log_prob(next_latents, mean, std_dev_t)
        else:
            std_dev_t = torch.tensor(0.0, device=latents.device, dtype=latents.dtype)
            next_latents = mean if prev_sample is None else prev_sample
            log_prob = self._zero_log_prob(latents)

        return StepResult(
            next_latents=next_latents,
            log_prob=log_prob,
            mean=mean,
            std_dev=self._scalar_like(std_dev_t, latents),
            state=state,
        )
