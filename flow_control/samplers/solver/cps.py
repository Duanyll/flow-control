import math
from typing import Literal

import torch

from .base import BaseSolver, SolverState, StepResult, solver_registry


@solver_registry.register("cps")
class CPSSolver(BaseSolver):
    type: Literal["cps"] = "cps"

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
        dt = sigma_next - sigma
        if step_eta == 0.0:
            return self._deterministic_step(latents, velocity, dt, state=state)

        std_dev_t = sigma_next * math.sin(step_eta * math.pi / 2)
        pred_original_sample = latents - sigma * velocity
        noise_estimate = latents + velocity * (1 - sigma)
        mean = pred_original_sample * (1 - sigma_next) + noise_estimate * torch.sqrt(
            sigma_next**2 - std_dev_t**2
        )

        next_latents = prev_sample
        if next_latents is None:
            noise = torch.randn(
                latents.shape,
                dtype=latents.dtype,
                device=latents.device,
                generator=generator,
            )
            next_latents = mean + std_dev_t * noise

        log_prob = -((next_latents.detach() - mean) ** 2)
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

        return StepResult(
            next_latents=next_latents,
            log_prob=log_prob,
            mean=mean,
            std_dev=self._scalar_like(std_dev_t, latents),
            state=state,
        )
