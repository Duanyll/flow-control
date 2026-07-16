from typing import Literal

import torch

from .base import BaseSolver, SolverState, StepResult, solver_registry


@solver_registry.register("flow")
class FlowSolver(BaseSolver):
    type: Literal["flow"] = "flow"

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

        sigma_denom = torch.where(sigma == 1.0, sigma_next, sigma)
        std_dev_t = torch.sqrt(sigma / (1 - sigma_denom)) * step_eta
        mean = (
            latents * (1 + std_dev_t**2 / (2 * sigma) * dt)
            + velocity * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
        )
        noise_scale = std_dev_t * torch.sqrt(-dt)

        next_latents = prev_sample
        if next_latents is None:
            noise = torch.randn(
                latents.shape,
                dtype=latents.dtype,
                device=latents.device,
                generator=generator,
            )
            next_latents = mean + noise_scale * noise

        return StepResult(
            next_latents=next_latents,
            log_prob=self._normal_log_prob(next_latents, mean, noise_scale),
            mean=mean,
            std_dev=self._scalar_like(std_dev_t, latents),
            state=state,
        )
