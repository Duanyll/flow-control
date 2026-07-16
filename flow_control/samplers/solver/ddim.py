from typing import Literal

import torch

from .base import BaseSolver, SolverState, StepResult, solver_registry


@solver_registry.register("ddim")
class DDIMSolver(BaseSolver):
    type: Literal["ddim"] = "ddim"

    @property
    def supports_step_log_prob(self) -> bool:
        return self.eta > 0.0

    @staticmethod
    def _ddim_update(
        pred_original_sample: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        eta: float,
        prev_sample: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        std_dev_t = eta * sigma_next
        dt_sqrt = torch.sqrt(
            torch.clamp(
                1.0
                - sigma_next**2 * (1 - sigma) ** 2 / (sigma**2 * (1 - sigma_next) ** 2),
                min=0.0,
            )
        )
        noise_scale = std_dev_t * dt_sqrt
        noise_pred = (sample - (1 - sigma) * pred_original_sample) / sigma
        mean = (1 - sigma_next) * pred_original_sample + torch.sqrt(
            torch.clamp(sigma_next**2 - noise_scale**2, min=0.0)
        ) * noise_pred

        next_latents = prev_sample
        if next_latents is None:
            noise = torch.randn(
                sample.shape,
                dtype=sample.dtype,
                device=sample.device,
                generator=generator,
            )
            next_latents = mean + noise_scale * noise

        return next_latents, mean, noise_scale

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
        pred_original_sample = self._velocity_to_x0(velocity, latents, sigma)
        next_latents, mean, noise_scale = self._ddim_update(
            pred_original_sample=pred_original_sample,
            sample=latents,
            sigma=sigma,
            sigma_next=sigma_next,
            eta=step_eta,
            prev_sample=prev_sample,
            generator=generator,
        )

        if step_eta > 0.0:
            log_prob = self._normal_log_prob(next_latents, mean, noise_scale)
        else:
            log_prob = self._zero_log_prob(latents)

        return StepResult(
            next_latents=next_latents,
            log_prob=log_prob,
            mean=mean,
            std_dev=self._scalar_like(noise_scale, latents),
            state=state,
        )
