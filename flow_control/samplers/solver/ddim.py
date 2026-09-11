from typing import Literal

import torch

from ..plan import (
    EvalRequest,
    StepContext,
    Transition,
    TransitionGen,
    TransitionResult,
)
from ..prediction import Predictor
from .base import BaseSolver, solver_registry


@solver_registry.register("ddim")
class DDIMSolver(BaseSolver):
    type: Literal["ddim"] = "ddim"

    @staticmethod
    def step_parts(
        latents: torch.Tensor,
        velocity: torch.Tensor,
        sigma: float,
        sigma_next: float,
        eta: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """DDIM update moments: ``(mean, noise_scale)``; ``noise_scale`` is the std."""
        sigma_t = latents.new_tensor(sigma)
        sigma_next_t = latents.new_tensor(sigma_next)
        pred_original_sample = latents - sigma_t * velocity
        std_dev_t = eta * sigma_next_t
        dt_sqrt = torch.sqrt(
            torch.clamp(
                1.0
                - sigma_next_t**2
                * (1 - sigma_t) ** 2
                / (sigma_t**2 * (1 - sigma_next_t) ** 2),
                min=0.0,
            )
        )
        noise_scale = std_dev_t * dt_sqrt
        noise_pred = (latents - (1 - sigma_t) * pred_original_sample) / sigma_t
        mean = (1 - sigma_next_t) * pred_original_sample + torch.sqrt(
            torch.clamp(sigma_next_t**2 - noise_scale**2, min=0.0)
        ) * noise_pred
        return mean, noise_scale

    def run_transition(
        self, tr: Transition, ctx: StepContext, predict: Predictor
    ) -> TransitionGen:
        velocity = yield from predict(
            EvalRequest(
                latents=ctx.latents,
                sigma=tr.sigma,
                sigma_next=tr.sigma_next,
                eta=tr.eta,
                solver=self,
            ),
            ctx,
        )
        latents = ctx.latents

        mean, noise_scale = self.step_parts(
            latents, velocity, tr.sigma, tr.sigma_next, tr.eta
        )

        if tr.eta == 0.0:
            # Deterministic DDIM: the legacy step drew (and zero-multiplied) a
            # useless randn here; the plan path drops the draw.
            next_latents = mean
        else:
            noise = torch.randn(
                latents.shape,
                dtype=latents.dtype,
                device=latents.device,
                generator=ctx.generator,
            )
            next_latents = mean + noise_scale * noise

        return TransitionResult(next_latents=next_latents)
