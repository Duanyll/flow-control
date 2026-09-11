import math
from typing import Literal

import torch

from ..plan import (
    EvalRequest,
    StepContext,
    Transition,
    TransitionGen,
    TransitionResult,
    euler_step,
)
from ..prediction import Predictor
from .base import BaseSolver, solver_registry


@solver_registry.register("cps")
class CPSSolver(BaseSolver):
    type: Literal["cps"] = "cps"

    @staticmethod
    def step_parts(
        latents: torch.Tensor,
        velocity: torch.Tensor,
        sigma: float,
        sigma_next: float,
        eta: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """CPS stochastic update moments: ``(mean, std_dev_t)``."""
        sigma_t = latents.new_tensor(sigma)
        sigma_next_t = latents.new_tensor(sigma_next)
        std_dev_t = sigma_next_t * math.sin(eta * math.pi / 2)
        pred_original_sample = latents - sigma_t * velocity
        noise_estimate = latents + velocity * (1 - sigma_t)
        mean = pred_original_sample * (1 - sigma_next_t) + noise_estimate * torch.sqrt(
            sigma_next_t**2 - std_dev_t**2
        )
        return mean, std_dev_t

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

        if tr.eta == 0.0:
            next_latents = euler_step(latents, velocity, tr.sigma, tr.sigma_next)
        else:
            mean, std_dev_t = self.step_parts(
                latents, velocity, tr.sigma, tr.sigma_next, tr.eta
            )
            noise = torch.randn(
                latents.shape,
                dtype=latents.dtype,
                device=latents.device,
                generator=ctx.generator,
            )
            next_latents = mean + std_dev_t * noise

        return TransitionResult(next_latents=next_latents)
