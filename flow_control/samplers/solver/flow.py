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


@solver_registry.register("flow")
class FlowSolver(BaseSolver):
    type: Literal["flow"] = "flow"

    @staticmethod
    def step_parts(
        latents: torch.Tensor,
        velocity: torch.Tensor,
        sigma: float,
        sigma_next: float,
        eta: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flow SDE-Euler moments: ``(mean, std_dev_t, noise_scale)``."""
        sigma_t = latents.new_tensor(sigma)
        sigma_next_t = latents.new_tensor(sigma_next)
        dt = sigma_next_t - sigma_t
        sigma_denom = torch.where(sigma_t == 1.0, sigma_next_t, sigma_t)
        std_dev_t = torch.sqrt(sigma_t / (1 - sigma_denom)) * eta
        mean = (
            latents * (1 + std_dev_t**2 / (2 * sigma_t) * dt)
            + velocity * (1 + std_dev_t**2 * (1 - sigma_t) / (2 * sigma_t)) * dt
        )
        noise_scale = std_dev_t * torch.sqrt(-dt)
        return mean, std_dev_t, noise_scale

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
            mean, _, noise_scale = self.step_parts(
                latents, velocity, tr.sigma, tr.sigma_next, tr.eta
            )
            noise = torch.randn(
                latents.shape,
                dtype=latents.dtype,
                device=latents.device,
                generator=ctx.generator,
            )
            next_latents = mean + noise_scale * noise

        return TransitionResult(next_latents=next_latents)
