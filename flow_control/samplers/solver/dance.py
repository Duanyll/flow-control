from typing import Literal

import torch

from ..plan import (
    EvalRequest,
    SamplingPlan,
    StepContext,
    Transition,
    TransitionGen,
    TransitionResult,
    euler_step,
)
from ..prediction import Predictor
from .base import BaseSolver, solver_registry


@solver_registry.register("dance")
class DanceSolver(BaseSolver):
    type: Literal["dance"] = "dance"

    @staticmethod
    def step_parts(
        latents: torch.Tensor,
        velocity: torch.Tensor,
        sigma: float,
        sigma_next: float,
        eta: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dance stochastic update moments (``eta > 0``): ``(mean, std_dev_t)``."""
        sigma_t = latents.new_tensor(sigma)
        sigma_next_t = latents.new_tensor(sigma_next)
        dsigma = sigma_next_t - sigma_t
        mean = latents + dsigma * velocity
        pred_original_sample = latents - sigma_t * velocity
        delta_t = sigma_t - sigma_next_t
        std_dev_t = eta * torch.sqrt(delta_t)
        score_estimate = -(latents - pred_original_sample * (1 - sigma_t)) / sigma_t**2
        mean = mean - 0.5 * eta**2 * score_estimate * dsigma
        return mean, std_dev_t

    def plan(self, sigmas: list[float]) -> SamplingPlan:
        plan = super().plan(sigmas)
        for item in plan:
            assert isinstance(item, Transition)
            if item.eta > 0.0 and item.sigma_next >= item.sigma:
                raise NotImplementedError(
                    "DanceSolver's stochastic step is only defined for "
                    f"decreasing sigmas; got {item.sigma} -> {item.sigma_next} "
                    f"with eta={item.eta}."
                )
        return plan

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
