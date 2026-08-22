from dataclasses import dataclass
from typing import ClassVar, Literal

import torch

from ..plan import (
    EvalRequest,
    ReplayStep,
    SamplingPlan,
    SolverRuntimeState,
    StepContext,
    StepLogProbOutput,
    Transition,
    TransitionGen,
    TransitionResult,
    euler_step,
    normal_log_prob,
    zero_log_prob,
)
from ..transforms import invert_first_order
from .base import BaseSolver, solver_registry


@dataclass(frozen=True, slots=True)
class DanceReplayStep(ReplayStep):
    eta: float

    def logprob(
        self,
        velocity: torch.Tensor,
        latent_t: torch.Tensor,
        latent_next: torch.Tensor,
        solver_state: SolverRuntimeState | None = None,
    ) -> StepLogProbOutput:
        if self.eta == 0.0:
            return self._deterministic(velocity, latent_t)
        mean, std_dev_t = DanceSolver.step_parts(
            latent_t, velocity, self.sigma, self.sigma_next, self.eta
        )
        return StepLogProbOutput(
            log_prob=normal_log_prob(latent_next, mean, std_dev_t),
            mean=mean,
            std_dev=std_dev_t,
        )


@solver_registry.register("dance")
class DanceSolver(BaseSolver):
    type: Literal["dance"] = "dance"

    supports_step_log_prob: ClassVar[bool] = True

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

    def make_replay(self, sigma: float, sigma_next: float, eta: float) -> ReplayStep:
        return DanceReplayStep(sigma=sigma, sigma_next=sigma_next, eta=eta)

    def invert(self, plan: SamplingPlan) -> SamplingPlan:
        # Inversion reuses only the forced-eta=0 first-order Euler path (the
        # stochastic Dance step is undefined for ascending sigma anyway).
        return invert_first_order(plan)

    def run_transition(self, tr: Transition, ctx: StepContext) -> TransitionGen:
        out = yield EvalRequest(latents=ctx.latents, sigma=tr.sigma)
        latents = ctx.latents
        velocity = out.velocity

        if tr.eta == 0.0:
            next_latents = euler_step(latents, velocity, tr.sigma, tr.sigma_next)
            log_prob = zero_log_prob(latents)
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
            log_prob = normal_log_prob(next_latents, mean, std_dev_t)

        recorded = (
            self._make_recorded_step(tr, ctx, next_latents, log_prob)
            if tr.record
            else None
        )
        return TransitionResult(next_latents=next_latents, recorded=recorded)
