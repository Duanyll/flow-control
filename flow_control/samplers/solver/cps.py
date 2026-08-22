import math
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
    zero_log_prob,
)
from ..transforms import invert_first_order
from .base import BaseSolver, solver_registry


@dataclass(frozen=True, slots=True)
class CpsReplayStep(ReplayStep):
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
        mean, std_dev_t = CPSSolver.step_parts(
            latent_t, velocity, self.sigma, self.sigma_next, self.eta
        )
        return StepLogProbOutput(
            log_prob=CPSSolver.residual_log_prob(latent_next, mean),
            mean=mean,
            std_dev=std_dev_t,
        )


@solver_registry.register("cps")
class CPSSolver(BaseSolver):
    type: Literal["cps"] = "cps"

    supports_step_log_prob: ClassVar[bool] = True

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

    @staticmethod
    def residual_log_prob(sample: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        """CPS's unnormalized squared-residual objective (not a Gaussian density)."""
        log_prob = -((sample.detach() - mean) ** 2)
        return log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    def make_replay(self, sigma: float, sigma_next: float, eta: float) -> ReplayStep:
        return CpsReplayStep(sigma=sigma, sigma_next=sigma_next, eta=eta)

    def invert(self, plan: SamplingPlan) -> SamplingPlan:
        # Inversion reuses only the forced-eta=0 first-order Euler path.
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
            # CPS keeps its unnormalized squared-residual objective verbatim.
            log_prob = self.residual_log_prob(next_latents, mean)

        recorded = (
            self._make_recorded_step(tr, ctx, next_latents, log_prob)
            if tr.record
            else None
        )
        return TransitionResult(next_latents=next_latents, recorded=recorded)
