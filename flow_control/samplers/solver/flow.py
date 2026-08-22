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
class FlowReplayStep(ReplayStep):
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
        mean, std_dev_t, noise_scale = FlowSolver.step_parts(
            latent_t, velocity, self.sigma, self.sigma_next, self.eta
        )
        return StepLogProbOutput(
            log_prob=normal_log_prob(latent_next, mean, noise_scale),
            mean=mean,
            std_dev=std_dev_t,
        )


@solver_registry.register("flow")
class FlowSolver(BaseSolver):
    type: Literal["flow"] = "flow"

    supports_step_log_prob: ClassVar[bool] = True

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

    def make_replay(self, sigma: float, sigma_next: float, eta: float) -> ReplayStep:
        return FlowReplayStep(sigma=sigma, sigma_next=sigma_next, eta=eta)

    def invert(self, plan: SamplingPlan) -> SamplingPlan:
        return invert_first_order(plan)

    def run_transition(self, tr: Transition, ctx: StepContext) -> TransitionGen:
        out = yield EvalRequest(latents=ctx.latents, sigma=tr.sigma)
        latents = ctx.latents
        velocity = out.velocity

        if tr.eta == 0.0:
            next_latents = euler_step(latents, velocity, tr.sigma, tr.sigma_next)
            log_prob = zero_log_prob(latents)
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
            log_prob = normal_log_prob(next_latents, mean, noise_scale)

        recorded = (
            self._make_recorded_step(tr, ctx, next_latents, log_prob)
            if tr.record
            else None
        )
        return TransitionResult(next_latents=next_latents, recorded=recorded)
