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
class DdimReplayStep(ReplayStep):
    eta: float

    def logprob(
        self,
        velocity: torch.Tensor,
        latent_t: torch.Tensor,
        latent_next: torch.Tensor,
        solver_state: SolverRuntimeState | None = None,
    ) -> StepLogProbOutput:
        if self.eta == 0.0 and self.sigma == 0.0:
            # Mirrors DDIMSolver.run_transition: only inverted plans record a
            # step starting at exactly sigma 0, where the DDIM mean formula
            # divides by sigma; the deterministic update is the Euler step.
            return self._deterministic(velocity, latent_t)
        mean, noise_scale = DDIMSolver.step_parts(
            latent_t, velocity, self.sigma, self.sigma_next, self.eta
        )
        if self.eta == 0.0:
            log_prob = zero_log_prob(latent_t)
        else:
            log_prob = normal_log_prob(latent_next, mean, noise_scale)
        return StepLogProbOutput(log_prob=log_prob, mean=mean, std_dev=noise_scale)


@solver_registry.register("ddim")
class DDIMSolver(BaseSolver):
    type: Literal["ddim"] = "ddim"

    supports_step_log_prob: ClassVar[bool] = True

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

    def make_replay(self, sigma: float, sigma_next: float, eta: float) -> ReplayStep:
        return DdimReplayStep(sigma=sigma, sigma_next=sigma_next, eta=eta)

    def invert(self, plan: SamplingPlan) -> SamplingPlan:
        # Deterministic DDIM on the RF parameterization is algebraically the
        # Euler step, so the shared first-order mirror applies.
        return invert_first_order(plan)

    def run_transition(self, tr: Transition, ctx: StepContext) -> TransitionGen:
        out = yield EvalRequest(latents=ctx.latents, sigma=tr.sigma)
        latents = ctx.latents

        if tr.eta == 0.0 and tr.sigma == 0.0:
            # Only inverted plans start a transition at exactly sigma == 0,
            # where the DDIM mean formula divides by sigma. The deterministic
            # DDIM update equals the Euler step algebraically; take it
            # directly (descending plans never hit this branch).
            next_latents = euler_step(latents, out.velocity, tr.sigma, tr.sigma_next)
            recorded = (
                self._make_recorded_step(tr, ctx, next_latents, zero_log_prob(latents))
                if tr.record
                else None
            )
            return TransitionResult(next_latents=next_latents, recorded=recorded)

        mean, noise_scale = self.step_parts(
            latents, out.velocity, tr.sigma, tr.sigma_next, tr.eta
        )

        if tr.eta == 0.0:
            # Deterministic DDIM: the legacy step drew (and zero-multiplied) a
            # useless randn here; the plan path drops the draw.
            next_latents = mean
            log_prob = zero_log_prob(latents)
        else:
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
