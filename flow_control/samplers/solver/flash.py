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
from .base import BaseSolver, solver_registry


@dataclass(frozen=True, slots=True)
class FlashTransition(Transition):
    """Flash re-noise step; the per-step noise scale is compiled at plan time."""

    noise_scale: float = 1.0


@dataclass(frozen=True, slots=True)
class FlashReplayStep(ReplayStep):
    eta: float
    noise_scale: float
    """Plan-compiled per-step noise scale; the clipped-noise transition uses an
    approximate Gaussian log-prob."""

    def logprob(
        self,
        velocity: torch.Tensor,
        latent_t: torch.Tensor,
        latent_next: torch.Tensor,
        solver_state: SolverRuntimeState | None = None,
    ) -> StepLogProbOutput:
        if self.eta == 0.0 or self.sigma_next <= 0.0:
            return self._deterministic(velocity, latent_t)
        mean, std_dev = FlashSolver.renoise_parts(
            latent_t, velocity, self.sigma, self.sigma_next, self.noise_scale
        )
        return StepLogProbOutput(
            log_prob=normal_log_prob(latent_next, mean, std_dev),
            mean=mean,
            std_dev=std_dev,
        )


@solver_registry.register("flash")
class FlashSolver(BaseSolver):
    """HiDream-O1 Dev's "flash" sampler: full re-noise at every step.

    Unlike an SDE-Euler step (which perturbs the ODE step with partial noise),
    each step extracts x0 and jumps to ``sigma_next`` with completely fresh
    noise: ``z' = (1 - sigma') * x0 + sigma' * s_noise * clip(eps)``, where
    ``s_noise`` interpolates linearly from ``noise_scale_start`` to
    ``noise_scale_end`` over the run and the noise is clamped at
    ``noise_clip_std`` times its own empirical std. Defaults follow the official
    Dev pipeline (7.5 pixel-space noise scale / 8 latent scaling = 0.9375;
    clip 2.5).

    The transition is Gaussian (``mean = (1 - sigma') * x0``,
    ``std = sigma' * s_noise``), so step log-probs are available for GRPO-style
    replay; with clipping enabled the Gaussian log-prob is an approximation
    (~1.2% of mass clipped at 2.5 std). ``eta`` acts as a gate only (window
    transforms zero it outside the trajectory window): ``eta == 0`` or a zero
    ``sigma_next`` falls back to the deterministic Euler step, which coincides
    with the re-noise formula at ``sigma' = 0``. The per-step noise scale is
    compiled into :class:`FlashTransition` at plan time; no runtime solver
    state remains.
    """

    type: Literal["flash"] = "flash"

    supports_step_log_prob: ClassVar[bool] = True

    eta: float = 1.0
    noise_scale_start: float = 0.9375
    noise_scale_end: float = 0.9375
    noise_clip_std: float = 2.5

    @staticmethod
    def renoise_parts(
        latents: torch.Tensor,
        velocity: torch.Tensor,
        sigma: float,
        sigma_next: float,
        noise_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Flash full re-noise moments: ``(mean, std_dev)``."""
        sigma_t = latents.new_tensor(sigma)
        sigma_next_t = latents.new_tensor(sigma_next)
        pred_original_sample = latents - sigma_t * velocity
        mean = (1.0 - sigma_next_t) * pred_original_sample
        std_dev = sigma_next_t * noise_scale
        return mean, std_dev

    def _noise_scale_at(self, step_index: int, num_steps: int) -> float:
        if num_steps <= 1:
            return self.noise_scale_start
        frac = step_index / (num_steps - 1)
        return (
            self.noise_scale_start
            + (self.noise_scale_end - self.noise_scale_start) * frac
        )

    def plan(self, sigmas: list[float]) -> SamplingPlan:
        num_steps = len(sigmas) - 1
        return [
            FlashTransition(
                solver=self,
                sigma=sigma,
                sigma_next=sigma_next,
                eta=self.eta if index < num_steps - 1 else 0.0,
                noise_scale=self._noise_scale_at(index, num_steps),
            )
            for index, (sigma, sigma_next) in enumerate(
                zip(sigmas[:-1], sigmas[1:], strict=True)
            )
        ]

    def run_transition(self, tr: Transition, ctx: StepContext) -> TransitionGen:
        assert isinstance(tr, FlashTransition)
        out = yield EvalRequest(latents=ctx.latents, sigma=tr.sigma)
        latents = ctx.latents
        velocity = out.velocity

        if tr.eta == 0.0 or tr.sigma_next <= 0.0:
            next_latents = euler_step(latents, velocity, tr.sigma, tr.sigma_next)
            log_prob = zero_log_prob(latents)
        else:
            mean, std_dev = self.renoise_parts(
                latents, velocity, tr.sigma, tr.sigma_next, tr.noise_scale
            )
            noise = torch.randn(
                latents.shape,
                dtype=latents.dtype,
                device=latents.device,
                generator=ctx.generator,
            )
            if self.noise_clip_std > 0:
                clip_val = self.noise_clip_std * noise.std()
                noise = noise.clamp(min=-clip_val, max=clip_val)
            next_latents = mean + std_dev * noise
            log_prob = normal_log_prob(next_latents, mean, std_dev)

        recorded = None
        if tr.record:
            recorded = self._make_recorded_step(
                tr,
                ctx,
                next_latents,
                log_prob,
                replay=FlashReplayStep(
                    sigma=tr.sigma,
                    sigma_next=tr.sigma_next,
                    eta=tr.eta,
                    noise_scale=tr.noise_scale,
                ),
            )
        return TransitionResult(next_latents=next_latents, recorded=recorded)
