from dataclasses import dataclass
from typing import Literal

import torch

from .base import BaseSolver, SolverState, StepResult, solver_registry


@dataclass(slots=True)
class FlashSolverState(SolverState):
    num_steps: int
    step_index: int = 0


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
    (~1.2% of mass clipped at 2.5 std). ``eta`` acts as a gate only (the
    sampler zeroes it outside the trajectory window): ``eta == 0`` or a zero
    ``sigma_next`` falls back to the deterministic Euler step, which coincides
    with the re-noise formula at ``sigma' = 0``.
    """

    type: Literal["flash"] = "flash"
    eta: float = 1.0
    noise_scale_start: float = 0.9375
    noise_scale_end: float = 0.9375
    noise_clip_std: float = 2.5

    @property
    def supports_step_log_prob(self) -> bool:
        return self.eta > 0.0

    def init_state(self, num_steps: int) -> FlashSolverState:
        return FlashSolverState(num_steps=num_steps)

    def _noise_scale(self, state: FlashSolverState) -> float:
        if state.num_steps <= 1:
            return self.noise_scale_start
        frac = state.step_index / (state.num_steps - 1)
        return (
            self.noise_scale_start
            + (self.noise_scale_end - self.noise_scale_start) * frac
        )

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
        if not isinstance(state, FlashSolverState):
            msg = f"{self.type} requires FlashSolverState."
            raise TypeError(msg)
        next_state = FlashSolverState(
            num_steps=state.num_steps, step_index=state.step_index + 1
        )

        step_eta = self.eta if eta is None else eta
        if step_eta == 0.0 or bool((sigma_next <= 0.0).all()):
            dt = sigma_next - sigma
            return self._deterministic_step(latents, velocity, dt, state=next_state)

        x0 = self._velocity_to_x0(velocity, latents, sigma)
        mean = (1.0 - sigma_next) * x0
        std_dev = sigma_next * self._noise_scale(state)

        next_latents = prev_sample
        if next_latents is None:
            noise = torch.randn(
                latents.shape,
                dtype=latents.dtype,
                device=latents.device,
                generator=generator,
            )
            if self.noise_clip_std > 0:
                clip_val = self.noise_clip_std * noise.std()
                noise = noise.clamp(min=-clip_val, max=clip_val)
            next_latents = mean + std_dev * noise

        return StepResult(
            next_latents=next_latents,
            log_prob=self._normal_log_prob(next_latents, mean, std_dev),
            mean=mean,
            std_dev=self._scalar_like(std_dev, latents),
            state=next_state,
        )
