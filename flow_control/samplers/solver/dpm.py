from dataclasses import dataclass
from typing import Literal

import torch

from ..plan import (
    EvalRequest,
    SamplingPlan,
    SolverRuntimeState,
    StepContext,
    Transition,
    TransitionGen,
    TransitionResult,
    zero_log_prob,
)
from .base import BaseSolver, solver_registry
from .ddim import DDIMSolver


@dataclass(frozen=True, slots=True)
class DpmTransition(Transition):
    """DPM transition with plan-compiled per-step order metadata."""

    lower_order_final: bool = False
    """Final grid step: fall back to the deterministic DDIM update."""


@dataclass(frozen=True, slots=True)
class DpmRuntimeState(SolverRuntimeState):
    """Rolling x0/sigma history (most recent last), rebuilt per transition."""

    x0_history: tuple[torch.Tensor, ...]
    sigma_history: tuple[float, ...]


@solver_registry.register("dpm")
class DPMSolver(BaseSolver):
    type: Literal["dpm"] = "dpm"
    order: Literal[1, 2]

    def plan(self, sigmas: list[float]) -> SamplingPlan:
        num_steps = len(sigmas) - 1
        return [
            DpmTransition(
                solver=self,
                sigma=sigma,
                sigma_next=sigma_next,
                eta=0.0,
                lower_order_final=index == num_steps - 1,
            )
            for index, (sigma, sigma_next) in enumerate(
                zip(sigmas[:-1], sigmas[1:], strict=True)
            )
        ]

    def run_transition(self, tr: Transition, ctx: StepContext) -> TransitionGen:
        assert isinstance(tr, DpmTransition)
        state = ctx.solver_state
        assert state is None or isinstance(state, DpmRuntimeState)

        out = yield EvalRequest(latents=ctx.latents, sigma=tr.sigma)
        latents = ctx.latents
        sigma_t = latents.new_tensor(tr.sigma)
        x0 = self._velocity_to_x0(out.velocity, latents, sigma_t)

        # Warmup follows the runtime history, so a sliced plan restarts cleanly:
        # empty history behaves like the first step of a full run.
        if state is None or tr.lower_order_final:
            next_latents, _ = DDIMSolver.step_parts(
                latents, out.velocity, tr.sigma, tr.sigma_next, eta=0.0
            )
        elif self.order == 1:
            next_latents = self._dpm_solver_first_order_update(
                model_output=x0,
                sample=latents,
                sigma=sigma_t,
                sigma_next=latents.new_tensor(tr.sigma_next),
            )
        else:
            next_latents = self._multistep_dpm_solver_second_order_update(
                m0=x0,
                m1=state.x0_history[-1],
                sample=latents,
                sigma=sigma_t,
                sigma_next=latents.new_tensor(tr.sigma_next),
                sigma_prev=latents.new_tensor(state.sigma_history[-1]),
            )

        history = state.x0_history if state is not None else ()
        sigma_history = state.sigma_history if state is not None else ()
        keep = self.order - 1
        new_history = (*history, x0)
        new_sigma_history = (*sigma_history, tr.sigma)
        next_state = DpmRuntimeState(
            x0_history=new_history[len(new_history) - keep :],
            sigma_history=new_sigma_history[len(new_sigma_history) - keep :],
        )

        recorded = (
            self._make_recorded_step(tr, ctx, next_latents, zero_log_prob(latents))
            if tr.record
            else None
        )
        return TransitionResult(
            next_latents=next_latents,
            recorded=recorded,
            next_solver_state=next_state,
        )

    def _dpm_solver_first_order_update(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
    ) -> torch.Tensor:
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_next)
        alpha_s, sigma_s = self._sigma_to_alpha_sigma_t(sigma)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s = torch.log(alpha_s) - torch.log(sigma_s)
        h = lambda_t - lambda_s
        return (sigma_t / sigma_s) * sample - (
            alpha_t * (torch.exp(-h) - 1.0)
        ) * model_output

    def _multistep_dpm_solver_second_order_update(
        self,
        m0: torch.Tensor,
        m1: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        sigma_prev: torch.Tensor,
    ) -> torch.Tensor:
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_next)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_prev)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)

        h = lambda_t - lambda_s0
        h_0 = lambda_s0 - lambda_s1
        r0 = h_0 / h
        d0 = m0
        d1 = (1.0 / r0) * (m0 - m1)

        return (
            (sigma_t / sigma_s0) * sample
            - (alpha_t * (torch.exp(-h) - 1.0)) * d0
            - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * d1
        )
