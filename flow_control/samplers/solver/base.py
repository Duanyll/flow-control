from abc import ABC
from typing import ClassVar, Literal

import torch
from pydantic import BaseModel, ConfigDict

from flow_control.utils.registry import Registry

from ..plan import (
    RecordedStep,
    ReplayStep,
    SamplingPlan,
    StepContext,
    Transition,
    TransitionGen,
)


class BaseSolver(BaseModel, ABC):
    type: Literal["base"] = "base"
    model_config = ConfigDict(extra="forbid")

    supports_step_log_prob: ClassVar[bool] = False
    """Static, config-independent capability: can this solver's stochastic
    steps be replayed for step-wise log-probs? Whether a given transition is
    actually stochastic/recorded is decided by its own ``eta`` / ``record``."""

    eta: float = 0.0

    def plan(self, sigmas: list[float]) -> SamplingPlan:
        """Compile a sigma grid into transitions.

        This default is for solvers whose ``eta`` directly expresses per-step
        stochasticity (flow/ddim/cps/dance); the terminal transition is always
        deterministic. Other solvers must override and write their actual
        per-step semantics into the transitions.
        """
        return [
            Transition(
                solver=self,
                sigma=sigma,
                sigma_next=sigma_next,
                eta=self.eta if i < len(sigmas) - 2 else 0.0,
            )
            for i, (sigma, sigma_next) in enumerate(
                zip(sigmas[:-1], sigmas[1:], strict=True)
            )
        ]

    def run_transition(self, tr: Transition, ctx: StepContext) -> TransitionGen:
        """Generator performing one transition; yields EvalRequests for velocities."""
        raise NotImplementedError(
            f"Solver '{self.type}' must implement run_transition()."
        )

    def make_replay(self, sigma: float, sigma_next: float, eta: float) -> ReplayStep:
        """Build the pure-float replay descriptor for one transition.

        The default is the null descriptor used when recording steps that have
        no step-wise transition density (deterministic multistep solvers, SA);
        its ``logprob`` raises if ever invoked.
        """
        return ReplayStep(sigma=sigma, sigma_next=sigma_next)

    def requires_replay_state(self, plan: SamplingPlan, index: int) -> bool:
        """Whether a recorded transition must snapshot pre-step solver state."""
        return False

    def invert(self, plan: SamplingPlan) -> SamplingPlan:
        raise NotImplementedError(
            f"Solver '{self.type}' does not support plan inversion."
        )

    def _make_recorded_step(
        self,
        tr: Transition,
        ctx: StepContext,
        next_latents: torch.Tensor,
        log_prob: torch.Tensor,
        replay: ReplayStep | None = None,
    ) -> RecordedStep:
        """Build the RecordedStep for one recorded transition.

        ``replay`` overrides the default ``make_replay`` descriptor for
        solvers whose transitions carry extra plan-compiled fields (flash).

        ctx.guidance_state advances per eval, so by the time a transition
        builds its RecordedStep it may already be post-eval; the executor
        snapshots the pre-first-eval state at every transition boundary
        (execution contract rule 7).
        """
        return RecordedStep(
            latent_t=ctx.latents,
            latent_next=next_latents,
            log_prob=log_prob,
            replay=replay
            if replay is not None
            else self.make_replay(tr.sigma, tr.sigma_next, tr.eta),
            solver_state=ctx.solver_state if tr.save_solver_state else None,
            guidance_state=ctx.pre_transition_guidance_state,
        )

    @staticmethod
    def _velocity_to_x0(
        velocity: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        return sample - sigma * velocity

    @staticmethod
    def _sigma_to_alpha_sigma_t(
        sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return 1 - sigma, sigma


solver_registry: Registry[BaseSolver] = Registry("solver", base=BaseSolver)
