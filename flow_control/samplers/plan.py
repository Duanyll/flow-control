"""Pure sampling plans and per-run state shared by solvers and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .calls import Calls

if TYPE_CHECKING:
    from .prediction import Predictor
    from .solver.base import BaseSolver


class SolverRuntimeState:
    """Live multistep history owned by one sampling run."""


@dataclass(slots=True)
class EvalRequest:
    latents: torch.Tensor
    sigma: float | torch.Tensor
    sigma_next: float | None = None
    eta: float = 0.0
    solver: BaseSolver | None = None
    variant: str | None = None


@dataclass(slots=True)
class StepContext:
    latents: torch.Tensor
    generator: torch.Generator | None
    solver_state: SolverRuntimeState | None
    item_index: int = 0
    num_items: int = 1


def euler_step(
    latents: torch.Tensor,
    velocity: torch.Tensor,
    sigma: float,
    sigma_next: float,
) -> torch.Tensor:
    dt = latents.new_tensor(sigma_next) - latents.new_tensor(sigma)
    return latents + velocity * dt


@dataclass(slots=True)
class TransitionResult:
    next_latents: torch.Tensor
    next_solver_state: SolverRuntimeState | None = None


@dataclass(frozen=True, slots=True)
class Transition:
    solver: BaseSolver
    sigma: float
    sigma_next: float
    eta: float = 0.0

    def run(self, ctx: StepContext, predict: Predictor) -> TransitionGen:
        return self.solver.run_transition(self, ctx, predict)


SamplingPlan = list[Transition]
type TransitionGen = Calls[TransitionResult]
