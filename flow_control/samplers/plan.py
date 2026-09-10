"""Pure sampling plans and per-run state shared by solvers and evaluation."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .solver.base import BaseSolver


class SolverRuntimeState:
    """Live multistep history owned by the executor."""


class GuidanceState:
    """Immutable per-run state returned by guidance after each evaluation."""


@dataclass(slots=True)
class EvalRequest:
    latents: torch.Tensor
    sigma: float
    sigma_next: float | None = None
    eta: float = 0.0
    solver: BaseSolver | None = None


@dataclass(slots=True)
class BranchEvals:
    velocities: dict[str, torch.Tensor]
    latents: torch.Tensor
    sigma: float
    sigma_next: float | None = None
    eta: float = 0.0
    solver: BaseSolver | None = None


@dataclass(slots=True)
class GuidanceOutput:
    velocity: torch.Tensor


@dataclass(slots=True)
class StepContext:
    latents: torch.Tensor
    generator: torch.Generator | None
    solver_state: SolverRuntimeState | None
    guidance_state: GuidanceState | None
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

    def run(self, ctx: StepContext) -> TransitionGen:
        return self.solver.run_transition(self, ctx)


SamplingPlan = list[Transition]
TransitionGen = Generator[EvalRequest, GuidanceOutput, TransitionResult]
