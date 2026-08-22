"""Plan-as-data core types for the sampler/solver stack.

A :data:`SamplingPlan` is a plain ``list`` of transition dataclasses produced by
``BaseSolver.plan``: pure data (floats plus a shared solver config reference)
that can be sliced, rewritten and inspected without touching models, latents or
RNG. Transforms over plans live in ``flow_control/samplers/transforms.py``; the
rendezvous loop that executes them lives in ``flow_control/samplers/executor.py``.

Only the solver-agnostic protocol lives here: the transition/eval/record types
plus the update primitives shared by every solver (``euler_step``,
``zero_log_prob``, ``normal_log_prob``). Solver-specific transition subclasses,
runtime state, step formulas and :class:`ReplayStep` subclasses live next to
their solver in ``flow_control/samplers/solver/``.

Everything in this module is process-local runtime data (dataclasses), not
configuration; only ``init=True`` tensor fields are used so instances round-trip
through ``deep_apply_tensor_fn`` / ``dataclasses.replace``.
"""

from __future__ import annotations

import math
from collections.abc import Generator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .solver.base import BaseSolver


class SolverRuntimeState:
    """Base class for live per-run solver state, owned by :class:`StepContext`."""


class GuidanceState:
    """Persistent per-run guidance state owned by :class:`StepContext`.

    Guidance implementations must treat instances, including contained
    tensors, as immutable values. ``combine`` returns a new state instead of
    mutating its input so transition-boundary snapshots remain replayable.
    """


@dataclass(slots=True)
class EvalRequest:
    """A transition's request to the executor: one velocity eval at ``sigma``."""

    latents: torch.Tensor
    sigma: float
    wants_grad: bool = False


@dataclass(slots=True)
class BranchEvals:
    """Raw branch velocities of one eval, handed to ``BaseGuidance.combine``.

    ``uncond`` is ``None`` when the sample ran no real negative pass (guidance
    does not need one, or the sample has no negative batch and its dummy
    forward was discarded).
    """

    cond: torch.Tensor
    uncond: torch.Tensor | None
    latents: torch.Tensor
    sigma: float


@dataclass(slots=True)
class GuidanceOutput:
    """Guided velocity handed back to a transition generator.

    ``branches`` carries the raw branch evals for CFG++-style solvers; it is
    populated by ``BaseGuidance.combine`` (``None`` only when a test harness
    feeds velocities directly).
    """

    velocity: torch.Tensor
    branches: BranchEvals | None = None


@dataclass(slots=True)
class StepContext:
    """Per-run runtime state, owned and advanced by the executor."""

    latents: torch.Tensor
    generator: torch.Generator | None
    solver_state: SolverRuntimeState | None
    guidance_state: GuidanceState | None
    pre_transition_guidance_state: GuidanceState | None = None
    """Snapshot of ``guidance_state`` taken by the executor at each transition
    boundary, before the transition's first eval. ``guidance_state`` advances
    per eval (execution contract rule 7) while ``RecordedStep`` must capture
    the pre-first-eval state; this slot keeps that value addressable."""


@dataclass(slots=True)
class StepLogProbOutput:
    log_prob: torch.Tensor
    mean: torch.Tensor
    std_dev: torch.Tensor


def zero_log_prob(latents: torch.Tensor) -> torch.Tensor:
    return torch.zeros(latents.shape[0], device=latents.device)


def normal_log_prob(
    sample: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    log_prob = (
        -((sample.detach() - mean) ** 2) / (2 * scale**2)
        - torch.log(scale)
        - torch.log(
            torch.sqrt(
                torch.tensor(2 * math.pi, device=sample.device, dtype=sample.dtype)
            )
        )
    )
    return log_prob.mean(dim=tuple(range(1, log_prob.ndim)))


def euler_step(
    latents: torch.Tensor,
    velocity: torch.Tensor,
    sigma: float,
    sigma_next: float,
) -> torch.Tensor:
    dt = latents.new_tensor(sigma_next) - latents.new_tensor(sigma)
    return latents + velocity * dt


@dataclass(frozen=True, slots=True)
class ReplayStep:
    """Pure-float descriptor of a recorded step, replayable for RL training.

    Every recorded step carries the executed sigma pair so replay consumers
    can re-evaluate the model at the recorded pre-step point. The base class
    is the null descriptor for recorded steps that have no step-wise
    transition density (its ``logprob`` raises); solvers with a density
    subclass it (next to their solver) and store the ACTUAL per-step eta
    written into the plan.
    """

    sigma: float
    sigma_next: float

    def logprob(
        self,
        velocity: torch.Tensor,
        latent_t: torch.Tensor,
        latent_next: torch.Tensor,
        solver_state: SolverRuntimeState | None = None,
    ) -> StepLogProbOutput:
        raise NotImplementedError(
            "This recorded step has no step-wise transition density to replay."
        )

    def _deterministic(
        self,
        velocity: torch.Tensor,
        latent_t: torch.Tensor,
    ) -> StepLogProbOutput:
        """Replay of a deterministic (``eta == 0``) step: the Euler update."""
        mean = euler_step(latent_t, velocity, self.sigma, self.sigma_next)
        return StepLogProbOutput(
            log_prob=zero_log_prob(latent_t),
            mean=mean,
            std_dev=latent_t.new_tensor(0.0),
        )


@dataclass(slots=True)
class RecordedStep:
    """One recorded transition of a rollout trajectory (RL consumption)."""

    latent_t: torch.Tensor
    latent_next: torch.Tensor
    log_prob: torch.Tensor
    replay: ReplayStep
    solver_state: SolverRuntimeState | None = None
    guidance_state: GuidanceState | None = None


@dataclass(slots=True)
class TransitionResult:
    next_latents: torch.Tensor
    recorded: RecordedStep | None = None
    next_solver_state: SolverRuntimeState | None = None
    reset_runtime_state: bool = False
    """Clear solver/guidance live state after this transition (renoise jumps)."""


@dataclass(frozen=True, slots=True)
class Transition:
    """One latent-space move ``sigma -> sigma_next``. Pure plan-time data."""

    solver: BaseSolver
    sigma: float
    sigma_next: float
    eta: float = 0.0
    record: bool = False
    save_solver_state: bool = False
    """Capture the pre-step solver state in the RecordedStep (set by planners)."""

    def run(self, ctx: StepContext) -> TransitionGen:
        return self.solver.run_transition(self, ctx)

    def eval_topology(self) -> str:
        """Structural key for the executor's lockstep fingerprint.

        Two plan items with equal keys must request the same eval pattern
        given the same pre-step runtime state. Subclasses extend the key with
        any field that changes their eval count (e.g. ``SaTransition.final``);
        sigma values stay out so resolution-shifted grids still batch.
        """
        return f"{type(self).__name__}:{type(self.solver).__name__}"


@dataclass(frozen=True, slots=True)
class RenoiseTransition:
    """Zero-eval jump back to a higher sigma. Never recorded, holds no solver."""

    sigma: float
    sigma_next: float

    def run(self, ctx: StepContext) -> TransitionGen:
        raise NotImplementedError(
            "RenoiseTransition execution is not implemented yet; it arrives "
            "with time-travel style plan transforms."
        )

    def eval_topology(self) -> str:
        """Zero-eval control transition (see :meth:`Transition.eval_topology`)."""
        return "renoise"


PlanItem = Transition | RenoiseTransition
SamplingPlan = list[PlanItem]
TransitionGen = Generator[EvalRequest, GuidanceOutput, TransitionResult]
