from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from flow_control.adapters.base import Batch, SamplerModel
from flow_control.utils.logging import console, get_logger, warn_once
from flow_control.utils.progress import report_progress
from flow_control.utils.tensor import deep_move_to_device

from .executor import (
    Run,
    evaluate_branches,
    execute,
    validate_distributed_request_count,
)
from .guidance import ClassifierFreeGuidance, Guidance
from .plan import (
    BranchEvals,
    GuidanceState,
    RecordedStep,
    SamplingPlan,
    StepContext,
    StepLogProbOutput,
)
from .shift import ConstantShift, Shift
from .solver import FlowSolver, Solver
from .transforms import finalize_replay_state

logger = get_logger(__name__)


def derive_seed(base_seed: int, key: str) -> int:
    """Derive a deterministic per-sample seed from a base seed and a sample key."""
    h = hashlib.sha256(f"{base_seed}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "little") % (2**63)


def make_sample_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description:<20}"),
        BarColumn(complete_style="blue", finished_style="bold blue"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    )


@dataclass(slots=True)
class SampleRequest:
    batch: Batch
    negative_batch: Batch | None = None
    generator: torch.Generator | None = None


@dataclass(slots=True)
class SampleOutput:
    final_latents: torch.Tensor
    timesteps: torch.Tensor
    """The executed sigma grid, independent of trajectory recording.
    ``Sampler.sample`` reports the raw config grid ``sigmas[:-1]``;
    ``run_phases`` reports the plan-compiled grid (each plan item's start
    sigma — for SA this includes the adjusted head/penultimate values)."""
    trajectory: list[RecordedStep] | None = None
    """Recorded steps for RL replay; populated by the recipe runner
    (``run_phases``) when the plan marks transitions with ``record=True``.
    ``Sampler.sample`` never records."""


@dataclass(slots=True)
class ReplayItem:
    """One recorded rollout step plus the conditioning needed to replay it."""

    batch: Batch
    recorded: RecordedStep
    negative_batch: Batch | None = None


class Sampler(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    guidance: Guidance = Field(default_factory=ClassifierFreeGuidance)
    """Branch-combination rule; the default is ``ClassifierFreeGuidance``
    with ``scale=1.0`` (no negative pass)."""

    steps: int = 50
    solver: Solver = Field(default_factory=FlowSolver)
    shift: Shift = Field(default_factory=ConstantShift)
    """Sigma-grid shift; the default ``ConstantShift`` factor of 1.0 is no
    shift. A bare number is a constant factor: ``"shift": 3.0``."""
    sigma_schedule: Literal["linear", "diffusers_flow"] = "linear"
    """Sigma-grid construction. ``diffusers_flow`` reproduces
    ``FlowMatchEulerDiscreteScheduler.set_timesteps`` including its shifted
    training-grid endpoints."""
    num_train_timesteps: int = 1000
    custom_sigmas: list[float] | None = None
    """Explicit sigma grid of length ``steps + 1`` (descending, terminal usually
    0.0), e.g. a distilled model's official timestep table. When set it replaces
    the linspace grid and *bypasses* ``shift`` (and the ``t_start``/``t_end``
    arguments of :meth:`make_sigmas`)."""

    @staticmethod
    def _validate_distributed_request_count(
        count: int,
        device: torch.device,
        operation: str,
    ) -> None:
        validate_distributed_request_count(count, device, f"Sampler.{operation}")

    def _make_sigmas(
        self,
        batch: Batch,
        t_start: float,
        t_end: float,
    ) -> torch.Tensor:
        if self.custom_sigmas is not None:
            # Serving may mutate steps at runtime, so validate this per call.
            if len(self.custom_sigmas) != self.steps + 1:
                raise ValueError(
                    f"custom_sigmas must have steps + 1 = {self.steps + 1} entries, "
                    f"got {len(self.custom_sigmas)}."
                )
            return torch.tensor(self.custom_sigmas, dtype=torch.float32)

        if self.sigma_schedule == "diffusers_flow":
            if t_start != 1.0 or t_end != 0.0:
                raise ValueError(
                    "diffusers_flow sigma schedule currently requires "
                    "t_start=1.0 and t_end=0.0."
                )
            training_grid = torch.linspace(
                1.0,
                1.0 / self.num_train_timesteps,
                self.num_train_timesteps,
            )
            shifted_training_grid = self.shift.apply(
                training_grid, batch, self.num_train_timesteps
            )
            inference_grid = torch.linspace(
                shifted_training_grid[0], shifted_training_grid[-1], self.steps
            )
            inference_grid = self.shift.apply(inference_grid, batch, self.steps)
            return torch.cat([inference_grid, inference_grid.new_zeros(1)])

        sigmas = torch.linspace(t_start, t_end, self.steps + 1)
        return self.shift.apply(sigmas, batch, self.steps)

    def make_sigmas(
        self,
        batch: Batch,
        t_start: float = 1.0,
        t_end: float = 0.0,
    ) -> list[float]:
        """The actual (shifted) sigma grid for one batch; canonical-time args."""
        return self._make_sigmas(batch, t_start, t_end).tolist()

    def plan(self, batch: Batch) -> SamplingPlan:
        """Compile the full base plan for one batch (not yet finalized)."""
        return self.solver.plan(self.make_sigmas(batch))

    def plan_from_sigma(self, batch: Batch, sigma_start: float) -> SamplingPlan:
        """Compile a partial plan whose grid starts at an actual sigma.

        Only defined for the analytic ``linear`` schedule: the actual sigma is
        pulled back through ``shift.inverse_sigma`` to a canonical ``t_start``
        and a fresh partial grid is built from there.
        """
        if self.custom_sigmas is not None:
            raise NotImplementedError(
                "plan_from_sigma is not supported with custom_sigmas: an "
                "explicit sigma table has no canonical-time inverse to rebuild "
                "a partial grid from."
            )
        if self.sigma_schedule != "linear":
            raise NotImplementedError(
                f"plan_from_sigma is not supported with sigma_schedule="
                f"{self.sigma_schedule!r}; only the analytic 'linear' schedule "
                "can be inverted to a partial grid."
            )
        if sigma_start <= 0.0:
            raise ValueError(
                f"plan_from_sigma requires sigma_start > 0, got {sigma_start} "
                "(there is nothing left to denoise from sigma 0)."
            )
        t_start = self.shift.inverse_sigma(sigma_start, batch, self.steps)
        plan = self.solver.plan(self.make_sigmas(batch, t_start=t_start))
        if not plan or not math.isclose(
            plan[0].sigma,
            sigma_start,
            rel_tol=1e-6,
            abs_tol=1e-7,
        ):
            actual = plan[0].sigma if plan else None
            raise ValueError(
                f"{type(self.solver).__name__} cannot start a partial plan at "
                f"sigma {sigma_start}; it produced {actual}."
            )
        return plan

    def sample(
        self,
        model: SamplerModel,
        requests: list[SampleRequest],
    ) -> list[SampleOutput]:
        """Plain full-grid sampling: base plan, no recording, no transforms.

        Consumers that need trajectory recording, SDE windows, SDEdit or
        inversion phases build a recipe instead (``PhasesRecipe`` +
        ``run_phases``).
        """
        if not requests:
            raise ValueError("sample requires at least one request.")
        self._validate_distributed_request_count(len(requests), model.device, "sample")

        if self.guidance.needs_negative() and any(
            request.negative_batch is None for request in requests
        ):
            warn_once(
                logger,
                "The configured guidance needs a negative branch but at least "
                "one request has no negative_batch; those samples fall back to "
                "the conditional velocity.",
            )

        runs: list[Run] = []
        request_sigmas: list[list[float]] = []
        for request in requests:
            batch = deep_move_to_device(request.batch, model.device)
            negative_batch = (
                deep_move_to_device(request.negative_batch, model.device)
                if request.negative_batch is not None
                else None
            )
            sigmas = self.make_sigmas(batch)
            plan = finalize_replay_state(self.solver.plan(sigmas))
            request_sigmas.append(sigmas)
            runs.append(
                Run(
                    plan=plan,
                    ctx=StepContext(
                        latents=batch["noisy_latents"].float(),
                        generator=request.generator,
                        solver_state=None,
                        guidance_state=self.guidance.init_state(),
                    ),
                    batch=batch,
                    negative_batch=negative_batch,
                )
            )

        with make_sample_progress() as progress:
            task = progress.add_task("Sampling", total=self.steps)
            for event in execute(model, runs, self.guidance):
                progress.update(task, total=event.total_steps, advance=1)
                report_progress(
                    (event.step_idx + 1) / event.total_steps,
                    f"Sampling {event.step_idx + 1}/{event.total_steps}",
                )

        return [
            SampleOutput(
                final_latents=run.ctx.latents.to(model.dtype),
                timesteps=torch.tensor(
                    sigmas[:-1], dtype=torch.float32, device=model.device
                ),
            )
            for run, sigmas in zip(runs, request_sigmas, strict=True)
        ]

    def _combine_with_states(
        self,
        evals: list[BranchEvals],
        states: list[GuidanceState | None],
    ) -> list[torch.Tensor]:
        """Combine evaluated branches outside the executor loop.

        Each sample gets a transient ``StepContext`` around the provided
        state; the post-combine state is discarded (single-eval semantics).
        """
        velocities: list[torch.Tensor] = []
        for branch_evals, state in zip(evals, states, strict=True):
            ctx = StepContext(
                latents=branch_evals.latents,
                generator=None,
                solver_state=None,
                guidance_state=state,
                pre_transition_guidance_state=state,
            )
            output, _ = self.guidance.combine(branch_evals, ctx, state)
            velocities.append(output.velocity)
        return velocities

    def get_guided_velocity(
        self,
        model: SamplerModel,
        batches: list[Batch],
        negative_batches: list[Batch | None],
        latents: list[torch.Tensor],
        timesteps: list[torch.Tensor],
        sigmas: list[float],
    ) -> list[torch.Tensor]:
        """One batched branch eval plus guidance combine with fresh state.

        Thin public wrapper over the split internals for consumers that need
        a guided velocity outside the executor loop (NFT's
        ``_predict_batched``). Stateful guidance sees a fresh ``init_state()``
        per sample; for stateless CFG this is a no-op.
        """
        evals = evaluate_branches(
            model=model,
            guidance=self.guidance,
            batches=batches,
            negative_batches=negative_batches,
            latents=latents,
            sigmas=sigmas,
            timesteps=timesteps,
        )
        return self._combine_with_states(
            evals, [self.guidance.init_state() for _ in evals]
        )

    def replay_recorded_steps(
        self,
        model: SamplerModel,
        items: list[ReplayItem],
    ) -> list[StepLogProbOutput]:
        """Recompute transition log-probs for recorded rollout steps.

        One batched branch eval at every item's recorded pre-step inputs
        ``(latent_t, replay.sigma)``, combined by the guidance with the
        recorded pre-eval ``guidance_state`` (``None`` in/out for stateless
        CFG), then each pure-float :class:`ReplayStep` rebuilds the transition
        moments with the ACTUAL eta it stored at rollout time (never
        re-reading ``solver.eta``).
        """
        if not items:
            raise ValueError("replay_recorded_steps requires at least one item.")
        self._validate_distributed_request_count(
            len(items), model.device, "replay_recorded_steps"
        )

        evals = evaluate_branches(
            model=model,
            guidance=self.guidance,
            batches=[item.batch for item in items],
            negative_batches=[item.negative_batch for item in items],
            latents=[item.recorded.latent_t for item in items],
            sigmas=[item.recorded.replay.sigma for item in items],
            timesteps=[
                item.recorded.latent_t.new_full((1,), item.recorded.replay.sigma)
                for item in items
            ],
        )
        velocities = self._combine_with_states(
            evals, [item.recorded.guidance_state for item in items]
        )
        return [
            item.recorded.replay.logprob(
                velocity,
                item.recorded.latent_t,
                item.recorded.latent_next,
                solver_state=item.recorded.solver_state,
            )
            for item, velocity in zip(items, velocities, strict=True)
        ]
