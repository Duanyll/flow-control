from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Literal, cast

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

from .evaluation import evaluate
from .executor import Run, StepObserver, execute, validate_distributed_request_count
from .guidance import ClassifierFreeGuidance, Guidance
from .plan import EvalRequest, SamplingPlan, StepContext
from .projectors import Projector
from .shift import ConstantShift, Shift
from .solver import FlowSolver, Solver
from .transforms import PlanTransform

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
    """Executed plan-item start sigmas on the model device."""


class Start(BaseModel):
    """Read a source unchanged, or re-noise it at an aligned SDEdit strength."""

    model_config = ConfigDict(extra="forbid")
    source: str | None = None
    """Batch tensor to start from. Defaults to ``noisy_latents`` (pure noise), or
    to ``clean_latents`` when ``strength`` is set (SDEdit)."""
    strength: float | None = Field(default=None, gt=0.0, le=1.0)

    def slice(self, plan: SamplingPlan) -> SamplingPlan:
        if self.strength is None:
            return plan
        for index, item in enumerate(plan):
            if item.sigma <= self.strength or math.isclose(
                item.sigma, self.strength, rel_tol=1e-6, abs_tol=1e-6
            ):
                return plan[index:]
        raise ValueError(
            f"SDEdit strength={self.strength} is below every denoising grid point."
        )

    def latents(
        self, batch: Batch, sigma: float, generator: torch.Generator | None
    ) -> torch.Tensor:
        key = self.source or (
            "noisy_latents" if self.strength is None else "clean_latents"
        )
        source = cast("dict[str, Any]", batch).get(key)
        if not isinstance(source, torch.Tensor):
            raise ValueError(f"Start source {key!r} must name a tensor in the batch.")
        source = source.float()
        if self.strength is None:
            return source
        noise = torch.randn(
            source.shape, dtype=source.dtype, device=source.device, generator=generator
        )
        return (1.0 - sigma) * source + sigma * noise


class Sampler(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: Start = Field(default_factory=Start)
    transforms: list[PlanTransform] = Field(default_factory=list)
    projectors: list[Projector] = Field(default_factory=list)

    seed: int = 42
    guidance: Guidance = Field(default_factory=ClassifierFreeGuidance)
    """Sampling middleware; the default is ``ClassifierFreeGuidance`` with
    ``scale=1.0`` (no negative pass)."""

    steps: int = Field(default=50, gt=0)
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

    def plan(
        self, batch: Batch, generator: torch.Generator | None = None
    ) -> SamplingPlan:
        plan = self.start.slice(self.solver.plan(self.make_sigmas(batch)))
        for transform in self.transforms:
            plan = transform.apply(plan, generator)
        return plan

    def sample(
        self,
        model: SamplerModel,
        requests: list[SampleRequest],
        *,
        observer: StepObserver | None = None,
    ) -> list[SampleOutput]:
        if not requests:
            raise ValueError("sample requires at least one request.")
        validate_distributed_request_count(
            len(requests), model.device, "Sampler.sample"
        )
        if (
            isinstance(self.guidance, ClassifierFreeGuidance)
            and self.guidance.requires_negative(self.steps)
            and any(request.negative_batch is None for request in requests)
        ):
            warn_once(
                logger,
                "The configured guidance needs a negative branch but at least "
                "one request has no negative_batch; those samples fall back to "
                "the conditional velocity.",
            )
        runs = []
        for request in requests:
            batch = deep_move_to_device(request.batch, model.device)
            negative_batch = (
                deep_move_to_device(request.negative_batch, model.device)
                if request.negative_batch is not None
                else None
            )
            # Window selection precedes start noise, preserving per-sample RNG order.
            plan = self.plan(batch, request.generator)
            runs.append(
                Run(
                    plan=plan,
                    ctx=StepContext(
                        latents=self.start.latents(
                            batch, plan[0].sigma, request.generator
                        ),
                        generator=request.generator,
                        solver_state=None,
                        guidance_state=self.guidance.init_state(),
                        num_items=len(plan),
                    ),
                    batch=batch,
                    negative_batch=negative_batch,
                )
            )
        with make_sample_progress() as progress:
            task = progress.add_task("Sampling", total=len(runs[0].plan))
            for event in execute(model, runs, self.guidance, self.projectors, observer):
                progress.update(task, total=event.total_steps, advance=1)
                report_progress(
                    (event.step_idx + 1) / event.total_steps,
                    f"Sampling {event.step_idx + 1}/{event.total_steps}",
                )
        return [
            SampleOutput(
                final_latents=run.ctx.latents.to(model.dtype),
                timesteps=torch.tensor(
                    [item.sigma for item in run.plan],
                    dtype=torch.float32,
                    device=model.device,
                ),
            )
            for run in runs
        ]

    def get_guided_velocity(
        self,
        model: SamplerModel,
        batches: list[Batch],
        negative_batches: list[Batch | None],
        latents: list[torch.Tensor],
        timesteps: list[torch.Tensor],
        sigmas: list[float],
        *,
        sigma_nexts: list[float | None] | None = None,
        etas: list[float] | None = None,
        item_indices: list[int] | None = None,
        num_items: list[int] | None = None,
    ) -> list[torch.Tensor]:
        """Evaluate guidance and whole-image projections with fresh per-item state.

        ``item_indices`` are positions in each request's executed plan, so they
        must come with those plans' lengths ``num_items``; without them every
        item is the first of ``steps``.
        """
        if len(timesteps) != len(sigmas):
            raise ValueError("timesteps and sigmas must have equal lengths.")
        if item_indices is not None and (
            num_items is None or len(num_items) != len(item_indices)
        ):
            raise ValueError(
                "item_indices requires an equally long num_items list of executed "
                "plan lengths."
            )
        requests = [
            EvalRequest(
                latent,
                sigma,
                sigma_next=sigma_nexts[index] if sigma_nexts is not None else None,
                eta=etas[index] if etas is not None else 0.0,
                solver=self.solver,
            )
            for index, (latent, sigma) in enumerate(zip(latents, sigmas, strict=True))
        ]
        contexts = [
            StepContext(
                latents=latent,
                generator=None,
                solver_state=None,
                guidance_state=self.guidance.init_state(),
                item_index=item_indices[index] if item_indices is not None else 0,
                num_items=num_items[index] if num_items is not None else self.steps,
            )
            for index, latent in enumerate(latents)
        ]
        outputs = evaluate(
            model=model,
            guidance=self.guidance,
            batches=batches,
            negative_batches=negative_batches,
            requests=requests,
            contexts=contexts,
            projectors=self.projectors,
            timesteps=timesteps,
        )
        return [output.velocity for output in outputs]
