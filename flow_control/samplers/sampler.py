from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

import torch
from pydantic import BaseModel, ConfigDict, Field

from flow_control.adapters.base import Batch, SamplerModel
from flow_control.utils.logging import get_logger, warn_once
from flow_control.utils.tensor import deep_move_to_device

from .executor import Executor
from .guidance import ClassifierFreeGuidance, Guidance
from .plan import SamplingPlan, StepContext
from .projectors import Projector
from .run import SampleRun, StepCollector
from .shift import ConstantShift, Shift
from .solver import FlowSolver, Solver
from .transforms import PlanTransform

logger = get_logger(__name__)


def derive_seed(base_seed: int, key: str) -> int:
    """Derive a deterministic per-sample seed from a base seed and a sample key."""
    h = hashlib.sha256(f"{base_seed}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "little") % (2**63)


@dataclass(slots=True)
class SampleRequest:
    batch: Batch
    negative_batch: Batch | None = None
    generator: torch.Generator | None = None


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

    def variant_keys(self) -> list[str | None]:
        """Every weight variant the guidance may ask for, in a fixed order."""
        return list(
            dict.fromkeys(
                spec.variant
                for index in range(self.steps)
                for spec in self.guidance.branches(index)
            )
        )

    def make_run(
        self,
        request: SampleRequest,
        *,
        plan: SamplingPlan | None = None,
        collector: StepCollector | None = None,
    ) -> SampleRun:
        """Bind a request to a plan and fresh per-run state.

        Without ``plan`` this is a sampling run: the plan is built for the
        batch and latents start per ``start``. Training passes the executed
        ``plan`` back and supplies latents through ``guided_velocity``.
        """
        batch, negative = request.batch, request.negative_batch
        if plan is None:
            plan = self.plan(batch, request.generator)
            latents = self.start.latents(batch, plan[0].sigma, request.generator)
        else:
            latents = batch["noisy_latents"].float()
        if negative is None:
            negative_branches = [
                spec
                for index in range(len(plan))
                for spec in self.guidance.branches(index)
                if spec.batch_key == "negative"
            ]
            if any(not spec.optional for spec in negative_branches):
                raise ValueError(
                    f"Guidance {self.guidance.type!r} needs a negative batch but the "
                    "request has none; enable the processor's negative conditioning."
                )
            if negative_branches:
                warn_once(
                    logger,
                    "The configured guidance can use a negative branch but at least "
                    "one request has no negative_batch; those samples fall back to "
                    "the conditional velocity.",
                )
        ctx = StepContext(
            latents=latents,
            generator=request.generator,
            solver_state=None,
            guidance_state=self.guidance.init_state(),
            num_items=len(plan),
        )
        return SampleRun(self, batch, negative, plan, ctx, collector)

    def sample(
        self,
        model: SamplerModel,
        requests: Iterable[SampleRequest],
        *,
        collector: StepCollector | None = None,
    ) -> Iterator[SampleRun]:
        """Sample lazily through one Executor; runs come out as they finish."""

        def runs() -> Iterator[SampleRun]:
            for request in requests:
                moved = replace(
                    request,
                    batch=deep_move_to_device(request.batch, model.device),
                    negative_batch=deep_move_to_device(
                        request.negative_batch, model.device
                    ),
                )
                yield self.make_run(moved, collector=collector)

        return Executor(model, self.variant_keys()).stream(runs())
