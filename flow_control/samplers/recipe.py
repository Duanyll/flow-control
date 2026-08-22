"""Recipe layer: config algebra over phases and plan transforms.

The config surface mirrors the underlying algebra — phases concatenate in
order, plan transforms compose in list order — instead of enumerating named
variants (TextToImage / SDEdit / ...). Core ships only the combinators:

- ``InitOp``: where a phase's latents come from
  (``pure_noise | renoise | from_latents | from_previous``);
- ``PlanTransform``: config wrappers over pure plan functions
  (``sde_window | invert``, contrib extends freely);
- ``Recipe``: ``phases`` is the single built-in member; flows that do not fit
  the phase structure register a custom recipe and do arbitrary plan surgery
  in ``build()``.

``PhasesRecipe.build`` produces runtime :class:`Phase` objects with everything
resolved (effective sampler, batch, guidance, finalized plan); a runner
executes them sequentially, one ``execute()`` per phase.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal, cast

import torch
from pydantic import BaseModel, ConfigDict, Field

from flow_control.adapters.base import Batch
from flow_control.utils.logging import get_logger, warn_once
from flow_control.utils.registry import Registry, RegistryUnion

from .guidance import BaseGuidance
from .plan import SamplingPlan, Transition
from .sampler import Sampler
from .transforms import finalize_replay_state, select_sde_window, with_sde_window

logger = get_logger(__name__)

_SIGMA_TOLERANCE = 1e-6
"""Grids are built in float32; config sigmas (e.g. ``strength: 0.6``) match a
grid point only up to float32 rounding."""


def _sigma_close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=_SIGMA_TOLERANCE, abs_tol=_SIGMA_TOLERANCE)


def _batch_tensor(batch: Batch, source: str) -> torch.Tensor:
    value = cast("dict[str, Any]", batch).get(source)
    if not isinstance(value, torch.Tensor):
        raise KeyError(
            f"Init op source {source!r} is not a tensor in the batch; "
            f"available keys: {sorted(cast('dict[str, Any]', batch))}."
        )
    return value


# ---------------------------------------------------------------------------
# InitOp: the source algebra for a phase's latents
# ---------------------------------------------------------------------------


class BaseInitOp(BaseModel, ABC):
    type: Literal["base"] = "base"
    model_config = ConfigDict(extra="forbid")

    @abstractmethod
    def build_latents(
        self,
        batch: Batch,
        generator: torch.Generator | None,
        aligned_sigma: float,
    ) -> torch.Tensor:
        """Construct the phase's initial latents.

        ``aligned_sigma`` is the built plan's start sigma
        (:func:`plan_start_sigma`) — the single source of truth after the
        builder aligned the plan, never the raw config value.
        """


init_op_registry: Registry[BaseInitOp] = Registry("init_op", base=BaseInitOp)


@init_op_registry.register("pure_noise")
class PureNoise(BaseInitOp):
    """Use the request batch's pre-initialized ``noisy_latents`` as-is."""

    type: Literal["pure_noise"] = "pure_noise"

    def build_latents(
        self,
        batch: Batch,
        generator: torch.Generator | None,
        aligned_sigma: float,
    ) -> torch.Tensor:
        return batch["noisy_latents"].float()


@init_op_registry.register("renoise")
class Renoise(BaseInitOp):
    """RF-interpolate a source latent to the phase's start sigma (SDEdit).

    ``strength`` is an ACTUAL sigma (SDEdit convention): the phase enters the
    trajectory at the first grid point at or below it. The interpolation
    ``x_sigma = (1 - sigma) * x0 + sigma * eps`` uses the plan's aligned start
    sigma, not ``strength`` itself.
    """

    type: Literal["renoise"] = "renoise"
    strength: float = Field(gt=0.0, le=1.0)
    source: str = "clean_latents"

    def build_latents(
        self,
        batch: Batch,
        generator: torch.Generator | None,
        aligned_sigma: float,
    ) -> torch.Tensor:
        clean = _batch_tensor(batch, self.source).float()
        noise = torch.randn(
            clean.shape,
            dtype=clean.dtype,
            device=clean.device,
            generator=generator,
        )
        return (1.0 - aligned_sigma) * clean + aligned_sigma * noise


@init_op_registry.register("from_latents")
class FromLatents(BaseInitOp):
    """Start from a batch latent unchanged (e.g. clean latents for inversion)."""

    type: Literal["from_latents"] = "from_latents"
    source: str

    def build_latents(
        self,
        batch: Batch,
        generator: torch.Generator | None,
        aligned_sigma: float,
    ) -> torch.Tensor:
        return _batch_tensor(batch, self.source).float()


@init_op_registry.register("from_previous")
class FromPrevious(BaseInitOp):
    """Marker: the runner hands over the previous phase's final latents."""

    type: Literal["from_previous"] = "from_previous"

    def build_latents(
        self,
        batch: Batch,
        generator: torch.Generator | None,
        aligned_sigma: float,
    ) -> torch.Tensor:
        raise RuntimeError(
            "FromPrevious does not build latents; the recipe runner passes the "
            "previous phase's final latents directly."
        )


InitOp = Annotated[BaseInitOp, RegistryUnion(init_op_registry, "type")]


# ---------------------------------------------------------------------------
# PlanTransform: config wrappers over the pure plan functions (axis 1)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TransformContext:
    """Effective per-phase values a transform may draw on (design BuildContext)."""

    batch: Batch
    generator: torch.Generator | None
    sampler: Sampler


class BasePlanTransform(BaseModel, ABC):
    type: Literal["base"] = "base"
    model_config = ConfigDict(extra="forbid")

    @abstractmethod
    def apply(self, plan: SamplingPlan, ctx: TransformContext) -> SamplingPlan: ...


plan_transform_registry: Registry[BasePlanTransform] = Registry(
    "plan_transform", base=BasePlanTransform
)


@plan_transform_registry.register("sde_window")
class SdeWindow(BasePlanTransform):
    """Gate per-step eta to a (possibly random) denoise-index window.

    Omitting ``size`` selects the full window (terminal transition excluded);
    otherwise the start index is drawn from the per-sample generator. With
    ``record=True`` the same window is marked as the replayable trajectory.
    """

    type: Literal["sde_window"] = "sde_window"
    size: int | None = None
    range: tuple[int, int] | None = None
    record: bool = False

    def apply(self, plan: SamplingPlan, ctx: TransformContext) -> SamplingPlan:
        num_transitions = sum(1 for item in plan if isinstance(item, Transition))
        start, end = select_sde_window(
            num_transitions, self.size, self.range, ctx.generator
        )
        return with_sde_window(plan, start, end, record=self.record)


@plan_transform_registry.register("invert")
class Invert(BasePlanTransform):
    """Reverse the full base plan (DDIM/Euler inversion); must be listed first."""

    type: Literal["invert"] = "invert"

    def apply(self, plan: SamplingPlan, ctx: TransformContext) -> SamplingPlan:
        return ctx.sampler.solver.invert(plan)


PlanTransform = Annotated[
    BasePlanTransform, RegistryUnion(plan_transform_registry, "type")
]


# ---------------------------------------------------------------------------
# Recipe: phases, build context and the runtime Phase product
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Phase:
    """One executable phase: everything resolved, plan finalized."""

    init: BaseInitOp
    plan: SamplingPlan
    batch: Batch
    negative_batch: Batch | None
    guidance: BaseGuidance
    generator: torch.Generator | None
    """All phases of one request share the same mutable generator reference."""


@dataclass(slots=True)
class RecipeBuildContext:
    default_sampler: Sampler
    batches: Mapping[str, Batch]
    negative_batch_for: Callable[[str], Batch | None]
    generator: torch.Generator | None


class PhaseConfig(BaseModel):
    """One phase of :class:`PhasesRecipe` (plain nested model, not a union)."""

    model_config = ConfigDict(extra="forbid")

    init: InitOp = Field(default_factory=PureNoise)
    transforms: list[PlanTransform] = Field(default_factory=list)
    batch: str = "main"
    """Which conditioning batch this phase uses (edit phases pick another set;
    the consumer provides the corresponding key)."""
    sampler: Sampler | None = None
    """``None`` inherits the consumer-level sampler and SLICES its plan (enter
    the same trajectory midway — standard SDEdit); an explicit sampler builds a
    NEW grid on [init start sigma, end] with its own steps/shift/solver and
    guidance."""


class BaseRecipe(BaseModel, ABC):
    type: Literal["base"] = "base"
    model_config = ConfigDict(extra="forbid")

    @abstractmethod
    def build(self, ctx: RecipeBuildContext) -> list[Phase]: ...


recipe_registry: Registry[BaseRecipe] = Registry("recipe", base=BaseRecipe)


@recipe_registry.register("phases")
class PhasesRecipe(BaseRecipe):
    """The only core recipe: phases concatenate, transforms compose in order."""

    type: Literal["phases"] = "phases"
    phases: list[PhaseConfig] = Field(default_factory=lambda: [PhaseConfig()])

    def build(self, ctx: RecipeBuildContext) -> list[Phase]:
        built: list[Phase] = []
        previous_final_sigma: float | None = None
        for index, cfg in enumerate(self.phases):
            sampler = cfg.sampler if cfg.sampler is not None else ctx.default_sampler
            if cfg.batch not in ctx.batches:
                raise KeyError(
                    f"Phase {index} requests batch {cfg.batch!r}, but the "
                    f"build context only provides {sorted(ctx.batches)}."
                )
            batch = ctx.batches[cfg.batch]
            guidance = sampler.guidance
            negative_batch = (
                ctx.negative_batch_for(cfg.batch) if guidance.needs_negative() else None
            )
            if guidance.needs_negative() and negative_batch is None:
                warn_once(
                    logger,
                    f"Phase {index}'s guidance needs a negative branch but the "
                    f"build context resolved no negative batch for key "
                    f"{cfg.batch!r}; those samples fall back to the "
                    "conditional velocity.",
                )
            transform_ctx = TransformContext(
                batch=batch, generator=ctx.generator, sampler=sampler
            )
            plan = _build_phase_plan(
                cfg, index, batch, transform_ctx, previous_final_sigma
            )
            built.append(
                Phase(
                    init=cfg.init,
                    plan=plan,
                    batch=batch,
                    negative_batch=negative_batch,
                    guidance=guidance,
                    generator=ctx.generator,
                )
            )
            previous_final_sigma = _final_sigma(plan)
        return built


Recipe = Annotated[
    BaseRecipe, RegistryUnion(recipe_registry, "type", list_as=("phases", "phases"))
]


def _build_phase_plan(
    cfg: PhaseConfig,
    index: int,
    batch: Batch,
    transform_ctx: TransformContext,
    previous_final_sigma: float | None,
) -> SamplingPlan:
    """Assemble one phase's finalized plan in the fixed pipeline order:
    full base plan -> optional leading invert -> alignment to the init's
    start sigma -> remaining transforms in list order -> finalize."""
    sampler = transform_ctx.sampler
    transforms = list(cfg.transforms)
    leading_invert = bool(transforms) and isinstance(transforms[0], Invert)
    post_align = transforms[1:] if leading_invert else transforms
    if any(isinstance(transform, Invert) for transform in post_align):
        raise NotImplementedError(
            f"Phase {index}: 'invert' must be the single, first entry of "
            "transforms; later positions or repeats are not supported."
        )

    start_sigma = _init_start_sigma(cfg.init, index, previous_final_sigma)

    if leading_invert:
        # Invert runs BEFORE alignment: slicing the descending full plan at a
        # clean latent's near-zero sigma would empty it.
        plan = transforms[0].apply(sampler.plan(batch), transform_ctx)
        if start_sigma is not None:
            plan = _align_inverted(plan, start_sigma, index)
    elif start_sigma is not None:
        if cfg.sampler is not None:
            plan = sampler.plan_from_sigma(batch, start_sigma)
        else:
            plan = _align_inherited(sampler.plan(batch), cfg.init, start_sigma, index)
    else:
        plan = sampler.plan(batch)

    for transform in post_align:
        plan = transform.apply(plan, transform_ctx)
    return finalize_replay_state(plan)


def _align_inverted(plan: SamplingPlan, start_sigma: float, index: int) -> SamplingPlan:
    sliced = _slice_at_sigma(plan, start_sigma, allow_below=False)
    if sliced is None:
        raise NotImplementedError(
            f"Phase {index}: partial inversion start sigma {start_sigma} does "
            "not land on the inverted grid; only starts at exact grid points "
            "are supported."
        )
    return sliced


def _align_inherited(
    plan: SamplingPlan, init: BaseInitOp, start_sigma: float, index: int
) -> SamplingPlan:
    if isinstance(init, Renoise):
        # SDEdit convention: enter the inherited trajectory at the first grid
        # point at or below the requested strength.
        sliced = _slice_at_sigma(plan, start_sigma, allow_below=True)
        if sliced is None:
            lowest = min(item.sigma for item in plan if isinstance(item, Transition))
            raise ValueError(
                f"Phase {index}: no transition starts at or below sigma "
                f"{start_sigma}; the grid's lowest transition starts at "
                f"{lowest}."
            )
        return sliced
    # FromPrevious hands over exact (x, sigma): silently re-labelling the
    # latents at a lower grid sigma would misalign the phase boundary.
    sliced = _slice_at_sigma(plan, start_sigma, allow_below=False)
    if sliced is None:
        raise NotImplementedError(
            f"Phase {index}: the previous phase ends at sigma {start_sigma}, "
            "which is not a grid point of this phase's inherited plan; give "
            "the phase an explicit sampler to rebuild its grid from that "
            "sigma."
        )
    return sliced


def _init_start_sigma(
    init: BaseInitOp, index: int, previous_final_sigma: float | None
) -> float | None:
    """The sigma an init op delivers latents at; ``None`` = no alignment."""
    if isinstance(init, Renoise):
        return init.strength
    if isinstance(init, FromPrevious):
        if previous_final_sigma is None:
            raise ValueError(
                f"Phase {index} uses 'from_previous' but is the first phase."
            )
        return previous_final_sigma
    # PureNoise / FromLatents deliver latents wherever the plan starts: the
    # full grid top, or — after a leading invert — the inverted grid's start.
    return None


def _slice_at_sigma(
    plan: SamplingPlan, start_sigma: float, *, allow_below: bool
) -> SamplingPlan | None:
    """Slice a plan from the first transition matching ``start_sigma``.

    With ``allow_below`` (Renoise's at-or-below convention) the first
    transition whose sigma is <= ``start_sigma`` also matches; otherwise only
    an exact grid point (up to float32 rounding) does. Returns ``None`` when
    no transition matches — the caller raises its own error.
    """
    for item_index, item in enumerate(plan):
        if not isinstance(item, Transition):
            continue
        if _sigma_close(item.sigma, start_sigma) or (
            allow_below and item.sigma <= start_sigma
        ):
            return plan[item_index:]
    return None


def _final_sigma(plan: SamplingPlan) -> float:
    for item in reversed(plan):
        if isinstance(item, Transition):
            return item.sigma_next
    raise ValueError("Plan contains no denoise transitions.")


def plan_start_sigma(plan: SamplingPlan) -> float:
    """The aligned start sigma of a phase plan (its first denoise transition).

    Single source of truth for init ops: the runner passes this value to
    :meth:`BaseInitOp.build_latents`, never a raw config field.
    """
    for item in plan:
        if isinstance(item, Transition):
            return item.sigma
    raise ValueError("Plan contains no denoise transitions.")


def plan_has_recordable_stochastic_step(phases: list[Phase]) -> bool:
    """Whether any phase records at least one stochastic (replayable) step.

    Trainers that need likelihood replay (GRPO) validate this after build and
    should point users at ``{"type": "sde_window", "record": true}``.
    """
    return any(
        isinstance(item, Transition) and item.record and item.eta > 0.0
        for phase in phases
        for item in phase.plan
    )
