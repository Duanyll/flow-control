"""One sample's sampling loop, written as a generator of leaf model calls."""

from __future__ import annotations

from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
from einops import rearrange

from flow_control.adapters.base import Batch
from flow_control.utils.tiling import TileLayout, TileSpec, extract_tiles, stitch_tiles

from .guidance import BranchSpec
from .plan import (
    BranchEvals,
    EvalRequest,
    GuidanceOutput,
    StepContext,
    Transition,
    TransitionResult,
)
from .projectors import apply_pre_transition

if TYPE_CHECKING:
    from .sampler import Sampler


@dataclass(slots=True)
class ModelCall:
    """One leaf model forward; the only thing a sampler hands to an Executor."""

    batch: Batch
    """Model input including ``noisy_latents`` for one logical sample."""
    timestep: torch.Tensor
    variant: str | None = None
    """Weight variant to run under; calls with different variants never share a forward."""


@dataclass(slots=True)
class StepRecord:
    """One completed transition, passed to the caller without retaining tensors."""

    index: int
    transition: Transition
    latents: torch.Tensor
    velocity: torch.Tensor | None
    """Combined whole-image velocity of the transition's last evaluation, if any."""
    next_latents: torch.Tensor


type Calls[T] = Generator[list[ModelCall], list[torch.Tensor], T]
"""A generator that yields batches of concurrent model calls, receives their
velocities in the same order, and finally returns ``T``."""

type StepCollector = Callable[[SampleRun, StepRecord], None]


@dataclass(slots=True)
class _Tiling:
    specs: list[TileSpec]
    height: int
    width: int
    """Token-grid extent of the whole image."""


def _expand_tiles(batch: Batch) -> tuple[list[Batch], _Tiling | None]:
    """Cut one packed BND image into per-tile model inputs on the token grid."""
    if "tiling" not in batch:
        return [batch], None
    source = cast(dict[str, Any], batch)
    layout = TileLayout.model_validate(source["tiling"])
    specs = layout.token_specs(source["image_size"])
    height, width = (length // layout.stride for length in source["image_size"])
    x = source["noisy_latents"]
    if x.ndim != 3 or x.shape[:2] != (1, height * width):
        raise ValueError(
            "Tiled evaluation requires one packed BND image matching image_size; "
            f"got {tuple(x.shape)}, image_size={source['image_size']}, "
            f"stride={layout.stride}."
        )
    conditions = source.get("tiles")
    if conditions is not None and len(conditions) != len(specs):
        raise ValueError(
            f"Expected {len(specs)} row-major tile batches, got {len(conditions)}."
        )
    grid = rearrange(x, "b (h w) d -> b d h w", h=height, w=width)
    tiles: list[Batch] = []
    for index, (spec, latents) in enumerate(
        zip(specs, extract_tiles(grid, specs), strict=True)
    ):
        tile = dict(source if conditions is None else conditions[index])
        for key in ("tiling", "tiles", "model_image_size", "negative", "clean_latents"):
            tile.pop(key, None)
        tile["image_size"] = (spec.height * layout.stride, spec.width * layout.stride)
        tile["noisy_latents"] = rearrange(latents, "b d h w -> b (h w) d")
        tiles.append(cast(Batch, tile))
    return tiles, _Tiling(specs, height, width)


def _merge_tiles(
    velocities: list[torch.Tensor], tiling: _Tiling | None
) -> torch.Tensor:
    if tiling is None:
        return velocities[0]
    tiles = [
        rearrange(velocity, "b (h w) d -> b d h w", h=spec.height, w=spec.width)
        for velocity, spec in zip(velocities, tiling.specs, strict=True)
    ]
    stitched = stitch_tiles(tiles, tiling.specs, tiling.height, tiling.width)
    return rearrange(stitched, "b d h w -> b (h w) d")


def conditional_velocity(batch: Batch, timestep: torch.Tensor) -> Calls[torch.Tensor]:
    """Plain conditional forward with tiles expanded and stitched; no guidance.

    The training targets of SFT/AWM/RAM reach the model through this, so they
    see exactly the tiles the sampler saw.
    """
    tiles, tiling = _expand_tiles(batch)
    velocities = yield [ModelCall(tile, timestep) for tile in tiles]
    return _merge_tiles(velocities, tiling)


@dataclass(slots=True)
class SampleRun:
    """Everything one sample needs to be denoised: configuration, data and state.

    ``run()`` is the whole trajectory. ``guided_velocity()`` is one evaluation of
    the same guidance at a chosen plan item, for training code that re-evaluates
    an executed plan.
    """

    sampler: Sampler
    batch: Batch
    negative_batch: Batch | None
    plan: list[Transition]
    ctx: StepContext
    collector: StepCollector | None = None

    def run(self) -> Calls[None]:
        ctx = self.ctx
        ctx.num_items = len(self.plan)
        for index, transition in enumerate(self.plan):
            ctx.item_index = index
            ctx.latents = apply_pre_transition(
                self.sampler.projectors, self.batch, ctx, transition
            )
            solver = transition.run(ctx)
            output: GuidanceOutput | None = None
            try:
                request = next(solver)
                while True:
                    output = yield from self._evaluate(request, ctx)
                    request = solver.send(output)
            except StopIteration as stop:
                result = cast(TransitionResult, stop.value)
            if self.collector is not None:
                self.collector(
                    self,
                    StepRecord(
                        index,
                        transition,
                        ctx.latents,
                        None if output is None else output.velocity,
                        result.next_latents,
                    ),
                )
            ctx.latents = result.next_latents
            if result.next_solver_state is not None:
                ctx.solver_state = result.next_solver_state
        ctx.latents = ctx.latents.to(self.batch["noisy_latents"].dtype)

    def guided_velocity(
        self, latents: torch.Tensor, item_index: int
    ) -> Calls[torch.Tensor]:
        """Evaluate plan item ``item_index`` at ``latents`` as the loop would."""
        transition = self.plan[item_index]
        latents = latents.float()
        ctx = StepContext(
            latents=latents,
            generator=None,
            solver_state=None,
            guidance_state=self.sampler.guidance.init_state(),
            item_index=item_index,
            num_items=len(self.plan),
        )
        output = yield from self._evaluate(
            EvalRequest(
                latents,
                transition.sigma,
                transition.sigma_next,
                transition.eta,
                transition.solver,
            ),
            ctx,
        )
        return output.velocity

    def _branch_source(self, spec: BranchSpec) -> Batch | None:
        if spec.batch_key == "positive":
            return self.batch
        if spec.batch_key == "negative":
            return self.negative_batch
        raise ValueError(
            f"Branch {spec.name!r} has unknown batch_key {spec.batch_key!r}."
        )

    def _evaluate(
        self, request: EvalRequest, ctx: StepContext
    ) -> Calls[GuidanceOutput]:
        """One whole-image evaluation: branches x tiles -> stitch -> combine -> project."""
        timestep = request.latents.new_full((1,), request.sigma)
        calls: list[ModelCall] = []
        branches: list[tuple[BranchSpec, _Tiling | None]] = []
        for spec in self.sampler.guidance.branches(ctx.item_index):
            source = self._branch_source(spec)
            if source is None:
                continue  # Optional branch without a condition; make_run rejected the rest.
            batch = dict(source)
            batch["noisy_latents"] = request.latents
            tiles, tiling = _expand_tiles(cast(Batch, batch))
            calls.extend(ModelCall(tile, timestep, spec.variant) for tile in tiles)
            branches.append((spec, tiling))
        velocities = yield calls
        evals = BranchEvals(
            {},
            request.latents,
            request.sigma,
            request.sigma_next,
            request.eta,
            request.solver,
        )
        offset = 0
        for spec, tiling in branches:
            count = 1 if tiling is None else len(tiling.specs)
            evals.velocities[spec.name] = _merge_tiles(
                velocities[offset : offset + count], tiling
            )
            offset += count
        output, ctx.guidance_state = self.sampler.guidance.combine(
            evals, ctx, ctx.guidance_state
        )
        for projector in self.sampler.projectors:
            output = projector.post_combine(output, request, self.batch, ctx)
        return output
