"""Shared branch evaluation and whole-image guidance for sampling and training."""

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.distributed as dist
from einops import rearrange

from flow_control.adapters.base import Batch, SamplerModel
from flow_control.utils.tiling import TileLayout, TileSpec, extract_tiles, stitch_tiles

from .guidance import BaseGuidance, BranchSpec
from .plan import BranchEvals, EvalRequest, GuidanceOutput, StepContext
from .projectors import BaseProjector


def _branch_batch(
    spec: BranchSpec, positive: Batch, negative: Batch | None
) -> Batch | None:
    if spec.batch_key == "positive":
        return positive
    if spec.batch_key == "negative":
        return negative
    raise ValueError(f"Branch {spec.name!r} has unknown batch_key {spec.batch_key!r}.")


def _schedule(guidance: BaseGuidance, num_items: int) -> list[BranchSpec]:
    # Derived from configuration alone, so every rank runs one forward schedule
    # without exchanging its locally present branches.
    return list(
        dict.fromkeys(
            spec for index in range(num_items) for spec in guidance.branches(index)
        )
    )


def _active_passes(
    sources: list[list[Batch | None]], count: int, device: torch.device, missing: bool
) -> list[bool]:
    active = [any(source is not None for source in group) for group in sources]
    values = [count, -count, int(missing), *map(int, active)]
    if dist.is_initialized():
        status = torch.tensor(values, device=device, dtype=torch.int64)
        dist.all_reduce(status, op=dist.ReduceOp.MAX)
        values = status.tolist()
    if values[0] != -values[1]:
        raise ValueError(
            "All ranks must submit the same number of branch evaluation requests."
        )
    if values[2]:
        raise ValueError(
            "A required guidance branch is missing from an input batch on this or another rank. "
            "Supply negative_batch for CFG++, or the named branch's batch_key condition."
        )
    return [bool(value) for value in values[3:]]


def _keep_dummy_gradients(
    velocities: list[dict[str, torch.Tensor]], dummy_outputs: list[torch.Tensor]
) -> None:
    dummy_outputs = [output for output in dummy_outputs if output.requires_grad]
    if not dummy_outputs:
        return
    # FSDP also synchronizes backward. Keep every dummy forward in the graph,
    # with zero contribution, when another rank trains this branch.
    zero = sum(output.sum() * 0 for output in dummy_outputs)
    for target in velocities:
        first = next(iter(target))
        target[first] = target[first] + zero


@dataclass(slots=True)
class _Tiling:
    specs: list[TileSpec]
    height: int
    width: int
    """Token-grid extent of the whole image."""


def _tile_batches(batch: Batch) -> tuple[list[Batch], _Tiling]:
    """Split one packed BND image into per-tile model inputs on the token grid."""
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


def _stitch(velocities: list[torch.Tensor], tiling: _Tiling) -> torch.Tensor:
    tiles = [
        rearrange(velocity, "b (h w) d -> b d h w", h=spec.height, w=spec.width)
        for velocity, spec in zip(velocities, tiling.specs, strict=True)
    ]
    stitched = stitch_tiles(tiles, tiling.specs, tiling.height, tiling.width)
    return rearrange(stitched, "b d h w -> b (h w) d")


def predict_velocity(
    model: SamplerModel, batches: list[Batch], timesteps: list[torch.Tensor]
) -> list[torch.Tensor]:
    """The single leaf call: expand tiled batches, run the model once, merge tiles back."""
    if not batches or len(batches) != len(timesteps):
        raise ValueError(
            "predict_velocity requires nonempty batches and equally many timesteps."
        )
    inputs: list[Batch] = []
    times: list[torch.Tensor] = []
    tilings: list[_Tiling | None] = []
    for batch, timestep in zip(batches, timesteps, strict=True):
        tiles, tiling = _tile_batches(batch) if "tiling" in batch else ([batch], None)
        inputs.extend(tiles)
        times.extend([timestep] * len(tiles))
        tilings.append(tiling)
    # Unequal input counts across ranks are rejected by the adapter's own sync.
    velocities = model.predict_velocity_batched(inputs, times)
    outputs = []
    offset = 0
    for tiling in tilings:
        count = 1 if tiling is None else len(tiling.specs)
        values = velocities[offset : offset + count]
        outputs.append(values[0] if tiling is None else _stitch(values, tiling))
        offset += count
    return outputs


def evaluate_branches(
    model: SamplerModel,
    guidance: BaseGuidance,
    batches: list[Batch],
    negative_batches: list[Batch | None],
    requests: list[EvalRequest],
    contexts: list[StepContext],
    timesteps: list[torch.Tensor] | None = None,
) -> list[BranchEvals]:
    lengths = {len(batches), len(negative_batches), len(requests), len(contexts)}
    if timesteps is not None:
        lengths.add(len(timesteps))
    if len(lengths) != 1 or not requests:
        raise ValueError(
            "Branch evaluation requires nonempty, equally sized batches, requests and contexts."
        )
    branches = [guidance.branches(ctx.item_index) for ctx in contexts]
    if any(len({spec.name for spec in specs}) != len(specs) for specs in branches):
        raise ValueError("Guidance branch names must be unique within each evaluation.")
    schedule = _schedule(guidance, max(ctx.num_items for ctx in contexts))
    sources = [
        [
            _branch_batch(spec, batch, negative) if spec in specs else None
            for specs, batch, negative in zip(
                branches, batches, negative_batches, strict=True
            )
        ]
        for spec in schedule
    ]
    missing = any(
        not spec.optional and spec in specs and source is None
        for spec, group in zip(schedule, sources, strict=True)
        for specs, source in zip(branches, group, strict=True)
    )
    active = _active_passes(sources, len(requests), model.device, missing)
    if timesteps is None:
        timesteps = [
            request.latents.new_full((1,), request.sigma) for request in requests
        ]
    velocities: list[dict[str, torch.Tensor]] = [{} for _ in requests]
    dummy_outputs: list[torch.Tensor] = []
    for spec, group, run_pass in zip(schedule, sources, active, strict=True):
        if not run_pass:
            continue
        inputs = []
        for source, positive, request in zip(group, batches, requests, strict=True):
            batch = (positive if source is None else source).copy()
            batch["noisy_latents"] = request.latents
            inputs.append(batch)
        # Preserve the established per-branch batch shape, including CFG
        # dummies. Variant changes never mutate the shared guidance config.
        with (
            model.use_variant(spec.variant)
            if spec.variant is not None
            else nullcontext()
        ):
            outputs = predict_velocity(model, inputs, timesteps)
        for target, source, velocity in zip(velocities, group, outputs, strict=True):
            if source is not None:
                target[spec.name] = velocity
            else:
                dummy_outputs.append(velocity)
    _keep_dummy_gradients(velocities, dummy_outputs)
    return [
        BranchEvals(
            velocities=velocity,
            latents=request.latents,
            sigma=request.sigma,
            sigma_next=request.sigma_next,
            eta=request.eta,
            solver=request.solver,
        )
        for velocity, request in zip(velocities, requests, strict=True)
    ]


def evaluate(
    model: SamplerModel,
    guidance: BaseGuidance,
    batches: list[Batch],
    negative_batches: list[Batch | None],
    requests: list[EvalRequest],
    contexts: list[StepContext],
    projectors: Sequence[BaseProjector] = (),
    timesteps: list[torch.Tensor] | None = None,
) -> list[GuidanceOutput]:
    evals = evaluate_branches(
        model, guidance, batches, negative_batches, requests, contexts, timesteps
    )
    outputs = []
    for branches, request, batch, ctx in zip(
        evals, requests, batches, contexts, strict=True
    ):
        output, ctx.guidance_state = guidance.combine(branches, ctx, ctx.guidance_state)
        for projector in projectors:
            output = projector.post_combine(output, request, batch, ctx)
        outputs.append(output)
    return outputs
