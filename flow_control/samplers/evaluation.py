"""Shared branch evaluation and whole-image guidance for sampling and training."""

from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any, cast

import torch
import torch.distributed as dist

from flow_control.adapters.base import Batch, SamplerModel

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
    value = cast(dict[str, Any], positive).get(spec.batch_key)
    return cast(Batch, value) if isinstance(value, dict) else None


def _schedule(branches: list[list[BranchSpec]]) -> list[BranchSpec]:
    local = list(dict.fromkeys(spec for specs in branches for spec in specs))
    if not dist.is_initialized():
        return local
    # Replay microbatches can contain different step indices, hence different
    # variants on each rank. Materialize one ordered union before any forwards.
    gathered: list[list[BranchSpec] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    return list(
        dict.fromkeys(spec for group in gathered if group is not None for spec in group)
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
    schedule = _schedule(branches)
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
            outputs = model.predict_velocity_batched(inputs, timesteps)
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
