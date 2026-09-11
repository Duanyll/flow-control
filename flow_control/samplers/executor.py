"""One rendezvous loop for cross-request and distributed sampling."""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterator, Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist

from flow_control.adapters.base import Batch, SamplerModel

from .evaluation import evaluate
from .guidance import BaseGuidance
from .plan import (
    EvalRequest,
    GuidanceOutput,
    SamplingPlan,
    StepContext,
    Transition,
    TransitionResult,
)
from .projectors import BaseProjector, apply_pre_transition


@dataclass(slots=True)
class Run:
    plan: SamplingPlan
    ctx: StepContext
    batch: Batch
    negative_batch: Batch | None


@dataclass(slots=True)
class StepEvent:
    step_idx: int
    total_steps: int


StepObserver = Callable[
    [int, Transition, StepContext, torch.Tensor | None, torch.Tensor], None
]
"""Observe a completed transition before the executor advances its context."""


@dataclass(slots=True)
class _EvalRound:
    entries: list[tuple[Run, EvalRequest]]


def validate_distributed_request_count(
    count: int, device: torch.device, operation: str
) -> None:
    if not dist.is_initialized():
        return
    counts = torch.tensor([count, -count], device=device, dtype=torch.int64)
    dist.all_reduce(counts, op=dist.ReduceOp.MIN)
    if int(counts[0].item()) != -int(counts[1].item()):
        raise ValueError(
            f"All distributed ranks must submit the same number of requests "
            f"to {operation}."
        )


def _sync_eval_target(local_count: int, device: torch.device) -> int:
    if not dist.is_initialized():
        return local_count
    counts = torch.tensor([local_count, -local_count], device=device, dtype=torch.int64)
    dist.all_reduce(counts, op=dist.ReduceOp.MIN)
    target = -int(counts[1].item())
    if int(counts[0].item()) != target:
        raise ValueError(
            "Sampling transitions requested different evaluation counts across "
            "ranks; solver configurations and execution topology must match."
        )
    return target


def _result(stop: StopIteration) -> TransitionResult:
    result = stop.value
    assert isinstance(result, TransitionResult)
    return result


def _apply_result(run: Run, result: TransitionResult) -> None:
    run.ctx.latents = result.next_latents
    if result.next_solver_state is not None:
        run.ctx.solver_state = result.next_solver_state


def _drive(
    runs: list[Run],
    projectors: Sequence[BaseProjector],
    observer: StepObserver | None,
) -> Generator[_EvalRound | StepEvent, list[GuidanceOutput] | None, None]:
    num_items = len(runs[0].plan)
    for item_index in range(num_items):
        for run in runs:
            run.ctx.item_index = item_index
            run.ctx.num_items = num_items
            run.ctx.latents = apply_pre_transition(
                projectors, run.batch, run.ctx, run.plan[item_index]
            )
        generators = [run.plan[item_index].run(run.ctx) for run in runs]
        results: list[TransitionResult | None] = [None] * len(runs)
        velocities: list[torch.Tensor | None] = [None] * len(runs)
        pending: dict[int, EvalRequest] = {}
        for index, generator in enumerate(generators):
            try:
                pending[index] = next(generator)
            except StopIteration as stop:
                results[index] = _result(stop)
        while pending:
            order = list(pending)
            outputs = yield _EvalRound([(runs[i], pending[i]) for i in order])
            assert outputs is not None and len(outputs) == len(order)
            pending = {}
            for index, output in zip(order, outputs, strict=True):
                velocities[index] = output.velocity
                try:
                    pending[index] = generators[index].send(output)
                except StopIteration as stop:
                    results[index] = _result(stop)
        for index, (run, result) in enumerate(zip(runs, results, strict=True)):
            assert result is not None
            if observer is not None:
                observer(
                    index,
                    run.plan[item_index],
                    run.ctx,
                    velocities[index],
                    result.next_latents,
                )
            _apply_result(run, result)
        yield StepEvent(item_index, num_items)


def execute(
    model: SamplerModel,
    runs: list[Run],
    guidance: BaseGuidance,
    projectors: Sequence[BaseProjector] = (),
    observer: StepObserver | None = None,
) -> Iterator[StepEvent]:
    """Execute matching plans; each model round rendezvous is collective."""
    if not runs:
        raise ValueError("execute requires at least one run.")
    driver = _drive(runs, projectors, observer)
    try:
        item: _EvalRound | StepEvent | None = next(driver)
    except StopIteration:
        item = None
    while True:
        while isinstance(item, StepEvent):
            yield item
            try:
                item = next(driver)
            except StopIteration:
                item = None
        entries = item.entries if isinstance(item, _EvalRound) else []
        if _sync_eval_target(len(entries), model.device) == 0:
            return
        outputs = evaluate(
            model=model,
            guidance=guidance,
            batches=[run.batch for run, _ in entries],
            negative_batches=[run.negative_batch for run, _ in entries],
            requests=[request for _, request in entries],
            contexts=[run.ctx for run, _ in entries],
            projectors=projectors,
        )
        try:
            item = driver.send(outputs)
        except StopIteration:
            item = None
