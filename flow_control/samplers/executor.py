"""Batches leaf model calls from many generators into collective adapter forwards."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist

from flow_control.adapters.base import SamplerModel

from .calls import Calls, ModelCall
from .run import SampleRun


@dataclass(slots=True)
class _Active[T]:
    gen: Calls[T]
    calls: list[ModelCall]
    velocities: dict[int, torch.Tensor] = field(default_factory=dict)


class Executor:
    """Drive call generators together so each adapter forward sees a full microbatch.

    Every round is collective: one all-reduce decides whether any rank still
    has work, then ``predict_velocity_batched`` runs once per variant in the
    fixed ``variants`` order, with an empty list where this rank has nothing,
    so all ranks issue the same forward sequence. Unequal sample counts are the
    adapter's business (it pads with dummy forwards). Grad mode is the caller's.
    """

    def __init__(
        self, model: SamplerModel, variants: Sequence[str | None] = (None,)
    ) -> None:
        self.model = model
        self.variants = list(variants)

    def stream(self, runs: Iterable[SampleRun]) -> Iterator[SampleRun]:
        """Sample runs lazily, pulling a new one only while pending calls fit a
        microbatch; finished runs come out in completion order."""
        source = ((run, run.run()) for run in runs)
        for run, _ in self._drive(source, self.model.micro_batch_size):
            yield run

    def evaluate[T](self, gens: Sequence[Calls[T]]) -> list[T]:
        """Run every generator at once (training microbatches); order preserved."""
        results = dict(self._drive(iter(enumerate(gens)), None))
        return [results[index] for index in range(len(gens))]

    def _drive[K, T](
        self, source: Iterator[tuple[K, Calls[T]]], window: int | None
    ) -> Iterator[tuple[K, T]]:
        active: list[tuple[K, _Active[T]]] = []
        exhausted = False
        while True:
            while not exhausted and (
                window is None or sum(len(entry.calls) for _, entry in active) < window
            ):
                try:
                    key, gen = next(source)
                except StopIteration:
                    exhausted = True
                    break
                try:
                    active.append((key, _Active(gen, self._checked(next(gen)))))
                except StopIteration as stop:
                    yield key, stop.value
            if not self._anyone_active(bool(active)):
                return
            self._forward(active)
            remaining: list[tuple[K, _Active[T]]] = []
            for key, entry in active:
                velocities = [entry.velocities[i] for i in range(len(entry.calls))]
                try:
                    calls = entry.gen.send(velocities)
                except StopIteration as stop:
                    yield key, stop.value
                else:
                    remaining.append((key, _Active(entry.gen, self._checked(calls))))
            active = remaining

    def _checked(self, calls: list[ModelCall]) -> list[ModelCall]:
        unknown = {call.variant for call in calls} - set(self.variants)
        if unknown:
            raise ValueError(
                f"Model calls ask for weight variants {sorted(unknown, key=str)} "
                f"but this Executor only runs {self.variants}; build it with "
                "Sampler.variant_keys()."
            )
        return calls

    def _anyone_active(self, local: bool) -> bool:
        if not dist.is_initialized():
            return local
        flag = torch.tensor(int(local), device=self.model.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def _forward(self, active: list[tuple[Any, _Active[Any]]]) -> None:
        dummies: list[torch.Tensor] = []
        for variant in self.variants:
            slots = [
                (entry, index)
                for _, entry in active
                for index, call in enumerate(entry.calls)
                if call.variant == variant
            ]
            with self.model.use_variant(variant):
                velocities = self.model.predict_velocity_batched(
                    [entry.calls[index].batch for entry, index in slots],
                    [entry.calls[index].timestep for entry, index in slots],
                    dummy_outputs=dummies,
                )
            for (entry, index), velocity in zip(slots, velocities, strict=True):
                entry.velocities[index] = velocity
        if dummies:
            if not active:
                raise RuntimeError(
                    "Distributed training needs at least one local evaluation "
                    "to carry dummy forward gradients; only no-grad streams may "
                    "have an entirely empty rank."
                )
            entry = active[0][1]
            entry.velocities[0] = entry.velocities[0] + sum(
                output.sum() * 0 for output in dummies
            )
