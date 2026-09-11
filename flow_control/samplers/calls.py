"""The shared coroutine protocol; only the executor performs model forwards."""

from collections.abc import Generator, Sequence
from dataclasses import dataclass
from typing import cast

import torch

from flow_control.adapters.base import Batch


@dataclass(slots=True)
class ModelCall:
    batch: Batch
    timestep: torch.Tensor
    variant: str | None = None


type Calls[T] = Generator[list[ModelCall], list[torch.Tensor], T]


def gather[T](generators: Sequence[Calls[T]]) -> Calls[list[T]]:
    """Join independent coroutines, yielding their ready leaf calls together.

    Child results retain input order, including children that return without
    a forward. This is local composition: no model, rank or microbatch logic.
    """
    results: dict[int, T] = {}
    pending: dict[int, list[ModelCall]] = {}
    try:
        for index, generator in enumerate(generators):
            try:
                pending[index] = next(generator)
            except StopIteration as stop:
                results[index] = cast(T, stop.value)
        while pending:
            calls = [call for calls in pending.values() for call in calls]
            velocities = (yield calls) if calls else []
            offset = 0
            for index, calls in list(pending.items()):
                values = velocities[offset : offset + len(calls)]
                offset += len(calls)
                try:
                    pending[index] = generators[index].send(values)
                except StopIteration as stop:
                    results[index] = cast(T, stop.value)
                    del pending[index]
        return [results[index] for index in range(len(generators))]
    finally:
        for generator in generators:
            generator.close()
