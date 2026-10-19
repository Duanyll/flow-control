"""Batches leaf model calls from many generators into collective adapter forwards."""

from __future__ import annotations

from collections.abc import Generator, Iterable, Iterator, Sequence
from contextlib import closing

import torch
import torch.distributed as dist

from flow_control.adapters.base import SamplerModel

from .calls import Calls, ModelCall
from .node import Row


def timestep(row: Row) -> torch.Tensor:
    return row["noisy_latents"].new_full((1,), row["sigma"], dtype=torch.float32)


class Executor:
    """Drive call generators together so each adapter forward sees a full microbatch.

    Every round is collective: one all-reduce decides whether any rank still
    has work, then ``predict_velocity_batched`` runs once per variant in the
    fixed ``variants`` order, with an empty list where this rank has nothing,
    so all ranks issue the same forward sequence. Grad mode is the caller's.
    """

    def __init__(
        self, model: SamplerModel, variants: Sequence[str | None] = (None,)
    ) -> None:
        self.model, self.variants = model, list(variants)

    def stream(self, runs: Iterable[tuple[Row, Calls[None]]]) -> Iterator[Row]:
        """Pull runs lazily while pending calls fit a microbatch; rows come out in completion order."""
        with closing(self._drive(iter(runs), self.model.micro_batch_size)) as driven:
            for row, _ in driven:
                yield row

    def evaluate[T](self, gens: Sequence[Calls[T]]) -> list[T]:
        """Run every generator at once (training microbatches); order preserved."""
        results = dict(self._drive(iter(enumerate(gens)), None))
        return [results[index] for index in range(len(gens))]

    def _drive[K, T](
        self, source: Iterator[tuple[K, Calls[T]]], window: int | None
    ) -> Generator[tuple[K, T], None, None]:
        active: list[tuple[K, Calls[T], list[ModelCall]]] = []
        exhausted = False
        try:
            while True:
                while not exhausted and (
                    window is None or sum(len(calls) for *_, calls in active) < window
                ):
                    try:
                        key, gen = next(source)
                    except StopIteration:
                        exhausted = True
                        break
                    try:
                        active.append((key, gen, next(gen)))
                    except StopIteration as stop:
                        yield key, stop.value
                if not self._anyone_active(bool(active)):
                    return
                velocities = self._forward(
                    [call for *_, calls in active for call in calls]
                )
                remaining: list[tuple[K, Calls[T], list[ModelCall]]] = []
                for key, gen, calls in active:
                    values, velocities = (
                        velocities[: len(calls)],
                        velocities[len(calls) :],
                    )
                    try:
                        remaining.append((key, gen, gen.send(values)))
                    except StopIteration as stop:
                        yield key, stop.value
                active = remaining
        finally:
            for _, gen, _ in active:
                gen.close()

    def _anyone_active(self, local: bool) -> bool:
        if not dist.is_initialized():
            return local
        flag = torch.tensor(int(local), device=self.model.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def _forward(self, calls: list[ModelCall]) -> list[torch.Tensor]:
        if len({id(call.row) for call in calls}) < len(calls):
            raise RuntimeError(
                "A row object appears in two model calls of one round; derive one "
                "row per concurrent branch (derived/follow)."
            )
        unknown = {call.variant for call in calls} - set(self.variants)
        if unknown:
            raise ValueError(
                f"Model calls ask for variants {sorted(unknown, key=str)} but this "
                f"Executor runs {self.variants}; build it with variants(tree)."
            )
        velocities: dict[int, torch.Tensor] = {}
        dummies: list[torch.Tensor] = []
        for variant in self.variants:
            slots = [
                index for index, call in enumerate(calls) if call.variant == variant
            ]
            with self.model.use_variant(variant):
                outputs = self.model.predict_velocity_batched(
                    [calls[index].row for index in slots],
                    [timestep(calls[index].row) for index in slots],
                    dummy_outputs=dummies,
                )
            velocities.update(zip(slots, outputs, strict=True))
        if dummies:
            if not calls:
                raise RuntimeError(
                    "Distributed training needs at least one local evaluation to "
                    "carry dummy forward gradients; only no-grad streams may have "
                    "an entirely empty rank."
                )
            velocities[0] = velocities[0] + sum(output.sum() * 0 for output in dummies)
        for call in calls:
            if (on_call := call.row["_frame"].root.on_call) is not None:
                on_call(call.row)
        return [velocities[index] for index in range(len(calls))]
