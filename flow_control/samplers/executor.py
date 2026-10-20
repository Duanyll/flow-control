"""Batches leaf model calls from many generators into collective adapter forwards."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from contextlib import closing
from itertools import islice
from typing import Any

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
    so all ranks issue the same forward sequence. A call runs under its own
    variant, else its row's ``RootFrame.variant``; the name goes to
    ``use_variant`` as given, and ``None`` is the adapter's active variant.
    Grad mode is the caller's. `stream` and `evaluate` drive,
    and on exit close, everything admitted, so they own the executor while
    they run.
    """

    def __init__(
        self, model: SamplerModel, variants: Sequence[str | None] = (None,)
    ) -> None:
        self.model = model
        self.variants = list(dict.fromkeys(variants))
        self._waiting: deque[tuple[Any, Calls[Any]]] = deque()
        self._active: list[tuple[Any, Calls[Any], list[ModelCall]]] = []

    @property
    def backlog(self) -> int:
        """Active leaf calls plus waiting generators; admit more while below ``model.micro_batch_size``."""
        return self._calls() + len(self._waiting)

    def admit(self, key: Any, gen: Calls[Any]) -> None:
        """Queue a generator; a later round primes it once the window has room."""
        self._waiting.append((key, gen))

    def round(self) -> list[tuple[Any, Any]]:
        """One collective round; returns ``(key, result)`` for every generator it finished.

        Waiting generators join while the active leaf calls are below
        ``model.micro_batch_size``; one that returns at priming finishes in
        this round. If no rank has active calls, nothing is forwarded. Every
        rank must call ``round`` the same number of times; agreeing on that
        count is the caller's job. After it raises, only `close` is safe.
        """
        return self._round(windowed=True)[0]

    def close(self) -> None:
        """Close every active and waiting generator, e.g. when the consumer stops early."""
        for _, gen, _ in self._active:
            gen.close()
        for _, gen in self._waiting:
            gen.close()
        self._active.clear()
        self._waiting.clear()

    def stream(self, runs: Iterable[tuple[Row, Calls[None]]]) -> Iterator[Row]:
        """Pull runs lazily while the backlog is below a microbatch; rows come out in completion order."""
        source = iter(runs)
        with closing(self):
            busy = True
            while busy:
                room = self.model.micro_batch_size - self.backlog
                for row, gen in islice(source, max(room, 0)):
                    self.admit(row, gen)
                finished, busy = self._round(windowed=True)
                for row, _ in finished:
                    yield row

    def evaluate[T](self, gens: Sequence[Calls[T]]) -> list[T]:
        """Run every generator at once (training microbatches); order preserved."""
        results: dict[int, T] = {}
        with closing(self):
            for index, gen in enumerate(gens):
                self.admit(index, gen)
            busy = True
            while busy:
                finished, busy = self._round(windowed=False)
                results.update(finished)
        return [results[index] for index in range(len(gens))]

    def _calls(self) -> int:
        return sum(len(calls) for *_, calls in self._active)

    def _round(self, windowed: bool) -> tuple[list[tuple[Any, Any]], bool]:
        """The round's finished generators, and whether any rank had work in it."""
        finished: list[tuple[Any, Any]] = []
        count = self._calls()
        while self._waiting and (not windowed or count < self.model.micro_batch_size):
            key, gen = self._waiting.popleft()
            try:
                self._active.append((key, gen, calls := next(gen)))
                count += len(calls)
            except StopIteration as stop:
                finished.append((key, stop.value))
        # 2: some rank forwards. 1: a generator finished at priming, so its
        # rank's caller may have more to admit and every rank keeps going.
        level = self._max_over_ranks(2 if self._active else int(bool(finished)))
        if level < 2:
            return finished, level > 0
        velocities = self._forward(
            [call for *_, calls in self._active for call in calls]
        )
        remaining: list[tuple[Any, Calls[Any], list[ModelCall]]] = []
        for key, gen, calls in self._active:
            values, velocities = velocities[: len(calls)], velocities[len(calls) :]
            try:
                remaining.append((key, gen, gen.send(values)))
            except StopIteration as stop:
                finished.append((key, stop.value))
        self._active = remaining
        return finished, True

    def _max_over_ranks(self, local: int) -> int:
        if not dist.is_initialized():
            return local
        flag = torch.tensor(local, device=self.model.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return int(flag.item())

    def _forward(self, calls: list[ModelCall]) -> list[torch.Tensor]:
        if len({id(call.row) for call in calls}) < len(calls):
            raise RuntimeError(
                "A row object appears in two model calls of one round; derive one "
                "row per concurrent branch (derived/follow)."
            )
        names = [call.variant or call.row["_frame"].root.variant for call in calls]
        unknown = set(names) - set(self.variants)
        if unknown:
            raise ValueError(
                f"Model calls ask for variants {sorted(unknown, key=str)} (the call's "
                f"own, else its row's RootFrame.variant) but this Executor runs "
                f"{self.variants}; build it with variants(tree) and the root variants."
            )
        velocities: dict[int, torch.Tensor] = {}
        dummies: list[torch.Tensor] = []
        for variant in self.variants:
            slots = [index for index, name in enumerate(names) if name == variant]
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


if __name__ == "__main__":
    from contextlib import contextmanager

    from rich import print

    from .node import RootFrame

    class FakeModel:
        device = torch.device("cpu")
        dtype = torch.float32
        micro_batch_size = 4

        def __init__(self) -> None:
            self.variant: str | None = None
            self.forwards: list[tuple[str | None, int]] = []

        @contextmanager
        def use_variant(self, variant: str | None, /) -> Iterator[None]:
            self.variant = variant
            try:
                yield
            finally:
                self.variant = None

        def predict_velocity_batched(
            self,
            batches: list[Any],
            timesteps: list[torch.Tensor],
            *,
            dummy_outputs: list[torch.Tensor] | None = None,
        ) -> list[torch.Tensor]:
            self.forwards.append((self.variant, len(batches)))
            return [torch.zeros_like(batch["noisy_latents"]) for batch in batches]

    def make_row(
        name: object = None, tokens: int = 1, variant: str | None = None
    ) -> Row:
        frame = RootFrame(None, variant=variant)
        return {
            "name": name,
            "noisy_latents": torch.zeros(1, tokens, 1),
            "sigma": 1.0,
            "_frame": frame,
        }

    closed: list[Row] = []

    def steps(row: Row, n: int, variant: str | None = None) -> Calls[Any]:
        """``n`` rounds of one call; ``n = 0`` returns at priming."""
        try:
            for _ in range(n):
                (velocity,) = yield [ModelCall(row, variant)]
                assert velocity.shape == row["noisy_latents"].shape  # routed back
            return n
        except GeneratorExit:
            closed.append(row)
            raise

    # admit + round: a generator finishes in its n-th round, at priming for n = 0;
    # a round where no rank has active calls forwards nothing.
    model = FakeModel()
    executor = Executor(model)
    for key, n in (("two", 2), ("zero", 0), ("one", 1)):
        executor.admit(key, steps(make_row(), n))
    assert executor.backlog == 3
    assert executor.round() == [("zero", 0), ("one", 1)] and executor.backlog == 1
    assert executor.round() == [("two", 2)] and executor.backlog == 0
    executor.admit("alone", steps(make_row(), 0))
    assert executor.round() == [("alone", 0)] and executor.round() == []
    assert model.forwards == [(None, 2), (None, 1)], model.forwards

    # The window: four calls per round; waiting generators take freed slots.
    model = FakeModel()
    executor = Executor(model)
    for index, n in enumerate((1, 2, 2, 2, 1, 1)):
        executor.admit(index, steps(make_row(), n))
    finished = [[key for key, _ in executor.round()] for _ in range(4)]
    assert finished == [[0], [1, 2, 3, 4], [5], []], finished
    assert [size for _, size in model.forwards] == [4, 4, 1], model.forwards

    # Variants: a call's own, else its root's; one batch per variant in the
    # executor's order, whatever the call order.
    model = FakeModel()
    gens = [
        steps(make_row(tokens=1, variant="b"), 1),
        steps(make_row(tokens=2, variant="a"), 1),
        steps(make_row(tokens=3, variant="a"), 1, variant="b"),
    ]
    assert Executor(model, ["a", "b"]).evaluate(gens) == [1, 1, 1]
    assert model.forwards == [("a", 1), ("b", 2)], model.forwards
    for variants, row, variant in (
        (["a"], make_row(), "c"),
        ([None], make_row(variant="a"), None),
    ):
        try:
            Executor(FakeModel(), variants).evaluate([steps(row, 1, variant)])
        except ValueError as error:
            assert "RootFrame.variant" in str(error), error
        else:
            raise AssertionError(f"{variant or 'root'} variant outside {variants}")

    # stream: completion order, and generators done at priming do not end it.
    rows = [make_row(n) for n in (3, 1, 0, 2)]
    streamed = Executor(FakeModel()).stream(
        (row, steps(row, row["name"])) for row in rows
    )
    assert [row["name"] for row in streamed] == [0, 1, 2, 3]
    rows = [make_row(n) for n in (0, 0, 0, 0, 0, 1)]
    streamed = Executor(FakeModel()).stream(
        (row, steps(row, row["name"])) for row in rows
    )
    assert len(list(streamed)) == 6
    # evaluate: input order, every generator in the first round (no window).
    model = FakeModel()
    gens = [steps(make_row(), n) for n in (3, 1, 0, 2, 1, 1)]
    assert Executor(model).evaluate(gens) == [3, 1, 0, 2, 1, 1]
    assert [size for _, size in model.forwards] == [5, 2, 1], model.forwards

    # Early exit closes what is in flight; stream pulls only a window of runs.
    closed.clear()
    pulled: list[Row] = []

    def runs() -> Iterator[tuple[Row, Calls[None]]]:
        for n in (1, 3, 3, 3, 3, 3):
            pulled.append(row := make_row(n))
            yield row, steps(row, n)

    executor = Executor(FakeModel())
    with closing(executor.stream(runs())) as streamed:
        assert next(streamed)["name"] == 1
    assert len(pulled) == 4 and len(closed) == 3 and executor.backlog == 0
    closed.clear()
    row = make_row()
    try:
        Executor(FakeModel()).evaluate([steps(row, 2), steps(row, 2)])
    except RuntimeError as error:
        assert "two model calls" in str(error), error
    assert len(closed) == 2, closed
    print("[green]executor checks passed[/green]")
