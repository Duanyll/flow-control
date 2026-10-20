"""The device worker and the reward loop behind rollout programs (``docs/rollout-reward-design.md``).

A :class:`Runner` owns two threads for the whole run. The main thread is the
*device worker*: it runs the model forwards through one :class:`Executor`,
starts and decodes rows, batches local rewards and snapshots rows for remote
ones, one *round* at a time inside :meth:`Pending.sampled` / :meth:`Pending.wait`
/ :meth:`Pending.as_completed`. A daemon thread runs the *reward loop*, an
asyncio loop where the programs (``sample`` / ``score`` coroutines) and the
remote reward requests live; it never touches CUDA or the model. The loop asks
the worker for device work through one queue of request objects, each carrying
an asyncio future the worker resolves back on the loop.

Rows are shared, not copied. One thread touches a row at a time: the device
worker inside an awaited ``ctx`` call, the program otherwise. Programs run no
CUDA ops and no collectives; every ``ctx`` method does its device work on the
worker. Every round is collective (the ranks agree on whether anyone still has
executor work), so the three ``Pending`` methods must be called by every rank.

A group has two phases: ``sample`` (every model call of the group; the prepared
prompt row stays on the device until it returns) and ``score`` (rewards only).
An exception in either, in a reward or in a sampler node fails the group, and
the next round raises on every rank.
"""

from __future__ import annotations

import asyncio
import contextlib
import queue
import threading
from collections import deque
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, NoReturn, Protocol, cast

import torch
import torch.distributed as dist

from flow_control.adapters.base import SamplerModel
from flow_control.processors.base import BaseProcessor
from flow_control.rewards.base import LocalReward, RemoteReward, Reward, run_local_batch
from flow_control.rewards.profile import RewardProfile
from flow_control.samplers import (
    Calls,
    Executor,
    Observer,
    RootFrame,
    Row,
    Trajectory,
    data,
    derive_seed,
)
from flow_control.utils.logging import get_logger
from flow_control.utils.tensor import deep_move_to_device

logger = get_logger(__name__)

_CPU = torch.device("cpu")
_SAMPLING, _SAMPLED, _SCORED = 0, 1, 2
_LEVEL = {"sampled": _SAMPLED, "scored": _SCORED}
_IDLE_WAIT_S = 0.02


class RunnerModel(SamplerModel, Protocol):
    """``SamplerModel`` plus the variant table of ``VariantsMixin``."""

    @property
    def variants(self) -> Mapping[str, object]: ...


# --------------------------------- requests ---------------------------------- #
# Built on the loop (the future binds to the running loop), served on the worker.


def _future() -> asyncio.Future[Any]:
    return asyncio.get_running_loop().create_future()


def _settle(
    future: asyncio.Future[Any], value: Any = None, error: BaseException | None = None
) -> None:
    """Resolve a loop future from the worker thread."""
    if error is None:
        future.get_loop().call_soon_threadsafe(future.set_result, value)
    else:
        future.get_loop().call_soon_threadsafe(future.set_exception, error)


@dataclass(slots=True)
class _Start:
    prepared: Row
    seed: int
    variant: str | None
    observers: Sequence[tuple[type, Observer]]
    future: asyncio.Future[Any] = field(default_factory=_future)


@dataclass(slots=True)
class _Admit:
    group: _Group
    gen: Calls[Any]
    future: asyncio.Future[Any] = field(default_factory=_future)


@dataclass(slots=True)
class _Decode:
    row: Row
    future: asyncio.Future[Any] = field(default_factory=_future)


@dataclass(slots=True)
class _Snapshot:
    fields: set[str]
    items: list[Any]
    future: asyncio.Future[Any] = field(default_factory=_future)


@dataclass(slots=True)
class _Local:
    """Items for one local reward; one future per item so the worker can rebatch across groups."""

    reward: LocalReward[Any, Any]
    items: list[Any]
    futures: list[asyncio.Future[Any]] = field(init=False)

    def __post_init__(self) -> None:
        self.futures = [_future() for _ in self.items]


type _Request = _Start | _Admit | _Decode | _Snapshot | _Local


@dataclass(slots=True)
class _Raised:
    """A sampler node's exception, carried out of the executor as the generator's value."""

    error: Exception


def _guarded(gen: Calls[Any]) -> Calls[Any]:
    """Keep a failing node from raising out of ``Executor.round`` and taking every other generator with it."""
    try:
        return (yield from gen)
    except Exception as error:
        return _Raised(error)


async def _profiled[T](request: Awaitable[T], profile: RewardProfile) -> T:
    """Stamp issue and completion on the loop thread, before the result reaches the waiter."""
    index = profile.on_submit()
    try:
        return await request
    finally:
        profile.on_done(index)


# ---------------------------------- groups ----------------------------------- #


@dataclass(eq=False)
class _Group:
    """One prompt's rollouts: a ``sample`` then a ``score`` run on the loop."""

    pending: Pending
    prompt_id: int
    item: Row
    state: int = _SAMPLING
    """Written by the loop thread, read by the worker."""
    admitted: bool = False
    """Has admitted a generator (worker-side); until then it counts as ``k`` prospective calls."""
    rows: list[Row] = field(default_factory=list)
    error: BaseException | None = None


class RewardContext:
    """What ``score()`` gets: the prompt and the one way to call any reward."""

    def __init__(self, runner: Runner, group: _Group) -> None:
        self._runner, self._group = runner, group

    @property
    def prompt(self) -> Row:
        """The store item, on CPU."""
        return self._group.item

    @property
    def prompt_id(self) -> int:
        return self._group.prompt_id

    @property
    def k(self) -> int:
        """``num_rollouts_per_prompt``; 1 in validation and inference."""
        return self._group.pending.k

    @property
    def epoch(self) -> int:
        return self._group.pending.epoch

    def seed(self, *parts: object) -> int:
        return derive_seed(self._runner.seed, ":".join(map(str, parts)))

    async def run[I, O](self, reward: Reward[I, O], items: list[I]) -> list[O]:
        """Outputs in item order: a local reward batched on the worker, a remote one
        as concurrent requests over CPU snapshots of the rows inside each item."""
        if isinstance(reward, LocalReward):
            request = _Local(reward, list(items))
            self._runner._requests.put(request)
            return list(await asyncio.gather(*request.futures))
        snapshots = await self._runner._request(_Snapshot(reward.fields, list(items)))
        profile = self._group.pending.profile
        return list(
            await asyncio.gather(
                *(_profiled(reward.run_one(item), profile) for item in snapshots)
            )
        )


class SampleContext(RewardContext):
    """What ``sample()`` gets; its methods raise once ``sample()`` has returned."""

    def __init__(self, runner: Runner, group: _Group, prepared: Row) -> None:
        super().__init__(runner, group)
        self._prepared: Row | None = prepared
        self._next_rollout = 0

    def _sampling(self) -> Row:
        if self._prepared is None:
            raise RuntimeError(
                "start/admit/decode/rollouts belong to the sample phase; this "
                "group's sample() has returned."
            )
        return self._prepared

    async def start(self, seed: int, variant: str | None = None) -> Row:
        """The prepared prompt row with fresh noise at σ = 1, its generator seeded,
        and a ``RootFrame`` under ``variant`` (None: ``submit()``'s; if that is None
        too, the adapter's active variant)."""
        return await self._start(seed, variant, ())

    async def _start(
        self, seed: int, variant: str | None, observers: Sequence[tuple[type, Observer]]
    ) -> Row:
        prepared = self._sampling()
        variant = variant or self._group.pending.variant
        return await self._runner._request(_Start(prepared, seed, variant, observers))

    async def admit[T](self, gen: Calls[T]) -> T:
        """Run a sampler node's generator; it joins the executor at its next round."""
        self._sampling()
        return await self._runner._request(_Admit(self._group, gen))

    async def decode(self, row: Row) -> list[str]:
        """``processor.decode_output`` in place; the row then moves to the runner's
        storage. Returns the keys ``decode_output`` wrote."""
        self._sampling()
        return await self._runner._request(_Decode(row))

    async def rollouts(self, n: int | None = None) -> list[Row]:
        """The policy samples, in index order: ``start(seed("rollout", epoch,
        prompt_id, i))`` under the submit's observers, ``sampler`` admitted,
        decoded, ``row["epoch"]`` set. ``n`` defaults to ``k``; a later call
        continues the indices."""
        self._sampling()
        # submit(sampler=) is what ctx.rollouts() samples with.
        sampler = cast("Trajectory", self._group.pending.sampler)
        count = self.k if n is None else n
        first, self._next_rollout = self._next_rollout, self._next_rollout + count
        return list(
            await asyncio.gather(
                *(self._rollout(sampler, i) for i in range(first, first + count))
            )
        )

    async def _rollout(self, sampler: Trajectory, index: int) -> Row:
        pending = self._group.pending
        seed = self.seed("rollout", self.epoch, self.prompt_id, index)
        row = await self._start(seed, None, pending.observers)
        row["epoch"] = self.epoch
        await self.admit(sampler(row))
        await self.decode(row)
        return row


class Pending:
    """The groups of one ``submit()``. Its methods drive the device worker and
    are collective: every rank calls the same one."""

    def __init__(
        self,
        runner: Runner,
        items: Iterator[tuple[int, Row]],
        sample: Callable[[SampleContext], Awaitable[Any]],
        score: Callable[[RewardContext, Any], Awaitable[list[Row]]],
        prepare: Callable[[Row], Row],
        k: int,
        epoch: int,
        variant: str | None,
        sampler: Trajectory | None,
        observers: Sequence[tuple[type, Observer]],
    ) -> None:
        self._runner, self._items = runner, items
        self.sample, self.score, self.prepare = sample, score, prepare
        self.k, self.epoch, self.variant = k, epoch, variant
        self.sampler, self.observers = sampler, tuple(observers)
        self.profile = RewardProfile()
        """One entry per remote request of these groups."""
        self._groups: list[_Group] = []
        self._completed: deque[_Group] = deque()
        """Scored groups in completion order (appended on the loop thread)."""
        self._exhausted = False
        self._error: BaseException | None = None
        """A failure pulling or preparing an item (worker-side)."""

    def sampled(self) -> None:
        """Serve device work until every group, on every rank, has left ``sample()``."""
        while not self._runner._round(self, "sampled"):
            pass

    def wait(self) -> list[list[Row]]:
        """Until every ``score()`` returned; the groups' rows in submission order."""
        while not self._runner._round(self, "scored"):
            pass
        return [group.rows for group in self._groups]

    def as_completed(self) -> Iterator[list[Row]]:
        """Each group's rows as it finishes, in completion order."""
        while True:
            finished = self._runner._round(self, "scored")
            while self._completed:
                yield self._completed.popleft().rows
            if finished:
                return

    def _reached(self, level: int) -> bool:
        return self._exhausted and all(g.state >= level for g in self._groups)


# ---------------------------------- runner ----------------------------------- #


class Runner:
    """The device worker (the calling thread) and the reward loop (a daemon
    thread), owned by the trainer for the whole run; inference opens its own."""

    def __init__(
        self,
        model: RunnerModel,
        processor: BaseProcessor,
        storage: torch.device,
        seed: int,
    ) -> None:
        """``storage``: where decoded rows go (``rollout_storage``); ``seed``: the
        trainer's, the base of every ``ctx.seed()``."""
        self.model, self.processor, self.storage, self.seed = (
            model,
            processor,
            storage,
            seed,
        )
        # Every variant of the model, in one order on every rank; None is the
        # adapter's active variant.
        self.executor = Executor(model, [None, "base", *model.variants])
        self._requests: queue.SimpleQueue[_Request] = queue.SimpleQueue()
        self._pending: list[Pending] = []
        self._rows: deque[_Start | _Decode | _Snapshot] = deque()
        self._local: dict[
            int, tuple[LocalReward[Any, Any], deque[tuple[Any, asyncio.Future[Any]]]]
        ] = {}
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="reward-loop", daemon=True
        )
        self._thread.start()

    def submit(
        self,
        items: Iterable[tuple[int, Row]],
        sample: Callable[[SampleContext], Awaitable[Any]],
        score: Callable[[RewardContext, Any], Awaitable[list[Row]]],
        *,
        prepare: Callable[[Row], Row],
        k: int,
        epoch: int,
        variant: str | None,
        sampler: Trajectory | None = None,
        observers: Sequence[tuple[type, Observer]] = (),
    ) -> Pending:
        """Register ``(prompt_id, store item)`` groups, pulled lazily as the executor
        has room; ``prepare`` turns an item into the prepared row on the device
        when its group starts. Nothing runs until a ``Pending`` method is called."""
        pending = Pending(
            self,
            iter(items),
            sample,
            score,
            prepare,
            k,
            epoch,
            variant,
            sampler,
            observers,
        )
        self._pending.append(pending)
        return pending

    def _groups(self) -> Iterator[_Group]:
        for pending in self._pending:
            yield from pending._groups

    def close(self) -> None:
        unfinished = sum(g.state < _SCORED and g.error is None for g in self._groups())
        if unfinished:
            logger.warning(f"Runner closed with {unfinished} groups still in flight.")
        self.executor.close()
        asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
        self._thread.join()
        self._loop.close()

    async def _shutdown(self) -> None:
        """Cancel the programs and requests still on the loop, then stop it."""
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._loop.stop()

    def _error(self) -> BaseException | None:
        """The first failure among the registered submissions and their groups."""
        for pending in self._pending:
            if pending._error is not None:
                return pending._error
            for group in pending._groups:
                if group.error is not None:
                    return group.error
        return None

    # ------------------------------- the round -------------------------------- #

    @torch.no_grad()
    def _round(self, pending: Pending, until: Literal["sampled", "scored"]) -> bool:
        """One device-worker round, all of it under ``no_grad``; True once every
        group of ``pending`` on every rank reached ``until``. Raises when any
        group failed, on every rank."""
        self._take_requests()
        self._start_group()
        error = self._error()
        rank = dist.get_rank() if dist.is_initialized() else 0
        busy, unfinished, failed_rank = self._all_reduce(
            [
                int(self._busy()),
                int(not pending._reached(_LEVEL[until])),
                rank + 1 if error is not None else 0,
            ]
        )
        if failed_rank:
            self._raise(error, failed_rank - 1)
        if busy:  # collective: every rank calls it, or none
            self._forward()
        self._serve_local()
        while self._rows:
            self._serve(self._rows.popleft())
        self._pending = [p for p in self._pending if not p._reached(_SCORED)]
        if not busy and unfinished:
            self._idle()
        return not unfinished

    def _raise(self, error: BaseException | None, rank: int) -> NoReturn:
        if error is not None:
            raise error  # this rank's own failure, with its traceback
        raise RuntimeError(
            f"A rollout group failed on rank {rank}; its traceback is in that rank's log."
        )

    def _forward(self) -> None:
        for request, value in self.executor.round():
            if isinstance(value, _Raised):
                _settle(request.future, error=value.error)
            else:
                _settle(request.future, value)

    def _idle(self) -> None:
        """Nothing to forward on any rank: wait for a request instead of spinning."""
        with contextlib.suppress(queue.Empty):
            self._dispatch(self._requests.get(timeout=_IDLE_WAIT_S))

    def _busy(self) -> bool:
        """This rank has, or is about to have, executor work."""
        return (
            self.executor.backlog > 0
            or bool(self._rows)
            or any(buffer for _, buffer in self._local.values())
            or (any(not p._exhausted for p in self._pending) and self._has_room())
        )

    def _all_reduce(self, flags: list[int]) -> list[int]:
        if not dist.is_initialized():
            return flags
        tensor = torch.tensor(flags, device=self.model.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return tensor.tolist()

    def _take_requests(self) -> None:
        while True:
            try:
                request = self._requests.get_nowait()
            except queue.Empty:
                return
            self._dispatch(request)

    def _dispatch(self, request: _Request) -> None:
        if isinstance(request, _Admit):
            request.group.admitted = True
            self.executor.admit(request, _guarded(request.gen))
        elif isinstance(request, _Local):
            _, buffer = self._local.setdefault(
                id(request.reward), (request.reward, deque())
            )
            buffer.extend(zip(request.items, request.futures, strict=True))
        else:
            self._rows.append(request)

    def _has_room(self) -> bool:
        """The executor's window has room for another group, where a started
        group that has not admitted yet counts as its ``k`` calls."""
        prospective = sum(
            g.pending.k
            for g in self._groups()
            if g.state == _SAMPLING and not g.admitted
        )
        return self.executor.backlog + prospective < self.model.micro_batch_size

    def _start_group(self) -> None:
        """Start the next group of the earliest submission with items left, at most
        one per round and only while the executor's window has room."""
        if not self._has_room():
            return
        for pending in self._pending:
            if pending._exhausted:
                continue
            try:
                prompt_id, item = next(pending._items)
                prepared = pending.prepare(item)
            except StopIteration:
                pending._exhausted = True
                continue
            except Exception as error:  # the next all-reduce raises it on every rank
                pending._error, pending._exhausted = error, True
                return
            group = _Group(pending, prompt_id, item)
            pending._groups.append(group)
            ctx = SampleContext(self, group, prepared)
            asyncio.run_coroutine_threadsafe(self._run(group, ctx), self._loop)
            return

    async def _run(self, group: _Group, ctx: SampleContext) -> None:
        """The group's program, on the loop."""
        pending = group.pending
        try:
            try:
                result = await pending.sample(ctx)
            finally:
                ctx._prepared = None  # the prepared row leaves the device
            group.state = _SAMPLED
            group.rows = await pending.score(ctx, result)
            pending._completed.append(group)
            group.state = _SCORED
        except BaseException as error:
            group.error = error

    # -------------------------- loop -> worker plumbing ------------------------- #

    async def _request(self, request: _Start | _Admit | _Decode | _Snapshot) -> Any:
        self._requests.put(request)
        return await request.future

    # ------------------------------ device work ------------------------------- #

    def _serve_local(self) -> None:
        """Drain every buffered item of every local reward in ``batch_size`` chunks."""
        for reward, buffer in self._local.values():
            while buffer:
                batch = [
                    buffer.popleft() for _ in range(min(reward.batch_size, len(buffer)))
                ]
                try:
                    outputs = run_local_batch(reward, [item for item, _ in batch])
                except Exception as error:
                    for _, future in batch:
                        _settle(future, error=error)
                else:
                    for (_, future), output in zip(batch, outputs, strict=True):
                        _settle(future, output)

    def _serve(self, request: _Start | _Decode | _Snapshot) -> None:
        try:
            if isinstance(request, _Start):
                value = self._start(request)
            elif isinstance(request, _Decode):
                value = self._decode(request.row)
            else:
                value = self._snapshot(request.items, request.fields)
        except Exception as error:
            _settle(request.future, error=error)
        else:
            _settle(request.future, value)

    def _start(self, request: _Start) -> Row:
        """A fresh row sharing the prepared tensors, at σ = 1 with its own noise and RNG."""
        model = self.model
        row = data(request.prepared)
        generator = torch.Generator(device=model.device).manual_seed(request.seed)
        row["noisy_latents"] = self.processor.initialize_latents(
            row, generator=generator, device=model.device, dtype=model.dtype
        ).float()
        row["sigma"] = 1.0
        row["_generator"] = generator
        row["_frame"] = RootFrame(
            self,
            request.observers,
            variant=request.variant,
        )
        return row

    def _decode(self, row: Row) -> list[str]:
        """Decode in place, then move the row to storage keeping the dict object
        (the program holds it) and dropping the runtime keys; the decoded keys."""
        row["clean_latents"] = row["noisy_latents"]
        decoded = self.processor.decode_output(row["noisy_latents"], row)
        row.update(decoded)
        moved = deep_move_to_device(data(row), self.storage, preserve_aliases=True)
        row.clear()
        row.update(moved)
        return list(decoded)

    def _snapshot(self, items: list[Any], fields: set[str]) -> list[Any]:
        """Every row inside an item (a row, or a tuple / list holding rows) filtered
        to ``fields``, anything else passed through, then moved to CPU in one pass:
        a row in many items (round-robin pairs) copies each tensor once."""

        def keep(value: Any) -> Any:
            if isinstance(value, dict):
                return {key: value[key] for key in value.keys() & fields}
            if isinstance(value, list | tuple):
                return type(value)(keep(v) for v in value)
            return value

        return deep_move_to_device(
            [keep(item) for item in items], _CPU, preserve_aliases=True
        )


if __name__ == "__main__":
    import random
    import time
    from contextlib import contextmanager

    from pydantic import PrivateAttr
    from rich import print

    from flow_control.samplers import Grid

    class FakeModel:
        device = torch.device("cpu")
        dtype = torch.float32
        micro_batch_size = 4
        variants: dict[str, object] = {"default": None, "ema": None}

        def __init__(self) -> None:
            self.active_variant = "default"
            self.forwards: list[tuple[str, int]] = []
            """``(active variant, rows)`` of every non-empty forward."""

        @contextmanager
        def use_variant(self, name: str | None, /) -> Iterator[None]:
            previous = self.active_variant
            if name is not None:
                self.active_variant = name
            try:
                yield
            finally:
                self.active_variant = previous

        def predict_velocity_batched(
            self,
            batches: list[Any],
            timesteps: list[torch.Tensor],
            *,
            dummy_outputs: list[torch.Tensor] | None = None,
        ) -> list[torch.Tensor]:
            if batches:
                self.forwards.append((self.active_variant, len(batches)))
            return [torch.zeros_like(batch["noisy_latents"]) for batch in batches]

    class FakeProcessor:
        def initialize_latents(
            self,
            row: Any,
            generator: torch.Generator | None = None,
            device: Any = None,
            dtype: torch.dtype = torch.float32,
        ) -> torch.Tensor:
            return torch.randn(1, 2, 4, 4, generator=generator, dtype=dtype)

        def decode_output(self, latents: torch.Tensor, row: Any) -> dict[str, Any]:
            return {"clean_image": latents.clamp(0, 1)}

    def check_snapshot(row: Row) -> None:
        assert set(row) == {"clean_image"}, sorted(row)
        assert row["clean_image"].device.type == "cpu"

    class FakeLocal(LocalReward[Row, torch.Tensor]):
        batch_size: int = 8
        delay: float = 0.0
        _sizes: list[int] = PrivateAttr(default_factory=list)

        def run_batch(self, rows: list[Row]) -> list[torch.Tensor]:
            self._sizes.append(len(rows))
            time.sleep(self.delay)
            return [row["clean_image"].mean().reshape(1) for row in rows]

    class FakeRemote(RemoteReward[Row, torch.Tensor]):
        delay: float = 0.0
        _done: int = PrivateAttr(default=0)

        @property
        def fields(self) -> set[str]:
            return {"clean_image"}

        async def run_one(self, row: Row) -> torch.Tensor:
            check_snapshot(row)
            await asyncio.sleep(random.uniform(0.001, 0.005) + self.delay)
            self._done += 1
            return row["clean_image"].mean().reshape(1)

    class FakeJudge(RemoteReward[tuple[Row, Row], float]):
        @property
        def fields(self) -> set[str]:
            return {"clean_image"}

        async def run_one(self, pair: tuple[Row, Row]) -> float:
            a, b = pair
            check_snapshot(a)
            check_snapshot(b)
            await asyncio.sleep(random.uniform(0.001, 0.005))
            return float(a["clean_image"].mean() > b["clean_image"].mean())

    def items(n: int, start: int = 0) -> list[tuple[int, Row]]:
        return [
            (
                i,
                {
                    "key": f"p{i}",
                    "image_size": (32, 32),
                    "prompt_embeds": torch.zeros(1, 3),
                },
            )
            for i in range(start, start + n)
        ]

    def check_decoded(row: Row) -> None:
        assert not any(key.startswith("_") for key in row), sorted(row)
        assert row["clean_image"].device.type == "cpu" and "clean_latents" in row

    async def sample(ctx: SampleContext) -> list[Row]:
        return await ctx.rollouts()

    def write(rows: list[Row], *parts: list[Any]) -> list[Row]:
        for row, *values in zip(rows, *parts, strict=True):
            row["reward_raw"] = torch.cat(
                [torch.as_tensor(v).reshape(-1) for v in values]
            )
            row["reward"] = row["reward_raw"] * 2
        return rows

    def scored_by(*rewards: Reward[Row, torch.Tensor]):
        async def score(ctx: RewardContext, rows: list[Row]) -> list[Row]:
            parts = await asyncio.gather(*(ctx.run(r, rows) for r in rewards))
            return write(rows, *parts)

        return score

    local, remote, judge = FakeLocal(), FakeRemote(), FakeJudge()
    model, grid = FakeModel(), Grid(steps=2)
    processor = cast("BaseProcessor", FakeProcessor())
    runner = Runner(model, processor, torch.device("cpu"), seed=0)

    def submit(
        items: list[tuple[int, Row]], sample: Any, score: Any, k: int = 3
    ) -> Pending:
        return runner.submit(
            items, sample, score, prepare=dict, k=k, epoch=0, variant=None, sampler=grid
        )

    # (a) sampled() returns once every group left sample(), remote scores still in
    # flight; wait() returns rows in submission order with rewards written.
    # (c) the remote scorer saw CPU snapshots of only its field. (g) one profile
    # entry per remote request.
    pending = submit(items(2), sample, scored_by(local, remote))
    pending.sampled()
    assert [g.state for g in pending._groups] == [_SAMPLED, _SAMPLED]
    assert remote._done < 6, remote._done
    groups = pending.wait()
    assert [[row["key"] for row in rows] for rows in groups] == [["p0"] * 3, ["p1"] * 3]
    for rows in groups:
        for row in rows:
            check_decoded(row)
            assert row["reward"].shape == (2,) and row["epoch"] == 0
        assert not torch.equal(rows[0]["clean_latents"], rows[1]["clean_latents"])
    assert remote._done == 6 and pending.profile.count == 6
    assert sum(local._sizes) == 6 and sum(n for _, n in model.forwards) == 12
    print("(a)(c)(g) ok:", local._sizes, model.forwards)

    # (b) local items of several groups share run_batch calls: the first group's
    # slow batch holds the worker while the other groups' items queue up behind it.
    slow_local, slow_remote = FakeLocal(delay=0.1), FakeRemote(delay=0.05)

    async def score_b(ctx: RewardContext, rows: list[Row]) -> list[Row]:
        await ctx.run(slow_remote, rows)
        return write(rows, await ctx.run(slow_local, rows))

    submit(items(3), sample, score_b).wait()
    assert sum(slow_local._sizes) == 9 and max(slow_local._sizes) <= 8
    assert len(slow_local._sizes) < 3, slow_local._sizes
    print("(b) ok:", slow_local._sizes)

    # (d) lookahead: two submissions interleave on one worker.
    a, b = (
        submit(items(2), sample, scored_by(local, remote)),
        submit(items(2, start=2), sample, scored_by(local, remote)),
    )
    a.sampled()
    assert all(g.state >= _SAMPLED for g in a._groups) and len(a._groups) == 2
    assert len(b._groups) >= 1, "B's groups start while A is still sampling"
    b.sampled()
    assert [[r["key"] for r in rows] for rows in a.wait()] == [["p0"] * 3, ["p1"] * 3]
    assert [[r["key"] for r in rows] for rows in b.wait()] == [["p2"] * 3, ["p3"] * 3]
    print("(d) ok")

    # (e) as_completed() yields in completion order.
    async def score_e(ctx: RewardContext, rows: list[Row]) -> list[Row]:
        await asyncio.sleep((2 - ctx.prompt_id) * 0.03)
        return write(rows, await ctx.run(local, rows))

    order = [
        rows[0]["key"] for rows in submit(items(3), sample, score_e).as_completed()
    ]
    assert order == ["p2", "p1", "p0"], order
    print("(e) ok:", order)

    # (h) an anchor sampled under another variant alongside the rollouts; both
    # decoded, forwards grouped by variant.
    async def sample_h(ctx: SampleContext) -> tuple[list[Row], Row]:
        anchor = await ctx.start(
            ctx.seed("anchor", ctx.epoch, ctx.prompt_id), variant="ema"
        )
        rows, _ = await asyncio.gather(ctx.rollouts(), ctx.admit(Grid(steps=1)(anchor)))
        await ctx.decode(anchor)
        return rows, anchor

    async def score_h(ctx: RewardContext, group: tuple[list[Row], Row]) -> list[Row]:
        rows, anchor = group
        return write(rows, await ctx.run(judge, [(row, anchor) for row in rows]))

    model.forwards.clear()
    pending = submit(items(2), sample_h, score_h)
    for rows in pending.wait():
        for row in rows:
            check_decoded(row)
            assert row["reward"].shape == (1,)
    by_variant: dict[str, int] = {}
    for name, n in model.forwards:
        by_variant[name] = by_variant.get(name, 0) + n
    assert by_variant == {"default": 12, "ema": 2}, model.forwards
    assert pending.profile.count == 6
    print("(h) ok:", model.forwards)

    # (i) SampleContext methods raise once sample() returned.
    late: list[str] = []

    async def sample_i(ctx: SampleContext) -> SampleContext:
        return ctx

    async def score_i(ctx: RewardContext, sampled: SampleContext) -> list[Row]:
        for call in (sampled.rollouts(), sampled.start(1)):
            try:
                await call
            except RuntimeError as error:
                late.append(str(error))
        return []

    assert submit(items(1), sample_i, score_i).wait() == [[]]
    assert len(late) == 2 and all("sample() has returned" in m for m in late), late
    print("(i) ok")

    # (f) a failing score() or sampler node fails wait() with that exception.
    async def bad_score(ctx: RewardContext, rows: list[Row]) -> list[Row]:
        raise ValueError("boom")

    def bad_node(row: Row) -> Calls[None]:
        raise RuntimeError("node")
        yield

    async def bad_sample(ctx: SampleContext) -> None:
        await ctx.admit(bad_node(await ctx.start(0)))

    class BadLocal(FakeLocal):
        def run_batch(self, rows: list[Row]) -> list[torch.Tensor]:
            raise ValueError("batch")

    for program, kind, message in (
        ((sample, bad_score), ValueError, "boom"),
        ((bad_sample, bad_score), RuntimeError, "node"),
        ((sample, scored_by(BadLocal())), ValueError, "batch"),
    ):
        other = Runner(FakeModel(), processor, torch.device("cpu"), seed=0)
        try:
            other.submit(
                items(1),
                *program,
                prepare=dict,
                k=1,
                epoch=0,
                variant=None,
                sampler=grid,
            ).wait()
        except kind as error:
            assert str(error) == message, error
        else:
            raise AssertionError(f"wait() must raise {message!r}")
        finally:
            other.close()
    print("(f) ok")

    runner.close()
    print("[green]runner checks passed[/green]")
