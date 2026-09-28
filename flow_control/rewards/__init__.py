import asyncio
import concurrent.futures
import threading
import time
from collections.abc import Callable, Coroutine, Generator
from dataclasses import dataclass, field
from typing import Annotated, Any

import torch
from pydantic import TypeAdapter

from flow_control.utils.registry import RegistryUnion

from .aesthetic import AestheticReward
from .base import BaseReward, RewardResult, reward_registry
from .clip_image_similarity import CLIPImageSimilarityReward
from .clip_score import CLIPScoreReward
from .composite import CompositeReward
from .geneval import GenevalReward
from .hpsv2 import HPSv2Reward
from .image_reward import ImageRewardReward
from .normalize import (
    AffineNormalize,
    ClampNormalize,
    IdentityNormalize,
    Normalize,
    SigmoidNormalize,
    parse_normalize,
)
from .ocr import OcrReward
from .pairwise import PairwiseReward
from .pickscore import PickScoreReward
from .rational_rewards import RationalRewardsEditReward, RationalRewardsT2IReward
from .unified_reward import UnifiedReward

# A bare list of reward configs is shorthand for a composite reward:
# ``"reward": [{...}, {...}]`` == ``{"type": "composite", "rewards": [...]}``.
Reward = Annotated[
    BaseReward,
    RegistryUnion(reward_registry, "type", list_as=("composite", "rewards")),
]

_reward_ta = TypeAdapter(Reward)


def parse_reward(conf: dict[str, Any] | list[Any] | str) -> BaseReward:
    """Parse a reward config (dict, bare tag, or composite-list shorthand)."""
    return _reward_ta.validate_python(conf)


class RewardLoopThread:
    """Run async reward requests in a dedicated event loop thread.

    ``execute_reward`` opens one per call unless handed a caller-owned instance;
    a trainer that keeps reward futures in flight across epochs owns one loop for
    the whole run and closes it at the end.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="reward-loop",
            daemon=True,
        )
        self._started = False

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def submit(self, coro: Any) -> concurrent.futures.Future[Any]:
        if not self._started:
            self._thread.start()
            self._started = True
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def close(self) -> None:
        if not self._started:
            self._loop.close()
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()


def _score_blocking(reward: BaseReward, row: dict[str, Any]) -> RewardResult:
    try:
        return reward.score(row)
    except NotImplementedError:
        return asyncio.run(reward.async_score(row))


@dataclass
class RewardProfile:
    """Pipeline-level timing for the async (overlap) reward path.

    Records when each reward request is submitted and when it completes, so we
    can tell whether the reward backend (e.g. a vLLM judge) keeps up with rollout
    production.  Submissions happen on the main thread; completions are recorded
    on the reward-loop thread as each request's coroutine returns, before its
    result is published to the waiter.  Each index is written exactly once, so
    no lock is needed.

    Only populated on the overlap path and the pairwise path, where every pair
    request (each order) and every per-row request of a remote child is one
    entry, so ``count`` and the throughput are judge requests, not rows; left
    empty (``count == 0``) otherwise.
    """

    _submit_times: list[float] = field(default_factory=list)
    _done_times: list[float] = field(default_factory=list)

    def on_submit(self) -> int:
        """Record a submission timestamp; return its index for ``on_done``."""
        idx = len(self._submit_times)
        self._submit_times.append(time.perf_counter())
        self._done_times.append(0.0)  # placeholder until on_done fires
        return idx

    def on_done(self, idx: int) -> None:
        self._done_times[idx] = time.perf_counter()

    @property
    def count(self) -> int:
        return len(self._submit_times)

    def timestamps(self) -> dict[str, list[float]]:
        """Raw ``perf_counter`` submit/done times (done ``0.0`` = still pending)."""
        return {"submit": list(self._submit_times), "done": list(self._done_times)}

    @staticmethod
    def _max_in_flight(submits: list[float], dones: list[float]) -> int:
        # Sweep +1 on submit / -1 on completion; ties resolve completion first.
        events = sorted(
            [(t, 1) for t in submits] + [(t, -1) for t in dones],
            key=lambda e: (e[0], e[1]),
        )
        cur = mx = 0
        for _, delta in events:
            cur += delta
            mx = max(mx, cur)
        return mx

    def local_payload(self) -> dict[str, Any]:
        """Raw per-rank quantities for cross-rank reduction (no percentiles)."""
        pairs = [
            (s, d)
            for s, d in zip(self._submit_times, self._done_times, strict=True)
            if d > 0.0
        ]
        if not pairs:
            return {
                "latencies": [],
                "produce_span_s": 0.0,
                "reward_span_s": 0.0,
                "tail_wait_s": 0.0,
                "max_in_flight": 0,
                "count": 0,
            }
        submits = [s for s, _ in pairs]
        dones = [d for _, d in pairs]
        return {
            "latencies": [d - s for s, d in pairs],
            "produce_span_s": max(submits) - min(submits),
            "reward_span_s": max(dones) - min(submits),
            "tail_wait_s": max(0.0, max(dones) - max(submits)),
            "max_in_flight": self._max_in_flight(submits, dones),
            "count": len(pairs),
        }


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated percentile (``q`` in [0, 100]) of a sorted list."""
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (q / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def reduce_reward_profiles(payloads: list[dict[str, Any]]) -> dict[str, float]:
    """Reduce gathered per-rank :meth:`RewardProfile.local_payload` dicts.

    Latency *durations* pool across ranks for honest global percentiles; spans
    and ``max_in_flight`` reduce by ``max`` (the slowest rank gates the
    downstream ``all_gather``); ``count`` sums.  Returns flat ``profile/reward/*``
    metrics, or an empty dict when no rank scored anything (nothing to log).
    """
    total_count = sum(int(p.get("count", 0)) for p in payloads)
    if total_count == 0:
        return {}

    latencies = sorted(float(x) for p in payloads for x in p.get("latencies", []))
    max_produce = max(float(p.get("produce_span_s", 0.0)) for p in payloads)
    max_reward = max(float(p.get("reward_span_s", 0.0)) for p in payloads)
    max_tail = max(float(p.get("tail_wait_s", 0.0)) for p in payloads)
    max_in_flight = max(int(p.get("max_in_flight", 0)) for p in payloads)

    mean_lat = sum(latencies) / len(latencies) if latencies else 0.0
    return {
        "profile/reward/count": float(total_count),
        "profile/reward/produce_span_s": max_produce,
        "profile/reward/reward_span_s": max_reward,
        "profile/reward/tail_wait_s": max_tail,
        "profile/reward/overlap_ratio": (max_produce / max_reward)
        if max_reward > 0
        else 0.0,
        "profile/reward/throughput_per_s": (total_count / max_reward)
        if max_reward > 0
        else 0.0,
        "profile/reward/max_in_flight": float(max_in_flight),
        "profile/reward/latency_mean_s": mean_lat,
        "profile/reward/latency_p50_s": _percentile(latencies, 50.0),
        "profile/reward/latency_p95_s": _percentile(latencies, 95.0),
        "profile/reward/latency_max_s": _percentile(latencies, 100.0),
    }


class PendingRewards[TTag]:
    """Reward requests launched by :func:`submit_reward`, waiting to be handled.

    On the overlap path each entry is a ``Future`` still running on the reward
    loop; on the blocking path scoring already happened during submission and
    the future is complete.  Either way :meth:`wait` hands results to the
    handler in submission order.  The loop stays open: it belongs to the caller.
    """

    def __init__(
        self, futures: list[tuple[TTag, concurrent.futures.Future[RewardResult]]]
    ) -> None:
        self._futures = futures

    def wait[TResult](
        self, handler: Callable[[TTag, RewardResult], TResult]
    ) -> list[TResult]:
        """Block until every result is in; call *handler* in submission order."""
        return [handler(tag, future.result()) for tag, future in self._futures]


async def _record_done[T](
    coro: Coroutine[Any, Any, T], profile: RewardProfile, idx: int
) -> T:
    """Stamp completion on the loop thread before the result reaches the waiter.

    A ``Future`` done-callback would fire *after* ``set_result`` has woken
    ``future.result()`` on the main thread, so a profile read right after
    :meth:`PendingRewards.wait` could still miss the last rows.
    """
    try:
        return await coro
    finally:
        profile.on_done(idx)


def submit_reward[TRow: dict, TTag](
    reward: BaseReward,
    submitter: Generator[tuple[TRow, TTag]],
    loop: RewardLoopThread,
    profile: RewardProfile | None = None,
) -> PendingRewards[TTag]:
    """Drive *submitter* to completion, launching one reward request per row.

    A reward that supports rollout overlap (i.e. remote rewards) has its
    ``async_score`` submitted to *loop*, so the generator keeps producing rows
    while earlier rewards are in flight, and *profile* records the
    submit/complete timestamps.  A local reward is scored blocking as each row
    is yielded and the profile stays empty.  The returned
    :class:`PendingRewards` collects the results; the loop is left open.
    """
    overlap = reward.supports_rollout_overlap()
    futures: list[tuple[TTag, concurrent.futures.Future[RewardResult]]] = []

    for row, tag in submitter:
        if overlap:
            coro = reward.async_score(reward.prepare_row_for_async(row))
            if profile is not None:
                coro = _record_done(coro, profile, profile.on_submit())
            future = loop.submit(coro)
        else:
            future = concurrent.futures.Future()
            future.set_result(_score_blocking(reward, row))
        futures.append((tag, future))

    return PendingRewards(futures)


def execute_reward[TRow: dict, TTag, TResult](
    reward: BaseReward,
    submitter: Generator[tuple[TRow, TTag]],
    handler: Callable[[TTag, RewardResult], TResult],
    profile: RewardProfile | None = None,
    loop: RewardLoopThread | None = None,
) -> list[TResult]:
    """Score rows from *submitter* and pass each reward to *handler*.

    A local reward is scored and handled row by row, so a streaming consumer
    (the inference report) never holds more than one row.  For a reward that
    supports rollout overlap this is :func:`submit_reward` followed by
    :meth:`PendingRewards.wait`; when *loop* is None a :class:`RewardLoopThread`
    is opened for this call and closed afterwards, a caller-provided one is left
    open.
    """
    if not reward.supports_rollout_overlap():
        return [handler(tag, _score_blocking(reward, row)) for row, tag in submitter]

    owned_loop = None
    if loop is None:
        loop = owned_loop = RewardLoopThread()
    try:
        return submit_reward(reward, submitter, loop, profile).wait(handler)
    finally:
        if owned_loop is not None:
            owned_loop.close()


def _has_pairwise_child(reward: BaseReward) -> bool:
    """Check if *reward* or any child is a PairwiseReward."""
    if isinstance(reward, PairwiseReward):
        return True
    if isinstance(reward, CompositeReward):
        return any(
            _has_pairwise_child(r)
            for r in reward._reward_instances  # noqa: SLF001
        )
    return False


def _stamped[T](
    coro: Coroutine[Any, Any, T], profile: RewardProfile | None
) -> Coroutine[Any, Any, T]:
    """Give *coro* a profile entry now (main thread) if profiling is on."""
    if profile is None:
        return coro
    return _record_done(coro, profile, profile.on_submit())


async def _ready[T](value: T) -> T:
    return value


def _pairwise_group(
    reward: PairwiseReward,
    rows: list[dict[str, Any]],
    profile: RewardProfile | None,
    gated: list[bool] | None = None,
) -> Coroutine[Any, Any, list[RewardResult]]:
    """One prompt group's per-row results from every pair comparison.

    The pair requests are created (and stamped) here on the caller's thread;
    the returned coroutine runs them concurrently on the reward loop, fills
    the ``[K, K]`` win matrix and aggregates it through the reward. Rows
    flagged in *gated* (see ``PairwiseReward.gate_component``) are asked no
    comparison: they lose every pair and are left out of the others' means.
    """
    K = len(rows)
    active = [not (gated and gated[i]) for i in range(K)]
    pairs = [(i, j) for i in range(K) for j in range(i) if active[i] and active[j]]
    orders = pairs + ([(j, i) for i, j in pairs] if reward.swap_orders else [])
    calls = [
        _stamped(reward.async_score_pair(rows[a], rows[b]), profile) for a, b in orders
    ]

    async def run() -> list[RewardResult]:
        scores = [float(s) for s in await asyncio.gather(*calls)]
        win_matrix = torch.full((K, K), 0.5)
        for n, (i, j) in enumerate(pairs):
            score = scores[n]
            if reward.swap_orders:
                score = (score + 1.0 - scores[len(pairs) + n]) / 2.0
            win_matrix[i, j] = score
            win_matrix[j, i] = 1.0 - score
        if gated is None:
            aggregated = reward.aggregate(win_matrix)
        else:
            # A rejected row loses every pair (both sides recorded); its
            # column is masked out of the active rows' means.
            active_mask = torch.tensor(active)
            win_matrix[:, ~active_mask] = 1.0
            win_matrix[~active_mask, :] = 0.0
            aggregated = reward.aggregate(win_matrix, active_mask)
        result = reward._make_result(aggregated.unsqueeze(-1))  # noqa: SLF001
        return [result.row(i) for i in range(K)]

    return run()


def _row_requests(
    reward: BaseReward,
    rows: list[dict[str, Any]],
    profile: RewardProfile | None,
) -> Coroutine[Any, Any, list[RewardResult]]:
    """One request per row for a remote (overlap-capable) child."""
    calls = [_stamped(reward.async_score(row), profile) for row in rows]

    async def run() -> list[RewardResult]:
        return list(await asyncio.gather(*calls))

    return run()


class _PairwiseGroupPlan:
    """How one prompt group of *reward* is scored.

    The reward is a :class:`PairwiseReward` or a :class:`CompositeReward` whose
    direct children may be pairwise (need the whole group), remote (one request
    per row) or local (scored blocking on the trainer's device as each row is
    yielded, like :func:`submit_reward` does for a local reward).
    """

    def __init__(self, reward: BaseReward) -> None:
        self.reward = reward
        self.children: list[BaseReward] = (
            list(reward._reward_instances)  # noqa: SLF001
            if isinstance(reward, CompositeReward)
            else [reward]
        )
        for child in self.children:
            if isinstance(child, CompositeReward) and _has_pairwise_child(child):
                raise ValueError(
                    "A pairwise reward must be the reward itself or a direct child "
                    "of the top-level composite reward; a nested composite cannot "
                    "carry one."
                )
        self.local = [
            index
            for index, child in enumerate(self.children)
            if not isinstance(child, PairwiseReward)
            and not child.supports_rollout_overlap()
        ]
        # Pairwise child index -> (local child index, component index) of its gate.
        self.gates: dict[int, tuple[int, int]] = {}
        for index, child in enumerate(self.children):
            if isinstance(child, PairwiseReward) and child.gate_component is not None:
                self.gates[index] = self._resolve_gate(child.gate_component)

    def _resolve_gate(self, label: str) -> tuple[int, int]:
        """Find the local sibling component that ``gate_component`` names."""
        if not isinstance(self.reward, CompositeReward):
            raise ValueError(
                f"gate_component {label!r} needs a local sibling reward: put the "
                "pairwise reward in a composite next to the reward that scores "
                "the gate."
            )
        found: list[tuple[int, int]] = []
        elsewhere: list[str] = []
        for index, child in enumerate(self.children):
            hits = [
                c
                for c, name in enumerate(child.component_labels)
                if label in (name, f"{child.type}/{name}")
            ]
            if not hits:
                continue
            if index in self.local:
                found.extend((index, c) for c in hits)
            else:
                elsewhere.append(child.type)
        if len(found) == 1:
            return found[0]
        if found:
            raise ValueError(
                f"gate_component {label!r} matches {len(found)} components of the "
                "local sibling rewards; write it as '<type>/<label>'."
            )
        if elsewhere:
            raise ValueError(
                f"gate_component {label!r} belongs to {', '.join(elsewhere)}, which "
                "is scored remotely or pairwise, after the group's comparisons are "
                "formed; only a local sibling (scored as the row is decoded) can "
                "gate."
            )
        available = [
            f"{self.children[index].type}/{name}"
            for index in self.local
            for name in self.children[index].component_labels
        ]
        raise ValueError(
            f"gate_component {label!r} is not a component of any local sibling "
            f"reward; available: {available or 'none'}."
        )

    def score_local(self, row: dict[str, Any]) -> dict[int, RewardResult]:
        """Local children score the row now, from the device tensors."""
        return {
            index: _score_blocking(self.children[index], row) for index in self.local
        }

    def group(
        self,
        rows: list[dict[str, Any]],
        local_results: list[dict[int, RewardResult]],
        profile: RewardProfile | None,
    ) -> Coroutine[Any, Any, list[RewardResult]]:
        parts: list[Coroutine[Any, Any, list[RewardResult]]] = []
        for index, child in enumerate(self.children):
            if isinstance(child, PairwiseReward):
                gated = None
                if index in self.gates:
                    local_index, component = self.gates[index]
                    gated = [
                        float(result[local_index].raw[0, component]) > 0.5
                        for result in local_results
                    ]
                parts.append(_pairwise_group(child, rows, profile, gated))
            elif index in self.local:
                parts.append(_ready([result[index] for result in local_results]))
            else:
                parts.append(_row_requests(child, rows, profile))
        return self._combine(parts)

    async def _combine(
        self, parts: list[Coroutine[Any, Any, list[RewardResult]]]
    ) -> list[RewardResult]:
        per_child = await asyncio.gather(*parts)
        if not isinstance(self.reward, CompositeReward):
            return per_child[0]
        return [
            self.reward._combine_results([child[k] for child in per_child])  # noqa: SLF001
            for k in range(len(per_child[0]))
        ]


async def _resolve_group(
    futures: list[concurrent.futures.Future[RewardResult]],
    coro: Coroutine[Any, Any, list[RewardResult]],
) -> None:
    """Publish a group's results (or its failure) to its rows' futures, so a
    waiter never blocks on a group that died."""
    try:
        results = await coro
    except BaseException as exc:  # noqa: BLE001 - re-raised on the waiter's thread
        for future in futures:
            future.set_exception(exc)
        return
    for future, result in zip(futures, results, strict=True):
        future.set_result(result)


def submit_pairwise_reward[TTag](
    reward: BaseReward,
    submitter: Generator[tuple[dict[str, Any], TTag]],
    loop: RewardLoopThread,
    num_rollouts_per_prompt: int,
    profile: RewardProfile | None = None,
) -> PendingRewards[TTag]:
    """Drive *submitter*, scoring each prompt group as soon as its K rows are in.

    Rows are grouped by ``row["key"]`` and may arrive in any order (a streaming
    sampler finishes prompts out of order), but all K rollouts of a prompt must
    come through this submitter (i.e. stay on one rank).  The moment a group is
    complete its pair requests (both orders when ``swap_orders``) and the
    per-row requests of any remote child go to *loop* together; the generator
    keeps producing rows meanwhile, so sampling and the judge overlap, and the
    returned :class:`PendingRewards` can be waited on any time later (a batch
    sampled ahead under ``rollout_lookahead``).  Local children are scored
    blocking as their row is yielded, like :func:`submit_reward`; a pairwise
    child's ``gate_component`` reads one of their components to drop rejected
    rows from the comparisons before the requests are formed.  *profile* gets
    one entry per request, so its count and throughput are judge requests, not
    rows.  Results reach the waiter's handler in submission order.
    """
    plan = _PairwiseGroupPlan(reward)
    futures: list[tuple[TTag, concurrent.futures.Future[RewardResult]]] = []
    groups: dict[
        str,
        list[tuple[dict[str, Any], dict[int, RewardResult], concurrent.futures.Future]],
    ] = {}

    for row, tag in submitter:
        key = row.get("key")
        if not isinstance(key, str):
            raise ValueError("Pairwise rewards require a string key for each prompt.")
        future: concurrent.futures.Future[RewardResult] = concurrent.futures.Future()
        futures.append((tag, future))
        group = groups.setdefault(key, [])
        group.append((reward.prepare_row_for_async(row), plan.score_local(row), future))
        if len(group) == num_rollouts_per_prompt:
            del groups[key]
            rows = [entry[0] for entry in group]
            local_results = [entry[1] for entry in group]
            row_futures = [entry[2] for entry in group]
            loop.submit(
                _resolve_group(row_futures, plan.group(rows, local_results, profile))
            )

    if groups:
        sizes = {key: len(group) for key, group in groups.items()}
        raise ValueError(
            f"Incomplete pairwise prompt groups: {sizes}; expected "
            f"{num_rollouts_per_prompt} rollouts per prompt on this rank."
        )
    return PendingRewards(futures)


def execute_pairwise_reward[TTag, TResult](
    reward: BaseReward,
    submitter: Generator[tuple[dict[str, Any], TTag]],
    handler: Callable[[TTag, RewardResult], TResult],
    num_rollouts_per_prompt: int,
    profile: RewardProfile | None = None,
    loop: RewardLoopThread | None = None,
) -> list[TResult]:
    """:func:`submit_pairwise_reward` followed by :meth:`PendingRewards.wait`;
    opens a :class:`RewardLoopThread` for the call when *loop* is None."""
    owned_loop = None
    if loop is None:
        loop = owned_loop = RewardLoopThread()
    try:
        return submit_pairwise_reward(
            reward, submitter, loop, num_rollouts_per_prompt, profile
        ).wait(handler)
    finally:
        if owned_loop is not None:
            owned_loop.close()


__all__ = [
    "AestheticReward",
    "AffineNormalize",
    "BaseReward",
    "CLIPImageSimilarityReward",
    "CLIPScoreReward",
    "ClampNormalize",
    "CompositeReward",
    "GenevalReward",
    "HPSv2Reward",
    "IdentityNormalize",
    "ImageRewardReward",
    "Normalize",
    "OcrReward",
    "PairwiseReward",
    "PendingRewards",
    "PickScoreReward",
    "RationalRewardsEditReward",
    "RationalRewardsT2IReward",
    "Reward",
    "RewardLoopThread",
    "RewardProfile",
    "RewardResult",
    "SigmoidNormalize",
    "UnifiedReward",
    "execute_pairwise_reward",
    "execute_reward",
    "parse_normalize",
    "parse_reward",
    "reduce_reward_profiles",
    "reward_registry",
    "submit_pairwise_reward",
    "submit_reward",
]


if __name__ == "__main__":
    from rich import print

    # (a) RewardProfile.local_payload from synthetic submit/done timestamps.
    # Three requests: submitted at 0/1/2, completing at 5/6/10. The last rollout
    # is submitted at t=2 but the last reward finishes at t=10 -> tail_wait=8.
    prof = RewardProfile()
    prof._submit_times = [0.0, 1.0, 2.0]
    prof._done_times = [5.0, 6.0, 10.0]
    payload = prof.local_payload()
    print("local_payload:", payload)
    assert payload["count"] == 3, payload
    assert payload["produce_span_s"] == 2.0, payload
    assert payload["reward_span_s"] == 10.0, payload
    assert payload["tail_wait_s"] == 8.0, payload
    # At t=2 all three are in flight (none done before t=5).
    assert payload["max_in_flight"] == 3, payload
    assert payload["latencies"] == [5.0, 5.0, 8.0], payload

    # Incomplete entries (placeholder 0.0) are dropped.
    prof2 = RewardProfile()
    prof2._submit_times = [0.0, 1.0]
    prof2._done_times = [3.0, 0.0]
    assert prof2.local_payload()["count"] == 1, prof2.local_payload()

    # (b) Cross-rank reduction must pool latencies, NOT average per-rank p95.
    rank0 = {
        "latencies": [1.0, 1.0, 1.0, 1.0, 100.0],
        "produce_span_s": 2.0,
        "reward_span_s": 10.0,
        "tail_wait_s": 8.0,
        "max_in_flight": 3,
        "count": 5,
    }
    rank1 = {
        "latencies": [2.0, 2.0, 2.0, 2.0, 2.0],
        "produce_span_s": 3.0,
        "reward_span_s": 4.0,
        "tail_wait_s": 1.0,
        "max_in_flight": 2,
        "count": 5,
    }
    reduced = reduce_reward_profiles([rank0, rank1])
    print("reduced:", reduced)
    assert reduced["profile/reward/count"] == 10.0, reduced
    assert reduced["profile/reward/tail_wait_s"] == 8.0, reduced  # max
    assert reduced["profile/reward/reward_span_s"] == 10.0, reduced  # max
    assert reduced["profile/reward/max_in_flight"] == 3.0, reduced  # max
    pooled_p95 = reduced["profile/reward/latency_p95_s"]
    per_rank_p95_avg = (_percentile(sorted(rank0["latencies"]), 95.0) + 2.0) / 2.0
    assert abs(pooled_p95 - per_rank_p95_avg) > 1e-6, (pooled_p95, per_rank_p95_avg)
    assert reduce_reward_profiles([]) == {}
    assert reduce_reward_profiles([{"count": 0, "latencies": []}]) == {}

    # (c) submit/wait split. A fake reward whose async path finishes rows in
    # REVERSE submission order (later rows sleep less) must still reach the
    # handler in submission order, on the overlap path and the blocking one.
    class _FakeReward(BaseReward):
        type: str = "fake"
        overlap: bool = True

        @property
        def _row_fields(self) -> set[str]:
            return {"value"}

        def _load_model(self, device: torch.device) -> None:
            pass

        def _score(self, row: dict[str, Any]) -> torch.Tensor:
            return torch.tensor([float(row["value"])])

        async def _async_score(self, row: dict[str, Any]) -> torch.Tensor:
            await asyncio.sleep(0.002 * (8 - row["value"] % 8))
            return self._score(row)

        def supports_rollout_overlap(self) -> bool:
            return self.overlap

    def rows(start: int, count: int) -> Generator[tuple[dict[str, Any], int]]:
        for value in range(start, start + count):
            yield {"value": value, "dropped": torch.zeros(1)}, value

    def handler(tag: int, result: RewardResult) -> int:
        assert result.raw.item() == float(tag), (tag, result.raw)
        return tag

    for overlap in (True, False):
        prof3 = RewardProfile()
        got = execute_reward(_FakeReward(overlap=overlap), rows(0, 8), handler, prof3)
        assert got == list(range(8)), (overlap, got)
        assert prof3.count == (8 if overlap else 0), (overlap, prof3.count)
        # Every completion is stamped by the time wait() returns.
        assert prof3.local_payload()["count"] == prof3.count, prof3.timestamps()

    # A local reward streams: each row is handled before the next one is drawn
    # (the inference report writes rows as they come instead of buffering).
    order: list[str] = []

    def traced_rows() -> Generator[tuple[dict[str, Any], int]]:
        for value in range(3):
            order.append(f"yield{value}")
            yield {"value": value}, value

    def traced_handler(tag: int, result: RewardResult) -> int:
        order.append(f"handle{tag}")
        return tag

    execute_reward(_FakeReward(overlap=False), traced_rows(), traced_handler)
    assert order == [
        "yield0",
        "handle0",
        "yield1",
        "handle1",
        "yield2",
        "handle2",
    ], order

    # A persistent loop survives interleaved rounds: submit A, submit B, wait A,
    # wait B; then execute_reward(loop=...) leaves it open for one more round.
    loop = RewardLoopThread()
    prof4 = RewardProfile()
    try:
        reward = _FakeReward()
        pending_a = submit_reward(reward, rows(0, 8), loop, prof4)
        pending_b = submit_reward(reward, rows(100, 4), loop, prof4)
        assert prof4.count == 12, prof4.count
        assert pending_a.wait(handler) == list(range(8))
        assert pending_b.wait(handler) == list(range(100, 104))
        assert prof4.local_payload()["count"] == 12, prof4.local_payload()
        assert execute_reward(reward, rows(200, 3), handler, loop=loop) == [
            200,
            201,
            202,
        ]
        assert submit_reward(reward, rows(300, 2), loop).wait(handler) == [300, 301]
        # A local reward on the persistent loop scores blocking during submit.
        pending_local = submit_reward(_FakeReward(overlap=False), rows(0, 3), loop)
        assert pending_local.wait(handler) == [0, 1, 2]
    finally:
        loop.close()

    # (d) Pairwise path. Two prompts' rows arrive interleaved; each group is
    # submitted the moment its K rows are in, every pair is asked in both
    # orders under swap_orders, results come back in submission order, and the
    # profile counts requests (not rows).
    from pydantic import PrivateAttr

    class _FakePairwise(PairwiseReward):
        """Prefers the larger ``value``; ``bias`` favours whichever row is A."""

        bias: float = 0.0
        _calls: list[tuple[int, int]] = PrivateAttr(default_factory=list)

        @property
        def _row_fields(self) -> set[str]:
            return {"value"}

        async def async_score_pair(self, row_a, row_b) -> float:
            a, b = int(row_a["value"]), int(row_b["value"])
            self._calls.append((a, b))
            await asyncio.sleep(0.001 * (a % 3))
            return min(1.0, max(0.0, float(a > b) + self.bias))

    def interleaved(k: int) -> Generator[tuple[dict[str, Any], int]]:
        # Group "A" holds values 0..k-1, group "B" values 10..10+k-1.
        for i in range(k):
            yield {"key": "A", "value": i, "dropped": torch.zeros(1)}, i
            yield {"key": "B", "value": 10 + i}, 10 + i

    def win_rate(tag: int, result: RewardResult) -> tuple[int, float]:
        return tag, round(result.raw.item(), 4)

    # Full round robin, perfect judge: rank r of K -> (r + 0.5) / K.
    values = (0, 1, 2, 3, 10, 11, 12, 13)
    expected = {v: round((v % 10 + 0.5) / 4, 4) for v in values}
    prof5 = RewardProfile()
    pairwise = _FakePairwise()
    got_pairs = execute_pairwise_reward(pairwise, interleaved(4), win_rate, 4, prof5)
    assert [tag for tag, _ in got_pairs] == [0, 10, 1, 11, 2, 12, 3, 13], got_pairs
    assert dict(got_pairs) == expected, got_pairs
    assert len(pairwise._calls) == 2 * 4 * 3, len(pairwise._calls)  # both orders
    assert prof5.count == len(pairwise._calls), prof5.count
    assert prof5.local_payload()["count"] == prof5.count, prof5.timestamps()
    # Swapping cancels a position bias; a single order does not.
    biased = _FakePairwise(bias=0.2)
    swapped = dict(execute_pairwise_reward(biased, interleaved(4), win_rate, 4))
    assert swapped == {
        v: round((0.9 * (v % 10) + 0.1 * (3 - v % 10) + 0.5) / 4, 4) for v in values
    }, swapped
    single = _FakePairwise(bias=0.2, swap_orders=False)
    unswapped = dict(execute_pairwise_reward(single, interleaved(4), win_rate, 4))
    assert len(single._calls) == 4 * 3, len(single._calls)
    assert unswapped != swapped, unswapped

    # Composite: pairwise child + remote child + local child. The local child is
    # scored while its row is yielded; the others when the group completes.
    composite = CompositeReward(
        rewards=[
            _FakePairwise(weight=0.5),
            _FakeReward(weight=0.25),
            _FakeReward(weight=0.25, overlap=False),
        ]
    )
    prof6 = RewardProfile()

    def components(tag: int, result: RewardResult) -> tuple[int, list[float]]:
        return tag, [round(x, 4) for x in result.raw.squeeze(0).tolist()]

    got_composite = dict(
        execute_pairwise_reward(composite, interleaved(4), components, 4, prof6)
    )
    assert got_composite == {v: [expected[v], float(v), float(v)] for v in values}, (
        got_composite
    )
    assert prof6.count == 2 * (12 + 4), prof6.count  # pair requests + remote rows

    # A local sibling's component gates the pairwise child: a rejected row is
    # asked no comparison, scores 0, and drops out of the others' means.
    class _FakeGate(_FakeReward):
        """Local; components ``[value, reject]``, reject = 1 for ``reject_values``."""

        overlap: bool = False
        reject_values: list[int] = [2]

        @property
        def component_labels(self) -> list[str]:
            return ["value", "reject"]

        @property
        def component_weights(self) -> list[float]:
            return [1.0, 0.0]

        def _score(self, row: dict[str, Any]) -> torch.Tensor:
            value = int(row["value"])
            return torch.tensor([float(value), float(value % 10 in self.reject_values)])

    def gated_composite(reject_values: list[int]) -> CompositeReward:
        return CompositeReward(
            rewards=[
                _FakePairwise(weight=0.5, gate_component="reject"),
                _FakeGate(weight=0.5, reject_values=reject_values),
            ]
        )

    # Value 2 (and 12) rejected: 3 survivors per group -> 3 pairs x 2 orders.
    prof7 = RewardProfile()
    one_out = gated_composite([2])
    got_gated = dict(
        execute_pairwise_reward(one_out, interleaved(4), components, 4, prof7)
    )
    survivors = {0: 0.1667, 1: 0.5, 3: 0.8333}
    assert got_gated == {
        v: [0.0 if v % 10 == 2 else survivors[v % 10], float(v), float(v % 10 == 2)]
        for v in values
    }, got_gated
    assert prof7.count == 2 * (3 * 2), prof7.count
    judge = one_out._reward_instances[0]
    assert isinstance(judge, _FakePairwise)
    assert all(2 not in (a % 10, b % 10) for a, b in judge._calls), judge._calls
    # One survivor scores 0.5 without a request; none surviving all score 0.
    prof8 = RewardProfile()
    lone = dict(
        execute_pairwise_reward(
            gated_composite([0, 1, 2]), interleaved(4), components, 4, prof8
        )
    )
    assert {v: comps[0] for v, comps in lone.items()} == {
        v: (0.5 if v % 10 == 3 else 0.0) for v in values
    }, lone
    assert prof8.count == 0, prof8.count
    none = dict(
        execute_pairwise_reward(
            gated_composite([0, 1, 2, 3]), interleaved(4), components, 4
        )
    )
    assert all(comps[0] == 0.0 for comps in none.values()), none
    # A gate must name a component of a local sibling.
    for bad, reason in (
        (_FakePairwise(gate_component="reject"), "needs a local sibling"),
        (
            CompositeReward(
                rewards=[_FakePairwise(gate_component="nope"), _FakeGate()]
            ),
            "not a component",
        ),
        (
            CompositeReward(
                rewards=[_FakePairwise(gate_component="fake"), _FakeReward()]
            ),
            "scored remotely",
        ),
        (
            CompositeReward(
                rewards=[
                    _FakePairwise(gate_component="reject"),
                    _FakeGate(),
                    _FakeGate(),
                ]
            ),
            "'<type>/<label>'",
        ),
    ):
        try:
            _PairwiseGroupPlan(bad)
        except ValueError as exc:
            assert reason in str(exc), (reason, exc)
        else:
            raise AssertionError(f"gate {bad!r} must be rejected: {reason}")
    print("[green]gate_component checks passed[/]")

    # Nested pairwise is refused up front; an incomplete group is an error.
    try:
        _PairwiseGroupPlan(CompositeReward(rewards=[composite]))
    except ValueError as exc:
        print(f"[green]nested pairwise rejected:[/] {exc}")
    else:
        raise AssertionError("a nested pairwise child must be rejected")
    try:
        execute_pairwise_reward(_FakePairwise(), interleaved(3), win_rate, 4)
    except ValueError as exc:
        assert "Incomplete" in str(exc), exc
    else:
        raise AssertionError("an incomplete prompt group must raise")

    # A failing pair request surfaces at wait() instead of hanging it.
    class _Broken(_FakePairwise):
        async def async_score_pair(self, row_a, row_b) -> float:
            raise RuntimeError("judge down")

    try:
        execute_pairwise_reward(_Broken(), interleaved(2), win_rate, 2)
    except RuntimeError as exc:
        assert str(exc) == "judge down", exc
    else:
        raise AssertionError("a failed group must raise at wait()")

    # Batches in flight together on the run-long loop (rollout_lookahead): the
    # second batch is submitted before the first is waited on.
    loop = RewardLoopThread()
    try:
        pending_a = submit_pairwise_reward(_FakePairwise(), interleaved(4), loop, 4)
        pending_b = submit_pairwise_reward(_FakePairwise(), interleaved(2), loop, 2)
        assert dict(pending_a.wait(win_rate)) == expected
        assert dict(pending_b.wait(win_rate)) == {
            0: 0.25,
            1: 0.75,
            10: 0.25,
            11: 0.75,
        }
    finally:
        loop.close()

    print("[green]reward self-test passed[/green]")
