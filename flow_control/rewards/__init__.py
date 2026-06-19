import asyncio
import concurrent.futures
import threading
import time
from collections.abc import Callable, Generator
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

Reward = Annotated[BaseReward, RegistryUnion(reward_registry, "type")]

_reward_ta = TypeAdapter(Reward)


def parse_reward(conf: dict[str, Any]) -> BaseReward:
    """Parse a reward config dict into the appropriate reward instance."""
    return _reward_ta.validate_python(conf)


class _RewardLoopThread:
    """Run async reward requests in a dedicated event loop thread."""

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


def _score_blocking(reward: BaseReward, batch: dict[str, Any]) -> RewardResult:
    try:
        return reward.score(batch)
    except NotImplementedError:
        return asyncio.run(reward.async_score(batch))


@dataclass
class RewardProfile:
    """Pipeline-level timing for the async (overlap) reward path.

    Records when each reward request is submitted and when it completes, so we
    can tell whether the reward backend (e.g. a vLLM judge) keeps up with rollout
    production.  Submissions happen on the main thread; completions are recorded
    by a ``Future`` done-callback on the reward-loop thread.  Each index is
    written exactly once, so no lock is needed.

    Only populated on the overlap path; left empty (``count == 0``) otherwise.
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


def execute_reward[TBatch: dict, TTag, TResult](
    reward: BaseReward,
    submitter: Generator[tuple[TBatch, TTag]],
    handler: Callable[[TTag, RewardResult], TResult],
    profile: RewardProfile | None = None,
) -> list[TResult]:
    """Score batches from *submitter* and pass each reward to *handler*.

    When the reward supports rollout overlap (i.e. remote rewards), scoring is
    launched asynchronously so that the generator can continue producing batches
    while earlier rewards are still in flight.  Otherwise scoring is synchronous.

    When *profile* is given and the overlap path is taken, submit/complete
    timestamps are recorded into it for throughput analysis.
    """
    overlap = reward.supports_rollout_overlap()
    reward_loop = _RewardLoopThread() if overlap else None

    pending: list[tuple[TTag, concurrent.futures.Future[RewardResult] | None]] = []
    results: list[TResult] = []

    try:
        for batch, tag in submitter:
            if reward_loop is not None:
                async_batch = reward.prepare_batch_for_async(batch)
                idx = profile.on_submit() if profile is not None else None
                future = reward_loop.submit(reward.async_score(async_batch))
                if profile is not None and idx is not None:
                    future.add_done_callback(lambda _f, i=idx: profile.on_done(i))
                pending.append((tag, future))
            else:
                reward_value = _score_blocking(reward, batch)
                results.append(handler(tag, reward_value))

        # Collect async results in submission order
        for tag, future in pending:
            assert future is not None
            reward_value = future.result()
            results.append(handler(tag, reward_value))
    finally:
        if reward_loop is not None:
            reward_loop.close()

    return results


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


def execute_pairwise_reward[TTag, TResult](
    reward: BaseReward,
    submitter: Generator[tuple[dict[str, Any], TTag]],
    handler: Callable[[TTag, RewardResult], TResult],
    num_rollouts_per_prompt: int,
) -> list[TResult]:
    """Score batches using the pairwise execution path.

    Expects rollouts to arrive contiguously per prompt (K at a time from the
    ``keep_prompt_local`` sampler).

    For a CompositeReward with mixed children, non-pairwise children are scored
    independently while pairwise children go through the pairwise comparison
    path.  Results are concatenated to produce the full ``[C]`` score vector.

    Args:
        reward: The reward (may be PairwiseReward, CompositeReward with
            pairwise children, or a regular reward).
        submitter: Yields ``(batch, tag)`` pairs, K per prompt contiguously.
        handler: Called with ``(tag, reward_tensor)`` for each sample.
        num_rollouts_per_prompt: K value for grouping.

    Returns:
        List of handler results.
    """
    reward_loop = _RewardLoopThread()
    results: list[TResult] = []

    try:
        prompt_group: list[tuple[dict[str, Any], TTag]] = []

        def _flush_prompt_group() -> None:
            """Process a completed prompt group of K rollouts."""
            batches = [b for b, _ in prompt_group]
            tags = [t for _, t in prompt_group]

            if isinstance(reward, PairwiseReward):
                scores = _score_pairwise_group(reward, reward_loop, batches)
            elif isinstance(reward, CompositeReward):
                scores = _score_composite_pairwise_group(reward, reward_loop, batches)
            else:
                # No pairwise children, score independently
                scores = []
                for batch in batches:
                    scores.append(_score_blocking(reward, batch))

            for tag, score in zip(tags, scores, strict=True):
                results.append(handler(tag, score))

        for batch, tag in submitter:
            async_batch = reward.prepare_batch_for_async(batch)
            prompt_group.append((async_batch, tag))

            if len(prompt_group) == num_rollouts_per_prompt:
                _flush_prompt_group()
                prompt_group = []

        # Handle any remaining partial group
        if prompt_group:
            _flush_prompt_group()
    finally:
        reward_loop.close()

    return results


def _score_pairwise_group(
    reward: PairwiseReward,
    loop: _RewardLoopThread,
    batches: list[dict[str, Any]],
) -> list[RewardResult]:
    """Build win matrix for a prompt group and aggregate."""
    K = len(batches)
    # Launch pairwise comparisons incrementally
    futures: dict[tuple[int, int], concurrent.futures.Future[Any]] = {}
    for i in range(K):
        for j in range(i):
            futures[(i, j)] = loop.submit(
                reward.async_score_pair(batches[i], batches[j])
            )

    # Build win matrix
    win_matrix = torch.full((K, K), 0.5)
    for (i, j), fut in futures.items():
        score = fut.result()
        win_matrix[i, j] = score
        win_matrix[j, i] = 1.0 - score

    # Aggregate to per-sample raw scores [K] and normalize through the reward.
    aggregated = reward.aggregate(win_matrix)
    result = reward._make_result(aggregated.unsqueeze(-1))  # noqa: SLF001
    return [result.row(i) for i in range(K)]


def _score_composite_pairwise_group(
    reward: CompositeReward,
    loop: _RewardLoopThread,
    batches: list[dict[str, Any]],
) -> list[RewardResult]:
    """Score a composite reward with mixed pairwise and non-pairwise children."""
    K = len(batches)
    # Per-child scores: list of K results per child
    child_scores: list[list[RewardResult]] = []

    for child in reward._reward_instances:  # noqa: SLF001
        if isinstance(child, PairwiseReward):
            child_scores.append(_score_pairwise_group(child, loop, batches))
        else:
            # Score each batch independently
            per_batch: list[RewardResult] = []
            if child.supports_rollout_overlap():
                futs = [loop.submit(child.async_score(b)) for b in batches]
                for fut in futs:
                    per_batch.append(fut.result())
            else:
                for b in batches:
                    per_batch.append(_score_blocking(child, b))
            child_scores.append(per_batch)

    # Concatenate per-child results for each sample.
    result: list[RewardResult] = []
    for k in range(K):
        parts = [child_scores[c][k] for c in range(len(child_scores))]
        result.append(reward._combine_results(parts))  # noqa: SLF001

    return result


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
    "PickScoreReward",
    "RationalRewardsEditReward",
    "RationalRewardsT2IReward",
    "Reward",
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

    print("[green]reward profile self-test passed[/green]")
