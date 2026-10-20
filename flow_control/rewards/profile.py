import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RewardProfile:
    """Pipeline-level timing of remote reward requests.

    Records when each request is issued and when it completes, so we can tell
    whether the reward backend (e.g. a vLLM judge) keeps up with rollout
    production. Both stamps are taken on the reward-loop thread: the issue stamp
    when the request is issued (so ``produce_span_s`` / ``overlap_ratio`` measure
    issue times, not decode times), the completion stamp as the request's
    coroutine returns, before its result is published to the waiter. Each index
    is written exactly once, so no lock is needed.

    One entry per remote request (each order of a judged pair, each row of a
    remote scorer), so ``count`` and the throughput are requests, not rows; a
    local reward adds none.
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

    print("[green]rewards.profile self-test passed[/green]")
