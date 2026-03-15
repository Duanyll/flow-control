import asyncio
import concurrent.futures
import threading
from collections.abc import Callable, Generator
from typing import Annotated, Any

import torch
from pydantic import Discriminator, Tag, TypeAdapter

from .base import BaseReward
from .clip_score import CLIPScoreReward
from .composite import CompositeReward
from .geneval import GenevalReward
from .pickscore import PickScoreReward
from .unified_reward import UnifiedReward

Reward = Annotated[
    Annotated[CLIPScoreReward, Tag("clip_score")]
    | Annotated[PickScoreReward, Tag("pickscore")]
    | Annotated[GenevalReward, Tag("geneval")]
    | Annotated[UnifiedReward, Tag("unified_reward")]
    | Annotated[CompositeReward, Tag("composite")],
    Discriminator("type"),
]

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

    def submit(self, coro: Any) -> concurrent.futures.Future[torch.Tensor]:
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


def _score_blocking(reward: BaseReward, batch: dict[str, Any]) -> torch.Tensor:
    try:
        return reward.score(batch)
    except NotImplementedError:
        return asyncio.run(reward.async_score(batch))


def execute_reward[TBatch: dict, TTag, TResult](
    reward: BaseReward,
    submitter: Generator[tuple[TBatch, TTag]],
    handler: Callable[[TTag, torch.Tensor], TResult],
) -> list[TResult]:
    """Score batches from *submitter* and pass each reward to *handler*.

    When the reward supports rollout overlap (i.e. remote rewards), scoring is
    launched asynchronously so that the generator can continue producing batches
    while earlier rewards are still in flight.  Otherwise scoring is synchronous.
    """
    overlap = reward.supports_rollout_overlap()
    reward_loop = _RewardLoopThread() if overlap else None

    pending: list[tuple[TTag, concurrent.futures.Future[torch.Tensor] | None]] = []
    results: list[TResult] = []

    try:
        for batch, tag in submitter:
            if reward_loop is not None:
                async_batch = reward.prepare_batch_for_async(batch)
                future = reward_loop.submit(reward.async_score(async_batch))
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


__all__ = ["Reward", "parse_reward", "execute_reward"]
