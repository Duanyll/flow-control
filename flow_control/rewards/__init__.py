import asyncio
import concurrent.futures
import threading
from collections.abc import Callable, Generator
from typing import Annotated, Any

import torch
from pydantic import Discriminator, Tag, TypeAdapter

from .aesthetic import AestheticReward
from .base import BaseReward, RewardResult
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
from .unified_reward import UnifiedReward

Reward = Annotated[
    Annotated[CLIPScoreReward, Tag("clip_score")]
    | Annotated[PickScoreReward, Tag("pickscore")]
    | Annotated[GenevalReward, Tag("geneval")]
    | Annotated[UnifiedReward, Tag("unified_reward")]
    | Annotated[CompositeReward, Tag("composite")]
    | Annotated[PairwiseReward, Tag("pairwise")]
    | Annotated[AestheticReward, Tag("aesthetic")]
    | Annotated[CLIPImageSimilarityReward, Tag("clip_image_similarity")]
    | Annotated[HPSv2Reward, Tag("hpsv2")]
    | Annotated[OcrReward, Tag("ocr")]
    | Annotated[ImageRewardReward, Tag("image_reward")],
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


def execute_reward[TBatch: dict, TTag, TResult](
    reward: BaseReward,
    submitter: Generator[tuple[TBatch, TTag]],
    handler: Callable[[TTag, RewardResult], TResult],
) -> list[TResult]:
    """Score batches from *submitter* and pass each reward to *handler*.

    When the reward supports rollout overlap (i.e. remote rewards), scoring is
    launched asynchronously so that the generator can continue producing batches
    while earlier rewards are still in flight.  Otherwise scoring is synchronous.
    """
    overlap = reward.supports_rollout_overlap()
    reward_loop = _RewardLoopThread() if overlap else None

    pending: list[tuple[TTag, concurrent.futures.Future[RewardResult] | None]] = []
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
    "AffineNormalize",
    "ClampNormalize",
    "IdentityNormalize",
    "Normalize",
    "PairwiseReward",
    "Reward",
    "RewardResult",
    "SigmoidNormalize",
    "execute_pairwise_reward",
    "execute_reward",
    "parse_normalize",
    "parse_reward",
]
