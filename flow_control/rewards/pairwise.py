"""Pairwise reward base class.

Compares rollouts from the same prompt pairwise (e.g., preference models),
producing a pairwise win matrix that is aggregated into per-sample scores.
"""

from abc import abstractmethod
from typing import Any, Literal

import torch
from pydantic import ConfigDict

from .base import BaseReward, reward_registry


@reward_registry.register("pairwise")
class PairwiseReward(BaseReward):
    """Base class for pairwise comparison rewards.

    Instead of scoring each sample independently, pairwise rewards compare
    pairs of rollouts from the same prompt.  Subclasses implement
    :meth:`async_score_pair` which returns a preference score in ``[0, 1]``
    (1 means *a* is preferred over *b*).

    The ``_score`` / ``score`` path is not used; pairwise scoring goes through
    :func:`submit_pairwise_reward` (or :func:`execute_pairwise_reward`), which
    calls ``async_score_pair`` for every pair of a prompt group as soon as the
    group's rollouts are in and aggregates the win matrix into per-sample
    scalars.  That makes it a training-time reward only: validation and
    inference score rows one at a time and need a pointwise reward.
    """

    type: Literal["pairwise"] = "pairwise"
    swap_orders: bool = True
    """Ask each pair in both orders and average: ``s = (s_ab + 1 - s_ba) / 2``.
    A judge that must answer A or B has a position bias (Qwen3.8 without
    thinking picked the first candidate 67% of the time in the judge pilot);
    asking both orders cancels it, and a pair whose answer flips with the order
    lands on 0.5, the tie such a judge never says. ``False`` asks once per pair
    and halves the requests, for a judge known to be order-invariant."""
    model_config = ConfigDict(extra="forbid")

    @property
    def _row_fields(self) -> set[str]:
        return {"clean_image", "prompt"}

    def _load_model(self, device: torch.device) -> None:
        pass

    def _score(self, row: dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError(
            "PairwiseReward does not support single-sample scoring. "
            "Use the pairwise execution path via execute_pairwise_reward()."
        )

    @abstractmethod
    async def async_score_pair(
        self, row_a: dict[str, Any], row_b: dict[str, Any]
    ) -> float:
        """Return a score in ``[0, 1]`` indicating preference of *a* over *b*.

        - 1.0 means *a* is strongly preferred.
        - 0.5 means no preference (tie).
        - 0.0 means *b* is strongly preferred.
        """
        ...

    def aggregate(self, win_matrix: torch.Tensor) -> torch.Tensor:
        """Aggregate a ``[K, K]`` pairwise win matrix into ``[K]`` per-sample scores.

        Default: average win rate (mean across columns for each row).
        """
        return win_matrix.mean(dim=1)

    def supports_rollout_overlap(self) -> bool:
        return True


if __name__ == "__main__":
    import asyncio

    from rich import print as rprint

    class MockPairwiseReward(PairwiseReward):
        """Mock pairwise reward for testing: prefers higher 'quality' values."""

        type: Literal["pairwise"] = "pairwise"

        async def async_score_pair(
            self, row_a: dict[str, Any], row_b: dict[str, Any]
        ) -> float:
            import math

            qa: float = row_a.get("quality", 0.5)
            qb: float = row_b.get("quality", 0.5)
            # Simple sigmoid-like preference
            diff = qa - qb
            return 1.0 / (1.0 + math.exp(-diff * 5.0))

        def _load_model(self, device: torch.device) -> None:
            pass

    rprint("[bold cyan]===== PairwiseReward aggregation test =====[/]")

    # Simulate K=4 rollouts with known quality ordering
    K = 4
    qualities = [0.2, 0.8, 0.5, 0.9]

    reward = MockPairwiseReward()

    # Build win matrix
    win_matrix = torch.zeros(K, K)
    for i in range(K):
        for j in range(K):
            if i == j:
                win_matrix[i, j] = 0.5
            else:
                row_a = {"quality": qualities[i]}
                row_b = {"quality": qualities[j]}
                win_matrix[i, j] = asyncio.run(reward.async_score_pair(row_a, row_b))

    rprint(f"  Qualities: {qualities}")
    rprint(f"  Win matrix:\n{win_matrix}")

    scores = reward.aggregate(win_matrix)
    rprint(f"  Aggregated scores: {scores}")
    rprint(f"  Ranking (highest first): {scores.argsort(descending=True).tolist()}")

    # Verify ranking matches quality ordering
    expected_rank = sorted(range(K), key=lambda i: qualities[i], reverse=True)
    actual_rank = scores.argsort(descending=True).tolist()
    rprint(f"  Expected rank: {expected_rank}")
    assert actual_rank == expected_rank, (
        f"Ranking mismatch: {actual_rank} != {expected_rank}"
    )

    rprint("\n[bold green]Pairwise reward test passed.[/]")
