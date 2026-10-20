"""Pref-GRPO's round robin: a rollout's reward is its win rate among its prompt's rollouts."""

import asyncio
from typing import TYPE_CHECKING, Annotated, Literal

import torch
from pydantic import Field

from flow_control.rewards.base import Judge
from flow_control.samplers.node import Row
from flow_control.training.rollouts.base import (
    PointwiseRollout,
    rollout_registry,
    write,
)
from flow_control.training.rollouts.pairwise import win_rates

if TYPE_CHECKING:
    from flow_control.training.rollouts.runner import RewardContext


@rollout_registry.register("round_robin")
class RoundRobinRollout(PointwiseRollout):
    """Pref-GRPO: :attr:`judge` compares a prompt's rollouts in pairs, and each
    rollout's win rate is the component ``round_robin``, ahead of the scorers of
    :attr:`reward`."""

    type: Literal["round_robin"] = "round_robin"
    judge: Judge
    """P(a ≻ b) of two rollouts of one prompt."""
    swap_orders: bool = True
    """Ask each pair in both orders, ``s = (s_ab + 1 - s_ba) / 2``. A judge forced
    to answer A or B has a position bias; asking both orders cancels it. ``False``
    halves the requests, for an order-invariant judge."""
    opponents: Annotated[int, Field(ge=2, multiple_of=2)] | None = None
    """Opponents per rollout: the prompt's rollouts form a random cycle and each
    meets the ``opponents / 2`` nearest on either side. ``None`` (or at least
    K - 1): every pair, K(K-1)/2 games."""
    weight: float = 1.0
    """Weight of the win-rate component ``round_robin``."""

    @property
    def labels(self) -> list[str]:
        return ["round_robin", *self.reward.labels]

    @property
    def weights(self) -> list[float]:
        return [self.weight, *self.reward.weights]

    async def score(self, ctx: "RewardContext", rows: list[Row]) -> list[Row]:
        wins, raw = await asyncio.gather(
            win_rates(
                ctx,
                self.judge,
                rows,
                swap_orders=self.swap_orders,
                opponents=self.opponents,
            ),
            self.reward.score(ctx, rows),
        )
        write(
            rows,
            torch.cat([wins, raw], -1),
            torch.cat([wins, self.reward.normalize(raw)], -1),
        )
        return rows


if __name__ == "__main__":
    from typing import cast

    from pydantic import TypeAdapter
    from rich import print
    from torch import Tensor

    from flow_control.rewards.base import (
        BaseJudge,
        BaseScorer,
        LocalReward,
        RemoteReward,
        Reward,
        judge_registry,
        scorer_registry,
    )
    from flow_control.rewards.normalize import AffineNormalize, Normalize
    from flow_control.samplers import derive_seed
    from flow_control.training.rollouts.base import Rollout

    @scorer_registry.register("_test_rr_scorer")
    class _Scorer(BaseScorer, LocalReward[Row, Tensor]):
        """``[q, -q]`` of the row's quality; ``2x + 1`` by default."""

        type: Literal["_test_rr_scorer"] = "_test_rr_scorer"
        normalize: Normalize = Field(
            default_factory=lambda: AffineNormalize(scale=2, offset=1)
        )

        @property
        def labels(self) -> list[str]:
            return ["value", "negated"]

        @property
        def weights(self) -> list[float]:
            return [1.0, 0.0]

        @property
        def fields(self) -> set[str]:
            return {"quality"}

        def run_batch(self, rows: list[Row]) -> list[Tensor]:
            return [torch.tensor([row["quality"], -row["quality"]]) for row in rows]

    @judge_registry.register("_test_rr_judge")
    class _Judge(BaseJudge, RemoteReward[tuple[Row, Row], float]):
        """``0.5 + (q_a - q_b) / 2``, plus a position bias for the first row."""

        type: Literal["_test_rr_judge"] = "_test_rr_judge"
        bias: float = 0.1

        @property
        def fields(self) -> set[str]:
            return {"quality"}

        async def run_one(self, pair: tuple[Row, Row]) -> float:
            a, b = pair
            return 0.5 + (a["quality"] - b["quality"]) / 2 + self.bias

    class _Ctx:
        """Stands in for the runner's RewardContext: runs rewards in place."""

        epoch, prompt_id = 0, 0

        def seed(self, *parts: object) -> int:
            return derive_seed(0, ":".join(map(str, parts)))

        async def run[I, O](self, reward: Reward[I, O], items: list[I]) -> list[O]:
            if isinstance(reward, LocalReward):
                return reward.run_batch(items)
            return list(await asyncio.gather(*(reward.run_one(i) for i in items)))

    ctx = cast("RewardContext", _Ctx())
    config = {
        "type": "round_robin",
        "sampler": {"type": "grid", "steps": 2},
        "judge": {"type": "_test_rr_judge"},
        "opponents": 4,
        "weight": 0.5,
        "reward": {"type": "_test_rr_scorer", "weight": 0.5},
    }
    rollout = TypeAdapter(Rollout).validate_python(config)

    # K = 4, so 4 opponents is every pair: rate_i = 0.5 + (4 q_i - sum q) / 8 with
    # the judge's bias folded out; the scorer columns are raw [q, -q], normalized
    # 2x + 1.
    q = torch.tensor([0.2, 0.8, 0.5, 0.9])
    rows: list[Row] = [{"quality": v} for v in q.tolist()]
    assert asyncio.run(rollout.score(ctx, rows)) is rows
    rates = 0.5 + (4 * q - q.sum()) / 8
    raw = torch.stack([q, -q], -1)
    assert all(len(row["reward"]) == len(rollout.labels) for row in rows)
    reward_raw = torch.stack([row["reward_raw"] for row in rows])
    reward = torch.stack([row["reward"] for row in rows])
    assert torch.allclose(reward_raw, torch.cat([rates[:, None], raw], -1)), reward_raw
    assert torch.allclose(reward, torch.cat([rates[:, None], 2 * raw + 1], -1)), reward

    # Without scorers the win rate is the only component.
    bare = TypeAdapter(Rollout).validate_python(
        {k: v for k, v in config.items() if k != "reward"}
    )
    assert bare.labels == ["round_robin"] and bare.weights == [0.5], bare
    asyncio.run(bare.score(ctx, rows))
    assert torch.allclose(torch.stack([row["reward"] for row in rows]), rates[:, None])

    print("[green]rollouts.round_robin self-test passed[/green]")
