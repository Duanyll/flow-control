"""Pairwise helpers for rollout programs that judge rows in pairs.

Shared by :class:`~flow_control.training.rollouts.round_robin.RoundRobinRollout` and
out-of-tree programs (a judge against an anchor sample, say).
"""

import itertools
import random
from typing import TYPE_CHECKING, cast

import torch
from einops import rearrange
from torch import Tensor

from flow_control.rewards.base import BaseJudge, Reward
from flow_control.samplers.node import Row

if TYPE_CHECKING:
    from flow_control.training.rollouts.runner import RewardContext


def schedule(
    n: int, opponents: int | None, rng: random.Random
) -> list[tuple[int, int]]:
    """Index pairs of the games among *n* rows.

    *opponents* ``None`` or at least ``n - 1``: every pair ``(i, j)``, ``i < j``.
    Otherwise (even): *rng* shuffles the rows into a cycle and each row plays
    the ``opponents / 2`` next along it, so every row has exactly *opponents*
    distinct opponents and every pair plays once.
    """
    if opponents is None or opponents >= n - 1:
        return list(itertools.combinations(range(n), 2))
    cycle = list(range(n))
    rng.shuffle(cycle)
    return [
        (a, cycle[(i + step) % n])
        for i, a in enumerate(cycle)
        for step in range(1, opponents // 2 + 1)
    ]


async def preferences(
    ctx: "RewardContext",
    judge: BaseJudge,
    pairs: list[tuple[Row, Row]],
    swap_orders: bool,
) -> Tensor:
    """``[P]`` float32 P(a ≻ b) of each pair ``(a, b)``.

    With *swap_orders* one ``ctx.run`` asks both orders, folded as
    ``(s_ab + 1 - s_ba) / 2``.
    """
    # The judge registry admits only classes with exactly one placement.
    reward = cast("Reward[tuple[Row, Row], float]", judge)
    items = (pairs + [(b, a) for a, b in pairs]) if swap_orders else pairs
    s = torch.tensor(await ctx.run(reward, items), dtype=torch.float32)
    if not swap_orders:
        return s
    ab, ba = rearrange(s, "(order p) -> order p", order=2)
    return (ab + 1 - ba) / 2


async def win_rates(
    ctx: "RewardContext",
    judge: BaseJudge,
    rows: list[Row],
    *,
    swap_orders: bool,
    opponents: int | None,
) -> Tensor:
    """``[K, 1]`` win rate of each row over its :func:`schedule`, ``(0.5 + wins) / (played + 1)``.

    The extra half game is a tie with itself: over every pair this is the mean
    of the win matrix with a 0.5 diagonal. The schedule is seeded by the epoch
    and the prompt.
    """
    rng = random.Random(ctx.seed("round_robin", ctx.epoch, ctx.prompt_id))
    games = schedule(len(rows), opponents, rng)
    s = await preferences(
        ctx, judge, [(rows[i], rows[j]) for i, j in games], swap_orders
    )
    wins, played = [0.5] * len(rows), [1] * len(rows)
    for (i, j), p in zip(games, s.tolist(), strict=True):
        wins[i] += p
        wins[j] += 1 - p
        played[i] += 1
        played[j] += 1
    return rearrange(torch.tensor(wins) / torch.tensor(played), "k -> k 1")


if __name__ == "__main__":
    import asyncio
    from typing import Literal

    from rich import print

    from flow_control.rewards.base import RemoteReward, judge_registry
    from flow_control.samplers import derive_seed

    @judge_registry.register("_test_pair_judge")
    class _Judge(BaseJudge, RemoteReward[tuple[Row, Row], float]):
        """``0.5 + (q_a - q_b) / 2`` on the rows' quality, plus ``bias`` for the first."""

        type: Literal["_test_pair_judge"] = "_test_pair_judge"
        bias: float = 0.0

        @property
        def fields(self) -> set[str]:
            return {"quality"}

        async def run_one(self, pair: tuple[Row, Row]) -> float:
            a, b = pair
            return 0.5 + (a["quality"] - b["quality"]) / 2 + self.bias

    class _Ctx:
        """Stands in for the runner's RewardContext."""

        epoch = 3

        def __init__(self, prompt_id: int = 0) -> None:
            self.prompt_id = prompt_id
            self.calls: list[int] = []  # items per ctx.run()

        def seed(self, *parts: object) -> int:
            return derive_seed(0, ":".join(map(str, parts)))

        async def run[I, O](self, reward: Reward[I, O], items: list[I]) -> list[O]:
            self.calls.append(len(items))
            assert isinstance(reward, RemoteReward), reward
            return list(await asyncio.gather(*(reward.run_one(i) for i in items)))

    def opponents_of(games: list[tuple[int, int]], n: int) -> list[set[int]]:
        seen: list[set[int]] = [set() for _ in range(n)]
        for i, j in games:
            seen[i].add(j)
            seen[j].add(i)
        return seen

    # schedule: d distinct opponents per row along a seeded cycle, each pair once.
    for d in (2, 4):
        games = schedule(6, d, random.Random(0))
        assert all(i != j for i, j in games), games
        assert len({frozenset(g) for g in games}) == len(games) == 6 * d // 2, games
        assert all(len(o) == d for o in opponents_of(games, 6)), games
        assert games == schedule(6, d, random.Random(0)), "same seed, same schedule"
        assert any(games != schedule(6, d, random.Random(s)) for s in (1, 2, 3))
    every = list(itertools.combinations(range(6), 2))
    for d in (None, 5, 6):
        assert schedule(6, d, random.Random(0)) == every, d

    # preferences: asking both orders in one call folds the position bias out.
    biased = _Judge(bias=0.1)
    lo, hi, mid = {"quality": 0.2}, {"quality": 0.8}, {"quality": 0.5}
    pairs = [(lo, hi), (mid, dict(mid))]
    ctx = _Ctx()
    stub = cast("RewardContext", ctx)
    swapped = asyncio.run(preferences(stub, biased, pairs, swap_orders=True))
    assert ctx.calls == [4], ctx.calls
    assert torch.allclose(swapped, torch.tensor([0.2, 0.5])), swapped
    once = asyncio.run(preferences(stub, biased, pairs, swap_orders=False))
    assert ctx.calls == [4, 2], ctx.calls
    assert torch.allclose(once, torch.tensor([0.3, 0.6])), once

    # win_rates over every pair: the mean of the win matrix with a 0.5 diagonal.
    qualities = [0.2, 0.8, 0.5, 0.9]
    rows: list[Row] = [{"quality": q} for q in qualities]
    ascending = sorted(range(4), key=qualities.__getitem__)

    def p(a: Row, b: Row) -> float:
        return asyncio.run(biased.run_one((a, b)))

    matrix = torch.tensor(
        [[0.5 if a is b else (p(a, b) + 1 - p(b, a)) / 2 for b in rows] for a in rows]
    )
    every_pair = cast("RewardContext", _Ctx())  # the stub stands in for it
    rates = asyncio.run(
        win_rates(every_pair, biased, rows, swap_orders=True, opponents=None)
    )
    assert rates.shape == (4, 1), rates.shape
    assert torch.allclose(rates[:, 0], matrix.mean(1)), (rates, matrix)
    assert rates[:, 0].argsort().tolist() == ascending, rates

    # opponents=2 of the 3 others: each row's rate over its two scheduled games
    # only. A row's rate then depends on the opponent it missed, so close rows can
    # swap in general; for these qualities the order holds, ties allowed (prompt
    # ids 0-3 draw all three ways to split 4 rows into missed pairs).
    for prompt_id in range(4):
        ctx = _Ctx(prompt_id)
        rates = asyncio.run(
            win_rates(
                cast("RewardContext", ctx), biased, rows, swap_orders=True, opponents=2
            )
        )[:, 0]
        rng = random.Random(ctx.seed("round_robin", ctx.epoch, prompt_id))
        met = opponents_of(schedule(4, 2, rng), 4)
        expected = [(0.5 + sum(matrix[i, j] for j in met[i])) / 3 for i in range(4)]
        assert ctx.calls == [8], ctx.calls  # 4 games, both orders
        assert torch.allclose(rates, torch.tensor(expected)), (rates, expected)
        assert (rates[ascending].diff() > -1e-6).all(), rates

    print("[green]rollouts.pairwise self-test passed[/green]")
