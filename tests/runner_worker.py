"""Two-rank ``Runner`` check (gloo, CPU fakes); launch by hand:

    uv run torchrun --standalone --nproc_per_node=2 tests/runner_worker.py

The ranks do unequal work: rank 0 submits three prompts and rank 1 one (K = 4,
one local and one remote scorer with random sleeps), then a validation-shaped
pass (k = 1, rank 1 with an extra item) consumed with ``as_completed()``. Every
rank must get all its rows back with rewards written, and both ranks must have
issued the identical sequence of adapter forwards: the executor's per-variant
calls are collective, so a round skipped on one rank only would deadlock or
desync the real model's collectives.
"""

import asyncio
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

import torch
import torch.distributed as dist
from pydantic import PrivateAttr
from rich import print

from flow_control.processors.base import BaseProcessor
from flow_control.rewards.base import LocalReward, RemoteReward
from flow_control.samplers import Grid, Row
from flow_control.training.rollouts.runner import RewardContext, Runner, SampleContext


class FakeModel:
    device = torch.device("cpu")
    dtype = torch.float32
    micro_batch_size = 4
    variants: dict[str, object] = {"default": None, "ema": None}

    def __init__(self) -> None:
        self.active_variant = "default"
        self.forwards: list[tuple[str, int]] = []
        """``(variant, rows)`` of every forward call, empty ones included."""

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


class LocalScorer(LocalReward[Row, torch.Tensor]):
    batch_size: int = 8
    _sizes: list[int] = PrivateAttr(default_factory=list)

    def run_batch(self, rows: list[Row]) -> list[torch.Tensor]:
        self._sizes.append(len(rows))
        time.sleep(random.uniform(0.001, 0.005))
        return [row["clean_image"].mean().reshape(1) for row in rows]


class RemoteScorer(RemoteReward[Row, torch.Tensor]):
    @property
    def fields(self) -> set[str]:
        return {"clean_image"}

    async def run_one(self, row: Row) -> torch.Tensor:
        assert set(row) == {"clean_image"}, sorted(row)
        assert row["clean_image"].device.type == "cpu"
        await asyncio.sleep(random.uniform(0.001, 0.02))
        return row["clean_image"].mean().reshape(1)


LOCAL, REMOTE, GRID = LocalScorer(), RemoteScorer(), Grid(steps=2)


def items(prompt_ids: list[int]) -> list[tuple[int, Row]]:
    return [(i, {"key": f"p{i}", "image_size": (32, 32)}) for i in prompt_ids]


async def sample(ctx: SampleContext) -> list[Row]:
    return await ctx.rollouts()


async def score(ctx: RewardContext, rows: list[Row]) -> list[Row]:
    local, remote = await asyncio.gather(ctx.run(LOCAL, rows), ctx.run(REMOTE, rows))
    for row, a, b in zip(rows, local, remote, strict=True):
        row["reward_raw"] = row["reward"] = torch.cat([a, b])
    return rows


async def sample_one(ctx: SampleContext) -> list[Row]:
    """Validation-shaped: one row per item under an explicit seed."""
    row = await ctx.start(ctx.seed("validation", ctx.prompt_id))
    await ctx.admit(GRID(row))
    await ctx.decode(row)
    return [row]


async def score_one(ctx: RewardContext, rows: list[Row]) -> list[Row]:
    (value,) = await ctx.run(LOCAL, rows)
    rows[0]["reward_raw"] = rows[0]["reward"] = value
    return rows


def check_row(row: Row, components: int) -> None:
    assert not any(key.startswith("_") for key in row), sorted(row)
    assert row["clean_image"].device.type == "cpu" and "clean_latents" in row
    assert row["reward"].shape == (components,), row["reward"]


def main() -> None:
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2, "launch with --nproc_per_node=2"
    model = FakeModel()
    processor = cast("BaseProcessor", FakeProcessor())
    runner = Runner(model, processor, torch.device("cpu"), seed=0)
    try:
        prompts = [0, 1, 2] if rank == 0 else [3]
        pending = runner.submit(
            items(prompts),
            sample,
            score,
            prepare=dict,
            k=4,
            epoch=0,
            variant=None,
            sampler=GRID,
        )
        pending.sampled()
        groups = pending.wait()
        assert [[row["key"] for row in rows] for rows in groups] == [
            [f"p{i}"] * 4 for i in prompts
        ]
        for rows in groups:
            for row in rows:
                check_row(row, 2)
        assert pending.profile.count == 4 * len(prompts), pending.profile.count

        val_items = [10, 11] if rank == 0 else [20, 21, 22]
        pending = runner.submit(
            items(val_items),
            sample_one,
            score_one,
            prepare=dict,
            k=1,
            epoch=1,
            variant=None,
        )
        done: list[str] = []
        for rows in pending.as_completed():
            check_row(rows[0], 1)
            done.append(rows[0]["key"])
        assert sorted(done) == [f"p{i}" for i in val_items], done

        # A failure on one rank raises on both in the same round: the original
        # there, a RuntimeError naming it elsewhere. Last, as the runner keeps
        # raising for a failed submission.
        async def failing_score(ctx: RewardContext, rows: list[Row]) -> list[Row]:
            if rank == 0:
                raise ValueError("rank 0 boom")
            return rows

        pending = runner.submit(
            items([30]),
            sample_one,
            failing_score,
            prepare=dict,
            k=1,
            epoch=2,
            variant=None,
        )
        try:
            pending.wait()
        except ValueError as error:
            assert rank == 0 and str(error) == "rank 0 boom", error
        except RuntimeError as error:
            assert rank == 1 and "rank 0" in str(error), error
        else:
            raise AssertionError("wait() must raise on both ranks")

        sequences: list[Any] = [None] * world
        dist.all_gather_object(sequences, [name for name, _ in model.forwards])
        assert sequences[0] == sequences[1], (
            f"forward sequences differ: {len(sequences[0])} vs {len(sequences[1])} calls"
        )
        rows_forwarded = sum(n for _, n in model.forwards)
        nonempty = sum(1 for _, n in model.forwards if n)
        print(
            f"rank {rank}: {len(model.forwards)} forward calls on both ranks "
            f"({nonempty} non-empty here, {rows_forwarded} rows); local batches "
            f"{LOCAL._sizes}"
        )
    finally:
        runner.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
