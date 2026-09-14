"""SDEdit sampling and GRPO replay across the public sampler API."""

import unittest

import torch
from test_microbatching import FakeSamplerModel, make_sampler_batch

from flow_control.samplers import Sampler, SampleRequest, Start
from flow_control.samplers.solver import FlowSolver
from flow_control.training.grpo_sampling import (
    GrpoCollector,
    ReplayItem,
    replay_steps,
)


def assert_bitwise(a, b):
    torch.testing.assert_close(a, b, rtol=0, atol=0)


class SdeditRecipeTest(unittest.TestCase):
    SAMPLER = Sampler(
        steps=10,
        solver=FlowSolver(eta=0.7),
        start=Start(source="clean_latents", strength=0.6),
    )

    def run_once(self, seed: int):
        batch = make_sampler_batch(0.5, initial=0.3)
        collector = GrpoCollector(self.SAMPLER)
        run = next(
            iter(
                self.SAMPLER.sample(
                    FakeSamplerModel(),
                    [
                        SampleRequest(
                            row=batch, generator=torch.Generator().manual_seed(seed)
                        )
                    ],
                    collector=collector,
                )
            )
        )
        return run, collector.take(run)

    def test_plan_slice_determinism_and_replay(self) -> None:
        run, trajectory = self.run_once(7)
        self.assertEqual(len(run.plan), 6)
        self.assertAlmostEqual(run.plan[0].sigma, 0.6, places=5)
        repeat, _ = self.run_once(7)
        assert_bitwise(repeat.ctx.latents, run.ctx.latents)
        other, _ = self.run_once(8)
        self.assertFalse(torch.equal(other.ctx.latents, run.ctx.latents))
        self.assertEqual(len(trajectory), 5)
        self.assertTrue(all((step.log_prob != 0).all() for step in trajectory))
        replayed = replay_steps(
            FakeSamplerModel(), [ReplayItem(run, step) for step in trajectory]
        )
        for step, replay in zip(trajectory, replayed, strict=True):
            assert_bitwise(replay.log_prob, step.log_prob)
