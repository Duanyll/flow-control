"""SDEdit sampling and GRPO replay across the public sampler API."""

import unittest

import torch
from test_microbatching import FakeSamplerModel, make_sampler_batch

from flow_control.samplers import Sampler, SampleRequest, Start
from flow_control.samplers.solver import FlowSolver
from flow_control.training.grpo_sampling import (
    ReplayItem,
    collect_samples,
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
        outputs, records = collect_samples(
            self.SAMPLER,
            FakeSamplerModel(),
            [SampleRequest(batch=batch, generator=torch.Generator().manual_seed(seed))],
        )
        return batch, outputs[0], records[0]

    def test_plan_slice_determinism_and_replay(self) -> None:
        batch, output, trajectory = self.run_once(7)
        self.assertEqual(output.timesteps.numel(), 6)
        self.assertAlmostEqual(float(output.timesteps[0]), 0.6, places=5)
        _, repeat, _ = self.run_once(7)
        assert_bitwise(repeat.final_latents, output.final_latents)
        _, other, _ = self.run_once(8)
        self.assertFalse(torch.equal(other.final_latents, output.final_latents))
        self.assertEqual(len(trajectory), 5)
        self.assertTrue(all((step.log_prob != 0).all() for step in trajectory))
        replayed = replay_steps(
            self.SAMPLER,
            FakeSamplerModel(),
            [ReplayItem(batch=batch, recorded=step) for step in trajectory],
        )
        for step, replay in zip(trajectory, replayed, strict=True):
            assert_bitwise(replay.log_prob, step.log_prob)
