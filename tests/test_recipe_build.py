"""SDEdit initialization and sampler RNG semantics."""

import random
import unittest

import torch
from test_microbatching import FakeSamplerModel, make_sampler_batch

from flow_control.samplers import Sampler, SampleRequest, Start
from flow_control.samplers.solver import SASolver
from flow_control.samplers.transforms import select_sde_window


class RecipeBuildTest(unittest.TestCase):
    def test_sdedit_renoise_slices_inherited_plan(self) -> None:
        # SDEdit must use the actual sliced grid sigma, including SA head rewrites.
        for solver in (None, SASolver(eta=0.0)):
            sampler = Sampler(
                steps=6, start=Start(source="clean_latents", strength=0.6)
            )
            if solver is not None:
                sampler.solver = solver
            batch = make_sampler_batch(0.5, initial=0.3)
            steps = []
            run = next(
                iter(
                    sampler.sample(
                        FakeSamplerModel(),
                        [
                            SampleRequest(
                                batch=batch, generator=torch.Generator().manual_seed(5)
                            )
                        ],
                        collector=lambda run, step, steps=steps: steps.append(step),
                    )
                )
            )
            first = steps[0]
            sigma, latents = first.transition.sigma, first.latents
            self.assertLessEqual(sigma, 0.6)
            self.assertEqual(sigma, run.plan[0].sigma)
            clean = batch["clean_latents"].float()
            noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(5))
            torch.testing.assert_close(latents, (1 - sigma) * clean + sigma * noise)


class SelectSdeWindowTest(unittest.TestCase):
    """RNG semantics of the window selection (ported from the deleted
    Sampler.trajectory_window_* surface)."""

    def test_window_uses_the_given_generator_not_global_rng(self) -> None:
        first_generator = torch.Generator().manual_seed(1234)
        random.seed(1)
        first = select_sde_window(12, 3, (1, 10), first_generator)

        for _ in range(100):
            random.random()
        second = select_sde_window(12, 3, (1, 10), torch.Generator().manual_seed(1234))

        self.assertEqual(first, second)
        # The draw consumes the generator's state.
        self.assertNotEqual(
            first_generator.get_state().tolist(),
            torch.Generator().manual_seed(1234).get_state().tolist(),
        )
