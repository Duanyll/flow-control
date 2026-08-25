"""Config-level tests for the recipe layer (plan-as-data Phase 4a).

Everything runs on CPU without a real model: builds produce plans that are
compared against directly-constructed plans; inversion is exercised through
the executor with duck-typed velocity models.
"""

import random
import unittest

import torch

from flow_control.adapters.base import Batch
from flow_control.samplers.executor import Run, execute
from flow_control.samplers.guidance import ClassifierFreeGuidance
from flow_control.samplers.plan import StepContext, Transition
from flow_control.samplers.recipe import (
    FromPrevious,
    Invert,
    PhaseConfig,
    PhasesRecipe,
    RecipeBuildContext,
    Renoise,
    SdeWindow,
    plan_start_sigma,
)
from flow_control.samplers.sampler import Sampler
from flow_control.samplers.shift import ConstantShift
from flow_control.samplers.solver import (
    DDIMSolver,
    DPMSolver,
    FlowSolver,
    SASolver,
)
from flow_control.samplers.solver.ddim import DdimReplayStep
from flow_control.samplers.transforms import (
    finalize_replay_state,
    select_sde_window,
    with_sde_window,
)


def make_batch(tokens: int = 4, seed: int = 0) -> Batch:
    generator = torch.Generator().manual_seed(seed)
    return {
        "image_size": (32, 32),
        "clean_latents": torch.randn(1, tokens, 2, generator=generator),
        "noisy_latents": torch.randn(1, tokens, 2, generator=generator),
    }


def make_context(
    sampler: Sampler,
    batch: Batch,
    seed: int = 1,
    **extra_batches: Batch,
) -> RecipeBuildContext:
    return RecipeBuildContext(
        default_sampler=sampler,
        batches={"main": batch, **extra_batches},
        negative_batch_for=lambda name: None,
        generator=torch.Generator().manual_seed(seed),
    )


class FieldVelocityModel:
    """Duck-typed model whose velocity is a function of the latents."""

    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self, fn) -> None:
        self.fn = fn

    def predict_velocity_batched(
        self,
        batches: list,
        timesteps: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        return [self.fn(batch["noisy_latents"]) for batch in batches]


def run_plan(plan, latents: torch.Tensor, model) -> torch.Tensor:
    batch = make_batch()
    batch["noisy_latents"] = latents
    run = Run(
        plan=plan,
        ctx=StepContext(
            latents=latents,
            generator=None,
            solver_state=None,
            guidance_state=None,
        ),
        batch=batch,
        negative_batch=None,
    )
    list(execute(model, [run], ClassifierFreeGuidance()))
    return run.ctx.latents


class PlanInversionTest(unittest.TestCase):
    def test_invert_mirrors_transitions_and_strips_stochasticity(self) -> None:
        solver = FlowSolver(eta=0.7)
        plan = with_sde_window(
            solver.plan([1.0, 0.75, 0.5, 0.25, 0.0]), 0, 3, record=True
        )
        inverted = solver.invert(plan)
        self.assertEqual(len(inverted), len(plan))
        for mirrored, original in zip(inverted, reversed(plan), strict=True):
            assert isinstance(mirrored, Transition)
            assert isinstance(original, Transition)
            self.assertEqual(mirrored.sigma, original.sigma_next)
            self.assertEqual(mirrored.sigma_next, original.sigma)
            self.assertEqual(mirrored.eta, 0.0)
            self.assertFalse(mirrored.record)
            self.assertFalse(mirrored.save_solver_state)

    def test_constant_velocity_inverts_exactly(self) -> None:
        model = FieldVelocityModel(lambda x: torch.full_like(x, 0.37))
        initial = torch.randn(1, 4, 2, generator=torch.Generator().manual_seed(7))
        for solver in (FlowSolver(), DDIMSolver()):
            with self.subTest(solver=solver.type):
                plan = finalize_replay_state(
                    solver.plan(torch.linspace(1.0, 0.0, 9).tolist())
                )
                final = run_plan(plan, initial.clone(), model)
                inverted = finalize_replay_state(solver.invert(plan))
                self.assertEqual(plan_start_sigma(inverted), 0.0)
                recovered = run_plan(inverted, final, model)
                torch.testing.assert_close(recovered, initial, rtol=0, atol=1e-5)

    def test_ddim_sigma_zero_step_replays_finite(self) -> None:
        # A recorded inverted DDIM step starts at exactly sigma 0; its replay
        # must take the same Euler path as run_transition instead of dividing
        # by sigma in DDIMSolver.step_parts.
        latents = torch.randn(1, 4, 2, generator=torch.Generator().manual_seed(1))
        velocity = torch.full_like(latents, 0.3)
        replay = DdimReplayStep(sigma=0.0, sigma_next=0.25, eta=0.0)
        output = replay.logprob(velocity, latents, latents)
        self.assertTrue(torch.isfinite(output.mean).all())
        torch.testing.assert_close(output.mean, latents + 0.25 * velocity)
        self.assertTrue((output.log_prob == 0).all())

    def test_near_linear_velocity_inverts_approximately(self) -> None:
        # Euler inversion is the standard approximate DDIM inversion: with a
        # weakly state-dependent field the reconstruction error is small but
        # not zero.
        model = FieldVelocityModel(lambda x: 0.3 + 0.05 * x)
        initial = torch.randn(1, 4, 2, generator=torch.Generator().manual_seed(9))
        solver = FlowSolver()
        plan = finalize_replay_state(solver.plan(torch.linspace(1.0, 0.0, 9).tolist()))
        final = run_plan(plan, initial.clone(), model)
        displacement = (final - initial).abs().max().item()
        self.assertGreater(displacement, 0.1)
        recovered = run_plan(finalize_replay_state(solver.invert(plan)), final, model)
        error = (recovered - initial).abs().max().item()
        self.assertGreater(error, 0.0)
        self.assertLess(error, 0.01 * displacement)


class RecipeBuildTest(unittest.TestCase):
    def test_sde_window_counts_denoise_indices_of_the_sliced_plan(self) -> None:
        # After renoise-slicing, window indices are relative to the sliced
        # plan, not the full grid.
        batch = make_batch()
        sampler = Sampler(steps=10, solver=FlowSolver(eta=0.7))
        recipe = PhasesRecipe(
            phases=[
                PhaseConfig(
                    init=Renoise(strength=0.45),
                    transforms=[SdeWindow(record=True)],
                )
            ]
        )
        plan = recipe.build(make_context(sampler, batch))[0].plan
        transitions = [item for item in plan if isinstance(item, Transition)]
        self.assertEqual(len(transitions), 4)
        # Full window over the slice: every step but the terminal one records.
        self.assertEqual(
            [item.record for item in transitions], [True, True, True, False]
        )

    def test_sdedit_renoise_slices_inherited_plan(self) -> None:
        batch = make_batch(tokens=1024)
        sampler = Sampler(steps=10, shift=ConstantShift(shift_value=3.0))
        recipe = PhasesRecipe(phases=[PhaseConfig(init=Renoise(strength=0.6))])
        phase = recipe.build(make_context(sampler, batch))[0]

        full = sampler.plan(batch)
        expected_index = next(
            index
            for index, item in enumerate(full)
            if isinstance(item, Transition) and item.sigma <= 0.6
        )
        self.assertEqual(phase.plan, finalize_replay_state(full[expected_index:]))
        self.assertGreater(expected_index, 0)

        # Single source of truth: the init op interpolates at the plan's
        # aligned start sigma, not at the raw strength.
        aligned_sigma = plan_start_sigma(phase.plan)
        first = phase.plan[0]
        assert isinstance(first, Transition)
        self.assertEqual(aligned_sigma, first.sigma)
        self.assertNotEqual(aligned_sigma, 0.6)

        latents = phase.init.build_latents(
            batch, torch.Generator().manual_seed(5), aligned_sigma
        )
        clean = batch["clean_latents"].float()
        noise = torch.randn(
            clean.shape, dtype=clean.dtype, generator=torch.Generator().manual_seed(5)
        )
        torch.testing.assert_close(
            latents, (1.0 - aligned_sigma) * clean + aligned_sigma * noise
        )

    def test_explicit_sa_sampler_keeps_partial_start_sigma(self) -> None:
        batch = make_batch()
        partial_sampler = Sampler(steps=6, solver=SASolver())
        recipe = PhasesRecipe(
            phases=[
                PhaseConfig(
                    init=Renoise(strength=0.6),
                    sampler=partial_sampler,
                )
            ]
        )

        phase = recipe.build(make_context(Sampler(), batch))[0]

        first = phase.plan[0]
        assert isinstance(first, Transition)
        self.assertAlmostEqual(first.sigma, 0.6, places=6)
        self.assertAlmostEqual(plan_start_sigma(phase.plan), 0.6, places=6)

    def test_partial_plan_rejects_solver_head_rewrite(self) -> None:
        sampler = Sampler(steps=6, solver=SASolver(initial_time=0.999))

        with self.assertRaisesRegex(ValueError, "cannot start a partial plan"):
            sampler.plan_from_sigma(make_batch(), 0.9995)

    def test_negative_batch_resolved_only_when_guidance_needs_it(self) -> None:
        batch = make_batch()
        negative = make_batch(seed=4)
        requested: list[str] = []

        def negative_batch_for(name: str):
            requested.append(name)
            return negative

        def build(sampler: Sampler):
            ctx = RecipeBuildContext(
                default_sampler=sampler,
                batches={"main": batch},
                negative_batch_for=negative_batch_for,
                generator=None,
            )
            return PhasesRecipe().build(ctx)[0]

        phase = build(Sampler(steps=2))
        self.assertIsNone(phase.negative_batch)
        self.assertEqual(requested, [])

        phase = build(Sampler(steps=2, guidance=ClassifierFreeGuidance(scale=4.5)))
        self.assertIs(phase.negative_batch, negative)
        self.assertEqual(requested, ["main"])


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


class RecipeBuildErrorsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.batch = make_batch()
        self.sampler = Sampler(steps=6)

    def build(self, recipe: PhasesRecipe, sampler: Sampler | None = None):
        return recipe.build(make_context(sampler or self.sampler, self.batch))

    def test_invert_on_multistep_solver_raises(self) -> None:
        recipe = PhasesRecipe(phases=[PhaseConfig(transforms=[Invert()])])
        for solver in (DPMSolver(order=2), SASolver()):
            with (
                self.subTest(solver=solver.type),
                self.assertRaises(NotImplementedError),
            ):
                self.build(recipe, Sampler(steps=6, solver=solver))

    def test_from_previous_off_grid_handover_raises(self) -> None:
        # A fully denoised phase hands over sigma 0.0, which is not a grid
        # point of the inherited descending plan; silently re-labelling the
        # latents at a lower sigma would misalign the phase boundary.
        recipe = PhasesRecipe(phases=[PhaseConfig(), PhaseConfig(init=FromPrevious())])
        with self.assertRaises(NotImplementedError) as caught:
            self.build(recipe)
        self.assertIn("explicit sampler", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
