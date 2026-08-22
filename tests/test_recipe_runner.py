"""Recipe runner tests (plan-as-data Phase 4b).

CPU-only end-to-end coverage of ``run_phases`` against ``Sampler.sample``
(bitwise for the default recipe) and against direct plan+executor construction
(bitwise for the GRPO-style windowed recipe), SDEdit / inversion recipes,
phase-boundary state isolation, and the executor fingerprint hardening.
"""

import importlib
import sys
import unittest
from pathlib import Path
from typing import Literal

import torch

from flow_control.adapters.base import Batch
from flow_control.samplers import (
    PhaseConfig,
    PhasesRecipe,
    RecipeBuildContext,
    ReplayItem,
    Sampler,
    SampleRequest,
    SdeWindow,
    run_phases,
)
from flow_control.samplers.executor import (
    Run,
    _apply_result,
    _plan_fingerprint,
    execute,
)
from flow_control.samplers.guidance import BaseGuidance, ClassifierFreeGuidance
from flow_control.samplers.plan import (
    BranchEvals,
    GuidanceOutput,
    GuidanceState,
    StepContext,
    Transition,
    TransitionResult,
)
from flow_control.samplers.recipe import (
    BaseRecipe,
    FromLatents,
    FromPrevious,
    Invert,
    Renoise,
    plan_start_sigma,
)
from flow_control.samplers.solver import (
    DPMSolver,
    FlowSolver,
    FlowUniPCSolver,
    SASolver,
)
from flow_control.samplers.transforms import (
    finalize_replay_state,
    select_sde_window,
    with_sde_window,
)

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
microbatching = importlib.import_module("test_microbatching")


def assert_bitwise(a: torch.Tensor, b: torch.Tensor) -> None:
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def build_phases(
    recipe: BaseRecipe,
    sampler: Sampler,
    batch: Batch,
    seed: int,
    negative: Batch | None = None,
    **extra_batches: Batch,
):
    ctx = RecipeBuildContext(
        default_sampler=sampler,
        batches={"main": batch, **extra_batches},
        negative_batch_for=lambda name: negative if name == "main" else None,
        generator=torch.Generator().manual_seed(seed),
    )
    return recipe.build(ctx)


class RunPhasesEquivalenceTest(unittest.TestCase):
    def test_default_recipe_matches_sample_bitwise(self) -> None:
        # Stochastic flow solver + true CFG with a mixed negative-batch pair:
        # the exact configuration Sampler.sample runs today.
        sampler = Sampler(
            steps=6,
            solver=FlowSolver(eta=0.7),
            guidance=ClassifierFreeGuidance(scale=2.0),
        )
        batches = [
            microbatching.make_sampler_batch(1.0, initial=0.3),
            microbatching.make_sampler_batch(2.0, initial=-0.2),
        ]
        negatives = [microbatching.make_sampler_batch(0.5), None]

        old = sampler.sample(
            microbatching.FakeSamplerModel(),
            [
                SampleRequest(
                    batch=batch,
                    negative_batch=negative,
                    generator=torch.Generator().manual_seed(11 + index),
                )
                for index, (batch, negative) in enumerate(
                    zip(batches, negatives, strict=True)
                )
            ],
        )
        new = run_phases(
            microbatching.FakeSamplerModel(),
            [
                build_phases(
                    PhasesRecipe(), sampler, batch, 11 + index, negative=negative
                )
                for index, (batch, negative) in enumerate(
                    zip(batches, negatives, strict=True)
                )
            ],
        )
        for old_output, new_output in zip(old, new, strict=True):
            assert_bitwise(new_output.final_latents, old_output.final_latents)
            assert_bitwise(new_output.timesteps, old_output.timesteps)
            self.assertIsNone(new_output.trajectory)
            self.assertIsNone(old_output.trajectory)

    def test_grpo_recipe_matches_direct_plan_execution_bitwise(self) -> None:
        # The recipe path (build + run_phases) must reproduce the manual
        # transform pipeline through the executor bitwise — same generator
        # stream (window randint first, then per-step noise), same recorded
        # window, same replay descriptors. This pins the semantics of the
        # deleted sample(return_trajectory=True) windowed path.
        sampler = Sampler(steps=6, solver=FlowSolver(eta=0.7))
        recipe = PhasesRecipe(
            phases=[
                PhaseConfig(transforms=[SdeWindow(size=2, range=(1, 5), record=True)])
            ]
        )
        batch = microbatching.make_sampler_batch(0.4, initial=0.1)

        generator = torch.Generator().manual_seed(11)
        start, end = select_sde_window(sampler.steps, 2, (1, 5), generator)
        plan = finalize_replay_state(
            with_sde_window(sampler.plan(batch), start, end, record=True)
        )
        direct = Run(
            plan=plan,
            ctx=StepContext(
                latents=batch["noisy_latents"].float(),
                generator=generator,
                solver_state=None,
                guidance_state=sampler.guidance.init_state(),
            ),
            batch=batch,
            negative_batch=None,
        )
        list(execute(microbatching.FakeSamplerModel(), [direct], sampler.guidance))

        new = run_phases(
            microbatching.FakeSamplerModel(),
            [build_phases(recipe, sampler, batch, 11)],
        )[0]

        assert_bitwise(new.final_latents, direct.ctx.latents)
        assert_bitwise(
            new.timesteps,
            torch.tensor([item.sigma for item in plan], dtype=torch.float32),
        )
        assert new.trajectory is not None
        self.assertEqual(len(new.trajectory), len(direct.recorded))
        for new_step, direct_step in zip(new.trajectory, direct.recorded, strict=True):
            assert_bitwise(new_step.latent_t, direct_step.latent_t)
            assert_bitwise(new_step.latent_next, direct_step.latent_next)
            assert_bitwise(new_step.log_prob, direct_step.log_prob)
            self.assertEqual(new_step.replay, direct_step.replay)
            self.assertIsNone(new_step.solver_state)


class SdeditRecipeTest(unittest.TestCase):
    SAMPLER = Sampler(steps=10, solver=FlowSolver(eta=0.7))
    RECIPE = PhasesRecipe(
        phases=[
            PhaseConfig(
                init=Renoise(strength=0.6),
                transforms=[SdeWindow(record=True)],
            )
        ]
    )

    def run_once(self, seed: int):
        batch = microbatching.make_sampler_batch(0.5, initial=0.3)
        phases = build_phases(self.RECIPE, self.SAMPLER, batch, seed)
        output = run_phases(
            microbatching.FakeSamplerModel(),
            [phases],
        )[0]
        return batch, phases, output

    def test_plan_slice_determinism_and_replay(self) -> None:
        batch, phases, output = self.run_once(7)
        plan = phases[0].plan
        transitions = [item for item in plan if isinstance(item, Transition)]
        self.assertEqual(len(transitions), 6)
        self.assertAlmostEqual(plan_start_sigma(plan), 0.6, places=5)
        self.assertEqual(output.timesteps.numel(), 6)
        assert_bitwise(
            output.timesteps,
            torch.tensor([item.sigma for item in transitions], dtype=torch.float32),
        )

        # Deterministic given the seed; stochastic across seeds.
        _, _, repeat = self.run_once(7)
        assert_bitwise(repeat.final_latents, output.final_latents)
        _, _, other = self.run_once(8)
        self.assertFalse(torch.equal(other.final_latents, output.final_latents))

        # Full window over the slice records all but the terminal transition,
        # and the recorded window replays bitwise (deterministic fake model).
        trajectory = output.trajectory
        assert trajectory is not None
        self.assertEqual(len(trajectory), 5)
        self.assertTrue(all((step.log_prob != 0).all() for step in trajectory))
        items = [ReplayItem(batch=batch, recorded=step) for step in trajectory]
        replayed = self.SAMPLER.replay_recorded_steps(
            microbatching.FakeSamplerModel(),
            items,
        )
        for step, replay in zip(trajectory, replayed, strict=True):
            assert_bitwise(replay.log_prob, step.log_prob)


class InversionRecipeTest(unittest.TestCase):
    """Two-phase DDIM-style inversion edit with exact float arithmetic.

    steps=4 gives dt=0.25 and the fake velocities are powers of two, so every
    Euler update is exact in float32 and the checks can be bitwise.
    """

    INVERT_PHASE = PhaseConfig(
        init=FromLatents(source="clean_latents"), transforms=[Invert()]
    )

    def test_invert_phase_recovers_noise_exactly(self) -> None:
        sampler = Sampler(steps=4, solver=FlowSolver())
        batch = microbatching.make_sampler_batch(0.5)
        phases = build_phases(
            PhasesRecipe(phases=[self.INVERT_PHASE]), sampler, batch, 1
        )
        output = run_phases(
            microbatching.FakeSamplerModel(),
            [phases],
        )[0]
        # Constant velocity v: ascending 0 -> 1 from x0 recovers x1 = x0 + v.
        clean = batch["clean_latents"].float()
        assert_bitwise(output.final_latents, clean + 0.5)
        assert_bitwise(
            output.timesteps, torch.tensor([0.0, 0.25, 0.5, 0.75], dtype=torch.float32)
        )
        self.assertIsNone(output.trajectory)

    def run_two_phase(self, edit_velocity: float):
        sampler = Sampler(steps=4, solver=FlowSolver())
        recipe = PhasesRecipe(
            phases=[
                self.INVERT_PHASE,
                PhaseConfig(init=FromPrevious(), batch="edit"),
            ]
        )
        batch = microbatching.make_sampler_batch(0.5)
        edit = microbatching.make_sampler_batch(edit_velocity)
        phases = build_phases(recipe, sampler, batch, 1, edit=edit)
        self.assertIs(phases[1].batch, edit)
        output = run_phases(
            microbatching.FakeSamplerModel(),
            [phases],
        )[0]
        return batch, output

    def test_from_previous_reconstructs_clean_latents_exactly(self) -> None:
        batch, output = self.run_two_phase(edit_velocity=0.5)
        assert_bitwise(output.final_latents, batch["clean_latents"].float())
        assert_bitwise(
            output.timesteps,
            torch.tensor(
                [0.0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25], dtype=torch.float32
            ),
        )

    def test_second_phase_uses_the_edit_batch(self) -> None:
        # Different edit velocity w: final = clean + v - w, so the value pins
        # that the edit conditioning actually drove the second phase.
        batch, output = self.run_two_phase(edit_velocity=0.25)
        assert_bitwise(
            output.final_latents, batch["clean_latents"].float() + 0.5 - 0.25
        )


class PhaseBoundaryIsolationTest(unittest.TestCase):
    def test_multistep_phase_two_warms_up_from_empty_history(self) -> None:
        # Invert (first-order flow) then hand over to an inherited multistep
        # solver: phase 2 must run bitwise like a fresh sample() started from
        # the handover latents (fresh StepContext, empty history warmup).
        inversion = PhaseConfig(
            init=FromLatents(source="clean_latents"),
            transforms=[Invert()],
            sampler=Sampler(steps=6, solver=FlowSolver()),
        )
        for solver in (DPMSolver(order=2), FlowUniPCSolver()):
            with self.subTest(solver=solver.type):
                default = Sampler(steps=6, solver=solver)
                batch = microbatching.make_sampler_batch(0.5)

                mid = run_phases(
                    microbatching.FakeSamplerModel(),
                    [build_phases(PhasesRecipe(phases=[inversion]), default, batch, 3)],
                )[0].final_latents

                full = run_phases(
                    microbatching.FakeSamplerModel(),
                    [
                        build_phases(
                            PhasesRecipe(
                                phases=[inversion, PhaseConfig(init=FromPrevious())]
                            ),
                            default,
                            batch,
                            3,
                        )
                    ],
                )[0]

                reference_batch = microbatching.make_sampler_batch(0.5)
                reference_batch["noisy_latents"] = mid
                reference = default.sample(
                    microbatching.FakeSamplerModel(),
                    [SampleRequest(batch=reference_batch)],
                )[0]
                assert_bitwise(full.final_latents, reference.final_latents)


class PlanFingerprintTest(unittest.TestCase):
    def test_sa_final_flag_changes_the_fingerprint(self) -> None:
        solver = SASolver()
        five = solver.plan(torch.linspace(1.0, 0.0, 6).tolist())
        four = solver.plan(torch.linspace(1.0, 0.0, 5).tolist())
        # A front slice drops the zero-eval terminal marker: same length and
        # classes as a fresh 4-step plan, but a different eval topology.
        self.assertNotEqual(_plan_fingerprint(five[:4]), _plan_fingerprint(four))
        # A tail slice keeps the terminal marker: same topology as fresh.
        self.assertEqual(_plan_fingerprint(five[1:]), _plan_fingerprint(four))


class _ResetState(GuidanceState):
    pass


class _StatefulGuidance(BaseGuidance):
    type: Literal["stateful_test"] = "stateful_test"

    def init_state(self) -> GuidanceState | None:
        return _ResetState()

    def needs_negative(self) -> bool:
        return False

    def combine(
        self,
        evals: BranchEvals,
        ctx: StepContext,
        state: GuidanceState | None,
    ) -> tuple[GuidanceOutput, GuidanceState | None]:
        return GuidanceOutput(velocity=evals.cond, branches=evals), state


class GuidanceResetTest(unittest.TestCase):
    def test_reset_runtime_state_reseeds_guidance_state(self) -> None:
        # Contract rule 2 resolution: after a reset transition the guidance
        # state is a fresh init_state() of the current execute() call's
        # guidance, not None.
        guidance = _StatefulGuidance()
        stale = guidance.init_state()
        run = Run(
            plan=[],
            ctx=StepContext(
                latents=torch.zeros(1, 1, 1),
                generator=None,
                solver_state=None,
                guidance_state=stale,
            ),
            batch=microbatching.make_sampler_batch(0.0),
            negative_batch=None,
        )
        _apply_result(
            run,
            TransitionResult(
                next_latents=torch.ones(1, 1, 1), reset_runtime_state=True
            ),
            guidance,
        )
        self.assertIsInstance(run.ctx.guidance_state, _ResetState)
        self.assertIsNot(run.ctx.guidance_state, stale)
        self.assertIsNone(run.ctx.solver_state)


if __name__ == "__main__":
    unittest.main()
