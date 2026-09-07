"""Guidance-axis tests for config, execution, and replay semantics."""

import importlib
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import PrivateAttr

from flow_control.processors.tasks.inpaint import InpaintProcessor
from flow_control.samplers import (
    PhaseConfig,
    PhasesRecipe,
    RecipeBuildContext,
    SdeWindow,
    run_phases,
)
from flow_control.samplers.executor import Run, execute
from flow_control.samplers.guidance import (
    BaseGuidance,
    ClassifierFreeGuidance,
    DifferentialDiffusionGuidance,
    MomentumGuidance,
    MomentumGuidanceState,
)
from flow_control.samplers.plan import (
    BranchEvals,
    GuidanceOutput,
    GuidanceState,
    RecordedStep,
    StepContext,
    euler_step,
)
from flow_control.samplers.sampler import ReplayItem, Sampler, SampleRequest
from flow_control.samplers.solver import (
    DPMSolver,
    FlashSolver,
    FlowSolver,
    SASolver,
)
from flow_control.samplers.solver.flow import FlowReplayStep
from flow_control.samplers.transforms import finalize_replay_state

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
microbatching = importlib.import_module("test_microbatching")


@dataclass(slots=True)
class _CounterState(GuidanceState):
    count: int


class _CountingGuidance(BaseGuidance):
    """Stateful dummy guidance: velocity = cond, state = eval counter.

    Never registered — instances pass through the union validator directly,
    the same way an out-of-tree plugin instance would.
    """

    type: Literal["counting"] = "counting"

    _seen: list[int] = PrivateAttr(default_factory=list)
    """Counter value received at each combine, in eval order."""

    def init_state(self) -> GuidanceState | None:
        return _CounterState(count=0)

    def needs_negative(self) -> bool:
        return False

    def combine(
        self,
        evals: BranchEvals,
        ctx: StepContext,
        state: GuidanceState | None,
    ) -> tuple[GuidanceOutput, GuidanceState | None]:
        assert isinstance(state, _CounterState)
        self._seen.append(state.count)
        return (
            GuidanceOutput(velocity=evals.cond, branches=evals),
            _CounterState(count=state.count + 1),
        )


class _PreparingCountingGuidance(_CountingGuidance):
    _prepared: list[int] = PrivateAttr(default_factory=list)

    def prepare_transition(
        self,
        run: Run,
        item_index: int,
    ) -> tuple[torch.Tensor, GuidanceState | None]:
        self._prepared.append(item_index)
        return run.ctx.latents, run.ctx.guidance_state


class GuidedVelocityCompatibilityTest(unittest.TestCase):
    def test_get_guided_velocity_matches_executor_path(self) -> None:
        sampler = Sampler(
            steps=1,
            guidance=ClassifierFreeGuidance(scale=2.0),
            solver=FlowSolver(),
        )
        cond = microbatching.make_sampler_batch(3.0)
        negative = microbatching.make_sampler_batch(1.0)

        velocity = sampler.get_guided_velocity(
            microbatching.FakeSamplerModel(),
            batches=[cond],
            negative_batches=[negative],
            latents=[cond["noisy_latents"]],
            timesteps=[torch.tensor([1.0])],
            sigmas=[1.0],
        )[0]
        torch.testing.assert_close(velocity, torch.tensor([[[5.0]]]))

        output = sampler.sample(
            microbatching.FakeSamplerModel(),
            [SampleRequest(batch=cond, negative_batch=negative)],
        )[0]
        # The executor's single Euler step must use the same guided velocity.
        torch.testing.assert_close(
            output.final_latents,
            euler_step(cond["noisy_latents"], velocity, 1.0, 0.0),
        )


class DifferentialDiffusionGuidanceTest(unittest.TestCase):
    @staticmethod
    def _make_run() -> Run:
        batch: Any = microbatching.make_batch(
            10.0,
            tokens=4,
            image_size=(32, 32),
        )
        batch["inpaint_latents"] = torch.full((1, 4, 2), 2.0)
        batch["inpaint_mask_latents"] = torch.tensor(
            [[[0.0], [0.25], [0.75], [1.0]]],
        )
        guidance = DifferentialDiffusionGuidance()
        return Run(
            plan=FlowSolver().plan([1.0, 0.5, 0.0]),
            ctx=StepContext(
                latents=batch["noisy_latents"].float(),
                generator=None,
                solver_state=None,
                guidance_state=guidance.init_state(),
            ),
            batch=batch,
            negative_batch=None,
        )

    def test_config_wraps_inner_guidance(self) -> None:
        sampler = Sampler.model_validate(
            {
                "guidance": {
                    "type": "differential",
                    "inner": {"type": "cfg", "scale": 2.0, "renorm": True},
                }
            }
        )

        guidance = sampler.guidance
        assert isinstance(guidance, DifferentialDiffusionGuidance)
        self.assertNotIsInstance(guidance, ClassifierFreeGuidance)
        inner = guidance.inner
        assert isinstance(inner, ClassifierFreeGuidance)
        self.assertTrue(guidance.needs_negative())
        self.assertEqual(inner.scale, 2.0)

        restored = Sampler.model_validate_json(sampler.model_dump_json())
        self.assertEqual(restored, sampler)

    def test_inpaint_mask_controls_reference_release_time(self) -> None:
        guidance = DifferentialDiffusionGuidance()
        run = self._make_run()

        first, first_state = guidance.prepare_transition(run, 0)
        # At sigma=1 the noised reference is exactly the fixed initial noise.
        torch.testing.assert_close(first, torch.full((1, 4, 2), 10.0))
        self.assertIsNone(first_state)

        run.ctx.latents = torch.full((1, 4, 2), 20.0)
        second, second_state = guidance.prepare_transition(run, 1)
        # At progress=1/2, edit strengths 0 and .25 still follow the
        # reference (which is 6 at sigma=.5); .75 and 1 have been released.
        torch.testing.assert_close(
            second,
            torch.tensor([[[6.0, 6.0], [6.0, 6.0], [20.0, 20.0], [20.0, 20.0]]]),
        )
        self.assertIsNone(second_state)

    def test_packed_mask_preserves_intra_token_spatial_detail(self) -> None:
        processor = InpaintProcessor.model_construct(
            vae_scale_factor=1,
            patch_size=2,
        )
        raw_mask = torch.tensor(
            [
                [
                    [
                        [0.0, 0.25, 1.0, 1.0],
                        [0.5, 0.75, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ]
        )

        luminance, packed = processor._prepare_inpaint_mask(raw_mask, (4, 4))

        self.assertEqual(tuple(luminance.shape), (1, 4, 4))
        self.assertEqual(tuple(packed.shape), (1, 4, 4))
        torch.testing.assert_close(packed[0, 0], torch.tensor([0.0, 0.25, 0.5, 0.75]))
        for multichannel in (
            raw_mask.repeat(1, 3, 1, 1),
            torch.cat([raw_mask, torch.zeros_like(raw_mask)], dim=1),
        ):
            normalized, packed_multichannel = processor._prepare_inpaint_mask(
                multichannel, (4, 4)
            )
            torch.testing.assert_close(normalized, luminance)
            torch.testing.assert_close(packed_multichannel, packed)

        run = self._make_run()
        run.ctx.latents = torch.zeros(1, 1, 8)
        batch: Any = run.batch
        batch["inpaint_mask_latents"] = packed[:, :1]
        expanded = DifferentialDiffusionGuidance._edit_strength(run, run.ctx.latents)
        torch.testing.assert_close(
            expanded,
            torch.tensor([[[0.0, 0.25, 0.5, 0.75] * 2]]),
        )

    def test_partial_denoise_uses_local_step_progress(self) -> None:
        run = self._make_run()
        run.plan = FlowSolver().plan([0.75, 0.5])
        run.ctx.latents = torch.full((1, 4, 2), 20.0)
        batch: Any = run.batch
        batch["inpaint_mask_latents"] = torch.tensor([[[0.0], [0.25], [0.75], [1.0]]])

        projected, _ = DifferentialDiffusionGuidance().prepare_transition(run, 0)

        # A sliced plan defines its own editing interval. At its first step,
        # every non-white region remains on the reference trajectory.
        torch.testing.assert_close(
            projected,
            torch.tensor([[[8.0, 8.0], [8.0, 8.0], [8.0, 8.0], [20.0, 20.0]]]),
        )

    def test_executor_feeds_projected_latents_to_the_model(self) -> None:
        class RecordingModel(microbatching.FakeSamplerModel):
            def __init__(self) -> None:
                super().__init__()
                self.latents: list[list[torch.Tensor]] = []

            def predict_velocity_batched(self, batches, timesteps):
                self.latents.append(
                    [batch["noisy_latents"].clone() for batch in batches]
                )
                return super().predict_velocity_batched(batches, timesteps)

        guidance = DifferentialDiffusionGuidance()
        sampler = Sampler(steps=2, guidance=guidance, solver=FlowSolver())
        runs = []
        for strength in (0.0, 1.0):
            batch: Any = microbatching.make_sampler_batch(0.0, initial=10.0)
            batch["inpaint_latents"] = torch.full((1, 1, 1), 2.0)
            batch["inpaint_mask_latents"] = torch.tensor([[[strength]]])
            runs.append(SampleRequest(batch=batch))

        model = RecordingModel()
        sampler.sample(model, runs)

        self.assertEqual(model.forward_batch_sizes, [2, 2])
        torch.testing.assert_close(model.latents[0][0], torch.tensor([[[10.0]]]))
        torch.testing.assert_close(model.latents[0][1], torch.tensor([[[10.0]]]))
        torch.testing.assert_close(model.latents[1][0], torch.tensor([[[6.0]]]))
        torch.testing.assert_close(model.latents[1][1], torch.tensor([[[10.0]]]))

    def test_all_white_mask_preserves_multistep_solver(self) -> None:
        batch: Any = microbatching.make_batch(0.9, tokens=1)
        batch["inpaint_latents"] = torch.full((1, 1, 2), 0.2)
        batch["inpaint_mask_latents"] = torch.ones(1, 1, 1)
        model = microbatching.FakeDenseAdapter.model_construct(
            arch="fake",
            type="fake",
        )

        plain = Sampler(
            steps=4,
            guidance=ClassifierFreeGuidance(),
            solver=DPMSolver(order=2),
        ).sample(model, [SampleRequest(batch=batch)])[0]
        differential = Sampler(
            steps=4,
            guidance=DifferentialDiffusionGuidance(),
            solver=DPMSolver(order=2),
        ).sample(model, [SampleRequest(batch=batch)])[0]

        torch.testing.assert_close(differential.final_latents, plain.final_latents)

    def test_recorded_projection_gap_remains_independently_replayable(self) -> None:
        guidance = DifferentialDiffusionGuidance()
        sampler = Sampler(
            steps=3,
            guidance=guidance,
            solver=FlowSolver(eta=0.5),
        )
        batch: Any = microbatching.make_sampler_batch(0.0, initial=0.9)
        batch["inpaint_latents"] = torch.full((1, 1, 1), 0.2)
        batch["inpaint_mask_latents"] = torch.zeros(1, 1, 1)
        phases = PhasesRecipe(
            phases=[PhaseConfig(transforms=[SdeWindow(record=True)])]
        ).build(
            RecipeBuildContext(
                default_sampler=sampler,
                batches={"main": batch},
                negative_batch_for=lambda name: None,
                generator=torch.Generator().manual_seed(5),
            )
        )

        output = run_phases(microbatching.FakeSamplerModel(), [phases])[0]
        trajectory = output.trajectory
        assert trajectory is not None
        self.assertEqual(len(trajectory), 2)
        self.assertFalse(torch.equal(trajectory[0].latent_next, trajectory[1].latent_t))

        replayed = sampler.replay_recorded_steps(
            microbatching.FakeSamplerModel(),
            [ReplayItem(batch=batch, recorded=step) for step in trajectory],
        )
        for recorded, replay in zip(trajectory, replayed, strict=True):
            torch.testing.assert_close(replay.log_prob, recorded.log_prob)

    def test_combine_delegates_to_inner_guidance(self) -> None:
        guidance = DifferentialDiffusionGuidance(
            inner=ClassifierFreeGuidance(scale=2.0)
        )
        latents = torch.zeros(1, 1, 1)
        ctx = StepContext(
            latents=latents,
            generator=None,
            solver_state=None,
            guidance_state=None,
        )
        cond = torch.full_like(latents, 3.0)
        uncond = torch.full_like(latents, 1.0)
        output, state = guidance.combine(
            BranchEvals(
                cond=cond,
                uncond=uncond,
                latents=latents,
                sigma=0.5,
            ),
            ctx,
            None,
        )
        torch.testing.assert_close(output.velocity, torch.full_like(latents, 5.0))
        self.assertIsNone(state)


class MomentumGuidanceTest(unittest.TestCase):
    def test_legacy_formula_uses_functional_per_run_state(self) -> None:
        guidance = MomentumGuidance(alpha=0.5, beta=0.25)
        latents = torch.zeros(1, 1, 1)
        ctx = StepContext(
            latents=latents,
            generator=None,
            solver_state=None,
            guidance_state=guidance.init_state(),
        )

        first_output, first_state = guidance.combine(
            BranchEvals(
                cond=torch.full_like(latents, 2.0),
                uncond=None,
                latents=latents,
                sigma=1.0,
            ),
            ctx,
            ctx.guidance_state,
        )
        second_output, second_state = guidance.combine(
            BranchEvals(
                cond=torch.full_like(latents, 4.0),
                uncond=None,
                latents=latents,
                sigma=0.5,
            ),
            ctx,
            first_state,
        )

        torch.testing.assert_close(first_output.velocity, torch.full_like(latents, 2.0))
        torch.testing.assert_close(
            second_output.velocity, torch.full_like(latents, 5.0)
        )
        assert isinstance(first_state, MomentumGuidanceState)
        assert isinstance(second_state, MomentumGuidanceState)
        torch.testing.assert_close(first_state.momentum, torch.full_like(latents, 2.0))
        torch.testing.assert_close(second_state.momentum, torch.full_like(latents, 3.5))
        fresh_state = guidance.init_state()
        assert isinstance(fresh_state, MomentumGuidanceState)
        self.assertIsNone(fresh_state.momentum)

    def test_recorded_momentum_states_replay_without_mutation(self) -> None:
        guidance = MomentumGuidance(alpha=0.3, beta=0.5)
        sampler = Sampler(steps=3, guidance=guidance, solver=FlowSolver(eta=0.5))
        batch = microbatching.make_sampler_batch(1.0)
        phases = PhasesRecipe(
            phases=[PhaseConfig(transforms=[SdeWindow(record=True)])]
        ).build(
            RecipeBuildContext(
                default_sampler=sampler,
                batches={"main": batch},
                negative_batch_for=lambda name: None,
                generator=torch.Generator().manual_seed(5),
            )
        )

        output = run_phases(microbatching.FakeSamplerModel(), [phases])[0]
        trajectory = output.trajectory
        assert trajectory is not None
        first_state = trajectory[0].guidance_state
        second_state = trajectory[1].guidance_state
        assert isinstance(first_state, MomentumGuidanceState)
        assert isinstance(second_state, MomentumGuidanceState)
        self.assertIsNone(first_state.momentum)
        self.assertIsNotNone(second_state.momentum)

        replayed = sampler.replay_recorded_steps(
            microbatching.FakeSamplerModel(),
            [ReplayItem(batch=batch, recorded=step) for step in trajectory],
        )
        for step, replay in zip(trajectory, replayed, strict=True):
            torch.testing.assert_close(replay.log_prob, step.log_prob)


class GuidanceStateTimingTest(unittest.TestCase):
    def test_prepare_runs_once_per_transition_not_once_per_eval(self) -> None:
        guidance = _PreparingCountingGuidance()
        solver = SASolver(eta=0.0)
        plan = finalize_replay_state(solver.plan(torch.linspace(1.0, 0.0, 5).tolist()))
        batch = microbatching.make_sampler_batch(1.0)
        run = Run(
            plan=plan,
            ctx=StepContext(
                latents=batch["noisy_latents"].float(),
                generator=None,
                solver_state=None,
                guidance_state=guidance.init_state(),
            ),
            batch=batch,
            negative_batch=None,
        )

        list(execute(microbatching.FakeSamplerModel(), [run], guidance))

        self.assertEqual(guidance._prepared, list(range(len(plan))))
        # SA evaluates four times across the same four transitions, but the
        # first transition alone accounts for two of those evaluations.
        self.assertEqual(guidance._seen, [0, 1, 2, 3])

    def test_state_advances_per_eval_including_mid_transition(self) -> None:
        # SA's first PEC transition evaluates twice (seeding eval + predicted
        # point); the counter must advance at the mid-transition eval too.
        guidance = _CountingGuidance()
        solver = SASolver(eta=0.0)
        plan = finalize_replay_state(solver.plan(torch.linspace(1.0, 0.0, 5).tolist()))
        batch = microbatching.make_sampler_batch(1.0)
        run = Run(
            plan=plan,
            ctx=StepContext(
                latents=batch["noisy_latents"].float(),
                generator=None,
                solver_state=None,
                guidance_state=guidance.init_state(),
            ),
            batch=batch,
            negative_batch=None,
        )
        events = list(execute(microbatching.FakeSamplerModel(), [run], guidance))
        self.assertEqual(len(events), len(plan))
        # 4 transitions -> 2 + 1 + 1 + 0 evals; [0, 1] within the first
        # transition proves per-eval (not per-transition) advancement.
        self.assertEqual(guidance._seen, [0, 1, 2, 3])
        state = run.ctx.guidance_state
        assert isinstance(state, _CounterState)
        self.assertEqual(state.count, 4)

    def test_recorded_step_carries_pre_first_eval_state(self) -> None:
        # Both single-eval replayable solvers, including flash (which builds
        # its RecordedStep with a plan-compiled replay descriptor).
        for solver in (FlowSolver(eta=0.5), FlashSolver(eta=1.0)):
            with self.subTest(solver=solver.type):
                guidance = _CountingGuidance()
                sampler = Sampler(steps=3, guidance=guidance, solver=solver)
                recipe = PhasesRecipe(
                    phases=[PhaseConfig(transforms=[SdeWindow(record=True)])]
                )
                phases = recipe.build(
                    RecipeBuildContext(
                        default_sampler=sampler,
                        batches={"main": microbatching.make_sampler_batch(1.0)},
                        negative_batch_for=lambda name: None,
                        generator=torch.Generator().manual_seed(5),
                    )
                )
                output = run_phases(microbatching.FakeSamplerModel(), [phases])[0]
                trajectory = output.trajectory
                assert trajectory is not None
                # The default window excludes the terminal transition.
                self.assertEqual(len(trajectory), 2)
                counts: list[int] = []
                for step in trajectory:
                    state = step.guidance_state
                    assert isinstance(state, _CounterState)
                    counts.append(state.count)
                # Transitions evaluate once each: the state recorded for step
                # i must be the pre-first-eval value i, not post-eval i + 1.
                self.assertEqual(counts, [0, 1])

    def test_replay_feeds_recorded_pre_eval_state(self) -> None:
        guidance = _CountingGuidance()
        sampler = Sampler(guidance=guidance, solver=FlowSolver(eta=0.5))
        item = ReplayItem(
            batch=microbatching.make_sampler_batch(0.0),
            recorded=RecordedStep(
                latent_t=torch.tensor([[[0.2]]]),
                latent_next=torch.tensor([[[0.1]]]),
                log_prob=torch.zeros(1),
                replay=FlowReplayStep(sigma=0.8, sigma_next=0.6, eta=0.5),
                guidance_state=_CounterState(count=7),
            ),
        )
        sampler.replay_recorded_steps(
            microbatching.FakeSamplerModel(),
            [item],
        )
        # The recorded pre-eval state (not a fresh init_state) reaches combine.
        self.assertEqual(guidance._seen, [7])


if __name__ == "__main__":
    unittest.main()
