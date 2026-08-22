"""Guidance-axis tests for config, execution, and replay semantics."""

import importlib
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from pydantic import PrivateAttr, ValidationError

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
from flow_control.samplers.solver import FlashSolver, FlowSolver, SASolver
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


class GuidanceConfigTest(unittest.TestCase):
    def test_removed_guidance_keys_are_rejected(self) -> None:
        removed = {
            "cfg_scale": 4.5,
            "enable_cfg_renorm": True,
            "cfg_renorm_eps": 1e-6,
            "cfg_renorm_min": 0.2,
        }
        for key, value in removed.items():
            with self.subTest(key=key), self.assertRaises(ValidationError) as error:
                Sampler.model_validate({key: value})
            self.assertEqual(error.exception.errors()[0]["loc"], (key,))
            self.assertEqual(error.exception.errors()[0]["type"], "extra_forbidden")

    def test_momentum_guidance_registry_config(self) -> None:
        sampler = Sampler.model_validate(
            {
                "guidance": {
                    "type": "momentum",
                    "scale": 2.0,
                    "alpha": 0.3,
                    "beta": 0.7,
                }
            }
        )

        guidance = sampler.guidance
        assert isinstance(guidance, MomentumGuidance)
        self.assertEqual(guidance.scale, 2.0)
        self.assertEqual(guidance.alpha, 0.3)
        self.assertEqual(guidance.beta, 0.7)


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
