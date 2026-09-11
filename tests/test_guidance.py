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
from flow_control.samplers import Executor, SampleRun
from flow_control.samplers.guidance import (
    BaseGuidance,
    BranchSpec,
    ClassifierFreeGuidance,
)
from flow_control.samplers.plan import (
    BranchEvals,
    GuidanceOutput,
    GuidanceState,
    StepContext,
    euler_step,
)
from flow_control.samplers.projectors import DifferentialDiffusion
from flow_control.samplers.sampler import Sampler, SampleRequest
from flow_control.samplers.solver import (
    DPMSolver,
    FlowSolver,
    SASolver,
)
from flow_control.training.grpo_sampling import (
    GrpoCollector,
    ReplayItem,
    replay_steps,
)

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
    _indices: list[int] = PrivateAttr(default_factory=list)

    def init_state(self) -> GuidanceState | None:
        return _CounterState(count=0)

    def branches(self, item_index: int) -> list[BranchSpec]:
        return [BranchSpec("cond")]

    def combine(
        self,
        evals: BranchEvals,
        ctx: StepContext,
        state: GuidanceState | None,
    ) -> tuple[GuidanceOutput, GuidanceState | None]:
        assert isinstance(state, _CounterState)
        self._seen.append(state.count)
        self._indices.append(ctx.item_index)
        return (
            GuidanceOutput(velocity=evals.velocities["cond"]),
            _CounterState(count=state.count + 1),
        )


class GuidedVelocityCompatibilityTest(unittest.TestCase):
    def test_guided_velocity_matches_sampling_path(self) -> None:
        # Training re-evaluates a plan item through SampleRun.guided_velocity;
        # it must be the very velocity the sampling loop stepped with.
        sampler = Sampler(
            steps=1,
            guidance=ClassifierFreeGuidance(scale=2.0),
            solver=FlowSolver(),
        )
        cond = microbatching.make_sampler_batch(3.0)
        negative = microbatching.make_sampler_batch(1.0)
        request = SampleRequest(batch=cond, negative_batch=negative)

        run = sampler.make_run(request, plan=sampler.plan(cond))
        velocity = Executor(
            microbatching.FakeSamplerModel(), sampler.variant_keys()
        ).evaluate([run.guided_velocity(cond["noisy_latents"], 0)])[0]
        torch.testing.assert_close(velocity, torch.tensor([[[5.0]]]))

        sampled = next(
            iter(sampler.sample(microbatching.FakeSamplerModel(), [request]))
        )
        # The single Euler step must use the same guided velocity.
        torch.testing.assert_close(
            sampled.ctx.latents,
            euler_step(cond["noisy_latents"], velocity, 1.0, 0.0),
        )


class DifferentialDiffusionTest(unittest.TestCase):
    @staticmethod
    def _make_run() -> SampleRun:
        batch: Any = microbatching.make_batch(
            10.0,
            tokens=4,
            image_size=(32, 32),
        )
        batch["inpaint_latents"] = torch.full((1, 4, 2), 2.0)
        batch["inpaint_mask_latents"] = torch.tensor(
            [[[0.0], [0.25], [0.75], [1.0]]],
        )
        return SampleRun(
            sampler=Sampler(),
            batch=batch,
            negative_batch=None,
            plan=FlowSolver().plan([1.0, 0.5, 0.0]),
            ctx=StepContext(
                latents=batch["noisy_latents"].float(),
                generator=None,
                solver_state=None,
                guidance_state=None,
                num_items=2,
            ),
        )

    def test_config_wraps_inner_guidance(self) -> None:
        sampler = Sampler.model_validate(
            {
                "guidance": {"type": "cfg", "scale": 2.0, "renorm": True},
                "projectors": [{"type": "differential"}],
            }
        )
        restored = Sampler.model_validate_json(sampler.model_dump_json())
        self.assertEqual(restored, sampler)
        self.assertIsInstance(restored.projectors[0], DifferentialDiffusion)
        self.assertTrue(restored.guidance.requires_negative(restored.steps))

    def test_inpaint_mask_controls_reference_release_time(self) -> None:
        guidance = DifferentialDiffusion()
        run = self._make_run()

        first = guidance.pre_transition(
            run.ctx.latents, run.batch, run.ctx, run.plan[0]
        )
        # At sigma=1 the noised reference is exactly the fixed initial noise.
        torch.testing.assert_close(first, torch.full((1, 4, 2), 10.0))

        run.ctx.latents = torch.full((1, 4, 2), 20.0)
        run.ctx.item_index = 1
        second = guidance.pre_transition(
            run.ctx.latents, run.batch, run.ctx, run.plan[1]
        )
        # At progress=1/2, edit strengths 0 and .25 still follow the
        # reference (which is 6 at sigma=.5); .75 and 1 have been released.
        torch.testing.assert_close(
            second,
            torch.tensor([[[6.0, 6.0], [6.0, 6.0], [20.0, 20.0], [20.0, 20.0]]]),
        )

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
        expanded = DifferentialDiffusion._edit_strength(run.batch, run.ctx.latents)
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

        run.ctx.num_items = len(run.plan)
        projected = DifferentialDiffusion().pre_transition(
            run.ctx.latents, run.batch, run.ctx, run.plan[0]
        )

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

            def predict_velocity_batched(
                self, batches, timesteps, *, dummy_outputs=None
            ):
                self.latents.append(
                    [batch["noisy_latents"].clone() for batch in batches]
                )
                return super().predict_velocity_batched(
                    batches, timesteps, dummy_outputs=dummy_outputs
                )

        guidance = DifferentialDiffusion()
        sampler = Sampler(steps=2, projectors=[guidance], solver=FlowSolver())
        runs = []
        for strength in (0.0, 1.0):
            batch: Any = microbatching.make_sampler_batch(0.0, initial=10.0)
            batch["inpaint_latents"] = torch.full((1, 1, 1), 2.0)
            batch["inpaint_mask_latents"] = torch.tensor([[[strength]]])
            runs.append(SampleRequest(batch=batch))

        model = RecordingModel()
        list(sampler.sample(model, runs))

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

        plain = next(
            iter(
                Sampler(
                    steps=4,
                    guidance=ClassifierFreeGuidance(),
                    solver=DPMSolver(order=2),
                ).sample(model, [SampleRequest(batch=batch)])
            )
        )
        differential = next(
            iter(
                Sampler(
                    steps=4,
                    projectors=[DifferentialDiffusion()],
                    solver=DPMSolver(order=2),
                ).sample(model, [SampleRequest(batch=batch)])
            )
        )

        torch.testing.assert_close(differential.ctx.latents, plain.ctx.latents)

    def test_recorded_projection_gap_remains_independently_replayable(self) -> None:
        guidance = DifferentialDiffusion()
        sampler = Sampler(
            steps=3,
            projectors=[guidance],
            solver=FlowSolver(eta=0.5),
        )
        batch: Any = microbatching.make_sampler_batch(0.0, initial=0.9)
        batch["inpaint_latents"] = torch.full((1, 1, 1), 0.2)
        batch["inpaint_mask_latents"] = torch.zeros(1, 1, 1)
        collector = GrpoCollector(sampler)
        run = next(
            iter(
                sampler.sample(
                    microbatching.FakeSamplerModel(),
                    [
                        SampleRequest(
                            batch=batch, generator=torch.Generator().manual_seed(5)
                        )
                    ],
                    collector=collector,
                )
            )
        )
        trajectory = collector.take(run)
        self.assertEqual(len(trajectory), 2)
        self.assertFalse(torch.equal(trajectory[0].latent_next, trajectory[1].latent_t))

        replayed = replay_steps(
            microbatching.FakeSamplerModel(),
            [ReplayItem(run, step) for step in trajectory],
        )
        for recorded, replay in zip(trajectory, replayed, strict=True):
            torch.testing.assert_close(replay.log_prob, recorded.log_prob)


class GuidanceStateTimingTest(unittest.TestCase):
    def test_state_advances_per_eval_including_mid_transition(self) -> None:
        # SA's first PEC transition evaluates twice (seeding eval + predicted
        # point); the counter must advance at the mid-transition eval too.
        guidance = _CountingGuidance()
        sampler = Sampler(steps=4, solver=SASolver(eta=0.0), guidance=guidance)
        batch = microbatching.make_sampler_batch(1.0)
        run = next(
            iter(
                sampler.sample(microbatching.FakeSamplerModel(), [SampleRequest(batch)])
            )
        )
        self.assertEqual(len(run.plan), 4)
        # 4 transitions -> 2 + 1 + 1 + 0 evals; [0, 1] within the first
        # transition proves per-eval (not per-transition) advancement.
        self.assertEqual(guidance._seen, [0, 1, 2, 3])
        state = run.ctx.guidance_state
        assert isinstance(state, _CounterState)
        self.assertEqual(state.count, 4)

        # S2 review reproduced [2, 2, 2] here: priming multiple evaluations
        # of one run must not overwrite earlier contexts or reuse its state.
        guidance._seen.clear()
        guidance._indices.clear()
        saved_index, saved_latents = run.ctx.item_index, run.ctx.latents
        for _ in range(2):
            Executor(microbatching.FakeSamplerModel()).evaluate(
                [run.guided_velocity(batch["noisy_latents"], i) for i in range(3)]
            )
        self.assertEqual(guidance._indices, [0, 1, 2] * 2)
        self.assertEqual(guidance._seen, [0] * 6)
        self.assertEqual(run.ctx.item_index, saved_index)
        self.assertIs(run.ctx.latents, saved_latents)
        self.assertIs(run.ctx.guidance_state, state)
