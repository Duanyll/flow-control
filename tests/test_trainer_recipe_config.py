"""Trainer config, rollout metadata and GRPO replay integration tests."""

import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch
from pydantic import BaseModel, Field
from test_microbatching import FakeSamplerModel, make_sampler_batch

from flow_control.contrib.momentum_guidance import MomentumGuidance
from flow_control.processors import parse_processor
from flow_control.samplers import (
    CfgPlusPlusGuidance,
    ClassifierFreeGuidance,
    ModelPrediction,
    Sampler,
    SampleRequest,
    SdeWindow,
    TiledPrediction,
)
from flow_control.samplers.solver import FlowSolver
from flow_control.training.awm import AwmTrainer
from flow_control.training.endpoint import EndpointTrainItem
from flow_control.training.grpo import GrpoTrainer
from flow_control.training.grpo_sampling import (
    GrpoCollector,
    replay_steps,
    step_log_prob,
)
from flow_control.training.mixins import Rollout
from flow_control.training.nft import NftObjective, NftTrainer
from flow_control.training.ram import RamTrainer
from flow_control.training.sft import SftTrainer
from flow_control.training.weighting import LogitNormalTimestepWeighting


class _ProbeOverrides(BaseModel):
    """Defaults for the heavy required trainer fields (config-only tests)."""

    model: Any = None
    processor: Any = None
    reward: Any = None
    dataset: Any = None
    launch: Any = None
    checkpoint_root: str = ""
    experiment_name: str = "probe"
    seed_checkpoint_dir: str = ""
    num_prompts_per_epoch: int = 1
    num_rollouts_per_prompt: int = 1
    rollout_sampler: Sampler = Field(default_factory=Sampler)
    validation_sampler: Sampler = Field(default_factory=Sampler)

    @property
    def device(self):
        return torch.device("cpu")


class _GrpoProbe(_ProbeOverrides, GrpoTrainer):
    pass


class _NftProbe(_ProbeOverrides, NftTrainer):
    pass


class _SftProbe(_ProbeOverrides, SftTrainer):
    pass


class _AwmProbe(_ProbeOverrides, AwmTrainer):
    pass


class _RamProbe(_ProbeOverrides, RamTrainer):
    pass


class _ConditionModel(FakeSamplerModel):
    def __init__(self):
        super().__init__()
        self.gain = torch.tensor(1.0, requires_grad=True)

    def predict_velocity_batched(self, batches, timesteps, *, dummy_outputs=None):
        self.forward_batch_sizes.append(len(batches))
        return [
            self.gain * (b["prompt_embeds"] + 0.1 * b["noisy_latents"]) for b in batches
        ]


def _check_independent_predictions(case: unittest.TestCase) -> None:
    """The removed conditional_velocity silently forced tiling and excluded CFG."""
    processor = parse_processor({"task": "t2i", "preset": "flux1"})
    batch: Any = {
        "image_size": (2, 2),
        "clean_latents": torch.zeros(1, 4, 1),
        "noisy_latents": torch.zeros(1, 4, 1, dtype=torch.bfloat16),
        "prompt_embeds": torch.tensor([[[3.0]]]),
        "negative": {"prompt_embeds": torch.tensor([[[1.0]]])},
        "tiling": {"tile_size": 1, "overlap": 0, "stride": 1},
        "tiles": [{"prompt_embeds": torch.tensor([[[float(i)]]])} for i in range(1, 5)],
    }
    configs = [
        ("model", torch.full((1, 4, 1), 3.0), [1]),
        ("tiled", torch.arange(1.0, 5.0).reshape(1, 4, 1), [4]),
        (
            {"type": "cfg", "scale": 3, "inner": "model"},
            torch.full((1, 4, 1), 7.0),
            [2],
        ),
    ]
    for cls in (_SftProbe, _AwmProbe, _RamProbe):
        with case.assertRaisesRegex(ValueError, "train_predictor"):
            cls.model_validate({})
        for config, expected, calls in configs:
            trainer = cls.model_validate({"train_predictor": config})
            trainer.processor, trainer.model = processor, _ConditionModel()
            # Auxiliary passes use the same tree even under no_grad.
            with torch.no_grad():
                (aux,) = trainer.predict_training([batch], [torch.tensor([0.5])])
            (prediction,) = trainer.predict_training([batch], [torch.tensor([0.5])])
            case.assertEqual(prediction.dtype, torch.float32)
            torch.testing.assert_close(aux, expected)
            torch.testing.assert_close(prediction, expected)
            torch.testing.assert_close(
                torch.autograd.grad(prediction.sum(), trainer.model.gain)[0],
                expected.sum(),
            )
            case.assertEqual(trainer.model.forward_batch_sizes, calls * 2)

    # Dropout changes the positive condition. Keep the original negative for
    # train CFG as well; asking the dropped batch for its overlay loses it.
    trainer = _SftProbe.model_validate(
        {"train_predictor": configs[-1][0], "cfg_drop_prob": 1}
    )
    trainer.processor, trainer.model = processor, _ConditionModel()
    with (
        patch.object(
            LogitNormalTimestepWeighting,
            "sample_timesteps",
            return_value=torch.tensor([0.5]),
        ),
        patch("torch.randn_like", return_value=torch.ones(1, 4, 1)),
    ):
        loss = trainer.train_step([batch])
    torch.testing.assert_close(loss, torch.tensor(0.05**2))
    case.assertEqual(trainer.model.forward_batch_sizes, [2])
    missing_negative: Any = {k: v for k, v in batch.items() if k != "negative"}
    with case.assertRaisesRegex(ValueError, "negative condition"):
        trainer.predict_training([missing_negative], [torch.tensor([0.5])])
    trainer.train_predictor = CfgPlusPlusGuidance(scale=3, inner=ModelPrediction())
    with case.assertRaisesRegex(ValueError, "transition with sigma_next"):
        trainer.predict_training([batch], [torch.tensor([0.5])])


class TrainerRolloutPlanTest(unittest.TestCase):
    """How trainers consume the executed sampling plan: GRPO collection needs a
    stochastic step to record, NFT trains on the grid the rollout actually ran
    (including per-step eta and CFG++), and SDE windows index the sliced plan."""

    BATCH: Any = {
        "image_size": (32, 32),
        "clean_latents": torch.zeros(1, 1, 1),
        "noisy_latents": torch.zeros(1, 1, 1),
    }

    def test_grpo_collection_requires_a_stochastic_step(self) -> None:
        from test_microbatching import FakeSamplerModel

        for eta in (0.0, 0.7):
            trainer = _GrpoProbe.model_validate(
                {
                    "train_predictor": "model",
                    "rollout_sampler": {
                        "steps": 4,
                        "solver": {"type": "flow", "eta": eta},
                    },
                }
            )
            collector = GrpoCollector(trainer.rollout_sampler)
            run = next(
                iter(
                    trainer.rollout_sampler.sample(
                        FakeSamplerModel(),
                        [SampleRequest(row=self.BATCH)],
                        collector=collector,
                    )
                )
            )
            if eta == 0:
                with self.assertRaisesRegex(ValueError, "stochastic"):
                    collector.take(run)
            else:
                self.assertEqual(len(collector.take(run)), 3)

        with self.subTest("sde_window indexes the sliced plan"):
            # steps=10 sliced at strength 0.45 leaves 4 transitions; range/size
            # pin the window to slice indices 1-2 (on the full grid these would
            # be sigmas 0.9/0.8, outside the slice, and nothing would record).
            sliced = Sampler.model_validate(
                {
                    "steps": 10,
                    "solver": {"type": "flow", "eta": 0.7},
                    "start": {"strength": 0.45},
                    "transforms": [{"type": "sde_window", "size": 2, "range": [1, 3]}],
                }
            )
            self.assertEqual(
                [item.eta for item in sliced.plan(self.BATCH)], [0.0, 0.7, 0.7, 0.0]
            )

        # A stateful behavior tree is legal: the actual rollout score is already
        # recorded. Test step 1, after Momentum has a history, with a separate
        # stateless training tree and an unguided rollout missing negative_row.
        for rollout_scale, train_scale in ((1, 3), (3, 1)):
            model = _ConditionModel()
            batch: Any = {
                **make_sampler_batch(0, 0.9),
                "prompt_embeds": torch.tensor([[[3.0]]]),
                "negative": {"prompt_embeds": torch.tensor([[[1.0]]])},
            }
            processor = parse_processor({"task": "t2i", "preset": "flux1"})
            negative: Any = (
                processor.get_negative_row(batch) if rollout_scale > 1 else None
            )
            sampler = Sampler(
                steps=4,
                solver=FlowSolver(eta=0.7),
                transforms=[SdeWindow(size=2, range=(1, 3))],
                guidance=MomentumGuidance(
                    alpha=0.5,
                    beta=0.25,
                    inner=ClassifierFreeGuidance(scale=rollout_scale),
                ),
            )
            collector = GrpoCollector(sampler)
            with torch.no_grad():
                (run,) = sampler.sample(
                    model,
                    [SampleRequest(batch, negative, torch.Generator().manual_seed(5))],
                    collector=collector,
                )
            records = collector.take(run)
            self.assertEqual(records[0].item_index, 1)
            old = records[0].log_prob.clone()
            raw_initial = 1 + 2 * rollout_scale + 0.1 * batch["noisy_latents"]
            raw_next = 1 + 2 * rollout_scale + 0.1 * records[0].latent_t
            behavior = step_log_prob(records[0], 1.5 * raw_next - 0.5 * raw_initial)
            torch.testing.assert_close(old, behavior.log_prob)
            rollout = Rollout(
                run.plan,
                torch.zeros(1),
                torch.zeros(1),
                torch.ones(1),
                ["reward"],
                "probe",
                batch,
                negative,
                records,
            )
            trainer = _GrpoProbe.model_validate(
                {"train_predictor": {"type": "cfg", "scale": train_scale}}
            )
            trainer.model, trainer.processor, trainer.rollout_sampler = (
                model,
                processor,
                sampler,
            )
            replay, denominator = trainer._make_replay_item(rollout, 0)
            (output,) = replay_steps(model, [replay])
            target_velocity = 1 + 2 * train_scale + 0.1 * records[0].latent_t
            expected = step_log_prob(records[0], target_velocity)
            torch.testing.assert_close(output.log_prob, expected.log_prob)
            torch.testing.assert_close(denominator, old, rtol=0, atol=0)
            torch.testing.assert_close(records[0].log_prob, old, rtol=0, atol=0)
            self.assertFalse(torch.allclose(output.log_prob.detach(), old))
            self.assertIs(replay.run.predictor, trainer.train_predictor)

        for cls in (_GrpoProbe, _NftProbe):
            with self.assertRaisesRegex(ValueError, "stateless"):
                cls.model_validate(
                    {
                        "train_predictor": TiledPrediction(
                            inner=MomentumGuidance(
                                alpha=0.5, beta=0.25, inner=ModelPrediction()
                            )
                        )
                    }
                )

    def test_nft_trains_on_the_executed_rollout_plan(self) -> None:
        _check_independent_predictions(self)
        trainer = _NftProbe.model_validate(
            {"num_inner_epochs": 2, "train_predictor": "model"}
        )
        sigmas = [0.9, 0.6, 0.3]
        rollout = Rollout(
            sampling_plan=FlowSolver().plan([*sigmas, 0.0]),
            reward=torch.zeros(1),
            raw_reward=torch.zeros(1),
            reward_weights=torch.ones(1),
            reward_labels=["reward"],
            key="sample",
            row=self.BATCH,
            negative_row=None,
        )

        plan = trainer._build_train_plan([rollout])

        self.assertEqual(len(plan), 2)
        for epoch in plan:
            self.assertEqual(len(epoch), len(sigmas))
            for item in epoch:
                self.assertIsInstance(item, EndpointTrainItem)
                self.assertIsInstance(item.sigma, float)
                assert item.grid_index is not None
                self.assertEqual(item.sigma, sigmas[item.grid_index])

        # The R3/R4 observer migration initially left ordinary NFT rollouts
        # without a plan, so per-step variants and CFG++ failed during training.
        # Exercise real collection with only reward/decode/logging stubbed out.
        from test_microbatching import FakeSamplerModel, make_sampler_batch

        from flow_control.data import Index, IndexEntry, RowCursor

        model: Any = FakeSamplerModel()
        model.transformer = torch.nn.Identity()
        trainer.model = model
        trainer.rollout_sampler = Sampler.model_validate(
            {
                "steps": 4,
                "solver": {"type": "flow", "eta": 0.7},
                "guidance": {"type": "cfg_pp", "scale": 2.0},
                "transforms": [{"type": "sde_window", "size": 1, "range": [1, 3]}],
            }
        )
        trainer.train_predictor = trainer.rollout_sampler.guidance
        trainer.processor = SimpleNamespace(
            initialize_latents=lambda row, **kwargs: None,
            decode_output=lambda latents, row: {},
            get_negative_row=lambda row: make_sampler_batch(-0.2),
            resample=lambda row, generator: row,
        )

        class OneRowStore:
            row = {**make_sampler_batch(0.3, 0.9), "key": "sample"}
            index = Index([IndexEntry("sample", 1, "", None, "sample")], {})

            def __len__(self):
                return 1

            def get(self, row_id):
                return dict(self.row)

        store: Any = OneRowStore()
        trainer._store = store
        trainer._cursor = RowCursor(
            store, trainer.make_planner(store, shuffle=True), trainer.seed
        )

        def drain(reward, submitter, handler, profile):
            list(submitter)

        with (
            patch(
                "flow_control.training.mixins.rollout.execute_reward", side_effect=drain
            ),
            patch.object(_NftProbe, "log_progress_timing"),
            patch.object(_NftProbe, "log_reduced_metrics"),
        ):
            collected = trainer._collect_rollouts(0)[0]
        self.assertIsNone(collected.recorded_steps)
        self.assertEqual(len(collected.sampling_plan), 4)
        self.assertEqual(sum(step.eta > 0 for step in collected.sampling_plan), 1)
        item = EndpointTrainItem(
            rollout_idx=0, sigma=collected.sampling_plan[1].sigma, grid_index=1
        )
        (prediction,) = trainer._predict(
            trainer._prepare([item], [collected], torch.zeros(1))
        )
        self.assertTrue(torch.isfinite(prediction).all())

        # S2 precision review: teacher-cache scalar products ran in bf16,
        # while .double() in normalization made the final NFT loss fp64. The
        # cached noise / old / ref tensors must be promoted to fp32 before the
        # objective sees them, so a bf16 cache trains bitwise like an fp32 one.
        trainer.objective = NftObjective(beta=0.3, kl_beta=0.2)
        trainer.model = _ConditionModel()
        conditioned: Any = collected.row
        conditioned["prompt_embeds"] = torch.tensor([[[3.0]]])
        negative_conditioned: Any = collected.negative_row
        negative_conditioned["prompt_embeds"] = torch.tensor([[[1.0]]])
        cached = EndpointTrainItem(
            rollout_idx=0,
            sigma=0.317,
            grid_index=1,
            noise=torch.full((1, 1, 1), 0.734, dtype=torch.bfloat16),
            cache={
                "old": torch.full((1, 1, 1), 0.121, dtype=torch.bfloat16),
                "ref": torch.full((1, 1, 1), -0.219, dtype=torch.bfloat16),
            },
        )
        assert cached.noise is not None
        promoted = EndpointTrainItem(
            rollout_idx=0,
            sigma=0.317,
            grid_index=1,
            noise=cached.noise.float(),
            cache={role: value.float() for role, value in cached.cache.items()},
        )
        loss_values = []
        gradients = []
        with patch.object(_NftProbe, "log_aggregated_metrics"):
            for targets in (cached, promoted):
                loss = trainer._loss_batched(
                    [targets], [collected], torch.tensor([0.23], dtype=torch.bfloat16)
                )
                self.assertEqual(loss.dtype, torch.float32)
                loss_values.append(loss.detach())
                gradients.append(torch.autograd.grad(loss, trainer.model.gain)[0])
        torch.testing.assert_close(*loss_values, rtol=0, atol=0)
        torch.testing.assert_close(*gradients, rtol=0, atol=0)
        self.assertEqual(trainer.model.forward_batch_sizes, [2, 2])

    def test_nft_timestep_window_is_index_based(self) -> None:
        # NFT's sigma-threshold ``timestep_range`` made the per-rank item count
        # depend on the rollout's sigma values (resolution-dependent shift);
        # the grid window keeps the noisiest index fraction instead.
        trainer = _NftProbe.model_validate(
            {
                "train_timesteps": {"type": "grid", "window": 0.3},
                "train_predictor": "model",
            }
        )
        uniform = [1.0 - i / 10 for i in range(10)]
        shifted = [s / (s + (1 - s) / 3) for s in uniform]
        for sigmas, expected in (
            ([0.7, 0.699, 0.8], [0]),
            ([0.95, 0.5, 0.05], [0]),
            (uniform, [0, 1, 2]),
            (shifted, [0, 1, 2]),
        ):
            drawn = trainer.train_timesteps.draw(sigmas)
            self.assertEqual(sorted(i for _, i in drawn if i is not None), expected)
            self.assertEqual(
                sorted(sigma for sigma, _ in drawn), sorted(sigmas[i] for i in expected)
            )


if __name__ == "__main__":
    unittest.main()
