"""Trainer config, rollout metadata and GRPO replay integration tests."""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch
from pydantic import BaseModel, Field

from flow_control.samplers import Sampler
from flow_control.samplers.solver import FlowSolver
from flow_control.training.grpo import GrpoTrainer
from flow_control.training.mixins import Rollout
from flow_control.training.nft import NftCachedTargets, NftTrainer, NftTrainItem


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
    num_batches_per_epoch: int = 1
    num_prompts_per_batch: int = 1
    num_rollouts_per_prompt: int = 1
    rollout_sampler: Sampler = Field(default_factory=Sampler)
    validation_sampler: Sampler = Field(default_factory=Sampler)


class _GrpoProbe(_ProbeOverrides, GrpoTrainer):
    pass


class _NftProbe(_ProbeOverrides, NftTrainer):
    @property
    def device(self):
        return torch.device("cpu")


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

        from flow_control.samplers import SampleRequest
        from flow_control.training.grpo_sampling import GrpoCollector

        for eta in (0.0, 0.7):
            trainer = _GrpoProbe.model_validate(
                {
                    "rollout_sampler": {
                        "steps": 4,
                        "solver": {"type": "flow", "eta": eta},
                    }
                }
            )
            collector = GrpoCollector(trainer.rollout_sampler)
            run = next(
                iter(
                    trainer.rollout_sampler.sample(
                        FakeSamplerModel(),
                        [SampleRequest(batch=self.BATCH)],
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

    def test_nft_trains_on_the_executed_rollout_plan(self) -> None:
        trainer = _NftProbe.model_validate({"num_inner_epochs": 2})
        sigmas = [0.9, 0.6, 0.3]
        rollout = Rollout(
            sampling_plan=FlowSolver().plan([*sigmas, 0.0]),
            reward=torch.zeros(1),
            raw_reward=torch.zeros(1),
            reward_weights=torch.ones(1),
            reward_labels=["reward"],
            key="sample",
            batch=self.BATCH,
            negative_batch=None,
        )

        plan = trainer._build_train_plan([rollout])

        self.assertEqual(len(plan), 2)
        for epoch in plan:
            self.assertEqual(len(epoch), len(sigmas))
            for item in epoch:
                self.assertIsInstance(item.sigma, float)
                self.assertEqual(item.sigma, sigmas[item.timestep_idx])

        # The R3/R4 observer migration initially left ordinary NFT rollouts
        # without a plan, so per-step variants and CFG++ failed during training.
        # Exercise real collection with only reward/decode/logging stubbed out.
        from test_microbatching import FakeSamplerModel, make_sampler_batch
        from torchdata.stateful_dataloader import StatefulDataLoader

        from flow_control.training.data import (
            DistributedKRepeatSampler,
            PaddingAwareDatasetWrapper,
            collate_fn,
        )

        model: Any = FakeSamplerModel()
        model.transformer = torch.nn.Identity()
        trainer.model = model
        trainer.rollout_sampler = Sampler.model_validate(
            {
                "steps": 4,
                "solver": {"type": "flow", "eta": 0.7},
                "guidance": {"type": "cfg_pp", "inner": 2.0},
                "transforms": [{"type": "sde_window", "size": 1, "range": [1, 3]}],
            }
        )
        trainer.processor = SimpleNamespace(
            initialize_latents=lambda batch, **kwargs: None,
            decode_output=lambda latents, batch: {},
            get_negative_batch=lambda batch: make_sampler_batch(-0.2),
        )
        dataset = PaddingAwareDatasetWrapper([make_sampler_batch(0.3, 0.9)])
        trainer._dataloader = StatefulDataLoader(
            dataset,
            batch_size=1,
            collate_fn=collate_fn,
            sampler=DistributedKRepeatSampler(dataset, 1, 1, 1, num_replicas=1, rank=0),
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
        item = NftTrainItem(
            rollout_idx=0, timestep_idx=1, sigma=collected.sampling_plan[1].sigma
        )
        predictions = trainer._predict_batched([trainer._make_run(collected)], [item])
        self.assertTrue(torch.isfinite(predictions[0]).all())

        # S2 precision review: teacher-cache scalar products ran in bf16,
        # while .double() in normalization made the final NFT loss fp64.
        trainer.beta = 0.3
        trainer.kl_beta = 0.2
        cached = NftCachedTargets(
            timestep=torch.tensor([0.317], dtype=torch.bfloat16),
            noisy_latents=torch.full((1, 1, 1), 0.734, dtype=torch.bfloat16),
            old_prediction=torch.full((1, 1, 1), 0.121, dtype=torch.bfloat16),
            ref_prediction=torch.full((1, 1, 1), -0.219, dtype=torch.bfloat16),
        )
        assert cached.ref_prediction is not None
        promoted = replace(
            cached,
            timestep=cached.timestep.float(),
            noisy_latents=cached.noisy_latents.float(),
            old_prediction=cached.old_prediction.float(),
            ref_prediction=cached.ref_prediction.float(),
        )
        loss_values = []
        gradients = []
        with patch.object(_NftProbe, "log_aggregated_metrics"):
            for targets in (cached, promoted):
                prepared = trainer._prepare_nft_loss_input(
                    collected,
                    torch.tensor([0.23], dtype=torch.bfloat16),
                    0.317,
                    targets,
                )
                prediction = torch.full((1, 1, 1), 0.354421, requires_grad=True)
                loss = trainer._nft_objective(prepared, prediction)
                self.assertEqual(loss.dtype, torch.float32)
                loss.backward()
                loss_values.append(loss.detach())
                gradients.append(prediction.grad)
        torch.testing.assert_close(*loss_values, rtol=0, atol=0)
        torch.testing.assert_close(*gradients, rtol=0, atol=0)

    def test_nft_timestep_range_keeps_float32_boundary(self) -> None:
        trainer = _NftProbe.model_validate({"timestep_range": 0.3})
        sigmas = torch.tensor([0.7, 0.699, 0.8], dtype=torch.float32).tolist()

        self.assertEqual(trainer._eligible_timestep_indices(sigmas), [0, 2])


if __name__ == "__main__":
    unittest.main()
