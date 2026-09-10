"""Trainer config, rollout metadata and GRPO replay integration tests."""

import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch
from pydantic import BaseModel, Field

from flow_control.samplers import SampleOutput, Sampler
from flow_control.training.grpo import GrpoTrainer
from flow_control.training.mixins import Rollout
from flow_control.training.nft import NftTrainer


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


class RolloutPhaseBuildCheckTest(unittest.TestCase):
    """GRPO's build-time recordable-step check (first rollout batch is the
    cheapest correct spot: the plan only exists once a batch is available)."""

    BATCH: Any = {
        "image_size": (32, 32),
        "clean_latents": torch.zeros(1, 1, 1),
        "noisy_latents": torch.zeros(1, 1, 1),
    }

    def test_grpo_rejects_recipes_without_recordable_stochastic_step(self) -> None:
        from test_microbatching import FakeSamplerModel

        from flow_control.samplers import SampleRequest
        from flow_control.training.grpo_sampling import collect_samples

        for eta in (0.0, 0.7):
            trainer = _GrpoProbe.model_validate(
                {
                    "rollout_sampler": {
                        "steps": 4,
                        "solver": {"type": "flow", "eta": eta},
                    }
                }
            )
            if eta == 0:
                with self.assertRaisesRegex(ValueError, "stochastic"):
                    collect_samples(
                        trainer.rollout_sampler,
                        FakeSamplerModel(),
                        [SampleRequest(batch=self.BATCH)],
                    )
            else:
                outputs, records = collect_samples(
                    trainer.rollout_sampler,
                    FakeSamplerModel(),
                    [SampleRequest(batch=self.BATCH)],
                )
                self.assertEqual(len(outputs), 1)
                self.assertEqual(len(records[0]), 3)
                self.assertNotIn("rollout_recipe", trainer.model_dump())

    def test_nft_train_plan_carries_cpu_sigma_values(self) -> None:
        trainer = _NftProbe.model_validate({"num_inner_epochs": 2})
        timesteps = torch.tensor([0.9, 0.6, 0.3])
        rollout = Rollout(
            trajectory=SampleOutput(
                final_latents=torch.zeros(1, 1, 1),
                timesteps=timesteps,
            ),
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
            self.assertEqual(len(epoch), len(timesteps))
            for item in epoch:
                self.assertIsInstance(item.sigma, float)
                self.assertEqual(item.sigma, float(timesteps[item.timestep_idx]))

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
        for index, transition in enumerate(collected.sampling_plan):
            self.assertEqual(
                transition.sigma, float(collected.trajectory.timesteps[index])
            )
        predictions = trainer._predict_batched(
            [collected.batch],
            [collected.trajectory.timesteps[1:2]],
            [float(collected.trajectory.timesteps[1])],
            [collected.negative_batch],
            [collected.sampling_plan[1]],
            [1],
        )
        self.assertTrue(torch.isfinite(predictions[0]).all())

    def test_nft_timestep_range_keeps_float32_boundary(self) -> None:
        trainer = _NftProbe.model_validate({"timestep_range": 0.3})
        sigmas = torch.tensor([0.7, 0.699, 0.8], dtype=torch.float32).tolist()

        self.assertEqual(trainer._eligible_timestep_indices(sigmas), [0, 2])


if __name__ == "__main__":
    unittest.main()
