"""Config-level trainer tests for rollout recipes and replay requirements."""

import unittest
from typing import Any

import torch
from pydantic import BaseModel, Field

from flow_control.samplers import SampleOutput, Sampler
from flow_control.training.grpo import GrpoTrainer
from flow_control.training.mixins import Rollout
from flow_control.training.nft import NftTrainer
from flow_control.utils.logging import _warn_once


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
    pass


class RolloutPhaseBuildCheckTest(unittest.TestCase):
    """GRPO's build-time recordable-step check (first rollout batch is the
    cheapest correct spot: the plan only exists once a batch is available)."""

    BATCH: Any = {
        "image_size": (32, 32),
        "clean_latents": torch.zeros(1, 1, 1),
        "noisy_latents": torch.zeros(1, 1, 1),
    }

    def test_grpo_rejects_recipes_without_recordable_stochastic_step(self) -> None:
        deterministic = _GrpoProbe.model_validate(
            {"rollout_sampler": {"steps": 4, "solver": {"type": "flow", "eta": 0.0}}}
        )
        with self.assertRaisesRegex(ValueError, "sde_window"):
            deterministic._build_rollout_phases(self.BATCH, torch.Generator())

        stochastic = _GrpoProbe.model_validate(
            {"rollout_sampler": {"steps": 4, "solver": {"type": "flow", "eta": 0.7}}}
        )
        phases, negative = stochastic._build_rollout_phases(
            self.BATCH, torch.Generator()
        )
        self.assertEqual(len(phases), 1)
        # Default guidance (scale 1.0) never resolves a negative batch.
        self.assertIsNone(negative)

    def test_grpo_rejects_recording_phase_with_overridden_guidance(self) -> None:
        # GRPO replay reconstructs velocities with rollout_sampler.guidance;
        # a recording phase whose sampler override changes the guidance would
        # silently produce wrong policy ratios.
        trainer = _GrpoProbe.model_validate(
            {
                "rollout_recipe": [
                    {
                        "sampler": {
                            "steps": 4,
                            "solver": {"type": "flow", "eta": 0.7},
                            "guidance": {"type": "cfg", "scale": 1.0, "renorm": True},
                        },
                        "transforms": [{"type": "sde_window", "record": True}],
                    }
                ]
            }
        )
        with self.assertRaisesRegex(NotImplementedError, "guidance"):
            trainer._build_rollout_phases(self.BATCH, torch.Generator())

    def test_nft_warns_about_phase_guidance_override(self) -> None:
        trainer = _NftProbe.model_validate(
            {
                "rollout_recipe": [
                    {
                        "sampler": {
                            "steps": 4,
                            "guidance": {"type": "cfg", "renorm": True},
                        }
                    }
                ]
            }
        )
        _warn_once.cache_clear()

        with self.assertLogs("flow_control.training.nft", "WARNING") as logs:
            trainer._build_rollout_phases(self.BATCH, torch.Generator())

        self.assertIn("different policies", "\n".join(logs.output))

    def test_nft_warns_about_stateful_guidance(self) -> None:
        trainer = _NftProbe.model_validate(
            {
                "rollout_sampler": {
                    "guidance": {
                        "type": "momentum",
                        "alpha": 0.3,
                        "beta": 0.7,
                    }
                }
            }
        )
        _warn_once.cache_clear()

        with self.assertLogs("flow_control.training.nft", "WARNING") as logs:
            trainer._build_rollout_phases(self.BATCH, torch.Generator())

        self.assertIn("fresh guidance state", "\n".join(logs.output))

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

    def test_nft_timestep_range_keeps_float32_boundary(self) -> None:
        trainer = _NftProbe.model_validate({"timestep_range": 0.3})
        sigmas = torch.tensor([0.7, 0.699, 0.8], dtype=torch.float32).tolist()

        self.assertEqual(trainer._eligible_timestep_indices(sigmas), [0, 2])


if __name__ == "__main__":
    unittest.main()
