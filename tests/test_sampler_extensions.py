"""Four integration regressions authorized for sampler-rethink R1–R5/A1–A4."""

import unittest
from collections import Counter
from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch
from peft import LoraConfig
from test_lora_tools import TinyTransformer, make_lora_model
from test_microbatching import FakeDenseAdapter, FakeSamplerModel, make_sampler_batch
from torch.utils.checkpoint import checkpoint

from flow_control.samplers import Sampler, SampleRequest, SdeWindow
from flow_control.samplers.guidance import CfgPlusPlusGuidance, ClassifierFreeGuidance
from flow_control.samplers.plan import BranchEvals, StepContext
from flow_control.samplers.solver import DDIMSolver, FlowSolver
from flow_control.samplers.tiled import Tiled
from flow_control.training.data import (
    DistributedKRepeatSampler,
    PaddingAwareDatasetWrapper,
)
from flow_control.training.grpo_sampling import (
    ReplayItem,
    collect_samples,
    replay_steps,
)


class _TileLeaf:
    device = torch.device("cpu")
    dtype = torch.float32
    patch_size = 1
    vae_scale_factor = 1

    def __init__(self):
        self.calls = []

    def use_variant(self, variant):
        return nullcontext()

    def prepare_tile_batch(self, batch, origin, size, position):
        return {**batch, "image_size": size}

    def predict_velocity_batched(self, batches, timesteps):
        self.calls.append(len(batches))
        return [
            2 * batch["noisy_latents"] + batch.get("prompt", 0) for batch in batches
        ]


class SamplerExtensionsTest(unittest.TestCase):
    def test_cfg_pp_solver_means_and_grpo_replay(self):
        # CHECKLIST A2 originally reused Flow's kappa for stochastic DDIM and
        # read solver.eta; either mistake changes policy ratios after sde_window.
        generator = torch.Generator().manual_seed(29)
        x, cond, uncond = [
            torch.randn(1, 4, 2, dtype=torch.float64, generator=generator)
            for _ in range(3)
        ]
        for solver_cls in (FlowSolver, DDIMSolver):
            for sigma, target in ((1.0, 0.75), (0.9, 0.6), (0.5, 0.25)):
                for eta in (0.0, 0.7, 1.0):
                    for scale in (0.5, 1.0, 3.0):
                        solver = solver_cls(eta=0.123)
                        guidance = CfgPlusPlusGuidance(
                            inner=ClassifierFreeGuidance(scale=scale)
                        )
                        evals = BranchEvals(
                            {"cond": cond, "uncond": uncond},
                            x,
                            sigma,
                            target,
                            eta,
                            solver,
                        )
                        velocity, _ = guidance.combine(
                            evals, StepContext(x, None, None, None), None
                        )
                        guided = uncond + scale * (cond - uncond)
                        x0, epsilon = x - sigma * guided, x + (1 - sigma) * uncond
                        actual = solver.step_parts(
                            x, velocity.velocity, sigma, target, eta
                        )[0]
                        if solver_cls is DDIMSolver:
                            noise = solver.step_parts(x, uncond, sigma, target, eta)[1]
                            expected = (1 - target) * x0 + torch.sqrt(
                                torch.clamp(x.new_tensor(target**2) - noise**2, min=0)
                            ) * epsilon
                        else:
                            dt = target - sigma
                            diffusion = (
                                sigma / (1 - (target if sigma == 1 else sigma))
                            ) ** 0.5 * eta
                            coefficient = (
                                sigma * (1 + diffusion**2 / (2 * sigma) * dt)
                                + (1 + diffusion**2 * (1 - sigma) / (2 * sigma)) * dt
                            )
                            expected = (1 - target) * x0 + coefficient * epsilon
                        torch.testing.assert_close(
                            actual, expected, rtol=1e-12, atol=1e-12
                        )
                        self.assertTrue(guidance.requires_negative(1))
            sampler = Sampler(
                steps=6,
                solver=solver_cls(eta=0.7),
                guidance=CfgPlusPlusGuidance(inner=ClassifierFreeGuidance(scale=0.5)),
                transforms=[SdeWindow(size=2, range=(1, 5))],
            )
            batch, negative = make_sampler_batch(0.3, 0.9), make_sampler_batch(-0.2)
            _, trajectories = collect_samples(
                sampler,
                FakeSamplerModel(),
                [
                    SampleRequest(
                        batch=batch,
                        negative_batch=negative,
                        generator=torch.Generator().manual_seed(3),
                    )
                ],
            )
            self.assertEqual(len(trajectories[0]), 2)
            outputs = replay_steps(
                sampler,
                FakeSamplerModel(),
                [ReplayItem(batch, step, negative) for step in trajectories[0]],
            )
            for recorded, replayed in zip(trajectories[0], outputs, strict=True):
                torch.testing.assert_close(
                    replayed.log_prob, recorded.log_prob, rtol=0, atol=0
                )

    def test_named_variants_preserve_adapter_state(self):
        # A1 shares the training reference context with named branch selection;
        # PEFT toggles can otherwise thaw frozen parameters or leak active weights.
        class CheckpointTransformer(TinyTransformer):
            checkpoint_enabled = False

            def forward(self, value):
                if self.checkpoint_enabled:
                    return self._gradient_checkpointing_func(self.proj, value)
                return self.proj(value)

        model = make_lora_model(model_type=CheckpointTransformer)
        model._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
        model.add_adapter(
            LoraConfig(
                r=2, lora_alpha=4, target_modules=["proj"], init_lora_weights=False
            ),
            adapter_name="other",
        )
        model.set_adapter("default")
        model.proj.lora_A["other"].weight.requires_grad_(False)
        model.proj.lora_B["other"].weight.requires_grad_(False)
        model.proj.base_layer.weight.requires_grad_(True)
        flags = {name: param.requires_grad for name, param in model.named_parameters()}

        class Adapter(FakeDenseAdapter):
            def _predict_velocity(self, batch, timestep):
                return self.transformer(batch["noisy_latents"])

        adapter = Adapter.model_construct(
            arch="fake",
            type="fake",
            hf_model=SimpleNamespace(model=model, dtype=torch.float32),
        )
        x = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4) / 16
        batch: Any = {
            "image_size": (32, 32),
            "clean_latents": torch.zeros_like(x),
            "noisy_latents": x,
        }
        outputs = {}
        for variant in ("default", "other", "base"):
            with adapter.use_variant(variant):
                outputs[variant] = model(x).detach()
                self.assertEqual(
                    {
                        name: param.requires_grad
                        for name, param in model.named_parameters()
                    },
                    flags,
                )
        sampler = Sampler(
            guidance=ClassifierFreeGuidance(
                scale=2,
                positive_variant=["default", "other"],
                negative_variant="base",
                negative_condition="positive",
            )
        )
        actual = sampler.get_guided_velocity(
            adapter,
            [batch, batch],
            [None, None],
            [x, x],
            [torch.tensor([0.5])] * 2,
            [0.5] * 2,
            item_indices=[0, 1],
        )
        for variant, value in zip(("default", "other"), actual, strict=True):
            torch.testing.assert_close(
                value, outputs["base"] + 2 * (outputs[variant] - outputs["base"])
            )
        with adapter.use_variant("base"):
            nested = sampler.get_guided_velocity(
                adapter, [batch], [None], [x], [torch.tensor([0.5])], [0.5]
            )[0]
            torch.testing.assert_close(nested, outputs["base"])
            self.assertTrue(model.proj.disable_adapters)
        with (
            self.assertRaisesRegex(RuntimeError, "probe"),
            adapter.use_variant("other"),
        ):
            raise RuntimeError("probe")
        self.assertEqual(model.active_adapters(), ["default"])
        self.assertFalse(model.proj.disable_adapters)
        self.assertEqual(
            {name: param.requires_grad for name, param in model.named_parameters()},
            flags,
        )

        # Delayed activation-checkpoint recomputation must re-enter the exact
        # variant used during forward, including the base branch. A non-LoRA
        # trainable base weight also requires backward through both branches.
        original_checkpoint = model._gradient_checkpointing_func
        parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        gradients = []
        for enabled in (False, True):
            model.checkpoint_enabled = enabled
            value = x.detach().clone().requires_grad_()
            values = sampler.get_guided_velocity(
                adapter,
                [batch, batch],
                [None, None],
                [value, value],
                [torch.tensor([0.5])] * 2,
                [0.5] * 2,
                item_indices=[0, 1],
            )
            loss = torch.stack([output.square().mean() for output in values]).sum()
            gradients.append(torch.autograd.grad(loss, [value, *parameters]))
            self.assertIs(model._gradient_checkpointing_func, original_checkpoint)
            self.assertEqual(model.active_adapters(), ["default"])
            self.assertFalse(model.proj.disable_adapters)
            self.assertEqual(
                {name: param.requires_grad for name, param in model.named_parameters()},
                flags,
            )
        for eager, recomputed in zip(*gradients, strict=True):
            torch.testing.assert_close(recomputed, eager)

    def test_tiled_reassembly_conditions_padding_and_gradients(self):
        # A4 must blend branch velocities before whole-image guidance and draw
        # solver noise once; padded duplicate tiles must never change the image.
        x = torch.arange(36, dtype=torch.float32).reshape(1, 36, 1).requires_grad_()
        batch: Any = {
            "image_size": (6, 6),
            "clean_latents": torch.zeros_like(x),
            "noisy_latents": x,
            "tiles": [{"image_size": (4, 4), "prompt": i} for i in range(4)],
        }
        leaf = _TileLeaf()
        wrapper = Tiled(tile_size=(4, 4), overlap=(2, 2)).wrap(leaf)
        output = wrapper.predict_velocity_batched([batch], [torch.tensor([0.5])])[0]
        offsets = torch.tensor(
            [[0, 0, 0.5, 0.5, 1, 1]] * 2
            + [[1, 1, 1.5, 1.5, 2, 2]] * 2
            + [[2, 2, 2.5, 2.5, 3, 3]] * 2
        ).reshape(1, 36, 1)
        torch.testing.assert_close(output, 2 * x + offsets)
        output.sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 2))
        self.assertEqual(leaf.calls, [4])

        def pad_count(count, op):
            count.fill_(6)

        with (
            patch("flow_control.samplers.tiled.dist.is_initialized", return_value=True),
            patch(
                "flow_control.samplers.tiled.dist.all_reduce", side_effect=pad_count
            ) as reduce,
        ):
            padded = wrapper.predict_velocity_batched([batch], [torch.tensor([0.5])])[0]
        self.assertEqual(leaf.calls, [4, 6])
        self.assertEqual(reduce.call_count, 1)
        torch.testing.assert_close(padded, output)
        with self.assertRaisesRegex(ValueError, "tile batches"):
            wrapper.predict_velocity_batched(
                [{**batch, "tiles": batch["tiles"][:3]}], [torch.tensor([0.5])]
            )
        sampler = Sampler(
            steps=3,
            solver=FlowSolver(eta=0.5),
            tiled=Tiled(tile_size=(4, 4), overlap=(2, 2)),
        )
        _, records = collect_samples(
            sampler,
            leaf,
            [SampleRequest(batch=batch, generator=torch.Generator().manual_seed(5))],
        )
        replayed = replay_steps(
            sampler, leaf, [ReplayItem(batch, step) for step in records[0]]
        )
        for step, replay in zip(records[0], replayed, strict=True):
            torch.testing.assert_close(replay.log_prob, step.log_prob, rtol=0, atol=0)

    def test_krepeat_buckets_counts_and_resume(self):
        # R5 fixes resolution divergence between equal-position distributed
        # samples and the iterator cursor repeating the last yielded item on resume.
        class Dataset(PaddingAwareDatasetWrapper):
            def __init__(self):
                pass

            bucket_lengths = [5, 7]

            def __len__(self):
                return sum(self.bucket_lengths)

            def __getitem__(self, index):
                return index

        for keep_local, repeats in ((False, 2), (True, 3)):

            def make(rank, repeats=repeats, keep_local=keep_local):
                return DistributedKRepeatSampler(
                    Dataset(),
                    num_batches_per_epoch=2,
                    num_prompts_per_batch=8,
                    num_rollouts_per_prompt=repeats,
                    num_replicas=4,
                    rank=rank,
                    seed=17,
                    keep_prompt_local=keep_local,
                )

            per_rank = [list(make(rank)) for rank in range(4)]
            for block in zip(*per_rank, strict=True):
                self.assertEqual(len({index >= 5 for index in block}), 1)
            local_count = 8 * repeats // 4
            for epoch_batch in range(2):
                counts = Counter(
                    index
                    for rank in per_rank
                    for index in rank[
                        epoch_batch * local_count : (epoch_batch + 1) * local_count
                    ]
                )
                self.assertEqual(len(counts), 8)
                self.assertEqual(set(counts.values()), {repeats})
                if keep_local:
                    for index in counts:
                        self.assertEqual(
                            sum(
                                index
                                in rank[
                                    epoch_batch * local_count : (epoch_batch + 1)
                                    * local_count
                                ]
                                for rank in per_rank
                            ),
                            1,
                        )
            sampler = make(0)
            iterator = iter(sampler)
            next(iterator)
            state = sampler.state_dict()
            remaining = list(iterator)
            restored = make(0)
            restored.load_state_dict(state)
            self.assertEqual(list(restored), remaining)
        dataset = Dataset()
        dataset.bucket_lengths = [1, 1, 1, 1]
        with self.assertRaisesRegex(ValueError, "bucket|resolution"):
            list(
                DistributedKRepeatSampler(
                    dataset,
                    num_batches_per_epoch=1,
                    num_prompts_per_batch=2,
                    num_rollouts_per_prompt=2,
                    num_replicas=4,
                    rank=0,
                )
            )
