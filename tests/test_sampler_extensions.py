"""Four integration regressions authorized for sampler-rethink R1–R5/A1–A4."""

import asyncio
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

from flow_control.processors import get_processor_input_typeddict, parse_processor
from flow_control.processors.tasks.tiled_t2i import TiledT2IProcessor
from flow_control.processors.tiles import TileConfig
from flow_control.samplers import Sampler, SampleRequest, SdeWindow
from flow_control.samplers.guidance import CfgPlusPlusGuidance, ClassifierFreeGuidance
from flow_control.samplers.plan import BranchEvals, StepContext
from flow_control.samplers.shift import LinearShift
from flow_control.samplers.solver import DDIMSolver, FlowSolver
from flow_control.samplers.tiled import TiledModel
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
            2 * batch["noisy_latents"] + batch.get("prompt_embeds", 0)
            for batch in batches
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
        # The 2026-09-11 tiled correction moves geometry to preprocessing. The
        # old sampler-owned layout used full-image sequence length for shift,
        # required redundant tile sizes and could not mix layouts per batch.
        def encode_prompt(processor, prompt, **kwargs):
            return {
                "prompt_embeds": torch.tensor([[[float(prompt.strip() or "0")]]]),
                "pooled_prompt_embeds": None,
            }

        processor = parse_processor(
            {
                "task": "tiled_t2i",
                "preset": "flux1",
                "tile_size": (4, 4),
                "overlap": (2, 2),
                "vae_scale_factor": 1,
                "patch_size": 1,
                "latent_channels": 1,
                "save_negative": True,
            }
        )
        assert isinstance(processor, TiledT2IProcessor)
        self.assertIn(
            "tiles",
            get_processor_input_typeddict(type(processor), "inference").__annotations__,
        )
        with patch.object(
            TiledT2IProcessor, "encode_prompt", autospec=True, side_effect=encode_prompt
        ) as encode:
            batch: Any = asyncio.run(
                processor.prepare_inference_batch(
                    {
                        "image_size": (6, 6),
                        "prompt": "0",
                        "negative_prompt": "-1",
                        "tiles": [
                            {"prompt": str(i), "negative_prompt": str(-i - 1)}
                            for i in range(4)
                        ],
                    }
                )
            )
            small: Any = asyncio.run(
                processor.prepare_inference_batch(
                    {
                        "image_size": (3, 2),
                        "prompt": "7",
                        "tiles": [{"prompt": "7"}],
                    }
                )
            )
            previous_calls = encode.call_count
            inherited: Any = asyncio.run(
                processor.prepare_inference_batch(
                    {
                        "image_size": (6, 6),
                        "prompt": "7",
                    }
                )
            )
        self.assertEqual(batch["image_size"], (6, 6))
        self.assertEqual([tile["image_size"] for tile in batch["tiles"]], [(4, 4)] * 4)
        self.assertEqual(small["tiles"][0]["image_size"], (3, 2))
        self.assertEqual(encode.call_count - previous_calls, 2)
        self.assertTrue(
            all(
                tile["prompt_embeds"] is inherited["prompt_embeds"]
                for tile in inherited["tiles"]
            )
        )
        # The metadata contract also permits one shared condition without a tiles list.
        inherited.pop("tiles")
        self.assertEqual(TileConfig.model_validate(batch["tiling"]).tile_size, (4, 4))
        processor.initialize_latents(
            batch,
            generator=torch.Generator().manual_seed(11),
            device=torch.device("cpu"),
        )
        self.assertEqual(batch["noisy_latents"].shape, (1, 36, 1))
        x = torch.arange(36, dtype=torch.float32).reshape(1, 36, 1).requires_grad_()
        batch["noisy_latents"] = x
        negative: Any = processor.get_negative_batch(batch)
        assert negative is not None
        self.assertEqual(negative["tiling"], batch["tiling"])
        self.assertEqual(
            [float(tile["prompt_embeds"].item()) for tile in negative["tiles"]],
            [-1, -2, -3, -4],
        )
        leaf = _TileLeaf()
        wrapper = TiledModel(leaf)
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

        for blend in ("uniform", "gaussian", "hann"):
            with self.subTest(blend=blend):
                value = x.detach().clone().requires_grad_()
                configuration = {**batch["tiling"], "blend": blend}
                constant = {
                    **inherited,
                    "tiling": configuration,
                    "noisy_latents": value,
                }
                result = wrapper.predict_velocity_batched(
                    [constant], [torch.tensor([0.5])]
                )[0]
                torch.testing.assert_close(result, 2 * value + 7)
                gradient = torch.autograd.grad(result.sum(), value)[0]
                torch.testing.assert_close(gradient, torch.full_like(value, 2))
                self.assertTrue(torch.isfinite(result).all())
                prompts = [
                    tile["prompt_embeds"].detach().clone().requires_grad_()
                    for tile in batch["tiles"]
                ]
                varied = {
                    **batch,
                    "tiling": configuration,
                    "noisy_latents": value,
                    "tiles": [
                        {**tile, "prompt_embeds": prompt}
                        for tile, prompt in zip(batch["tiles"], prompts, strict=True)
                    ],
                }
                stitched = wrapper.predict_velocity_batched(
                    [varied], [torch.tensor([0.5])]
                )[0]
                offset = (stitched - 2 * value).reshape(6, 6)
                torch.testing.assert_close(
                    offset + torch.flip(offset, (0, 1)), torch.full_like(offset, 3)
                )
                self.assertTrue(torch.isfinite(offset).all())
                if blend == "uniform":
                    torch.testing.assert_close(offset, offsets.reshape(6, 6))
                else:
                    self.assertGreater(float(offset[0, 2].detach()), 0)
                    self.assertLess(float(offset[0, 2].detach()), 0.5)
                    self.assertGreater(float(offset[0, 3].detach()), 0.5)
                    self.assertLess(float(offset[0, 3].detach()), 1)
                prompt_grads = torch.autograd.grad(stitched[0, 2, 0], prompts)
                torch.testing.assert_close(
                    prompt_grads[0] + prompt_grads[1], torch.ones_like(prompts[0])
                )
                torch.testing.assert_close(
                    prompt_grads[1].squeeze(), offset[0, 2].detach()
                )
                for gradient in prompt_grads[2:]:
                    torch.testing.assert_close(gradient, torch.zeros_like(gradient))

        # Public sampling and plugin replay must discover the same layout
        # on both positive and negative batches, with one whole-image noise.
        configuration = {**batch["tiling"], "blend": "gaussian"}
        sampler = Sampler(
            steps=3,
            solver=FlowSolver(eta=0.5),
            guidance=ClassifierFreeGuidance(scale=2),
        )
        request_batch: Any = {**batch, "tiling": configuration}
        request_negative: Any = {**negative, "tiling": configuration}
        with torch.no_grad():
            _, records = collect_samples(
                sampler,
                leaf,
                [
                    SampleRequest(
                        batch=request_batch,
                        negative_batch=request_negative,
                        generator=torch.Generator().manual_seed(5),
                    )
                ],
            )
        replayed = replay_steps(
            sampler,
            leaf,
            [ReplayItem(request_batch, step, request_negative) for step in records[0]],
        )
        for step, replay in zip(records[0], replayed, strict=True):
            torch.testing.assert_close(replay.log_prob, step.log_prob, rtol=0, atol=0)

        # Smaller-than-tile images, another layout and an ordinary batch can share
        # one call. All inherit constant conditions, so overlap must be invisible.
        small["noisy_latents"] = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1)
        different = {
            **inherited,
            "tiling": TileConfig(tile_size=(3, 3), overlap=(1, 1)).model_dump(),
            "noisy_latents": x.detach(),
        }
        plain: Any = {
            "image_size": (2, 2),
            "noisy_latents": torch.ones(1, 4, 1),
            "prompt_embeds": torch.tensor([[[7.0]]]),
        }
        mixed: list[Any] = [small, different, plain]
        actual = wrapper.predict_velocity_batched(mixed, [torch.tensor([0.5])] * 3)
        self.assertEqual(leaf.calls[-1], 11)
        for actual_batch, original in zip(actual, mixed, strict=True):
            torch.testing.assert_close(actual_batch, 2 * original["noisy_latents"] + 7)
        single = wrapper.predict_velocity_batched([plain], [torch.tensor([0.5])])[0]
        torch.testing.assert_close(single, 2 * plain["noisy_latents"] + 7)

        full: Any = {
            "image_size": (4096, 4096),
            "noisy_latents": torch.zeros(1, 256 * 256, 1),
            "tiling": TileConfig(
                tile_size=(1024, 1024), overlap=(128, 128)
            ).model_dump(),
        }
        reference: Any = {
            "image_size": (1024, 1024),
            "noisy_latents": torch.zeros(1, 64 * 64, 1),
        }
        for source in ("actual", "image_size"):
            with self.subTest(shift_source=source):
                sampler = Sampler(steps=6, shift=LinearShift(latent_length_from=source))
                self.assertEqual(
                    sampler.make_sigmas(full), sampler.make_sigmas(reference)
                )
                untiled: Any = {
                    key: value for key, value in full.items() if key != "tiling"
                }
                self.assertNotEqual(
                    sampler.make_sigmas(untiled), sampler.make_sigmas(reference)
                )
                smaller: Any = {
                    "image_size": (512, 512),
                    "noisy_latents": torch.zeros(1, 32 * 32, 1),
                }
                tiled_smaller: Any = {**smaller, "tiling": full["tiling"]}
                self.assertEqual(
                    sampler.make_sigmas(tiled_smaller), sampler.make_sigmas(smaller)
                )

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
