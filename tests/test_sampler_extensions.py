"""Four integration regressions authorized for sampler-rethink R1–R5/A1–A4."""

import asyncio
import unittest
from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
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
from flow_control.rewards import PairwiseReward, execute_pairwise_reward
from flow_control.samplers import (
    Executor,
    Sampler,
    SampleRequest,
    SdeWindow,
    Start,
    conditional_velocity,
)
from flow_control.samplers.guidance import CfgPlusPlusGuidance, ClassifierFreeGuidance
from flow_control.samplers.plan import BranchEvals, StepContext
from flow_control.samplers.shift import LinearShift
from flow_control.samplers.solver import DDIMSolver, FlowSolver, SASolver
from flow_control.training.data import (
    DistributedKRepeatSampler,
    PaddingAwareDatasetWrapper,
)
from flow_control.training.grpo_sampling import (
    GrpoCollector,
    ReplayItem,
    replay_steps,
)


class _TileLeaf:
    """Tokenwise model: tile stitching must reproduce 2x + prompt exactly."""

    device = torch.device("cpu")
    dtype = torch.float32
    micro_batch_size = 4

    def __init__(self):
        self.calls = []

    def use_variant(self, variant):
        return nullcontext()

    def predict_velocity_batched(self, batches, timesteps, *, dummy_outputs=None):
        self.calls.append(len(batches))
        return [
            2 * batch["noisy_latents"] + batch.get("prompt_embeds", 0)
            for batch in batches
        ]


# A 6x6 image with 4-pixel tiles overlapping by 2 (stride 1): four tiles at
# (0, 0), (0, 2), (2, 0), (2, 2), each row-major prompt encoding to its index.
_VARIED: Any = {
    "image_size": (6, 6),
    "prompt": "0",
    "negative_prompt": "-1",
    "tiles": [{"prompt": str(i), "negative_prompt": str(-i - 1)} for i in range(4)],
}
_SMALL: Any = {"image_size": (3, 2), "prompt": "7", "tiles": [{"prompt": "7"}]}
_SHARED: Any = {"image_size": (6, 6), "prompt": "7"}


def _tiled_processor() -> TiledT2IProcessor:
    processor = parse_processor(
        {
            "task": "tiled_t2i",
            "preset": "flux1",
            "tile_size": 4,
            "overlap": 2,
            "vae_scale_factor": 1,
            "patch_size": 1,
            "latent_channels": 1,
            "save_negative": True,
        }
    )
    assert isinstance(processor, TiledT2IProcessor)
    return processor


def _encode_prompt(processor, prompt, **kwargs):
    return {
        "prompt_embeds": torch.tensor([[[float(prompt.strip() or "0")]]]),
        "pooled_prompt_embeds": None,
    }


def _preprocess(processor: TiledT2IProcessor, *inputs: Any) -> tuple[list[Any], int]:
    """Preprocess copies of ``inputs``; also return the encoder call count."""
    with patch.object(
        TiledT2IProcessor, "encode_prompt", autospec=True, side_effect=_encode_prompt
    ) as encode:
        batches = [
            asyncio.run(processor.prepare_inference_batch(deepcopy(item)))
            for item in inputs
        ]
    return batches, encode.call_count


def _conditional(leaf: _TileLeaf, batches: list[Any], timestep: torch.Tensor):
    return Executor(leaf).evaluate(
        [conditional_velocity(batch, timestep) for batch in batches]
    )


def _tiled_latents(processor: TiledT2IProcessor, batch: Any) -> torch.Tensor:
    processor.initialize_latents(
        batch, generator=torch.Generator().manual_seed(11), device=torch.device("cpu")
    )
    return torch.arange(36, dtype=torch.float32).reshape(batch["noisy_latents"].shape)


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
            collector = GrpoCollector(sampler)
            run = next(
                iter(
                    sampler.sample(
                        FakeSamplerModel(),
                        [
                            SampleRequest(
                                batch=batch,
                                negative_batch=negative,
                                generator=torch.Generator().manual_seed(3),
                            )
                        ],
                        collector=collector,
                    )
                )
            )
            trajectory = collector.take(run)
            self.assertEqual(len(trajectory), 2)
            outputs = replay_steps(
                FakeSamplerModel(), [ReplayItem(run, step) for step in trajectory]
            )
            for recorded, replayed in zip(trajectory, outputs, strict=True):
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
            steps=2,
            guidance=ClassifierFreeGuidance(
                scale=2,
                positive_variant=["default", "other"],
                negative_variant="base",
                negative_condition="positive",
            ),
        )
        executor = Executor(adapter, sampler.variant_keys())
        runs = [
            sampler.make_run(SampleRequest(batch), plan=sampler.plan(batch))
            for _ in range(2)
        ]
        # Item 0 runs the "default" branch, item 1 the "other" branch.
        actual = executor.evaluate(
            [run.guided_velocity(x, index) for index, run in enumerate(runs)]
        )
        for variant, value in zip(("default", "other"), actual, strict=True):
            torch.testing.assert_close(
                value, outputs["base"] + 2 * (outputs[variant] - outputs["base"])
            )
        with adapter.use_variant("base"):
            nested = executor.evaluate([runs[0].guided_velocity(x, 0)])[0]
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
            values = executor.evaluate(
                [run.guided_velocity(value, index) for index, run in enumerate(runs)]
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

    def test_tiled_processor_writes_layout_and_negative_tiles(self):
        # The 2026-09-11 rework moves tile geometry to preprocessing: the batch
        # carries a stride-aligned layout, the model-visible size for shift, and
        # negative tiles inside the negative overlay, so no consumer recurses.
        processor = _tiled_processor()
        self.assertIn(
            "tiles",
            get_processor_input_typeddict(type(processor), "inference").__annotations__,
        )
        (batch, small), _ = _preprocess(processor, _VARIED, _SMALL)
        (shared,), calls = _preprocess(processor, _SHARED)
        self.assertEqual(batch["tiling"], {"tile_size": 4, "overlap": 2, "stride": 1})
        self.assertEqual(batch["model_image_size"], (4, 4))
        self.assertEqual([tile["image_size"] for tile in batch["tiles"]], [(4, 4)] * 4)
        self.assertEqual(
            [tile["prompt_embeds"].item() for tile in batch["tiles"]], [0, 1, 2, 3]
        )
        self.assertTrue(all("negative" not in tile for tile in batch["tiles"]))
        # A smaller image is one tile of its own size; a shared prompt is encoded
        # once per branch and referenced by every tile.
        self.assertEqual(small["model_image_size"], (3, 2))
        self.assertEqual([tile["image_size"] for tile in small["tiles"]], [(3, 2)])
        self.assertEqual(calls, 2)
        self.assertTrue(
            all(
                tile["prompt_embeds"] is shared["prompt_embeds"]
                for tile in shared["tiles"]
            )
        )
        negative: Any = processor.get_negative_batch(batch)
        self.assertEqual(negative["tiling"], batch["tiling"])
        self.assertNotIn("negative", negative)
        self.assertEqual(negative["prompt_embeds"].item(), -1)
        self.assertEqual(
            [tile["prompt_embeds"].item() for tile in negative["tiles"]],
            [-1, -2, -3, -4],
        )
        shared_negative: Any = processor.get_negative_batch(shared)
        self.assertEqual(
            [tile["prompt_embeds"].item() for tile in shared_negative["tiles"]], [0] * 4
        )
        with self.assertRaisesRegex(ValueError, "row-major tile prompts"):
            _preprocess(processor, {**_SHARED, "tiles": [{"prompt": "1"}]})

    def test_conditional_velocity_stitches_tiles_with_gradients(self):
        # Tiles are cut and feathered on the token grid inside the run layer;
        # the model sees plain per-tile batches and the stitched velocity keeps
        # gradients to latents and to every tile's condition.
        processor = _tiled_processor()
        (batch, shared), _ = _preprocess(processor, _VARIED, _SHARED)
        x = _tiled_latents(processor, batch).requires_grad_()
        self.assertEqual(x.shape, (1, 36, 1))
        leaf = _TileLeaf()
        timestep = torch.tensor([0.5])
        # Constant conditions reconstruct 2x + 7 exactly; the contract permits
        # one shared condition without a tiles list.
        shared.pop("tiles")
        constant: Any = {**shared, "noisy_latents": x}
        result = _conditional(leaf, [constant], timestep)[0]
        torch.testing.assert_close(result, 2 * x + 7)
        torch.testing.assert_close(
            torch.autograd.grad(result.sum(), x)[0], torch.full_like(x, 2)
        )
        self.assertEqual(leaf.calls, [4])
        # Prompts 0..3 blend with Hann ramps only along shared edges: single
        # coverage is exact, the overlap crossfades, and the layout is symmetric.
        prompts = [
            tile["prompt_embeds"].clone().requires_grad_() for tile in batch["tiles"]
        ]
        varied: Any = {
            **batch,
            "noisy_latents": x.detach(),
            "tiles": [
                {**tile, "prompt_embeds": prompt}
                for tile, prompt in zip(batch["tiles"], prompts, strict=True)
            ],
        }
        stitched = _conditional(leaf, [varied], timestep)[0]
        offset = (stitched - 2 * x.detach()).detach().reshape(6, 6)
        torch.testing.assert_close(
            offset + torch.flip(offset, (0, 1)), torch.full_like(offset, 3)
        )
        torch.testing.assert_close(offset[0, :2], torch.zeros(2))
        torch.testing.assert_close(offset[0, 4:], torch.ones(2))
        self.assertLess(0, float(offset[0, 2]))
        self.assertLess(float(offset[0, 2]), 0.5)
        self.assertLess(0.5, float(offset[0, 3]))
        self.assertLess(float(offset[0, 3]), 1)
        prompt_grads = torch.autograd.grad(stitched[0, 2, 0], prompts)
        torch.testing.assert_close(
            prompt_grads[0] + prompt_grads[1], torch.ones_like(prompts[0])
        )
        torch.testing.assert_close(prompt_grads[1].squeeze(), offset[0, 2])
        for gradient in prompt_grads[2:]:
            torch.testing.assert_close(gradient, torch.zeros_like(gradient))
        fewer_tiles: Any = {**varied, "tiles": varied["tiles"][:3]}
        with self.assertRaisesRegex(ValueError, "tile batches"):
            _conditional(leaf, [fewer_tiles], timestep)
        fewer_tokens: Any = {**constant, "noisy_latents": x[:, :30]}
        with self.assertRaisesRegex(ValueError, "packed BND"):
            _conditional(leaf, [fewer_tokens], timestep)

    def test_tiled_collect_and_replay_share_one_layout(self):
        # Sampling and GRPO replay expand tiles through the same run code, so
        # both expand positive and negative tiles identically and log
        # probabilities replay bitwise from one whole-image noise draw.
        processor = _tiled_processor()
        (batch,), _ = _preprocess(processor, _VARIED)
        batch["noisy_latents"] = _tiled_latents(processor, batch)
        negative: Any = processor.get_negative_batch(batch)
        sampler = Sampler(
            steps=3,
            solver=FlowSolver(eta=0.5),
            guidance=ClassifierFreeGuidance(scale=2),
        )
        leaf = _TileLeaf()
        with torch.no_grad():
            collector = GrpoCollector(sampler)
            run = next(
                iter(
                    sampler.sample(
                        leaf,
                        [
                            SampleRequest(
                                batch=batch,
                                negative_batch=negative,
                                generator=torch.Generator().manual_seed(5),
                            )
                        ],
                        collector=collector,
                    )
                )
            )
        records = collector.take(run)
        replayed = replay_steps(leaf, [ReplayItem(run, step) for step in records])
        for step, replay in zip(records, replayed, strict=True):
            torch.testing.assert_close(replay.log_prob, step.log_prob, rtol=0, atol=0)
        # Both branches share one forward per sampling step (four tiles each),
        # then both recorded steps replay in one call: 2 items x 2 branches x 4.
        self.assertEqual(leaf.calls, [8] * 3 + [16])

    def test_stream_batches_mixed_runs_and_matches_sequential_sampling(self):
        # One executor call may hold a sub-tile image, two different layouts, a
        # stride above one, and an ordinary batch; all expand into one forward.
        processor = _tiled_processor()
        (small, shared), _ = _preprocess(processor, _SMALL, _SHARED)
        small["noisy_latents"] = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1)
        different: Any = {
            **{key: value for key, value in shared.items() if key != "tiles"},
            "tiling": {"tile_size": 3, "overlap": 1, "stride": 1},
            "noisy_latents": torch.arange(36, dtype=torch.float32).reshape(1, 36, 1),
        }
        strided: Any = {
            "image_size": (6, 6),
            "noisy_latents": torch.arange(9, dtype=torch.float32).reshape(1, 9, 1),
            "prompt_embeds": torch.tensor([[[7.0]]]),
            "tiling": {"tile_size": 4, "overlap": 2, "stride": 2},
        }
        plain: Any = {
            "image_size": (2, 2),
            "noisy_latents": torch.ones(1, 4, 1),
            "prompt_embeds": torch.tensor([[[7.0]]]),
        }
        mixed: list[Any] = [small, different, strided, plain]
        leaf = _TileLeaf()
        actual = _conditional(leaf, mixed, torch.tensor([0.5]))
        self.assertEqual(leaf.calls, [1 + 9 + 4 + 1])
        for output, original in zip(actual, mixed, strict=True):
            torch.testing.assert_close(output, 2 * original["noisy_latents"] + 7)

        # One stream mixes runs with different plan lengths, solvers, guidance
        # and tiling. The executor batches whatever is pending (up to the
        # model's micro batch of 4), and every run ends bitwise where it would
        # have ended sampled alone, because per-sample RNG and per-sample math
        # never depend on who shares the forward.
        (tiled,), _ = _preprocess(processor, _VARIED)
        tiled["noisy_latents"] = _tiled_latents(processor, tiled)
        negative: Any = processor.get_negative_batch(tiled)
        plain["clean_latents"] = torch.zeros_like(plain["noisy_latents"])
        samplers = [
            Sampler(
                steps=3,
                solver=FlowSolver(eta=0.5),
                guidance=ClassifierFreeGuidance(scale=2),
            ),
            Sampler(steps=4, solver=SASolver(eta=0.4)),
            Sampler(steps=10, solver=FlowSolver(eta=0.7), start=Start(strength=0.6)),
            Sampler(steps=2),
        ]

        def runs():
            requests = [
                SampleRequest(tiled, negative),
                SampleRequest(plain),
                SampleRequest(plain),
                SampleRequest(plain),
            ]
            return [
                sampler.make_run(
                    replace(request, generator=torch.Generator().manual_seed(index))
                )
                for index, (sampler, request) in enumerate(
                    zip(samplers, requests, strict=True)
                )
            ]

        leaf = _TileLeaf()
        stream_runs = runs()
        list(Executor(leaf).stream(stream_runs))
        # The tiled CFG run alone fills the window (8 calls per round); the
        # other three then share rounds until each finishes: SA (2+1+1+0 evals)
        # and the 6-transition SDEdit run outlast the 2-step one.
        self.assertEqual(leaf.calls, [8, 8, 8, 3, 3, 2, 2, 1, 1])
        alone_calls = 0
        for streamed, run in zip(stream_runs, runs(), strict=True):
            leaf = _TileLeaf()
            (alone,) = Executor(leaf).stream([run])
            alone_calls += len(leaf.calls)
            torch.testing.assert_close(
                streamed.ctx.latents, alone.ctx.latents, rtol=0, atol=0
            )
        self.assertLess(9, alone_calls)

        # S2 streamed completions interleave prompt groups. The reward layer
        # must group by __key__ BEFORE prepare_batch_for_async drops metadata.
        class SamePromptReward(PairwiseReward):
            async def async_score_pair(self, batch_a, batch_b) -> float:
                assert batch_a["prompt"] == batch_b["prompt"]
                return float(
                    batch_a["clean_image"].mean() > batch_b["clean_image"].mean()
                )

        pair_runs = []
        for index, steps in enumerate((3, 4, 1, 5)):
            key = "A" if index < 2 else "B"
            batch: Any = make_sampler_batch(0.2)
            batch.update(
                __key__=key,
                prompt=key,
                clean_image=torch.full((1, 3, 1, 1), float(index % 2)),
            )
            pair_runs.append(Sampler(steps=steps).make_run(SampleRequest(batch)))
        completions = []

        def submitter():
            for run in Executor(_TileLeaf()).stream(pair_runs):
                index = next(
                    i for i, candidate in enumerate(pair_runs) if candidate is run
                )
                completions.append(index)
                yield dict(run.batch), index

        scored = dict(
            execute_pairwise_reward(
                SamePromptReward(),
                submitter(),
                lambda index, result: (index, result.raw.item()),
                num_rollouts_per_prompt=2,
            )
        )
        self.assertEqual(completions, [2, 0, 1, 3])
        self.assertEqual(scored, {0: 0.25, 1: 0.75, 2: 0.25, 3: 0.75})

    def test_shift_reads_model_image_size(self):
        # Resolution-dependent shift follows the size the model actually sees:
        # a 4096 image evaluated in 1024 tiles gets the 1024 grid, whole images
        # keep theirs, and the shift layer needs no tile knowledge for that.
        full: Any = {
            "image_size": (4096, 4096),
            "noisy_latents": torch.zeros(1, 256 * 256, 1),
            "model_image_size": (1024, 1024),
        }
        reference: Any = {
            "image_size": (1024, 1024),
            "noisy_latents": torch.zeros(1, 64 * 64, 1),
        }
        whole: Any = {
            key: value for key, value in full.items() if key != "model_image_size"
        }
        for source in ("actual", "image_size"):
            with self.subTest(shift_source=source):
                sampler = Sampler(steps=6, shift=LinearShift(latent_length_from=source))
                self.assertEqual(
                    sampler.make_sigmas(full), sampler.make_sigmas(reference)
                )
                self.assertNotEqual(
                    sampler.make_sigmas(whole), sampler.make_sigmas(reference)
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
