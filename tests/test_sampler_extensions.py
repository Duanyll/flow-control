"""Sampler extensions: CFG++, tiling, streaming, shift grids and trajectory composition."""

import asyncio
import importlib
import sys
import unittest
from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from einops import rearrange

from flow_control.contrib.momentum_guidance import Momentum
from flow_control.data import (
    Index,
    IndexEntry,
    RowCursor,
    expand_rollouts,
    groups_shuffled,
)
from flow_control.processors import get_processor_input_typeddict, parse_processor
from flow_control.processors.tasks.tiled_t2i import TiledT2IProcessor
from flow_control.rewards import PairwiseReward, execute_pairwise_reward
from flow_control.samplers import (
    BaseGrid,
    BaseTransition,
    Calls,
    Cfg,
    CfgPlusPlus,
    Ddim,
    DifferentialDiffusion,
    DpmGrid,
    Euler,
    Executor,
    Grid,
    InitLatent,
    Model,
    Renoise,
    Row,
    SaGrid,
    SdeWindowGrid,
    Sequential,
    Tiled,
    data,
    gather,
    select_sde_window,
)
from flow_control.samplers.prediction import kappa
from flow_control.samplers.shift import ConstantShift, LinearShift
from flow_control.samplers.transition import ddim_moments, euler_moments
from flow_control.utils.tiling import TileLayout, extract_tiles, stitch_tiles

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
capture = importlib.import_module("sampler_baseline_capture")
microbatching = importlib.import_module("test_microbatching")
sampling_row, sample = microbatching.sampling_row, microbatching.sample
FakeSamplerModel, make_sampler_batch = (
    microbatching.FakeSamplerModel,
    microbatching.make_sampler_batch,
)


def _repeated(row: Row, count: int) -> Calls[torch.Tensor]:
    value = torch.zeros_like(row["noisy_latents"])
    for _ in range(count):
        value = value + (yield from Model()(row))
    return value


def _nested(row: Row) -> Calls[torch.Tensor]:
    values = yield from gather([_repeated({**row}, n) for n in (0, 1, 3)])
    return sum(values, start=torch.zeros_like(values[0]))


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
            asyncio.run(processor.prepare_inference_row(deepcopy(item)))
            for item in inputs
        ]
    return batches, encode.call_count


def _predict(
    leaf: Any, prediction: Any, batches: list[Any], sigma: float = 0.5
) -> list[torch.Tensor]:
    """Evaluate ``prediction`` on fresh rows at ``sigma``; one executor round."""
    rows = [sampling_row(batch) for batch in batches]
    for row in rows:
        row["sigma"] = sigma
    return Executor(leaf).evaluate([prediction(row) for row in rows])


def _tiled_latents(processor: TiledT2IProcessor, batch: Any) -> torch.Tensor:
    processor.initialize_latents(
        batch, generator=torch.Generator().manual_seed(11), device=torch.device("cpu")
    )
    return torch.arange(36, dtype=torch.float32).reshape(batch["noisy_latents"].shape)


def _negative_overlay(batch: Any) -> Any:
    return {**batch, **batch["negative"]}


class SamplerExtensionsTest(unittest.TestCase):
    def test_cfg_pp_preserves_the_step_mean_and_replays(self):
        # CHECKLIST A2 originally reused Flow's kappa for stochastic DDIM and
        # read the config eta; either mistake changes policy ratios after an
        # SDE window. The guided velocity must step to the CFG x0 with the
        # unconditional epsilon under each solver's own moments.
        generator = torch.Generator().manual_seed(29)
        x, cond, uncond = [
            torch.randn(1, 4, 2, dtype=torch.float64, generator=generator)
            for _ in range(3)
        ]
        for ddim in (False, True):
            for sigma, target in ((1.0, 0.75), (0.9, 0.6), (0.5, 0.25)):
                for eta in (0.0, 0.7, 1.0):
                    for scale in (0.5, 1.0, 3.0):
                        guided = uncond + scale * (cond - uncond)
                        velocity = uncond + kappa(sigma, target, eta, ddim) * (
                            guided - uncond
                        )
                        x0, epsilon = x - sigma * guided, x + (1 - sigma) * uncond
                        if ddim:
                            actual, noise = ddim_moments(
                                x, velocity, sigma, target, eta
                            )
                            expected = (1 - target) * x0 + torch.sqrt(
                                torch.clamp(x.new_tensor(target**2) - noise**2, min=0)
                            ) * epsilon
                        else:
                            actual = euler_moments(x, velocity, sigma, target, eta)[0]
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

        # CFG++ always evaluates both branches (two calls per step) and reads
        # the enclosing step's actual eta: windowed rollouts replay bitwise.
        for transition in (Euler, Ddim):
            with self.subTest(transition=transition.__name__):
                sampler = SdeWindowGrid(
                    steps=6,
                    size=2,
                    window_range=(1, 5),
                    eta=0.7,
                    transition=transition(prediction=CfgPlusPlus(scale=0.5)),
                )
                batch = {
                    **make_sampler_batch(0.3, 0.9),
                    "negative": {"clean_latents": torch.tensor([[[-0.2]]])},
                }
                row = sampling_row(
                    batch,
                    torch.Generator().manual_seed(3),
                    [(BaseTransition, capture.record)],
                )
                model = FakeSamplerModel()
                (rollout,) = sample(model, sampler, [row])
                self.assertEqual(model.forward_batch_sizes, [2] * 6)
                points = capture.stochastic_points(rollout)
                self.assertEqual(len(points), 2)
                replayed = Executor(FakeSamplerModel()).evaluate(
                    [capture.replay(rollout, p) for p in points]
                )
                for point, output in zip(points, replayed, strict=True):
                    torch.testing.assert_close(
                        output.log_prob, point["log_prob"], rtol=0, atol=0
                    )
        with self.assertRaisesRegex(
            ValueError, "cfg_pp needs an enclosing euler or ddim step.*DpmGrid"
        ):
            row = sampling_row(
                {
                    **make_sampler_batch(0.3),
                    "negative": {"clean_latents": torch.zeros(1, 1, 1)},
                }
            )
            sample(
                FakeSamplerModel(),
                DpmGrid(steps=2, order=1, prediction=CfgPlusPlus()),
                [row],
            )

    def test_tiled_processor_writes_layout_and_per_tile_negatives(self):
        # Tile geometry lives in preprocessing: the row carries a stride-aligned
        # layout, the model-visible size for shift, one conditioning row per
        # tile (each with its own negative, for CFG under Tiled) and negative
        # tiles under the whole-image negative overlay (for CFG over Tiled).
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
        self.assertEqual(
            [tile["negative"]["prompt_embeds"].item() for tile in batch["tiles"]],
            [-1, -2, -3, -4],
        )
        self.assertEqual(batch["negative"]["prompt_embeds"].item(), -1)
        self.assertEqual(
            [tile["prompt_embeds"].item() for tile in batch["negative"]["tiles"]],
            [-1, -2, -3, -4],
        )
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
        self.assertEqual(
            [tile["prompt_embeds"].item() for tile in shared["negative"]["tiles"]],
            [0] * 4,
        )
        with self.assertRaisesRegex(ValueError, "row-major tile prompts"):
            _preprocess(processor, {**_SHARED, "tiles": [{"prompt": "1"}]})

    def test_conditional_velocity_stitches_tiles_with_gradients(self):
        # Tiles are cut and feathered on the token grid inside the prediction;
        # the model sees plain per-tile rows and the stitched velocity keeps
        # gradients to the latents and to every tile's condition.
        processor = _tiled_processor()
        (batch, shared), _ = _preprocess(processor, _VARIED, _SHARED)
        x = _tiled_latents(processor, batch).requires_grad_()
        self.assertEqual(x.shape, (1, 36, 1))
        leaf = _TileLeaf()
        # Constant conditions reconstruct 2x + 7 exactly; the contract permits
        # one shared condition without a tiles list.
        shared.pop("tiles")
        constant: Any = {**shared, "noisy_latents": x}
        (result,) = _predict(leaf, Tiled(), [constant])
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
        (stitched,) = _predict(leaf, Tiled(), [varied])
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
        with self.assertRaises(ValueError):
            _predict(leaf, Tiled(), [{**varied, "tiles": varied["tiles"][:3]}])
        with self.assertRaisesRegex(ValueError, "no `tiled` node"):
            _predict(leaf, Model(), [varied])

    def test_tiled_cfg_nesting_records_and_replays(self):
        # CFG over Tiled expands both branches through one layout (the negative
        # overlay swaps in the negative tiles); stochastic steps record with one
        # whole-image noise draw and replay bitwise from the row data alone.
        processor = _tiled_processor()
        (batch,), _ = _preprocess(processor, _VARIED)
        batch["noisy_latents"] = _tiled_latents(processor, batch)
        sampler = Grid(
            steps=3,
            transition=Euler(eta=0.5, prediction=Cfg(scale=2, prediction=Tiled())),
        )
        leaf = _TileLeaf()
        row = sampling_row(
            batch, torch.Generator().manual_seed(5), [(BaseTransition, capture.record)]
        )
        with torch.no_grad():
            (rollout,) = sample(leaf, sampler, [row])
        points = capture.stochastic_points(rollout)
        self.assertEqual(len(points), 2)
        replayed = Executor(leaf).evaluate([capture.replay(rollout, p) for p in points])
        for point, replay in zip(points, replayed, strict=True):
            torch.testing.assert_close(
                replay.log_prob, point["log_prob"], rtol=0, atol=0
            )
        # Both branches share one forward per sampling step (four tiles each),
        # then both recorded steps replay in one call: 2 items x 2 branches x 4.
        self.assertEqual(leaf.calls, [8] * 3 + [16])

        # Moving CFG inside Tiled changes where nonlinear renorm happens;
        # CFG(Tiled(Model)) keeps whole-image branch semantics. Momentum under
        # Tiled keeps one history per tile row across calls.
        layout = TileLayout.model_validate(batch["tiling"])
        specs = layout.token_specs(batch["image_size"])
        x = torch.linspace(-3, 3, 36).reshape(1, 36, 1)
        cfg = Cfg(scale=3, renorm=True)
        configs = [
            Cfg(scale=3, renorm=True, prediction=Tiled()),
            Tiled(prediction=cfg),
            Tiled(prediction=Momentum(alpha=0.5, beta=0.25, prediction=cfg)),
        ]
        rows = [sampling_row(batch) for _ in configs]
        expected_tiles = []
        ordered_values = []
        for index in range(3):
            current = x + index * 0.7
            tile_inputs = extract_tiles(
                rearrange(current, "b (h w) d -> b d h w", h=6, w=6), specs
            )

            def guide(c, u):
                combined = u + 3 * (c - u)
                return combined * (c.abs() / (combined.abs() + 1e-8)).clamp(max=1)

            tiles = [
                guide(2 * tile + i, 2 * tile - i - 1)
                for i, tile in enumerate(tile_inputs)
            ]
            expected_tiles.append(tiles)
            raw_c, raw_u = _predict(
                _TileLeaf(),
                Tiled(),
                [
                    {**b, "noisy_latents": current}
                    for b in (batch, _negative_overlay(batch))
                ],
            )
            for row in rows:
                row.update(noisy_latents=current, sigma=0.5)
            value = Executor(_TileLeaf()).evaluate(
                [config(row) for config, row in zip(configs, rows, strict=True)]
            )
            torch.testing.assert_close(value[0], guide(raw_c, raw_u))
            stitched = rearrange(
                stitch_tiles(tiles, specs, 6, 6), "b d h w -> b (h w) d"
            )
            torch.testing.assert_close(value[1], stitched)
            ordered_values.append(value[:2])
            history = expected_tiles
            if index == 0:
                momentum_tiles = tiles
            elif index == 1:
                momentum_tiles = [
                    1.5 * v1 - 0.5 * v0 for v0, v1 in zip(*history, strict=True)
                ]
            else:
                momentum_tiles = [
                    1.5 * v2 - 0.5 * (0.75 * v1 + 0.25 * v0)
                    for v0, v1, v2 in zip(*history, strict=True)
                ]
            expected = rearrange(
                stitch_tiles(momentum_tiles, specs, 6, 6), "b d h w -> b (h w) d"
            )
            torch.testing.assert_close(value[2], expected)
        self.assertTrue(any(not torch.allclose(a, b) for a, b in ordered_values))

    def test_stream_batches_mixed_runs_and_matches_sequential_sampling(self):
        # One executor call may hold a sub-tile image, two different layouts, a
        # stride above one, and an ordinary row; all expand into one forward.
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
        actual = _predict(leaf, Tiled(), mixed)
        self.assertEqual(leaf.calls, [1 + 9 + 4 + 1])
        for output, original in zip(actual, mixed, strict=True):
            torch.testing.assert_close(output, 2 * original["noisy_latents"] + 7)

        # One stream mixes runs with different step counts, grids, guidance
        # and tiling. The executor batches whatever is pending (up to the
        # model's micro batch of 4), and every run ends bitwise where it would
        # have ended sampled alone, because per-sample RNG and per-sample math
        # never depend on who shares the forward.
        (tiled,), _ = _preprocess(processor, _VARIED)
        tiled["noisy_latents"] = _tiled_latents(processor, tiled)
        plain["clean_latents"] = torch.zeros_like(plain["noisy_latents"])
        samplers = [
            Grid(
                steps=3,
                transition=Euler(eta=0.5, prediction=Cfg(scale=2, prediction=Tiled())),
            ),
            SaGrid(steps=4, eta=0.4),
            Sequential(
                steps=[
                    InitLatent(source="clean", sigma_end=0.6),
                    Grid(steps=6, transition=Euler(eta=0.7)),
                ]
            ),
            Grid(steps=2),
        ]

        def runs():
            return [
                (row, sampler(row))
                for sampler, row in zip(
                    samplers,
                    [
                        sampling_row(batch, torch.Generator().manual_seed(index))
                        for index, batch in enumerate((tiled, plain, plain, plain))
                    ],
                    strict=True,
                )
            ]

        leaf = _TileLeaf()
        streamed = runs()
        list(Executor(leaf).stream(streamed))
        # The tiled CFG run alone fills the window (8 calls per round); the
        # other three then share rounds until each finishes: SA (2+1+1+0 evals)
        # and the 6-step SDEdit run outlast the 2-step one.
        self.assertEqual(leaf.calls, [8, 8, 8, 3, 3, 2, 2, 1, 1])
        alone_calls = 0
        for (row, _), run in zip(streamed, runs(), strict=True):
            leaf = _TileLeaf()
            (alone,) = Executor(leaf).stream([run])
            alone_calls += len(leaf.calls)
            torch.testing.assert_close(
                row["noisy_latents"], alone["noisy_latents"], rtol=0, atol=0
            )
        self.assertLess(9, alone_calls)

        # Nested joins may finish without a call or need several waves; the
        # executor must see only ready leaves and return results in input
        # order. One row object may not appear twice in a round.
        row = sampling_row(plain)
        row["sigma"] = 0.5
        leaf = _TileLeaf()
        (values,) = Executor(leaf).evaluate(
            [gather([_nested(row), _repeated({**row}, 2)])]
        )
        self.assertEqual(leaf.calls, [3, 2, 1])
        for value, count in zip(values, (4, 2), strict=True):
            torch.testing.assert_close(value, count * (2 * plain["noisy_latents"] + 7))
        with self.assertRaisesRegex(RuntimeError, "two model calls"):
            Executor(leaf).evaluate([gather([Model()(row), Model()(row)])])

        # S2 streamed completions interleave prompt groups. The reward layer
        # must group by key BEFORE prepare_row_for_async drops metadata.
        class SamePromptReward(PairwiseReward):
            async def async_score_pair(self, row_a, row_b) -> float:
                assert row_a["prompt"] == row_b["prompt"]
                return float(row_a["clean_image"].mean() > row_b["clean_image"].mean())

        pair_runs = []
        for index, steps in enumerate((3, 4, 1, 5)):
            key = "A" if index < 2 else "B"
            batch: Any = make_sampler_batch(0.2)
            batch.update(
                key=key,
                prompt=key,
                clean_image=torch.full((1, 3, 1, 1), float(index % 2)),
            )
            row = sampling_row(batch)
            pair_runs.append((row, Grid(steps=steps)(row)))
        completions = []

        def submitter():
            for row in Executor(_TileLeaf()).stream(pair_runs):
                index = next(
                    i for i, (candidate, _) in enumerate(pair_runs) if candidate is row
                )
                completions.append(index)
                yield data(row), index

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

    def test_shift_grid_matches_reference_schedules(self):
        # Resolution-dependent shift follows the size the model actually sees:
        # a 4096 image evaluated in 1024 tiles gets the 1024 grid, whole images
        # keep theirs, and the shift layer needs no tile knowledge for that.
        full: Any = {
            "image_size": (4096, 4096),
            "noisy_latents": torch.zeros(1, 256 * 256, 1),
            "model_image_size": (1024, 1024),
            "sigma": 1.0,
        }
        reference: Any = {
            "image_size": (1024, 1024),
            "noisy_latents": torch.zeros(1, 64 * 64, 1),
            "sigma": 1.0,
        }
        whole: Any = {
            key: value for key, value in full.items() if key != "model_image_size"
        }
        for source in ("actual", "image_size"):
            with self.subTest(shift_source=source):
                grid = Grid(steps=6, shift=LinearShift(image_seq_len_from=source))
                self.assertEqual(grid.sigmas(full), grid.sigmas(reference))
                self.assertNotEqual(grid.sigmas(whole), grid.sigmas(reference))

        # Qwen21's shift_terminal=0.02 exposed that our full grid stretched
        # the final zero as well, stopping denoising at 0.02. Match the official
        # explicit-sigma path without introducing another sampler schedule.
        shift = LinearShift(
            base_image_seq_len=256,
            max_image_seq_len=8192,
            base_shift=0.5,
            max_shift=0.9,
            shift_terminal=0.02,
        )
        for side in (1024, 2048):
            tokens = side * side // 256
            row: Any = {
                "image_size": (side, side),
                "noisy_latents": torch.zeros(1, tokens, 1),
                "sigma": 1.0,
            }
            for steps in (2, 40):
                with self.subTest(side=side, steps=steps):
                    official = FlowMatchEulerDiscreteScheduler(
                        use_dynamic_shifting=True,
                        shift_terminal=0.02,
                        time_shift_type="exponential",
                    )
                    official.set_timesteps(
                        sigmas=np.linspace(1.0, 1.0 / steps, steps).tolist(),
                        mu=0.5 + (tokens - 256) * (0.9 - 0.5) / (8192 - 256),
                    )
                    expected = official.sigmas
                    assert isinstance(expected, torch.Tensor)
                    actual = torch.tensor(Grid(steps=steps, shift=shift).sigmas(row))
                    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-7)
                    self.assertEqual(actual[-1].item(), 0.0)
                    self.assertAlmostEqual(actual[-2].item(), 0.02)
        self.assertEqual(Grid(steps=1, shift=shift).sigmas(reference), [1.0, 0.0])
        self.assertEqual(
            Grid(
                steps=6, shift=shift.model_copy(update={"shift_terminal": 0.0})
            ).sigmas(reference),
            Grid(
                steps=6, shift=shift.model_copy(update={"shift_terminal": None})
            ).sigmas(reference),
        )
        # A grid from a mid-point follows the same curve through its inverse,
        # forwards (the tail of the full grid) and backwards (inversion).
        full_grid = Grid(steps=4, shift=ConstantShift(shift_value=3.0)).sigmas(
            reference
        )
        mid = {**reference, "sigma": full_grid[2]}
        self.assertEqual(
            Grid(steps=2, shift=ConstantShift(shift_value=3.0)).sigmas(mid),
            full_grid[2:],
        )
        self.assertEqual(
            Grid(steps=2, shift=ConstantShift(shift_value=3.0), sigma_end=1.0).sigmas(
                mid
            ),
            full_grid[2::-1],
        )
        with self.assertRaisesRegex(ValueError, "custom_sigmas"):
            Grid(steps=2, custom_sigmas=[1.0, 0.5, 0.0]).sigmas(mid)

    def test_trajectory_composition_and_sde_window(self):
        # Time travel as a sequential tree: init_latent re-noises the clean
        # latents, every grid starts at the row's sigma, renoise is the forward
        # process from the current point, and each stochastic node draws from
        # the row generator in order.
        x0 = torch.full((1, 4, 2), 2.0)
        batch = {**microbatching.make_batch(0.0), "clean_latents": x0}
        seen: list[tuple[str, float, torch.Tensor]] = []
        observers = [
            (
                InitLatent,
                lambda row, frame: seen.append(
                    ("init", row["sigma"], row["noisy_latents"])
                ),
            ),
            (
                Renoise,
                lambda row, frame: seen.append(
                    ("renoise", row["sigma"], row["noisy_latents"])
                ),
            ),
            (
                BaseGrid,
                lambda row, grid: seen.append(
                    ("grid", grid.sigmas, row["noisy_latents"])
                ),
            ),
        ]
        sampler = Sequential(
            steps=[
                InitLatent(source="clean", sigma_end=0.5),
                Grid(steps=2, sigma_end=0.2),
                Renoise(sigma_end=0.3),
                Grid(steps=2),
            ]
        )
        row = sampling_row(batch, torch.Generator().manual_seed(9), observers)
        (final,) = sample(
            capture.ConstVelocityModel(torch.zeros(1, 4, 2)), sampler, [row]
        )
        draws = torch.Generator().manual_seed(9)
        e1, e2 = [torch.randn(x0.shape, generator=draws) for _ in range(2)]
        self.assertEqual(
            [kind for kind, *_ in seen], ["renoise", "init", "grid", "renoise", "grid"]
        )
        torch.testing.assert_close(seen[1][2], 0.5 * x0 + 0.5 * e1)
        self.assertEqual(seen[1][1], 0.5)
        torch.testing.assert_close(
            torch.tensor(seen[2][1]), torch.tensor([0.5, 0.35, 0.2])
        )
        c = 0.7 / 0.8
        torch.testing.assert_close(
            seen[3][2], c * seen[2][2] + (0.09 - (c * 0.2) ** 2) ** 0.5 * e2
        )
        self.assertEqual(seen[3][1], 0.3)
        torch.testing.assert_close(
            torch.tensor(seen[4][1]), torch.tensor([0.3, 0.15, 0.0])
        )
        self.assertEqual(final["sigma"], 0.0)
        self.assertEqual(sampler.estimate_calls(row), 4)

        # The SDE window needs a transition with eta; a window that does not
        # fit its range is a config error. Observers see the eta copy only on
        # the window steps.
        with self.assertRaisesRegex(ValueError, "transition with `eta`"):
            SdeWindowGrid(steps=4, size=2, eta=0.5, transition=DifferentialDiffusion())
        with self.assertRaisesRegex(ValueError, "does not fit"):
            select_sde_window(4, 3, (1, 3), torch.Generator().manual_seed(0))
        sampler = SdeWindowGrid(
            steps=5, size=2, window_range=(0, 4), eta=0.9, transition=Ddim()
        )
        etas: list[float] = []
        row = sampling_row(
            batch,
            torch.Generator().manual_seed(2),
            [(BaseTransition, lambda row, step: etas.append(step.node.eta))],
        )
        sample(capture.ConstVelocityModel(torch.zeros(1, 4, 2)), sampler, [row])
        start, end = select_sde_window(5, 2, (0, 4), torch.Generator().manual_seed(2))
        self.assertEqual(etas, [0.9 if start <= i < end else 0.0 for i in range(5)])

    def test_cursor_rollouts_counts_and_resume(self):
        # R5 fixed resolution divergence between equal-position distributed
        # samples and the iterator cursor repeating the last yielded item on
        # resume. The data rework moved both guarantees to RowCursor +
        # expand_rollouts: every rank's i-th rollout comes from one plan group
        # (cost-adjacent rows), an epoch's M prompts each get exactly K rollouts
        # (all on one rank under a pairwise reward), and a restored cursor
        # continues exactly where the saved one stopped.
        class Store:
            costs = [5] * 5 + [7] * 7
            index = Index(
                [
                    IndexEntry(str(i), cost, "", None, str(i))
                    for i, cost in enumerate(costs)
                ],
                {},
            )

            def __len__(self):
                return len(self.costs)

            def get(self, row_id):
                return {"key": str(row_id)}

        store: Any = Store()
        n, world_size, prompts = 4, 4, 8
        planner = partial(groups_shuffled, store.index, n, 3, 17)
        walk = [{row for row, _ in g} for p in (0, 1) for g in planner(p)]
        for whole_prompts, repeats in ((False, 2), (True, 3)):
            cursor = RowCursor(store, planner, seed=17)
            epochs = [cursor.take(prompts, epoch=epoch) for epoch in range(2)]
            self.assertEqual(set(epochs[0]) | set(epochs[1][:4]), set(range(12)))
            self.assertEqual(cursor.state_dict(), {"pass": 1, "position": 8})
            for epoch, prompt_ids in enumerate(epochs):
                self.assertEqual(len(set(prompt_ids)), prompts)
                per_rank = [
                    expand_rollouts(
                        prompt_ids,
                        repeats,
                        rank,
                        world_size,
                        whole_prompts=whole_prompts,
                    )
                    for rank in range(world_size)
                ]
                local_count = prompts * repeats // world_size
                self.assertEqual({len(share) for share in per_rank}, {local_count})
                # Epoch 1 straddles two passes and de-duplicates across them
                # (§15 14.3), which breaks group contiguity; within a pass the
                # i-th rollout of every rank comes from one group.
                for position in range(local_count if epoch == 0 else 0):
                    block = {share[position][0] for share in per_rank}
                    self.assertTrue(any(block <= group for group in walk))
                counts = Counter(p for share in per_rank for p, _ in share)
                self.assertEqual(set(counts), set(prompt_ids))
                self.assertEqual(set(counts.values()), {repeats})
                if whole_prompts:
                    for p in prompt_ids:
                        self.assertEqual(
                            sum(p in {q for q, _ in share} for share in per_rank), 1
                        )
            state = cursor.state_dict()
            remaining = [cursor.take(prompts, epoch=epoch) for epoch in (2, 3)]
            restored = RowCursor(store, planner, seed=17)
            restored.load_state_dict(state)
            self.assertEqual(
                [restored.take(prompts, epoch=epoch) for epoch in (2, 3)], remaining
            )
        with self.assertRaisesRegex(ValueError, "distinct prompts"):
            RowCursor(store, planner, seed=17).take(13, epoch=0)


if __name__ == "__main__":
    unittest.main()
