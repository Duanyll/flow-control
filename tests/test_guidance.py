"""Differential diffusion and momentum under the node tree."""

import importlib
import sys
import unittest
from itertools import product
from pathlib import Path
from typing import Any

import torch

from flow_control.contrib.momentum_guidance import Momentum
from flow_control.processors.tasks.inpaint import InpaintProcessor
from flow_control.samplers import (
    BaseTransition,
    Cfg,
    CfgPlusPlus,
    DifferentialDiffusion,
    Euler,
    Executor,
    Grid,
    GridFrame,
    Model,
    StepFrame,
)

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
capture = importlib.import_module("sampler_baseline_capture")
microbatching = importlib.import_module("test_microbatching")


class RecordingModel(microbatching.FakeSamplerModel):
    """Zero velocity; keeps the latents every forward saw."""

    def __init__(self) -> None:
        super().__init__()
        self.latents: list[list[torch.Tensor]] = []

    def predict_velocity_batched(self, batches, timesteps, *, dummy_outputs=None):
        self.latents.append([batch["noisy_latents"].clone() for batch in batches])
        return [torch.zeros_like(batch["noisy_latents"]) for batch in batches]


def _inpaint_batch(mask: torch.Tensor, latents: float = 10.0) -> Any:
    batch: Any = microbatching.make_batch(latents, tokens=mask.shape[1])
    batch["inpaint_latents"] = torch.full((1, mask.shape[1], 2), 2.0)
    batch["inpaint_mask_latents"] = mask
    return batch


def _project(
    row: Any, sigmas: list[float], index: int, model: RecordingModel
) -> torch.Tensor:
    """Run one differential-diffusion step at grid position ``index``; return what the model saw."""
    grid = GridFrame(Grid(steps=len(sigmas) - 1), row["_frame"])
    grid.sigmas, grid.index = sigmas, index
    row["_frame"], row["sigma"] = grid, sigmas[index]
    Executor(model).evaluate([DifferentialDiffusion()(row, sigmas[index + 1])])
    return model.latents[-1][0]


class DifferentialDiffusionTest(unittest.TestCase):
    def test_inpaint_mask_controls_reference_release_time(self) -> None:
        # The projection happens before the wrapped step, on the latents the
        # model then sees. At sigma=1 the noised reference is exactly the
        # initial noise; at progress 1/2 of a two-step grid, edit strengths 0
        # and .25 still follow the reference (6 at sigma=.5), .75 and 1 have
        # been released. A grid that starts mid-way measures progress on its
        # own steps, against the noise kept from the first projection.
        mask = torch.tensor([[[0.0], [0.25], [0.75], [1.0]]])
        row = microbatching.sampling_row(_inpaint_batch(mask))
        model = RecordingModel()
        torch.testing.assert_close(
            _project(row, [1.0, 0.5, 0.0], 0, model), torch.full((1, 4, 2), 10.0)
        )
        row["noisy_latents"] = torch.full((1, 4, 2), 20.0)
        torch.testing.assert_close(
            _project(row, [1.0, 0.5, 0.0], 1, model),
            torch.tensor([[[6.0, 6.0], [6.0, 6.0], [20.0, 20.0], [20.0, 20.0]]]),
        )
        row["noisy_latents"] = torch.full((1, 4, 2), 20.0)
        torch.testing.assert_close(
            _project(row, [0.75, 0.5], 0, model),
            torch.tensor([[[8.0, 8.0], [8.0, 8.0], [8.0, 8.0], [20.0, 20.0]]]),
        )
        # The tree round-trips through its JSON spelling (the `from` alias).
        sampler = Grid(
            steps=2,
            transition=DifferentialDiffusion(
                source="inpaint",
                transition=Euler(prediction=Cfg(scale=2.0, renorm=True)),
            ),
        )
        self.assertEqual(Grid.model_validate_json(sampler.model_dump_json()), sampler)
        self.assertIn('"from":"inpaint"', sampler.model_dump_json())

    def test_packed_mask_preserves_intra_token_spatial_detail(self) -> None:
        processor = InpaintProcessor.model_construct(vae_scale_factor=1, patch_size=2)
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
        # The packed mask expands over the latent features of each token, so at
        # progress 1/2 only the features with strength < .5 keep the reference.
        batch: Any = microbatching.make_batch(20.0, tokens=1)
        batch["noisy_latents"] = torch.full((1, 1, 8), 20.0)
        batch["inpaint_latents"] = torch.full((1, 1, 8), 2.0)
        batch["inpaint_mask_latents"] = packed[:, :1]
        row = microbatching.sampling_row(batch)
        row["_noise"] = torch.full((1, 1, 8), 10.0)
        torch.testing.assert_close(
            _project(row, [1.0, 0.5, 0.0], 1, RecordingModel()),
            torch.tensor([[[6.0, 6.0, 20.0, 20.0, 6.0, 6.0, 20.0, 20.0]]]),
        )

    def test_recorded_projection_gap_remains_independently_replayable(self) -> None:
        # The projection between steps breaks the latent chain (latent_next of
        # one record is not latent_t of the next), yet each record replays on
        # its own from the row data.
        sampler = Grid(
            steps=3, transition=DifferentialDiffusion(transition=Euler(eta=0.5))
        )
        batch: Any = microbatching.make_sampler_batch(0.0, initial=0.9)
        batch["inpaint_latents"] = torch.full((1, 1, 1), 0.2)
        batch["inpaint_mask_latents"] = torch.zeros(1, 1, 1)
        row = microbatching.sampling_row(
            batch, torch.Generator().manual_seed(5), [(BaseTransition, capture.record)]
        )
        (rollout,) = microbatching.sample(
            microbatching.FakeSamplerModel(), sampler, [row]
        )
        points = capture.stochastic_points(rollout)
        self.assertEqual(len(points), 2)
        self.assertEqual(
            [type(p["_frame"].node).__name__ for p in points], ["Euler", "Euler"]
        )
        self.assertFalse(
            torch.equal(points[0]["next_latents"], points[1]["noisy_latents"])
        )
        replayed = Executor(microbatching.FakeSamplerModel()).evaluate(
            [capture.replay(rollout, p) for p in points]
        )
        for point, replay in zip(points, replayed, strict=True):
            torch.testing.assert_close(replay.log_prob, point["log_prob"])


class MomentumTest(unittest.TestCase):
    def test_momentum_history_is_per_row_and_branch(self) -> None:
        # Momentum must own its history per sample AND per condition branch.
        # Renorm is nonlinear, so placing Momentum inside/outside CFG exercises
        # different algebra; plain linear CFG would hide a misplaced EMA. CFG++
        # reads the enclosing step's sigma_next and eta.
        class TimeDependentModel(microbatching.FakeSamplerModel):
            def predict_velocity_batched(
                self, batches, timesteps, *, dummy_outputs=None
            ):
                self.forward_batch_sizes.append(len(batches))
                return [
                    b["clean_latents"] * t
                    + torch.cat([t.square(), 1 - t]).reshape(1, 1, 2)
                    for b, t in zip(batches, timesteps, strict=True)
                ]

        def cfg(c, u, sigma, cfg_pp):
            guided = u + 3 * (c - u)
            guided = guided * (
                c.norm(dim=2, keepdim=True) / (guided.norm(dim=2, keepdim=True) + 1e-8)
            ).clamp(max=1)
            # Flow's CFG++ mean conversion uses the actual transition eta.
            kappa = sigma * (1 - 0.15) / ((1 + 0.3**2 / 2) * (sigma - 0.15))
            return u + kappa * (guided - u) if cfg_pp else guided

        def ema(values):
            v0, v1, v2 = values
            return [v0, 1.5 * v1 - 0.5 * v0, 1.5 * v2 - 0.5 * (0.75 * v1 + 0.25 * v0)]

        batch = microbatching.make_sampler_batch(1.0)
        sigmas = (0.9, 0.6, 0.3)
        all_outputs = []
        for cfg_cls, outside in product((Cfg, CfgPlusPlus), (True, False)):
            config = (
                Momentum(alpha=0.5, beta=0.25, prediction=cfg_cls(scale=3, renorm=True))
                if outside
                else cfg_cls(
                    scale=3,
                    renorm=True,
                    prediction=Momentum(alpha=0.5, beta=0.25, prediction=Model()),
                )
            )
            cfg_pp = cfg_cls is CfgPlusPlus
            self.assertEqual(
                Grid.model_validate_json(
                    Grid(transition=Euler(prediction=config)).model_dump_json()
                ),
                Grid(transition=Euler(prediction=config)),
            )
            rows, expected = [], []
            for index in range(2):
                positive, negative = [
                    torch.tensor([[vector]], dtype=torch.float32)
                    for vector in ([2 + index, -1], [-1, 2 - index])
                ]
                rows.append(
                    microbatching.sampling_row(
                        {
                            **batch,
                            "clean_latents": positive,
                            "negative": {"clean_latents": negative},
                        }
                    )
                )
                condition_values = [
                    [b * s + torch.tensor([[[s * s, 1 - s]]]) for s in sigmas]
                    for b in (positive, negative)
                ]
                expected.append(
                    ema(
                        [
                            cfg(c, u, s, cfg_pp)
                            for c, u, s in zip(*condition_values, sigmas, strict=True)
                        ]
                    )
                    if outside
                    else [
                        cfg(c, u, s, cfg_pp)
                        for c, u, s in zip(
                            *(ema(v) for v in condition_values), sigmas, strict=True
                        )
                    ]
                )
            model = TimeDependentModel()
            outputs = []
            for index, sigma in enumerate(sigmas):
                for row in rows:
                    step = StepFrame(Euler(eta=0.3), row["_frame"])
                    step.sigma, step.sigma_next = sigma, 0.15
                    row["_frame"], row["sigma"] = step, sigma
                values = Executor(model).evaluate([config(row) for row in rows])
                for row in rows:
                    row["_frame"] = row["_frame"].parent
                for sample, value in enumerate(values):
                    self.assertEqual(value.dtype, torch.float32)
                    torch.testing.assert_close(value, expected[sample][index])
                outputs.append(values[0])
            self.assertEqual(model.forward_batch_sizes, [4, 4, 4])
            all_outputs.append(outputs)
        self.assertFalse(torch.allclose(all_outputs[0][-1], all_outputs[1][-1]))
        self.assertFalse(torch.allclose(all_outputs[2][-1], all_outputs[3][-1]))


if __name__ == "__main__":
    unittest.main()
