"""How the trainers consume rollouts and train points: GRPO record / replay,
endpoint grids and independent train predictions. CPU, with the DiT, store,
processor and reward faked around the real trainer code."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import patch

import torch
from pydantic import BaseModel, Field
from test_microbatching import FakeSamplerModel

from flow_control.contrib.momentum_guidance import Momentum
from flow_control.data import KEY, Index, IndexEntry, RowCursor
from flow_control.processors import parse_processor
from flow_control.rewards.base import BaseScorer, LocalReward, Scorers, scorer_registry
from flow_control.samplers import Cfg, CfgPlusPlus, Grid, Row, Trajectory
from flow_control.training.awm import AwmTrainer
from flow_control.training.endpoint import EndpointTrainItem
from flow_control.training.grpo import GrpoTrainer
from flow_control.training.nft import NftObjective, NftTrainer
from flow_control.training.ram import RamTrainer
from flow_control.training.rollouts import PointwiseRollout, Rollout
from flow_control.training.sft import SftTrainer
from flow_control.training.weighting import LogitNormalTimestepWeighting


@scorer_registry.register("_test_key")
class _KeyScorer(BaseScorer, LocalReward[Row, torch.Tensor]):
    """Local scorer ``float(row[KEY])``: a rollout's reward names its prompt, so
    the probes can tell which batch a reward came from."""

    type: Literal["_test_key"] = "_test_key"
    name: str | None = "key"

    @property
    def fields(self) -> set[str]:
        return {KEY}

    def run_batch(self, rows: list[Row]) -> list[torch.Tensor]:
        return [torch.tensor([float(row[KEY])]) for row in rows]


class _ProbeOverrides(BaseModel):
    """Defaults for the heavy required trainer fields (config-only tests)."""

    model: Any = None
    processor: Any = None
    dataset: Any = None
    launch: Any = None
    checkpoint_root: str = ""
    experiment_name: str = "probe"
    seed_checkpoint_dir: str = ""
    num_prompts_per_epoch: int = 1
    num_rollouts_per_prompt: int = 1
    rollout: Rollout = Field(
        default_factory=lambda: PointwiseRollout(
            sampler=Grid(), reward=Scorers([_KeyScorer()])
        )
    )
    validation_sampler: Trajectory = Field(default_factory=Grid)

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


class _VariantModel(FakeSamplerModel):
    """``gain[variant] * (prompt_embeds + 0.1 * x)``; journals the variant of every
    non-empty forward (the runner's executor forwards every variant each round)."""

    def __init__(self, **gains: float) -> None:
        super().__init__()
        self.gain = torch.tensor(1.0, requires_grad=True)
        self.gains = {"default": self.gain} | {
            name: torch.tensor(value) for name, value in gains.items()
        }
        self.variants: dict[str, object] = dict.fromkeys(self.gains)
        self.active = "default"
        self.forwards: list[tuple[str, int]] = []
        self.transformer = torch.nn.Identity()

    @contextmanager
    def use_variant(self, variant: str | None):
        previous, self.active = self.active, variant or self.active
        try:
            yield
        finally:
            self.active = previous

    def predict_velocity_batched(self, batches, timesteps, *, dummy_outputs=None):
        if not batches:
            return []
        self.forward_batch_sizes.append(len(batches))
        self.forwards.append((self.active, len(batches)))
        return [
            self.gains[self.active] * (b["prompt_embeds"] + 0.1 * b["noisy_latents"])
            for b in batches
        ]


class _Store:
    """One prompt row keyed ``"0"``; ``prompt_embeds`` 3 with negative 1."""

    index = Index([IndexEntry("0", 1, "", None, "0")], {})

    def __len__(self) -> int:
        return 1

    def get(self, row_id: int) -> dict[str, Any]:
        return {
            KEY: "0",
            "image_size": (32, 32),
            "clean_latents": torch.zeros(1, 1, 1),
            "noisy_latents": torch.full((1, 1, 1), 0.9),
            "prompt_embeds": torch.tensor([[[3.0]]]),
            "negative": {"prompt_embeds": torch.tensor([[[1.0]]])},
        }


def _collect(case: unittest.TestCase, trainer: Any, model: Any) -> Row:
    """One real synchronous rollout batch of the single prompt through the runner."""
    trainer.set_seed()
    trainer.model = model
    trainer.processor = SimpleNamespace(
        initialize_latents=lambda row, **kwargs: row["noisy_latents"],
        decode_output=lambda latents, row: {},
        resample=lambda row, generator: row,
    )
    store: Any = _Store()
    trainer._store = store
    trainer._cursor = RowCursor(
        store, trainer.make_planner(store, shuffle=True), trainer.seed
    )
    trainer.make_runner(model)
    case.addCleanup(trainer.close_runner)
    (row,) = trainer._finish_rollouts(trainer._start_rollouts(0))
    return row


def _point(batch: dict[str, Any], trainer: Any) -> dict[str, Any]:
    """A training point at σ = 0.5 whose ``x_t`` is the batch's zero latents."""
    return trainer.noised_point(batch, 0.5, torch.zeros_like(batch["clean_latents"]))


class TrainerRolloutTest(unittest.TestCase):
    def test_grpo_records_and_replays_through_train_predictor(self) -> None:
        """A rollout records only its stochastic steps, each with the behaviour
        tree's likelihood (momentum history included); training replays every
        recorded point through ``train_predictor`` on the recorded call chain,
        so the ratio follows the train tree, and ``kl_beta`` reads the ``ref``
        variant inline. A deterministic rollout is refused before training."""
        for rollout_scale, train_scale in ((1, 3), (3, 1)):
            branches = {1: 1, 3: 2}  # cfg scale 1 is the conditional alone
            model = _VariantModel(ref=0.5)
            trainer = _GrpoProbe.model_validate(
                {
                    "train_predictor": {"type": "cfg", "scale": train_scale},
                    "rollout": {
                        "type": "pointwise",
                        "sampler": {
                            "type": "sde_window_grid",
                            "steps": 4,
                            "size": 2,
                            "window_range": [1, 3],
                            "eta": 0.7,
                            "transition": {
                                "type": "euler",
                                "prediction": Momentum(
                                    alpha=0.5,
                                    beta=0.25,
                                    prediction=Cfg(scale=rollout_scale),
                                ),
                            },
                        },
                        "reward": {"type": "_test_key"},
                    },
                    "kl_beta": 0.1,
                }
            )
            rollout = _collect(self, trainer, model)
            trajectory = rollout["trajectory"]
            self.assertEqual([p["_frame"].parent.index for p in trajectory], [1, 2])
            self.assertEqual(model.forwards, [("default", branches[rollout_scale])] * 4)
            self.assertNotIn("_frame", rollout)
            # Recorded = the momentum-guided velocity at step 1 under the
            # rollout's eta 0.7 (history from step 0 at the initial latents).
            first = trainer._point(rollout, 0)
            guided = 1 + 2 * rollout_scale
            behaviour = 1.5 * (guided + 0.1 * first["noisy_latents"]) - 0.5 * (
                guided + 0.1 * 0.9
            )
            node = first["_frame"].node
            self.assertEqual(node.eta, 0.7)
            torch.testing.assert_close(
                first["log_prob"], node.log_prob(first, behaviour).log_prob
            )
            # Replay evaluates the train tree (no momentum) at the same point.
            model.forwards.clear()
            (output,) = trainer._replay([first])
            expected = node.log_prob(
                first, 1 + 2 * train_scale + 0.1 * first["noisy_latents"]
            )
            torch.testing.assert_close(output.log_prob, expected.log_prob)
            self.assertFalse(
                torch.allclose(output.log_prob.detach(), first["log_prob"])
            )
            self.assertEqual(model.forwards, [("default", branches[train_scale])])

            plan = trainer._build_train_plan([rollout])
            self.assertEqual([item.timestep_idx for item in plan[0]], [0, 1])
            model.forwards.clear()
            loss = trainer._loss_batched(plan[0], [rollout], torch.tensor([0.4]))
            calls = 2 * branches[train_scale]
            self.assertEqual(model.forwards, [("default", calls), ("ref", calls)])
            self.assertIn("train/kl_loss", trainer._aggregated_metrics)
            self.assertTrue(torch.isfinite(loss))
            self.assertIsNotNone(torch.autograd.grad(loss, model.gain)[0])

        deterministic = _GrpoProbe.model_validate({"train_predictor": "model"})
        rollout = _collect(self, deterministic, _VariantModel())
        self.assertNotIn("trajectory", rollout)
        with self.assertRaisesRegex(RuntimeError, "stochastic"):
            deterministic._check_rollouts([rollout])

    def test_endpoint_trains_on_the_executed_grid(self) -> None:
        """Endpoint trainers keep the executed grid (concatenated over
        sequential grids) and draw train sigmas from its step starts; every
        point is an independent ``train_predictor`` evaluation (model / tiled /
        cfg, with the row's own negative; cfg_pp has no step to read), and the
        objective's variants are read inline, under no_grad, in sorted order."""
        trainer = _NftProbe.model_validate(
            {
                "num_inner_epochs": 2,
                "train_predictor": "model",
                "rollout": {
                    "type": "pointwise",
                    "sampler": [
                        {"type": "grid", "steps": 2, "sigma_end": 0.5},
                        {"type": "grid", "steps": 2},
                    ],
                    "reward": {"type": "_test_key"},
                },
            }
        )
        model = _VariantModel(old=0.8, ref=0.5)
        rollout = _collect(self, trainer, model)
        self.assertEqual(rollout["sigmas"], [1.0, 0.75, 0.5, 0.25, 0.0])
        torch.testing.assert_close(rollout["clean_latents"], rollout["noisy_latents"])
        plan = trainer._build_train_plan([rollout])
        self.assertEqual(len(plan), 2)
        for items in plan:
            self.assertEqual(
                sorted(item.sigma for item in items), [0.25, 0.5, 0.75, 1.0]
            )
            self.assertTrue(all(isinstance(item, EndpointTrainItem) for item in items))

        trainer.objective = NftObjective(beta=0.3, kl_beta=0.2)
        model.forwards.clear()
        loss = trainer._loss_batched(plan[0][:2], [rollout], torch.tensor([0.23]))
        self.assertEqual(model.forwards, [("old", 2), ("ref", 2), ("default", 2)])
        self.assertEqual(loss.dtype, torch.float32)
        self.assertIsNotNone(torch.autograd.grad(loss, model.gain)[0])

        processor = parse_processor({"task": "t2i", "preset": "flux1"})
        plain: Any = {
            "image_size": (2, 2),
            "clean_latents": torch.zeros(1, 4, 1),
            "prompt_embeds": torch.tensor([[[3.0]]]),
            "negative": {"prompt_embeds": torch.tensor([[[1.0]]])},
        }
        tiled: Any = {
            **plain,
            "tiling": {"tile_size": 1, "overlap": 0, "stride": 1},
            "tiles": [
                {"prompt_embeds": torch.tensor([[[float(i)]]])} for i in range(1, 5)
            ],
        }
        configs = [
            ("model", plain, torch.full((1, 4, 1), 3.0), [1]),
            ("tiled", tiled, torch.arange(1.0, 5.0).reshape(1, 4, 1), [4]),
            ({"type": "cfg", "scale": 3}, plain, torch.full((1, 4, 1), 7.0), [2]),
        ]
        for cls in (_SftProbe, _AwmProbe, _RamProbe):
            with self.assertRaisesRegex(ValueError, "train_predictor"):
                cls.model_validate({})
            for config, batch, expected, calls in configs:
                trainer = cls.model_validate({"train_predictor": config})
                trainer.processor, trainer.model = processor, _VariantModel()
                with torch.no_grad():
                    (aux,) = trainer.predict_training([_point(batch, trainer)])
                (prediction,) = trainer.predict_training([_point(batch, trainer)])
                self.assertEqual(prediction.dtype, torch.float32)
                torch.testing.assert_close(aux, expected)
                torch.testing.assert_close(prediction, expected)
                torch.testing.assert_close(
                    torch.autograd.grad(prediction.sum(), trainer.model.gain)[0],
                    expected.sum(),
                )
                self.assertEqual(trainer.model.forward_batch_sizes, calls * 2)

        # CFG dropout swaps in the negative overlay but keeps the negative, so
        # a guided train_predictor still evaluates both branches.
        trainer = _SftProbe.model_validate(
            {"train_predictor": configs[-1][0], "cfg_drop_prob": 1}
        )
        trainer.processor, trainer.model = processor, _VariantModel()
        with (
            patch.object(
                LogitNormalTimestepWeighting,
                "sample_timesteps",
                return_value=torch.tensor([0.5]),
            ),
            patch("torch.randn_like", return_value=torch.ones(1, 4, 1)),
        ):
            loss = trainer.train_step([plain])
        torch.testing.assert_close(loss, torch.tensor(0.05**2))
        self.assertEqual(trainer.model.forward_batch_sizes, [2])
        missing_negative: Any = {k: v for k, v in plain.items() if k != "negative"}
        with self.assertRaisesRegex(ValueError, "negative"):
            trainer.predict_training([_point(missing_negative, trainer)])
        trainer.train_predictor = CfgPlusPlus(scale=3)
        with self.assertRaisesRegex(ValueError, "cfg_pp"):
            trainer.predict_training([_point(plain, trainer)])

    def test_nft_timestep_window_is_index_based(self) -> None:
        # NFT's sigma-threshold ``timestep_range`` made the per-rank item count
        # depend on the rollout's sigma values (resolution-dependent shift);
        # the grid window keeps the noisiest index fraction of the step starts.
        trainer = _NftProbe.model_validate(
            {
                "train_timesteps": {"type": "grid", "window": 0.3},
                "train_predictor": "model",
            }
        )
        uniform = [1.0 - i / 10 for i in range(11)]
        shifted = [s / (s + (1 - s) / 3) for s in uniform]
        for sigmas, expected in (
            ([0.7, 0.699, 0.8, 0.0], [0]),
            ([0.95, 0.5, 0.05, 0.0], [0]),
            (uniform, [0, 1, 2]),
            (shifted, [0, 1, 2]),
        ):
            drawn = trainer.train_timesteps.draw(sigmas)
            self.assertEqual(sorted(sigmas.index(s) for s in drawn), expected)


if __name__ == "__main__":
    unittest.main()
