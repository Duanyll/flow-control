import unittest
from dataclasses import dataclass
from typing import Any

import torch
from diffusers import ModelMixin
from pydantic import PrivateAttr

from flow_control.adapters.base import BaseModelAdapter, Batch
from flow_control.samplers import Sampler, SampleRequest
from flow_control.samplers.guidance import ClassifierFreeGuidance
from flow_control.samplers.shift import LinearShift
from flow_control.samplers.solver import FlowSolver
from flow_control.training.mixins.microbatch import MicrobatchTrainMixin


def make_batch(
    value: float = 0.0,
    *,
    tokens: int = 4,
    image_size: tuple[int, int] = (32, 32),
) -> Batch:
    latents = torch.full((1, tokens, 2), value)
    return {
        "image_size": image_size,
        "clean_latents": torch.zeros_like(latents),
        "noisy_latents": latents,
    }


class FakeDenseAdapter(BaseModelAdapter[ModelMixin, Batch]):
    supports_dense_batching = True
    dense_batch_fields = (
        "image_size",
        "noisy_latents",
        "clean_latents",
        "reference_latents",
        "reference_sizes",
    )
    _forward_batch_sizes: list[int] = PrivateAttr(default_factory=list)
    _scale: torch.Tensor = PrivateAttr(
        default_factory=lambda: torch.tensor(1.0, requires_grad=True)
    )

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    def _predict_velocity(
        self,
        batch: Batch,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        self._forward_batch_sizes.append(batch["noisy_latents"].shape[0])
        return batch["noisy_latents"] * self._scale + timestep[:, None, None]


class FakeFallbackAdapter(FakeDenseAdapter):
    supports_dense_batching = False


class FakeSamplerModel:
    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self) -> None:
        self.forward_batch_sizes: list[int] = []

    def predict_velocity_batched(
        self,
        batches: list[Batch],
        timesteps: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        self.forward_batch_sizes.append(len(batches))
        return [
            torch.full_like(batch["noisy_latents"], batch["clean_latents"].item())
            for batch in batches
        ]


def make_sampler_batch(velocity: float, initial: float = 0.0) -> Batch:
    batch = make_batch(initial, tokens=1)
    batch["clean_latents"] = torch.tensor([[[velocity]]])
    batch["noisy_latents"] = torch.tensor([[[initial]]])
    return batch


class AdapterBatchingTest(unittest.TestCase):
    def make_adapter(self) -> FakeDenseAdapter:
        return FakeDenseAdapter.model_construct(arch="fake", type="fake")

    def test_dense_inputs_use_one_forward(self) -> None:
        adapter = self.make_adapter()
        batches = [make_batch(1.0), make_batch(2.0)]
        outputs = adapter.predict_velocity_batched(
            batches,
            [torch.tensor([0.25]), torch.tensor([0.5])],
        )

        self.assertEqual(adapter._forward_batch_sizes, [2])
        self.assertEqual([tuple(output.shape) for output in outputs], [(1, 4, 2)] * 2)
        torch.testing.assert_close(outputs[0], torch.full((1, 4, 2), 1.25))
        torch.testing.assert_close(outputs[1], torch.full((1, 4, 2), 2.5))

    def test_incompatible_inputs_fall_back_to_single_sample(self) -> None:
        adapter = self.make_adapter()
        outputs = adapter.predict_velocity_batched(
            [make_batch(tokens=3), make_batch(tokens=5)],
            [torch.tensor([0.0]), torch.tensor([0.0])],
        )

        self.assertEqual(adapter._forward_batch_sizes, [1, 1])
        self.assertEqual([output.shape[1] for output in outputs], [3, 5])

    def test_different_reference_topology_falls_back(self) -> None:
        adapter = self.make_adapter()
        batches = [make_batch(1.0), make_batch(2.0)]
        first: Any = batches[0]
        second: Any = batches[1]
        first["reference_latents"] = [torch.zeros(1, 2, 2)]
        first["reference_sizes"] = [(16, 16)]
        second["reference_latents"] = [torch.zeros(1, 2, 2)]
        second["reference_sizes"] = [(32, 16)]

        adapter.predict_velocity_batched(
            batches,
            [torch.tensor([0.0]), torch.tensor([0.0])],
        )
        self.assertEqual(adapter._forward_batch_sizes, [1, 1])

    def test_dense_and_fallback_gradients_match(self) -> None:
        dense = self.make_adapter()
        fallback = FakeFallbackAdapter.model_construct(arch="fake", type="fake")
        batches = [make_batch(1.0), make_batch(2.0)]
        timesteps = [torch.tensor([0.25]), torch.tensor([0.5])]

        dense_loss = torch.stack(
            [
                output.mean()
                for output in dense.predict_velocity_batched(batches, timesteps)
            ]
        ).sum()
        fallback_loss = torch.stack(
            [
                output.mean()
                for output in fallback.predict_velocity_batched(batches, timesteps)
            ]
        ).sum()
        dense_loss.backward()
        fallback_loss.backward()

        torch.testing.assert_close(dense_loss, fallback_loss)
        torch.testing.assert_close(dense._scale.grad, fallback._scale.grad)


class SamplerBatchingTest(unittest.TestCase):
    def test_mixed_cfg_uses_dummy_forward_without_guiding_missing_negative(
        self,
    ) -> None:
        sampler = Sampler(steps=1, guidance=ClassifierFreeGuidance(scale=2.0))
        model = FakeSamplerModel()
        cond_a = make_sampler_batch(3.0)
        cond_b = make_sampler_batch(4.0)
        negative_a = make_sampler_batch(1.0)

        velocities = sampler.get_guided_velocity(
            model,
            batches=[cond_a, cond_b],
            negative_batches=[negative_a, None],
            latents=[cond_a["noisy_latents"], cond_b["noisy_latents"]],
            timesteps=[torch.tensor([1.0]), torch.tensor([1.0])],
            sigmas=[1.0, 1.0],
        )

        self.assertEqual(model.forward_batch_sizes, [2, 2])
        torch.testing.assert_close(velocities[0], torch.tensor([[[5.0]]]))
        torch.testing.assert_close(velocities[1], torch.tensor([[[4.0]]]))

    def test_cfg_renorm_is_applied_per_sample(self) -> None:
        sampler = Sampler(
            steps=1,
            guidance=ClassifierFreeGuidance(scale=3.0, renorm=True, renorm_min=0.0),
        )
        model = FakeSamplerModel()
        conditional = [make_sampler_batch(2.0), make_sampler_batch(8.0)]
        velocities = sampler.get_guided_velocity(
            model,
            conditional,
            [make_sampler_batch(-2.0), make_sampler_batch(4.0)],
            [batch["noisy_latents"] for batch in conditional],
            [torch.tensor([1.0]), torch.tensor([1.0])],
            [1.0, 1.0],
        )
        torch.testing.assert_close(velocities[0], torch.tensor([[[2.0]]]))
        torch.testing.assert_close(velocities[1], torch.tensor([[[8.0]]]))

    def test_each_request_keeps_its_own_shifted_schedule(self) -> None:
        sampler = Sampler(
            steps=3,
            shift=LinearShift(
                base_image_seq_len=1,
                max_image_seq_len=8,
                base_shift=0.0,
                max_shift=1.0,
            ),
        )
        long_batch = make_batch(tokens=8)
        long_batch["clean_latents"] = torch.tensor(0.0)
        outputs = sampler.sample(
            FakeSamplerModel(),
            [
                SampleRequest(batch=make_sampler_batch(0.0)),
                SampleRequest(batch=long_batch),
            ],
        )
        self.assertFalse(torch.equal(outputs[0].timesteps, outputs[1].timesteps))

    def test_stochastic_generators_are_isolated_per_sample(self) -> None:
        sampler = Sampler(steps=3, solver=FlowSolver(eta=0.4))
        model = FakeSamplerModel()
        batched = sampler.sample(
            model,
            [
                SampleRequest(
                    batch=make_sampler_batch(0.0),
                    generator=torch.Generator().manual_seed(7),
                ),
                SampleRequest(
                    batch=make_sampler_batch(0.0),
                    generator=torch.Generator().manual_seed(19),
                ),
            ],
        )

        individual = [
            sampler.sample(
                FakeSamplerModel(),
                [
                    SampleRequest(
                        batch=make_sampler_batch(0.0),
                        generator=torch.Generator().manual_seed(seed),
                    )
                ],
            )[0]
            for seed in (7, 19)
        ]
        for batched_output, individual_output in zip(batched, individual, strict=True):
            torch.testing.assert_close(
                batched_output.final_latents, individual_output.final_latents
            )


@dataclass(slots=True)
class _IndexedItem:
    rollout_idx: int


class MicrobatchArithmeticTest(unittest.TestCase):
    def make_mixin(
        self, train_batch_size: int, micro_batch_size: int
    ) -> MicrobatchTrainMixin:
        return MicrobatchTrainMixin.model_construct(
            train_batch_size=train_batch_size,
            train_micro_batch_size=micro_batch_size,
        )

    def test_micro_updates_cover_items_in_order(self) -> None:
        mixin = self.make_mixin(train_batch_size=4, micro_batch_size=3)
        updates = list(mixin.iter_micro_updates(list(range(10))))

        self.assertEqual(
            [update.items for update in updates],
            [[0, 1, 2], [3], [4, 5, 6], [7], [8, 9]],
        )
        self.assertEqual(
            [update.is_sync_step for update in updates],
            [False, True, False, True, True],
        )
        self.assertEqual(
            [update.loss_scale for update in updates],
            [0.75, 0.25, 0.75, 0.25, 1.0],
        )

    def test_non_finite_loss_reports_rollout_indices(self) -> None:
        mixin = self.make_mixin(train_batch_size=4, micro_batch_size=2)
        items = [_IndexedItem(rollout_idx=3), _IndexedItem(rollout_idx=5)]
        mixin._check_finite_loss(torch.tensor(1.0), items)
        with self.assertRaisesRegex(RuntimeError, r"rollout_indices=\[3, 5\]"):
            mixin._check_finite_loss(torch.tensor(float("nan")), items)


if __name__ == "__main__":
    unittest.main()
