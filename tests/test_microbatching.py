import random
import unittest
from dataclasses import dataclass
from typing import Any

import torch
from diffusers import ModelMixin
from pydantic import PrivateAttr
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from flow_control.adapters.base import BaseModelAdapter, Batch
from flow_control.samplers import ReplayRequest, Sampler, SampleRequest
from flow_control.samplers.shift import LinearShift
from flow_control.samplers.solver import FlowSolver
from flow_control.training.data import collate_fn
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

    def test_empty_and_mismatched_inputs_are_rejected(self) -> None:
        adapter = self.make_adapter()
        with self.assertRaisesRegex(ValueError, "at least one batch"):
            adapter.predict_velocity_batched([], [])
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            adapter.predict_velocity_batched([make_batch()], [])

    def test_incompatible_inputs_fall_back_to_single_sample(self) -> None:
        adapter = self.make_adapter()
        outputs = adapter.predict_velocity_batched(
            [make_batch(tokens=3), make_batch(tokens=5)],
            [torch.tensor([0.0]), torch.tensor([0.0])],
        )

        self.assertEqual(adapter._forward_batch_sizes, [1, 1])
        self.assertEqual([output.shape[1] for output in outputs], [3, 5])

    def test_irrelevant_metadata_does_not_block_dense_batching(self) -> None:
        adapter = self.make_adapter()
        batches = [make_batch(1.0), make_batch(2.0)]
        for batch, prompt in zip(batches, ("first", "second"), strict=True):
            batch_any: Any = batch
            batch_any["prompt"] = prompt
            batch_any["__key__"] = prompt

        adapter.predict_velocity_batched(
            batches,
            [torch.tensor([0.0]), torch.tensor([0.0])],
        )
        self.assertEqual(adapter._forward_batch_sizes, [2])

    def test_matching_reference_topology_is_collated_recursively(self) -> None:
        adapter = self.make_adapter()
        batches = [make_batch(1.0), make_batch(2.0)]
        for batch, value in zip(batches, (3.0, 4.0), strict=True):
            batch_any: Any = batch
            batch_any["reference_latents"] = [torch.full((1, 2, 2), value)]
            batch_any["reference_sizes"] = [(16, 16)]

        adapter.predict_velocity_batched(
            batches,
            [torch.tensor([0.0]), torch.tensor([0.0])],
        )
        self.assertEqual(adapter._forward_batch_sizes, [2])

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

    def test_invalid_logical_batch_dimension_is_rejected(self) -> None:
        adapter = self.make_adapter()
        batch = make_batch()
        batch["noisy_latents"] = torch.zeros(2, 4, 2)
        with self.assertRaisesRegex(ValueError, "singleton leading batch"):
            adapter.predict_velocity_batched([batch], [torch.tensor([0.0])])


class SamplerBatchingTest(unittest.TestCase):
    def test_trajectory_window_uses_request_generator(self) -> None:
        sampler = Sampler(
            steps=12,
            trajectory_window_size=3,
            trajectory_window_range=(1, 10),
        )
        first_generator = torch.Generator().manual_seed(1234)
        random.seed(1)
        first = sampler._select_trajectory_window(12, first_generator)

        for _ in range(100):
            random.random()
        second_generator = torch.Generator().manual_seed(1234)
        second = sampler._select_trajectory_window(12, second_generator)

        self.assertEqual(first, second)
        self.assertNotEqual(
            first_generator.get_state().tolist(),
            torch.Generator().manual_seed(1234).get_state().tolist(),
        )

    def test_trajectory_window_generators_are_independent(self) -> None:
        sampler = Sampler(steps=12, trajectory_window_size=3)
        untouched = torch.Generator().manual_seed(77)
        advanced = torch.Generator().manual_seed(77)
        _ = torch.rand(9, generator=advanced)

        expected = sampler._select_trajectory_window(
            12, torch.Generator().manual_seed(77)
        )
        self.assertEqual(sampler._select_trajectory_window(12, untouched), expected)
        sampler._select_trajectory_window(12, advanced)

    def test_trajectory_window_validates_size_and_reaches_endpoints(self) -> None:
        invalid = Sampler(steps=8, trajectory_window_size=0)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            invalid._select_trajectory_window(8, torch.Generator().manual_seed(1))

        sampler = Sampler(
            steps=8,
            trajectory_window_size=2,
            trajectory_window_range=(1, 7),
        )
        starts = {
            sampler._select_trajectory_window(8, torch.Generator().manual_seed(seed))[0]
            for seed in range(100)
        }
        self.assertIn(1, starts)
        self.assertIn(5, starts)

    def test_sampler_batches_model_forwards_and_preserves_order(self) -> None:
        sampler = Sampler(steps=2, cfg_scale=1.0)
        model = FakeSamplerModel()
        outputs = sampler.sample(
            model,  # type: ignore[arg-type]
            [
                SampleRequest(batch=make_sampler_batch(1.0)),
                SampleRequest(batch=make_sampler_batch(2.0)),
            ],
        )

        self.assertEqual(model.forward_batch_sizes, [2, 2])
        torch.testing.assert_close(outputs[0].final_latents, torch.tensor([[[-1.0]]]))
        torch.testing.assert_close(outputs[1].final_latents, torch.tensor([[[-2.0]]]))

    def test_mixed_cfg_uses_dummy_forward_without_guiding_missing_negative(
        self,
    ) -> None:
        sampler = Sampler(steps=1, cfg_scale=2.0)
        model = FakeSamplerModel()
        cond_a = make_sampler_batch(3.0)
        cond_b = make_sampler_batch(4.0)
        negative_a = make_sampler_batch(1.0)

        velocities = sampler.get_guided_velocity(
            model,  # type: ignore[arg-type]
            batches=[cond_a, cond_b],
            negative_batches=[negative_a, None],
            latents=[cond_a["noisy_latents"], cond_b["noisy_latents"]],
            timesteps=[torch.tensor([1.0]), torch.tensor([1.0])],
        )

        self.assertEqual(model.forward_batch_sizes, [2, 2])
        torch.testing.assert_close(velocities[0], torch.tensor([[[5.0]]]))
        torch.testing.assert_close(velocities[1], torch.tensor([[[4.0]]]))

    def test_cfg_all_and_none_negative_paths(self) -> None:
        sampler = Sampler(steps=1, cfg_scale=2.0)
        conditional = [make_sampler_batch(3.0), make_sampler_batch(4.0)]
        timesteps = [torch.tensor([1.0]), torch.tensor([1.0])]

        no_negative_model = FakeSamplerModel()
        no_negative = sampler.get_guided_velocity(
            no_negative_model,  # type: ignore[arg-type]
            conditional,
            [None, None],
            [batch["noisy_latents"] for batch in conditional],
            timesteps,
        )
        self.assertEqual(no_negative_model.forward_batch_sizes, [2])
        torch.testing.assert_close(no_negative[0], torch.tensor([[[3.0]]]))

        all_negative_model = FakeSamplerModel()
        all_negative = sampler.get_guided_velocity(
            all_negative_model,  # type: ignore[arg-type]
            conditional,
            [make_sampler_batch(1.0), make_sampler_batch(2.0)],
            [batch["noisy_latents"] for batch in conditional],
            timesteps,
        )
        self.assertEqual(all_negative_model.forward_batch_sizes, [2, 2])
        torch.testing.assert_close(all_negative[0], torch.tensor([[[5.0]]]))
        torch.testing.assert_close(all_negative[1], torch.tensor([[[6.0]]]))

    def test_cfg_renorm_is_applied_per_sample(self) -> None:
        sampler = Sampler(
            steps=1,
            cfg_scale=3.0,
            enable_cfg_renorm=True,
            cfg_renorm_min=0.0,
        )
        model = FakeSamplerModel()
        conditional = [make_sampler_batch(2.0), make_sampler_batch(8.0)]
        velocities = sampler.get_guided_velocity(
            model,  # type: ignore[arg-type]
            conditional,
            [make_sampler_batch(-2.0), make_sampler_batch(4.0)],
            [batch["noisy_latents"] for batch in conditional],
            [torch.tensor([1.0]), torch.tensor([1.0])],
        )
        torch.testing.assert_close(velocities[0], torch.tensor([[[2.0]]]))
        torch.testing.assert_close(velocities[1], torch.tensor([[[8.0]]]))

    def test_each_request_keeps_its_own_shifted_schedule(self) -> None:
        sampler = Sampler(
            steps=3,
            cfg_scale=1.0,
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
            FakeSamplerModel(),  # type: ignore[arg-type]
            [
                SampleRequest(batch=make_sampler_batch(0.0)),
                SampleRequest(batch=long_batch),
            ],
        )
        assert outputs[0].timesteps is not None
        assert outputs[1].timesteps is not None
        self.assertFalse(torch.equal(outputs[0].timesteps, outputs[1].timesteps))

    def test_stochastic_generators_are_isolated_per_sample(self) -> None:
        sampler = Sampler(steps=3, cfg_scale=1.0, solver=FlowSolver(eta=0.4))
        model = FakeSamplerModel()
        batched = sampler.sample(
            model,  # type: ignore[arg-type]
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
                FakeSamplerModel(),  # type: ignore[arg-type]
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

    def test_replay_returns_named_outputs(self) -> None:
        sampler = Sampler(solver=FlowSolver(eta=0.5), cfg_scale=1.0)
        model = FakeSamplerModel()
        batch = make_sampler_batch(0.0)
        output = sampler.compute_logprob_at_step(
            model,  # type: ignore[arg-type]
            [
                ReplayRequest(
                    batch=batch,
                    latent_t=torch.tensor([[[0.2]]]),
                    latent_next=torch.tensor([[[0.1]]]),
                    sigma=torch.tensor([0.8]),
                    sigma_next=torch.tensor([0.6]),
                )
            ],
        )[0]

        self.assertEqual(tuple(output.log_prob.shape), (1,))
        self.assertEqual(tuple(output.mean.shape), (1, 1, 1))


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

    def test_loss_scales_sum_to_one_per_chunk(self) -> None:
        mixin = self.make_mixin(train_batch_size=4, micro_batch_size=3)
        chunk_scale = 0.0
        for update in mixin.iter_micro_updates(list(range(11))):
            chunk_scale += update.loss_scale
            if update.is_sync_step:
                self.assertAlmostEqual(chunk_scale, 1.0)
                chunk_scale = 0.0
        self.assertEqual(chunk_scale, 0.0)

    def test_micro_batches_preserve_item_identity(self) -> None:
        # Precompute passes fill cached fields in place through these slices.
        mixin = self.make_mixin(train_batch_size=4, micro_batch_size=2)
        items = [_IndexedItem(rollout_idx=index) for index in range(3)]
        sliced = [
            item
            for micro_items in mixin.iter_train_micro_batches(items)
            for item in micro_items
        ]
        for original, batched in zip(items, sliced, strict=True):
            self.assertIs(original, batched)

    def test_empty_items_are_rejected(self) -> None:
        mixin = self.make_mixin(train_batch_size=4, micro_batch_size=1)
        with self.assertRaisesRegex(RuntimeError, "no train items"):
            next(mixin.iter_micro_updates([]))

    def test_batch_divisibility_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "train_micro_batch_size"):
            _ = self.make_mixin(train_batch_size=8, micro_batch_size=3).grad_acc_steps
        self.assertEqual(
            self.make_mixin(train_batch_size=8, micro_batch_size=2).grad_acc_steps, 4
        )

    def test_non_finite_loss_reports_rollout_indices(self) -> None:
        mixin = self.make_mixin(train_batch_size=4, micro_batch_size=2)
        items = [_IndexedItem(rollout_idx=3), _IndexedItem(rollout_idx=5)]
        mixin._check_finite_loss(torch.tensor(1.0), items)
        with self.assertRaisesRegex(RuntimeError, r"rollout_indices=\[3, 5\]"):
            mixin._check_finite_loss(torch.tensor(float("nan")), items)


class _OrderedDataset(Dataset[dict[str, int]]):
    def __init__(self, size: int) -> None:
        self.items = [{"index": index} for index in range(size)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, int]:
        return self.items[index]


class DataLoaderBatchingTest(unittest.TestCase):
    def test_list_collation_preserves_order_and_stateful_resume(self) -> None:
        dataset = _OrderedDataset(5)
        loader = StatefulDataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn,
        )
        iterator = iter(loader)
        first = next(iterator)
        state = loader.state_dict()
        expected_remaining = [item for batch in iterator for item in batch]

        resumed = StatefulDataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn,
        )
        resumed.load_state_dict(state)
        actual_remaining = [item for batch in resumed for item in batch]

        self.assertEqual(first, dataset.items[:2])
        self.assertEqual(actual_remaining, expected_remaining)


if __name__ == "__main__":
    unittest.main()
