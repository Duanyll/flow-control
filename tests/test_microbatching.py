import unittest
from contextlib import closing, nullcontext
from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import patch

import torch
from diffusers import Flux2Transformer2DModel, ModelMixin, QwenImage21Transformer2DModel
from pydantic import PrivateAttr

from flow_control.adapters.base import BaseModelAdapter, Batch
from flow_control.adapters.flux1.base import Flux1Adapter
from flow_control.adapters.flux2.base import Flux2Adapter
from flow_control.adapters.flux2.kv import Flux2KVAdapter
from flow_control.adapters.qwen21 import QwenImage21Adapter
from flow_control.samplers import Executor, Sampler, SampleRequest
from flow_control.samplers.guidance import ClassifierFreeGuidance
from flow_control.samplers.shift import LinearShift
from flow_control.samplers.solver import FlowSolver
from flow_control.training.mixins.microbatch import MicrobatchTrainMixin
from flow_control.utils.model_cache import cache_enabled, cache_fields


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


class FakeCacheAdapter(FakeDenseAdapter):
    # This cache depends on the condition, so it cannot use layout-only sharing.
    supports_dense_batching = False
    _seen_rows: list[list[Batch]] = PrivateAttr(default_factory=list)
    _cache_misses: int = PrivateAttr(default=0)

    def predict_velocity_batched(self, batches, timesteps, *, dummy_outputs=None):
        self._seen_rows.append(list(batches))
        return super().predict_velocity_batched(
            batches, timesteps, dummy_outputs=dummy_outputs
        )

    def _predict_velocity(self, batch: Batch, timestep: torch.Tensor) -> torch.Tensor:
        data = cast(dict[str, Any], batch)
        if "_condition" not in data:
            data["_condition"] = batch["clean_latents"].clone()
            self._cache_misses += 1
        # A different CFG branch or logical sample must not reuse this value.
        torch.testing.assert_close(data["_condition"], batch["clean_latents"])
        return super()._predict_velocity(batch, timestep)


class IdRecordingTransformer(ModelMixin):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.config = {"guidance_embeds": False}
        self.batch_sizes: list[int] = []

    def forward(
        self,
        hidden_states: torch.Tensor,
        txt_ids: torch.Tensor,
        img_ids: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor]:
        b = hidden_states.shape[0]
        self.batch_sizes.append(b)
        if img_ids.ndim == 3:
            assert img_ids.shape[0] == txt_ids.shape[0] == b
        assert img_ids.shape[-2] == hidden_states.shape[1]
        return (hidden_states + img_ids.float().mean() + txt_ids.float().mean(),)


class FakeFallbackAdapter(FakeDenseAdapter):
    supports_dense_batching = False


class FakeSamplerModel:
    device = torch.device("cpu")
    dtype = torch.float32
    micro_batch_size = 2
    """Two runs share each forward, so cross-request batching is observable."""

    def __init__(self) -> None:
        self.forward_batch_sizes: list[int] = []

    def use_variant(self, variant: str | None):
        if variant is not None:
            raise ValueError(f"Test model has no variant {variant!r}.")
        return nullcontext()

    def predict_velocity_batched(
        self,
        batches: list[Batch],
        timesteps: list[torch.Tensor],
        *,
        dummy_outputs: list[torch.Tensor] | None = None,
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
    def make_adapter(self, micro_batch_size: int = 2) -> FakeDenseAdapter:
        return FakeDenseAdapter.model_construct(
            arch="fake", type="fake", micro_batch_size=micro_batch_size
        )

    def test_dense_inputs_use_one_forward_per_chunk(self) -> None:
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

        # Sampler-rethink S1: lists longer than micro_batch_size are chunked in
        # order, and an empty list is a legal call that runs no forward.
        adapter = self.make_adapter()
        outputs = adapter.predict_velocity_batched(
            [make_batch(float(value)) for value in range(5)],
            [torch.tensor([0.0])] * 5,
        )
        self.assertEqual(adapter._forward_batch_sizes, [2, 2, 1])
        for value, output in enumerate(outputs):
            torch.testing.assert_close(output, torch.full((1, 4, 2), float(value)))
        self.assertEqual(adapter.predict_velocity_batched([], []), [])
        self.assertEqual(adapter._forward_batch_sizes, [2, 2, 1])

        # The dense/fallback decision is per chunk: a ragged pair after a
        # compatible pair falls back only for its own chunk.
        adapter = self.make_adapter()
        outputs = adapter.predict_velocity_batched(
            [make_batch(), make_batch(), make_batch(tokens=3), make_batch(tokens=5)],
            [torch.tensor([0.0])] * 4,
        )
        self.assertEqual(adapter._forward_batch_sizes, [2, 1, 1])
        self.assertEqual([output.shape[1] for output in outputs], [4, 4, 3, 5])

        # The temporary-batch refactor lost Flux ID reuse across solver steps.
        # Exercise the real adapters/collator while mocking only the transformer;
        # unequal trajectory lengths also change physical B from two to one.
        for adapter_type in (Flux1Adapter, Flux2Adapter):
            with self.subTest(adapter=adapter_type.__name__):
                flux = adapter_type(micro_batch_size=2)
                transformer = IdRecordingTransformer()
                flux.transformer = cast(Any, transformer)
                rows: list[Any] = [
                    {
                        **make_batch(float(i)),
                        "prompt_embeds": torch.zeros(1, 3, 2),
                        "pooled_prompt_embeds": torch.zeros(1, 2),
                    }
                    for i in range(2)
                ]
                outputs_by_mode = []
                for use_cache in (True, False):
                    flux.use_cache = use_cache
                    runs = [
                        Sampler(steps=steps).make_run(SampleRequest(row))
                        for steps, row in zip((1, 3), rows, strict=True)
                    ]
                    transformer.batch_sizes.clear()
                    with (
                        torch.no_grad(),
                        patch.object(
                            adapter_type,
                            "_make_batch_img_ids",
                            side_effect=flux._make_batch_img_ids,
                        ) as make_ids,
                    ):
                        finished = list(Executor(flux).stream(runs))
                    self.assertEqual(transformer.batch_sizes, [2, 1, 1])
                    self.assertEqual(make_ids.call_count, 1 if use_cache else 3)
                    self.assertTrue(all(not cache_fields(run.row) for run in finished))
                    outputs_by_mode.append([run.ctx.latents for run in finished])
                for cached, uncached in zip(*outputs_by_mode, strict=True):
                    torch.testing.assert_close(cached, uncached, rtol=0, atol=0)

    def test_incompatible_inputs_fall_back_to_single_sample(self) -> None:
        adapter = self.make_adapter()
        outputs = adapter.predict_velocity_batched(
            [make_batch(tokens=3), make_batch(tokens=5)],
            [torch.tensor([0.0]), torch.tensor([0.0])],
        )

        self.assertEqual(adapter._forward_batch_sizes, [1, 1])
        self.assertEqual([output.shape[1] for output in outputs], [3, 5])

        # Real KV adapters must preserve per-sample/CFG caches through the
        # sequential fallback, including mixed T2I and reference-image rows.
        for adapter_type, transformer in (
            (
                QwenImage21Adapter,
                QwenImage21Transformer2DModel(
                    num_layers=1,
                    num_attention_heads=2,
                    attention_head_dim=16,
                    context_in_dim=8,
                    axes_dims_rope=(4, 6, 6),
                ),
            ),
            (
                Flux2KVAdapter,
                Flux2Transformer2DModel(
                    num_layers=1,
                    num_single_layers=1,
                    num_attention_heads=2,
                    attention_head_dim=16,
                    joint_attention_dim=8,
                    axes_dims_rope=(4, 4, 4, 4),
                    guidance_embeds=False,
                ),
            ),
        ):
            with self.subTest(adapter=adapter_type.__name__):
                kv = adapter_type(micro_batch_size=2)
                kv.hf_model.dtype = torch.float32
                kv.transformer = transformer
                kv._install_modules()
                channels = kv.latent_channels * kv.patch_size**2
                rows: list[Any] = []
                for refs in (0, 1):
                    rows.append(
                        {
                            "image_size": (32, 32),
                            "noisy_latents": torch.randn(1, 4, channels),
                            "prompt_embeds": torch.randn(1, 3, 8),
                            "prompt_embeds_mask": torch.ones(1, 3, dtype=torch.long),
                            "image_pad_mask": torch.tensor(
                                [[False, bool(refs), False]]
                            ),
                            "reference_latents": [torch.randn(1, 4, channels)] * refs,
                            "reference_sizes": [(32, 32)] * refs,
                        }
                    )
                requests = [
                    SampleRequest(
                        row,
                        cast(Batch, {**row, "prompt_embeds": -row["prompt_embeds"]}),
                    )
                    for row in rows
                ]
                sampler = Sampler(steps=3, guidance=ClassifierFreeGuidance(scale=2))
                modes: list[str | None] = []
                handle = transformer.register_forward_pre_hook(
                    lambda module, args, kwargs, modes=modes: modes.append(
                        kwargs.get("kv_cache_mode")
                    ),
                    with_kwargs=True,
                )
                results = []
                for reuse in (True, False):
                    kv.use_cache = reuse
                    modes.clear()
                    with torch.no_grad():
                        completed = list(sampler.sample(kv, requests))
                    results.append([run.ctx.latents for run in completed])
                    self.assertEqual(
                        modes.count("cached"),
                        (8 if isinstance(kv, QwenImage21Adapter) else 4)
                        if reuse
                        else 0,
                    )
                    self.assertTrue(
                        all(
                            not cache_fields(run.row)
                            and not cache_fields(run.negative_row or {})
                            for run in completed
                        )
                    )
                handle.remove()
                for cached, uncached in zip(*results, strict=True):
                    torch.testing.assert_close(cached, uncached, atol=2e-5, rtol=2e-5)
                kv.use_cache = True
                loss = torch.stack(
                    [
                        output.square().mean()
                        for output in kv.predict_velocity_batched(
                            rows, [torch.tensor([0.5])] * 2
                        )
                    ]
                ).sum()
                loss.backward()
                self.assertTrue(
                    any(
                        p.grad is not None and p.grad.abs().sum() > 0
                        for p in transformer.parameters()
                    )
                )
                self.assertTrue(all(not cache_fields(row) for row in rows))

                if isinstance(kv, QwenImage21Adapter):
                    # The trainer materializes a meta model before loading DCP.
                    # Upstream plain RoPE tables stay meta, and nonpersistent
                    # timestep buffers otherwise become uninitialized here.
                    restored = QwenImage21Adapter()
                    restored.hf_model.dtype = torch.float32
                    with torch.device("meta"):
                        restored.transformer = (
                            QwenImage21Transformer2DModel.from_config(
                                transformer.config
                            )
                        )
                    restored._install_modules()
                    restored.transformer.to_empty(device="cpu")
                    restored.transformer.load_state_dict(transformer.state_dict())
                    with torch.no_grad():
                        expected = kv.predict_velocity_batched(
                            rows, [torch.tensor([0.5])] * 2
                        )
                        actual = restored.predict_velocity_batched(
                            rows, [torch.tensor([0.5])] * 2
                        )
                    for reference, materialized in zip(expected, actual, strict=True):
                        torch.testing.assert_close(
                            reference, materialized, rtol=0, atol=0
                        )

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

        # Dummy snapshots may be broadcast before Executor has a chance to
        # release the run: exclude runtime fields before detach/CPU transfer.
        first["_kv"] = torch.ones(1)
        first["key"] = "first"
        with (
            torch.no_grad(),
            patch("flow_control.adapters.base.dist.is_initialized", return_value=True),
            patch("flow_control.adapters.base.dist.get_world_size", return_value=2),
            patch("flow_control.adapters.base.dist.get_rank", return_value=0),
            patch("flow_control.adapters.base.dist.all_reduce"),
            patch("flow_control.adapters.base.dist.broadcast_object_list") as broadcast,
        ):
            adapter.predict_velocity_batched(batches, [torch.zeros(1)] * 2)
            adapter._share_dummy(0)
        payload = broadcast.call_args.args[0][0][0]
        self.assertFalse(any(key.startswith("_") for key in payload))
        self.assertEqual(payload["noisy_latents"].device.type, "cpu")
        self.assertIn("_kv", first)
        self.assertEqual(payload["key"], "first")

    def test_dense_and_fallback_gradients_match(self) -> None:
        dense = self.make_adapter()
        fallback = FakeFallbackAdapter.model_construct(
            arch="fake", type="fake", micro_batch_size=2
        )
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


def guided_velocities(
    sampler: Sampler, model: FakeSamplerModel, requests: list[SampleRequest]
) -> list[torch.Tensor]:
    """Training-style evaluation of plan item 0 for every request, batched."""
    runs = [
        sampler.make_run(request, plan=sampler.plan(request.row))
        for request in requests
    ]
    return Executor(model, sampler.variant_keys()).evaluate(
        [run.guided_velocity(run.row["noisy_latents"], 0) for run in runs]
    )


class SamplerBatchingTest(unittest.TestCase):
    def test_mixed_cfg_batches_only_the_branches_present(self) -> None:
        # A request without a negative row skips the optional branch instead
        # of getting a dummy forward; the present branches share one forward
        # and the missing one falls back to the conditional velocity.
        sampler = Sampler(steps=1, guidance=ClassifierFreeGuidance(scale=2.0))
        model = FakeSamplerModel()
        velocities = guided_velocities(
            sampler,
            model,
            [
                SampleRequest(make_sampler_batch(3.0), make_sampler_batch(1.0)),
                SampleRequest(make_sampler_batch(4.0)),
            ],
        )

        self.assertEqual(model.forward_batch_sizes, [3])
        torch.testing.assert_close(velocities[0], torch.tensor([[[5.0]]]))
        torch.testing.assert_close(velocities[1], torch.tensor([[[4.0]]]))

        # Runtime cache must survive copied/dense rows, but stop at the
        # executor boundary even when streaming pauses or fails mid-trajectory.
        sampler = Sampler(steps=3, guidance=ClassifierFreeGuidance(scale=2.0))
        request = SampleRequest(make_batch(3.0), make_batch(1.0))
        request.row["clean_latents"].fill_(3.0)
        assert request.negative_row is not None
        request.negative_row["clean_latents"].fill_(1.0)
        adapter = FakeCacheAdapter.model_construct(
            arch="fake", type="fake", micro_batch_size=2
        )
        with torch.no_grad():
            finished = list(sampler.sample(adapter, [request]))
        self.assertEqual(adapter._cache_misses, 2)
        self.assertEqual(adapter._forward_batch_sizes, [1] * 6)
        self.assertTrue(
            all(not cache_fields(row) for rows in adapter._seen_rows for row in rows)
        )
        self.assertFalse(cache_fields(finished[0].row))
        self.assertFalse(cache_fields(finished[0].negative_row or {}))
        self.assertFalse(cache_enabled.get())

        # Neither a direct no-grad call nor an executor's training forward may
        # read or persist runtime fields. A poisoned input would break the probe.
        direct_row = cast(
            Batch, {**make_batch(), "_condition": torch.full((1, 4, 2), 99.0)}
        )
        with torch.no_grad():
            adapter.predict_velocity_batched([direct_row], [torch.zeros(1)])
        torch.testing.assert_close(
            cache_fields(direct_row)["_condition"], torch.full((1, 4, 2), 99.0)
        )
        run = sampler.make_run(request)
        Executor(adapter).evaluate([run.guided_velocity(run.row["noisy_latents"], 0)])
        self.assertTrue(all(not cache_fields(row) for row in adapter._seen_rows[-1]))

        for exit_mode in ("close", "error"):
            with self.subTest(exit_mode=exit_mode), torch.no_grad():
                adapter = FakeCacheAdapter.model_construct(
                    arch="fake", type="fake", micro_batch_size=2
                )
                runs = [
                    Sampler(steps=n).make_run(SampleRequest(make_batch()))
                    for n in (1, 3)
                ]
                with closing(Executor(adapter).stream(runs)) as stream:
                    self.assertIs(next(stream), runs[0])
                    first, pending = adapter._seen_rows[0]
                    self.assertFalse(cache_fields(first))
                    self.assertTrue(cache_fields(pending))
                    self.assertFalse(cache_enabled.get())
                    if exit_mode == "error":
                        with (
                            patch.object(
                                FakeCacheAdapter,
                                "_predict_velocity",
                                side_effect=RuntimeError("forward failed"),
                            ),
                            self.assertRaisesRegex(RuntimeError, "forward failed"),
                        ):
                            next(stream)
                self.assertFalse(cache_fields(pending))
                self.assertFalse(cache_enabled.get())

    def test_cfg_renorm_is_applied_per_sample(self) -> None:
        sampler = Sampler(
            steps=1,
            guidance=ClassifierFreeGuidance(scale=3.0, renorm=True, renorm_min=0.0),
        )
        velocities = guided_velocities(
            sampler,
            FakeSamplerModel(),
            [
                SampleRequest(make_sampler_batch(2.0), make_sampler_batch(-2.0)),
                SampleRequest(make_sampler_batch(8.0), make_sampler_batch(4.0)),
            ],
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
        short, long = sampler.sample(
            FakeSamplerModel(),
            [
                SampleRequest(row=make_sampler_batch(0.0)),
                SampleRequest(row=long_batch),
            ],
        )
        self.assertNotEqual(
            [item.sigma for item in short.plan], [item.sigma for item in long.plan]
        )

    def test_stochastic_generators_are_isolated_per_sample(self) -> None:
        sampler = Sampler(steps=3, solver=FlowSolver(eta=0.4))
        model = FakeSamplerModel()
        batched = list(
            sampler.sample(
                model,
                [
                    SampleRequest(
                        row=make_sampler_batch(0.0),
                        generator=torch.Generator().manual_seed(7),
                    ),
                    SampleRequest(
                        row=make_sampler_batch(0.0),
                        generator=torch.Generator().manual_seed(19),
                    ),
                ],
            )
        )
        self.assertEqual(model.forward_batch_sizes, [2, 2, 2])

        individual = [
            next(
                iter(
                    sampler.sample(
                        FakeSamplerModel(),
                        [
                            SampleRequest(
                                row=make_sampler_batch(0.0),
                                generator=torch.Generator().manual_seed(seed),
                            )
                        ],
                    )
                )
            )
            for seed in (7, 19)
        ]
        for batched_run, individual_run in zip(batched, individual, strict=True):
            torch.testing.assert_close(
                batched_run.ctx.latents, individual_run.ctx.latents
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
