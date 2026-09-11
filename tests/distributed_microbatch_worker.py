from contextlib import nullcontext

import torch
import torch.distributed as dist
from diffusers import ModelMixin
from pydantic import PrivateAttr
from torchdata.stateful_dataloader import StatefulDataLoader

from flow_control.adapters.base import BaseModelAdapter, Batch
from flow_control.samplers import Sampler, SampleRequest
from flow_control.samplers.guidance import CfgPlusPlusGuidance, ClassifierFreeGuidance
from flow_control.training.data import (
    DistributedBucketSampler,
    PaddingAwareDatasetWrapper,
    collate_fn,
)


def make_batch(tokens: int, value: float = 0.0) -> Batch:
    latents = torch.full((1, tokens, 2), value)
    return {
        "image_size": (32, 32),
        "clean_latents": torch.zeros_like(latents),
        "noisy_latents": latents,
    }


class DistributedFakeAdapter(BaseModelAdapter[ModelMixin, Batch]):
    supports_dense_batching = True
    dense_batch_fields = ("image_size", "noisy_latents")
    _forward_batch_sizes: list[int] = PrivateAttr(default_factory=list)
    _scale: torch.Tensor = PrivateAttr(
        default_factory=lambda: torch.tensor(1.0, requires_grad=True)
    )
    _forward_scales: list[torch.Tensor] = PrivateAttr(default_factory=list)
    """One fresh leaf per forward, so backward reaching a forward is observable."""

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
        self._forward_scales.append(torch.tensor(1.0, requires_grad=True))
        return batch["noisy_latents"] * self._scale * self._forward_scales[-1]


class DistributedSamplerModel:
    device = torch.device("cpu")
    dtype = torch.float32
    micro_batch_size = 1

    def __init__(self) -> None:
        self.calls: list[int] = []

    def use_variant(self, variant: str | None):
        if variant is not None:
            raise ValueError(f"Test model has no variant {variant!r}.")
        return nullcontext()

    def predict_velocity_batched(
        self,
        batches: list[Batch],
        timesteps: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        self.calls.append(len(batches))
        return [batch["clean_latents"] for batch in batches]


def make_adapter(micro_batch_size: int) -> DistributedFakeAdapter:
    return DistributedFakeAdapter.model_construct(
        arch="fake", type="fake", micro_batch_size=micro_batch_size
    )


def test_synchronized_fallback(rank: int) -> None:
    # One rank's incompatible shapes force sequential forwards on every rank.
    adapter = make_adapter(2)
    token_counts = (4, 4) if rank == 0 else (4, 5)
    adapter.predict_velocity_batched(
        [make_batch(tokens) for tokens in token_counts],
        [torch.tensor([0.5]), torch.tensor([0.5])],
    )
    assert adapter._forward_batch_sizes == [1, 1]

    # Sampler-rethink S1: the fallback path runs one forward per sample, so a
    # shorter rank pads to the longest rank's chunk with dummy forwards.
    adapter = make_adapter(2)
    token_counts = (4, 5) if rank == 0 else (4,)
    outputs = adapter.predict_velocity_batched(
        [make_batch(tokens) for tokens in token_counts],
        [torch.tensor([0.5]) for _ in token_counts],
    )
    assert adapter._forward_batch_sizes == [1, 1]
    assert [output.shape[1] for output in outputs] == list(token_counts)


def test_unequal_counts_pad_with_dummy_forwards(rank: int) -> None:
    # Sampler-rethink S1: unequal logical counts no longer raise. Rank 0 has
    # chunks [2, 2, 1], rank 1 [2, 1] plus one dummy chunk: forward counts match
    # (dense sizes need not), and every real output stays in order.
    adapter = make_adapter(2)
    values = [float(value) for value in range(5 if rank == 0 else 3)]
    with torch.no_grad():
        outputs = adapter.predict_velocity_batched(
            [make_batch(4, value) for value in values],
            [torch.tensor([0.5]) for _ in values],
        )
    assert adapter._forward_batch_sizes == ([2, 2, 1] if rank == 0 else [2, 1, 1])
    for value, output in zip(values, outputs, strict=True):
        torch.testing.assert_close(output, torch.full((1, 4, 2), value))

    # A drained rank keeps participating with an empty list; its dummy comes
    # from the sample cached by the previous call.
    adapter._forward_batch_sizes.clear()
    local = ([make_batch(4, 7.0)], [torch.tensor([0.5])]) if rank == 0 else ([], [])
    with torch.no_grad():
        outputs = adapter.predict_velocity_batched(*local)
    assert adapter._forward_batch_sizes == [1]
    assert len(outputs) == len(local[0])

    # Under autograd the dummy output is folded into the first real output with
    # zero weight: backward must reach every forward (a detached dummy would
    # leave its leaf without a gradient and desync FSDP's reduce-scatters),
    # while the gradient value stays the real samples' alone.
    adapter = make_adapter(1)
    count = 2 if rank == 0 else 1
    outputs = adapter.predict_velocity_batched(
        [make_batch(4, 3.0) for _ in range(count)],
        [torch.tensor([0.5]) for _ in range(count)],
    )
    assert adapter._forward_batch_sizes == [1, 1]
    torch.stack([output.mean() for output in outputs]).sum().backward()
    assert all(scale.grad is not None for scale in adapter._forward_scales)
    torch.testing.assert_close(adapter._scale.grad, torch.tensor(3.0 * count))


def test_mixed_cfg_is_globally_synchronized(rank: int) -> None:
    sampler = Sampler(guidance=ClassifierFreeGuidance(scale=2.0), steps=1)
    model = DistributedSamplerModel()
    batch = make_batch(1, value=3.0)
    negative_batch = make_batch(1, value=1.0) if rank == 0 else None
    sampler.get_guided_velocity(
        model,
        batches=[batch],
        negative_batches=[negative_batch],
        latents=[batch["noisy_latents"]],
        timesteps=[torch.tensor([1.0])],
        sigmas=[1.0],
    )
    assert model.calls == [1, 1]

    # CFG++ cannot fall back to a conditional-only velocity. If only one rank
    # lacks its negative batch, every rank must fail before model collectives.
    sampler = Sampler(guidance=CfgPlusPlusGuidance(), steps=1)
    model = DistributedSamplerModel()
    try:
        sampler.get_guided_velocity(
            model,
            batches=[batch],
            negative_batches=[negative_batch],
            latents=[batch["noisy_latents"]],
            timesteps=[torch.tensor([1.0])],
            sigmas=[1.0],
            sigma_nexts=[0.0],
        )
    except ValueError as error:
        assert "required guidance branch" in str(error)
    else:
        raise AssertionError(
            "Missing CFG++ negative batch was not rejected on every rank."
        )
    assert model.calls == []


def test_sampler_request_count_mismatch_is_rejected(rank: int) -> None:
    sampler = Sampler(steps=1)
    model = DistributedSamplerModel()
    request_count = 2 if rank == 0 else 1
    try:
        sampler.sample(
            model,
            [SampleRequest(batch=make_batch(1)) for _ in range(request_count)],
        )
    except ValueError as error:
        assert "same number of requests" in str(error)
    else:
        raise AssertionError("Distributed sampler request mismatch was not rejected.")


class _TinyDataset:
    def __len__(self) -> int:
        return 5

    def __getitem__(self, index: int) -> dict[str, int]:
        return {"index": index}


def test_final_padded_microbatch(rank: int) -> None:
    dataset = PaddingAwareDatasetWrapper(_TinyDataset())
    sampler = DistributedBucketSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        grad_acc_steps=4,
    )
    loader = StatefulDataLoader(
        dataset,
        batch_size=2,
        sampler=sampler,
        collate_fn=collate_fn,
    )
    batches = list(loader)
    assert [len(batch) for batch in batches] == [2, 2]
    local_padding = sum(
        int(item.get("_is_padding_sample", False))
        for batch in batches
        for item in batch
    )
    total_padding = torch.tensor(local_padding)
    dist.all_reduce(total_padding)
    assert total_padding.item() == 3


def main() -> None:
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        test_synchronized_fallback(rank)
        test_unequal_counts_pad_with_dummy_forwards(rank)
        test_mixed_cfg_is_globally_synchronized(rank)
        test_sampler_request_count_mismatch_is_rejected(rank)
        test_final_padded_microbatch(rank)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
