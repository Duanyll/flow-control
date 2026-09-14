from contextlib import nullcontext
from functools import partial

import torch
import torch.distributed as dist
from diffusers import ModelMixin
from pydantic import PrivateAttr

from flow_control.adapters.base import BaseModelAdapter, Batch
from flow_control.data import (
    Index,
    IndexEntry,
    RowStream,
    build_loader,
    groups_plain,
    is_padding,
)
from flow_control.samplers import Executor, Sampler, SampleRequest
from flow_control.samplers.guidance import CfgPlusPlusGuidance, ClassifierFreeGuidance


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
        *,
        dummy_outputs: list[torch.Tensor] | None = None,
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
    # S2 cold start: rank 1 has never seen a sample. The adapter must transfer
    # a dummy collectively before its first forward, and all-empty stays a no-op.
    adapter = make_adapter(2)
    local = ([make_batch(4, 7.0)], [torch.tensor([0.5])]) if rank == 0 else ([], [])
    with torch.no_grad():
        assert adapter.predict_velocity_batched([], []) == []
        outputs = adapter.predict_velocity_batched(*local)
    assert adapter._forward_batch_sizes == [1]
    assert len(outputs) == len(local[0])
    assert adapter._dummy_sample is not None
    torch.testing.assert_close(
        adapter._dummy_sample[0]["noisy_latents"], torch.full((1, 4, 2), 7.0)
    )

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
    # A rank with a negative batch and one without issue the same forward
    # sequence: one predict_velocity_batched per variant per round, carrying
    # whatever branches the rank has (two here, one there); equal physical
    # forward counts are the adapter's job, so the fake only sees its own list.
    sampler = Sampler(guidance=ClassifierFreeGuidance(scale=2.0), steps=1)
    model = DistributedSamplerModel()
    batch = make_batch(1, value=3.0)
    negative_batch = make_batch(1, value=1.0) if rank == 0 else None
    run = sampler.make_run(
        SampleRequest(batch, negative_batch), plan=sampler.plan(batch)
    )
    Executor(model, sampler.variant_keys()).evaluate(
        [run.guided_velocity(batch["noisy_latents"], 0)]
    )
    assert model.calls == ([2] if rank == 0 else [1])

    # CFG++ cannot fall back to a conditional-only velocity: a request without
    # its negative batch is rejected when the run is built, before any
    # collective, so no rank can be left waiting in a forward.
    sampler = Sampler(guidance=CfgPlusPlusGuidance(), steps=1)
    try:
        sampler.make_run(SampleRequest(batch), plan=sampler.plan(batch))
    except ValueError as error:
        assert "negative batch" in str(error)
    else:
        raise AssertionError("Missing CFG++ negative batch was not rejected.")


def test_stream_drains_unequal_request_counts(rank: int) -> None:
    # Rank 0 streams two requests and rank 1 one. The executor keeps rank 1 in
    # the extra round with an empty call list and the adapter fills it with a
    # dummy forward, so both ranks run two forwards and neither hangs; a
    # one-step Euler run with velocity == latents lands on zero.
    adapter = make_adapter(1)
    sampler = Sampler(steps=1)
    values = [3.0, 5.0] if rank == 0 else [3.0]
    with torch.no_grad():
        runs = list(
            sampler.sample(
                adapter, [SampleRequest(batch=make_batch(4, value)) for value in values]
            )
        )
    assert adapter._forward_batch_sizes == [1, 1]
    assert len(runs) == len(values)
    for run in runs:
        torch.testing.assert_close(run.ctx.latents, torch.zeros(1, 4, 2))


class _TinyStore:
    index = Index([IndexEntry(str(i), 0, "", None, str(i)) for i in range(5)], {})

    def __len__(self) -> int:
        return 5

    def get(self, row_id: int) -> dict[str, int]:
        return {"index": row_id}


def test_final_padded_microbatch(rank: int) -> None:
    # Five rows over the ranks in one plan group of two microbatches per rank:
    # the tail is padded with repeated rows flagged ``__padding__`` so every rank
    # runs the same number of full microbatches (equal collectives), and the
    # padding is counted, never dropped.
    world_size = dist.get_world_size()
    stream = RowStream(
        _TinyStore(),
        partial(groups_plain, 5, 4 * world_size, 0, shuffle=False),
        rank,
        world_size,
    )
    loader = build_loader(stream, batch_size=2, num_workers=0)
    batches = list(loader)
    assert [len(batch) for batch in batches] == [2, 2]
    local_padding = sum(int(is_padding(item)) for batch in batches for item in batch)
    total_padding = torch.tensor(local_padding)
    dist.all_reduce(total_padding)
    assert total_padding.item() == 4 * world_size - 5


def main() -> None:
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        test_synchronized_fallback(rank)
        test_unequal_counts_pad_with_dummy_forwards(rank)
        test_mixed_cfg_is_globally_synchronized(rank)
        test_stream_drains_unequal_request_counts(rank)
        test_final_padded_microbatch(rank)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
