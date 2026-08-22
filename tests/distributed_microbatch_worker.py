from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist
from diffusers import ModelMixin
from pydantic import PrivateAttr
from torchdata.stateful_dataloader import StatefulDataLoader

from flow_control.adapters.base import BaseModelAdapter, Batch
from flow_control.samplers import Sampler, SampleRequest
from flow_control.samplers.executor import Run, execute
from flow_control.samplers.guidance import BaseGuidance, ClassifierFreeGuidance
from flow_control.samplers.plan import (
    BranchEvals,
    GuidanceOutput,
    GuidanceState,
    StepContext,
)
from flow_control.samplers.solver import FlowSolver
from flow_control.samplers.transforms import finalize_replay_state
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
        return batch["noisy_latents"]


class DistributedSamplerModel:
    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self) -> None:
        self.calls: list[int] = []

    def predict_velocity_batched(
        self,
        batches: list[Batch],
        timesteps: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        self.calls.append(len(batches))
        return [batch["clean_latents"] for batch in batches]


@dataclass(frozen=True, slots=True)
class _CountState(GuidanceState):
    count: int = 0


class _CountingGuidance(BaseGuidance):
    type: Literal["distributed_counting"] = "distributed_counting"

    def init_state(self) -> GuidanceState | None:
        return _CountState()

    def needs_negative(self) -> bool:
        return False

    def combine(
        self,
        evals: BranchEvals,
        ctx: StepContext,
        state: GuidanceState | None,
    ) -> tuple[GuidanceOutput, GuidanceState | None]:
        assert isinstance(state, _CountState)
        return (
            GuidanceOutput(velocity=evals.cond, branches=evals),
            _CountState(count=state.count + 1),
        )


def test_synchronized_fallback(rank: int) -> None:
    adapter = DistributedFakeAdapter.model_construct(arch="fake", type="fake")
    token_counts = (4, 4) if rank == 0 else (4, 5)
    adapter.predict_velocity_batched(
        [make_batch(tokens) for tokens in token_counts],
        [torch.tensor([0.5]), torch.tensor([0.5])],
    )
    assert adapter._forward_batch_sizes == [1, 1]


def test_length_mismatch_is_rejected(rank: int) -> None:
    adapter = DistributedFakeAdapter.model_construct(arch="fake", type="fake")
    logical_batch_size = 2 if rank == 0 else 1
    try:
        adapter.predict_velocity_batched(
            [make_batch(4) for _ in range(logical_batch_size)],
            [torch.tensor([0.5]) for _ in range(logical_batch_size)],
        )
    except ValueError as error:
        assert "same number of logical samples" in str(error)
    else:
        raise AssertionError("Distributed logical batch mismatch was not rejected.")


def test_mixed_cfg_is_globally_synchronized(rank: int) -> None:
    sampler = Sampler(guidance=ClassifierFreeGuidance(scale=2.0), steps=1)
    model = DistributedSamplerModel()
    batch = make_batch(1, value=3.0)
    negative_batch = make_batch(1, value=1.0) if rank == 0 else None
    sampler.get_guided_velocity(
        model,  # type: ignore[arg-type]
        batches=[batch],
        negative_batches=[negative_batch],
        latents=[batch["noisy_latents"]],
        timesteps=[torch.tensor([1.0])],
        sigmas=[1.0],
    )
    assert model.calls == [1, 1]


def test_sampler_request_count_mismatch_is_rejected(rank: int) -> None:
    sampler = Sampler(steps=1)
    model = DistributedSamplerModel()
    request_count = 2 if rank == 0 else 1
    try:
        sampler.sample(
            model,  # type: ignore[arg-type]
            [SampleRequest(batch=make_batch(1)) for _ in range(request_count)],
        )
    except ValueError as error:
        assert "same number of requests" in str(error)
    else:
        raise AssertionError("Distributed sampler request mismatch was not rejected.")


def _make_run(steps: int, value: float) -> Run:
    solver = FlowSolver()
    batch = make_batch(1, value=value)
    return Run(
        plan=finalize_replay_state(
            solver.plan(torch.linspace(1.0, 0.0, steps + 1).tolist())
        ),
        ctx=StepContext(
            latents=batch["noisy_latents"].float(),
            generator=None,
            solver_state=None,
            guidance_state=None,
        ),
        batch=batch,
        negative_batch=None,
    )


def test_heterogeneous_executor_pads_each_rendezvous(rank: int) -> None:
    adapter = DistributedFakeAdapter.model_construct(arch="fake", type="fake")
    guidance = _CountingGuidance()
    runs = (
        [_make_run(4, 1.0), _make_run(4, 2.0)]
        if rank == 0
        else [_make_run(4, 1.0), _make_run(6, 2.0)]
    )
    for run in runs:
        run.ctx.guidance_state = guidance.init_state()

    list(execute(adapter, runs, guidance))

    # Rank 1 pads its one-request A group while rank 0 evaluates [A, A]; once
    # rank 0 drains, it submits one dummy request for rank 1's B group.
    assert adapter._forward_batch_sizes == [2, 2, 2, 2, 1, 1, 1, 1, 1, 1]
    assert all(torch.isfinite(run.ctx.latents).all() for run in runs)
    expected_counts = (4, 4) if rank == 0 else (4, 6)
    for run, expected_count in zip(runs, expected_counts, strict=True):
        state = run.ctx.guidance_state
        assert isinstance(state, _CountState)
        assert state.count == expected_count


def test_heterogeneous_executor_handles_early_rank_drain(rank: int) -> None:
    adapter = DistributedFakeAdapter.model_construct(arch="fake", type="fake")
    run = _make_run(4 if rank == 0 else 6, 1.0)

    list(execute(adapter, [run], ClassifierFreeGuidance()))

    assert adapter._forward_batch_sizes == [1, 1, 1, 1, 1, 1]
    assert torch.isfinite(run.ctx.latents).all()


class _TinyDataset:
    def __len__(self) -> int:
        return 5

    def __getitem__(self, index: int) -> dict[str, int]:
        return {"index": index}


def test_final_padded_microbatch(rank: int) -> None:
    dataset = PaddingAwareDatasetWrapper(_TinyDataset())  # type: ignore[arg-type]
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
        test_length_mismatch_is_rejected(rank)
        test_mixed_cfg_is_globally_synchronized(rank)
        test_sampler_request_count_mismatch_is_rejected(rank)
        test_heterogeneous_executor_pads_each_rendezvous(rank)
        test_heterogeneous_executor_handles_early_rank_drain(rank)
        test_final_padded_microbatch(rank)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
