"""Run with ``torchrun --standalone --nproc-per-node=2`` to check tiled evaluation across ranks."""

from typing import Any, cast

import torch
import torch.distributed as dist
from diffusers import ModelMixin
from pydantic import PrivateAttr

from flow_control.adapters.base import BaseModelAdapter, Batch
from flow_control.samplers.evaluation import predict_velocity
from flow_control.utils.tiling import TileLayout


class TileAdapter(BaseModelAdapter[ModelMixin, Batch]):
    supports_dense_batching = True
    dense_batch_fields = ("image_size", "noisy_latents")
    _forward_batch_sizes: list[int] = PrivateAttr(default_factory=list)

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    def _predict_velocity(self, batch: Batch, timestep: torch.Tensor) -> torch.Tensor:
        self._forward_batch_sizes.append(batch["noisy_latents"].shape[0])
        return 2 * batch["noisy_latents"]


def make_batch(height: int, width: int) -> Batch:
    x = torch.arange(height * width, dtype=torch.float32)[None, :, None]
    batch: dict[str, Any] = {
        "image_size": (height, width),
        "noisy_latents": x.requires_grad_(),
        "tiling": TileLayout(tile_size=4, overlap=2, stride=1).model_dump(),
    }
    return cast(Batch, batch)


def main() -> None:
    dist.init_process_group("gloo")
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        timesteps = [torch.tensor([0.5])]

        # Equal tile counts: four tiles per rank share one dense forward, and the
        # stitched velocity and gradients match the tokenwise oracle up to feathering rounding.
        adapter = TileAdapter.model_construct(
            arch="fake", type="fake", micro_batch_size=4
        )
        batch = make_batch(6, 6)
        velocity = predict_velocity(adapter, [batch], timesteps)[0]
        x = batch["noisy_latents"]
        torch.testing.assert_close(velocity, 2 * x, rtol=1e-6, atol=1e-5)
        velocity.sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 2), rtol=1e-6, atol=1e-5)
        assert adapter._forward_batch_sizes == [4]

        # Unequal tile counts (two versus four) at micro_batch_size=2 are legal:
        # rank 0 runs its one real chunk plus a dummy chunk to match rank 1's
        # two chunks (sampler-rethink S1), and the stitch stays exact.
        adapter = TileAdapter.model_construct(
            arch="fake", type="fake", micro_batch_size=2
        )
        batch = make_batch(4, 6) if rank == 0 else make_batch(6, 6)
        velocity = predict_velocity(adapter, [batch], timesteps)[0]
        torch.testing.assert_close(
            velocity, 2 * batch["noisy_latents"], rtol=1e-6, atol=1e-5
        )
        assert adapter._forward_batch_sizes == ([2, 1] if rank == 0 else [2, 2])
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
