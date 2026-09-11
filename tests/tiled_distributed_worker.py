"""Run with ``torchrun --standalone --nproc-per-node=2`` to check real tile padding."""

import torch
import torch.distributed as dist
from diffusers import ModelMixin
from pydantic import PrivateAttr

from flow_control.adapters.base import BaseModelAdapter, Batch
from flow_control.processors.tiles import TileConfig
from flow_control.samplers.tiled import TiledModel


class TileAdapter(BaseModelAdapter[ModelMixin, Batch]):
    supports_dense_batching = True
    dense_batch_fields = ("image_size", "noisy_latents")
    patch_size: int = 1
    vae_scale_factor: int = 1
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


def main() -> None:
    dist.init_process_group("gloo")
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        for mixed_ranks in (False, True):
            height, width = (4, 6) if rank == 0 else (6, 6)
            adapter = TileAdapter.model_construct(arch="fake", type="fake")
            wrapped = TiledModel(adapter)
            x = torch.arange(height * width, dtype=torch.float32)[None, :, None]
            x.requires_grad_()
            batch = {"image_size": (height, width), "noisy_latents": x}
            if not mixed_ranks or rank == 1:
                batch["tiling"] = TileConfig(
                    tile_size=(4, 4), overlap=(2, 2)
                ).model_dump()
            velocity = wrapped.predict_velocity_batched([batch], [torch.tensor([0.5])])[
                0
            ]
            torch.testing.assert_close(velocity, 2 * x, rtol=0, atol=0)
            velocity.sum().backward()
            torch.testing.assert_close(x.grad, torch.full_like(x, 2), rtol=0, atol=0)
            # Padding yields four compatible inputs on each rank, including the
            # plain rank. Dense forwards permit different token lengths across ranks.
            assert adapter._forward_batch_sizes == [4]
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
