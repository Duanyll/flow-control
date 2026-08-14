import os

import torch
import torch.distributed as dist
import torch.nn as nn
from diffusers import ModelMixin
from pydantic import PrivateAttr
from torch.distributed.fsdp import fully_shard

from flow_control.adapters.base import BaseModelAdapter, Batch


class TinyBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Linear(width, width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(self.proj(inputs))


class TinyTransformer(ModelMixin):
    _no_split_modules = ["TinyBlock"]

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.block = TinyBlock(width)
        self.output = nn.Linear(width, width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.block(inputs))


class TinyAdapter(BaseModelAdapter[TinyTransformer, Batch]):
    supports_dense_batching = True
    dense_batch_fields = ("image_size", "noisy_latents")
    _tiny_transformer: TinyTransformer = PrivateAttr(default_factory=TinyTransformer)

    @property
    def transformer(self) -> TinyTransformer:
        return self._tiny_transformer

    @transformer.setter
    def transformer(self, value: TinyTransformer) -> None:
        self._tiny_transformer = value

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    def _predict_velocity(
        self,
        batch: Batch,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        return self.transformer(batch["noisy_latents"])


def make_batch(tokens: int, device: torch.device) -> Batch:
    latents = torch.randn(1, tokens, 8, device=device)
    return {
        "image_size": (tokens * 16, 16),
        "clean_latents": torch.zeros_like(latents),
        "noisy_latents": latents,
    }


def run_case(
    adapter: TinyAdapter,
    optimizer: torch.optim.Optimizer,
    token_counts: tuple[int, int],
) -> None:
    device = adapter.device
    outputs = adapter.predict_velocity_batched(
        [make_batch(tokens, device) for tokens in token_counts],
        [torch.tensor([0.5], device=device) for _ in token_counts],
    )
    loss = torch.stack([output.square().mean() for output in outputs]).mean()
    loss.backward()
    for parameter in adapter.transformer.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    try:
        mesh = dist.device_mesh.init_device_mesh("cuda", (dist.get_world_size(),))
        adapter = TinyAdapter.model_construct(arch="tiny", type="tiny")
        nn.Module.to(adapter.transformer, device=device)
        fully_shard(adapter.transformer.block, mesh=mesh)
        fully_shard(adapter.transformer, mesh=mesh)
        optimizer = torch.optim.AdamW(adapter.transformer.parameters(), lr=1e-3)

        run_case(adapter, optimizer, (4, 4))
        run_case(adapter, optimizer, (4, 5))
        token_counts = (4, 4) if dist.get_rank() == 0 else (4, 5)
        run_case(adapter, optimizer, token_counts)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
