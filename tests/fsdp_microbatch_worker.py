import os
from copy import deepcopy
from typing import ClassVar

import torch
import torch.distributed as dist
import torch.nn as nn
from diffusers import ModelMixin
from diffusers.loaders import PeftAdapterMixin
from peft import LoraConfig
from pydantic import PrivateAttr
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from flow_control.adapters.base import BaseModelAdapter, Batch
from flow_control.samplers.evaluation import evaluate
from flow_control.samplers.guidance import ClassifierFreeGuidance
from flow_control.samplers.plan import EvalRequest, StepContext
from flow_control.samplers.tiled import Tiled
from flow_control.utils.logging import get_logger

logger = get_logger(__name__)


class TinyBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Linear(width, width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(self.proj(inputs))


class TinyTransformer(ModelMixin, PeftAdapterMixin):
    _no_split_modules = ["TinyBlock"]
    _supports_gradient_checkpointing = True

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.block = TinyBlock(width)
        self.output = nn.Linear(width, width)
        self.gradient_checkpointing = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = (
            self._gradient_checkpointing_func(self.block, inputs)
            if torch.is_grad_enabled() and self.gradient_checkpointing
            else self.block(inputs)
        )
        return self.output(hidden)


class TinyAdapter(BaseModelAdapter[TinyTransformer, Batch]):
    supports_dense_batching: ClassVar[bool] = True
    dense_batch_fields = ("image_size", "noisy_latents")
    _tiny_transformer: TinyTransformer = PrivateAttr(default_factory=TinyTransformer)
    _forward_count: int = PrivateAttr(default=0)

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
        self._forward_count += 1
        return self.transformer(batch["noisy_latents"])


class FallbackTinyAdapter(TinyAdapter):
    supports_dense_batching: ClassVar[bool] = False


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


def make_peft_pair(
    mesh: DeviceMesh,
    device: torch.device,
    *,
    fallback: bool = False,
    train_base: bool = False,
) -> tuple[TinyAdapter, TinyAdapter]:
    """Load both variants before sharding; the oracle has no checkpointing."""
    torch.manual_seed(2718)
    adapter_class = FallbackTinyAdapter if fallback else TinyAdapter
    sharded = adapter_class.model_construct(arch="tiny", type="tiny")
    for variant in ("default", "other"):
        sharded.transformer.add_adapter(
            LoraConfig(
                r=2, lora_alpha=2, target_modules=["proj"], init_lora_weights=False
            ),
            adapter_name=variant,
        )
    sharded.transformer.set_adapter("default")
    for name, parameter in sharded.transformer.named_parameters():
        parameter.requires_grad_(train_base or "lora_" in name)
    nn.Module.to(sharded.transformer, device=device)
    control = adapter_class.model_construct(arch="tiny", type="tiny")
    control.transformer = deepcopy(sharded.transformer)
    sharded.transformer.enable_gradient_checkpointing()
    fully_shard(sharded.transformer.block, mesh=mesh)
    fully_shard(sharded.transformer, mesh=mesh)
    return sharded, control


def compare_gradients(sharded: TinyAdapter, control: TinyAdapter, label: str) -> None:
    """FSDP gradients must match the average of ordinary per-rank objectives."""
    for (name, parameter), (reference_name, reference) in zip(
        sharded.transformer.named_parameters(),
        control.transformer.named_parameters(),
        strict=True,
    ):
        assert name == reference_name
        assert parameter.requires_grad == reference.requires_grad, name
        if not reference.requires_grad:
            continue
        expected = (
            reference.grad.detach().clone()
            if reference.grad is not None
            else torch.zeros_like(reference)
        )
        dist.all_reduce(expected)
        expected /= dist.get_world_size()
        gradient = parameter.grad
        actual = (
            gradient.full_tensor()
            if isinstance(gradient, DTensor)
            else gradient
            if gradient is not None
            else torch.zeros_like(expected)
        )
        torch.testing.assert_close(
            actual, expected, rtol=2e-5, atol=2e-6, msg=f"{label}: {name}"
        )
    if dist.get_rank() == 0:
        logger.info("FSDP regression passed: %s", label)


def run_branch_case(
    mesh: DeviceMesh, device: torch.device, *, different_variants: bool
) -> None:
    """Rethink A1: variant restoration must survive checkpoint recomputation,
    and rank-local branch dummies must participate in FSDP backward.
    """
    sharded, control = make_peft_pair(mesh, device, train_base=not different_variants)
    guidance = (
        ClassifierFreeGuidance(positive_variant=["default", "other"])
        if different_variants
        else ClassifierFreeGuidance(
            scale=2.0,
            positive_variant="default",
            negative_variant="base",
            negative_condition="positive",
        )
    )
    torch.manual_seed(100 + dist.get_rank())
    batch = make_batch(4, device)
    outputs = []
    for adapter in (sharded, control):
        output = evaluate(
            model=adapter,
            guidance=guidance,
            batches=[batch],
            negative_batches=[None],
            requests=[EvalRequest(batch["noisy_latents"], 0.5)],
            contexts=[
                StepContext(
                    batch["noisy_latents"],
                    None,
                    None,
                    None,
                    item_index=dist.get_rank() if different_variants else 0,
                    num_items=2,
                )
            ],
        )[0].velocity
        outputs.append(output)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-5, atol=1e-6)
    for output in outputs:
        output.square().mean().backward()
    label = (
        "rank-dependent LoRA branches"
        if different_variants
        else "active/base checkpoint replay"
    )
    compare_gradients(sharded, control, label)


def run_tiled_case(mesh: DeviceMesh, device: torch.device) -> None:
    """Rethink A4: discarded padding previously omitted whole FSDP backward
    graphs when unequal tile counts forced sequential leaf forwards.
    """
    sharded, control = make_peft_pair(mesh, device, fallback=True)
    torch.manual_seed(200 + dist.get_rank())
    batch = make_batch(4 if dist.get_rank() == 0 else 7, device)
    timesteps = [torch.tensor([0.5], device=device)]
    tiled = Tiled(tile_size=(32, 16), overlap=(16, 0)).wrap(sharded)
    actual = tiled.predict_velocity_batched([batch], timesteps)[0]
    # The toy network is tokenwise: an untiled, unsharded forward is an oracle
    # for both overlap normalization and all real/padded-tile gradients.
    expected = control.predict_velocity_batched([batch], timesteps)[0]
    assert sharded._forward_count == 6
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    actual.square().mean().backward()
    expected.square().mean().backward()
    compare_gradients(sharded, control, "unequal tiled fallback counts")


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
        run_branch_case(mesh, device, different_variants=False)
        run_branch_case(mesh, device, different_variants=True)
        run_tiled_case(mesh, device)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
