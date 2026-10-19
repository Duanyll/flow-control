import os
from typing import Any, ClassVar, cast
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn as nn
from diffusers import ModelMixin
from pydantic import Field, PrivateAttr
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from flow_control.adapters.base import BaseModelAdapter, Batch
from flow_control.contrib.momentum_guidance import Momentum
from flow_control.samplers import (
    BaseTrajectory,
    BaseTransition,
    Cfg,
    Euler,
    Executor,
    Grid,
    Model,
    RootFrame,
    Row,
    SaGrid,
    StepFrame,
    Tiled,
    variants,
)
from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.logging import get_logger
from flow_control.utils.tiling import TileLayout

logger = get_logger(__name__)

LORA: dict[str, Any] = {
    "default": {"type": "lora", "rank": 2, "alpha": 2},
    "other": {"type": "lora", "rank": 2, "alpha": 2},
}


class TinyBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Linear(width, width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(self.proj(inputs))


class TinyTransformer(ModelMixin):
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


class TinyAdapter(BaseModelAdapter[ModelMixin, Batch]):
    arch: str = "tiny"
    type: str = "tiny"
    hf_model: HfModelLoader[ModelMixin] = Field(
        default_factory=lambda: HfModelLoader(
            library="custom", class_name="Tiny", pretrained_model_id="tiny"
        )
    )
    supports_dense_batching: ClassVar[bool] = True
    dense_batch_fields = ("image_size", "noisy_latents")
    lora_target_modules = ("proj",)
    _forward_count: int = PrivateAttr(default=0)

    def _predict_velocity(
        self,
        batch: Batch,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        self._forward_count += 1
        return self.module()(batch["noisy_latents"])


class FallbackTinyAdapter(TinyAdapter):
    supports_dense_batching: ClassVar[bool] = False


def make_adapter(
    cls: type[TinyAdapter],
    device: torch.device,
    *,
    variants: dict[str, Any],
    trainable: bool,
    micro_batch_size: int = 1,
) -> TinyAdapter:
    """A seeded adapter realized like ``load_transformer`` does, with the loader faked."""
    torch.manual_seed(2718)
    adapter = cls(
        variants=variants, trainable=trainable, micro_batch_size=micro_batch_size
    )
    adapter.transformer = TinyTransformer().requires_grad_(trainable)
    with patch.object(HfModelLoader, "load_model", return_value=True):
        adapter.load_transformer(torch.device("cpu"))
    with torch.no_grad():  # LoRA B starts at zero; make every branch distinct
        for name in variants:
            for weight in adapter._weights(name):
                weight.normal_()
    nn.Module.to(adapter.transformer, device=device)
    return adapter


def make_batch(tokens: int, device: torch.device) -> Batch:
    latents = torch.randn(1, tokens, 8, device=device)
    return {
        "image_size": (tokens * 16, 16),
        "clean_latents": torch.zeros_like(latents),
        "noisy_latents": latents,
    }


def point(batch: Any, sigma: float = 0.5) -> Row:
    return {**batch, "sigma": sigma, "_generator": None, "_frame": RootFrame(None)}


def sample(model: Any, sampler: BaseTrajectory, rows: list[Row]) -> list[Row]:
    executor = Executor(model, variants(sampler))
    return list(executor.stream((row, sampler(row)) for row in rows))


def run_case(
    adapter: TinyAdapter,
    optimizer: torch.optim.Optimizer,
    token_counts: tuple[int, ...],
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


def make_lora_pair(
    mesh: DeviceMesh,
    device: torch.device,
    *,
    fallback: bool = False,
    train_base: bool = False,
) -> tuple[TinyAdapter, TinyAdapter]:
    """Two identically seeded adapters; the first is sharded with checkpointing."""
    adapter_class = FallbackTinyAdapter if fallback else TinyAdapter
    sharded = make_adapter(
        adapter_class, device, variants=LORA, trainable=train_base, micro_batch_size=2
    )
    control = make_adapter(adapter_class, device, variants=LORA, trainable=train_base)
    sharded.enable_gradient_checkpointing()
    fully_shard(cast(Any, sharded.transformer).block, mesh=mesh)
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
    """Rethink A1: the forward's variant must survive checkpoint recomputation,
    and rank-local branch dummies must participate in FSDP backward.
    """
    sharded, control = make_lora_pair(mesh, device, train_base=not different_variants)
    prediction = (
        Model(variant=("default", "other")[dist.get_rank()])
        if different_variants
        else Cfg(
            scale=2.0, negative_condition="positive", negative=Model(variant="base")
        )
    )
    torch.manual_seed(100 + dist.get_rank())
    batch = make_batch(4, device)
    outputs = [
        Executor(adapter, variants(prediction)).evaluate([prediction(point(batch))])[0]
        for adapter in (sharded, control)
    ]
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
    """Rethink A4: tiled evaluation runs sequential leaf forwards under FSDP and
    stitches them; the sharded gradients must match an untiled unsharded oracle.
    """
    sharded, control = make_lora_pair(mesh, device, fallback=True)
    torch.manual_seed(200 + dist.get_rank())
    batch = make_batch(7, device)
    timesteps = [torch.tensor([0.5], device=device)]
    stride = sharded.patch_size * sharded.vae_scale_factor
    # Seven tokens tall, one wide: two-token tiles overlapping by one give six.
    tiled: Any = {
        **batch,
        "tiling": TileLayout(
            tile_size=2 * stride, overlap=stride, stride=stride
        ).model_dump(),
    }
    actual = Executor(sharded).evaluate([Tiled()(point(tiled))])[0]
    # The toy network is tokenwise: an untiled, unsharded forward is an oracle
    # for overlap normalization and for every tile's gradient.
    expected = control.predict_velocity_batched([batch], timesteps)[0]
    assert sharded._forward_count == 6
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    actual.square().mean().backward()
    expected.square().mean().backward()
    compare_gradients(sharded, control, "tiled sequential forwards")

    # Deeply nested local joins must preserve FSDP's forward order. SA also
    # exercises two sequential EMA updates inside its first transition.
    composed = SaGrid(
        steps=4,
        eta=0,
        prediction=Tiled(
            prediction=Cfg(
                scale=2,
                renorm=True,
                negative_condition="positive",
                negative=Model(variant="base"),
                prediction=Momentum(alpha=0.5, beta=0.25, prediction=Model()),
            )
        ),
    )
    with torch.no_grad():
        (actual_run,) = sample(sharded, composed, [point(tiled, 1.0)])
        (expected_run,) = sample(control, composed, [point(batch, 1.0)])
    torch.testing.assert_close(
        actual_run["noisy_latents"],
        expected_run["noisy_latents"],
        rtol=1e-5,
        atol=1e-6,
    )
    logger.info("GPU regression passed: Tiled(CFG(Momentum(Model))) with SA and FSDP")

    # Complete tiled sampling must match the tokenwise whole-image oracle.
    # Run real bf16 Linear forwards while every step tensor stays fp32.
    def check_step(row: Row, step: StepFrame) -> None:
        assert step.latents.dtype == step.next_latents.dtype == torch.float32
        assert step.velocity is not None and step.velocity.dtype == torch.float32

    sampler = Grid(steps=3, transition=Euler(prediction=Momentum(alpha=0.5, beta=0.25)))
    nn.Module.to(control.transformer, dtype=torch.bfloat16)
    observed = point(tiled, 1.0)
    observed["_frame"] = RootFrame(None, [(BaseTransition, check_step)])
    with torch.no_grad():
        (tiled_run,) = sample(control, sampler, [observed])
        (plain_run,) = sample(control, sampler, [point(batch, 1.0)])
    torch.testing.assert_close(
        tiled_run["noisy_latents"], plain_run["noisy_latents"], rtol=1e-5, atol=1e-6
    )
    logger.info("GPU regression passed: bf16 tiled sampling with fp32 steps")


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    try:
        mesh = dist.device_mesh.init_device_mesh("cuda", (dist.get_world_size(),))
        adapter = make_adapter(
            TinyAdapter, device, variants={}, trainable=True, micro_batch_size=2
        )
        fully_shard(cast(Any, adapter.transformer).block, mesh=mesh)
        fully_shard(adapter.transformer, mesh=mesh)
        optimizer = torch.optim.AdamW(adapter.trainable_parameters(), lr=1e-3)

        run_case(adapter, optimizer, (4, 4))
        run_case(adapter, optimizer, (4, 5))
        token_counts = (4, 4) if dist.get_rank() == 0 else (4, 5)
        run_case(adapter, optimizer, token_counts)
        # Sampler-rethink S1: unequal logical counts (two versus four samples
        # at micro_batch_size=2) pad rank 0 with a dummy chunk under autograd;
        # forward counts must match and the folded backward must complete.
        count = 2 if dist.get_rank() == 0 else 4
        run_case(adapter, optimizer, (4,) * count)
        run_branch_case(mesh, device, different_variants=False)
        run_branch_case(mesh, device, different_variants=True)
        run_tiled_case(mesh, device)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
