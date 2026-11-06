"""What runs the loaded transformer: attention backend, activation checkpointing,
per-block torch.compile and FSDP2 units (docs/adapters/parallelize.md)."""

import functools
import importlib
import inspect
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar, Literal, cast

import torch
from pydantic import BaseModel, ConfigDict, model_validator
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts

from flow_control.utils.logging import get_logger

logger = get_logger(__name__)

CheckpointingPolicy = Literal["none", "block", "selective"]
_SDPA_BACKENDS = (None, "native", "_native_flash", "_native_efficient", "_native_cudnn")


class TorchCompileConfig(BaseModel):
    """`torch.compile` of every block's forward. Only the adapter's `dynamic_dims`
    are symbolic; every other dimension specializes."""

    model_config = ConfigDict(extra="forbid")

    backend: str = "inductor"
    mode: str | None = None
    """`"max-autotune-no-cudagraphs"` selects Inductor's GEMM templates; each new
    graph then takes minutes (cached on disk)."""
    fullgraph: bool = True
    """A graph break inside a block is a bug: a `ContextVar.set`, an `.item()`.
    False needs `gradient_checkpointing: "none"`: a block that falls back to eager
    would recompute under the backward's variant."""
    options: dict[str, str | int | bool] | None = None
    recompile_limit: int = 64
    """Graphs per compiled frame: weight sets x grad on / off x physical batch
    sizes x guard splits. Checkpointed blocks of every class share the wrapper's
    frame. Past it, `fullgraph` raises; without it the block runs eager."""

    def compile(self, function: Callable[..., Any]) -> Callable[..., Any]:
        # Process-wide. The config module declares `recompile_limit = 8` without an
        # annotation, so ty infers `Literal[8]`.
        torch._dynamo.config.recompile_limit = self.recompile_limit  # ty: ignore[invalid-assignment]
        options = self.model_dump(exclude_none=True, exclude={"recompile_limit"})
        return torch.compile(function, dynamic=False, **options)


class ParallelizeMixin(BaseModel):
    """Turns the loaded transformer into the module that runs. The adapter knows its
    blocks, their sequence dims and which ops to keep; the trainer supplies the mesh."""

    attention_backend: str | None = None
    """A diffusers `AttentionBackendName` (`"flash"`: the external flash-attn).
    None keeps the model's, `native` SDPA. Like `torch_compile`, applied by
    `parallelize`; serving loads without it."""
    gradient_checkpointing: CheckpointingPolicy = "block"
    """`"block"`: recompute each block in backward. `"selective"`: keep the
    attention outputs too, recompute the rest."""
    torch_compile: TorchCompileConfig | None = None
    block_classes: ClassVar[tuple[str, ...]] = ()
    """Class names of the repeated blocks; empty takes the transformer's
    `_no_split_modules` or `_repeated_blocks`."""
    dynamic_dims: ClassVar[dict[str, dict[str, int]]] = {}
    """Block class -> {forward argument: sequence dim} marked dynamic under compile;
    a tuple argument is marked element-wise."""

    # ------------------------- what the adapter provides ------------------------ #

    @property
    @abstractmethod
    def transformer(self) -> nn.Module: ...

    @abstractmethod
    def pin_variant(self, checkpoint: Callable[..., Any]) -> Callable[..., Any]: ...

    @model_validator(mode="after")
    def check_fullgraph(self):
        compile = self.torch_compile
        if compile and not compile.fullgraph and self.gradient_checkpointing != "none":
            raise ValueError(
                'torch_compile.fullgraph: false needs gradient_checkpointing: "none": '
                "a block that falls back to eager would recompute under the "
                "backward's variant."
            )
        return self

    # --------------------------------- entry ----------------------------------- #

    def parallelize(self, mesh: DeviceMesh | None) -> None:
        """Once, after `load_transformer` and before `to_empty` / `dcp.load`, in the
        same order on every rank. Adds no state and changes no FQN."""
        self._set_attention_backend()
        blocks = self._blocks()
        unknown = set(self.dynamic_dims) - {type(block).__name__ for block in blocks}
        if unknown:
            raise ValueError(f"dynamic_dims names no block class: {sorted(unknown)}.")
        if (
            self.gradient_checkpointing == "selective"
            and self.attention_backend not in (*_SDPA_BACKENDS, "flash", "flash_varlen")
        ):
            logger.warning(
                f'"selective" keeps nothing for attention_backend '
                f"{self.attention_backend!r}: _saved_ops does not list its op, so "
                "each block is recomputed whole."
            )
        for block in blocks:
            self._checkpoint(block)  # inside the compiled region
            if self.torch_compile is not None:
                self._compile(block)
        self._shard(blocks, mesh)
        logger.info(
            f"Parallelized {len(blocks)} blocks: checkpointing "
            f"{self.gradient_checkpointing!r}, compile {self.torch_compile}, "
            f"attention {self.attention_backend or 'default'}."
        )

    # ---------------------------- overridable steps ----------------------------- #

    def _set_attention_backend(self) -> None:
        if self.attention_backend is None:
            return
        set_backend = getattr(self.transformer, "set_attention_backend", None)
        if set_backend is None:
            raise ValueError(
                f"attention_backend needs diffusers' attention dispatch; "
                f"{type(self.transformer).__name__} has no set_attention_backend."
            )
        set_backend(self.attention_backend)

    def _blocks(self) -> list[nn.Module]:
        """Outermost modules of the block classes, full variant copies included."""
        transformer = self.transformer
        names = (
            self.block_classes
            or getattr(transformer, "_no_split_modules", None)
            or getattr(transformer, "_repeated_blocks", None)
        )
        if not names:
            raise ValueError(
                f"{type(transformer).__name__} has no _no_split_modules or "
                "_repeated_blocks; set the adapter's block_classes."
            )
        blocks: list[nn.Module] = []
        prefixes: list[str] = []
        for fqn, module in transformer.named_modules():  # parents before children
            nested = any(fqn.startswith(prefix) for prefix in prefixes)
            if type(module).__name__ in names and not nested:
                blocks.append(module)
                prefixes.append(fqn + ".")
        return blocks

    def _checkpoint(self, block: nn.Module) -> None:
        """Checkpoint `block.forward` when grad is enabled. A compiled block recomputes
        inside its compiled backward, under the variant its graph traced; an eager
        block re-reads the ContextVar, so its recompute is pinned."""
        if self.gradient_checkpointing == "none":
            return
        run = (
            checkpoint
            if self.torch_compile is not None
            else self.pin_variant(checkpoint)
        )
        options: dict[str, Any] = {"use_reentrant": False}
        if self.gradient_checkpointing == "selective":
            saved = list(self._saved_ops())
            options["context_fn"] = functools.partial(
                create_selective_checkpoint_contexts, saved
            )  # Dynamo traces this partial, not an explicit noop_context_fn
        forward = block.forward

        @functools.wraps(forward)
        def checkpointed(*args: Any, **kwargs: Any) -> Any:
            if not torch.is_grad_enabled():
                return forward(*args, **kwargs)
            return run(forward, *args, **options, **kwargs)

        block.forward = checkpointed

    def _saved_ops(self) -> set[Any]:
        """What `"selective"` keeps for backward: the attention op of the backend."""
        aten = torch.ops.aten
        ops: set[Any] = {
            aten._scaled_dot_product_flash_attention.default,
            aten._scaled_dot_product_efficient_attention.default,
            aten._scaled_dot_product_cudnn_attention.default,
        }
        if self.attention_backend in ("flash", "flash_varlen"):  # the external FA2
            importlib.import_module("flash_attn")  # registers torch.ops.flash_attn
            ops |= {
                torch.ops.flash_attn._flash_attn_forward.default,
                torch.ops.flash_attn._flash_attn_varlen_forward.default,
            }
        return ops

    def _compile(self, block: nn.Module) -> None:
        """Compile `block.forward` and mark its `dynamic_dims` in front of it. The
        marks run eagerly after FSDP2's pre-forward hook, which replaces inputs
        that require grad (`RegisterPostBackwardFunction`) and drops earlier marks."""
        forward = block.forward
        compiled = cast(TorchCompileConfig, self.torch_compile).compile(forward)
        dims = self.dynamic_dims.get(type(block).__name__)
        if not dims:
            block.forward = compiled
            return
        signature = inspect.signature(forward)
        if unknown := set(dims) - set(signature.parameters):
            raise ValueError(
                f"dynamic_dims[{type(block).__name__!r}] names no forward argument: "
                f"{sorted(unknown)}."
            )

        @functools.wraps(forward)
        def marked(*args: Any, **kwargs: Any) -> Any:
            self._mark_dynamic(dims, signature.bind(*args, **kwargs).arguments)
            return compiled(*args, **kwargs)

        block.forward = marked

    def _mark_dynamic(self, dims: dict[str, int], arguments: dict[str, Any]) -> None:
        """`maybe_mark_dynamic`: a guard on the dim (Inductor's SDPA backward lowering
        checks `S % 8`) adds a graph set instead of raising like `mark_dynamic`."""
        for name, dim in dims.items():
            value = arguments.get(name)
            for tensor in value if isinstance(value, tuple | list) else (value,):
                if isinstance(tensor, torch.Tensor):
                    torch._dynamo.maybe_mark_dynamic(tensor, dim)

    def _shard(self, blocks: list[nn.Module], mesh: DeviceMesh | None) -> None:
        """FSDP2 units: every block, every full variant copy, then the root. None:
        `fully_shard`'s default mesh, the whole world."""
        transformer = cast(Any, self.transformer)
        for block in blocks:
            fully_shard(block, mesh=mesh)
        for copy in transformer.variants.values():
            fully_shard(copy, mesh=mesh)
        fully_shard(transformer, mesh=mesh)


if __name__ == "__main__":
    # One GPU: `uv run -m flow_control.adapters.parallelize`. Forward under "default",
    # backward under "old": every policy must reproduce the plain gradients, and a
    # new sequence length must not compile a new graph.
    from unittest.mock import patch

    import torch.distributed as dist
    from diffusers import ModelMixin
    from pydantic import Field
    from rich import print
    from torch._dynamo.utils import counters
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor
    from torch.nn import functional as F

    from flow_control.adapters.base import BaseModelAdapter, Batch
    from flow_control.utils.hf_model import HfModelLoader

    class Block(nn.Module):
        def __init__(self, width: int = 64) -> None:
            super().__init__()
            self.qkv = nn.Linear(width, 3 * width)  # wider than its input: fused LoRA
            self.out = nn.Linear(width, width)

        def forward(self, hidden_states: torch.Tensor, rope: torch.Tensor):
            q, k, v = (
                t.unflatten(-1, (4, -1)).transpose(1, 2)
                for t in self.qkv(hidden_states).chunk(3, -1)
            )
            attn = F.scaled_dot_product_attention(q * rope, k * rope, v)
            return hidden_states + self.out(attn.transpose(1, 2).flatten(-2))

    class Tiny(ModelMixin):
        _repeated_blocks = ["Block"]

        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList(Block() for _ in range(3))

        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            rope = torch.arange(x.shape[1], device=x.device)[:, None].div(9).cos()
            for block in self.blocks:
                x = block(x * t.view(-1, 1, 1), rope)
            return x

    class TinyAdapter(BaseModelAdapter[ModelMixin, Batch]):
        type: str = "tiny"
        hf_model: HfModelLoader[ModelMixin] = Field(
            default_factory=lambda: HfModelLoader(
                library="custom", class_name="Tiny", pretrained_model_id="tiny"
            )
        )
        lora_target_modules = ("qkv", "out")
        dynamic_dims = {"Block": {"hidden_states": 1, "rope": 0}}

        def _predict_velocity(self, batch: Batch, timestep: torch.Tensor):
            return self.module()(batch["noisy_latents"], timestep)

    def build(**fields: Any) -> TinyAdapter:
        torch.manual_seed(0)
        variants = {
            "default": {"type": "lora", "rank": 4, "init": "gaussian"},
            "old": {"type": "ema", "of": "default", "decay": 0.5},
        }
        adapter = TinyAdapter.model_validate({"variants": variants, **fields})
        adapter.hf_model.dtype = torch.float32
        adapter.transformer = Tiny().cuda()
        with patch.object(HfModelLoader, "load_model", return_value=True):
            adapter.load_transformer(torch.device("cuda"))
        with torch.no_grad():
            for name, std in (("default", 0.1), ("old", 0.3)):
                for tensor in adapter._weights(name):
                    tensor.normal_(
                        0, std, generator=torch.Generator("cuda").manual_seed(1)
                    )
        return adapter

    def run(adapter: TinyAdapter, length: int) -> list[torch.Tensor]:
        generator = torch.Generator("cuda").manual_seed(length)
        x = torch.randn(2, length, 64, device="cuda", generator=generator)
        with adapter.use_variant("default"):
            y = adapter.module()(x, torch.full((2,), 0.5, device="cuda"))
        with adapter.use_variant("old"):  # what an unpinned eager recompute would read
            y.square().mean().backward()
        grads = []
        for p in adapter.trainable_parameters():
            grad = cast(torch.Tensor, p.grad)
            grads.append(grad.full_tensor() if isinstance(grad, DTensor) else grad)
            p.grad = None
        return grads

    dist.init_process_group("nccl", "tcp://127.0.0.1:29533", rank=0, world_size=1)
    mesh = init_device_mesh("cuda", (1,))
    lengths = (41, 53)  # neither a multiple of 8: one graph set for SDPA's backward
    plain = build(gradient_checkpointing="none")
    reference = {n: run(plain, n) for n in lengths}
    for policy in ("block", "selective"):
        for compiled in (False, True):
            torch._dynamo.reset()
            counters.clear()
            config = {"mode": "default"} if compiled else None
            adapter = build(gradient_checkpointing=policy, torch_compile=config)
            adapter.parallelize(mesh)
            graphs = []
            for n in lengths:
                for got, want in zip(run(adapter, n), reference[n], strict=True):
                    torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-5)
                graphs.append(counters["stats"]["unique_graphs"])
            assert graphs[0] == graphs[-1], f"a new length recompiled: {graphs}"
            print(f"{policy:9s} compiled={compiled!s:5s} ok, graphs {graphs[-1]}")
    dist.destroy_process_group()
