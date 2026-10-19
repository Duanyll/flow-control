"""Model variants: named weight sets of one adapter, selected per forward through a ContextVar."""

import copy
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from typing import Annotated, Any, Literal, cast

import torch
from pydantic import BaseModel, ConfigDict
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F

from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.registry import Registry, RegistryUnion


class BaseWarmup(BaseModel):
    """EMA decay warmup schedule."""

    model_config = ConfigDict(extra="forbid")
    type: str

    def get_decay(self, step: int, base_decay: float) -> float:
        raise NotImplementedError


warmup_registry: Registry[BaseWarmup] = Registry("warmup", base=BaseWarmup)


@warmup_registry.register("none")
class NoWarmup(BaseWarmup):
    type: Literal["none"] = "none"

    def get_decay(self, step: int, base_decay: float) -> float:
        return base_decay


@warmup_registry.register("standard")
class StandardWarmup(BaseWarmup):
    """`(1 + step) / (10 + step)`, capped at the base decay."""

    type: Literal["standard"] = "standard"

    def get_decay(self, step: int, base_decay: float) -> float:
        return min((1 + step) / (10 + step), base_decay)


@warmup_registry.register("linear_ramp")
class LinearRampWarmup(BaseWarmup):
    """NFT-style: 0 for `flat_steps`, then a linear ramp up to the base decay."""

    type: Literal["linear_ramp"] = "linear_ramp"
    flat_steps: int = 0
    ramp_rate: float = 0.001

    def get_decay(self, step: int, base_decay: float) -> float:
        if step < self.flat_steps:
            return 0.0
        return min((step - self.flat_steps) * self.ramp_rate, base_decay)


Warmup = Annotated[BaseWarmup, RegistryUnion(warmup_registry, "type")]


class BaseVariant(BaseModel):
    """A named weight set built on the variant `of` ("base" is the loaded weights)."""

    model_config = ConfigDict(extra="forbid")
    type: str
    of: str = "base"


variant_registry: Registry[BaseVariant] = Registry("variant", base=BaseVariant)


@variant_registry.register("lora")
class LoraVariant(BaseVariant):
    type: Literal["lora"] = "lora"
    rank: int
    alpha: float | None = None
    bias: bool = False
    target_modules: list[str] | Literal["all-linear"] | None = None
    """None: the adapter's `lora_target_modules`."""
    extra_trainable: list[str] | None = None
    """Modules cloned from `of` into this variant; None: the adapter's `lora_extra_trainable`."""
    path: str | None = None
    """Initial values: a native LoRA file (`save_lora`)."""
    init: Literal["kaiming", "gaussian"] = "kaiming"
    """A's init: `nn.Linear`'s kaiming-uniform, or N(0, 1/rank) (PEFT's `gaussian`)."""
    trainable: bool = True

    @property
    def scaling(self) -> float:
        return (self.rank if self.alpha is None else self.alpha) / self.rank


@variant_registry.register("full")
class FullVariant(BaseVariant):
    type: Literal["full"] = "full"
    hf_model: HfModelLoader[Any] | None = None
    trainable: bool = False


@variant_registry.register("ema")
class EmaVariant(BaseVariant):
    type: Literal["ema"] = "ema"
    of: str = "default"
    decay: float
    every: Literal["step", "epoch"] = "step"
    warmup: Warmup = NoWarmup()


@variant_registry.register("alias")
class AliasVariant(BaseVariant):
    type: Literal["alias"] = "alias"
    of: str


Variant = Annotated[BaseVariant, RegistryUnion(variant_registry, "type")]


class LoraPair(nn.Module):
    """One variant's low-rank update `B(A(x)) * scaling`; slots are cast to the input dtype."""

    def __init__(self, linear: nn.Linear, variant: LoraVariant) -> None:
        super().__init__()
        self.scaling = variant.scaling
        factory: dict[str, Any] = {
            "device": linear.weight.device,
            "dtype": linear.weight.dtype,
        }
        self.A = nn.Linear(linear.in_features, variant.rank, bias=False, **factory)
        self.B = nn.Linear(variant.rank, linear.out_features, variant.bias, **factory)
        if variant.init == "gaussian":
            nn.init.normal_(self.A.weight, std=1 / variant.rank)
        for parameter in self.B.parameters():
            nn.init.zeros_(parameter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = None if self.B.bias is None else self.B.bias.to(x.dtype)
        hidden = F.linear(x, self.A.weight.to(x.dtype))
        return F.linear(hidden, self.B.weight.to(x.dtype), bias) * self.scaling


class LoraLinear(nn.Linear):
    """`nn.Linear` plus per-variant LoRA slots; the active variant's slot, if any, is added."""

    lora: nn.ModuleDict
    active: ContextVar[str]

    @classmethod
    def convert(cls, linear: nn.Linear, active: ContextVar[str]) -> "LoraLinear":
        linear.__class__ = cls
        module = cast(LoraLinear, linear)
        module.lora = nn.ModuleDict()
        module.active = active
        return module

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        name = self.active.get()
        if name in self.lora:
            return output + self.lora[name](input)
        return output


class VariantModule(nn.Module):
    """Runs the active variant's clone of `base`, else `base`."""

    def __init__(self, base: nn.Module, active: ContextVar[str]) -> None:
        super().__init__()
        self.base = base
        self.clones = nn.ModuleDict()
        self.active = active

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        name = self.active.get()
        if name in self.clones:
            return self.clones[name](*args, **kwargs)
        return self.base(*args, **kwargs)


def matches(fqn: str, names: list[str] | tuple[str, ...]) -> bool:
    return any(fqn == name or fqn.endswith(f".{name}") for name in names)


def linears(module: nn.Module, prefix: str = "") -> Iterator[tuple[str, nn.Linear]]:
    """`nn.Linear` leaves outside variant clones, which are already whole copies."""
    for name, child in module.named_children():
        fqn = prefix + name
        if isinstance(child, nn.Linear):
            yield fqn, child
        elif not isinstance(child, VariantModule):
            yield from linears(child, fqn + ".")


def holders(root: nn.Module) -> Iterator[nn.ModuleDict]:
    """Every slot dict (`LoraLinear.lora`) and clone dict (`VariantModule.clones`)."""
    for module in list(root.modules()):  # callers add entries while walking
        if isinstance(module, LoraLinear):
            yield module.lora
        elif isinstance(module, VariantModule):
            yield module.clones


def slots(root: nn.Module, name: str) -> list[nn.Module]:
    """`name`'s LoRA pairs and clones."""
    return [holder[name] for holder in holders(root) if name in holder]


def install_clones(
    root: nn.Module,
    name: str,
    variant: LoraVariant,
    names: list[str] | tuple[str, ...],
    active: ContextVar[str],
) -> None:
    """Wrap each module named in `names` once and give `name` a copy of the base."""
    for fqn in [f for f, _ in root.named_modules() if matches(f, names)]:
        parent_name, _, child = fqn.rpartition(".")
        parent = root.get_submodule(parent_name)
        wrapper = getattr(parent, child)
        if not isinstance(wrapper, VariantModule):
            wrapper = VariantModule(wrapper, active)
            setattr(parent, child, wrapper)
        clone = copy.deepcopy(wrapper.base)
        wrapper.clones[name] = clone.requires_grad_(variant.trainable)


def install_slots(
    root: nn.Module,
    name: str,
    variant: LoraVariant,
    targets: list[str] | tuple[str, ...] | Literal["all-linear"],
    excludes: tuple[str, ...],
    active: ContextVar[str],
) -> None:
    """Add `name`'s slot to every targeted linear, converting it to `LoraLinear` once."""
    for fqn, linear in list(linears(root)):
        if (targets == "all-linear" or matches(fqn, targets)) and not matches(
            fqn, excludes
        ):
            module = (
                linear
                if isinstance(linear, LoraLinear)
                else LoraLinear.convert(linear, active)
            )
            module.lora[name] = LoraPair(linear, variant).requires_grad_(
                variant.trainable
            )


def copy_slots(root: nn.Module, name: str, source: str) -> None:
    """Give `name` frozen fp32 copies of `source`'s slots and clones (EMA of a LoRA)."""
    for holder in holders(root):
        if source in holder:
            holder[name] = copy.deepcopy(holder[source]).float().requires_grad_(False)


def lora_pairs(root: nn.Module, name: str) -> dict[str, LoraPair]:
    return {
        fqn: cast(LoraPair, module.lora[name])
        for fqn, module in root.named_modules()
        if isinstance(module, LoraLinear) and name in module.lora
    }


def lora_metadata(variant: LoraVariant) -> dict[str, str]:
    return {"rank": str(variant.rank), "alpha": str(variant.scaling * variant.rank)}


def save_lora(root: nn.Module, name: str, variant: LoraVariant, path: str) -> None:
    """Write `name`'s slots as a native file: `<linear_fqn>.{A.weight,B.weight,B.bias}`."""
    tensors = {
        f"{fqn}.{key}": tensor.detach().cpu().contiguous()
        for fqn, pair in lora_pairs(root, name).items()
        for key, tensor in pair.state_dict().items()
    }
    save_file(tensors, path, metadata=lora_metadata(variant))


def load_lora(root: nn.Module, name: str, variant: LoraVariant, path: str) -> None:
    """Load a native file into `name`'s slots; rank, alpha and the targeted set must match."""
    with safe_open(path, "pt") as file:
        metadata, keys = file.metadata() or {}, file.keys()
        tensors = {key: file.get_tensor(key) for key in keys}
    pairs = lora_pairs(root, name)
    expected = {
        f"{fqn}.{key}" for fqn, pair in pairs.items() for key in pair.state_dict()
    }
    shape = (int(metadata.get("rank", 0)), float(metadata.get("alpha", 0)))
    if (
        shape != (variant.rank, variant.scaling * variant.rank)
        or set(tensors) != expected
    ):
        raise ValueError(
            f"{path} does not fit variant {name!r}: file {metadata} vs config "
            f"{lora_metadata(variant)}; unexpected keys {sorted(set(tensors) - expected)[:5]}, "
            f"missing keys {sorted(expected - set(tensors))[:5]}."
        )
    for fqn, pair in pairs.items():
        pair.load_state_dict(
            {key: tensors[f"{fqn}.{key}"] for key in pair.state_dict()}
        )


def pinned_checkpoint(
    original: Callable[..., Any], active: ContextVar[str]
) -> Callable[..., Any]:
    """Wrap a `_gradient_checkpointing_func` so recompute runs under the forward's variant."""

    def checkpoint(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        name = active.get()

        def pinned(*inputs: Any, **inner: Any) -> Any:
            token = active.set(name)
            try:
                return function(*inputs, **inner)
            finally:
                active.reset(token)

        return original(pinned, *args, **kwargs)

    return checkpoint
