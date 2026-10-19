"""How variants live on the transformer: LoRA slots, module clones, full copies, EMA counters."""

import copy
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, ClassVar, Literal, cast

import torch
from pydantic import BaseModel, PrivateAttr
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F

from .variants import AliasVariant, EmaVariant, FullVariant, LoraVariant, Variant

# ------------------------------ runtime modules ------------------------------ #


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


# --------------------------- walking the transformer -------------------------- #


def _matches(fqn: str, names: list[str] | tuple[str, ...]) -> bool:
    return any(fqn == name or fqn.endswith(f".{name}") for name in names)


def _linears(module: nn.Module, prefix: str = "") -> Iterator[tuple[str, nn.Linear]]:
    """`nn.Linear` leaves outside variant clones, which are already whole copies."""
    for name, child in module.named_children():
        fqn = prefix + name
        if isinstance(child, nn.Linear):
            yield fqn, child
        elif not isinstance(child, VariantModule):
            yield from _linears(child, fqn + ".")


def _holders(root: nn.Module) -> Iterator[nn.ModuleDict]:
    """Every slot dict (`LoraLinear.lora`) and clone dict (`VariantModule.clones`)."""
    for module in list(root.modules()):  # callers add entries while walking
        if isinstance(module, LoraLinear):
            yield module.lora
        elif isinstance(module, VariantModule):
            yield module.clones


def _slots(root: nn.Module, name: str) -> list[nn.Module]:
    """`name`'s LoRA pairs and clones."""
    return [holder[name] for holder in _holders(root) if name in holder]


# ------------------------------ native LoRA files ----------------------------- #


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


# ---------------------------------- the mixin --------------------------------- #


class VariantsMixin(BaseModel):
    """Realizes `variants` on the transformer and selects one per forward through a ContextVar."""

    trainable: bool = False
    """Train the base weights themselves (full fine-tuning)."""
    variants: dict[str, Variant] = {}
    """Named weight sets (docs/model-variants.md); "base" is built in."""
    lora_target_modules: ClassVar[tuple[str, ...] | Literal["all-linear"]] = (
        "all-linear"
    )
    """Default LoRA targets: linears whose FQN ends with one of these names."""
    lora_exclude_modules: ClassVar[tuple[str, ...]] = ()
    lora_extra_trainable: ClassVar[tuple[str, ...]] = ()
    """Default modules a LoRA variant clones and trains whole."""

    _active: ContextVar[str] = PrivateAttr()
    """The resolved variant running this adapter's forwards."""
    _full: dict[str, nn.Module] = PrivateAttr(default_factory=dict)
    """Variants with a full module of their own (`transformer.variants`)."""
    _ema_tensors: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] = (
        PrivateAttr(default_factory=dict)
    )

    @property
    def transformer(self) -> nn.Module:
        raise NotImplementedError

    def model_post_init(self, context: object, /) -> None:
        super().model_post_init(context)
        default = self.resolve("default")
        self._active = ContextVar("variant", default=default)

    # -------------------------------- selection -------------------------------- #

    def resolve(self, name: str) -> str:
        """Follow aliases; an undeclared "default" is the base weights."""
        variant = self.variants.get(name)
        if isinstance(variant, AliasVariant):
            return self.resolve(variant.of)
        if variant is not None or name == "base":
            return name
        if name == "default":
            return "base"
        raise ValueError(
            f"Unknown model variant {name!r}; declared: {['base', *self.variants]}."
        )

    @contextmanager
    def use_variant(self, name: str | None) -> Iterator[None]:
        """Run forwards with variant `name`; None keeps the current one."""
        if name is None:
            yield
            return
        token = self._active.set(self.resolve(name))
        try:
            yield
        finally:
            self._active.reset(token)

    @property
    def active_variant(self) -> str:
        return self._active.get()

    def module(self) -> nn.Module:
        """The full module running the active variant."""
        return self._full.get(self._active.get(), self.transformer)

    def fp32_copy_active(self) -> bool:
        """An EMA of a full module is an fp32 copy; its forward runs under autocast."""
        active = self._active.get()
        variant = self.variants.get(active)
        return isinstance(variant, EmaVariant) and self._kind(active) == "full"

    # -------------------------------- training --------------------------------- #

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.transformer.parameters() if p.requires_grad]

    def variant_modules(self) -> set[nn.Module]:
        """Modules holding variant weights and EMA counters (checkpointing frozen variants)."""
        transformer = cast(Any, self.transformer)
        owners: set[nn.Module] = {
            transformer.variant_counts,
            *transformer.variants.values(),
        }
        for name in self.variants:
            owners.update(_slots(self.transformer, name))
        return owners

    @torch.no_grad()
    def update_variants(self, every: Literal["step", "epoch"]) -> None:
        """Move every EMA variant scheduled `every` toward its source."""
        counts = cast(Any, self.transformer).variant_counts
        for name, variant in self.variants.items():
            if not isinstance(variant, EmaVariant) or variant.every != every:
                continue
            count = counts.get_buffer(name)
            decay = variant.warmup.get_decay(int(count), variant.decay)
            if decay < 1:
                if name not in self._ema_tensors:
                    # Collected lazily: FSDP replaces the parameters after load.
                    self._ema_tensors[name] = (
                        self._weights(name),
                        self._weights(self.resolve(variant.of)),
                    )
                ema, source = self._ema_tensors[name]
                torch._foreach_lerp_(ema, [t.float() for t in source], 1 - decay)
            count += 1

    def enable_gradient_checkpointing(self) -> None:
        """Enable checkpointing with recompute pinned to the forward's variant."""
        cast(Any, self.transformer).enable_gradient_checkpointing()
        for module in self.transformer.modules():
            original = getattr(module, "_gradient_checkpointing_func", None)
            if callable(original):
                cast(Any, module)._gradient_checkpointing_func = pinned_checkpoint(
                    original, self._active
                )

    # ------------------------------- realization ------------------------------- #

    def _realize(self, device: torch.device) -> None:
        """Install the variants on a freshly loaded transformer: copies, clones, then slots."""
        transformer = cast(Any, self.transformer)
        order = self._ordered()
        copies, counts = nn.ModuleDict(), nn.Module()
        for name in order:  # copies first: they must not carry LoRA slots
            if self._kind(name) == "full":
                copies[name] = self._copy(name, copies)
        for name in order:  # clones before slots: a clone must not contain slots
            variant = self.variants[name]
            if isinstance(variant, LoraVariant):
                self._install_clones(name, variant)
        for name in order:
            variant = self.variants[name]
            if isinstance(variant, LoraVariant):
                self._install_slots(name, variant, device)
            elif isinstance(variant, EmaVariant):
                if self._kind(name) == "lora":
                    self._copy_slots(name, self.resolve(variant.of))
                counts.register_buffer(
                    name, torch.zeros((), dtype=torch.long, device=device)
                )
        transformer.variants, transformer.variant_counts = copies, counts
        self._full = dict(copies.items())

    def _attach(self) -> None:
        """Adopt a cached transformer another adapter realized: rebind its slots to this selector."""
        for module in self.transformer.modules():
            if isinstance(module, LoraLinear | VariantModule):
                module.active = self._active
        self._full = dict(cast(Any, self.transformer).variants.items())

    def _kind(self, name: str) -> str:
        """How a variant is realized: "base", "full" (own module) or "lora" (slots)."""
        name = self.resolve(name)
        variant = self.variants.get(name)
        if isinstance(variant, LoraVariant):
            return "lora"
        if isinstance(variant, EmaVariant):
            return "lora" if self._kind(variant.of) == "lora" else "full"
        return "base" if name == "base" else "full"

    def _ordered(self) -> list[str]:
        """Non-alias variant names, each after the variant it builds on."""
        order: list[str] = []

        def visit(name: str) -> None:
            name = self.resolve(name)
            variant = self.variants.get(name)
            if variant is not None and name not in order:
                visit(variant.of)
                order.append(name)

        for name in self.variants:
            visit(name)
        return order

    def _copy(self, name: str, copies: nn.ModuleDict) -> nn.Module:
        """A full-kind variant's own module: a copy of its source (fp32 for an EMA)."""
        variant = self.variants[name]
        if isinstance(variant, FullVariant) and variant.hf_model is not None:
            raise NotImplementedError("FullVariant.hf_model is not implemented.")
        source = self.resolve(variant.of)
        if self._kind(source) == "lora":
            raise NotImplementedError(
                "A full copy of a LoRA variant is not implemented."
            )
        origin: nn.Module = self.transformer if source == "base" else copies[source]
        module = copy.deepcopy(origin)
        if isinstance(variant, EmaVariant):
            return module.float().requires_grad_(False)
        return module.requires_grad_(cast(FullVariant, variant).trainable)

    def _install_clones(self, name: str, variant: LoraVariant) -> None:
        """Wrap each `extra_trainable` module once and give `name` a copy of the base."""
        if self.resolve(variant.of) != "base":
            raise NotImplementedError("LoRA on a full variant is not implemented.")
        names = variant.extra_trainable
        if names is None:
            names = self.lora_extra_trainable
        root = self.transformer
        for fqn in [f for f, _ in root.named_modules() if _matches(f, names)]:
            parent_name, _, child = fqn.rpartition(".")
            parent = root.get_submodule(parent_name)
            wrapper = getattr(parent, child)
            if not isinstance(wrapper, VariantModule):
                wrapper = VariantModule(wrapper, self._active)
                setattr(parent, child, wrapper)
            clone = copy.deepcopy(wrapper.base)
            wrapper.clones[name] = clone.requires_grad_(variant.trainable)

    def _install_slots(
        self, name: str, variant: LoraVariant, device: torch.device
    ) -> None:
        """Add `name`'s slot to every targeted linear, converting it to `LoraLinear` once."""
        targets = variant.target_modules
        if targets is None:
            targets = self.lora_target_modules
        for fqn, linear in list(_linears(self.transformer)):
            if (targets == "all-linear" or _matches(fqn, targets)) and not _matches(
                fqn, self.lora_exclude_modules
            ):
                module = (
                    linear
                    if isinstance(linear, LoraLinear)
                    else LoraLinear.convert(linear, self._active)
                )
                module.lora[name] = LoraPair(linear, variant).requires_grad_(
                    variant.trainable
                )
        if variant.path is not None and device.type != "meta":  # the seed carries it
            load_lora(self.transformer, name, variant, variant.path)

    def _copy_slots(self, name: str, source: str) -> None:
        """Give `name` frozen fp32 copies of `source`'s slots and clones (EMA of a LoRA)."""
        for holder in _holders(self.transformer):
            if source in holder:
                holder[name] = (
                    copy.deepcopy(holder[source]).float().requires_grad_(False)
                )

    def _weights(self, name: str) -> list[torch.Tensor]:
        """The tensors realizing a resolved variant."""
        if name in self._full:
            return list(self._full[name].parameters())
        if name != "base":
            return [
                p for slot in _slots(self.transformer, name) for p in slot.parameters()
            ]
        owned = {
            id(p) for module in self.variant_modules() for p in module.parameters()
        }
        return [p for p in self.transformer.parameters() if id(p) not in owned]
