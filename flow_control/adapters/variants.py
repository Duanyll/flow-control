"""Variant configs: the named weight sets of one adapter (docs/adapters/variants.md)."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict

from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.registry import Registry, RegistryUnion

# --------------------------------- EMA warmup --------------------------------- #


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

# ---------------------------------- variants ---------------------------------- #


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
    # Bare on purpose: a parametrized `HfModelLoader[Any]` would re-validate a loader
    # subclass instance (Ideogram 4's fp8 loader) into a plain HfModelLoader.
    hf_model: HfModelLoader | None = None
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
