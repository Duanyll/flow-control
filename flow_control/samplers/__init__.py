from typing import Annotated

from pydantic import PlainValidator

from .base import BaseSampler
from .euler import EulerSampler
from .momentum import MomentumGuidedSampler
from .shift import (
    ConstantShiftSampler,
    Flux2ShiftSampler,
    LinearShiftSampler,
    SquaredShiftSampler,
)

SAMPLER_REGISTRY = {
    "base": BaseSampler,
    "euler": EulerSampler,
    "momentum": MomentumGuidedSampler,
    "constant_shift": ConstantShiftSampler,
    "linear_shift": LinearShiftSampler,
    "squared_shift": SquaredShiftSampler,
    "flux2_shift": Flux2ShiftSampler,
}


def parse_sampler(conf: dict) -> BaseSampler:
    sampler_type = conf["type"]
    sampler_class = SAMPLER_REGISTRY.get(sampler_type)
    if sampler_class is None:
        raise ValueError(f"Unknown sampler type: {sampler_type}")
    return sampler_class(**conf)


Sampler = Annotated[BaseSampler, PlainValidator(parse_sampler)]

__all__ = [
    "Sampler",
]
