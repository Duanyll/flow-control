from typing import Annotated

from pydantic import PlainValidator

from .base import BaseSampler
from .momentum import MomentumGuidedSampler
from .shift import ShiftedEulerSampler
from .simple_euler import SimpleEulerSampler

SAMPLER_REGISTRY = {
    "base": BaseSampler,
    "simple_euler": SimpleEulerSampler,
    "momentum": MomentumGuidedSampler,
    "shifted_euler": ShiftedEulerSampler,
}


def parse_sampler(conf: dict) -> BaseSampler:
    sampler_type = conf.pop("type")
    sampler_class = SAMPLER_REGISTRY.get(sampler_type)
    if sampler_class is None:
        raise ValueError(f"Unknown sampler type: {sampler_type}")
    return sampler_class(**conf)


Sampler = Annotated[BaseSampler, PlainValidator(parse_sampler)]

__all__ = [
    "Sampler",
]
