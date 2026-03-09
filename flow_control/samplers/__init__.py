from typing import Annotated

from pydantic import Discriminator, Tag

from .base import BaseSampler
from .euler import EulerSampler
from .momentum import MomentumGuidedSampler
from .shift import (
    ConstantShiftSampler,
    Flux2ShiftSampler,
    LinearShiftSampler,
    SquaredShiftSampler,
)

Sampler = Annotated[
    Annotated[BaseSampler, Tag("base")]
    | Annotated[EulerSampler, Tag("euler")]
    | Annotated[MomentumGuidedSampler, Tag("momentum")]
    | Annotated[ConstantShiftSampler, Tag("constant_shift")]
    | Annotated[LinearShiftSampler, Tag("linear_shift")]
    | Annotated[SquaredShiftSampler, Tag("squared_shift")]
    | Annotated[Flux2ShiftSampler, Tag("flux2_shift")],
    Discriminator("type"),
]

__all__ = [
    "Sampler",
]
