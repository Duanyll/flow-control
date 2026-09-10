"""Plan transforms configure stochastic windows without changing solver math."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Annotated, Literal

import torch
from pydantic import BaseModel, ConfigDict

from flow_control.utils.registry import Registry, RegistryUnion

from .plan import SamplingPlan


def select_sde_window(
    num_transitions: int,
    size: int | None,
    window_range: tuple[int, int] | None,
    generator: torch.Generator | None,
) -> tuple[int, int]:
    """Pick a window, excluding the terminal transition, using the sample RNG."""
    if size is None:
        return 0, num_transitions - 1
    if size <= 0:
        raise ValueError("SDE window size must be positive.")
    range_start, range_end = window_range or (0, num_transitions - 1)
    if not 0 <= range_start < range_end < num_transitions:
        raise ValueError(
            "SDE window range must satisfy 0 <= start < end < num_transitions."
        )
    max_start = range_end - size
    if max_start < range_start:
        raise ValueError(
            f"SDE window size={size} does not fit in "
            f"range=({range_start}, {range_end})."
        )
    random_device = generator.device if generator is not None else "cpu"
    window_start = int(
        torch.randint(
            range_start,
            max_start + 1,
            (),
            generator=generator,
            device=random_device,
        ).item()
    )
    return window_start, window_start + size


def with_sde_window(plan: SamplingPlan, start: int, end: int) -> SamplingPlan:
    return [
        replace(item, eta=item.eta if start <= index < end else 0.0)
        for index, item in enumerate(plan)
    ]


class BasePlanTransform(BaseModel, ABC):
    model_config = ConfigDict(extra="forbid")
    type: Literal["base"] = "base"

    @abstractmethod
    def apply(
        self, plan: SamplingPlan, generator: torch.Generator | None
    ) -> SamplingPlan: ...


plan_transform_registry: Registry[BasePlanTransform] = Registry(
    "plan_transform", base=BasePlanTransform
)


@plan_transform_registry.register("sde_window")
class SdeWindow(BasePlanTransform):
    type: Literal["sde_window"] = "sde_window"
    size: int | None = None
    range: tuple[int, int] | None = None

    def apply(
        self, plan: SamplingPlan, generator: torch.Generator | None
    ) -> SamplingPlan:
        start, end = select_sde_window(len(plan), self.size, self.range, generator)
        return with_sde_window(plan, start, end)


PlanTransform = Annotated[
    BasePlanTransform, RegistryUnion(plan_transform_registry, "type")
]
