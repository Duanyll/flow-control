"""Composable velocity predictors: configuration trees and per-binding runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import Annotated, Any, ClassVar, Literal, cast

import torch
from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny

from flow_control.adapters.base import Batch
from flow_control.utils.model_cache import cache_fields
from flow_control.utils.registry import Registry, RegistryUnion

from .calls import Calls, ModelCall
from .plan import EvalRequest, StepContext

type Predictor = Callable[[EvalRequest, StepContext], Calls[torch.Tensor]]


class BasePrediction(BaseModel, ABC):
    """Bind once per run/branch; evaluate any number of times at that scope."""

    type: str
    model_config = ConfigDict(extra="forbid")
    stateful: ClassVar[bool] = False

    @abstractmethod
    def bind(self, row: Batch, negative_row: Batch | None = None) -> Predictor:
        """Return a predictor whose mutable history belongs only to this binding."""

    def velocity(
        self,
        row: Batch,
        timestep: torch.Tensor,
        negative_row: Batch | None = None,
    ) -> Calls[torch.Tensor]:
        """Evaluate this tree at an independent timestep, without a solver plan."""
        latents = row["noisy_latents"].float()
        return (
            yield from self.bind(row, negative_row)(
                EvalRequest(latents, timestep), StepContext(latents, None, None)
            )
        )

    def children(self) -> tuple[BasePrediction, ...]:
        return ()

    def walk(self) -> Iterator[BasePrediction]:
        yield self
        for child in self.children():
            yield from child.walk()

    def requires_negative(self, num_items: int) -> bool:
        return any(child.requires_negative(num_items) for child in self.children())

    def variant_keys(self, num_items: int) -> list[str | None]:
        """Declare explicit weight selections; an empty list inherits the caller's."""
        return list(
            dict.fromkeys(
                key
                for child in self.children()
                for key in child.variant_keys(num_items)
            )
        )


prediction_registry: Registry[BasePrediction] = Registry(
    "prediction", base=BasePrediction
)
Prediction = SerializeAsAny[
    Annotated[
        BasePrediction,
        RegistryUnion(prediction_registry, "type", number_as=("cfg", "scale")),
    ]
]


@prediction_registry.register("model")
class ModelPrediction(BasePrediction):
    type: Literal["model"] = "model"

    def bind(self, row: Batch, negative_row: Batch | None = None) -> Predictor:
        caches: dict[str | None, dict[str, Any]] = {}

        def predict(request: EvalRequest, ctx: StepContext) -> Calls[torch.Tensor]:
            timestep = (
                request.sigma.to(
                    device=request.latents.device, dtype=torch.float32
                ).expand(1)
                if isinstance(request.sigma, torch.Tensor)
                else request.latents.new_full((1,), request.sigma, dtype=torch.float32)
            )
            cache = caches.setdefault(request.variant, {})
            batch = cast(Batch, {**row, **cache, "noisy_latents": request.latents})
            (velocity,) = yield [ModelCall(batch, timestep, request.variant)]
            cache.update(cache_fields(batch))
            return velocity

        return predict


class WrappedPrediction(BasePrediction):
    """A configurable child, with generic traversal for execution requirements."""

    # Recursive members are defined before CFG registers the numeric shorthand.
    # Build their schemas after module imports have populated the registry.
    model_config = ConfigDict(defer_build=True)
    inner: Prediction = Field(default_factory=ModelPrediction)

    def children(self) -> tuple[BasePrediction, ...]:
        return (self.inner,)
