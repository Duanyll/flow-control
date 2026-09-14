"""Momentum of any child predictor; opt in via the launch config's imports."""

from typing import ClassVar, Literal

import torch
from pydantic import Field

from flow_control.adapters.base import Batch
from flow_control.samplers.calls import Calls
from flow_control.samplers.guidance import ClassifierFreeGuidance
from flow_control.samplers.plan import EvalRequest, StepContext
from flow_control.samplers.prediction import (
    Prediction,
    Predictor,
    WrappedPrediction,
    prediction_registry,
)


@prediction_registry.register("momentum")
class MomentumGuidance(WrappedPrediction):
    """Apply EMA momentum after each child evaluation, including SA substeps.

    Each bind owns its own history. Above CFG this tracks the guided velocity;
    below CFG the two separately bound children track their own conditions.
    """

    type: Literal["momentum"] = "momentum"
    inner: Prediction = Field(default_factory=ClassifierFreeGuidance)
    alpha: float
    beta: float
    stateful: ClassVar[bool] = True

    def bind(self, row: Batch, negative_row: Batch | None = None) -> Predictor:
        inner = self.inner.bind(row, negative_row)
        momentum: torch.Tensor | None = None
        in_flight = False

        def predict(request: EvalRequest, ctx: StepContext) -> Calls[torch.Tensor]:
            nonlocal momentum, in_flight
            if in_flight:
                raise ValueError(
                    "Momentum cannot evaluate one binding concurrently; "
                    "bind independent branches separately to give each its own history."
                )
            in_flight = True
            try:
                velocity = yield from inner(request, ctx)
                previous = velocity if momentum is None else momentum
                guided = velocity + self.alpha * (velocity - previous)
                momentum = (1 - self.beta) * velocity + self.beta * previous
                return guided
            finally:
                in_flight = False

        return predict
