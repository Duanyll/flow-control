"""Momentum of any child prediction; opt in via the launch config's imports."""

from typing import Literal

import torch

from flow_control.samplers.calls import Calls
from flow_control.samplers.node import Frame, Row
from flow_control.samplers.prediction import (
    BasePrediction,
    Model,
    Prediction,
    prediction_registry,
)


@prediction_registry.register("momentum")
class Momentum(BasePrediction):
    """EMA momentum of the child prediction; the history lives on the row, so per CFG branch."""

    type: Literal["momentum"] = "momentum"
    prediction: Prediction = Model()
    alpha: float
    beta: float

    def run(self, row: Row, frame: Frame) -> Calls[torch.Tensor]:
        velocity = yield from self.prediction(row)
        previous = row.get("_momentum", velocity)
        row["_momentum"] = (1 - self.beta) * velocity + self.beta * previous
        return velocity + self.alpha * (velocity - previous)
