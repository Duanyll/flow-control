"""Explicit training prediction trees, shared by current and auxiliary models."""

from typing import Self, cast

import torch
from pydantic import model_validator

from flow_control.adapters import ModelAdapter
from flow_control.adapters.base import Batch
from flow_control.processors.base import ProcessedRow
from flow_control.samplers import Executor, Prediction

from .data import DataMixin


class TrainingPredictionMixin(DataMixin):
    model: ModelAdapter
    train_predictor: Prediction
    """Required tree for all loss/teacher/reference predictions, independent of
    rollout and validation guidance. Use ``model``, ``tiled``, or a composed tree."""

    @model_validator(mode="after")
    def check_train_predictor(self) -> Self:
        if any(node.stateful for node in self.train_predictor.walk()):
            raise ValueError(
                "train_predictor must be stateless: training evaluates independent "
                "timesteps without trajectory history. Keep stateful nodes such as "
                "Momentum in the rollout tree and omit them from train_predictor."
            )
        return self

    def training_negative(
        self, row: Batch, num_items: int = 1, negative_row: Batch | None = None
    ) -> Batch | None:
        """Resolve conditions from training requirements, even with unguided rollouts."""
        if not self.train_predictor.requires_negative(num_items):
            return None
        if negative_row is not None:
            return negative_row
        negative = self.processor.get_negative_row(cast(ProcessedRow, row))
        if negative is None:
            raise ValueError(
                "train_predictor requires a negative condition, but the row has none; "
                "preprocess with processor.save_negative=true."
            )
        return cast(Batch, negative)

    def predict_training(
        self,
        rows: list[Batch],
        timesteps: list[torch.Tensor],
        negative_rows: list[Batch | None] | None = None,
    ) -> list[torch.Tensor]:
        """Independent timestep predictions; preserve the caller's weight/grad scope."""
        return Executor(
            self.model, self.train_predictor.variant_keys(1) or [None]
        ).evaluate(
            [
                self.train_predictor.velocity(
                    row,
                    timestep,
                    self.training_negative(row, negative_row=negative),
                )
                for row, timestep, negative in zip(
                    rows,
                    timesteps,
                    [None] * len(rows) if negative_rows is None else negative_rows,
                    strict=True,
                )
            ]
        )
