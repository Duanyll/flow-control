"""Explicit training prediction trees, shared by current and auxiliary models."""

from typing import Self, cast

import torch
from pydantic import model_validator

from flow_control.adapters import ModelAdapter
from flow_control.adapters.base import Batch
from flow_control.processors.base import ProcessedBatch
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
        self, batch: Batch, num_items: int = 1, negative_batch: Batch | None = None
    ) -> Batch | None:
        """Resolve conditions from training requirements, even with unguided rollouts."""
        if not self.train_predictor.requires_negative(num_items):
            return None
        if negative_batch is not None:
            return negative_batch
        negative = self.processor.get_negative_batch(cast(ProcessedBatch, batch))
        if negative is None:
            raise ValueError(
                "train_predictor requires a negative condition, but the batch has none; "
                "preprocess with processor.save_negative=true."
            )
        return cast(Batch, negative)

    def predict_training(
        self,
        batches: list[Batch],
        timesteps: list[torch.Tensor],
        negative_batches: list[Batch | None] | None = None,
    ) -> list[torch.Tensor]:
        """Independent timestep predictions; preserve the caller's weight/grad scope."""
        return Executor(
            self.model, self.train_predictor.variant_keys(1) or [None]
        ).evaluate(
            [
                self.train_predictor.velocity(
                    batch,
                    timestep,
                    self.training_negative(batch, negative_batch=negative),
                )
                for batch, timestep, negative in zip(
                    batches,
                    timesteps,
                    [None] * len(batches)
                    if negative_batches is None
                    else negative_batches,
                    strict=True,
                )
            ]
        )
