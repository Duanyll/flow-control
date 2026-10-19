"""Training predictions: independent points evaluated through ``train_predictor``."""

import torch

from flow_control.adapters import ModelAdapter
from flow_control.samplers import (
    Executor,
    Prediction,
    RootFrame,
    Row,
    data,
    variants,
)

from .data import DataMixin


class TrainingPredictionMixin(DataMixin):
    model: ModelAdapter
    train_predictor: Prediction
    """Prediction tree for every loss and auxiliary velocity, independent of
    the rollout and validation samplers (``model``, ``cfg``, ``tiled``, ...)."""

    def noised_point(self, row: Row, sigma: float, noise: torch.Tensor) -> Row:
        """A training point of ``row``: its clean latents re-noised at ``sigma``, chain rooted here."""
        x0 = row["clean_latents"].float()
        t = x0.new_tensor(sigma)
        return {
            **data(row),
            "clean_latents": x0,
            "noise": noise,
            "sigma": sigma,
            "noisy_latents": (1.0 - t) * x0 + t * noise,
            "_frame": RootFrame(self),
        }

    def predict_training(self, points: list[Row]) -> list[torch.Tensor]:
        """Velocities at independent points, in the caller's grad and variant scope."""
        return Executor(self.model, variants(self.train_predictor)).evaluate(
            [self.train_predictor(point) for point in points]
        )
