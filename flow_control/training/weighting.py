import math
from typing import Annotated, Literal

import torch
from pydantic import BaseModel, ConfigDict, Discriminator, Tag


class BaseTimestepWeighting(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        raise NotImplementedError


class UniformTimestepWeighting(BaseTimestepWeighting):
    type: Literal["uniform"] = "uniform"

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.rand(batch_size)


class LogitNormalTimestepWeighting(BaseTimestepWeighting):
    type: Literal["logit_normal"] = "logit_normal"
    mean: float = 0.0
    std: float = 1.0

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        u = torch.normal(mean=self.mean, std=self.std, size=(batch_size,))
        u = torch.sigmoid(u)
        return u


class ModeTimestepWeighting(BaseTimestepWeighting):
    type: Literal["mode"] = "mode"
    scale: float = 1.29

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        u = torch.rand(batch_size)
        u = 1 - u - self.scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
        return u


TimestepWeighting = Annotated[
    Annotated[UniformTimestepWeighting, Tag("uniform")]
    | Annotated[LogitNormalTimestepWeighting, Tag("logit_normal")]
    | Annotated[ModeTimestepWeighting, Tag("mode")],
    Discriminator("type"),
]


class BaseLossWeighting(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def get_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class UniformLossWeighting(BaseLossWeighting):
    type: Literal["uniform"] = "uniform"

    def get_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(timesteps)


class SigmaSquaredLossWeighting(BaseLossWeighting):
    type: Literal["sigma_squared"] = "sigma_squared"

    def get_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        return timesteps ** (-2.0)


class CosmapLossWeighting(BaseLossWeighting):
    type: Literal["cosmap"] = "cosmap"

    def get_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        bot = 1 - 2 * timesteps + 2 * (timesteps**2)
        weights = 2 / (math.pi * bot)
        return weights


LossWeighting = Annotated[
    Annotated[UniformLossWeighting, Tag("uniform")]
    | Annotated[SigmaSquaredLossWeighting, Tag("sigma_squared")]
    | Annotated[CosmapLossWeighting, Tag("cosmap")],
    Discriminator("type"),
]

__all__ = [
    "TimestepWeighting",
    "LossWeighting",
]
