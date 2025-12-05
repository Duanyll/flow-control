from typing import Annotated
from pydantic import BaseModel, PlainValidator
import torch
import math


class BaseTimestepWeighting(BaseModel):
    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        raise NotImplementedError


class UniformTimestepWeighting(BaseTimestepWeighting):
    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.rand(batch_size)


class LogitNormalTimestepWeighting(BaseTimestepWeighting):
    mean: float = 0.0
    std: float = 1.0

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        u = torch.normal(mean=self.mean, std=self.std, size=(batch_size,))
        u = torch.sigmoid(u)
        return u


class ModeTimestepWeighting(BaseTimestepWeighting):
    scale: float = 1.29

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        u = torch.rand(batch_size)
        u = 1 - u - self.scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
        return u


TIMESTEP_WEIGHTING_REGISTRY = {
    "uniform": UniformTimestepWeighting,
    "logit_normal": LogitNormalTimestepWeighting,
    "mode": ModeTimestepWeighting,
}


def parse_timestep_weighting(conf: dict) -> BaseTimestepWeighting:
    weighting_type = conf.pop("type")
    weighting_class = TIMESTEP_WEIGHTING_REGISTRY.get(weighting_type)
    if weighting_class is None:
        raise ValueError(f"Unknown timestep weighting type: {weighting_type}")
    return weighting_class(**conf)


TimestepWeighting = Annotated[
    BaseTimestepWeighting, PlainValidator(parse_timestep_weighting)
]


class BaseLossWeighting(BaseModel):
    def get_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
class UniformLossWeighting(BaseLossWeighting):
    def get_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(timesteps)
    
class SigmaSquaredLossWeighting(BaseLossWeighting):
    def get_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        return (timesteps ** (-2.0))
    
class CosmapLossWeighting(BaseLossWeighting):
    def get_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        bot = 1 - 2 * timesteps + 2 * (timesteps ** 2)
        weights = 2 / (math.pi * bot)
        return weights
    
LOSS_WEIGHTING_REGISTRY = {
    "uniform": UniformLossWeighting,
    "sigma_squared": SigmaSquaredLossWeighting,
    "cosmap": CosmapLossWeighting,
}

def parse_loss_weighting(conf: dict) -> BaseLossWeighting:
    weighting_type = conf.pop("type")
    weighting_class = LOSS_WEIGHTING_REGISTRY.get(weighting_type)
    if weighting_class is None:
        raise ValueError(f"Unknown loss weighting type: {weighting_type}")
    return weighting_class(**conf)

LossWeighting = Annotated[
    BaseLossWeighting, PlainValidator(parse_loss_weighting)]

__all__ = [
    "TimestepWeighting",
    "LossWeighting",
]