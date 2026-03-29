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


# -------------------- Flux2 blog training distributions ---------------------- #
# Reference: https://bfl.ai/techblog/representation-comparison/index.html
#
# These distributions use the timeshift function s(α, t) = αt / (1 + (α-1)t)
# to bias timestep sampling towards higher noise levels (t→1), which is
# beneficial for higher-dimensional latent spaces.


def _timeshift(shift: float, t: torch.Tensor) -> torch.Tensor:
    """Apply timeshift: s(α, t) = αt / (1 + (α-1)t)."""
    return shift * t / (1.0 + (shift - 1.0) * t)


class ShiftedUniformTimestepWeighting(BaseTimestepWeighting):
    """Uniform sampling composed with the timeshift function.

    Density: p(t; α) = α / (α + (1-α)t)²
    """

    type: Literal["shifted_uniform"] = "shifted_uniform"
    shift: float = 1.0

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        u = torch.rand(batch_size)
        return _timeshift(self.shift, u)


class ShiftedLogitNormalTimestepWeighting(BaseTimestepWeighting):
    """Logit-normal with mean shifted by log(α).

    Equivalent to sampling from LogitNormal(mean, std) then applying
    the timeshift, or directly sampling from LogitNormal(mean + log(α), std).
    """

    type: Literal["shifted_logit_normal"] = "shifted_logit_normal"
    mean: float = 0.0
    std: float = 1.0
    shift: float = 1.0

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        effective_mean = self.mean + math.log(self.shift)
        u = torch.normal(mean=effective_mean, std=self.std, size=(batch_size,))
        return torch.sigmoid(u)


def _logit_normal_mode(mu: float, sigma: float) -> float:
    """Find the mode of the logit-normal distribution via Newton's method.

    Solves: logit(t*) - μ - σ²(2t* - 1) = 0
    """
    t = 1.0 / (1.0 + math.exp(-mu))  # sigmoid(μ) as initial guess
    for _ in range(20):
        logit_t = math.log(t / (1.0 - t))
        g = logit_t - mu - sigma**2 * (2.0 * t - 1.0)
        dg = 1.0 / (t * (1.0 - t)) - 2.0 * sigma**2
        t = max(1e-6, min(1.0 - 1e-6, t - g / dg))
    return t


def _logit_normal_pdf(t: float, mu: float, sigma: float) -> float:
    """Evaluate the logit-normal probability density function at t."""
    logit_t = math.log(t / (1.0 - t))
    return math.exp(-((logit_t - mu) ** 2) / (2.0 * sigma**2)) / (
        sigma * math.sqrt(2.0 * math.pi) * t * (1.0 - t)
    )


def _logit_normal_cdf(t: float, mu: float, sigma: float) -> float:
    """Evaluate the logit-normal CDF at t: Φ((logit(t) - μ) / σ)."""
    logit_t = math.log(t / (1.0 - t))
    return 0.5 * (1.0 + math.erf((logit_t - mu) / (sigma * math.sqrt(2.0))))


class PlateauLogitNormalTimestepWeighting(BaseTimestepWeighting):
    """Logit-normal that stays constant after its mode, biasing towards noise.

    The density follows the logit-normal up to its mode t*, then remains
    at the constant value p_ln(t*) from t* to 1. This creates a plateau
    that places more probability mass on high-noise timesteps.
    """

    type: Literal["plateau_logit_normal"] = "plateau_logit_normal"
    mean: float = 0.0
    std: float = 1.0
    shift: float = 1.0

    def _compute_normalization(self) -> tuple[float, float, float, float]:
        """Return (t_star, p_star, P_star, Z) for the plateau distribution."""
        mu = self.mean + math.log(self.shift)
        sigma = self.std
        t_star = _logit_normal_mode(mu, sigma)
        p_star = _logit_normal_pdf(t_star, mu, sigma)
        P_star = _logit_normal_cdf(t_star, mu, sigma)
        Z = P_star + (1.0 - t_star) * p_star
        return t_star, p_star, P_star, Z

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        mu = self.mean + math.log(self.shift)
        sigma = self.std
        t_star, p_star, P_star, Z = self._compute_normalization()
        threshold = P_star / Z

        u = torch.rand(batch_size)

        # Inverse CDF for the logit-normal region (u <= threshold):
        # t = sigmoid(μ + σ · Φ⁻¹(u·Z))  where Φ⁻¹(x) = √2 · erfinv(2x - 1)
        clamped = (u * Z).clamp(1e-6, 1.0 - 1e-6)
        ln_part = torch.sigmoid(
            mu + sigma * math.sqrt(2.0) * torch.erfinv(2.0 * clamped - 1.0)
        )

        # Inverse CDF for the plateau region (u > threshold):
        # t = t* + (u·Z - P*) / p*
        plateau_part = (t_star + (u * Z - P_star) / p_star).clamp(0.0, 1.0)

        return torch.where(u <= threshold, ln_part, plateau_part)


TimestepWeighting = Annotated[
    Annotated[UniformTimestepWeighting, Tag("uniform")]
    | Annotated[LogitNormalTimestepWeighting, Tag("logit_normal")]
    | Annotated[ModeTimestepWeighting, Tag("mode")]
    | Annotated[ShiftedUniformTimestepWeighting, Tag("shifted_uniform")]
    | Annotated[ShiftedLogitNormalTimestepWeighting, Tag("shifted_logit_normal")]
    | Annotated[PlateauLogitNormalTimestepWeighting, Tag("plateau_logit_normal")],
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


if __name__ == "__main__":
    from rich import print

    N = 100_000

    configs: list[dict[str, object]] = [
        {"type": "shifted_uniform", "shift": 4.63},
        {"type": "shifted_logit_normal", "shift": 4.63},
        {"type": "plateau_logit_normal", "shift": 4.63},
        {"type": "shifted_logit_normal", "shift": 1.0},
        {"type": "plateau_logit_normal", "shift": 1.0},
    ]

    from pydantic import TypeAdapter

    adapter = TypeAdapter(TimestepWeighting)

    for cfg in configs:
        w = adapter.validate_python(cfg)
        samples = w.sample_timesteps(N)
        print(
            f"[bold]{cfg}[/bold]  "
            f"mean={samples.mean():.4f}  std={samples.std():.4f}  "
            f"min={samples.min():.4f}  max={samples.max():.4f}  "
            f"in [0,1]: {(samples >= 0).all() and (samples <= 1).all()}"
        )
