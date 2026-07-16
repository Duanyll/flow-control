import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict

from flow_control.utils.registry import Registry


class SolverState:
    """Marker base class for per-solver runtime state."""


@dataclass(slots=True)
class StepResult:
    next_latents: torch.Tensor
    log_prob: torch.Tensor
    mean: torch.Tensor
    std_dev: torch.Tensor
    state: SolverState | None = None


class BaseSolver(BaseModel, ABC):
    type: Literal["base"] = "base"
    model_config = ConfigDict(extra="forbid")

    eta: float = 0.0

    def init_state(self, num_steps: int) -> SolverState | None:
        return None

    @property
    def supports_step_log_prob(self) -> bool:
        return False

    def replay_step(
        self,
        velocity: torch.Tensor,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        prev_sample: torch.Tensor,
        state: SolverState | None = None,
    ) -> StepResult:
        return self.step(
            velocity=velocity,
            latents=latents,
            sigma=sigma,
            sigma_next=sigma_next,
            prev_sample=prev_sample,
            eta=self.eta,
            state=state,
        )

    @abstractmethod
    def step(
        self,
        velocity: torch.Tensor,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        prev_sample: torch.Tensor | None = None,
        eta: float | None = None,
        state: SolverState | None = None,
        generator: torch.Generator | None = None,
    ) -> StepResult:
        raise NotImplementedError()

    @staticmethod
    def _deterministic_step(
        latents: torch.Tensor,
        velocity: torch.Tensor,
        dt: torch.Tensor,
        state: SolverState | None = None,
    ) -> StepResult:
        next_latents = latents + velocity * dt
        return StepResult(
            next_latents=next_latents,
            log_prob=torch.zeros(latents.shape[0], device=latents.device),
            mean=next_latents,
            std_dev=torch.tensor(0.0, device=latents.device),
            state=state,
        )

    @staticmethod
    def _zero_log_prob(latents: torch.Tensor) -> torch.Tensor:
        return torch.zeros(latents.shape[0], device=latents.device)

    @staticmethod
    def _scalar_like(value: torch.Tensor | float, ref: torch.Tensor) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.squeeze() if value.ndim > 0 else value
        return torch.tensor(value, device=ref.device, dtype=ref.dtype)

    @staticmethod
    def _normal_log_prob(
        sample: torch.Tensor,
        mean: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        log_prob = (
            -((sample.detach() - mean) ** 2) / (2 * scale**2)
            - torch.log(scale)
            - torch.log(
                torch.sqrt(
                    torch.tensor(2 * math.pi, device=sample.device, dtype=sample.dtype)
                )
            )
        )
        return log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    @staticmethod
    def _velocity_to_x0(
        velocity: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        return sample - sigma * velocity

    @staticmethod
    def _sigma_to_alpha_sigma_t(
        sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return 1 - sigma, sigma


solver_registry: Registry[BaseSolver] = Registry("solver", base=BaseSolver)
