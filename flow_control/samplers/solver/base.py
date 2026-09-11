from abc import ABC
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict

from flow_control.utils.registry import Registry

from ..plan import (
    SamplingPlan,
    StepContext,
    Transition,
    TransitionGen,
)
from ..prediction import Predictor


class BaseSolver(BaseModel, ABC):
    type: Literal["base"] = "base"
    model_config = ConfigDict(extra="forbid")

    eta: float = 0.0

    def plan(self, sigmas: list[float]) -> SamplingPlan:
        """Compile a sigma grid into transitions.

        This default is for solvers whose ``eta`` directly expresses per-step
        stochasticity (flow/ddim/cps/dance); the terminal transition is always
        deterministic. Other solvers must override and write their actual
        per-step semantics into the transitions.
        """
        return [
            Transition(
                solver=self,
                sigma=sigma,
                sigma_next=sigma_next,
                eta=self.eta if i < len(sigmas) - 2 else 0.0,
            )
            for i, (sigma, sigma_next) in enumerate(
                zip(sigmas[:-1], sigmas[1:], strict=True)
            )
        ]

    def run_transition(
        self, tr: Transition, ctx: StepContext, predict: Predictor
    ) -> TransitionGen:
        """Perform one transition through a supplied predictor, yielding its leaf calls."""
        raise NotImplementedError(
            f"Solver '{self.type}' must implement run_transition()."
        )

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
