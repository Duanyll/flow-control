"""Optional momentum guidance; opt in with imports in the launch config."""

from dataclasses import dataclass
from typing import Literal

import torch

from flow_control.samplers.guidance import ClassifierFreeGuidance, guidance_registry
from flow_control.samplers.plan import (
    BranchEvals,
    GuidanceOutput,
    GuidanceState,
    StepContext,
)


@dataclass(frozen=True, slots=True)
class MomentumGuidanceState(GuidanceState):
    momentum: torch.Tensor | None = None


@guidance_registry.register("momentum")
class MomentumGuidance(ClassifierFreeGuidance):
    type: Literal["momentum"] = "momentum"
    alpha: float
    beta: float

    def init_state(self) -> GuidanceState | None:
        return MomentumGuidanceState()

    def combine(
        self,
        evals: BranchEvals,
        ctx: StepContext,
        state: GuidanceState | None,
    ) -> tuple[GuidanceOutput, GuidanceState | None]:
        if not isinstance(state, MomentumGuidanceState):
            raise TypeError("MomentumGuidance requires MomentumGuidanceState.")
        output, _ = super().combine(evals, ctx, state)
        velocity = output.velocity
        momentum = velocity if state.momentum is None else state.momentum
        return (
            GuidanceOutput(velocity=velocity + self.alpha * (velocity - momentum)),
            MomentumGuidanceState(
                momentum=(1 - self.beta) * velocity + self.beta * momentum
            ),
        )
