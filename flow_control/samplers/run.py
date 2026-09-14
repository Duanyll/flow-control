"""One sampling trajectory; solver and predictor composition share leaf calls."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from flow_control.adapters.base import Batch

from .calls import Calls
from .plan import EvalRequest, StepContext, Transition
from .prediction import BasePrediction, Predictor
from .projectors import apply_pre_transition

if TYPE_CHECKING:
    from .sampler import Sampler


@dataclass(slots=True)
class StepRecord:
    """A completed transition, passed to the caller without retaining tensors."""

    index: int
    transition: Transition
    latents: torch.Tensor
    velocity: torch.Tensor | None
    """The transition's last evaluated velocity, if any."""
    next_latents: torch.Tensor


type StepCollector = Callable[[SampleRun, StepRecord], None]


@dataclass(slots=True)
class SampleRun:
    sampler: Sampler
    row: Batch
    negative_row: Batch | None
    plan: list[Transition]
    ctx: StepContext
    predictor: BasePrediction
    collector: StepCollector | None = None
    _prediction: Predictor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._prediction = self._bind_prediction()

    def _bind_prediction(self) -> Predictor:
        inner = self.predictor.bind(self.row, self.negative_row)
        projectors, row = self.sampler.projectors, self.row

        def predict(request: EvalRequest, ctx: StepContext) -> Calls[torch.Tensor]:
            velocity = yield from inner(request, ctx)
            for projector in projectors:
                velocity = projector.post_combine(velocity, request, row, ctx)
            return velocity

        return predict

    def run(self) -> Calls[None]:
        ctx = self.ctx
        ctx.num_items = len(self.plan)
        predict = self._prediction
        velocity: torch.Tensor | None = None

        def observed(request: EvalRequest, ctx: StepContext) -> Calls[torch.Tensor]:
            nonlocal velocity
            velocity = yield from predict(request, ctx)
            return velocity

        for index, transition in enumerate(self.plan):
            ctx.item_index = index
            ctx.latents = apply_pre_transition(
                self.sampler.projectors, self.row, ctx, transition
            )
            velocity = None
            result = yield from transition.run(ctx, observed)
            if self.collector is not None:
                self.collector(
                    self,
                    StepRecord(
                        index, transition, ctx.latents, velocity, result.next_latents
                    ),
                )
            ctx.latents = result.next_latents
            if result.next_solver_state is not None:
                ctx.solver_state = result.next_solver_state
        ctx.latents = ctx.latents.to(self.row["noisy_latents"].dtype)

    def guided_velocity(
        self, latents: torch.Tensor, item_index: int
    ) -> Calls[torch.Tensor]:
        """Evaluate with fresh predictor bindings, isolated from all other calls."""
        transition = self.plan[item_index]
        latents = latents.float()
        ctx = StepContext(
            latents, None, None, item_index=item_index, num_items=len(self.plan)
        )
        return (
            yield from self._bind_prediction()(
                EvalRequest(
                    latents,
                    transition.sigma,
                    transition.sigma_next,
                    transition.eta,
                    transition.solver,
                ),
                ctx,
            )
        )
