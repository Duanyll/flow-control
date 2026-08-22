"""Guidance axis (axis 2 of the plan-as-data design): branch combination rules.

Branch *evaluation* (cond/uncond batched forwards + FSDP alignment) lives in
``flow_control/samplers/executor.py``; this module owns the pluggable
*combination* rule applied to the evaluated branches. Guidance configs are
pydantic registry members (same mechanics as ``solver_registry``); their live
per-run state is a plain :class:`GuidanceState` carried by ``StepContext`` and
advanced per eval (execution contract rule 7).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Annotated, Literal

import torch
from pydantic import BaseModel, ConfigDict

from flow_control.utils.registry import Registry, RegistryUnion

from .plan import BranchEvals, GuidanceOutput, GuidanceState, StepContext


class BaseGuidance(BaseModel, ABC):
    type: Literal["base"] = "base"
    model_config = ConfigDict(extra="forbid")

    def init_state(self) -> GuidanceState | None:
        """Fresh per-run state, seeded into ``StepContext`` at run start."""
        return None

    @abstractmethod
    def needs_negative(self) -> bool:
        """Whether this guidance wants an unconditional branch eval."""

    @abstractmethod
    def combine(
        self,
        evals: BranchEvals,
        ctx: StepContext,
        state: GuidanceState | None,
    ) -> tuple[GuidanceOutput, GuidanceState | None]:
        """Combine branch velocities into one guided velocity.

        Called once per eval; the returned state takes effect immediately
        (multi-eval transitions advance it at every intermediate eval). Treat
        ``state`` and any tensors it contains as immutable: recorded steps may
        retain the same object as a transition-boundary snapshot.
        """


guidance_registry: Registry[BaseGuidance] = Registry("guidance", base=BaseGuidance)


@guidance_registry.register("cfg")
class ClassifierFreeGuidance(BaseGuidance):
    """Classifier-free guidance with optional norm-preserving rescale."""

    type: Literal["cfg"] = "cfg"
    scale: float = 1.0
    """
    TRUE classifier-free guidance scale. For guidance distilled models like FLUX, true
    CFG should not be applied and this should be kept at 1.0. Set their guidance
    embeddings value in ModelAdapter instead.
    """
    renorm: bool = False
    renorm_eps: float = 1e-8
    renorm_min: float = 0.0

    def needs_negative(self) -> bool:
        return self.scale > 1.0

    def combine(
        self,
        evals: BranchEvals,
        ctx: StepContext,
        state: GuidanceState | None,
    ) -> tuple[GuidanceOutput, GuidanceState | None]:
        if evals.uncond is None:
            return GuidanceOutput(velocity=evals.cond, branches=evals), state

        combined = evals.uncond + (evals.cond - evals.uncond) * self.scale
        if self.renorm:
            cond_norm = torch.norm(evals.cond, dim=2, keepdim=True)
            noise_norm = torch.norm(combined, dim=2, keepdim=True)
            combined = combined * (cond_norm / (noise_norm + self.renorm_eps)).clamp(
                min=self.renorm_min, max=1.0
            )
        return GuidanceOutput(velocity=combined, branches=evals), state


@dataclass(frozen=True, slots=True)
class MomentumGuidanceState(GuidanceState):
    """Exponentially smoothed velocity carried between guidance evals."""

    momentum: torch.Tensor | None = None


@guidance_registry.register("momentum")
class MomentumGuidance(ClassifierFreeGuidance):
    """CFG followed by the legacy velocity-momentum extrapolation.

    This is the functional counterpart of the former ``MomentumGuidedSampler``:
    runtime momentum lives in each run's immutable guidance state rather than
    on the shared sampler configuration.
    """

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
        cfg_output, _ = super().combine(evals, ctx, state)
        velocity = cfg_output.velocity
        momentum = velocity if state.momentum is None else state.momentum
        guided_velocity = velocity + self.alpha * (velocity - momentum)
        next_momentum = (1.0 - self.beta) * velocity + self.beta * momentum
        return (
            GuidanceOutput(velocity=guided_velocity, branches=evals),
            MomentumGuidanceState(momentum=next_momentum),
        )


Guidance = Annotated[
    BaseGuidance,
    # A bare number is the CFG scale: ``"guidance": 4.5``.
    RegistryUnion(guidance_registry, "type", number_as=("cfg", "scale")),
]


if __name__ == "__main__":
    from pydantic import TypeAdapter
    from rich import print

    adapter: TypeAdapter[BaseGuidance] = TypeAdapter(Guidance)
    guidance = adapter.validate_python({"type": "cfg", "scale": 4.5, "renorm": True})
    assert isinstance(guidance, ClassifierFreeGuidance)
    assert guidance.needs_negative()
    assert adapter.validate_json(adapter.dump_json(guidance)) == guidance
    # Bare-number shorthand: the CFG scale.
    assert adapter.validate_python(4.5) == ClassifierFreeGuidance(scale=4.5)

    cond = torch.full((1, 2, 3), 3.0)
    uncond = torch.full((1, 2, 3), 1.0)
    latents = torch.zeros(1, 2, 3)
    ctx = StepContext(
        latents=latents, generator=None, solver_state=None, guidance_state=None
    )
    evals = BranchEvals(cond=cond, uncond=uncond, latents=latents, sigma=1.0)
    output, state = ClassifierFreeGuidance(scale=2.0).combine(evals, ctx, None)
    torch.testing.assert_close(output.velocity, uncond + (cond - uncond) * 2.0)
    assert output.branches is evals and state is None

    cond_only = BranchEvals(cond=cond, uncond=None, latents=latents, sigma=1.0)
    output, _ = guidance.combine(cond_only, ctx, None)
    assert output.velocity is cond

    print("[green]guidance smoke test passed[/green]")
